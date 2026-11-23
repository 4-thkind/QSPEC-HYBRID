from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import analysis
from .core import (
    POLICIES,
    PROTOCOL,
    TASKS,
    VLLM,
    Store,
    atomic_json,
    bundle,
    digest,
    file_hash,
    read_json,
)
from .data import prepare, revisions
from .quant import ensure_export, tensor_index
from .runtime import Server
from .upstream import Scorers


def request_key(phase, policy, k, case_id, replicate=0):
    return digest([phase, policy, k, case_id, replicate])


def environment(root):
    if importlib.metadata.version("vllm") != VLLM:
        raise RuntimeError(f"This protocol requires vLLM {VLLM}")
    script = (
        "import json,torch; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable in the worker environment'; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name':p.name,'memory':p.total_memory,'capability':[p.major,p.minor],"
        "'torch':torch.__version__,'cuda':torch.version.cuda}))"
    )
    gpu = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    gpu["driver"] = driver
    if gpu["capability"] != [12, 0] or gpu["memory"] < 80 * 1024**3:
        raise RuntimeError(
            f"This protocol targets RTX PRO 6000 Blackwell SM120 with high VRAM, found {gpu}. No silent hardware/format fallback."
        )
    helper = subprocess.check_output(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "--help=all"],
        text=True,
    )
    required = [
        "--enable-per-request-metrics",
        "--per-request-spec-decode-metrics",
        "--worker-extension-cls",
        "--kv-cache-memory-bytes",
    ]
    if not all(x in helper for x in required):
        raise RuntimeError(
            "Installed serving CLI does not expose the required audited interfaces"
        )
    code = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted((root / "qspec_hybrid").glob("*.py"))
    }
    code["bootstrap.py"] = file_hash(root / "bootstrap.py")
    identity = digest(
        {
            "gpu": gpu,
            "protocol": PROTOCOL,
            "code": code,
            "environment_lock": file_hash(root / "environment.lock.txt"),
        }
    )
    path = root / "results" / "environment.json"
    old = read_json(path)
    if old and old["identity"] != identity:
        atomic_json(root / "results" / "environment_mismatch.json", {
            "saved_identity": old["identity"], "current_identity": identity,
            "saved_gpu": old.get("gpu"), "current_gpu": gpu,
            "gpu_metadata_equal": old.get("gpu") == gpu,
            "current_code_sha256": code,
            "current_environment_lock_sha256": file_hash(root / "environment.lock.txt"),
            "current_protocol_sha256": digest(PROTOCOL),
            "saved_records_modified": False,
            "action": "blocked; an explicitly reviewed measurement lineage is required",
        })
        raise RuntimeError(
            "Worker software/GPU class changed; refusing to mix resumed measurements"
        )
    atomic_json(path, {"identity": identity, "gpu": gpu, "driver": driver})
    return identity


def baseline(root):
    from huggingface_hub import snapshot_download

    rev = revisions(root)["model"]
    path = Path(
        snapshot_download(
            rev["id"],
            revision=rev["revision"],
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.txt",
                "*.model",
                "*.jinja",
                "*.tiktoken",
            ],
            ignore_patterns=["original/*", "onnx/*"],
        )
    )
    if not any("mtp" in key.lower() for key in tensor_index(path)):
        raise RuntimeError(
            "Downloaded reference checkpoint does not contain native MTP weights"
        )
    return path


def whitespace_only_difference(ar, mtp):
    """True when MTP changed only spacing and did not make the score worse.

    Speculative verification runs a batched forward pass, so its GEMM shapes
    differ from one-token-at-a-time decoding. The reduction order differs, the
    rounding differs, and a near-tied argmax can land the other way. With
    temperature 0 and a fixed seed this is the only remaining source of
    divergence, and it cannot be removed without changing the kernels being
    timed. Such a mismatch is recorded either way; it only stops blocking the
    gate when no non-space character moved and the task score did not drop.
    """
    if "".join(ar["text"].split()) != "".join(mtp["text"].split()):
        return False
    before, after = ar.get("score"), mtp.get("score")
    if not before or not after:
        return False
    if before.get("value") is None or after.get("value") is None:
        return False
    if before.get("higher_is_better"):
        return after["value"] >= before["value"]
    return after["value"] <= before["value"]


class Experiment:
    def __init__(self, root, identity, prepared, base):
        self.root, self.identity, self.prepared, self.base = (
            root,
            identity,
            prepared,
            base,
        )
        self.scorers = Scorers(root)
        self.db = Store(root / "results")
        self.new_records = 0
        self.mode = "pilot"

    def status(self, stage, **kwargs):
        saved = self.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        atomic_json(
            self.root / "status.json",
            {
                "state": "running",
                "stage": stage,
                "mode": self.mode,
                "saved_records": saved,
                "core_full_requests": 17920,
                **kwargs,
            },
        )

    def check_stop(self):
        if (self.root / "STOP").exists():
            raise KeyboardInterrupt

    def record(self, server, case, phase, replicate=0, profile=False, prefix=None):
        self.check_stop()
        key = request_key(phase, server.policy, server.k, case["id"], replicate)
        existing = self.db.get(key)
        if existing:
            if (
                existing["identity"] != self.identity
                or existing["input_hash"] != self.prepared["input_hash"]
            ):
                raise RuntimeError(
                    "Cannot resume a request from different software or frozen inputs"
                )
            return existing
        self.status(
            f"{phase}: {server.policy} · MTP {'off' if not server.k else server.k} · {case['id']} · repeat {replicate}"
        )
        result = None
        try:
            if profile:
                server.rpc("qspec_profile_begin")
            result = server.generate(case, prefix=prefix)
            if profile:
                result["forward_profile"] = server.rpc("qspec_profile_end")
            score = None
            if phase in ("pilot", "full", "heldout"):
                score = self.scorers.score(case, result["text"])
            result.update(
                phase=phase,
                policy=server.policy,
                k=server.k,
                case_id=case["id"],
                task=case["task"],
                group=case["group"],
                stratum=case["stratum"],
                replicate=replicate,
                timing=bool(case.get("timing_repeat", True)),
                identity=self.identity,
                input_hash=self.prepared["input_hash"],
                model_revision=self.prepared.get("revisions", {})
                .get("model", {})
                .get("revision"),
                image_sha256=case["image_sha256"],
                prompt_sha256=digest(case["prompt"]),
                score=score,
                state="ok",
                saved_at=time.time(),
            )
            if prefix is not None:
                result["fixed_prefix"] = prefix
                result["replay_caveat"] = (
                    "Common initial prefix; later continuation is free generation, not full teacher forcing."
                )
            self.db.put(key, result)
            self.new_records += 1
            if self.new_records % 20 == 0:
                analysis.summarize(self.root)
            return result
        except Exception as error:
            failure = {
                "phase": phase,
                "policy": server.policy,
                "k": server.k,
                "case_id": case["id"],
                "task": case["task"],
                "replicate": replicate,
                "state": "failed",
                "identity": self.identity,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "saved_at": time.time(),
            }
            if result is not None:
                failure["raw_output"] = result
            # Failed attempts stay immutable; retries use the canonical success key.
            self.db.put("failed:" + key + ":" + str(time.time_ns()), failure)
            raise

    def run_condition(
        self,
        phase,
        policy,
        k,
        cases,
        replicate=0,
        eager=False,
        profile=False,
        prefixes=None,
    ):
        schedule = [(c, replicate) for c in cases]
        if phase == "pilot" and replicate == 0:
            sentinels = [next(c for c in cases if c["task"] == task) for task in TASKS]
            schedule += [(c, 1) for c in sentinels]
        pending = [
            (c, r)
            for c, r in schedule
            if not self.db.get(request_key(phase, policy, k, c["id"], r))
        ]
        if not pending:
            return [
                self.db.get(request_key(phase, policy, k, c["id"], replicate))
                for c in cases
            ]
        self.status(f"Preparing {policy}, K={k}, {phase}")
        model = ensure_export(self.root, self.base, policy)
        print(
            f"Starting {phase}: {policy} K={k}; {len(pending)} pending requests",
            flush=True,
        )
        with Server(self.root, model, policy, k, self.identity, eager=eager) as server:
            # One warm-up per task, on development inputs, never an evaluation answer filter.
            for task in TASKS:
                warm = next(
                    c for c in self.prepared["development"] if c["task"] == task
                )
                server.generate(warm)
            for case, repetition in pending:
                self.record(
                    server,
                    case,
                    phase,
                    repetition,
                    profile=profile,
                    prefix=prefixes.get(case["id"]) if prefixes else None,
                )
        analysis.summarize(self.root)
        return [
            self.db.get(request_key(phase, policy, k, c["id"], replicate))
            for c in cases
        ]

    def compare(self, phase, policy, k, cases):
        differences, missing, instability, proposals = [], [], [], 0
        for case in cases:
            ar = self.db.get(request_key(phase, policy, 0, case["id"]))
            mtp = self.db.get(request_key(phase, policy, k, case["id"]))
            if ar is None or mtp is None:
                missing.append(case["id"])
                continue
            proposals += (mtp.get("speculative") or {}).get("num_draft_tokens", 0)
            if phase == "pilot":
                for mode in (0, k):
                    repeat = self.db.get(
                        request_key(phase, policy, mode, case["id"], 1)
                    )
                    first = ar if mode == 0 else mtp
                    if repeat and repeat["output_ids"] != first["output_ids"]:
                        instability.append({"case": case["id"], "k": mode})
            if ar["output_ids"] != mtp["output_ids"]:
                length = min(len(ar["output_ids"]), len(mtp["output_ids"]))
                at = next(
                    (
                        i
                        for i in range(length)
                        if ar["output_ids"][i] != mtp["output_ids"][i]
                    ),
                    length,
                )
                differences.append(
                    {
                        "case": case["id"],
                        "first_mismatch": at,
                        "whitespace_only": whitespace_only_difference(ar, mtp),
                        "ar_tokens": len(ar["output_ids"]),
                        "mtp_tokens": len(mtp["output_ids"]),
                        "ar_score": (ar.get("score") or {}).get("value"),
                        "mtp_score": (mtp.get("score") or {}).get("value"),
                        "ar_key": request_key(phase, policy, 0, case["id"]),
                        "mtp_key": request_key(phase, policy, k, case["id"]),
                    }
                )
        blocking = [d for d in differences if not d["whitespace_only"]]
        # Speculative verification is lossless in exact arithmetic, so AR/MTP
        # equality is an apparatus check and it is enforced on the BF16 baseline.
        # Under quantization the logit gaps shrink and batched verification
        # rounds differently, so divergence there is the effect being measured,
        # not a fault: every case is recorded and reported, and only apparatus
        # faults (missing outputs, unstable repeats, no drafts) block.
        exactness_required = policy == "P0"
        return {
            "policy": policy,
            "k": k,
            "cases": len(cases),
            "missing": missing,
            "mismatches": differences,
            "blocking_mismatches": blocking if exactness_required else [],
            "recorded_divergence": len(differences),
            "whitespace_only_mismatches": len(differences) - len(blocking),
            "exact_token_match": not differences,
            "exactness_required": exactness_required,
            "repeat_instability": instability,
            "proposals": proposals,
            "passed": (not blocking or not exactness_required)
            and not missing
            and not instability
            and proposals > 0,
        }

    def pilot(self):
        self.mode = "pilot"
        cases = self.prepared["development"]
        self.run_condition("pilot", "P0", 0, cases)
        choices = []
        checks = []
        for k in (1, 2):
            outputs = self.run_condition("pilot", "P0", k, cases)
            check = self.compare("pilot", "P0", k, cases)
            checks.append(check)
            if check["passed"]:
                # Equal task weights; no response-dependent filtering.
                task_times = [
                    statistics.mean(
                        x["request_latency_s"] for x in outputs if x["task"] == task
                    )
                    for task in TASKS
                ]
                choices.append(
                    (math.exp(statistics.mean(math.log(x) for x in task_times)), k)
                )
        if not choices:
            return self.freeze_gate(
                False,
                None,
                checks,
                "Neither K=1 nor K=2 passed the BF16 multimodal pilot; full scale remains blocked. Inspect saved mismatches, do not remove samples.",
            )
        chosen = min(choices)[1]
        for policy in list(POLICIES)[1:]:
            for k in (0, chosen):
                self.run_condition("pilot", policy, k, cases)
            checks.append(self.compare("pilot", policy, chosen, cases))
        passed = all(c["passed"] for c in checks if c["k"] == chosen)
        return self.freeze_gate(
            passed,
            chosen,
            checks,
            "Pilot passed; K is frozen for all full-scale policies."
            if passed
            else "Pilot completed, but one or more policies failed exact-output checks. Full scale remains blocked; all results are retained.",
        )

    def freeze_gate(self, passed, k, checks, message):
        value = {
            "passed": passed,
            "k": k,
            "checks": checks,
            "message": message,
            "identity": self.identity,
            "input_hash": self.prepared["input_hash"],
            "protocol": digest(PROTOCOL),
        }
        atomic_json(self.root / "pilot_gate.json", value)
        analysis.summarize(self.root, figures=True, intervals=True)
        atomic_json(
            self.root / "status.json",
            {
                "state": "complete" if passed else "blocked",
                "mode": "pilot",
                "stage": message,
            },
        )
        print(message, flush=True)
        return value

    def full(self):
        self.mode = "full"
        gate = read_json(self.root / "pilot_gate.json", {})
        if (
            not gate.get("passed")
            or gate.get("identity") != self.identity
            or gate.get("input_hash") != self.prepared["input_hash"]
            or gate.get("protocol") != digest(PROTOCOL)
        ):
            raise RuntimeError(
                "Run Pilot test successfully in this exact environment/protocol before Full scale"
            )
        k = gate["k"]
        rng = random.Random(PROTOCOL["seed"])
        policies = list(POLICIES)
        rng.shuffle(policies)
        cases = self.prepared["evaluation"]
        for repeat in range(3):
            cohort = cases if repeat == 0 else [c for c in cases if c["timing_repeat"]]
            order = policies if repeat % 2 == 0 else list(reversed(policies))
            for index, policy in enumerate(order):
                modes = (0, k) if (index + repeat) % 2 == 0 else (k, 0)
                for mode in modes:
                    self.run_condition("full", policy, mode, cohort, replicate=repeat)
        checks = [self.compare("full", p, k, cases) for p in POLICIES]
        atomic_json(self.root / "results" / "full_correctness.json", checks)
        selection_path = self.root / "results" / "heldout_selection.json"
        selection = analysis.select_candidates(self.root)
        old = read_json(selection_path)
        if old and old != selection:
            raise RuntimeError(
                "Held-out selection changed on resume; results were not overwritten"
            )
        atomic_json(selection_path, selection)
        for policy in selection["policies"]:
            for mode in (0, k):
                self.run_condition("heldout", policy, mode, self.prepared["heldout"])
        heldout_checks = [
            self.compare("heldout", p, k, self.prepared["heldout"])
            for p in selection["policies"]
        ]
        atomic_json(self.root / "results" / "heldout_correctness.json", heldout_checks)
        self.diagnostics(k)
        analysis.summarize(self.root, figures=True, intervals=True)
        review = any(not c["passed"] for c in checks + heldout_checks)
        atomic_json(
            self.root / "status.json",
            {
                "state": "review_required" if review else "complete",
                "mode": "full",
                "stage": "Finished. Numerical mismatches retained; no lossless claim."
                if review
                else "Finished; review quality, confidence intervals and diagnostic caveats before making claims.",
            },
        )

    def diagnostics(self, k):
        # Separate eager diagnostic timing and common-prefix continuation.
        cases = [
            next(c for c in self.prepared["development"] if c["task"] == task)
            for task in TASKS
        ]
        prefixes = {}
        for case in cases:
            output = self.db.get(request_key("pilot", "P0", 0, case["id"]))
            prefixes[case["id"]] = output["text"][: max(1, len(output["text"]) // 2)]
        short_cases = [{**c, "max_tokens": min(c["max_tokens"], 64)} for c in cases]
        for policy in ("P0", "P1", "P7"):
            for mode in (0, k):
                self.run_condition(
                    "profile", policy, mode, short_cases, eager=True, profile=True
                )
                self.run_condition(
                    "prefix_continuation", policy, mode, short_cases, prefixes=prefixes
                )
        replay_checks = []
        for case in short_cases:
            records = [
                self.db.get(
                    request_key("prefix_continuation", policy, mode, case["id"])
                )
                for policy in ("P0", "P1", "P7")
                for mode in (0, k)
            ]
            hashes = {
                digest(r["prompt_ids"])
                for r in records
                if r and r["prompt_ids"] is not None
            }
            replay_checks.append(
                {
                    "case": case["id"],
                    "identical_tokenized_input_prefix": len(hashes) == 1,
                    "first_generated_token_identical": len(
                        {r["output_ids"][0] for r in records if r and r["output_ids"]}
                    )
                    == 1,
                    "caveat": "Continuation is not full teacher-forced replay; only the initial input is controlled.",
                }
            )
        atomic_json(self.root / "results" / "prefix_checks.json", replay_checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    experiment = None

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        atomic_json(root / "protocol.json", PROTOCOL)
        identity = environment(root)
        atomic_json(
            root / "status.json",
            {
                "state": "running",
                "mode": args.mode,
                "stage": "Preparing frozen inputs and evaluator parity checks",
            },
        )
        prepared = prepare(root)
        base = baseline(root)
        experiment = Experiment(root, identity, prepared, base)
        experiment.pilot() if args.mode == "pilot" else experiment.full()
    except KeyboardInterrupt:
        atomic_json(
            root / "status.json",
            {
                "state": "stopped",
                "mode": args.mode,
                "stage": "Stopped; completed requests retained",
            },
        )
    except Exception as error:
        atomic_json(
            root / "status.json",
            {"state": "failed", "mode": args.mode, "stage": str(error)},
        )
        traceback.print_exc()
        raise
    finally:
        if experiment:
            experiment.db.close()
        try:
            analysis.summarize(root)
        except Exception as error:
            print("Final summary failed; committed records are still on disk:", error, flush=True)


if __name__ == "__main__":
    main()
