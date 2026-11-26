"""Revision experiments, run after the frozen main study.

E1  root cause of the 31 BF16 AR/MTP divergences (logprobs, prefix replay, state check)
E2  second checkpoint (Qwen3.5-4B, native MTP): BF16 and quantized, MTP on/off
E3  format vs component: FFN-only FP8 (F1) against FFN NVFP4 (P1), attention FP8 (P2, P3)
E4  image tokens x output length sweep with TTFT and draft/verify GPU split
E5  fresh pre-registered non-inferiority hold-out for the fast policies

Reuses the main-study exports. PROTOCOL is never edited: new policies live in this
module only, under their own names, with their own export identities.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import gzip
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from collections import defaultdict
from pathlib import Path

from . import quant, runtime
from .core import POLICIES, PROTOCOL, SEED, TASKS, Store, atomic_json, digest, read_json
from .data import balanced, image_of, prepare, save_case
from .data import prompt as task_prompt
from .runner import baseline, environment
from .upstream import Scorers

EXT_POLICIES = {"F1": ("D",)}  # FFN only, FP8 instead of NVFP4
SMALL = {"id": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
SMALL_POLICIES = {"S0": (), "S1": ("D",), "S7": ("D", "A", "G")}
ALL_POLICIES = {**POLICIES, **EXT_POLICIES, **SMALL_POLICIES}
# Module-level rebind inside quant only; core.POLICIES and PROTOCOL stay byte-identical.
quant.POLICIES = ALL_POLICIES
K = 2  # frozen by the main-study pilot gate
EXPORT_MODULE = "qspec_hybrid.extensions"  # entry that registers the policy table in the export subprocess

PREREGISTRATION = {
    "written": "2026-11-10, before any fresh-sample generation",
    "question": "Do the fast policies stay within a fixed quality tolerance of BF16 on unseen inputs?",
    "policies": ["P1", "P7"],
    "reference": "P0",
    "decoding": "native MTP, K=2, for every policy (matched mode)",
    "tasks": {
        "mmstar": {"metric": "mc_accuracy", "margin": 0.02, "sample_cap": 1000},
        "chartqa": {"metric": "relaxed_accuracy", "margin": 0.02, "sample_cap": 1000},
        "omnidoc_ocr": {"metric": "normalized edit distance (lower is better)", "margin": 0.01, "sample_cap": 500},
    },
    "test": "paired difference (policy minus BF16, oriented so negative is worse); non-inferior when the "
            "lower bound of the two-sided 95% paired bootstrap interval (10,000 resamples, seed 20261018) "
            "is above minus the margin",
    "multiplicity": "Holm correction across the 6 policy x task tests, reported alongside the unadjusted intervals",
    "sampling": "all eligible rows not used by the main study (development, evaluation, held-out, calibration), "
                "seeded stratified round robin with seed SEED+5, capped as listed; no response-dependent filtering",
}

E4_TOKENS = (64, 144, 256, 576, 960)
E4_LENGTHS = (16, 64, 256, 1024)
E4_PROMPT = "Describe this image in detail, including all visible text."
E4_PROFILE = ((64, 256), (960, 256), (960, 16))
PROFILE_WAIT_S = 90
LOGPROBS = {"logprobs": True, "top_logprobs": 5, "return_tokens_as_token_ids": True}


# --------------------------------------------------------------------------- helpers

def ext_recipe(policy):
    base_policy = {"F1": "P1", "S1": "P1", "S7": "P7"}.get(policy, policy)
    value = quant.recipe(base_policy)
    if policy == "F1":
        for rule in value["quantize"]["quant_cfg"]:
            if rule.get("cfg") == {"$import": "nvfp4"}:
                rule["cfg"] = {"$import": "fp8"}
    value["metadata"]["description"] = f"QSpec revision {policy} ({'+'.join(ALL_POLICIES[policy]) or 'none'})"
    return value


def validate_runtime(audits, policy, k):
    """audit.validate_runtime with the extension policy table."""
    if not audits:
        raise RuntimeError("No GPU precision audit returned")
    for audit in audits:
        if audit.get("kv_cache_dtype") != "torch.bfloat16":
            raise RuntimeError(f"KV storage is not BF16: {audit.get('kv_cache_dtype')}")
        if k and not audit["mtp_found"]:
            raise RuntimeError("MTP requested but no native drafter model found")
        counts = {g: 0 for g in ("D", "A", "G")}
        for module in audit["modules"]:
            group = module["group"]
            if group in counts and module["scope"] == "target":
                counts[group] += 1
                if module["quantized"] != (group in ALL_POLICIES[policy]):
                    raise RuntimeError(f"Wrong runtime precision: {module}")
                if module["quantized"]:
                    kernel = module.get("kernel", "NoneType")
                    if kernel == "NoneType" or any(x in kernel.lower() for x in ("emulation", "marlin")):
                        raise RuntimeError(f"Native low-bit kernel not established: {module}")
            elif module["quantized"]:
                raise RuntimeError(f"Protected module was quantized: {module}")
        if not all(counts.values()):
            raise RuntimeError(f"Cannot establish all runtime projection groups: {counts}")


runtime.validate_runtime = validate_runtime
_command = runtime.command


def command(model, port, k, eager=False, smoke=False):
    args = _command(model, port, k, eager, smoke=smoke)
    folder = os.environ.get("QSPEC_PROFILE_DIR")
    if folder:
        args += ["--profiler-config", json.dumps({
            "profiler": "torch", "torch_profiler_dir": folder, "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False, "ignore_frontend": True,
        })]
    return args


runtime.command = command


def model_bytes(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))


def ext_export(root, base, name, revision):
    """quant.ensure_export for extension policies, with a correct export identity."""
    import yaml

    path = root / "models" / name
    expected = {
        "protocol": digest(PROTOCOL),
        "recipe": digest(ext_recipe(name)),
        "base_revision": revision,
        "calibration": digest(read_json(root / "data/prepared.json")["calibration"]),
        "extension": True,
    }
    manifest = read_json(path / "export_manifest.json")
    if manifest:
        if manifest["identity"] != expected:
            raise RuntimeError(f"{name} export identity changed; refusing to reuse it")
        for file, size in manifest["file_sizes"].items():
            if not (path / file).exists() or (path / file).stat().st_size != size:
                raise RuntimeError(f"Incomplete cached export: {name}/{file}")
        return path
    if path.exists():
        raise RuntimeError(f"Unverified directory exists at {path}; it was left untouched")
    if shutil.disk_usage(root).free < 25 * 1024**3:
        raise RuntimeError("Less than 25 GiB disk remains; export not started")
    config = root / "configs" / f"{name}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(ext_recipe(name), sort_keys=False))
    (root / "models").mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{name}-export-", dir=root / "models"))
    output = temporary / "weights"
    print(f"Exporting {name}; this takes a while", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", EXPORT_MODULE, "--export", str(root), str(base), str(config), str(output)],
            cwd=root, env=dict(os.environ, QSPEC_ROOT=str(root)), check=True,
        )
        result = quant.audit_export(output, base, name)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)  # never leave a half export eating disk
        raise
    result["identity"] = expected
    result["file_sizes"] = {f.name: f.stat().st_size for f in output.iterdir() if f.is_file()}
    atomic_json(output / "export_manifest.json", result)
    atomic_json(root / "results" / "export_audits" / f"{name}.json", result)
    os.replace(output, path)
    temporary.rmdir()
    return path


def small_baseline():
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        SMALL["id"], revision=SMALL["revision"],
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
    ))
    if not any("mtp" in key.lower() for key in quant.tensor_index(path)):
        raise RuntimeError("Second checkpoint has no native MTP weights")
    return path


_ERROR_LINE = re.compile(r"\b([A-Za-z_]*(?:Error|Exception)): (.+)")


def root_cause(log_path):
    """First real exception message in a vLLM server log, without the generic wrapper."""
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        match = _ERROR_LINE.search(line)
        if match and "Engine core initialization failed" not in line and "See root cause above" not in line:
            return f"{match.group(1)}: {match.group(2)}"[:900]
    return None


class ServerStartError(RuntimeError):
    pass


def post(server, path):
    request = urllib.request.Request(server.url + path, data=b"{}", method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + server.token})
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()


def summarize_trace(files):
    """Kernel totals and GPU time per runner region from torch profiler traces."""
    kernels, regions, forwards = defaultdict(lambda: [0, 0.0]), defaultdict(float), []
    for file in files:
        opener = gzip.open if file.suffix == ".gz" else open
        with opener(file, "rt") as stream:
            events = json.load(stream).get("traceEvents", [])
        for event in sorted(events, key=lambda e: e.get("ts", 0)):
            cat, dur = event.get("cat", ""), float(event.get("dur", 0) or 0)
            if cat == "kernel":
                kernels[event["name"]][0] += 1
                kernels[event["name"]][1] += dur
            elif cat == "gpu_user_annotation":
                regions[event["name"]] += dur
                if event["name"] == "gpu_model_runner: forward":
                    forwards.append(dur)
    total = sum(v[1] for v in kernels.values())
    top = sorted(kernels.items(), key=lambda kv: -kv[1][1])[:60]

    def kind(name):
        n = name.lower()
        for label, keys in (("fp4", ("fp4", "e2m1", "nvfp4")), ("fp8", ("fp8", "e4m3")),
                            ("gemm", ("gemm", "cutlass", "matmul", "sm90", "sm100", "sm120", "cublas"))):
            if any(k in n for k in keys):
                return label
        return "other"

    by_kind = defaultdict(float)
    for name, (_, dur) in kernels.items():
        by_kind[kind(name)] += dur
    return {
        "kernel_time_us": total,
        "kernel_time_by_kind_us": dict(by_kind),
        "top_kernels": [{"name": n, "calls": c, "time_us": d, "kind": kind(n)} for n, (c, d) in top],
        "region_gpu_us": dict(regions),
        "forward_gpu_us": forwards,
        "prefill_forward_us": forwards[0] if forwards else None,
        "decode_forward_us": sum(forwards[1:]) if len(forwards) > 1 else None,
        "files": [f.name for f in files],
    }


def compact_logprobs(entries):
    def tid(token):
        return int(token.split(":", 1)[1]) if isinstance(token, str) and token.startswith("token_id:") else token

    return [[tid(e["token"]), e["logprob"], [[tid(t["token"]), t["logprob"]] for t in e.get("top_logprobs", [])]]
            for e in entries or []]


# --------------------------------------------------------------------------- inputs

def fresh_cases(root, prepared):
    """E5 inputs: rows never used by the main study, frozen once before generation."""
    from datasets import Image as HFImage, load_dataset
    from huggingface_hub import hf_hub_download

    path = root / "data" / "ext_fresh.json"
    if path.exists():
        value = read_json(path)
        if value["hash"] != digest(value["cases"]) or value["preregistration"] != digest(PREREGISTRATION):
            raise RuntimeError("Fresh hold-out manifest changed; refusing to reuse it")
        return value["cases"]
    used = [c for part in ("development", "evaluation", "heldout", "calibration") for c in prepared[part]]
    used_images = {c["image_sha256"] for c in used}
    used_groups = {(c["task"], c["group"]) for c in used}
    used_pages = {Path(c["group"]).name for c in used if c["task"] == "omnidoc_ocr"}
    rev = prepared["revisions"]["datasets"]
    out = []
    for task, spec in PREREGISTRATION["tasks"].items():
        print(f"Freezing fresh {task} hold-out", flush=True)
        info = rev[task]
        chosen = []
        if task != "omnidoc_ocr":
            import hashlib

            dataset = load_dataset(info["id"], split=info["split"], revision=info["revision"])
            dataset = dataset.cast_column("image", HFImage(decode=False))
            groups, candidates = set(), []
            for index, doc in enumerate(dataset):
                raw = doc["image"]
                if doc.get("image_id") is not None:
                    group = str(doc["image_id"])
                elif isinstance(raw, dict) and raw.get("bytes"):
                    group = hashlib.sha256(raw["bytes"]).hexdigest()
                else:
                    group = str(raw.get("path") if isinstance(raw, dict) else index)
                if group in groups or (task, group) in used_groups:
                    continue
                groups.add(group)
                candidates.append({"index": index, "group": group,
                                   "stratum": str(doc.get("category", doc.get("type", "all")))})
            for item in balanced(candidates, min(len(candidates), spec["sample_cap"]), SEED + 5):
                doc = dict(dataset[item["index"]])
                target = doc["answer"]
                if isinstance(target, list):
                    target = target[0]
                identity = str(doc.get("question_id", doc.get("index", doc.get("id", item["index"]))))
                case = save_case(root, task, doc, image_of(doc["image"]), target,
                                 "fresh-" + identity + "@row" + str(item["index"]), item["stratum"], item["group"])
                if case["image_sha256"] not in used_images:
                    used_images.add(case["image_sha256"])
                    chosen.append(case)
        else:
            annotations = Path(hf_hub_download(info["id"], "OmniDocBench.json", repo_type="dataset",
                                               revision=info["revision"]))
            pages, candidates = json.loads(annotations.read_text()), []
            for index, page in enumerate(pages):
                name = Path(page["page_info"]["image_path"]).name
                if name in used_pages:
                    continue
                blocks = [b for b in page.get("layout_dets", [])
                          if b.get("category_type") == "text_block" and not b.get("ignore")
                          and len(str(b.get("text", ""))) >= 100
                          and b.get("attribute", {}).get("text_rotate", "normal") in ("normal", "horizontal")
                          and len(b.get("poly", [])) == 8]
                if not blocks:
                    continue
                attr = page["page_info"].get("page_attribute", {})
                candidates.append({"index": index, "block": max(blocks, key=lambda x: len(x["text"])),
                                   "group": name,
                                   "stratum": f"{attr.get('language', 'unknown')}/{attr.get('data_source', 'unknown')}"})
            files = None
            for item in balanced(candidates, min(len(candidates), spec["sample_cap"]), SEED + 5):
                name = item["group"]
                try:
                    local = hf_hub_download(info["id"], f"images/{name}", repo_type="dataset",
                                            revision=info["revision"])
                except Exception:
                    from huggingface_hub import HfApi

                    files = files or HfApi().list_repo_files(info["id"], repo_type="dataset",
                                                             revision=info["revision"])
                    matches = [f for f in files if f.endswith("/" + name)]
                    if len(matches) != 1:
                        continue  # unresolved name; recorded by the count, never replaced
                    local = hf_hub_download(info["id"], matches[0], repo_type="dataset", revision=info["revision"])
                image = image_of(str(local))
                points = item["block"]["poly"]
                box = (max(0, math.floor(min(points[0::2]))), max(0, math.floor(min(points[1::2]))),
                       min(image.width, math.ceil(max(points[0::2]))), min(image.height, math.ceil(max(points[1::2]))))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                case = save_case(root, task, {}, image.crop(box), item["block"]["text"],
                                 f"fresh-{name}:{item['block'].get('anno_id')}", item["stratum"], name)
                if case["image_sha256"] not in used_images:
                    used_images.add(case["image_sha256"])
                    chosen.append(case)
        out += chosen
        print(f"  {task}: {len(chosen)} fresh inputs", flush=True)
    atomic_json(path, {"preregistration": digest(PREREGISTRATION), "cases": out, "hash": digest(out),
                       "frozen_at": time.time()})
    return out


def sweep_cases(root, prepared):
    """E4 inputs: 4 OCR crops and 4 charts from the timing subset, resized to target token counts."""
    path = root / "data" / "ext_sweep.json"
    if path.exists():
        return read_json(path)
    import hashlib

    picks = []
    for task in ("omnidoc_ocr", "chartqa"):
        picks += [c for c in prepared["evaluation"] if c["task"] == task and c.get("timing_repeat")][:4]
    out = []
    for case in picks:
        image = image_of(str(root / case["image"]))
        for tokens in E4_TOKENS:
            scale = math.sqrt(tokens * 1024 / (image.width * image.height))
            size = (max(32, round(image.width * scale)), max(32, round(image.height * scale)))
            resized = image.resize(size, resample=3)  # bicubic
            sha = hashlib.sha256(str(resized.size).encode() + resized.tobytes()).hexdigest()
            dest = root / "data" / "ext_images" / f"{sha}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                resized.save(dest)
            out.append({"id": f"e4/{case['id']}/{tokens}", "task": "e4", "source": case["id"],
                        "source_task": case["task"], "target_tokens": tokens, "size": list(size),
                        "image": str(dest.relative_to(root)), "image_sha256": sha, "prompt": E4_PROMPT,
                        "group": case["group"], "stratum": case["stratum"]})
    atomic_json(path, out)
    return out


# --------------------------------------------------------------------------- worker

class Extension:
    mode = "extend"
    def __init__(self, root, identity, prepared, base):
        self.root, self.identity, self.prepared, self.base = root, identity, prepared, base
        self.db = Store(root / "results" / "ext")
        self.scorers = Scorers(root)
        self.progress_path = root / "results" / "ext" / "progress.json"
        self.progress = read_json(self.progress_path, {})
        self.profile_dir = root / "results" / "ext" / "traces"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        os.environ["QSPEC_PROFILE_DIR"] = str(self.profile_dir)
        # Named runner regions (forward, draft, sample) in traces; uniform tiny CPU cost in every ext session.
        os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(base))
        self.image_pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        cases = {c["id"]: c for part in ("development", "evaluation", "heldout") for c in prepared[part]}
        self.cases = cases
        self.timing = [c for c in prepared["evaluation"] if c.get("timing_repeat")]
        self.e1_cases = read_json(Path(__file__).with_name("ext_e1_cases.json"))["cases"]

    # ---- bookkeeping
    def mark(self, item, state, **kw):
        self.progress[item] = {"state": state, "time": time.time(), **kw}
        atomic_json(self.progress_path, self.progress)
        done = sum(v["state"] == "done" for v in self.progress.values())
        atomic_json(self.root / "status.json", {
            "state": "running", "mode": self.mode, "stage": f"{item}: {state}",
            "saved_records": self.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0],
            "jobs_done": done, "jobs_total": len(self.plan()),
        })

    def check_stop(self):
        if (self.root / "STOP").exists():
            raise KeyboardInterrupt

    def record(self, server, case, exp, rep=0, tag=None, extra=None, prefix=None):
        self.check_stop()
        key = digest([exp, server.policy, server.k, case["id"], rep, tag])
        old = self.db.get(key)
        if old:
            return old
        try:
            result = server.generate(case, prefix=prefix, extra=extra)
        except Exception as error:
            self.db.put(f"failed:{key}:{time.time_ns()}", {
                "exp": exp, "policy": server.policy, "k": server.k, "case_id": case["id"], "rep": rep,
                "tag": tag, "error": str(error), "traceback": traceback.format_exc(), "saved_at": time.time()})
            self.failures += 1
            if self.failures >= 5:
                raise RuntimeError(f"5 consecutive request failures in {server.policy} K={server.k}: {error}")
            return None
        self.failures = 0
        result["logprobs"] = compact_logprobs(result.get("logprobs"))
        if case["task"] in TASKS:
            result["score"] = self.scorers.score(case, result["text"])
        if result.get("prompt_ids"):
            result["image_tokens"] = result["prompt_ids"].count(self.image_pad)
        result.update(exp=exp, policy=server.policy, k=server.k, case_id=case["id"], task=case["task"],
                      rep=rep, tag=tag, identity=self.identity, saved_at=time.time(),
                      fixed_prefix=prefix, extra=extra)
        self.db.put(key, result)
        return result

    def profile(self, server, label, case, extra=None):
        key = digest(["profile", server.policy, server.k, label])
        if self.db.get(key) or not os.environ.get("QSPEC_PROFILE_DIR"):
            return
        before = set(self.profile_dir.rglob("*"))
        try:
            post(server, "/start_profile")
            result = server.generate(case, extra=extra)
            post(server, "/stop_profile")
            deadline = time.time() + PROFILE_WAIT_S
            files = []
            while time.time() < deadline:
                files = [f for f in set(self.profile_dir.rglob("*")) - before
                         if f.is_file() and (f.name.endswith(".json") or f.name.endswith(".json.gz"))]
                if files:
                    time.sleep(5)  # let the writer finish
                    break
                time.sleep(2)
            summary = summarize_trace(sorted(files)) if files else {"error": "no trace written"}
            for f in files:
                f.unlink(missing_ok=True)
        except Exception as error:
            summary, result = {"error": str(error), "traceback": traceback.format_exc()}, None
        self.db.put(key, {"exp": "profile", "label": label, "policy": server.policy, "k": server.k,
                          "case_id": case["id"], "trace": summary, "saved_at": time.time(),
                          "native_metrics": (result or {}).get("native_metrics"),
                          "usage": (result or {}).get("usage"), "image_tokens":
                          (result or {}).get("prompt_ids", []).count(self.image_pad) if result else None})

    # ---- experiment bodies (each is resumable request by request)
    def e1(self, server):
        cases = [{**self.cases[c["case_id"]], "_e1": c} for c in self.e1_cases]
        for rep in (0, 1):
            for case in cases:
                self.record(server, case, "e1", rep, extra=LOGPROBS)
        for case in cases:
            j = case["_e1"]["j"]
            prefix = self.tokenizer.decode(case["_e1"]["old_ar_ids"][:j], skip_special_tokens=False)
            self.record(server, {**case, "max_tokens": 4}, "e1_replay", 0, tag=j, extra=LOGPROBS, prefix=prefix)

    def e3(self, server, quality=False):
        cohort = self.prepared["evaluation"] if quality else self.timing
        for rep in range(3):
            for case in (cohort if rep == 0 else self.timing):
                self.record(server, case, "e3", rep)
        ocr = next(c for c in self.timing if c["task"] == "omnidoc_ocr")
        self.profile(server, "e3_ocr128", {**ocr, "max_tokens": 128})

    def e4(self, server):
        sweep = sweep_cases(self.root, self.prepared)
        for case in sweep:
            for length in E4_LENGTHS:
                self.record(server, {**case, "max_tokens": length}, "e4", tag=f"L{length}",
                            extra={"ignore_eos": True, "min_tokens": length})
            self.record(server, {**case, "max_tokens": 1024}, "e4", tag="natural")
        first = sweep[0]["source"]
        for tokens, length in E4_PROFILE:
            case = next(c for c in sweep if c["source"] == first and c["target_tokens"] == tokens)
            self.profile(server, f"e4_T{tokens}_L{length}", {**case, "max_tokens": length},
                         extra={"ignore_eos": True, "min_tokens": length})

    def e5(self, server):
        for case in fresh_cases(self.root, self.prepared):
            self.record(server, case, "e5")

    def e2(self, server):
        for rep in range(3):
            for case in (self.prepared["evaluation"] if rep == 0 else self.timing):
                self.record(server, case, "e2", rep)
        ocr = next(c for c in self.timing if c["task"] == "omnidoc_ocr")
        self.profile(server, "e2_ocr128", {**ocr, "max_tokens": 128})

    # ---- schedule: one server per (policy, K); K order alternates to balance drift
    def plan(self):
        p = []
        p.append(("P0", K, ["e1", "e5", "e3", "e4"]))
        p.append(("P0", 0, ["e1", "e3", "e4"]))
        p.append(("P7", 0, ["e4"]))
        p.append(("P7", K, ["e5", "e4"]))
        p.append(("P1", K, ["e5", "e3", "e4"]))
        p.append(("P1", 0, ["e3", "e4"]))
        p.append(("P2", 0, ["e3"]))
        p.append(("P2", K, ["e3"]))
        p.append(("P3", K, ["e3"]))
        p.append(("P3", 0, ["e3"]))
        p.append(("F1", 0, ["e3q"]))
        p.append(("F1", K, ["e3q"]))
        p.append(("S0", K, ["e2"]))
        p.append(("S0", 0, ["e2"]))
        p.append(("S1", 0, ["e2"]))
        p.append(("S1", K, ["e2"]))
        p.append(("S7", K, ["e2"]))
        p.append(("S7", 0, ["e2"]))
        return [f"{pol}:K{k}:{job}" for pol, k, jobs in p for job in jobs]

    def model_for(self, policy):
        if policy in POLICIES:
            return quant.ensure_export(self.root, self.base, policy)
        if policy == "F1":
            return ext_export(self.root, self.base, "F1", read_json(self.root / "revisions.json")["model"]["revision"])
        small = small_baseline()
        return small if policy == "S0" else ext_export(self.root, small, policy, SMALL["revision"])

    def start(self, model, policy, k):
        """Start a server and report the real root cause from its log if it fails.

        Profiling is dropped and the start retried only when the profiler itself is the cause.
        """
        server = runtime.Server(self.root, model, policy, k, self.identity)
        try:
            return server.__enter__()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            cause = root_cause(server.folder / "server.log") or str(error)
            if os.environ.get("QSPEC_PROFILE_DIR") and "profil" in cause.lower():
                print(f"server failed because of the profiler ({cause}); retrying without it", flush=True)
                os.environ.pop("QSPEC_PROFILE_DIR")
                return self.start(model, policy, k)
            raise ServerStartError(f"{policy} K={k} server did not start: {cause}") from error

    def run(self):
        sessions = []
        for item in self.plan():
            pol, k, job = item.split(":")
            key = (pol, int(k[1:]))
            if not sessions or sessions[-1][0] != key:
                sessions.append((key, []))
            sessions[-1][1].append((item, job))
        self.write_meta()
        for (policy, k), jobs in sessions:
            pending = [(i, j) for i, j in jobs if self.progress.get(i, {}).get("state") != "done"]
            if not pending:
                continue
            try:
                self.mark(pending[0][0], "preparing model")
                model = self.model_for(policy)
                self.mark(pending[0][0], "starting server")
                print(f"\n=== {policy} K={k}: {[j for _, j in pending]}", flush=True)
                with contextlib.closing(self.start(model, policy, k)) as server:
                    for task in TASKS:  # one untimed warm-up per task
                        try:
                            server.generate(next(c for c in self.prepared["development"] if c["task"] == task))
                        except Exception as error:
                            print(f"  warm-up {task} failed: {error}", flush=True)
                    for item, job in pending:
                        self.failures = 0
                        self.mark(item, "running")
                        started = time.time()
                        try:
                            ((lambda s: self.e3(s, quality=True)) if job == "e3q" else getattr(self, job))(server)
                            self.mark(item, "done", seconds=time.time() - started)
                            print(f"  done {item} in {(time.time() - started) / 60:.1f} min", flush=True)
                        except KeyboardInterrupt:
                            raise
                        except Exception as error:
                            traceback.print_exc()
                            self.mark(item, "failed", error=str(error)[:2000])
                            if "consecutive request failures" in str(error):
                                break  # the server is unhealthy; move to the next session
            except KeyboardInterrupt:
                raise
            except Exception as error:
                traceback.print_exc()
                for item, _ in pending:
                    if self.progress.get(item, {}).get("state") != "done":
                        self.mark(item, "failed", error=str(error)[:2000])
        failed = [i for i in self.plan() if self.progress.get(i, {}).get("state") != "done"]
        atomic_json(self.root / "status.json", {
            "state": "complete" if not failed else "partial", "mode": self.mode,
            "stage": "All extension jobs finished." if not failed else f"Finished with {len(failed)} job(s) not done: {failed}",
            "saved_records": self.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0],
        })

    def write_meta(self):
        sizes = {}
        for policy in ("P0", "P1", "P2", "P3", "P7", "F1", "S0", "S1", "S7"):
            path = self.base if policy == "P0" else self.root / "models" / policy
            if policy == "S0":
                continue
            if path.exists():
                sizes[policy] = model_bytes(path)
        atomic_json(self.root / "results" / "ext" / "meta.json", {
            "preregistration": PREREGISTRATION, "preregistration_sha256": digest(PREREGISTRATION),
            "image_pad_id": self.image_pad, "k": K, "second_model": SMALL,
            "checkpoint_bytes": sizes, "plan": self.plan(),
            "e4": {"tokens": E4_TOKENS, "lengths": E4_LENGTHS, "prompt": E4_PROMPT, "profile": E4_PROFILE},
        })


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        quant.exporter_main(*(Path(x).resolve() for x in sys.argv[2:6]))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", default="extend")
    args = parser.parse_args()
    root = args.root.resolve()

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    worker = None
    try:
        atomic_json(root / "protocol.json", PROTOCOL)
        identity = environment(root)
        atomic_json(root / "status.json", {"state": "running", "mode": "extend",
                                           "stage": "checking frozen inputs"})
        prepared = prepare(root)
        base = baseline(root)
        worker = Extension(root, identity, prepared, base)
        worker.run()
        worker.write_meta()
    except KeyboardInterrupt:
        atomic_json(root / "status.json", {"state": "stopped", "mode": "extend",
                                           "stage": "Stopped; completed requests retained"})
    except Exception as error:
        atomic_json(root / "status.json", {"state": "failed", "mode": "extend", "stage": str(error)})
        traceback.print_exc()
        raise
    finally:
        if worker:
            worker.db.close()


if __name__ == "__main__":
    main()
