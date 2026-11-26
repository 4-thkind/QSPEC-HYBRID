"""More model families with native MTP. One notebook (mode) per family.

fam_gemma   Gemma 4 12B (Google), official MTP assistant drafter. FP8 policies, because the
            vLLM 0.31 NVFP4 path for this architecture produced degenerate text in v8.1.
fam_glmocr  GLM-OCR (Zhipu), small GLM decoder with a built in MTP layer, OCR only. Its 59k
            vocabulary makes the output head small, which tests the drafting cost mechanism.
fam_exaone  EXAONE 4.5 33B (LG), built in MTP layer. Pure greedy decoding degenerates on this
            model, so every EXAONE condition uses the vendor recommended presence penalty.

Same frozen inputs, calibration images, recipe rules, audits and timing protocol as the Qwen
runs. PROTOCOL is untouched. A quality gate on the first answers of every session stops a
broken model within minutes instead of hours.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

from . import extensions as ext
from . import quant, runtime
from .core import TASKS, atomic_json, read_json

FAMILIES = {
    "gemma": {
        "id": "google/gemma-4-12B-it", "revision": "43c9d014b231474fa7212dbb48a762566c2b255d",
        "assistant": ("google/gemma-4-12B-it-assistant", "46d4c6f13f0ac0ad827b915669b8df9b81c64c51"),
        "pixels": False, "need_gib": 60, "eval_per_task": 200, "tasks": TASKS,
        "kv_gib": 8, "gpu_util": 0.85,  # ordinary attention: 16k context needs about 3.1 GiB of cache
    },
    "glmocr": {
        "id": "zai-org/GLM-OCR", "revision": "2e85a62840ccac27daa451df36c736c4636b8628",
        "assistant": None, "pixels": False, "need_gib": 15, "eval_per_task": 200,
        "tasks": ("omnidoc_ocr",), "prompts": {"omnidoc_ocr": "Text Recognition:"},
        "mtp_prefixes": ("language_model.layers.16.",),  # drafter = decoder layer num_hidden_layers
        # vLLM names the GLM drafter model.layers.16.*, not its checkpoint names, so the export must
        # also exclude it under that name or vLLM would build low-bit layers for a BF16 drafter.
        "extra_excludes": ("model.layers.16*",),
        "fused_ffn": True, "kv_gib": 4, "gpu_util": 0.85,
    },
    "exaone": {
        "id": "LGAI-EXAONE/EXAONE-4.5-33B", "revision": "570aa4b15a4f45ba1133072b45f50198f6e3b4fd",
        "assistant": None, "pixels": True, "need_gib": 130, "eval_per_task": 100, "tasks": TASKS,
        "extra": {"presence_penalty": 1.5},  # vendor recommendation, same for every EXAONE condition
        "kv_gib": 8, "gpu_util": 0.90,
    },
}
# policy: (family, groups, feed forward format, Qwen recipe it is derived from)
POLICY = {
    "G0": ("gemma", (), None, None),
    "GF1": ("gemma", ("D",), "fp8", "P1"),
    "GF4": ("gemma", ("D", "A"), "fp8", "P4"),
    "O0": ("glmocr", (), None, None),
    "O1": ("glmocr", ("D",), "nvfp4", "P1"),
    "OF1": ("glmocr", ("D",), "fp8", "P1"),
    "O4": ("glmocr", ("D", "A"), "nvfp4", "P4"),
    "X0": ("exaone", (), None, None),
    "X1": ("exaone", ("D",), "nvfp4", "P1"),
    "XF1": ("exaone", ("D",), "fp8", "P1"),
}
# BF16 K=0 first in every family: it is the quality gate reference for every later session.
PLANS = {
    "fam_gemma": [("G0", 0), ("G0", 2), ("GF1", 2), ("GF1", 0), ("GF4", 0), ("GF4", 2)],
    "fam_glmocr": [("O0", 0), ("O0", 2), ("O1", 2), ("O1", 0), ("OF1", 0), ("OF1", 2), ("O4", 2), ("O4", 0)],
    "fam_exaone": [("X0", 0), ("X0", 2), ("X1", 2), ("X1", 0), ("XF1", 0), ("XF1", 2)],
}
MODES = tuple(PLANS)

ext.ALL_POLICIES.update({name: spec[1] for name, spec in POLICY.items()})  # same dict as quant.POLICIES
quant.FP8_FFN_POLICIES.update(name for name, spec in POLICY.items() if spec[2] == "fp8")
quant.EXTERNAL_DRAFTER_POLICIES.update(n for n, s in POLICY.items() if FAMILIES[s[0]]["assistant"])
ext.EXPORT_MODULE = "qspec_hybrid.extensions2"
_ext_recipe = ext.ext_recipe


def recipe2(policy):
    if policy not in POLICY:
        return _ext_recipe(policy)
    family, groups, ffn, base = POLICY[policy]
    spec = FAMILIES[family]
    value = quant.recipe(base)
    rules = value["quantize"]["quant_cfg"]
    if spec.get("fused_ffn"):
        # GLM stores gate and up as one matrix. Quantize it like the other D matrices, placed before
        # the protection rules because the last matching rule wins.
        template = [r for r in rules if r.get("quantizer_name", "").startswith("*layers.*.mlp.gate_proj")]
        first_protect = next(i for i, r in enumerate(rules) if r.get("quantizer_name") == "*mtp*")
        fused = [{**r, "quantizer_name": r["quantizer_name"].replace("gate_proj", "gate_up_proj")} for r in template]
        if not fused:
            raise RuntimeError(f"{policy}: no gate_proj rule to copy for the fused gate_up_proj")
        rules[first_protect:first_protect] = fused
    if ffn == "fp8":
        for rule in rules:
            if rule.get("cfg") == {"$import": "nvfp4"}:
                rule["cfg"] = {"$import": "fp8"}
    rules.append({"quantizer_name": "*audio*", "enable": False})
    for prefix in spec.get("mtp_prefixes", ()):
        rules.append({"quantizer_name": f"*{prefix}*", "enable": False})
    value["metadata"]["description"] = f"QSpec family run {policy} ({'+'.join(groups)} {ffn})"
    return value


ext.ext_recipe = recipe2


def validate2(audits, policy, k):
    """ext.validate_runtime, but a family without gated linear attention has no G group."""
    needed = ("D", "A") if policy in POLICY else ("D", "A", "G")
    if not audits:
        raise RuntimeError("No GPU precision audit returned")
    for audit in audits:
        if audit.get("kv_cache_dtype") != "torch.bfloat16":
            raise RuntimeError(f"KV storage is not BF16: {audit.get('kv_cache_dtype')}")
        if k and not audit["mtp_found"]:
            raise RuntimeError("MTP requested but no drafter model found")
        counts = {g: 0 for g in ("D", "A", "G")}
        for module in audit["modules"]:
            group = module["group"]
            if group in counts and module["scope"] == "target":
                counts[group] += 1
                if module["quantized"] != (group in ext.ALL_POLICIES[policy]):
                    raise RuntimeError(f"Wrong runtime precision: {module}")
                if module["quantized"]:
                    kernel = module.get("kernel", "NoneType")
                    if kernel == "NoneType" or any(x in kernel.lower() for x in ("emulation", "marlin")):
                        raise RuntimeError(f"Native low-bit kernel not established: {module}")
            elif module["quantized"]:
                raise RuntimeError(f"Protected module was quantized: {module}")
        if not all(counts[g] for g in needed):
            raise RuntimeError(f"Cannot establish runtime projection groups: {counts}")


runtime.validate_runtime = validate2
_command = ext.command
CURRENT = {"family": None, "assistant": None, "kv_gib": None}
LEARNED_KV = {}  # family -> budget that a retry proved necessary, reused by later sessions


def kv_needed(message):
    """GiB of KV cache vLLM said it needs, from its startup error, or None."""
    match = re.search(r"([0-9.]+) GiB KV cache is needed", message or "")
    return float(match.group(1)) if match else None


def command2(model, port, k, eager=False, smoke=False):
    args = _command(model, port, k, eager, smoke=smoke)
    family = CURRENT["family"]
    if family and not FAMILIES[family]["pixels"]:
        i = args.index("--mm-processor-kwargs")
        del args[i:i + 2]
    if family:
        # Qwen keeps the frozen 2 GiB budget. These families need a larger reservation, which
        # changes reserved memory only, not the work done per token.
        i = args.index("--kv-cache-memory-bytes")
        args[i + 1] = str(int((CURRENT["kv_gib"] or FAMILIES[family]["kv_gib"]) * 2**30))
        i = args.index("--gpu-memory-utilization")
        args[i + 1] = str(FAMILIES[family]["gpu_util"])
    if k and CURRENT["assistant"]:
        i = args.index("--speculative-config")
        args[i + 1] = json.dumps({"method": "mtp", "model": str(CURRENT["assistant"]), "num_speculative_tokens": k})
    return args


runtime.command = command2

PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken"]


def download(repo, revision, need_gib=0, root=None):
    from huggingface_hub import snapshot_download

    try:  # already complete in the cache: no disk check, no network
        return Path(snapshot_download(repo, revision=revision, allow_patterns=PATTERNS, local_files_only=True))
    except Exception:
        pass
    free = shutil.disk_usage(root or Path.home()).free / 2**30
    print(f"downloading {repo}", flush=True)
    if free < need_gib:
        raise RuntimeError(f"Only {free:.0f} GiB free; {repo} needs about {need_gib} GiB with its exports")
    try:
        return Path(snapshot_download(repo, revision=revision, allow_patterns=PATTERNS))
    except BaseException:
        # A failed (for example disk full) download must not keep its partial blobs, or there is
        # no room left to bundle the results that already finished.
        from huggingface_hub.constants import HF_HUB_CACHE

        shutil.rmtree(Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")), ignore_errors=True)
        raise


def family_base(root, family):
    spec = FAMILIES[family]
    os.environ["QSPEC_MTP_PREFIXES"] = ",".join(spec.get("mtp_prefixes", ()))  # also read by the exporter
    os.environ["QSPEC_EXTRA_EXCLUDES"] = ",".join(spec.get("extra_excludes", ()))
    path = download(spec["id"], spec["revision"], spec["need_gib"], root)
    if not spec["assistant"] and not any(quant.is_mtp_name(key) for key in quant.tensor_index(path)):
        raise RuntimeError(f"{spec['id']} checkpoint has no MTP weights")
    assistant = download(*spec["assistant"]) if spec["assistant"] else None
    return path, assistant


# --------------------------------------------------------------------------- quality gate
GATE_PER_TASK = 8
BF16_FLOOR = {"textvqa": 0.05, "chartqa": 0.05, "mmstar": 0.10}  # accuracy floors for BF16
OCR_CEILING = 0.80  # normalized edit distance; 1.0 means nothing right


class GateFailed(RuntimeError):
    pass


def _stats(results):
    cap = sum(r.get("finish_reason") == "length" for r in results) / len(results)
    score = sum(r["score"]["value"] for r in results) / len(results)
    return cap, score, results[0]["score"]["higher_is_better"]


def gate_verdict(policy, k, gate, reference=None):
    """(passed, one line summary). gate and reference map task -> results on the same cases."""
    failures, parts = [], []
    for task, results in gate.items():
        cap, score, higher = _stats(results)
        if reference is None:  # BF16: absolute sanity floor
            bad = cap > 0.6 or (score < BF16_FLOOR.get(task, 0) if higher else score > OCR_CEILING)
            parts.append(f"{task} {score:.2f} cap {cap:.0%}{' FAIL' if bad else ''}")
        else:  # quantized: collapse relative to BF16 on the same cases
            ref_cap, ref, _ = _stats(reference[task])
            collapsed = (ref >= 0.2 and score <= 0.25 * ref) if higher else score >= max(OCR_CEILING, ref + 0.4)
            bad = (cap >= 0.6 and ref_cap <= 0.3) or collapsed
            parts.append(f"{task} {score:.2f} vs BF16 {ref:.2f} cap {cap:.0%}{' FAIL' if bad else ''}")
        if bad:
            failures.append(task)
    passed = len(failures) < max(1, math.ceil(len(gate) / 2))
    return passed, f"gate {policy} K={k}: {'pass' if passed else 'FAIL'} | " + " | ".join(parts)


class Families(ext.Extension):
    mode = "fam_gemma"

    def plan(self):
        return [f"{pol}:K{k}:fam" for pol, k in PLANS[self.mode]]

    @property
    def broken_path(self):
        return self.root / "results" / "ext" / f"gate_{self.mode}.json"

    def broken(self):
        return read_json(self.broken_path, {})

    def model_for(self, policy):
        family, groups, *_ = POLICY[policy]
        spec = FAMILIES[family]
        broken = self.broken()
        if family in broken or policy in broken:
            raise GateFailed(f"skipped by the quality gate: {broken.get(family) or broken.get(policy)}")
        base, assistant = family_base(self.root, family)
        CURRENT.update(family=family, assistant=assistant,
                       kv_gib=max(spec["kv_gib"], LEARNED_KV.get(family, 0)))
        if spec["pixels"]:
            os.environ.pop("QSPEC_NO_PIXEL_KWARGS", None)
        else:
            os.environ["QSPEC_NO_PIXEL_KWARGS"] = "1"
        return base if not groups else ext.ext_export(self.root, base, policy, spec["revision"])

    def start(self, model, policy, k):
        """ext.Extension.start, plus an automatic resize if vLLM reports the KV budget is short."""
        family = POLICY[policy][0]
        for attempt in range(3):
            try:
                return self.family_settings(super().start(model, policy, k))
            except ext.ServerStartError as error:
                need = kv_needed(str(error))
                if need is None or attempt == 2:
                    raise
                CURRENT["kv_gib"] = max(CURRENT["kv_gib"] * 2, math.ceil(need * 1.5) + 1)
                LEARNED_KV[family] = CURRENT["kv_gib"]
                print(f"{policy} K={k}: KV cache needs {need:.2f} GiB; retrying with {CURRENT['kv_gib']} GiB",
                      flush=True)

    @staticmethod
    def family_settings(server):
        """Every request on this server (warm-up, gate, timing, profile) gets the family prompt and
        sampling settings, e.g. the EXAONE presence penalty."""
        spec = FAMILIES[POLICY[server.policy][0]]
        generate = server.generate

        def family_generate(case, prefix=None, extra=None):
            case = {**case, "prompt": spec.get("prompts", {}).get(case["task"], case["prompt"])}
            return generate(case, prefix=prefix, extra={**spec.get("extra", {}), **(extra or {})} or None)

        server.generate = family_generate
        return server

    def record(self, server, case, exp, rep=0, tag=None, extra=None, prefix=None):
        spec = FAMILIES[POLICY[server.policy][0]]
        case = {**case, "prompt": spec.get("prompts", {}).get(case["task"], case["prompt"])}
        merged = {**spec.get("extra", {}), **(extra or {})} or None
        return super().record(server, case, exp, rep, tag=tag, extra=merged, prefix=prefix)

    def reference(self, family, gate_ids):
        """BF16 K=0 gate answers of this family on the same cases."""
        bf16 = next(p for p, s in POLICY.items() if s[0] == family and not s[1])
        rows = {r["case_id"]: r for r in self.db.rows()
                if r.get("exp") == "fam" and r.get("policy") == bf16 and r.get("k") == 0
                and r.get("rep") == 0 and r.get("score")}
        return {task: [rows[i] for i in ids if i in rows] for task, ids in gate_ids.items()}

    def fam(self, server):
        family, groups, *_ = POLICY[server.policy]
        spec = FAMILIES[family]
        tasks = spec["tasks"]
        per_task = {t: [x for x in self.prepared["evaluation"] if x["task"] == t][:spec["eval_per_task"]]
                    for t in tasks}
        # 1. quality gate on the first answers of every task, before anything long runs
        gate_ids = {t: [c["id"] for c in cases[:GATE_PER_TASK]] for t, cases in per_task.items()}
        gate = {t: [r for r in (self.record(server, c, "fam", 0) for c in cases[:GATE_PER_TASK]) if r]
                for t, cases in per_task.items()}
        if not all(gate.values()):
            raise GateFailed(f"gate {server.policy} K={server.k}: every gate request failed for a task")
        reference = None
        if groups:
            reference = self.reference(family, gate_ids)
            if not all(len(reference[t]) == len(gate[t]) for t in gate):
                raise GateFailed(f"gate {server.policy} K={server.k}: BF16 K=0 reference answers missing")
        passed, line = gate_verdict(server.policy, server.k, gate, reference)
        print("  " + line, flush=True)
        if not groups and not server.k:
            for t, results in gate.items():  # eyeball check of the reference session
                print(f"  sample {t}: {results[0]['text'][:200]!r}", flush=True)
        if not passed:
            broken = self.broken()
            broken[family if not groups else server.policy] = line
            atomic_json(self.broken_path, broken)
            raise GateFailed(line)
        # 2. the full measurement (gate answers are reused, never requested twice)
        cohort = [c for t in tasks for c in per_task[t]]
        timing = [c for c in self.timing if c["task"] in tasks]
        ids = {c["id"] for c in cohort}
        cohort += [c for c in timing if c["id"] not in ids]
        for rep in range(3):
            for case in (cohort if rep == 0 else timing):
                self.record(server, case, "fam", rep)
        ocr = [c for c in timing if c["task"] == "omnidoc_ocr"][:8]
        for case in ocr:  # forced length for per token and per step costs
            self.record(server, {**case, "max_tokens": 512}, "fam_forced", tag="L512",
                        extra={"ignore_eos": True, "min_tokens": 512})
        self.profile(server, "fam_ocr128", {**ocr[0], "max_tokens": 128})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        quant.exporter_main(*(Path(x).resolve() for x in sys.argv[2:6]))
        return
    import argparse
    import signal
    import traceback

    from .core import PROTOCOL
    from .data import prepare
    from .runner import baseline, environment

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    worker = None
    try:
        atomic_json(root / "protocol.json", PROTOCOL)
        identity = environment(root)
        atomic_json(root / "status.json", {"state": "running", "mode": args.mode, "stage": "checking frozen inputs"})
        prepared = prepare(root)
        base = baseline(root)  # Qwen tokenizer only feeds image token counting in shared helpers
        worker = Families(root, identity, prepared, base)
        worker.mode = args.mode
        worker.run()
    except KeyboardInterrupt:
        atomic_json(root / "status.json", {"state": "stopped", "mode": args.mode,
                                           "stage": "Stopped; completed requests retained"})
    except Exception as error:
        atomic_json(root / "status.json", {"state": "failed", "mode": args.mode, "stage": str(error)})
        traceback.print_exc()
        raise
    finally:
        if worker:
            worker.db.close()


if __name__ == "__main__":
    main()
