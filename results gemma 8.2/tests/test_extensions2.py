"""CPU checks for the extra model families (Gemma FP8, GLM-OCR, EXAONE)."""
import json

from qspec_hybrid import core, quant


def test_family_policies_do_not_touch_protocol():
    from qspec_hybrid import extensions as ext
    from qspec_hybrid import extensions2 as e2

    assert core.digest(core.PROTOCOL) == "9a6daff6ec06e087424c47d32d78420fa26de81c4de6735b5d7cdfa399f3ca6a"
    assert "GF1" in quant.POLICIES and "GF1" not in core.POLICIES
    for policy in core.POLICIES:
        assert ext.ext_recipe(policy)["quantize"] == quant.recipe(policy)["quantize"]
    x1 = ext.ext_recipe("X1")["quantize"]["quant_cfg"]
    assert x1[:-1] == quant.recipe("P1")["quantize"]["quant_cfg"]
    assert x1[-1] == {"quantizer_name": "*audio*", "enable": False}
    # FP8 policies: the same rules with every NVFP4 import swapped for FP8, nothing else
    gf4 = ext.ext_recipe("GF4")["quantize"]["quant_cfg"]
    p4 = quant.recipe("P4")["quantize"]["quant_cfg"]
    assert len(gf4) == len(p4) + 1
    for new, old in zip(gf4, p4):
        assert new["quantizer_name"] == old["quantizer_name"]
        assert new.get("cfg") == ({"$import": "fp8"} if old.get("cfg") else None) or new.get("enable") is False
    assert {"GF1", "GF4", "OF1", "XF1"} <= quant.FP8_FFN_POLICIES
    assert not ({"O1", "O4", "X1"} & quant.FP8_FFN_POLICIES)
    assert ext.EXPORT_MODULE == "qspec_hybrid.extensions2"


def test_classify_family_names():
    assert quant.classify("model.language_model.layers.0.mlp.gate_proj.weight") == "D"
    assert quant.classify("model.language_model.layers.3.self_attn.q_proj.weight") == "A"
    assert quant.classify("model.visual.blocks.0.mlp.gate_proj.weight") == "protected"
    assert quant.classify("model.audio_tower.layers.0.mlp.gate_proj.weight") == "protected"
    assert quant.classify("language_model.model.layers.0.mlp.gate_up_proj") == "D"


def test_validate_without_linear_attention():
    from qspec_hybrid import extensions2 as e2

    def module(group, quantized):
        return {"scope": "target", "group": group, "quantized": quantized, "kernel": "CutlassFp8"}
    audit = {"kv_cache_dtype": "torch.bfloat16", "mtp_found": True,
             "modules": [module("D", True), module("A", False), module("protected", False)]}
    e2.validate2([audit], "GF1", 2)
    e2.validate2([audit], "O1", 2)
    try:
        e2.validate2([audit], "GF4", 2)
        raise AssertionError("A should be quantized under GF4")
    except RuntimeError:
        pass


def test_command_for_gemma_and_exaone():
    from qspec_hybrid import extensions2 as e2

    e2.CURRENT.update(family="gemma", assistant="/x/assistant")
    args = e2.runtime.command("/m", 8000, 2)
    assert "--mm-processor-kwargs" not in args
    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec == {"method": "mtp", "model": "/x/assistant", "num_speculative_tokens": 2}
    e2.CURRENT.update(family="exaone", assistant=None)
    args = e2.runtime.command("/m", 8000, 2)
    assert "--mm-processor-kwargs" in args
    assert json.loads(args[args.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 2}
    e2.CURRENT.update(family=None, assistant=None)


def test_external_drafter_only_for_gemma():
    assert {"G0", "GF1", "GF4"} <= quant.EXTERNAL_DRAFTER_POLICIES
    assert not ({"P0", "P1", "P7", "S1", "X1", "XF1", "O0", "O1", "OF1", "O4"} & quant.EXTERNAL_DRAFTER_POLICIES)


REAL_LOG = """(EngineCore pid=3775) ERROR 11-10 07:13:18 [core.py:1433]   File "kv_cache_utils.py", line 925, in _check_enough_kv_cache_memory
(EngineCore pid=3775) ERROR 11-10 07:13:18 [core.py:1433]     raise ValueError(
(EngineCore pid=3775) ERROR 11-10 07:13:18 [core.py:1433] ValueError: To serve at least one request with the model's max seq len (16384), (3.07 GiB KV cache is needed, which is larger than the available KV cache memory (2.0 GiB). Based on the available memory, the estimated maximum model length is 6208.
Process EngineCore:
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


def test_root_cause_finds_real_error_not_wrapper(tmp_path):
    from qspec_hybrid import extensions as ext

    log = tmp_path / "server.log"
    log.write_text(REAL_LOG)
    cause = ext.root_cause(log)
    assert cause.startswith("ValueError: To serve at least one request")
    assert "Engine core initialization failed" not in cause
    assert ext.root_cause(tmp_path / "missing.log") is None


def test_kv_needed_parses_vllm_message():
    from qspec_hybrid import extensions2 as e2

    assert e2.kv_needed(REAL_LOG) == 3.07
    assert e2.kv_needed("NotImplementedError: something else") is None


def test_family_kv_budget_and_qwen_untouched():
    from qspec_hybrid import core
    from qspec_hybrid import extensions2 as e2

    def value(args, flag):
        return args[args.index(flag) + 1]

    e2.CURRENT.update(family=None, assistant=None, kv_gib=None)
    qwen = e2.runtime.command("/m", 8000, 2)
    assert value(qwen, "--kv-cache-memory-bytes") == str(core.PROTOCOL["kv_cache_bytes"])
    assert value(qwen, "--gpu-memory-utilization") == "0.85"
    for family, util, gib in (("gemma", "0.85", 8), ("exaone", "0.9", 8), ("glmocr", "0.85", 4)):
        e2.CURRENT.update(family=family, assistant=None, kv_gib=e2.FAMILIES[family]["kv_gib"])
        args = e2.runtime.command("/m", 8000, 0)
        assert int(value(args, "--kv-cache-memory-bytes")) == gib * 2**30 > 3.07 * 2**30
        assert value(args, "--gpu-memory-utilization") == util
    assert "--mm-processor-kwargs" not in args  # GLM-OCR uses its own processor defaults
    e2.CURRENT.update(family=None, assistant=None, kv_gib=None)


def test_failed_export_leaves_no_partial_directory(tmp_path, monkeypatch):
    import subprocess

    from qspec_hybrid import extensions as ext
    from qspec_hybrid.core import atomic_json

    (tmp_path / "data").mkdir()
    atomic_json(tmp_path / "data" / "prepared.json", {"calibration": []})

    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "export")

    monkeypatch.setattr(ext.subprocess, "run", boom)
    monkeypatch.setattr(ext.shutil, "disk_usage", lambda p: type("U", (), {"free": 10**13})())
    try:
        ext.ext_export(tmp_path, tmp_path / "base", "GF1", "rev")
        raise AssertionError("export should have failed")
    except subprocess.CalledProcessError:
        pass
    assert not [p for p in (tmp_path / "models").iterdir()], "partial export directory left behind"


# ----------------------------------------------------------------------------- GLM-OCR

def test_glm_mtp_layer_is_protected(monkeypatch):
    monkeypatch.setenv("QSPEC_MTP_PREFIXES", "language_model.layers.16.")
    assert quant.classify("model.language_model.layers.16.mlp.gate_up_proj.weight") == "protected"
    assert quant.classify("model.language_model.layers.16.eh_proj.weight") == "protected"
    assert quant.classify("model.language_model.layers.15.mlp.gate_up_proj.weight") == "D"
    assert quant.classify("model.language_model.layers.1.self_attn.q_proj.weight") == "A"
    # layer 1 and 16 share a prefix only without the trailing dot; the dot is what keeps 1 quantizable
    assert quant.classify("model.language_model.layers.1.mlp.gate_up_proj.weight") == "D"
    monkeypatch.delenv("QSPEC_MTP_PREFIXES")
    assert quant.classify("model.language_model.layers.16.mlp.gate_up_proj.weight") == "D"


def test_glm_recipe_quantizes_fused_ffn_and_protects_drafter():
    from fnmatch import fnmatch

    from qspec_hybrid import extensions as ext

    def enabled(rules, name):
        state = None
        for rule in rules:  # last match wins, as in ModelOpt
            if fnmatch(name, rule["quantizer_name"]):
                state = rule.get("cfg", {}).get("$import") if rule.get("enable", True) else None
        return state

    for policy, fmt in (("O1", "nvfp4"), ("OF1", "fp8"), ("O4", "nvfp4")):
        rules = ext.ext_recipe(policy)["quantize"]["quant_cfg"]
        mod = "model.language_model.layers.3.mlp.gate_up_proj."
        assert enabled(rules, mod + "weight_quantizer") == fmt
        assert enabled(rules, mod + "input_quantizer") == fmt
        assert enabled(rules, "model.language_model.layers.3.mlp.down_proj.weight_quantizer") == fmt
        assert enabled(rules, "model.language_model.layers.16.mlp.gate_up_proj.weight_quantizer") is None
        assert enabled(rules, "model.visual.blocks.0.mlp.gate_proj.weight_quantizer") is None
        assert enabled(rules, "model.visual.merger.gate_proj.weight_quantizer") is None
        assert enabled(rules, "lm_head.weight_quantizer") is None
        attn = enabled(rules, "model.language_model.layers.3.self_attn.q_proj.weight_quantizer")
        assert attn == ("fp8" if policy == "O4" else None)


def test_glm_drafter_excluded_under_vllm_names():
    from fnmatch import fnmatch

    from qspec_hybrid import extensions2 as e2

    patterns = e2.FAMILIES["glmocr"]["extra_excludes"]
    for name in ("model.layers.16.mlp.gate_up_proj", "model.layers.16.self_attn.qkv_proj",
                 "model.layers.16.shared_head.head"):
        assert any(fnmatch(name, p) for p in patterns), name
    for name in ("language_model.model.layers.1.mlp.gate_up_proj", "language_model.model.layers.15.self_attn.qkv_proj"):
        assert not any(fnmatch(name, p) or p.rstrip("*") in name for p in patterns), name


# ----------------------------------------------------------------------------- quality gate

def _result(score, higher=True, finish="stop", text="ok"):
    return {"score": {"value": score, "higher_is_better": higher}, "finish_reason": finish, "text": text}


def test_gate_bf16_floor():
    from qspec_hybrid import extensions2 as e2

    good = {"textvqa": [_result(0.7)] * 8, "chartqa": [_result(0.5)] * 8,
            "mmstar": [_result(0.6)] * 8, "omnidoc_ocr": [_result(0.15, False)] * 8}
    assert e2.gate_verdict("X0", 0, good)[0]
    garbage = {"textvqa": [_result(0.0, finish="length")] * 8, "chartqa": [_result(0.0)] * 8,
               "mmstar": [_result(0.0, finish="length")] * 8, "omnidoc_ocr": [_result(1.0, False, "length")] * 8}
    passed, line = e2.gate_verdict("X0", 0, garbage)
    assert not passed and "FAIL" in line
    # one weak task out of four is not enough to stop a family
    one_bad = {**good, "chartqa": [_result(0.0)] * 8}
    assert e2.gate_verdict("X0", 0, one_bad)[0]
    # OCR only family: a single failing task stops it
    assert not e2.gate_verdict("O0", 0, {"omnidoc_ocr": [_result(0.95, False)] * 8})[0]
    assert e2.gate_verdict("O0", 0, {"omnidoc_ocr": [_result(0.2, False)] * 8})[0]


def test_gate_quantized_vs_bf16():
    from qspec_hybrid import extensions2 as e2

    ref = {"textvqa": [_result(0.7)] * 8, "chartqa": [_result(0.5)] * 8,
           "mmstar": [_result(0.6)] * 8, "omnidoc_ocr": [_result(0.15, False)] * 8}
    small_loss = {"textvqa": [_result(0.6)] * 8, "chartqa": [_result(0.4)] * 8,
                  "mmstar": [_result(0.5)] * 8, "omnidoc_ocr": [_result(0.2, False)] * 8}
    assert e2.gate_verdict("XF1", 2, small_loss, ref)[0]
    collapse = {"textvqa": [_result(0.05)] * 8, "chartqa": [_result(0.0, finish="length")] * 8,
                "mmstar": [_result(0.6)] * 8, "omnidoc_ocr": [_result(1.0, False, "length")] * 8}
    assert not e2.gate_verdict("XF1", 2, collapse, ref)[0]


def test_plans_start_with_bf16_reference_and_cover_both_k():
    from qspec_hybrid import extensions2 as e2

    for mode, plan in e2.PLANS.items():
        first = plan[0]
        assert not e2.POLICY[first[0]][1] and first[1] == 0, mode
        families = {e2.POLICY[p][0] for p, _ in plan}
        assert len(families) == 1, mode
        for policy in {p for p, _ in plan}:
            assert sorted(k for p, k in plan if p == policy) == [0, 2], (mode, policy)


# ----------------------------------------------------------------------------- worker with a fake server

class FakeServer:
    def __init__(self, policy, k, texts):
        self.policy, self.k, self.texts, self.calls = policy, k, texts, []

    def generate(self, case, prefix=None, extra=None):
        self.calls.append((case, extra))
        text, finish = self.texts(case)
        return {"text": text, "finish_reason": finish, "prompt_ids": []}


class FakeScorers:
    def score(self, case, text):
        if case["task"] == "omnidoc_ocr":
            return {"value": 0.1 if text == "good" else 1.0, "higher_is_better": False}
        return {"value": 1.0 if text == "good" else 0.0, "higher_is_better": True}


def _worker(tmp_path, mode):
    from qspec_hybrid import extensions2 as e2
    from qspec_hybrid.core import TASKS, Store

    w = object.__new__(e2.Families)
    w.root, w.mode, w.identity, w.image_pad, w.failures = tmp_path, mode, {}, 0, 0
    w.db = Store(tmp_path / "results" / "ext")
    w.scorers = FakeScorers()
    w.progress = {}
    evaluation = [{"id": f"{t}-{i}", "task": t, "prompt": "orig", "max_tokens": 16, "image": "x.png",
                   "timing_repeat": i < 3} for t in TASKS for i in range(12)]
    w.prepared = {"evaluation": evaluation, "development": []}
    w.timing = [c for c in evaluation if c["timing_repeat"]]
    w.profile = lambda *a, **k: None
    return w


def test_exaone_requests_carry_presence_penalty_everywhere(tmp_path):
    w = _worker(tmp_path, "fam_exaone")
    server = FakeServer("X0", 0, lambda c: ("good", "stop"))
    w.fam(server)
    assert server.calls
    for case, extra in server.calls:
        assert extra["presence_penalty"] == 1.5
    forced = [extra for case, extra in server.calls if case["max_tokens"] == 512]
    assert forced and all(e["ignore_eos"] and e["presence_penalty"] == 1.5 for e in forced)
    w.db.close()


def test_glm_uses_its_prompt_and_only_ocr(tmp_path):
    w = _worker(tmp_path, "fam_glmocr")
    server = FakeServer("O0", 0, lambda c: ("good", "stop"))
    w.fam(server)
    assert {c["task"] for c, _ in server.calls} == {"omnidoc_ocr"}
    assert {c["prompt"] for c, _ in server.calls} == {"Text Recognition:"}
    assert all(extra in (None, {"ignore_eos": True, "min_tokens": 512}) for _, extra in server.calls)
    w.db.close()


def test_gate_stops_garbage_bf16_quickly_and_skips_family(tmp_path):
    import pytest

    from qspec_hybrid import extensions2 as e2

    w = _worker(tmp_path, "fam_exaone")
    server = FakeServer("X0", 0, lambda c: ("</think></think>", "length"))
    with pytest.raises(e2.GateFailed):
        w.fam(server)
    assert len(server.calls) == 4 * e2.GATE_PER_TASK  # stopped after the gate, nothing long ran
    for policy in ("X0", "X1", "XF1"):  # every later session of this family is skipped before any GPU work
        with pytest.raises(e2.GateFailed, match="skipped by the quality gate"):
            w.model_for(policy)
    w.db.close()


def test_gate_stops_collapsed_quantized_policy_only(tmp_path):
    import pytest

    from qspec_hybrid import extensions2 as e2

    w = _worker(tmp_path, "fam_gemma")
    w.fam(FakeServer("G0", 0, lambda c: ("good", "stop")))
    bad = FakeServer("GF4", 2, lambda c: ("garbage", "length"))
    with pytest.raises(e2.GateFailed):
        w.fam(bad)
    assert len(bad.calls) == 4 * e2.GATE_PER_TASK
    with pytest.raises(e2.GateFailed, match="skipped"):
        w.model_for("GF4")
    ok = FakeServer("GF1", 2, lambda c: ("good", "stop"))
    w.fam(ok)  # a healthy policy in the same family still runs in full
    assert len(ok.calls) > 4 * e2.GATE_PER_TASK
    w.db.close()


def test_quantized_gate_without_bf16_reference_refuses(tmp_path):
    import pytest

    from qspec_hybrid import extensions2 as e2

    w = _worker(tmp_path, "fam_gemma")
    with pytest.raises(e2.GateFailed, match="reference"):
        w.fam(FakeServer("GF1", 0, lambda c: ("good", "stop")))
    w.db.close()


def test_resume_reuses_gate_answers(tmp_path):
    w = _worker(tmp_path, "fam_glmocr")
    w.fam(FakeServer("O0", 0, lambda c: ("good", "stop")))
    again = FakeServer("O0", 0, lambda c: ("good", "stop"))
    w.fam(again)
    assert again.calls == []  # everything was already saved; nothing is requested twice
    w.db.close()
