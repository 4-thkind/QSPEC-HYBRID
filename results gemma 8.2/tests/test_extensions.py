"""CPU checks for the revision experiments: recipes, audits, trace parsing, protocol freeze."""

import gzip
import json

import pytest

from qspec_hybrid import core, quant


def test_protocol_and_main_policies_untouched():
    from qspec_hybrid import extensions as ext

    assert core.digest(core.PROTOCOL) == "9a6daff6ec06e087424c47d32d78420fa26de81c4de6735b5d7cdfa399f3ca6a"
    assert "F1" not in core.POLICIES and "F1" in quant.POLICIES
    for policy in core.POLICIES:  # existing export identities must not move
        assert ext.ext_recipe(policy)["quantize"] == quant.recipe(policy)["quantize"]


def test_f1_recipe_is_p1_with_fp8():
    from qspec_hybrid import extensions as ext

    f1, p1 = ext.ext_recipe("F1")["quantize"]["quant_cfg"], quant.recipe("P1")["quantize"]["quant_cfg"]
    assert len(f1) == len(p1)
    assert not any(r.get("cfg") == {"$import": "nvfp4"} for r in f1)
    assert sum(r.get("cfg") == {"$import": "fp8"} for r in f1) == sum(r.get("cfg") == {"$import": "nvfp4"} for r in p1) == 6
    assert ext.ext_recipe("S1")["quantize"] == quant.recipe("P1")["quantize"]


def test_f1_fp8_ffn_export_audit(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    import qspec_hybrid.extensions  # noqa: F401  (registers F1)

    original, quantized = tmp_path / "base", tmp_path / "quant"
    original.mkdir(); quantized.mkdir()
    values = {
        "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(2, 16, dtype=torch.bfloat16),
        "model.language_model.layers.3.self_attn.o_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.language_model.layers.0.linear_attn.out_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    save_file(values, original / "model.safetensors")
    fp8 = dict(values)
    fp8["model.language_model.layers.0.mlp.down_proj.weight"] = torch.ones(2, 16).to(torch.float8_e4m3fn)
    fp8["model.language_model.layers.0.mlp.down_proj.input_scale"] = torch.tensor(1.0)
    fp8["model.language_model.layers.0.mlp.down_proj.weight_scale"] = torch.tensor(1.0)
    save_file(fp8, quantized / "model.safetensors")
    assert quant.compare_export(quantized, original, "F1")["passed"]
    assert not quant.compare_export(quantized, original, "P1")["passed"]  # P1 still demands NVFP4


def test_trace_summary(tmp_path):
    from qspec_hybrid.extensions import summarize_trace

    events = [
        {"cat": "gpu_user_annotation", "name": "gpu_model_runner: forward", "ts": 1, "dur": 900},
        {"cat": "kernel", "name": "cutlass_sm120_fp4_gemm", "ts": 2, "dur": 300},
        {"cat": "kernel", "name": "flash_fwd", "ts": 3, "dur": 100},
        {"cat": "gpu_user_annotation", "name": "gpu_model_runner: forward", "ts": 5, "dur": 50},
        {"cat": "gpu_user_annotation", "name": "gpu_model_runner: draft", "ts": 6, "dur": 20},
    ]
    path = tmp_path / "t.pt.trace.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": events}, f)
    s = summarize_trace([path])
    assert s["prefill_forward_us"] == 900 and s["decode_forward_us"] == 50
    assert s["region_gpu_us"]["gpu_model_runner: draft"] == 20
    assert s["kernel_time_by_kind_us"]["fp4"] == 300


def test_logprob_compaction_and_e1_cases():
    from pathlib import Path
    from qspec_hybrid.extensions import compact_logprobs

    out = compact_logprobs([{"token": "token_id:7", "logprob": -0.1,
                             "top_logprobs": [{"token": "token_id:7", "logprob": -0.1},
                                              {"token": "token_id:9", "logprob": -2.4}]}])
    assert out == [[7, -0.1, [[7, -0.1], [9, -2.4]]]]
    cases = json.loads((Path(quant.__file__).with_name("ext_e1_cases.json")).read_text())["cases"]
    assert len(cases) == 62 and sum(c["role"] == "divergent" for c in cases) == 31
    assert all(0 < c["j"] < len(c["old_ar_ids"]) for c in cases)


def test_preregistration_frozen():
    from qspec_hybrid.extensions import PREREGISTRATION

    margins = {t: v["margin"] for t, v in PREREGISTRATION["tasks"].items()}
    assert margins == {"mmstar": 0.02, "chartqa": 0.02, "omnidoc_ocr": 0.01}
    assert core.digest(PREREGISTRATION)  # recorded in meta.json before fresh generation
