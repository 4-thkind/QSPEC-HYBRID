"""Real CPU safetensors audits, not GPU quantization/kernel validation."""

import pytest
from pathlib import Path

from qspec_hybrid.quant import audit_export, compare_export, restore_protected_tensors


@pytest.fixture
def checkpoint_pair(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    original, quantized = tmp_path / "base", tmp_path / "quant"
    original.mkdir()
    quantized.mkdir()
    values = {
        "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(
            2, 16, dtype=torch.bfloat16
        ),
        "model.language_model.layers.3.self_attn.o_proj.weight": torch.ones(
            2, 2, dtype=torch.bfloat16
        ),
        "model.language_model.layers.0.linear_attn.out_proj.weight": torch.ones(
            2, 2, dtype=torch.bfloat16
        ),
        "model.language_model.layers.0.linear_attn.A_log": torch.tensor(
            1.0, dtype=torch.float32
        ),
        "mtp.fc.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "visual.merger.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    save_file(values, original / "model.safetensors")
    compressed = dict(values)
    compressed["model.language_model.layers.0.mlp.down_proj.weight"] = torch.ones(
        2, 8, dtype=torch.uint8
    )
    # A scalar scale explicitly exercises the zero-dimensional tensor hashing path.
    compressed["model.language_model.layers.0.mlp.down_proj.input_scale"] = torch.tensor(1.0)
    compressed["model.language_model.layers.0.mlp.down_proj.weight_scale"] = torch.ones(2, 1, dtype=torch.float8_e4m3fn)
    compressed["model.language_model.layers.0.mlp.down_proj.weight_scale_2"] = torch.tensor(1.0)
    save_file(compressed, quantized / "model.safetensors")
    return original, quantized, compressed


def test_real_safetensor_storage_and_scalar_fingerprints(checkpoint_pair):
    original, quantized, _ = checkpoint_pair
    audit = audit_export(quantized, original, "P1")
    assert audit["stored_dtypes"]["D"] == ["U8"]
    assert len(audit["weights_content_sha256"]) == 64
    assert audit["mtp_tensor_count"] == 1
    assert (
        audit_export(quantized, original, "P1")["weights_content_sha256"]
        == audit["weights_content_sha256"]
    )


@pytest.mark.parametrize(
    "corruption", ["missing_mtp", "changed_protected", "wrong_selected_format"]
)
def test_real_safetensor_audit_rejects_invalid_export(checkpoint_pair, corruption):
    import torch
    from safetensors.torch import save_file

    original, quantized, values = checkpoint_pair
    if corruption == "missing_mtp":
        values.pop("mtp.fc.weight")
    elif corruption == "changed_protected":
        values["visual.merger.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    else:
        values["model.language_model.layers.0.mlp.down_proj.weight"] = torch.ones(
            2, 2, dtype=torch.bfloat16
        )
    save_file(values, quantized / "model.safetensors")
    with pytest.raises(RuntimeError):
        audit_export(quantized, original, "P1")


def test_disallowed_dtype_narrowing_is_reported_without_tolerance(checkpoint_pair):
    import torch
    from safetensors.torch import save_file

    original, quantized, values = checkpoint_pair
    values["model.language_model.layers.0.linear_attn.A_log"] = torch.tensor(
        1.0, dtype=torch.bfloat16
    )
    save_file(values, quantized / "model.safetensors")
    report = compare_export(quantized, original, "P1")
    assert report["comparison_counts"]["disallowed_dtype_only"] == 1
    assert report["disallowed_dtype_only"][0]["name"].endswith("A_log")
    with pytest.raises(RuntimeError, match="disallowed dtype conversion"):
        audit_export(quantized, original, "P1")


def test_lossless_widening_requires_an_explicit_record(checkpoint_pair):
    import torch
    from safetensors.torch import load_file, save_file

    original, quantized, values = checkpoint_pair
    source = load_file(original / "model.safetensors")
    source["model.language_model.layers.0.linear_attn.A_log"] = torch.tensor(
        1.0, dtype=torch.bfloat16
    )
    save_file(source, original / "model.safetensors")
    values["model.language_model.layers.0.linear_attn.A_log"] = torch.tensor(
        1.0, dtype=torch.float32
    )
    save_file(values, quantized / "model.safetensors")
    name = "model.language_model.layers.0.linear_attn.A_log"
    assert not compare_export(quantized, original, "P1")["passed"]
    report = compare_export(quantized, original, "P1", [name])
    assert report["passed"]
    assert report["lossless_dtype_only"] == [
        {
            "name": name,
            "group": "protected",
            "source_dtype": "torch.bfloat16",
            "export_dtype": "torch.float32",
            "source_shape": [],
            "export_shape": [],
            "explicitly_allowed": True,
        }
    ]


def test_generic_repair_restores_every_unselected_tensor_and_keeps_raw(checkpoint_pair):
    import json
    import torch
    from safetensors.torch import load_file, save_file

    original, quantized, values = checkpoint_pair
    source = load_file(original / "model.safetensors")
    source["model.language_model.layers.0.linear_attn.norm.weight"] = torch.tensor(
        [1.1, 2.2], dtype=torch.float32
    )
    source["model.language_model.layers.0.mlp.down_proj.bias"] = torch.tensor(
        [3.3, 4.4], dtype=torch.float32
    )
    save_file(source, original / "model.safetensors")
    values.update(
        {
            "model.language_model.layers.0.linear_attn.A_log": torch.tensor(1.0, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn.norm.weight": source[
                "model.language_model.layers.0.linear_attn.norm.weight"
            ].to(torch.bfloat16),
            "model.language_model.layers.0.mlp.down_proj.bias": source[
                "model.language_model.layers.0.mlp.down_proj.bias"
            ].to(torch.bfloat16),
        }
    )
    save_file(values, quantized / "model.safetensors")
    (original / "config.json").write_text("{}")
    (quantized / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "modelopt"}})
    )
    report = restore_protected_tensors(
        quantized, original, "P1", preserve_raw=True, max_shard_bytes=128
    )
    assert report["passed"]
    assert Path(report["raw_checkpoint_preserved_at"]).is_dir()
    repaired = load_file(next(quantized.glob("*.safetensors")))
    for name in (
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.mlp.down_proj.bias",
    ):
        # Some tiny tests are split across shards; use the index to find the tensor.
        index = json.loads((quantized / "model.safetensors.index.json").read_text())["weight_map"]
        tensor = load_file(quantized / index[name])[name]
        assert tensor.dtype == source[name].dtype
        assert torch.equal(tensor, source[name])
    assert audit_export(quantized, original, "P1")["passed"]


def test_unowned_extra_tensor_is_rejected(checkpoint_pair):
    import torch
    from safetensors.torch import save_file

    original, quantized, values = checkpoint_pair
    values["mystery.scale"] = torch.tensor(1.0)
    save_file(values, quantized / "model.safetensors")
    report = compare_export(quantized, original, "P1")
    assert report["unexpected_tensors"] == ["mystery.scale"]
    with pytest.raises(RuntimeError, match="unexpected exported tensors"):
        audit_export(quantized, original, "P1")


def test_float8_quantization_state_is_checked_via_safe_promotion(checkpoint_pair):
    import torch
    from safetensors.torch import save_file

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build has no Float8")
    original, quantized, values = checkpoint_pair
    values["model.language_model.layers.0.mlp.down_proj.weight_scale"] = torch.ones(
        2, 1, dtype=torch.float8_e4m3fn
    )
    save_file(values, quantized / "model.safetensors")
    assert audit_export(quantized, original, "P1")["passed"]


@pytest.mark.parametrize("corruption", ["wrong_packed_shape", "wrong_scale_shape", "missing_individual_scale", "changed_unselected_projection", "tiny_real_change"])
def test_complete_contract_rejects_storage_and_value_corruption(checkpoint_pair, corruption):
    import torch
    from safetensors.torch import save_file
    original, quantized, values = checkpoint_pair
    if corruption == "wrong_packed_shape":
        values["model.language_model.layers.0.mlp.down_proj.weight"] = torch.ones(2, 7, dtype=torch.uint8)
    elif corruption == "wrong_scale_shape":
        values["model.language_model.layers.0.mlp.down_proj.weight_scale"] = torch.ones(3, 1, dtype=torch.float8_e4m3fn)
    elif corruption == "missing_individual_scale":
        values.pop("model.language_model.layers.0.mlp.down_proj.input_scale")
    elif corruption == "changed_unselected_projection":
        values["model.language_model.layers.3.self_attn.o_proj.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    else:
        values["model.language_model.layers.0.linear_attn.A_log"] = torch.nextafter(torch.tensor(1.0),torch.tensor(2.0))
    save_file(values, quantized / "model.safetensors")
    assert not compare_export(quantized, original, "P1")["passed"]
    with pytest.raises(RuntimeError):
        audit_export(quantized, original, "P1")


def test_checkpoint_index_cannot_hide_unindexed_tensors(checkpoint_pair):
    import json
    from qspec_hybrid.quant import tensor_index
    original, quantized, values = checkpoint_pair
    declared = {name: "model.safetensors" for name in values if name != "visual.merger.weight"}
    (quantized / "model.safetensors.index.json").write_text(json.dumps({"weight_map": declared}))
    with pytest.raises(RuntimeError, match="index/shard mismatch"):
        tensor_index(quantized)


def test_checkpoint_duplicate_tensor_shards_are_rejected(checkpoint_pair):
    from safetensors.torch import save_file
    from qspec_hybrid.quant import tensor_index
    original, quantized, values = checkpoint_pair
    save_file({"mtp.fc.weight": values["mtp.fc.weight"]}, quantized / "duplicate.safetensors")
    with pytest.raises(RuntimeError, match="Duplicate tensor"):
        tensor_index(quantized)


def test_matching_empty_mtp_inventory_is_still_invalid(checkpoint_pair):
    from safetensors.torch import save_file,load_file
    original, quantized, values = checkpoint_pair
    source=load_file(original / "model.safetensors")
    source.pop("mtp.fc.weight")
    values.pop("mtp.fc.weight")
    save_file(source, original / "model.safetensors")
    save_file(values, quantized / "model.safetensors")
    report=compare_export(quantized, original, "P1")
    assert not report["passed"]
    assert "native-MTP source tensor inventory is empty" in report["violations"]
