from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .core import POLICIES, PROTOCOL, atomic_json, digest, read_json

# Revision control (extensions.F1): the FFN group stored as FP8 instead of NVFP4.
FP8_FFN_POLICIES = {"F1"}
# Family runs (extensions2) whose MTP drafter is a separate checkpoint, not tensors in the target.
EXTERNAL_DRAFTER_POLICIES = {"G0", "G1", "G4"}


def _extra_mtp_prefixes():
    # Families whose drafter is stored as an extra decoder layer (GLM-OCR: layers.<num_hidden_layers>.)
    return tuple(p.lower() for p in os.environ.get("QSPEC_MTP_PREFIXES", "").split(",") if p)


def extra_excludes():
    # Extra exclude patterns written into the export, in the drafter's vLLM module names.
    return [p for p in os.environ.get("QSPEC_EXTRA_EXCLUDES", "").split(",") if p]


def is_mtp_name(name):
    name = name.lower()
    return "mtp" in name or any(p in name for p in _extra_mtp_prefixes())


def classify(name):
    name = name.lower()
    if is_mtp_name(name) or any(x in name for x in ("visual", "vision", "audio", "lm_head", "embed_tokens")):
        return "protected"
    if ".mlp." in name and any(
        x in name for x in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
    ):
        return "D"
    if ".self_attn." in name and any(
        x in name for x in ("q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj")
    ):
        return "A"
    if ".linear_attn." in name and any(
        x in name for x in ("in_proj_qkv", "in_proj_z", "out_proj")
    ):
        return "G"
    return "protected"


def recipe(policy):
    groups = POLICIES[policy]
    rules = [{"quantizer_name": "*", "enable": False}]
    patterns = {
        "D": [
            "*layers.*.mlp.gate_proj*",
            "*layers.*.mlp.up_proj*",
            "*layers.*.mlp.down_proj*",
        ],
        "A": [
            "*layers.*.self_attn.q_proj*",
            "*layers.*.self_attn.k_proj*",
            "*layers.*.self_attn.v_proj*",
            "*layers.*.self_attn.o_proj*",
        ],
        "G": [
            "*layers.*.linear_attn.in_proj_qkv*",
            "*layers.*.linear_attn.in_proj_z*",
            "*layers.*.linear_attn.out_proj*",
        ],
    }
    for group in groups:
        for pattern in patterns[group]:
            for quantizer in ("weight_quantizer", "input_quantizer"):
                rules.append(
                    {
                        "quantizer_name": pattern + quantizer,
                        "cfg": {"$import": "nvfp4" if group == "D" else "fp8"},
                    }
                )
    # Last match wins. Protect cache, gates, visual branch and native MTP.
    for pattern in (
        "*mtp*",
        "*lm_head*",
        "*embed_tokens*",
        "*visual*",
        "*vision*",
        "*linear_attn.in_proj_a*",
        "*linear_attn.in_proj_b*",
        "*conv1d*",
        "*norm*",
    ):
        rules.append({"quantizer_name": pattern, "enable": False})
    return {
        "imports": {"fp8": "configs/numerics/fp8", "nvfp4": "configs/numerics/nvfp4"},
        "metadata": {
            "recipe_type": "ptq",
            "description": f"QSpec {policy}; controlled selective PTQ; cache/vision/head/MTP unquantized"
        },
        "quantize": {
            "algorithm": {"method": "max"},
            "quant_cfg": rules,
        },
    }


def tensor_index(path):
    """Index tensor names without loading model-sized tensors into memory."""
    from safetensors import safe_open

    index_path = path / "model.safetensors.index.json"
    mapping = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in mapping:
                    raise RuntimeError(f"Duplicate tensor across checkpoint shards: {key}")
                mapping[key] = file.name
    if not mapping:
        raise RuntimeError(f"No safetensors weights found in {path}")
    if index_path.exists():
        declared = read_json(index_path)["weight_map"]
        if declared != mapping:
            missing = sorted(set(declared) - set(mapping))
            unindexed = sorted(set(mapping) - set(declared))
            misplaced = sorted(key for key in set(mapping) & set(declared) if mapping[key] != declared[key])
            raise RuntimeError(f"Checkpoint index/shard mismatch: missing={missing[:5]}, unindexed={unindexed[:5]}, misplaced={misplaced[:5]}")
    return mapping


def is_selected_weight(name, policy):
    """Only named component weights selected by a policy may be quantized."""
    return name.endswith(".weight") and classify(name) in POLICIES[policy]


_QUANT_AUX_SUFFIXES = (".input_scale", ".weight_scale", ".weight_scale_2")


def quant_aux_owner(name):
    for suffix in _QUANT_AUX_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".weight", suffix[1:]
    return None, None


def _tensor_sha256(tensor):
    import hashlib
    import torch

    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _lossless_float_widening(source, exported):
    import torch

    widening = {
        torch.float16: {torch.float32, torch.float64},
        torch.bfloat16: {torch.float32, torch.float64},
        torch.float32: {torch.float64},
    }
    return (
        exported.dtype in widening.get(source.dtype, set())
        and source.shape == exported.shape
        and torch.equal(exported, source.to(exported.dtype))
    )


def storage_layout_errors(selected, auxiliary):
    """Validate the pinned ModelOpt NVFP4 packing and FP8 scale layouts."""
    states = {item["name"]: item for item in auxiliary}
    errors = []
    for item in selected:
        source_shape, export_shape = item["source_shape"], item["export_shape"]
        stored = item["export_dtype"]
        nvfp4 = item["expected_format"] == "U8/F4"
        if nvfp4:
            valid = (len(source_shape) == 2 and source_shape[-1] % 16 == 0
                     and stored == "U8"
                     and export_shape == [source_shape[0], source_shape[1] // 2])
        else:
            valid = stored == "F8_E4M3" and export_shape == source_shape
        if not valid:
            errors.append({"weight": item["name"], "reason": "quantized weight shape/format does not match recipe packing"})
        prefix = item["name"][:-len(".weight")]
        for suffix in ("input_scale", "weight_scale", "weight_scale_2"):
            state = states.get(prefix + "." + suffix)
            if state is None:
                continue
            expected = [source_shape[0], source_shape[1] // 16] if nvfp4 and suffix == "weight_scale" and len(source_shape) == 2 else []
            dtype = "F8_E4M3" if nvfp4 and suffix == "weight_scale" else "F32"
            if state["shape"] != expected or state["dtype"] != dtype:
                errors.append({"weight": item["name"], "state": state["name"], "reason": "scale layout/dtype does not match pinned recipe", "expected_shape": expected, "expected_dtype": dtype})
    return errors


def compare_export(path, original, policy, allowed_lossless_widenings=()):
    """Classify every source and exported tensor without numerical tolerances.

    A dtype widening is accepted only when it is mathematically lossless *and* the
    tensor name is explicitly supplied in ``allowed_lossless_widenings``.
    """
    from contextlib import ExitStack
    from safetensors import safe_open
    import torch

    allowed = set(allowed_lossless_widenings)
    exported_map, source_map = tensor_index(path), tensor_index(original)
    exact, widened, real_changes, dtype_only = [], [], [], []
    protected_missing, selected_missing, selected, auxiliary, unexpected = [], [], [], [], []
    wrong_selected_formats, missing_auxiliary, nonfinite = [], [], []
    hashes = {}

    with ExitStack() as stack:
        source_streams = {
            filename: stack.enter_context(safe_open(original / filename, framework="pt", device="cpu"))
            for filename in sorted(set(source_map.values()))
        }
        export_streams = {
            filename: stack.enter_context(safe_open(path / filename, framework="pt", device="cpu"))
            for filename in sorted(set(exported_map.values()))
        }

        for name in sorted(source_map):
            chosen = is_selected_weight(name, policy)
            if name not in exported_map:
                (selected_missing if chosen else protected_missing).append(name)
                continue
            source_tensor = source_streams[source_map[name]].get_tensor(name).contiguous()
            export_stream = export_streams[exported_map[name]]
            export_tensor = export_stream.get_tensor(name).contiguous()
            hashes[name] = {
                "sha256": _tensor_sha256(export_tensor),
                "dtype": str(export_tensor.dtype),
                "shape": list(export_tensor.shape),
            }
            if export_tensor.is_floating_point() and not torch.isfinite(export_tensor.to(torch.float64)).all().item():
                nonfinite.append(name)
            if chosen:
                group = classify(name)
                stored = export_stream.get_slice(name).get_dtype()
                nvfp4 = group == "D" and policy not in FP8_FFN_POLICIES
                expected = "U8/F4" if nvfp4 else "F8"
                valid = (stored == "U8" or stored.startswith("F4")) if nvfp4 else stored.startswith("F8")
                item = {
                    "name": name,
                    "group": group,
                    "source_dtype": str(source_tensor.dtype),
                    "export_dtype": stored,
                    "source_shape": list(source_tensor.shape),
                    "export_shape": list(export_tensor.shape),
                    "expected_format": expected,
                }
                selected.append(item)
                if not valid:
                    wrong_selected_formats.append(item)
                continue

            source_hash, export_hash = _tensor_sha256(source_tensor), hashes[name]["sha256"]
            item = {
                "name": name,
                "group": classify(name),
                "source_dtype": str(source_tensor.dtype),
                "export_dtype": str(export_tensor.dtype),
                "source_shape": list(source_tensor.shape),
                "export_shape": list(export_tensor.shape),
            }
            if source_tensor.shape == export_tensor.shape and source_tensor.dtype == export_tensor.dtype and source_hash == export_hash:
                exact.append(item)
            elif _lossless_float_widening(source_tensor, export_tensor):
                item["explicitly_allowed"] = name in allowed
                widened.append(item)
            elif source_tensor.shape == export_tensor.shape and source_tensor.dtype != export_tensor.dtype and torch.equal(source_tensor.to(torch.float64), export_tensor.to(torch.float64)):
                item["reason"] = "numeric values equal but dtype conversion is not an allowed lossless widening"
                dtype_only.append(item)
            else:
                item["reason"] = "shape or numerical values changed"
                real_changes.append(item)

        for name in sorted(set(exported_map) - set(source_map)):
            owner, state = quant_aux_owner(name)
            if owner in source_map and is_selected_weight(owner, policy):
                tensor = export_streams[exported_map[name]].get_tensor(name).contiguous()
                auxiliary.append({
                    "name": name,
                    "owner": owner,
                    "state": state,
                    "group": classify(owner),
                    "dtype": export_streams[exported_map[name]].get_slice(name).get_dtype(),
                    "shape": list(tensor.shape),
                })
                hashes[name] = {"sha256": _tensor_sha256(tensor), "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
                if tensor.is_floating_point() and not torch.isfinite(tensor.to(torch.float64)).all().item():
                    nonfinite.append(name)
            else:
                unexpected.append(name)

    auxiliary_names = {item["name"] for item in auxiliary}
    for item in selected:
        base = item["name"][:-len(".weight")]
        required = [base + ".input_scale", base + ".weight_scale"]
        if item["expected_format"] == "U8/F4":
            required.append(base + ".weight_scale_2")
        absent = [name for name in required if name not in auxiliary_names]
        if absent:
            missing_auxiliary.append({"weight": item["name"], "missing": absent})

    mtp_source = sorted(name for name in source_map if is_mtp_name(name))
    mtp_export = sorted(name for name in exported_map if is_mtp_name(name) and name in source_map)
    layout_errors = storage_layout_errors(selected, auxiliary)
    violations = []
    missing_groups = sorted(set(POLICIES[policy]) - {classify(name) for name in source_map if name.endswith(".weight")})
    if missing_groups: violations.append(f"source lacks selected component groups: {missing_groups}")
    # Gemma 4 ships its MTP drafter as a separate assistant checkpoint, so its target has none.
    if not mtp_source and policy not in EXTERNAL_DRAFTER_POLICIES:
        violations.append("native-MTP source tensor inventory is empty")
    if layout_errors: violations.append("selected weights or quantization state have invalid storage layouts")
    if protected_missing: violations.append("protected tensors missing")
    if selected_missing: violations.append("selected weights missing")
    if real_changes: violations.append("protected tensors changed shape or numerical values")
    if dtype_only: violations.append("protected tensors used a disallowed dtype conversion")
    if any(not item["explicitly_allowed"] for item in widened): violations.append("lossless widenings were not explicitly recorded")
    if wrong_selected_formats: violations.append("selected weights have incorrect stored formats")
    if missing_auxiliary: violations.append("selected weights lack required scale/state tensors")
    if unexpected: violations.append("unexpected exported tensors")
    if nonfinite: violations.append("non-finite exported tensors")
    if mtp_source != mtp_export: violations.append("native-MTP tensor inventory changed")

    return {
        "policy": policy,
        "source_tensor_count": len(source_map),
        "export_tensor_count": len(exported_map),
        "comparison_counts": {
            "exact_preservation": len(exact),
            "lossless_dtype_only": len(widened),
            "disallowed_dtype_only": len(dtype_only),
            "real_numerical_or_shape_change": len(real_changes),
            "protected_missing": len(protected_missing),
            "selected_missing": len(selected_missing),
            "selected_quantized_weights": len(selected),
            "quantization_scale_state_tensors": len(auxiliary),
            "unexpected_tensors": len(unexpected),
        },
        "exact_preservation": exact,
        "lossless_dtype_only": widened,
        "disallowed_dtype_only": dtype_only,
        "real_numerical_or_shape_change": real_changes,
        "protected_missing": protected_missing,
        "selected_missing": selected_missing,
        "selected_quantized_weights": selected,
        "quantization_scale_state_tensors": auxiliary,
        "wrong_selected_formats": wrong_selected_formats,
        "storage_layout_errors": layout_errors,
        "missing_scale_state": missing_auxiliary,
        "unexpected_tensors": unexpected,
        "nonfinite_tensors": sorted(set(nonfinite)),
        "explicit_lossless_widening_allowlist": sorted(allowed),
        "mtp_source_inventory": mtp_source,
        "mtp_export_inventory": mtp_export,
        "mtp_inventory_equal": mtp_source == mtp_export,
        "weights_content_sha256": digest(hashes),
        "weights_bytes": sum((path / filename).stat().st_size for filename in set(exported_map.values())),
        "violations": violations,
        "passed": not violations,
    }


def audit_export(path, original, policy, allowed_lossless_widenings=()):
    report = compare_export(path, original, policy, allowed_lossless_widenings)
    if not report["passed"]:
        details = []
        for key in (
            "protected_missing", "selected_missing", "real_numerical_or_shape_change",
            "disallowed_dtype_only", "wrong_selected_formats", "missing_scale_state", "unexpected_tensors",
        ):
            if report[key]:
                details.append(f"{key}={report[key][:3]}")
        raise RuntimeError("Conversion contract failed: " + "; ".join(report["violations"] + details))
    counts = {group: 0 for group in ("D", "A", "G")}
    storage = {group: set() for group in counts}
    for item in report["selected_quantized_weights"]:
        counts[item["group"]] += 1
        storage[item["group"]].add(item["export_dtype"])
    report.update({
        "projection_counts": counts,
        "stored_dtypes": {key: sorted(value) for key, value in storage.items()},
        "protected_weight_hashes": {item["name"]: item for item in report["exact_preservation"]},
        "mtp_tensor_count": len(report["mtp_source_inventory"]),
    })
    return report


def _write_repaired_checkpoint(stage, exported, original, policy, max_shard_bytes):
    from contextlib import ExitStack
    from safetensors import safe_open
    from safetensors.torch import save_file

    exported_map, source_map = tensor_index(exported), tensor_index(original)
    for item in exported.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
            shutil.copy2(item, stage / item.name)
        elif item.is_dir():
            shutil.copytree(item, stage / item.name)

    parts, current, current_bytes = [], {}, 0
    with ExitStack() as stack:
        source_streams = {
            filename: stack.enter_context(safe_open(original / filename, framework="pt", device="cpu"))
            for filename in sorted(set(source_map.values()))
        }
        export_streams = {
            filename: stack.enter_context(safe_open(exported / filename, framework="pt", device="cpu"))
            for filename in sorted(set(exported_map.values()))
        }
        for name in sorted(exported_map):
            use_source = name in source_map and not is_selected_weight(name, policy)
            stream = source_streams[source_map[name]] if use_source else export_streams[exported_map[name]]
            tensor = stream.get_tensor(name).contiguous()
            size = tensor.numel() * tensor.element_size()
            if current and current_bytes + size > max_shard_bytes:
                part = stage / f".part-{len(parts) + 1:05d}.safetensors"
                save_file(current, part, metadata={"format": "pt"})
                parts.append((part, tuple(current), current_bytes))
                current, current_bytes = {}, 0
            current[name] = tensor
            current_bytes += size
        if current:
            part = stage / f".part-{len(parts) + 1:05d}.safetensors"
            save_file(current, part, metadata={"format": "pt"})
            parts.append((part, tuple(current), current_bytes))

    weight_map, total_size = {}, 0
    total = len(parts)
    for number, (part, names, size) in enumerate(parts, 1):
        final_name = f"model-{number:05d}-of-{total:05d}.safetensors"
        os.replace(part, stage / final_name)
        weight_map.update({name: final_name for name in names})
        total_size += size
    atomic_json(stage / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": weight_map})


def restore_protected_tensors(path, original, policy, *, preserve_raw=False, max_shard_bytes=2 * 1024**3):
    """Rebuild an export so every non-selected source tensor is byte-exact.

    The rebuilt directory is validated before an atomic directory swap. Existing
    checkpoints can be retained with ``preserve_raw=True``.
    """
    import json
    import time

    path, original = Path(path), Path(original)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.protected-stage-", dir=path.parent))
    raw = path.with_name(f"{path.name}.modelopt-raw-{int(time.time())}")
    try:
        _write_repaired_checkpoint(stage, path, original, policy, max_shard_bytes)
        source_config = read_json(original / "config.json", {})
        export_config = read_json(stage / "config.json", {})
        quantization_config = export_config.get("quantization_config")
        if not quantization_config:
            raise RuntimeError("ModelOpt export did not write quantization_config")
        extra = extra_excludes()
        if extra:
            # The carried drafter is listed under its checkpoint names, but some vLLM drafters
            # (GLM-OCR) name their modules differently, so also exclude it under those names.
            key = "ignore" if "ignore" in quantization_config else "exclude_modules"
            quantization_config[key] = sorted(set(quantization_config.get(key, [])) | set(extra))
            legacy = read_json(stage / "hf_quant_config.json")
            if legacy and isinstance(legacy.get("quantization"), dict):
                q = legacy["quantization"]
                q["exclude_modules"] = sorted(set(q.get("exclude_modules") or []) | set(extra))
                atomic_json(stage / "hf_quant_config.json", legacy)
        repaired_config = dict(source_config)
        repaired_config["quantization_config"] = quantization_config
        atomic_json(stage / "config.json", repaired_config)
        report = compare_export(stage, original, policy)
        if not report["passed"]:
            raise RuntimeError(f"Rebuilt checkpoint failed conversion contract: {report['violations']}")
        os.replace(path, raw)
        try:
            os.replace(stage, path)
        except BaseException:
            os.replace(raw, path)
            raise
        if not preserve_raw:
            shutil.rmtree(raw)
        report["raw_checkpoint_preserved_at"] = str(raw) if preserve_raw else None
        return report
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def adopt_unfinished_export(root, base, policy, output):
    """Validate and atomically adopt an already-completed raw ModelOpt export.

    This is intentionally explicit: it never selects or deletes an arbitrary
    directory, and preserves the raw checkpoint beside its original staging
    directory for recovery/audit purposes.
    """
    output = Path(output).resolve()
    expected_parent = (root / "models").resolve()
    if expected_parent not in output.parents or output.name != "weights":
        raise RuntimeError(f"Refusing to adopt export outside the expected staging layout: {output}")
    destination = root / "models" / policy
    if destination.exists():
        raise RuntimeError(f"Destination already exists: {destination}")
    report = restore_protected_tensors(output, base, policy, preserve_raw=True)
    expected = {
        "protocol": digest(PROTOCOL),
        "recipe": digest(recipe(policy)),
        "base_revision": read_json(root / "revisions.json")["model"]["revision"],
        "calibration": digest(read_json(root / "data/prepared.json")["calibration"]),
    }
    report["identity"] = expected
    report["file_sizes"] = {
        item.name: item.stat().st_size for item in output.iterdir() if item.is_file()
    }
    atomic_json(output / "export_manifest.json", report)
    atomic_json(root / "results" / "export_audits" / f"{policy}.json", report)
    os.replace(output, destination)
    return destination

def ensure_export(root, base, policy):
    import yaml

    if policy == "P0":
        return base
    path = root / "models" / policy
    expected = {
        "protocol": digest(PROTOCOL),
        "recipe": digest(recipe(policy)),
        "base_revision": read_json(root / "revisions.json")["model"]["revision"],
        "calibration": digest(read_json(root / "data/prepared.json")["calibration"]),
    }
    manifest = read_json(path / "export_manifest.json")
    if manifest:
        if manifest["identity"] != expected:
            raise RuntimeError(
                f"{policy} export identity changed; use a new project directory"
            )
        # Detect missing/truncated files without re-reading every tensor on resume.
        for name, size in manifest["file_sizes"].items():
            if not (path / name).exists() or (path / name).stat().st_size != size:
                raise RuntimeError(f"Incomplete cached export: {policy}/{name}")
        return path
    if path.exists():
        raise RuntimeError(
            f"Unverified directory exists at {path}; it was left untouched"
        )
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise RuntimeError(
            "Less than 20 GiB disk remains; export stopped before filling storage"
        )
    config = root / "configs" / f"{policy}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(recipe(policy), sort_keys=False))
    (root / "models").mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{policy}-export-", dir=root / "models"))
    output = temporary / "weights"
    env = dict(os.environ, QSPEC_ROOT=str(root))
    print(f"Exporting {policy}; a first export can take several minutes", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "qspec_hybrid.quant",
            str(root),
            str(base),
            str(config),
            str(output),
        ],
        cwd=root,
        env=env,
        check=True,
    )
    result = audit_export(output, base, policy)
    result["identity"] = expected
    result["file_sizes"] = {
        f.name: f.stat().st_size for f in output.iterdir() if f.is_file()
    }
    receipt = root / "results" / "export_audits" / f"{policy}.json"
    old_receipt = read_json(receipt)
    if old_receipt and (
        old_receipt["identity"] != expected
        or old_receipt["weights_content_sha256"] != result["weights_content_sha256"]
    ):
        raise RuntimeError(
            f"Rebuilt {policy} weights differ from the saved experiment; refusing to mix results. Rebuild was preserved at {output}"
        )
    atomic_json(output / "export_manifest.json", result)
    atomic_json(receipt, result)
    os.replace(output, path)
    temporary.rmdir()
    return path


def exporter_main(root, base, recipe_path, output):
    """Official exporter, with one explicit frozen image-calibration adapter.

    No forward/attention/cache/quantization algorithm implementation is replaced.
    The adapter replaces the upstream random corpus loader, not the model.
    """
    import torch
    import modelopt.torch.utils.vlm_dataset_utils as module
    from .data import image_of

    cases = read_json(root / "data/prepared.json")["calibration"]

    def loader(*, processor, device=None, **kwargs):
        class FrozenCalibration:
            batch_size = 1
            dataset = cases

            def __len__(self):
                return len(cases)

            def __iter__(self):
                for case in cases:
                    image = image_of(str(root / case["image"]))
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": case["prompt"]},
                            ],
                        },
                        {"role": "assistant", "content": case["target"]},
                    ]
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    # Qwen style processors take a pixel budget. Others (Gemma) use their own resizing.
                    pixels = {} if os.environ.get("QSPEC_NO_PIXEL_KWARGS") else {
                        "max_pixels": PROTOCOL["max_pixels"], "min_pixels": PROTOCOL["min_pixels"]}
                    batch = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                        **pixels,
                    )
                    yield {
                        k: v.to(device or "cuda") if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

        return FrozenCalibration()

    module.get_vlm_dataset_dataloader = loader
    script = root / "tooling" / "modelopt" / "examples" / "hf_ptq" / "hf_ptq.py"
    sys.path.insert(0, str(script.parent))
    sys.argv = [
        str(script),
        "--pyt_ckpt_path",
        str(base),
        "--recipe",
        str(recipe_path),
        "--export_path",
        str(output),
        "--calib_with_images",
        "--calib_size",
        str(len(cases)),
        "--batch_size",
        "1",
        "--skip_generate",
        "--attn_implementation",
        "eager",
    ]
    runpy.run_path(str(script), run_name="__main__")
    policy = recipe_path.stem
    restore_protected_tensors(output, base, policy)


if __name__ == "__main__":
    exporter_main(*(Path(x).resolve() for x in sys.argv[1:]))
