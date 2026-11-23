from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .core import POLICIES, PROTOCOL, atomic_json, digest, read_json
from .quant import compare_export, ensure_export, recipe


def local_base(root: Path) -> Path:
    """Resolve the pinned checkpoint from cache only; never download in a preflight."""
    from huggingface_hub import snapshot_download

    revision = read_json(root / "revisions.json")["model"]
    return Path(
        snapshot_download(
            revision["id"],
            revision=revision["revision"],
            local_files_only=True,
        )
    )


def validate_recipes(root: Path):
    import yaml
    from modelopt.recipe import load_recipe

    result = {}
    with tempfile.TemporaryDirectory(prefix="qspec-recipe-check-", dir=root) as folder:
        folder = Path(folder)
        for policy in tuple(POLICIES)[1:]:
            configuration = recipe(policy)
            path = folder / f"{policy}.yaml"
            path.write_text(yaml.safe_dump(configuration, sort_keys=False))
            loaded = load_recipe(path)
            result[policy] = {
                "passed": type(loaded).__name__ == "ModelOptPTQRecipe",
                "loaded_type": type(loaded).__name__,
                "recipe_type": configuration["metadata"]["recipe_type"],
                "algorithm": configuration["quantize"]["algorithm"],
                "selected_groups": list(POLICIES[policy]),
            }
    return result


def validate_checkpoint_assets(base: Path, export: Path):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    source_config = read_json(base / "config.json", {})
    export_config = read_json(export / "config.json", {})
    quantization_config = export_config.get("quantization_config")
    architecture_export = dict(export_config)
    architecture_export.pop("quantization_config", None)
    config_contract = {
        "passed": bool(quantization_config) and architecture_export == source_config,
        "source_architecture_sha256": digest(source_config),
        "export_architecture_sha256": digest(architecture_export),
        "quantization_config_present": bool(quantization_config),
        "quantization_method": (quantization_config or {}).get("quant_method"),
        "quantization_algorithm": (quantization_config or {}).get("quant_algo"),
    }

    checks = {}
    loaded = {}
    for label, loader in (
        ("config", AutoConfig),
        ("tokenizer", AutoTokenizer),
        ("processor", AutoProcessor),
    ):
        try:
            value = loader.from_pretrained(
                export, local_files_only=True, trust_remote_code=False
            )
            loaded[label] = value
            checks[label] = {"passed": True, "class": type(value).__name__}
        except Exception as error:
            checks[label] = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }

    try:
        source_tokenizer = AutoTokenizer.from_pretrained(
            base, local_files_only=True, trust_remote_code=False
        )
        export_tokenizer = loaded.get("tokenizer")
        probes = ["QSpec", "चित्र में क्या है?", "<|im_start|>user"]
        tokenizer_contract = {
            "passed": export_tokenizer is not None
            and source_tokenizer.get_vocab() == export_tokenizer.get_vocab()
            and source_tokenizer.special_tokens_map == export_tokenizer.special_tokens_map
            and all(
                source_tokenizer.encode(text, add_special_tokens=False)
                == export_tokenizer.encode(text, add_special_tokens=False)
                for text in probes
            ),
            "source_class": type(source_tokenizer).__name__,
            "export_class": type(export_tokenizer).__name__ if export_tokenizer else None,
            "vocabulary_sha256": digest(source_tokenizer.get_vocab()),
            "probe_count": len(probes),
        }
    except Exception as error:
        tokenizer_contract = {"passed": False, "error": f"{type(error).__name__}: {error}"}

    try:
        source_processor = AutoProcessor.from_pretrained(
            base, local_files_only=True, trust_remote_code=False
        )
        export_processor = loaded.get("processor")
        source_image = getattr(source_processor, "image_processor", None)
        export_image = getattr(export_processor, "image_processor", None)
        source_image_dict = source_image.to_dict() if source_image else None
        export_image_dict = export_image.to_dict() if export_image else None
        processor_contract = {
            "passed": export_processor is not None
            and type(source_processor).__name__ == type(export_processor).__name__
            and source_image_dict == export_image_dict,
            "source_class": type(source_processor).__name__,
            "export_class": type(export_processor).__name__ if export_processor else None,
            "image_processor_sha256": digest(source_image_dict),
        }
    except Exception as error:
        processor_contract = {"passed": False, "error": f"{type(error).__name__}: {error}"}

    return {
        "config_contract": config_contract,
        "reload_checks": checks,
        "tokenizer_contract": tokenizer_contract,
        "processor_contract": processor_contract,
        "passed": config_contract["passed"]
        and all(item["passed"] for item in checks.values())
        and tokenizer_contract["passed"]
        and processor_contract["passed"],
    }


def preflight_policy(root: Path, base: Path, export: Path, policy: str, recipes=None):
    tensors = compare_export(export, base, policy)
    assets = validate_checkpoint_assets(base, export)
    recipes = recipes if recipes is not None else validate_recipes(root)
    recipe_passed = all(item["passed"] for item in recipes.values())
    blocking = list(tensors["violations"])
    if not recipe_passed:
        blocking.append("one or more P1-P7 ModelOpt recipes did not load")
    if not assets["passed"]:
        blocking.append("checkpoint configuration/tokenizer/processor reload contract failed")
    report = {
        "schema": 2,
        "status": "pass" if not blocking else "blocked",
        "policy": policy,
        "base": str(base),
        "export": str(export),
        "recipe_validation": recipes,
        "tensor_contract": tensors,
        "checkpoint_assets": assets,
        "native_mtp": {
            "source_count": len(tensors["mtp_source_inventory"]),
            "export_count": len(tensors["mtp_export_inventory"]),
            "inventory_equal": tensors["mtp_inventory_equal"],
            "source_inventory": tensors["mtp_source_inventory"],
            "export_inventory": tensors["mtp_export_inventory"],
        },
        "blocking_reasons": blocking,
        "hardware_kernel_validation_completed": False,
    }
    path = root / "results" / "conversion_preflight" / f"{policy}.json"
    atomic_json(path, report)
    return report


def run_export_phase(root: Path, policies):
    base = local_base(root)
    recipes = validate_recipes(root)
    reports = []
    for policy in policies:
        try:
            export = ensure_export(root, base, policy)
            report = preflight_policy(root, base, export, policy, recipes)
        except Exception as error:
            report = {
                "schema": 2, "policy": policy, "status": "blocked",
                "base": str(base), "recipe_validation": recipes,
                "blocking_reasons": [f"{type(error).__name__}: {error}"],
            }
            atomic_json(root / "results/conversion_preflight" / f"{policy}.json", report)
        reports.append(report)
        if report["status"] != "pass":
            break
    return reports


def smoke_exports(root: Path, policies):
    """Short vLLM loads: every export target-only; P1/P7 also native-MTP."""
    from .data import prepare
    from .runtime import Server

    prepared = prepare(root)
    case = dict(prepared["development"][0])
    case["max_tokens"] = min(2, case["max_tokens"])
    identity = digest({"protocol": PROTOCOL, "conversion_smoke": 1})
    smoke_path = root / "results/conversion_preflight/vllm_smoke.json"
    previous = read_json(smoke_path, {})
    if previous:
        import time
        atomic_json(smoke_path.with_name(f"vllm_smoke.history.{time.time_ns()}.json"), previous)
    prior = {(item["policy"], item["k"]): item for item in previous.get("results", []) if item.get("passed")}
    expected_count = sum(2 if policy in ("P1", "P7") else 1 for policy in policies)
    results = []
    targets = {}
    for policy in policies:
        model = root / "models" / policy
        modes = (0, 1) if policy in ("P1", "P7") else (0,)
        for k in modes:
            checkpoint_hash = read_json(model / "export_manifest.json", {}).get("weights_content_sha256")
            saved = prior.get((policy, k))
            if saved and checkpoint_hash and saved.get("checkpoint_sha256") == checkpoint_hash and saved.get("output_ids"):
                if k and saved["output_ids"] != targets.get(policy):
                    raise RuntimeError(f"Saved target/MTP smoke mismatch: {policy}")
                if not k:
                    targets[policy] = saved["output_ids"]
                results.append(saved)
                continue
            item = {"policy": policy, "k": k, "model": str(model), "checkpoint_sha256": checkpoint_hash}
            try:
                with Server(root, model, policy, k, identity, eager=True, smoke=True) as server:
                    output = server.generate(case)
                    token_ids = output.get("output_ids") or []
                    if not token_ids:
                        raise RuntimeError("Smoke request produced no token IDs")
                    if k and not (output.get("speculative") or {}).get("num_draft_tokens", 0):
                        raise RuntimeError("Native MTP loaded but produced no draft proposals")
                    if k and targets.get(policy) != token_ids:
                        raise RuntimeError("Target/native-MTP smoke token IDs differ")
                    if not k:
                        targets[policy] = token_ids
                    item.update(
                        passed=True,
                        output_ids=token_ids,
                        speculative=output.get("speculative"),
                        server_folder=str(server.folder),
                        eager=True,
                        autotune=False,
                        output_token_count=len(output.get("output_ids") or []),
                        native_mtp_requested=bool(k),
                        native_metrics_present=bool(output.get("native_metrics")),
                    )
            except Exception as error:
                item.update(passed=False, error=f"{type(error).__name__}: {error}")
            results.append(item)
            atomic_json(smoke_path, {"results": results, "completed": len(results) == expected_count, "passed": len(results) == expected_count and all(x["passed"] for x in results)})
            if not item["passed"]:
                return {"results": results, "passed": False}
    final = {"results": results, "completed": len(results) == expected_count, "passed": len(results) == expected_count and all(x["passed"] for x in results)}
    atomic_json(smoke_path, final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True, choices=tuple(POLICIES)[1:])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    reports = run_export_phase(args.root.resolve(), args.policies)
    if any(report["status"] != "pass" for report in reports):
        raise SystemExit(2)
    if args.smoke:
        smoke = smoke_exports(args.root.resolve(), args.policies)
        if not smoke["passed"]:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
