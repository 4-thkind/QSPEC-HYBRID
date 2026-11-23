"""Read-only model/precision audit and untimed memory queries inside a GPU worker."""

from __future__ import annotations

import os

from .quant import classify


def quantization_active(method_name, parameter_dtypes, *, kv_cache_adapter=False, cache_dtype=None):
    """Separate a cache-scale loader from actual cache/weight quantization."""
    if kv_cache_adapter:
        # The enclosing runtime audit independently checks actual cache storage.
        # Unknown or explicitly low-bit cache modes fail closed.
        return str(cache_dtype) not in ("auto", "bfloat16", "torch.bfloat16")
    low_bit_storage = any(
        any(marker in dtype for marker in ("float8", "float4", "uint8", "int8", "int4"))
        for dtype in parameter_dtypes
    )
    return low_bit_storage or "unquantized" not in method_name.lower()


class PrecisionAudit:
    def qspec_audit(self):
        import torch
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

        modules = []
        scopes = [("target", self.model_runner.model)]
        draft = getattr(self.model_runner, "speculator", None)
        if draft is not None and getattr(draft, "model", None) is not None:
            scopes.append(("mtp", draft.model))
        for scope, model in scopes:
            for name, module in model.named_modules():
                method = getattr(module, "quant_method", None)
                if method is None:
                    continue
                kv_adapter = isinstance(method, BaseKVCacheMethod)
                cache_dtype = str(getattr(module, "kv_cache_dtype", "unknown")) if kv_adapter else None
                quantized = quantization_active(
                    type(method).__name__,
                    [str(value.dtype) for _, value in module.named_parameters(recurse=False)],
                    kv_cache_adapter=kv_adapter,
                    cache_dtype=cache_dtype,
                )
                group = classify(name) if scope == "target" else "protected"
                modules.append(
                    {
                        "scope": scope,
                        "name": name,
                        "group": group,
                        "method": type(method).__name__,
                        "quantized": quantized,
                        "kv_cache_adapter": kv_adapter,
                        "module_cache_dtype": cache_dtype,
                        "kernel": type(getattr(method, "kernel", None)).__name__,
                        "quant_spec": str(getattr(method, "spec", "not applicable")),
                        "parameters": {
                            k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                            for k, v in module.named_parameters(recurse=False)
                        },
                    }
                )
        return {
            "pid": os.getpid(),
            "device": torch.cuda.get_device_name(),
            "kv_cache_dtype": str(self.model_runner.kv_cache_dtype),
            "capability": list(torch.cuda.get_device_capability()),
            "modules": modules,
            "mtp_found": any(scope == "mtp" for scope, _ in scopes),
        }

    def qspec_reset_memory(self):
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }

    def qspec_memory(self):
        import torch

        torch.cuda.synchronize()
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }

    def qspec_profile_begin(self):
        import torch

        self._qspec_events, self._qspec_hooks = [], []
        for scope, model in [
            ("target_forward", self.model_runner.model),
            (
                "draft_forward",
                getattr(getattr(self.model_runner, "speculator", None), "model", None),
            ),
        ]:
            if model is None:
                continue
            stack = []

            def before(_module, _args, label=scope, events=stack):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                events.append(start)

            def after(_module, _args, _out, label=scope, events=stack):
                finish = torch.cuda.Event(enable_timing=True)
                finish.record()
                self._qspec_events.append((label, events.pop(), finish))

            self._qspec_hooks += [
                model.register_forward_pre_hook(before),
                model.register_forward_hook(after),
            ]
        return {
            "enabled": True,
            "scope": "eager diagnostic only; shared logits and scheduler/state operations are not included",
        }

    def qspec_profile_end(self):
        import torch

        for hook in getattr(self, "_qspec_hooks", []):
            hook.remove()
        torch.cuda.synchronize()
        times = {"target_forward_gpu_ms": 0.0, "draft_forward_gpu_ms": 0.0}
        counts = {"target_forward_calls": 0, "draft_forward_calls": 0}
        for label, start, finish in getattr(self, "_qspec_events", []):
            times[label + "_gpu_ms"] += start.elapsed_time(finish)
            counts[label + "_calls"] += 1
        if counts["target_forward_calls"] == 0:
            times["target_forward_gpu_ms"] = None
        return {
            **times,
            **counts,
            "state_only_gpu_ms": None,
            "caveat": "Eager forward regions only, excluding shared logits/sampling/state overhead; not headline compiled timings.",
        }


def validate_runtime(audits, policy, k):
    from .core import POLICIES

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
                if module["quantized"] != (group in POLICIES[policy]):
                    raise RuntimeError(f"Wrong runtime precision: {module}")
                if module["quantized"]:
                    kernel = module.get("kernel", "NoneType")
                    if kernel == "NoneType" or any(
                        x in kernel.lower() for x in ("emulation", "marlin")
                    ):
                        raise RuntimeError(
                            f"Native low-bit kernel not established: {module}"
                        )
            elif module["quantized"]:
                raise RuntimeError(f"Protected module was quantized: {module}")
        if not all(counts.values()):
            raise RuntimeError(
                f"Cannot establish all runtime projection groups: {counts}"
            )
