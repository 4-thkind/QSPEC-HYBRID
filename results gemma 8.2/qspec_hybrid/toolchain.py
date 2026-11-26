"""Pinned, project-local CUDA 13.0 / FlashInfer JIT setup for Molab.

The worker's PyTorch and cuBLAS binaries are never replaced by this module.
Only an empty/pre-measurement project may create the compiler overlay.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

from .core import atomic_json, file_hash, read_json

CUDA_PINS = (
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-nvvm==13.0.88",
    "nvidia-cuda-runtime==13.0.96",
    "nvidia-curand==10.4.0.35",
)
NINJA_PIN = "1.13.2"


def _worker_site(worker: Path) -> Path:
    value = subprocess.check_output(
        [str(worker), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
        timeout=30,
    ).strip()
    site = Path(value).resolve()
    if not site.is_dir() or not site.is_relative_to(worker.parent.parent.resolve()):
        raise RuntimeError(f"Worker site-packages is outside its environment: {site}")
    return site


def ensure_ninja(worker: Path, uv: str | None, *, allow_install: bool) -> None:
    """Pin Ninja before the dependency lock is written; never alter a locked worker."""
    probe = (
        "import importlib.metadata,pathlib,subprocess,sys; "
        "installed=importlib.metadata.version('ninja'); "
        "binary=pathlib.Path(sys.argv[1]); "
        "assert binary.is_file(), 'ninja executable is missing from worker bin'; "
        "reported=subprocess.check_output([str(binary),'--version'],text=True).strip(); "
        "assert reported==installed or reported.startswith(installed+'.'), "
        "f'ninja binary {reported!r} does not match wheel {installed!r}'; "
        "print(installed)"
    )
    env = dict(os.environ)
    env["PATH"] = str(worker.parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [str(worker), "-c", probe, str(worker.parent / "ninja")],
        env=env, text=True, capture_output=True, timeout=30
    )
    if result.returncode == 0 and result.stdout.strip() == NINJA_PIN:
        return
    if not allow_install or not uv:
        raise RuntimeError(
            f"Pinned Ninja {NINJA_PIN} is missing from the locked worker; "
            f"do not change dependencies after measurements. Probe: {result.stderr.strip() or result.stdout.strip()}"
        )
    subprocess.run(
        [uv, "pip", "install", "--python", str(worker), f"ninja=={NINJA_PIN}"],
        check=True,
    )
    ensure_ninja(worker, None, allow_install=False)


def _toolkit(root: Path) -> tuple[Path, Path, Path, Path]:
    cuda = root / "tooling/cuda130-pinned/nvidia/cu13"
    return (
        cuda,
        cuda / "bin/nvcc",
        cuda / "include/cuda_runtime_api.h",
        cuda / "lib/libcudart.so.13",
    )


def _driver_library() -> Path:
    ctypes.CDLL("libcuda.so.1")
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = Path(line.split()[-1])
        if "/libcuda.so" in str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("NVIDIA driver library was not found after loading libcuda.so.1")


def _link(destination: Path, target: Path) -> None:
    target = target.resolve()
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != target:
            raise RuntimeError(f"Unexpected CUDA toolkit entry preserved: {destination}")
    else:
        destination.symlink_to(target, target_is_directory=target.is_dir())
    if not destination.exists():
        raise RuntimeError(f"CUDA toolkit link is broken: {destination}")


def _validate_toolkit(root: Path, receipt: dict) -> tuple[Path, Path]:
    cuda, nvcc, header, runtime = _toolkit(root)
    if not all(path.is_file() for path in (nvcc, header, runtime)):
        raise RuntimeError("Pinned CUDA 13.0 compiler, headers, or runtime library are missing")
    version = subprocess.check_output([str(nvcc), "--version"], text=True, timeout=30)
    if not re.search(r"release\s+13\.0\b", version):
        raise RuntimeError("FlashInfer must use the pinned CUDA 13.0 compiler")
    if not re.search(r"#\s*define\s+CUDART_VERSION\s+13000\b", header.read_text()):
        raise RuntimeError("FlashInfer must use CUDA 13.0 headers")
    allowed = {str(path.relative_to(root)) for path in (nvcc, header, runtime)}
    for relative, expected in receipt.get("file_sha256", {}).items():
        if relative not in allowed or file_hash(root / relative) != expected:
            raise RuntimeError(f"Pinned CUDA toolchain changed: {relative}")
    return cuda, nvcc


def validated_environment(
    root: Path, env: dict[str, str], *, _allow_pending: bool = False
) -> dict[str, str]:
    """Reconstruct JIT paths from a successful receipt; perform no installation."""
    root = Path(root).resolve()
    receipt = read_json(root / "results/cuda130_toolchain.json", {})
    if receipt.get("schema") != 2 or not receipt.get("file_sha256"):
        raise RuntimeError("CUDA/FlashInfer toolchain receipt is incomplete")
    if not receipt.get("sampler_preflight_passed") and not _allow_pending:
        raise RuntimeError("CUDA/FlashInfer sampler preflight has not passed")
    cuda, nvcc = _validate_toolkit(root, receipt)
    worker = root / ".venv-hybrid/bin/python"
    package = _worker_site(worker) / "nvidia/cu13"
    if not (package / "include/cublasLt.h").is_file():
        raise RuntimeError("Worker cuBLAS development headers are missing; JIT is not ready")
    from .runtime import configure_cublas_paths

    result = dict(env)
    cache = root / ".cache/flashinfer-cu130"
    parent = read_json(root / "lineage.json", {}).get("parent_root")
    if parent and (Path(parent) / ".cache/flashinfer-cu130").is_dir():
        cache = Path(parent) / ".cache/flashinfer-cu130"
    result.update(
        CUDA_HOME=str(cuda),
        CUDA_PATH=str(cuda),
        FLASHINFER_NVCC=str(nvcc),
        FLASHINFER_WORKSPACE_BASE=str(cache),
    )
    parts = [str(worker.parent), str(cuda / "bin")] + result.get("PATH", "").split(os.pathsep)
    result["PATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))
    return configure_cublas_paths(root, result, package=package)


def _has_measurements(root: Path) -> bool:
    database = root / "results/requests.sqlite"
    if not database.is_file():
        return False
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        return connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0] > 0


def ensure_toolchain(root: Path, worker: Path, uv: str, *, allow_install: bool) -> dict[str, str]:
    """Run the actual sampler once before the first measurement."""
    root = Path(root).resolve()
    receipt_path = root / "results/cuda130_toolchain.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path, {})
        if receipt.get("sampler_preflight_passed") and receipt.get("schema") == 2:
            return validated_environment(root, dict(os.environ))
        if not allow_install or _has_measurements(root):
            raise RuntimeError("Incomplete JIT preflight cannot be resumed after measurements or in a frozen lineage")
        # An interrupted preflight may leave a pending receipt. Rebuild it and
        # rerun the real GPU sample before publishing a passing receipt.
        receipt_path.unlink()
    if not allow_install or _has_measurements(root):
        raise RuntimeError("Cannot install a new JIT toolchain after measurements or in a frozen lineage")
    gpu_probe = (
        "import json,torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'torch_cuda':torch.version.cuda,'capability':[p.major,p.minor]}))"
    )
    gpu = json.loads(subprocess.check_output([str(worker), "-c", gpu_probe], text=True, timeout=90))
    if gpu != {"torch_cuda": "13.0", "capability": [12, 0]}:
        raise RuntimeError(f"This audited JIT path requires PyTorch CUDA 13.0 on SM120: {gpu}")
    site = root / "tooling/cuda130-pinned"
    site.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [uv, "pip", "install", "--python", str(worker), "--target", str(site), "--reinstall", *CUDA_PINS],
        check=True,
    )
    cuda, nvcc, header, runtime = _toolkit(root)
    _validate_toolkit(root, {})
    library = cuda / "lib"
    library.mkdir(exist_ok=True)
    _link(cuda / "lib64", library)
    _link(library / "libcudart.so", runtime)
    _link(library / "libcuda.so", _driver_library())
    # The receipt is published only after compilation and an actual GPU sample.
    files = (nvcc, header, runtime)
    hashes = {str(path.relative_to(root)): file_hash(path) for path in files}
    provisional = {"schema": 2, "sampler_preflight_passed": False, "file_sha256": hashes}
    atomic_json(receipt_path, provisional)
    try:
        env = validated_environment(root, dict(os.environ), _allow_pending=True)
        sampler = """import torch,flashinfer
assert torch.version.cuda == '13.0', torch.version.cuda
torch.manual_seed(0)
logits=torch.randn(1,1024,device='cuda',dtype=torch.float32)
tokens=flashinfer.sampling.top_k_top_p_sampling_from_logits(logits,top_k=10,top_p=0.9)
torch.cuda.synchronize()
print('CUDA 13.0 + FlashInfer sampler PASSED:',tokens.cpu().tolist())
"""
        result = subprocess.run(
            [str(worker), "-c", sampler], cwd=root, env=env,
            text=True, capture_output=True, timeout=900,
        )
        log = root / "results/cuda130_preflight.log"
        log.write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(f"FlashInfer JIT preflight failed; inspect {log}")
        atomic_json(receipt_path, {
            "schema": 2,
            "pins": list(CUDA_PINS),
            "ninja": NINJA_PIN,
            "torch_cuda": gpu["torch_cuda"],
            "capability": gpu["capability"],
            "file_sha256": hashes,
            "sampler_preflight_passed": True,
            "worker_packages_changed": False,
        })
        return env
    except BaseException:
        receipt_path.unlink(missing_ok=True)
        raise
