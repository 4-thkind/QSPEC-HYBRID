from pathlib import Path
import subprocess
import sys

import pytest

from qspec_hybrid import core, toolchain


def test_pinned_jit_environment_uses_worker_cublas_and_rejects_changed_header(tmp_path, monkeypatch):
    root = tmp_path
    cuda, nvcc, header, runtime = toolchain._toolkit(root)
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
    nvcc.chmod(0o755)
    header.parent.mkdir(parents=True)
    header.write_text("#define CUDART_VERSION 13000\n")
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    package = root / "fake-site/nvidia/cu13"
    (package / "include").mkdir(parents=True)
    (package / "include/cublasLt.h").write_text("/* worker header */")
    (package / "lib").mkdir()
    for name in ("libcublas.so.13", "libcublasLt.so.13"):
        (package / "lib" / name).write_bytes(b"worker-library")
    monkeypatch.setattr(toolchain, "_worker_site", lambda worker: root / "fake-site")
    core.atomic_json(root / "results/cuda130_toolchain.json", {
        "schema": 2,
        "sampler_preflight_passed": True,
        "file_sha256": {
            str(path.relative_to(root)): core.file_hash(path)
            for path in (nvcc, header, runtime)
        },
    })
    env = toolchain.validated_environment(root, {"PATH": "/usr/bin"})
    assert env["FLASHINFER_NVCC"] == str(nvcc)
    assert env["PATH"].startswith(str(root / ".venv-hybrid/bin"))
    assert env["LIBRARY_PATH"].startswith(str(root / ".cache/cublas-devel/lib"))
    assert (root / ".cache/cublas-devel/lib/libcublas.so").resolve() == (package / "lib/libcublas.so.13")
    header.write_text("#define CUDART_VERSION 13010\n")
    with pytest.raises(RuntimeError, match="CUDA 13.0 headers"):
        toolchain.validated_environment(root, {})


def test_pending_or_incomplete_jit_receipt_is_not_validated(tmp_path):
    path = tmp_path / "results/cuda130_toolchain.json"
    core.atomic_json(path, {"schema": 2, "sampler_preflight_passed": False, "file_sha256": {"compiler": "hash"}})
    with pytest.raises(RuntimeError, match="preflight has not passed"):
        toolchain.validated_environment(tmp_path, {})
    core.atomic_json(path, {"sampler_preflight_passed": True})
    with pytest.raises(RuntimeError, match="receipt is incomplete"):
        toolchain.validated_environment(tmp_path, {})


def test_locked_worker_never_installs_missing_ninja(tmp_path, monkeypatch):
    worker = tmp_path / ".venv-hybrid/bin/python"
    worker.parent.mkdir(parents=True)
    worker.write_text("")
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="1.13.2\n", stderr=""),
    )
    toolchain.ensure_ninja(worker, None, allow_install=False)
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="1.12.0\n", stderr=""),
    )
    with pytest.raises(RuntimeError, match="Pinned Ninja"):
        toolchain.ensure_ninja(worker, None, allow_install=False)


def test_ninja_wheel_accepts_kitware_binary_suffix(tmp_path, monkeypatch):
    worker = tmp_path / ".venv-hybrid/bin/python"
    worker.parent.mkdir(parents=True)
    worker.symlink_to(sys.executable)
    dist = tmp_path / "site/ninja-1.13.2.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ninja\nVersion: 1.13.2\n")
    monkeypatch.setenv("PYTHONPATH", str(dist.parent))
    binary = worker.parent / "ninja"
    binary.write_text("#!/bin/sh\necho 1.13.2.g7659b.kitware.jobserver-pipe-1\n")
    binary.chmod(0o755)
    toolchain.ensure_ninja(worker, None, allow_install=False)
    binary.write_text("#!/bin/sh\necho 1.13.1\n")
    with pytest.raises(RuntimeError, match="ninja binary .* does not match wheel"):
        toolchain.ensure_ninja(worker, None, allow_install=False)


def test_existing_measurements_block_new_compiler_install(tmp_path):
    with core.Store(tmp_path / "results") as store:
        store.put("saved", {"state": "ok"})
    with pytest.raises(RuntimeError, match="after measurements"):
        toolchain.ensure_toolchain(
            tmp_path, tmp_path / ".venv-hybrid/bin/python", "uv", allow_install=True
        )
