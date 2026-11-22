"""Molab bootstrap: separate Python 3.12 worker; safe to rerun after interruption."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def canonical_lock(text, repo, commit):
    """Make a pinned lock portable across fresh Molab filesystem paths."""
    from urllib.parse import unquote, urlparse

    lines = []
    for line in text.splitlines():
        if line.lower().startswith("nvidia-modelopt @ file:"):
            location = unquote(urlparse(line.split(" @ ", 1)[1]).path)
            if Path(location).resolve() != repo.resolve():
                raise RuntimeError(
                    "Installed ModelOpt did not come from this pinned checkout"
                )
            line = (
                "nvidia-modelopt @ git+https://github.com/NVIDIA/Model-Optimizer.git@"
                + commit
            )
        lines.append(line)
    return "\n".join(sorted(lines)) + "\n"


def _retry_run(cmd, attempts=4, pause=20):
    """Package downloads hit transient 5xx from PyPI/GitHub release assets.

    uv keeps what it already fetched, so a retry resumes rather than restarting.
    Non-network failures still surface on the final attempt.
    """
    import subprocess as _sp
    import time as _t

    for attempt in range(1, attempts + 1):
        try:
            return _sp.run(cmd, check=True)
        except _sp.CalledProcessError:
            if attempt == attempts:
                raise
            print(
                f"[install attempt {attempt}/{attempts} failed; retrying in {pause}s "
                f"-- usually a transient registry/CDN error]",
                flush=True,
            )
            _t.sleep(pause)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full", "extend", "fam_gemma", "fam_glmocr", "fam_exaone"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from qspec_hybrid.core import (
        MODEL_OPT,
        PROTOCOL,
        atomic_json,
        bundle,
        digest,
        exclusive,
        read_json,
    )
    from qspec_hybrid.toolchain import ensure_ninja, ensure_toolchain

    def status(stage, **kwargs):
        atomic_json(
            root / "status.json",
            {"state": "running", "stage": stage, "mode": args.mode, **kwargs},
        )

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with exclusive(root):
            try:
                status("checking GPU and setup")
                smi = shutil.which("nvidia-smi")
                if not smi:
                    raise RuntimeError(
                        "Attach RTX PRO 6000 in Molab notebook specifications; nvidia-smi is not available in this worker"
                    )
                gpu = subprocess.check_output(
                    [
                        smi,
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                print("Attached GPU:", gpu.strip(), flush=True)
                uv = shutil.which("uv")
                if not uv:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "uv"], check=True
                    )
                    uv = shutil.which("uv")
                if not uv:
                    raise RuntimeError("uv is unavailable; worker setup cannot continue")
                git = shutil.which("git")
                if not git:
                    raise RuntimeError("Molab Git executable is missing; the pinned ModelOpt commit cannot be verified")
                python = root / ".venv-hybrid" / "bin" / "python"
                if not python.exists():
                    subprocess.run(
                        [
                            uv,
                            "venv",
                            "--allow-existing",
                            "--python",
                            "3.12",
                            str(python.parent.parent),
                        ],
                        check=True,
                    )
                repo = root / "tooling" / "modelopt"
                if not (repo / ".git").exists():
                    repo.mkdir(parents=True, exist_ok=True)
                    subprocess.run([git, "init", str(repo)], check=True)
                head = subprocess.run(
                    [git, "-C", str(repo), "rev-parse", "--verify", "HEAD"],
                    capture_output=True,
                )
                if head.returncode:
                    subprocess.run(
                        [
                            git,
                            "-C",
                            str(repo),
                            "fetch",
                            "--depth",
                            "1",
                            "https://github.com/NVIDIA/Model-Optimizer.git",
                            MODEL_OPT,
                        ],
                        check=True,
                    )
                    subprocess.run(
                        [git, "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"],
                        check=True,
                    )
                commit = subprocess.check_output(
                    [git, "-C", str(repo), "rev-parse", "HEAD"], text=True
                ).strip()
                if commit != MODEL_OPT:
                    raise RuntimeError(
                        "ModelOpt checkout has the wrong revision; refusing to overwrite it"
                    )
                subprocess.run(
                    [git, "-C", str(repo), "diff", "--exit-code", "HEAD"], check=True
                )
                setup_id = digest(
                    {
                        "protocol": PROTOCOL,
                        "requirements": (root / "requirements.txt").read_text(),
                    }
                )
                marker = root / ".environment.json"
                if not marker.exists():
                    status("installing pinned serving/export tools; inspect live log")
                    lock_path = root / "environment.lock.txt"
                    if lock_path.exists():
                        _retry_run(
                            [
                                uv,
                                "pip",
                                "sync",
                                "--python",
                                str(python),
                                "--index-strategy",
                                "unsafe-best-match",
                                "--extra-index-url",
                                "https://flashinfer.ai/whl/",
                                str(lock_path),
                            ],
                        )
                    else:
                        _retry_run(
                            [
                                uv,
                                "pip",
                                "install",
                                "--python",
                                str(python),
                                "--index-strategy",
                                "unsafe-best-match",
                                "--extra-index-url",
                                "https://flashinfer.ai/whl/",
                                "-r",
                                str(root / "requirements.txt"),
                                str(repo),
                            ],
                        )
                    ensure_ninja(python, uv, allow_install=not lock_path.exists())
                    status("verifying pinned CUDA 13.0 and FlashInfer JIT; first compile may take minutes")
                    tool_env = ensure_toolchain(root, python, uv, allow_install=True)
                    subprocess.run([uv, "pip", "check", "--python", str(python)], check=True)
                    lock = subprocess.check_output(
                        [uv, "pip", "freeze", "--python", str(python)], text=True
                    )
                    lock = canonical_lock(lock, repo, commit)
                    if lock_path.exists() and lock_path.read_text() != lock:
                        raise RuntimeError(
                            "Restored environment did not reproduce the frozen dependency lock"
                        )
                    lock_path.write_text(lock)
                    atomic_json(
                        marker, {"identity": setup_id, "modelopt": commit, "gpu": gpu}
                    )
                elif read_json(marker)["identity"] != setup_id:
                    raise RuntimeError(
                        "Environment belongs to a different protocol; use a fresh project directory"
                    )
                else:
                    current_lock = subprocess.check_output(
                        [uv, "pip", "freeze", "--python", str(python)], text=True
                    )
                    if (
                        canonical_lock(current_lock, repo, commit)
                        != (root / "environment.lock.txt").read_text()
                    ):
                        raise RuntimeError(
                            "Worker dependencies changed after setup; refusing to mix measurements"
                        )
                    ensure_ninja(python, None, allow_install=False)
                    tool_env = ensure_toolchain(root, python, uv, allow_install=False)
                env = dict(
                    tool_env,
                    PYTHONPATH=str(root),
                    PYTHONUNBUFFERED="1",
                    TOKENIZERS_PARALLELISM="false",
                )
                status("starting experiment worker")
                subprocess.run(
                    [
                        str(python),
                        "-m",
                        "qspec_hybrid.extensions" if args.mode == "extend" else "qspec_hybrid.extensions2" if args.mode.startswith("fam_") else "qspec_hybrid.runner",
                        "--root",
                        str(root),
                        "--mode",
                        args.mode,
                    ],
                    cwd=root,
                    env=env,
                    check=True,
                )
            except KeyboardInterrupt:
                atomic_json(
                    root / "status.json",
                    {
                        "state": "stopped",
                        "stage": "interrupted; committed requests retained",
                        "mode": args.mode,
                    },
                )
            except Exception as error:
                previous = read_json(root / "status.json", {})
                if previous.get("state") not in ("blocked", "review_required", "failed"):
                    atomic_json(
                        root / "status.json",
                        {"state": "failed", "stage": str(error), "mode": args.mode},
                    )
                raise
            finally:
                pass

    finally:
        # The experiment worker has exited, its SQLite handle is closed, and the
        # exclusive experiment lock above has been released before snapshotting.
        try:
            print("Recovery bundle:", bundle(root), flush=True)
        except Exception as error:
            print(
                "Bundle could not be refreshed; raw results remain on disk:",
                error,
                flush=True,
            )


if __name__ == "__main__":
    main()
