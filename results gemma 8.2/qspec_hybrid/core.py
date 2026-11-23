from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import sqlite3
import time
import zipfile
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
MODEL_OPT = "d7cbff3486a3ffb1a357e32d9810da0bfc649709"
LMMS = "9466a70af5086fc0b8ebcc785d8f105e65240fdc"
OMNI = "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
VLLM = "0.31.0"
SEED = 20261018
POLICIES = {
    "P0": (),
    "P1": ("D",),
    "P2": ("A",),
    "P3": ("G",),
    "P4": ("D", "A"),
    "P5": ("D", "G"),
    "P6": ("A", "G"),
    "P7": ("D", "A", "G"),
}
TASKS = ("textvqa", "chartqa", "mmstar", "omnidoc_ocr")
PROTOCOL = {
    "schema": 1,
    "model": MODEL,
    "modelopt": MODEL_OPT,
    "lmms": LMMS,
    "omni_evaluator": OMNI,
    "vllm": VLLM,
    "seed": SEED,
    "policies": POLICIES,
    "full_samples_per_task": 200,
    "timing_samples_per_task": 40,
    "additional_repeats": 2,
    "pilot_samples_per_task": 6,
    "calibration_samples": 256,
    "context": 16384,
    "max_pixels": 1003520,
    "min_pixels": 65536,
    "kv_cache_bytes": 2147483648,
    "cache_dtype": "bfloat16",
    "max_tokens": {"textvqa": 128, "chartqa": 16, "mmstar": 128, "omnidoc_ocr": 2048},
    "omni_scope": "longest eligible horizontal text block per page; cropped OCR only; not full-page overall",
    "acceptance": "native MTP; no tolerance or sample exclusions",
    "mtp_equivalence": "exact token match recorded per case; whitespace-only divergence that does not lower the score is reported but does not block",
    "quality_screening_budget": 0.01,
    "heldout_samples_per_task": 40,
    "replay_scope": "fixed-prefix continuation, first native verification step; not full teacher-forced replay",
    "pilot_gate": "all policies load; preserved precisions; finite metrics; no score-changing token mismatches",
}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Store:
    """Transactional requests; a crash never rewrites a completed request."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "requests.sqlite"
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.db.commit()

    def get(self, key):
        row = self.db.execute(
            "SELECT payload FROM requests WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        payload = json.dumps(value, allow_nan=False, ensure_ascii=False)
        with self.db:
            self.db.execute("INSERT INTO requests VALUES (?, ?)", (key, payload))

    def rows(self):
        return [
            json.loads(x[0])
            for x in self.db.execute("SELECT payload FROM requests ORDER BY rowid")
        ]

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@contextlib.contextmanager
def exclusive(root: Path):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "An experiment worker is already running in this project"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity(pid):
    """Linux PID identity guards against accidentally stopping a reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def alive(job):
    return bool(
        job
        and job.get("pid")
        and process_identity(job["pid"]) == job.get("birth")
        and job.get("birth")
    )


def launch(root: Path, mode: str):
    import subprocess
    import sys

    if mode not in ("pilot", "full", "extend", "fam_gemma", "fam_glmocr", "fam_exaone"):
        raise ValueError("Choose pilot, full, extend, fam_gemma, fam_glmocr or fam_exaone")
    from .lineage import cached_environment
    launch_env = cached_environment(root, dict(os.environ))
    if (root / "lineage.json").exists():
        from .lineage import validate_lineage
        validate_lineage(root)
        launch_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if mode == "full" and not read_json(root / "pilot_gate.json", {}).get("passed"):
        raise RuntimeError("Full scale requires a passing pilot in this lineage")
    old = read_json(root / "job.json", {})
    if alive(old):
        raise RuntimeError(
            "A worker is already active. Stop it before starting another."
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "STOP").unlink(missing_ok=True)
    log = root / "worker.log"
    with log.open("ab") as output:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(root / "bootstrap.py"),
                "--root",
                str(root),
                "--mode",
                mode,
            ],
            cwd=root,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=launch_env,
        )
    job = {
        "pid": proc.pid,
        "birth": process_identity(proc.pid),
        "mode": mode,
        "started": time.time(),
        "log": str(log),
    }
    atomic_json(root / "job.json", job)
    return job


def stop(root: Path):
    (root / "STOP").touch()
    for name in ("job.json", "conversion_job.json", "lineage_job.json"):
        job = read_json(root / name, {})
        if alive(job):
            os.killpg(job["pid"], signal.SIGTERM)
    return "Stop requested. Already committed requests remain saved."


def bundle(root: Path) -> Path:
    """Portable source, inputs and SQLite snapshot, never environments/weights/secrets."""
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    # Refuse snapshots while an experiment owns open handles/its lock.
    for job_name in ("job.json", "conversion_job.json", "lineage_job.json"):
        job = read_json(root / job_name, {})
        if alive(job) and job.get("pid") != os.getpid():
            raise RuntimeError("Stop the active worker before preparing recovery")
    with (root / "worker.lock").open("a+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Recovery waits until the experiment lock is released") from None
        fcntl.flock(probe, fcntl.LOCK_UN)
    with (root / "backup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _bundle_unlocked(root)


def _bundle_unlocked(root: Path) -> Path:
    target = root / "artifacts" / "qspec_hybrid_results.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot = root / "artifacts" / "requests.snapshot.sqlite"
    ext_db, ext_snapshot = root / "results" / "ext" / "requests.sqlite", root / "artifacts" / "ext.snapshot.sqlite"
    if ext_db.exists():
        with (
            contextlib.closing(sqlite3.connect(ext_db)) as source,
            contextlib.closing(sqlite3.connect(ext_snapshot)) as destination,
        ):
            source.backup(destination)
            destination.commit()
    db_path = root / "results" / "requests.sqlite"
    if db_path.exists():
        # sqlite connection contexts commit/rollback but do not close handles.
        with (
            contextlib.closing(sqlite3.connect(db_path)) as source,
            contextlib.closing(sqlite3.connect(snapshot)) as destination,
        ):
            source.backup(destination)
            destination.commit()
    temporary = target.with_suffix(".tmp")
    allowed_dirs = {"qspec_hybrid", "tests", "configs", "data", "results", "sources"}
    allowed_files = {
        "bootstrap.py",
        "notebook.py",
        "notebook_template.py",
        "README.md",
        "requirements.txt",
        "validation.json",
        "protocol.json",
        "worker.log",
        "status.json",
        "job.json",
        "conversion_job.json",
        "lineage.json",
        "lineage_job.json",
        "active_run.json",
        "environment.lock.txt",
        "revisions.json",
        "pilot_gate.json",
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        candidates = [root / name for name in allowed_files if (root / name).is_file()]
        for directory in sorted(allowed_dirs):
            folder = root / directory
            if folder.is_dir():
                candidates.extend(folder.rglob("*"))
        for path in sorted(candidates):
            rel = path.relative_to(root)
            if path.is_symlink() or not path.is_file():
                continue
            if rel.parts[0] in allowed_dirs or (
                len(rel.parts) == 1 and path.name in allowed_files
            ):
                if (
                    path.name.startswith(".")
                    or "__pycache__" in rel.parts
                    or rel.parts[:2] == ("data", "ext_images")
                    or path.suffix == ".gz"
                    or path.name.startswith("requests.sqlite")
                ):
                    continue
                archive.write(path, str(rel))
        if snapshot.exists():
            archive.write(snapshot, "results/requests.sqlite")
        if ext_snapshot.exists():
            archive.write(ext_snapshot, "results/ext/requests.sqlite")
        live_notebook = root.parent / "notebook.py"
        if not live_notebook.is_file() and (root / "lineage.json").exists():
            parent_root = Path(read_json(root / "lineage.json")["parent_root"])
            live_notebook = parent_root.parent / "notebook.py"
        if live_notebook.is_file() and "def _qrun_full_conversion_preflight(root, control):" in live_notebook.read_text():
            archive.write(live_notebook, "live_notebook.py")
        # Audits are informational on restore. Do not create model directories
        # containing a manifest but no weights: the exporter must rebuild them.
        for path in sorted((root / "models").glob("*/export_manifest.json")):
            name = f"results/export_audits/{path.parent.name}.json"
            if name not in archive.namelist():
                archive.write(path, name)
    os.replace(temporary, target)
    return target
