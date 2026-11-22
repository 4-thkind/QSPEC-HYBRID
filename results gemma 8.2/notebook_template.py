# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo==0.25.1"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Hybrid VLM quantization × native MTP

    **Attach RTX PRO 6000 Blackwell (96 GB) in Molab notebook specifications.**
    Add `HF_TOKEN` through Molab Secrets if a dataset requires authentication.

    **1. Pilot test** installs a separate Python 3.12 worker, freezes data, calibrates and
    verifies all eight precision recipes, then freezes K=1 or K=2.
    **2. Full scale** runs the sixteen conditions only after the pilot passes.

    Main sweep: 200 images/task × four tasks × sixteen conditions, plus two extra
    timing passes on 40 images/task (**17,920 core requests**). A fresh 40/task hold-out
    and separately labelled diagnostic runs follow automatically.

    TextVQA · ChartQA · MMStar · **OmniDocBench cropped-text OCR only**, not its overall score.
    Source/interface checks and CPU tests do not certify GPU compatibility. Unsupported kernels,
    changed precisions or an unsuccessful pilot block full scale with saved diagnostics.
    Opening this notebook never starts installation, downloads or GPU work.
    Allow substantial scratch storage (roughly 150–200 GB free is a planning estimate;
    export sizes vary). Model weights are not included in result downloads.
    """)
    return


@app.cell
def _(mo):
    import base64 as _base64
    import hashlib as _hashlib
    import io as _io
    import sys as _sys
    import zipfile as _zipfile
    from pathlib import Path as _Path

    _payload = "@@PAYLOAD@@"
    _expected = "@@SHA256@@"
    _bytes = _base64.b64decode(_payload)
    if _hashlib.sha256(_bytes).hexdigest() != _expected:
        raise ValueError(
            "Embedded project checksum failed; upload the original notebook.py again"
        )
    _notebook_folder = mo.notebook_dir() or _Path.cwd()
    ROOT = (
        _notebook_folder
        if (_notebook_folder / "qspec_hybrid/core.py").exists()
        else _notebook_folder / ("qspec_hybrid_project_" + _expected[:8])
    )
    _marker = ROOT / ".source_bundle_sha256"
    if not (ROOT / "qspec_hybrid/core.py").exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        with _zipfile.ZipFile(_io.BytesIO(_bytes)) as _archive:
            for _entry in _archive.infolist():
                _destination = (ROOT / _entry.filename).resolve()
                if not _destination.is_relative_to(ROOT.resolve()):
                    raise ValueError("Unsafe embedded path")
                if not _entry.is_dir():
                    _destination.parent.mkdir(parents=True, exist_ok=True)
                    _destination.write_bytes(_archive.read(_entry))
        _marker.write_text(_expected)
    elif _marker.exists() and _marker.read_text().strip() != _expected:
        raise ValueError(
            "This project has a different source version; existing results were preserved"
        )
    if not (ROOT / "notebook.py").exists():
        _template = (ROOT / "notebook_template.py").read_text()
        _source = _template.replace("@@" + "PAYLOAD" + "@@", _payload).replace(
            "@@" + "SHA256" + "@@", _expected
        )
        (ROOT / "notebook.py").write_text(_source)
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    import qspec_hybrid.core as controller

    if _Path(controller.__file__).resolve().parent.parent != ROOT.resolve():
        raise ValueError(
            "Another project copy is already imported in this kernel. Restart the notebook kernel; existing results are untouched."
        )

    mo.md(
        f"Project ready at `{ROOT}`. Choose Pilot test first. Download recovery regularly: Molab storage may disappear when the session ends."
    )
    return ROOT, controller


@app.cell
def _(mo):
    pilot_button = mo.ui.run_button(label="1 · Pilot test", kind="neutral")
    full_button = mo.ui.run_button(label="2 · Full scale", kind="success")
    stop_button = mo.ui.run_button(label="Stop and save", kind="warn")
    mo.hstack([pilot_button, full_button, stop_button], justify="start")
    return full_button, pilot_button, stop_button


@app.cell
def _(ROOT, controller, full_button, mo, pilot_button):
    _message = (
        "Choose one run button. Clicking the same mode later resumes committed work."
    )
    if pilot_button.value or full_button.value:
        try:
            if pilot_button.value and full_button.value:
                raise ValueError("Click only one run button")
            _mode = "pilot" if pilot_button.value else "full"
            _gate = controller.read_json(ROOT / "pilot_gate.json", {})
            if _mode == "full" and not _gate.get("passed"):
                raise ValueError(
                    "Full scale is blocked until Pilot test passes. Read pilot_gate.json and the live log below."
                )
            controller.launch(ROOT, _mode)
            _message = f"Started {_mode}. Setup, data, export, measurements and saving run automatically in the worker."
        except Exception as _error:
            _message = f"Not started: {_error}"
    mo.md(_message)
    return


@app.cell
def _(ROOT, controller, mo, stop_button):
    _text = (
        controller.stop(ROOT)
        if stop_button.value
        else "Each completed request is committed immediately. Stop preserves it; restart the same mode to resume."
    )
    mo.md(_text)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(
        options=["5s", "15s"], default_interval="5s", label="Live progress"
    )
    refresh
    return (refresh,)


@app.cell
def _(ROOT, controller, mo, refresh):
    refresh.value
    _status = controller.read_json(ROOT / "status.json", {})
    _gate = controller.read_json(ROOT / "pilot_gate.json", {})
    _summary = controller.read_json(ROOT / "results/summary.json", {})
    _state = _status.get("state", "idle")
    _job = controller.read_json(ROOT / "job.json", {})
    if _state == "running" and _job and not controller.alive(_job):
        _state = "interrupted · worker exited; download or resume the same mode"
    _pieces = [
        mo.md(
            f"## Progress\n**State:** {_state} · **Stage:** {_status.get('stage', 'waiting')}\n\n**Saved records:** {_summary.get('saved_records', _status.get('saved_records', 0))} · **Pilot gate:** {'passed' if _gate.get('passed') else 'not passed yet'}"
        )
    ]
    if _gate.get("message"):
        _pieces.append(mo.md(_gate["message"]))
    _rows = _summary.get("summary", [])
    if _rows:
        _columns = (
            "phase",
            "policy",
            "k",
            "task",
            "scored_samples",
            "mean_score",
            "median_request_latency_s",
            "acceptance_rate",
            "peak_allocated_GiB",
            "token_mismatches",
        )
        _pieces.append(
            mo.ui.table(
                [{_key: _row.get(_key) for _key in _columns} for _row in _rows],
                page_size=8,
            )
        )
    if _state == "review_required":
        _pieces.append(
            mo.md(
                "Numerical differences remain in the results. Do not claim exact/lossless speedup or exclude those samples."
            )
        )
    mo.vstack(_pieces)
    return


@app.cell
def _(ROOT, mo, refresh):
    refresh.value
    _path = ROOT / "worker.log"
    _tail = "No worker started."
    if _path.exists():
        with _path.open("rb") as _stream:
            _stream.seek(max(0, _path.stat().st_size - 10000))
            _tail = _stream.read().decode(errors="replace")
    mo.accordion({"Live worker log": mo.plain_text(_tail)})
    return


@app.cell
def _(ROOT, mo, refresh):
    refresh.value
    _figures = [
        ROOT / "results/figures/mtp_speedup.png",
        ROOT / "results/figures/quality_latency.png",
        ROOT / "results/figures/acceptance_economics.png",
    ]
    _present = [_path for _path in _figures if _path.exists()]
    mo.vstack([mo.image(str(_path)) for _path in _present]) if _present else mo.md(
        "Figures appear after measured results are available. No example results are preloaded."
    )
    return


@app.cell
def _(mo):
    download_button = mo.ui.run_button(
        label="Prepare download · saved progress + results", kind="neutral"
    )
    download_button
    return (download_button,)


@app.cell
def _(ROOT, controller, download_button, mo):
    _view = mo.md(
        "Download includes source, frozen image inputs, evaluator versions, resumable SQLite results, scores, logs, audits and figures. Model weights/environments/secrets are excluded; a fresh session must rebuild/download them."
    )
    if download_button.value:
        try:
            _archive = controller.bundle(ROOT)
            _view = mo.vstack(
                [
                    mo.md(
                        f"Backup ready · {_archive.stat().st_size / 1024**2:.1f} MiB"
                    ),
                    mo.download(
                        lambda: _archive.read_bytes(),
                        filename="qspec_hybrid_results.zip",
                        label="Download ZIP",
                    ),
                ]
            )
        except Exception as _error:
            _view = mo.md(
                f"Download not prepared: {_error}. Raw committed results remain on disk."
            )
    _view
    return


if __name__ == "__main__":
    app.run()
