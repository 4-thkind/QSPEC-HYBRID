# QSpec Hybrid · Molab / marimo

## Start here

Upload **dist/qspec_hybrid_molab.py** to Molab as a Python/marimo notebook. It is self-contained: support files are embedded and unpacked into a versioned project directory. Do not convert it to an `.ipynb` or paste the whole file into one marimo cell.

1. Attach **RTX PRO 6000 Blackwell 96 GB / SM120** in notebook specifications. Confirm the GPU attachment survives a runtime restart.
2. If authentication is required, add `HF_TOKEN` using Molab Secrets. Do not put a token in a notebook cell.
3. Click **Pilot test**. Setup, dataset freezing, reference download, seven selective quantized exports, precision audits and numerical checks run automatically. This first pilot includes substantial installation/export work; it is not a ten-second smoke test.
4. After the pilot gate passes, click **Full scale**. No separate prepare/freeze/develop/configuration controls are needed.
5. Use **Prepare download** and **Download ZIP** regularly. Each completed request is already committed to SQLite; downloadable recovery bundles refresh automatically at condition boundaries and periodically during runs.

There are exactly two experiment modes. Stop/save and download are utility controls, not extra experiment stages. Opening the notebook starts no network, installation or GPU work. Background status/log cells do not depend on a stopped `mo.stop` ancestor.

The conventional source package is `dist/qspec_hybrid_molab.zip`; unpack it and open `notebook.py` if folder upload is preferable.

## What runs

- Main checkpoint: `Qwen/Qwen3.5-9B`, revision frozen on first preparation.
- vLLM **0.31.0**; official ModelOpt source **d7cbff3486a3ffb1a357e32d9810da0bfc649709**.
- Separate Python **3.12** worker; the Molab/marimo Python environment is not replaced.
- Decoder MLP (D): **NVFP4 W4A4**. Full-attention projections (A) and large GDN projections (G): **FP8 W8A8**.
- Eight precision policies: none, D, A, G, D+A, D+G, A+G, D+A+G. Every policy has ordinary decoding and native MTP.
- Vision, merger, embeddings, output head, MTP and all unselected weights are preserved. Native GDN gates/convolution/state remain unquantized. KV storage is explicitly audited as BF16 (`auto` with a BF16 model), with a fixed 2 GiB cache budget.
- One image/request, concurrency one, context 16,384, fixed pixel bounds, no CPU model offload, prefix caching off, async scheduling off, image processor cache off. Serving is loopback-only.
- Frozen 256-image ScienceQA/train image–text calibration corpus, shared across all policies. This is calibration, not training. It is not claimed to be optimal for OCR/document quantization.
- Calibration, pilot/development, main screen and fresh hold-out use disjoint decoded image hashes. Duplicate-input exclusions are fixed before generation and recorded; numerical failures are never used for sample selection.

The installed serving/exporter dependency combination has **not** been GPU-validated by the authoring environment. Do not confuse a source/interface audit or CPU tests with a hardware certification. In particular, CUDA/driver compatibility, the exact native MTP path and selective mixed-precision exports must pass the pilot. Newer vLLM binaries may require a newer NVIDIA driver than an old Molab image provides; setup reports the actual GPU/driver and does not silently downgrade the backend.

## Pilot

Six examples/task, 24 development inputs, including longer OCR outputs. P0 tests K=1 and K=2, including repeat sentinels; the fastest eligible K is frozen across every policy. All remaining recipes run AR and the selected K with repeat sentinels.

The successful path has **408 first-pass pilot requests plus 68 repeat sentinels**, and unscored warm-ups. All eight recipes must load, maintain the intended precision/kernel map, preserve unselected weights, return native counters, and pass token/repeat checks. A negative speedup is allowed. Mismatches and missing kernels are saved and block full scale, not hidden or “repaired” by tolerance.

## Full scale

| Task | Main samples | Generation/evaluation |
|---|---:|---|
| TextVQA | 200 | Pinned VQA consensus scoring over all ten human answers; 128-token cap |
| ChartQA | 200 | Pinned relaxed accuracy; 16-token cap |
| MMStar | 200 | Pinned MCQ extraction and accuracy; 128-token cap |
| OmniDocBench cropped-text OCR | 200 | Longest eligible horizontal text paragraph per selected page, cropped by annotation; raw text normalized edit distance; 2,048-token cap |

The OmniDocBench task is deliberately a **cropped text-block OCR subset**, not full-page parsing, TEDS, formula CDM, reading order or an overall OmniDocBench score. The official metric's per-block formula is checked against its pinned implementation. Ground-truth and predicted text are scored without invented text normalization. TextVQA/ChartQA/MMStar prompts and scoring helpers come from lmms-eval commit **9466a70af5086fc0b8ebcc785d8f105e65240fdc**. OmniDocBench metric source is commit **f133a71e9e91c3621c7ce8994200a7b394a06eb3**. Full source files and checksums are included under `sources/`; no fragile editable-install commit metadata check is used.

Each of 16 conditions runs all 800 main inputs once. Two extra timing passes use the same frozen 40 inputs/task, with condition/mode ordering reversed/balanced across passes: **17,920 core requests**. Quality uses one first-pass response per input, not three pseudoreplicated scores.

After this development screen, up to two candidate policies are selected using a predeclared 0.01 absolute score-loss screening budget on each task and equal-task geometric-mean speedup. This budget is **not a statistical noninferiority claim**. P0 plus those candidates run a fresh 40/task hold-out (320–960 additional requests).

Finally P0/P1/P7 run separate eager GPU-forward profiling and common-prefix continuation diagnostics (48 additional requests, plus warm-ups). The latter controls the initial image/text input; it is **not full teacher-forced MTP replay**, and subsequent generated prefixes can differ. Forward hooks exclude shared output projection/sampling/state operations. **State-only GPU latency remains unavailable**, rather than being incorrectly equated to residual wall time. No Qwen3.8-27B replication or full vision-encoder quantization is silently added to this single-model package.

## Results and definitions

- `results/requests.sqlite`: immutable successful request records and separately retained failed attempts; SQLite WAL + FULL synchronous transactions.
- `results/summary.json`, `summary.csv`: progressively refreshed task/policy summaries.
- `results/comparisons.csv`: quantization-only, incremental-MTP and combined speedup references; paired image/page-bootstrap 95% intervals at final analysis; quality deltas, truncation, mismatches and repeat changes.
- `results/request_metrics.csv`, `diagnostics.csv`: flat request metrics (including failed attempts), eager-forward timings and fixed-prefix diagnostic observations. Original token IDs/full responses remain in SQLite.
- `results/servers/*/runtime_precision.json`: actual loaded quantization methods, kernel classes, parameter dtypes/shapes and cache dtype.
- `results/export_audits/*.json`: recipes, calibration identity, preservation checks and weight-content fingerprints.
- `results/full_correctness.json`, `heldout_correctness.json`, `prefix_checks.json`: full-cohort checks, not cherry-picked passes.
- `results/figures/`: MTP speedup, quality/latency and acceptance/economics figures.
- `data/prepared.json`, `revisions.json`, `environment.lock.txt`, `protocol.json`: frozen inputs, revisions, exact resolved worker packages and definitions.
- `status.json`, `worker.log`, per-server logs, `artifacts/qspec_hybrid_results.zip`: live state, diagnostics and downloadable recovery.

Recorded metrics include token IDs, raw text, native detailed acceptance histograms/per-step counts, TTFT, decode time, client request latency, local payload preparation, end-to-end latency, output count/throughput, actual GPU-worker peak allocated/reserved memory, scores and stop reasons. Conventional mean acceptance length and observed request output/verification-step ratios are labelled separately because boundary/EOS tokens differ. Single-token outputs have no meaningful per-token decode interval. Requests with no draft opportunities have null acceptance rather than a fabricated zero.

Headline timings use compiled serving without profiling hooks. Eager diagnostic GPU-forward times are separately labelled and cannot be substituted for headline compiled latency. Optional NVML records add 50 ms sampled worker-process memory and a **device-level** energy-counter delta where supported; unavailable observations are null, energy is suppressed if another GPU compute process is detected, and short-request counter resolution may be inadequate. This is not per-process energy attribution. Source-PDF identities are not inferred from filenames: bootstrap units are selected images/pages, not guaranteed independent source documents. See the saved caveats before drawing causal or lossless conclusions.

If new main/hold-out numerical mismatches appear after a passing pilot, the full cohort stays saved and the final status becomes `review_required`; it is not reported as a lossless success. Actual execution/scoring errors stop that worker, preserve the failed attempt and leave all completed requests resumable.

## Stop, resume and recover

Click **Stop and save**, wait for shutdown, then download. Click the same experiment mode to resume. A reused PID is checked before a stop signal is sent. Only one project worker can run at a time. Interrupted environment creation is reusable; the setup does not unconditionally recreate an existing virtual environment.

Recovery ZIPs contain code, inputs/images, evaluator source, exact environment lock, coherent SQLite snapshot, logs, audits and figures. They exclude model weights, tool clones, virtual environments, `.env` files and secrets. A new session must download/rebuild weights. Rebuilt compressed tensor fingerprints must match the saved experiment before results can be mixed. A software/GPU-class/code change requires a new experiment, not an overwrite of previous results.

To restore, unzip a recovery bundle into a folder, open its `notebook.py`, reattach the GPU and click the same mode. `mo.notebook_dir()` locates the sibling support/results files even if the working directory differs. The source ZIP/standalone original file should be retained separately. Molab storage is not a durable cloud backup service; the notebook cannot guarantee persistence after the session is destroyed unless you downloaded the ZIP.

Exports require substantial disk storage; a low-disk guard stops before another export when less than 20 GiB remains. Partial exports are retained with distinct temporary names rather than overwriting a verified checkpoint. Installation, data download, export, server startup and warm-ups are excluded from headline request latency but remain part of total experiment wall time.

## Verification and limits

Run local CPU tests with `PYTHONPATH=. python -m pytest tests -q`; marimo format checks with `marimo check notebook.py`. The tests cover mathematical/scorer contracts, configuration guards, persistence, backup boundaries, API stream parsing and mocked workflow/resume behavior. They are not model experiments. `validation.json` records the authoring checks actually performed and the missing GPU validation.

The actual Molab **Pilot test** is the required next verification. This package does not promise error-free execution, a positive speedup, preserved benchmark quality, or conference acceptance.

## Upstream notices

The unmodified reference source files in `sources/` retain their original attribution. lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval ; its VQA processor cites the originating MMF/Pythia implementation. OmniDocBench: https://github.com/opendatalab/OmniDocBench . NVIDIA Model Optimizer and vLLM are installed from their own licensed distributions/repositories; this package does not redistribute model weights or evaluation datasets. Follow the upstream licenses and dataset research-use restrictions.
