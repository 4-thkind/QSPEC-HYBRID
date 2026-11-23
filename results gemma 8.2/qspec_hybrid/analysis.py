from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict

from .core import POLICIES, PROTOCOL, TASKS, atomic_json, digest, read_json


def compact_rows(root):
    path = root / "results" / "requests.sqlite"
    if not path.exists():
        return []
    fields = {
        "phase": "$.phase",
        "policy": "$.policy",
        "k": "$.k",
        "case": "$.case_id",
        "task": "$.task",
        "group": "$.group",
        "replicate": "$.replicate",
        "timing": "$.timing",
        "latency_s": "$.request_latency_s",
        "end_to_end_s": "$.end_to_end_s",
        "tokens": "$.usage.completion_tokens",
        "ttft_ms": "$.native_metrics.time_to_first_token_ms",
        "decode_ms": "$.native_metrics.generation_time_ms",
        "score": "$.score.value",
        "metric": "$.score.metric",
        "higher": "$.score.higher_is_better",
        "accepted": "$.speculative.num_accepted_draft_tokens",
        "proposed": "$.speculative.num_draft_tokens",
        "steps": "$.speculative.num_spec_steps",
        "allocated": "$.memory[0].peak_allocated_bytes",
        "reserved": "$.memory[0].peak_reserved_bytes",
        "ids": "$.output_ids",
        "truncated": "$.finish_reason",
        "state": "$.state",
        "nvml_peak": "$.device_observations.nvml_worker_process_peak_bytes",
        "energy": "$.device_observations.device_energy_j",
        "target_forward_gpu_ms": "$.forward_profile[0].target_forward_gpu_ms",
        "draft_forward_gpu_ms": "$.forward_profile[0].draft_forward_gpu_ms",
        "state_only_gpu_ms": "$.forward_profile[0].state_only_gpu_ms",
        "first_round_accepted": "$.speculative.per_step_accepted[0]",
        "first_round_proposed": "$.speculative.per_step_drafted[0]",
    }
    with sqlite3.connect(path) as db:
        sql = (
            "SELECT "
            + ",".join(f"json_extract(payload, '{v}')" for v in fields.values())
            + " FROM requests"
        )
        return [dict(zip(fields, values)) for values in db.execute(sql)]


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def paired_interval(values, seed=20261018, repetitions=2000, geometric=False):
    import numpy as np

    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    if geometric:
        array = np.log(array)
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            rng.choice(array, size=len(array), replace=True).mean()
            for _ in range(repetitions)
        ]
    )
    point = array.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    if geometric:
        point, lo, hi = math.exp(point), math.exp(lo), math.exp(hi)
    return float(point), float(lo), float(hi)


def summarize(root, figures=False, intervals=False):
    import statistics

    all_rows = compact_rows(root)
    rows = [r for r in all_rows if r["state"] == "ok"]
    groups = defaultdict(list)
    lookup = {
        (r["phase"], r["policy"], r["k"], r["case"], r["replicate"]): r for r in rows
    }
    summary = []
    for row in rows:
        if row["phase"] in ("pilot", "full", "heldout"):
            groups[(row["phase"], row["policy"], row["k"], row["task"])].append(row)
    for (phase, policy, k, task), items in sorted(groups.items()):
        first = [x for x in items if x["replicate"] == 0]
        timing = [x for x in items if x["timing"]]
        by_case = defaultdict(list)
        for item in timing:
            by_case[item["case"]].append(item["latency_s"])
        acceptance_rows = [x for x in first if x["proposed"] is not None]
        accepted = sum(x["accepted"] for x in acceptance_rows)
        proposed = sum(x["proposed"] for x in acceptance_rows)
        steps = sum(x["steps"] for x in acceptance_rows)
        mismatch, compared = 0, 0
        if k:
            for item in first:
                baseline = lookup.get((phase, policy, 0, item["case"], 0))
                if baseline:
                    compared += 1
                    mismatch += item["ids"] != baseline["ids"]
        repeated_differences = sum(
            x["ids"] != lookup.get((phase, policy, k, x["case"], 0), x)["ids"]
            for x in items
            if x["replicate"] > 0
        )
        scores = [x["score"] for x in first if x["score"] is not None]
        timing_medians = [statistics.median(v) for v in by_case.values()]
        record = {
            "phase": phase,
            "policy": policy,
            "k": k,
            "task": task,
            "scored_samples": len(scores),
            "metric": first[0]["metric"] if first else None,
            "mean_score": statistics.mean(scores) if scores else None,
            "timing_images": len(by_case),
            "timing_requests": len(timing),
            "median_request_latency_s": statistics.median(timing_medians)
            if timing_medians
            else None,
            "median_ttft_ms": statistics.median(
                x["ttft_ms"] for x in timing if x["ttft_ms"] is not None
            )
            if any(x["ttft_ms"] is not None for x in timing)
            else None,
            "acceptance_rate": accepted / proposed if proposed else None,
            "conventional_mean_acceptance_length": 1 + accepted / steps
            if steps
            else None,
            "accepted_tokens": accepted,
            "proposed_tokens": proposed,
            "verify_steps": steps,
            "observed_request_output_tokens_per_verify_step": sum(
                x["tokens"] for x in acceptance_rows
            )
            / steps
            if steps
            else None,
            "median_decode_ms_per_token_interval": statistics.median(
                x["decode_ms"] / (x["tokens"] - 1)
                for x in timing
                if x["decode_ms"] is not None and x["tokens"] > 1
            )
            if any(x["decode_ms"] is not None and x["tokens"] > 1 for x in timing)
            else None,
            "peak_allocated_GiB": max((x["allocated"] or 0) for x in items) / 1024**3,
            "peak_reserved_GiB": max((x["reserved"] or 0) for x in items) / 1024**3,
            "sampled_peak_worker_process_GiB": max(
                x["nvml_peak"] for x in items if x["nvml_peak"] is not None
            )
            / 1024**3
            if any(x["nvml_peak"] is not None for x in items)
            else None,
            "median_device_energy_j": statistics.median(
                x["energy"] for x in timing if x["energy"] is not None
            )
            if any(x["energy"] is not None for x in timing)
            else None,
            "compared_outputs": compared,
            "token_mismatches": mismatch,
            "repeat_output_changes": repeated_differences,
            "truncated_samples": sum(x["truncated"] == "length" for x in first),
            "lossless_claim_supported": bool(
                k
                and compared == len(first)
                and mismatch == 0
                and repeated_differences == 0
            ),
        }
        summary.append(record)
    comparisons = []
    for record in summary:
        if not record["k"]:
            continue
        phase, policy, k, task = (record[x] for x in ("phase", "policy", "k", "task"))
        (
            speed_ratios,
            quality_deltas,
            quant_ratios,
            total_ratios,
            quant_quality_deltas,
        ) = [], [], [], [], []
        for item in groups[(phase, policy, k, task)]:
            if item["replicate"] != 0:
                continue
            baseline = lookup.get((phase, policy, 0, item["case"], 0))
            if not baseline:
                continue
            if item["score"] is not None and baseline["score"] is not None:
                quality_deltas.append(item["score"] - baseline["score"])
            if item["timing"]:
                mtp = [
                    lookup[(phase, policy, k, item["case"], r)]["latency_s"]
                    for r in range(3)
                    if (phase, policy, k, item["case"], r) in lookup
                ]
                ar = [
                    lookup[(phase, policy, 0, item["case"], r)]["latency_s"]
                    for r in range(3)
                    if (phase, policy, 0, item["case"], r) in lookup
                ]
                speed_ratios.append(statistics.median(ar) / statistics.median(mtp))
                reference = [
                    lookup[(phase, "P0", 0, item["case"], r)]["latency_s"]
                    for r in range(3)
                    if (phase, "P0", 0, item["case"], r) in lookup
                ]
                if reference:
                    quant_ratios.append(
                        statistics.median(reference) / statistics.median(ar)
                    )
                    total_ratios.append(
                        statistics.median(reference) / statistics.median(mtp)
                    )
            reference_score = lookup.get((phase, "P0", 0, item["case"], 0))
            if (
                reference_score
                and reference_score["score"] is not None
                and baseline["score"] is not None
            ):
                quant_quality_deltas.append(
                    baseline["score"] - reference_score["score"]
                )
        estimate = (
            math.exp(statistics.mean(math.log(x) for x in speed_ratios))
            if speed_ratios
            else None
        )
        ci = (
            paired_interval(speed_ratios, geometric=True)
            if intervals
            else (estimate, None, None)
        )
        quality = (
            paired_interval(quality_deltas)
            if intervals
            else (
                statistics.mean(quality_deltas) if quality_deltas else None,
                None,
                None,
            )
        )
        quant_quality = (
            paired_interval(quant_quality_deltas)
            if intervals
            else (
                statistics.mean(quant_quality_deltas) if quant_quality_deltas else None,
                None,
                None,
            )
        )

        def speed_interval(values):
            return (
                paired_interval(values, geometric=True)
                if intervals
                else (
                    math.exp(statistics.mean(math.log(x) for x in values))
                    if values
                    else None,
                    None,
                    None,
                )
            )

        quant_ci, total_ci = speed_interval(quant_ratios), speed_interval(total_ratios)
        comparisons.append(
            {
                "phase": phase,
                "policy": policy,
                "k": k,
                "task": task,
                "paired_timing_images": len(speed_ratios),
                "mtp_incremental_speedup": ci[0],
                "speedup_ci95_low": ci[1],
                "speedup_ci95_high": ci[2],
                "quantization_only_speedup_vs_BF16_AR": quant_ci[0],
                "quantization_speedup_ci95_low": quant_ci[1],
                "quantization_speedup_ci95_high": quant_ci[2],
                "combined_speedup_vs_BF16_AR": total_ci[0],
                "combined_speedup_ci95_low": total_ci[1],
                "combined_speedup_ci95_high": total_ci[2],
                "quantized_AR_minus_BF16_AR_score": quant_quality[0],
                "quantized_AR_score_delta_ci95_low": quant_quality[1],
                "quantized_AR_score_delta_ci95_high": quant_quality[2],
                "mtp_minus_same_precision_ar_score": quality[0],
                "score_delta_ci95_low": quality[1],
                "score_delta_ci95_high": quality[2],
            }
        )
    result = {
        "summary": summary,
        "comparisons": comparisons,
        "saved_records": len(rows),
        "caveats": [
            "OmniDocBench is cropped-text OCR NED only, not full-page overall.",
            "Acceptance is native and strict; mismatched outputs are retained.",
            "Zero draft opportunities on tiny answers yield null acceptance, not zero.",
            "Eager diagnostic forward timings are separate from compiled headline results.",
            "Common-prefix continuations are not full teacher-forced MTP replay.",
            "Bootstrap units are selected images/pages, not tokens; source-PDF clustering is not inferred from filenames. Subsets are not complete benchmarks.",
            "Output tokens/verify-step includes request boundary tokens; conventional acceptance length is reported separately.",
            "ScienceQA image-text calibration is fixed, not claimed optimal for OCR/documents.",
        ],
    }
    atomic_json(root / "results" / "summary.json", result)
    csv_file(root / "results" / "summary.csv", summary)
    csv_file(root / "results" / "comparisons.csv", comparisons)
    diagnostics = [
        {k: v for k, v in row.items() if k != "ids"}
        for row in all_rows
        if row["phase"] in ("profile", "prefix_continuation")
    ]
    csv_file(root / "results" / "diagnostics.csv", diagnostics)
    if figures or intervals:
        csv_file(
            root / "results" / "request_metrics.csv",
            [{k: v for k, v in row.items() if k != "ids"} for row in all_rows],
        )
    if figures and summary:
        plot(root, summary, comparisons)
    return result


def plot(root, summary, comparisons):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = root / "results" / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    phase = "full" if any(x["phase"] == "full" for x in summary) else "pilot"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task in zip(axes.flat, TASKS):
        data = [
            x
            for x in comparisons
            if x["phase"] == phase
            and x["task"] == task
            and x["mtp_incremental_speedup"] is not None
        ]
        labels = [f"{x['policy']} K{x['k']}" for x in data]
        values = [x["mtp_incremental_speedup"] for x in data]
        ax.barh(labels, values, color="#376b9d")
        for index, item in enumerate(data):
            if item["speedup_ci95_low"] is not None:
                ax.errorbar(
                    item["mtp_incremental_speedup"],
                    index,
                    xerr=[
                        [
                            max(
                                0,
                                item["mtp_incremental_speedup"]
                                - item["speedup_ci95_low"],
                            )
                        ],
                        [
                            max(
                                0,
                                item["speedup_ci95_high"]
                                - item["mtp_incremental_speedup"],
                            )
                        ],
                    ],
                    color="#333333",
                    capsize=3,
                    fmt="none",
                )
        ax.axvline(1, color="#333333", linestyle="--", linewidth=1)
        ax.set(
            xlabel="MTP speedup vs same-precision AR (×)",
            title=f"{task} · {phase} subset",
            xlim=(0, None),
        )
    fig.savefig(folder / "mtp_speedup.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task in zip(axes.flat, TASKS):
        for item in summary:
            if (
                item["phase"] != phase
                or item["task"] != task
                or not item["k"]
                or item["mean_score"] is None
            ):
                continue
            x = item["median_request_latency_s"]
            if x is None:
                continue
            ax.scatter(x, item["mean_score"], color="#376b9d", marker="o")
            ax.annotate(
                item["policy"],
                (x, item["mean_score"]),
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set(
            xlabel="Median request latency (s)",
            ylabel="NED ↓" if task == "omnidoc_ocr" else "Accuracy ↑",
            title=f"{task} · {phase} subset",
        )
    fig.savefig(folder / "quality_latency.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task in zip(axes.flat, TASKS):
        for item in summary:
            if (
                item["phase"] != phase
                or item["task"] != task
                or not item["k"]
                or item["acceptance_rate"] is None
            ):
                continue
            comparison = next(
                (
                    x
                    for x in comparisons
                    if x["phase"] == phase
                    and x["policy"] == item["policy"]
                    and x["k"] == item["k"]
                    and x["task"] == task
                ),
                None,
            )
            if comparison and comparison["mtp_incremental_speedup"] is not None:
                x, y = (
                    item["acceptance_rate"] * 100,
                    comparison["mtp_incremental_speedup"],
                )
                ax.scatter(x, y, color="#376b9d")
                ax.annotate(
                    f"{item['policy']} K{item['k']}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax.axhline(1, color="#333333", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Accepted / proposed draft tokens (%)",
            ylabel="MTP incremental speedup (×)",
            title=f"{task} · {phase} subset",
            xlim=(0, 100),
        )
    fig.savefig(folder / "acceptance_economics.png", dpi=160)
    plt.close(fig)


def select_candidates(root):
    """Predeclared screening guardrail, not a proof of statistical equivalence."""
    import statistics

    rows = compact_rows(root)
    rows = [
        r
        for r in rows
        if r["phase"] == "full" and r["replicate"] == 0 and r["state"] == "ok"
    ]
    k = read_json(root / "pilot_gate.json")["k"]
    scores, latency = defaultdict(list), defaultdict(list)
    for row in rows:
        scores[(row["policy"], row["k"], row["task"])].append(row["score"])
        latency[(row["policy"], row["k"], row["task"])].append(row["latency_s"])
    ranked = []
    for policy in list(POLICIES)[1:]:
        valid, ratios = True, []
        for task in TASKS:
            ref = scores[("P0", 0, task)]
            current = scores[(policy, k, task)]
            if len(ref) != 200 or len(current) != 200:
                valid = False
                break
            difference = statistics.mean(current) - statistics.mean(ref)
            if (difference if task == "omnidoc_ocr" else -difference) > PROTOCOL[
                "quality_screening_budget"
            ]:
                valid = False
            ratios.append(
                statistics.median(latency[("P0", 0, task)])
                / statistics.median(latency[(policy, k, task)])
            )
        if valid:
            ranked.append(
                (math.exp(statistics.mean(math.log(x) for x in ratios)), policy)
            )
    return {
        "policies": ["P0"] + [p for _, p in sorted(ranked, reverse=True)[:2]],
        "quality_screening_budget": 0.01,
        "not_a_noninferiority_test": True,
        "criterion": "At most 0.01 absolute mean-score loss on each development-screen task; rank by equal-task geometric mean speedup vs P0 AR",
        "screen_digest": digest(rows),
    }
