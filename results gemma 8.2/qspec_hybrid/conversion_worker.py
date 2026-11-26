from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from .conversion import run_export_phase, smoke_exports
from .core import POLICIES, atomic_json, bundle, exclusive, launch, read_json
from .quant import adopt_unfinished_export
from .conversion import local_base


def run(root: Path, smoke_only=False):
    root = root.resolve()
    status_path = root / "results" / "conversion_preflight" / "status.json"
    phases = (("P1", "P2", "P3"), ("P4", "P5", "P6", "P7"))
    passed = False
    conversion_passed = False
    failure = None
    try:
        with exclusive(root):
            if smoke_only:
                for policy in tuple(POLICIES)[1:]:
                    report = read_json(root / "results/conversion_preflight" / f"{policy}.json", {})
                    manifest = read_json(root / "models" / policy / "export_manifest.json", {})
                    if (report.get("status") != "pass" or
                        report.get("tensor_contract", {}).get("weights_content_sha256") != manifest.get("weights_content_sha256")):
                        raise RuntimeError(f"Cannot resume smoke gate without matching passing audit: {policy}")
                    for name, size in manifest["file_sizes"].items():
                        path = root / "models" / policy / name
                        if not path.is_file() or path.stat().st_size != size:
                            raise RuntimeError(f"Checkpoint changed since conversion gate: {policy}/{name}")
            else:
                atomic_json(status_path, {"state": "running", "stage": "P1-P3", "pid": os.getpid()})
                destination = root / "models" / "P1"
                if not destination.exists():
                    candidates = [
                        path
                        for path in sorted((root / "models").glob("P1-export-*/weights"))
                        if (path / "config.json").exists() and any(path.glob("*.safetensors"))
                    ]
                    if len(candidates) != 1:
                        raise RuntimeError(f"Expected one completed P1 staging export, found {candidates}")
                    adopt_unfinished_export(root, local_base(root), "P1", candidates[0])

                first = run_export_phase(root, phases[0])
                if len(first) != len(phases[0]) or any(item["status"] != "pass" for item in first):
                    raise RuntimeError("P1-P3 conversion gate blocked")
                atomic_json(status_path, {"state": "running", "stage": "P4-P7", "pid": os.getpid()})
                second = run_export_phase(root, phases[1])
                if len(second) != len(phases[1]) or any(item["status"] != "pass" for item in second):
                    raise RuntimeError("P4-P7 conversion gate blocked")
            atomic_json(status_path, {"state": "running", "stage": "vLLM smoke", "pid": os.getpid()})
            smoke = smoke_exports(root, tuple(POLICIES)[1:])
            if not smoke["passed"]:
                raise RuntimeError("vLLM smoke gate blocked")
            conversion_passed = True
            for policy in tuple(POLICIES)[1:]:
                report_path = root / "results/conversion_preflight" / f"{policy}.json"
                report = read_json(report_path)
                evidence = [item for item in smoke["results"] if item["policy"] == policy]
                report["hardware_kernel_validation_completed"] = True
                report["hardware_kernel_validation"] = {
                    "passed": all(item["passed"] for item in evidence),
                    "eager": True, "max_generated_tokens": 2,
                    "modes_tested": [item["k"] for item in evidence],
                    "server_evidence": [item["server_folder"] for item in evidence],
                    "caveat": "Short loading/generation smoke; compiled execution and full pilot metrics remain separate gates.",
                }
                atomic_json(report_path, report)
            # Repaired source changes the measurement fingerprint. Validate the
            # saved lineage synchronously before claiming a pilot was resumed.
            from .runner import environment
            environment(root)
            passed = True
    except BaseException as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    # The conversion lock and every server/file context above have closed.
    status = {
        "state": "passed" if passed else "blocked",
        "stage": "conversion and smoke complete" if passed else ("pilot resume compatibility" if conversion_passed else "conversion gate"),
        "conversion_gates_passed": conversion_passed,
        "failure": failure,
        "pilot_resumed": False,
        "full_scale_started": False,
    }
    atomic_json(status_path, status)
    recovery = bundle(root)
    status["recovery_bundle"] = str(recovery)
    if passed:
        status["pilot_job"] = launch(root, "pilot")
        status["pilot_resumed"] = True
        status["stage"] = "pilot resumed"
    atomic_json(status_path, status)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.root, smoke_only=args.smoke_only))


if __name__ == "__main__":
    main()
