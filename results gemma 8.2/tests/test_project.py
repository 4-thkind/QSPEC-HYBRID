"""CPU contract tests. Mocked requests are NEVER research measurements."""

import json
import sqlite3
import sys
import types
import zipfile
from pathlib import Path

import pytest

from qspec_hybrid import analysis, core, data, quant, runner
from qspec_hybrid.audit import validate_runtime
from qspec_hybrid.runtime import command
from qspec_hybrid.upstream import Scorers, prepare as prepare_scorers


def test_matrix_is_complete():
    assert len(core.POLICIES) == 8
    assert len({frozenset(x) for x in core.POLICIES.values()}) == 8
    assert len(core.POLICIES) * 2 == 16
    assert 200 * 4 * 16 + 40 * 4 * 16 * 2 == 17920


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.1.mlp.gate_up_proj", "D"),
        ("model.layers.1.mlp.down_proj.weight", "D"),
        ("model.layers.3.self_attn.qkv_proj", "A"),
        ("model.layers.3.self_attn.o_proj", "A"),
        ("model.layers.0.linear_attn.in_proj_qkvz", "G"),
        ("model.layers.0.linear_attn.out_proj", "G"),
        ("model.layers.0.linear_attn.in_proj_ba", "protected"),
        ("model.layers.0.linear_attn.conv1d", "protected"),
        ("mtp.layers.0.mlp.down_proj", "protected"),
        ("visual.blocks.0.mlp.down_proj", "protected"),
        ("lm_head", "protected"),
        ("model.embed_tokens", "protected"),
    ],
)
def test_grouping(name, expected):
    assert quant.classify(name) == expected


@pytest.mark.parametrize("policy", core.POLICIES)
def test_recipe_exclusions_last(policy):
    cfg = quant.recipe(policy)
    assert cfg["quantize"]["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}
    rules = cfg["quantize"]["quant_cfg"]
    assert any(
        x.get("quantizer_name") == "*mtp*" and x.get("enable") is False for x in rules
    )
    assert any(
        x.get("quantizer_name") == "*lm_head*" and x.get("enable") is False
        for x in rules
    )
    assert not any("kv_" in str(x) for x in rules)
    assert all(x.get("enable") is False for x in rules[-9:])


def test_store_resume_and_failure_immutability(tmp_path):
    with core.Store(tmp_path / "results") as db:
        db.put("request", {"state": "ok", "text": "retained"})
        with pytest.raises(sqlite3.IntegrityError):
            db.put("request", {"text": "overwrite"})
    with core.Store(tmp_path / "results") as db:
        assert db.get("request")["text"] == "retained"


def test_bundle_excludes_weights_and_takes_sqlite_snapshot(tmp_path):
    (tmp_path / "models/P1").mkdir(parents=True)
    (tmp_path / "models/P1/model.safetensors").write_bytes(b"not a real model")
    core.atomic_json(tmp_path / "models/P1/export_manifest.json", {"audited": True})
    (tmp_path / ".env").write_text("HF_TOKEN=TEST_SECRET_DO_NOT_INCLUDE")
    with core.Store(tmp_path / "results") as db:
        db.put("x", {"state": "ok"})
        archive = core.bundle(tmp_path)
    with zipfile.ZipFile(archive) as z:
        assert "results/requests.sqlite" in z.namelist()
        assert "results/export_audits/P1.json" in z.namelist()
        assert not any(n.startswith("models/") for n in z.namelist())
        assert not any("safetensors" in n or n == ".env" for n in z.namelist())


def test_stratification_is_deterministic():
    rows = [{"id": i, "stratum": str(i % 4)} for i in range(100)]
    a, b = data.balanced(rows, 20), data.balanced(rows, 20)
    assert a == b
    assert len({x["id"] for x in a}) == 20
    assert all(sum(x["stratum"] == str(i) for x in a) == 5 for i in range(4))
    with pytest.raises(ValueError):
        data.balanced(rows, 101)


def test_runtime_command_has_pinned_safety_controls():
    args = command("/model", 9876, 2)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--no-enable-prefix-caching" in args and "--no-async-scheduling" in args
    assert args[args.index("--kv-cache-dtype") + 1] == "auto"
    assert args[args.index("--cpu-offload-gb") + 1] == "0"
    assert (
        json.loads(args[args.index("--speculative-config") + 1])[
            "num_speculative_tokens"
        ]
        == 2
    )
    assert "--enforce-eager" not in args


@pytest.mark.parametrize("bad", ["mtp", "cache", "target"])
def test_runtime_audit_fails_closed(bad):
    audit = {
        "mtp_found": True,
        "kv_cache_dtype": "torch.bfloat16",
        "modules": [
            {
                "scope": "target",
                "group": g,
                "quantized": g == "D",
                "name": g,
                "kernel": "FlashInferNvFp4LinearKernel",
            }
            for g in ("D", "A", "G")
        ],
    }
    if bad == "mtp":
        audit["modules"].append(
            {"scope": "mtp", "group": "protected", "quantized": True, "name": "bad"}
        )
    elif bad == "cache":
        audit["kv_cache_dtype"] = "torch.float8_e4m3fn"
    else:
        audit["modules"][1]["quantized"] = True
    with pytest.raises(RuntimeError):
        validate_runtime([audit], "P1", 2)


@pytest.fixture
def scorers_root(tmp_path):
    import shutil

    source = Path(__file__).resolve().parents[1] / "sources"
    shutil.copytree(source, tmp_path / "sources")
    prepare_scorers(tmp_path)
    return tmp_path


def test_official_vqa_and_mcq_scorers(scorers_root):
    scorer = Scorers(scorers_root)
    assert scorer.score({"task": "textvqa", "target": ["yes"] * 3 + ["no"] * 7}, "yes")[
        "value"
    ] == pytest.approx(0.9)
    assert scorer.score({"task": "textvqa", "target": ["2"] * 10}, "two")["value"] == 1
    assert scorer.score({"task": "chartqa", "target": "100"}, "105")["value"] == 1
    assert scorer.score({"task": "chartqa", "target": "0"}, "1")["value"] == 0
    assert (
        scorer.score({"task": "mmstar", "target": "B"}, "The answer is B.")["value"]
        == 1
    )
    assert scorer.score({"task": "mmstar", "target": "B"}, "A or B")["value"] == 0


def test_pinned_prompt_globals(scorers_root):
    assert "yes or no" not in data.prompt(
        scorers_root, "mmstar", {"question": "Q? Please answer yes or no."}
    )
    assert data.prompt(scorers_root, "textvqa", {"question": "WHAT?"}).startswith(
        "What?"
    )


def test_native_metric_formula_matches_our_ned_without_optional_dependencies(
    scorers_root, monkeypatch
):
    # A test-only distance double; deployed worker uses python-Levenshtein.
    def distance(a, b):
        row = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            nxt = [i]
            for j, y in enumerate(b, 1):
                nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
            row = nxt
        return row[-1]

    monkeypatch.setitem(
        sys.modules, "Levenshtein", types.SimpleNamespace(distance=distance)
    )
    assert Scorers(scorers_root).parity_check()["passed"]


def test_bootstrap_help_does_not_install():
    import subprocess

    path = Path(__file__).resolve().parents[1] / "bootstrap.py"
    result = subprocess.run(
        [sys.executable, str(path), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--mode" in result.stdout


def test_document_bootstrap_has_no_stop_ancestor():
    text = (Path(__file__).resolve().parents[1] / "notebook_template.py").read_text()
    assert "mo.stop(" not in text
    assert "1 · Pilot test" in text and "2 · Full scale" in text
    assert "Download ZIP" in text


def test_confidence_interval_unit():
    point, lo, hi = analysis.paired_interval([2.0] * 10, repetitions=50, geometric=True)
    assert (
        point == pytest.approx(2) and lo == pytest.approx(2) and hi == pytest.approx(2)
    )


class FakeServer:
    """Explicit simulated API used only by CPU integration tests."""

    def __init__(self, root, model, policy, k, identity, eager=False):
        self.root, self.policy, self.k, self.eager = root, policy, k, eager
        self.session = "CPU_TEST_ONLY"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def rpc(self, method):
        return [
            {
                "target_forward_gpu_ms": 2,
                "draft_forward_gpu_ms": 1,
                "state_only_gpu_ms": None,
            }
        ]

    def generate(self, case, prefix=None):
        return {
            "text": "B",
            "reasoning": "",
            "output_ids": [1, 2],
            "prompt_ids": [3, 4],
            "usage": {"completion_tokens": 2},
            "native_metrics": {"time_to_first_token_ms": 1, "generation_time_ms": 2},
            "speculative": {
                "num_spec_steps": 1,
                "num_accepted_draft_tokens": 1,
                "num_draft_tokens": self.k,
                "acceptance_histogram": [0, 1] + [0] * (self.k - 1),
            }
            if self.k
            else None,
            "request_latency_s": 0.01 if self.k == 1 else 0.02,
            "end_to_end_s": 0.02,
            "payload_preparation_s": 0,
            "client_ttft_s": 0.001,
            "finish_reason": "stop",
            "memory": [{"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}],
            "server_session": self.session,
            "eager": self.eager,
        }


def test_mocked_pilot_full_resume_end_to_end(scorers_root, monkeypatch):
    root = scorers_root
    monkeypatch.setattr(runner, "Server", FakeServer)
    monkeypatch.setattr(runner, "ensure_export", lambda root, base, policy: base)
    monkeypatch.setattr(
        Scorers,
        "score",
        lambda self, case, text: {
            "metric": "CPU_TEST_ONLY",
            "value": 1.0,
            "higher_is_better": True,
        },
    )
    prepared = {
        "development": [],
        "evaluation": [],
        "heldout": [],
        "input_hash": "CPU_TEST_ONLY",
    }
    for task in core.TASKS:
        for phase, count in [("development", 2), ("evaluation", 3), ("heldout", 1)]:
            for i in range(count):
                prepared[phase].append(
                    {
                        "id": f"{task}/{phase}/{i}",
                        "task": task,
                        "group": f"{task}/{phase}/{i}",
                        "stratum": "test",
                        "image_sha256": "test",
                        "prompt": "test",
                        "max_tokens": 4,
                        "timing_repeat": i == 0,
                    }
                )
    experiment = runner.Experiment(root, "CPU_TEST_ONLY", prepared, root)
    try:
        gate = experiment.pilot()
        assert gate["passed"] and gate["k"] == 1
        before = experiment.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        experiment.pilot()
        assert (
            experiment.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            == before
        )
        experiment.full()
        assert core.read_json(root / "status.json")["state"] == "complete"
        assert (root / "results/summary.csv").exists()
        count = experiment.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        experiment.full()
        assert (
            experiment.db.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            == count
        )
    finally:
        experiment.db.close()



def test_recovery_bundle_runs_after_database_and_exclusive_lock_release():
    import ast
    root = Path(__file__).resolve().parents[1]
    runner_text = (root / "qspec_hybrid/runner.py").read_text()
    assert "bundle(self.root)" not in runner_text
    tree = ast.parse((root / "bootstrap.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    locks = [node for node in ast.walk(main) if isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "exclusive" for item in node.items)]
    assert len(locks) == 1
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "bundle" for node in ast.walk(locks[0]))
    outer = next(node for node in main.body if isinstance(node, ast.Try) and node.finalbody)
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "bundle" for node in ast.walk(ast.Module(body=outer.finalbody, type_ignores=[])))


def test_smoke_options_do_not_change_pilot_command():
    normal = command("/model", 1234, 1)
    smoke = command("/model", 1234, 1, eager=True, smoke=True)
    assert "--kernel-config" not in normal
    assert "--enforce-eager" not in normal
    assert "--enforce-eager" in smoke
    assert json.loads(smoke[smoke.index("--kernel-config") + 1])["enable_flashinfer_autotune"] is False


def test_snapshot_connections_closed_before_archive_opens(tmp_path, monkeypatch):
    import contextlib
    with core.Store(tmp_path / "results") as store:
        store.put("saved", {"state": "ok"})
    real_connect, real_zip = core.sqlite3.connect, core.zipfile.ZipFile
    handles=[]
    class Connection:
        def __init__(self, *args, **kwargs):
            self.connection=real_connect(*args, **kwargs)
            self.closed=False
            handles.append(self)
        def backup(self, destination):
            return self.connection.backup(destination.connection)
        def commit(self): return self.connection.commit()
        def close(self):
            self.connection.close()
            self.closed=True
    def archive(*args, **kwargs):
        assert len(handles) == 2 and all(handle.closed for handle in handles)
        return real_zip(*args, **kwargs)
    monkeypatch.setattr(core.sqlite3, "connect", Connection)
    monkeypatch.setattr(core.zipfile, "ZipFile", archive)
    assert core.bundle(tmp_path).is_file()


def test_server_cleanup_reaps_orphan_engine_in_dedicated_process_group(tmp_path):
    import subprocess
    import time
    import psutil
    from qspec_hybrid.runtime import Server
    child_pid_path = tmp_path / "child.pid"
    script = ("import subprocess,sys,pathlib; "
              "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); "
              f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))")
    process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    process.wait(timeout=10)
    assert child_pid_path.exists()
    orphan = psutil.Process(int(child_pid_path.read_text()))
    server = Server.__new__(Server)
    server.proc = process
    server.log = (tmp_path / "server.log").open("wb")
    server.close()
    assert server.log.closed
    assert not orphan.is_running() or orphan.status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("cache_dtype,quantized", [("auto", False), ("bfloat16", False), ("fp8", True), ("fp8_e4m3", True), ("unknown", True)])
def test_kv_scale_adapter_is_not_confused_with_actual_quantization(cache_dtype, quantized):
    from qspec_hybrid.audit import quantization_active
    assert quantization_active("ModelOptKVCacheMethod", [], kv_cache_adapter=True, cache_dtype=cache_dtype) is quantized


def test_low_bit_storage_cannot_hide_behind_unquantized_method_name():
    from qspec_hybrid.audit import quantization_active
    assert quantization_active("UnquantizedLinearMethod", ["torch.float8_e4m3fn"])
    assert not quantization_active("UnquantizedLinearMethod", ["torch.bfloat16"])


def test_existing_cublas_files_are_exposed_without_overwriting_toolkit(tmp_path):
    from qspec_hybrid.runtime import configure_cublas_paths
    package=tmp_path / "installed"
    (package / "include").mkdir(parents=True)
    (package / "lib").mkdir()
    (package / "include/cublasLt.h").write_text("existing header")
    (package / "include/cuda_runtime.h").write_text("must not be overlaid")
    for name in ("libcublas.so.13", "libcublasLt.so.13"):
        (package / "lib" / name).write_bytes(b"existing library")
    original={"CUDA_HOME": "/pinned/toolkit", "LD_LIBRARY_PATH": "/existing/lib"}
    result=configure_cublas_paths(tmp_path / "project", original, package)
    assert original == {"CUDA_HOME": "/pinned/toolkit", "LD_LIBRARY_PATH": "/existing/lib"}
    assert result["CUDA_HOME"] == "/pinned/toolkit"
    assert (Path(result["CPATH"]) / "cublasLt.h").resolve() == (package / "include/cublasLt.h").resolve()
    assert not (Path(result["CPATH"]) / "cuda_runtime.h").exists()
    assert result["LD_LIBRARY_PATH"].endswith(":/existing/lib")
    assert configure_cublas_paths(tmp_path / "project", original, package) == result


def test_smoke_resume_is_bound_to_checkpoint_digest(tmp_path, monkeypatch):
    from qspec_hybrid import conversion, runtime
    calls=[]
    monkeypatch.setattr(data, "prepare", lambda root: {"development": [{"max_tokens": 2}]})
    class SmokeServer:
        def __init__(self, root, model, policy, k, identity, eager=False, smoke=False):
            self.policy,self.k=policy,k
            self.folder=root/"CPU_TEST_ONLY"
            calls.append((policy,k))
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def generate(self,case):
            return {"output_ids": [1,2], "native_metrics": {"CPU_TEST_ONLY": True},
                    "speculative": {"num_draft_tokens": 1} if self.k else None}
    monkeypatch.setattr(runtime,"Server",SmokeServer)
    for policy in ("P1","P2"):
        core.atomic_json(tmp_path/"models"/policy/"export_manifest.json", {"weights_content_sha256": policy+"_CPU_TEST_ONLY"})
    assert conversion.smoke_exports(tmp_path,["P1","P2"])["passed"]
    assert calls == [("P1",0),("P1",1),("P2",0)]
    calls.clear()
    assert conversion.smoke_exports(tmp_path,["P1","P2"])["passed"]
    assert not calls
    core.atomic_json(tmp_path/"models/P2/export_manifest.json", {"weights_content_sha256": "CHANGED_CPU_TEST_ONLY"})
    assert conversion.smoke_exports(tmp_path,["P1","P2"])["passed"]
    assert calls == [("P2",0)]


def test_conversion_worker_never_launches_pilot_when_saved_identity_rejects(tmp_path, monkeypatch):
    import contextlib
    from qspec_hybrid import conversion_worker
    locked={"value": False}
    @contextlib.contextmanager
    def exclusive(root):
        locked["value"]=True
        try: yield
        finally: locked["value"]=False
    monkeypatch.setattr(conversion_worker,"exclusive",exclusive)
    for policy in list(core.POLICIES)[1:]:
        core.atomic_json(tmp_path/"models"/policy/"export_manifest.json", {"weights_content_sha256": policy,"file_sizes":{}})
        core.atomic_json(tmp_path/"results/conversion_preflight"/f"{policy}.json", {"status":"pass","tensor_contract":{"weights_content_sha256":policy}})
    def smoke(root,policies):
        return {"passed": True,"results":[{"policy":p,"k":0,"passed":True,"server_folder":"CPU_TEST_ONLY"} for p in policies]}
    monkeypatch.setattr(conversion_worker,"smoke_exports",smoke)
    def reject(root): raise RuntimeError("CPU_TEST_ONLY identity mismatch")
    monkeypatch.setattr(runner,"environment",reject)
    def bundle(root):
        assert not locked["value"]
        return root/"recovery.zip"
    monkeypatch.setattr(conversion_worker,"bundle",bundle)
    def forbidden(*args): raise AssertionError("pilot must not launch")
    monkeypatch.setattr(conversion_worker,"launch",forbidden)
    assert conversion_worker.run(tmp_path,smoke_only=True)==1
    status=core.read_json(tmp_path/"results/conversion_preflight/status.json")
    assert status["conversion_gates_passed"]
    assert not status["pilot_resumed"]
    assert status["stage"]=="pilot resume compatibility"
    assert status["failure"]["message"]=="CPU_TEST_ONLY identity mismatch"
