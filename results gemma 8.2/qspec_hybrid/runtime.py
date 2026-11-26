from __future__ import annotations

import base64
import json
import math
import os
import socket
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from .audit import validate_runtime
from .core import PROTOCOL, atomic_json, digest
from .monitor import Monitor


def configure_cublas_paths(root, env, package=None):
    """Expose existing cuBLAS development files to the pinned JIT compiler.

    Only cuBLAS files are linked, keeping the pinned toolkit's CUDA headers and
    other libraries authoritative. No dependency is installed or replaced.
    """
    from pathlib import Path
    import sysconfig

    package = Path(package) if package is not None else Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    if not (package / "include/cublasLt.h").is_file():
        return env
    overlay = Path(root) / ".cache" / "cublas-devel"
    include, library = overlay / "include", overlay / "lib"
    include.mkdir(parents=True, exist_ok=True)
    library.mkdir(parents=True, exist_ok=True)
    links = [(include / path.name, path) for path in sorted((package / "include").glob("cublas*.h"))]
    for name in ("libcublas", "libcublasLt"):
        target = package / "lib" / (name + ".so.13")
        if not target.is_file():
            raise RuntimeError(f"Installed cuBLAS library missing: {target}")
        links += [(library / (name + ".so"), target), (library / target.name, target)]
    for destination, target in links:
        if destination.is_symlink() or destination.exists():
            if destination.resolve() != target.resolve():
                raise RuntimeError(f"cuBLAS overlay has unexpected content; preserved: {destination}")
        else:
            destination.symlink_to(target.resolve())
    result = dict(env)
    for variable, folder in (("CPATH", include), ("CPLUS_INCLUDE_PATH", include),
                             ("LIBRARY_PATH", library), ("LD_LIBRARY_PATH", library)):
        previous = result.get(variable, "")
        result[variable] = str(folder) + (os.pathsep + previous if previous else "")
    return result


def command(model, port, k, eager=False, smoke=False):
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "qspec",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--kv-cache-memory-bytes",
        str(PROTOCOL["kv_cache_bytes"]),
        "--max-model-len",
        str(PROTOCOL["context"]),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--cpu-offload-gb",
        "0",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--enable-per-request-metrics",
        "--worker-extension-cls",
        "qspec_hybrid.audit.PrecisionAudit",
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
        "--mm-processor-cache-gb",
        "0",
        "--mm-processor-kwargs",
        json.dumps(
            {"max_pixels": PROTOCOL["max_pixels"], "min_pixels": PROTOCOL["min_pixels"]}
        ),
        "--no-enable-log-requests",
        "--seed",
        str(PROTOCOL["seed"]),
    ]
    if k:
        args += [
            "--per-request-spec-decode-metrics",
            "detailed",
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": k}),
        ]
    if eager:
        args.append("--enforce-eager")
    if smoke:
        args += ["--kernel-config", json.dumps({"enable_flashinfer_autotune": False})]
    return args


class Server:
    def __init__(self, root, model, policy, k, identity, eager=False, smoke=False):
        self.root, self.policy, self.k, self.eager = root, policy, k, eager
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.token = uuid.uuid4().hex
        self.session = uuid.uuid4().hex[:12]
        self.folder = root / "results" / "servers" / self.session
        self.folder.mkdir(parents=True)
        env = dict(os.environ)
        env.update(
            VLLM_SERVER_DEV_MODE="1",
            VLLM_API_KEY=self.token,
            VLLM_CACHE_ROOT=str(
                root / ".cache" / "vllm" / digest([identity, policy, k, eager])[:24]
            ),
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
        )
        env = configure_cublas_paths(root, env)
        self.log = (self.folder / "server.log").open("wb")
        args = command(model, port, k, eager, smoke=smoke)
        atomic_json(
            self.folder / "launch.json",
            {
                "args": args,
                "cache_root": env["VLLM_CACHE_ROOT"],
                "policy": policy,
                "k": k,
                "eager": eager,
                "smoke": smoke,
            },
        )
        self.proc = subprocess.Popen(
            args, env=env, cwd=root, stdout=self.log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def rpc(self, method):
        return self.json("/collective_rpc", {"method": method, "timeout": 60})[
            "results"
        ]

    def json(self, path, body=None, timeout=90):
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read(4000).decode(errors="replace")
            raise RuntimeError(
                f"Local inference API {path}: HTTP {error.code}: {message}"
            ) from error

    def __enter__(self):
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"Server exited ({self.proc.returncode}); inspect {self.folder / 'server.log'}"
                    )
                try:
                    info = self.json("/v1/models", timeout=2)
                    if any(x["id"] == "qspec" for x in info["data"]):
                        break
                except (OSError, RuntimeError, ValueError):
                    time.sleep(2)
                if (self.root / "STOP").exists():
                    raise KeyboardInterrupt
            else:
                raise TimeoutError(
                    f"Server startup exceeded 30 minutes; see {self.folder}"
                )
            audits = self.rpc("qspec_audit")
            self.worker_pids = {a["pid"] for a in audits}
            atomic_json(self.folder / "runtime_precision.json", audits)
            validate_runtime(audits, self.policy, self.k)
            self.rpc("qspec_reset_memory")
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        # The API server and EngineCore share this dedicated process group.
        # EngineCore can outlive a terminated API server, so clean the group
        # even when the leader has already exited.
        try:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    pass
            import psutil
            survivors = []
            for process in psutil.process_iter(["pid", "status"]):
                try:
                    if (process.info["status"] != psutil.STATUS_ZOMBIE
                            and os.getpgid(process.pid) == self.proc.pid):
                        survivors.append(process)
                except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                    continue
            if survivors:
                _, survivors = psutil.wait_procs(survivors, timeout=5)
                if survivors:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    psutil.wait_procs(survivors, timeout=5)
            if self.proc.poll() is None:
                self.proc.wait(timeout=10)
        finally:
            self.log.close()

    def __exit__(self, *_):
        self.close()

    def generate(self, case, prefix=None, extra=None):
        self.rpc("qspec_reset_memory")
        prep = time.perf_counter()
        image = base64.b64encode((self.root / case["image"]).read_bytes()).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + image},
                    },
                    {"type": "text", "text": case["prompt"]},
                ],
            }
        ]
        body = {
            "model": "qspec",
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "max_tokens": case["max_tokens"],
            "seed": PROTOCOL["seed"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
            "include_metrics": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if prefix is not None:
            body["messages"].append({"role": "assistant", "content": prefix})
            body.update(continue_final_message=True, add_generation_prompt=False)
        if case["task"] == "textvqa":
            body["stop"] = ["ASSISTANT:"]
        body.update(extra or {})
        payload = json.dumps(body).encode()
        preparation_s = time.perf_counter() - prep
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            },
        )
        monitor = Monitor(getattr(self, "worker_pids", set())).start()
        started = time.perf_counter()
        first, text, reasoning, token_ids, prompt_ids = None, [], [], [], None
        usage, metrics, finish_reason, request_id = None, None, None, None
        logprobs = []
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    request_id = chunk.get("id", request_id)
                    if chunk.get("prompt_token_ids") is not None:
                        prompt_ids = chunk["prompt_token_ids"]
                    usage = chunk.get("usage") or usage
                    metrics = chunk.get("metrics") or metrics
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        ids = choice.get("token_ids")
                        if ids is not None:
                            token_ids.extend(ids)
                        if first is None and (
                            ids
                            or delta.get("content")
                            or delta.get("reasoning_content")
                            or delta.get("reasoning")
                        ):
                            first = time.perf_counter() - started
                        text.append(delta.get("content") or "")
                        reasoning.append(
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if (choice.get("logprobs") or {}).get("content"):
                            logprobs.extend(choice["logprobs"]["content"])
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read(4000).decode(errors="replace")) from error
        finally:
            elapsed = time.perf_counter() - started
            device_observations = monitor.stop()
        memory = self.rpc("qspec_memory")
        if usage is None or metrics is None or not token_ids:
            raise RuntimeError(
                "Missing native usage/timing/token IDs; do not substitute retokenized output"
            )
        ttft = metrics.get("time_to_first_token_ms")
        if (
            ttft is None
            or not math.isfinite(ttft)
            or ttft < 0
            or usage.get("completion_tokens", 0) <= 0
        ):
            raise RuntimeError("Invalid or missing native request timing/token counts")
        speculative = metrics.get("speculative_decoding")
        if self.k and speculative is None:
            raise RuntimeError("Native per-request speculative counters are absent")
        if speculative:
            steps = speculative["num_spec_steps"]
            accepted = speculative["num_accepted_draft_tokens"]
            proposed = speculative["num_draft_tokens"]
            hist = speculative["acceptance_histogram"]
            if len(hist) != self.k + 1 or any(
                not isinstance(x, int) or x < 0 for x in hist
            ):
                raise RuntimeError("Invalid native acceptance histogram")
            if (
                sum(hist) != steps
                or sum(i * n for i, n in enumerate(hist)) != accepted
                or accepted > proposed
            ):
                raise RuntimeError("Native acceptance counters do not reconcile")
            # Zero proposals on a naturally tiny output is valid and remains saved.
        return {
            "request_id": request_id,
            "text": "".join(text),
            "reasoning": "".join(reasoning),
            "output_ids": token_ids,
            "prompt_ids": prompt_ids,
            "finish_reason": finish_reason,
            "usage": usage,
            "native_metrics": metrics,
            "speculative": speculative,
            "request_latency_s": elapsed,
            "payload_preparation_s": preparation_s,
            "end_to_end_s": preparation_s + elapsed,
            "client_ttft_s": first,
            "output_tokens_per_s": usage["completion_tokens"] / elapsed,
            "memory": memory,
            "device_observations": device_observations,
            "server_session": self.session,
            "eager": self.eager,
            "logprobs": logprobs or None,
        }
