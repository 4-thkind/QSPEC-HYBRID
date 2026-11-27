"""Pinned API contract tests; all responses below are explicit test fixtures."""

import io
import json
import urllib.request

import pytest

from qspec_hybrid.runtime import Server


def response(k=2, ids=True, opportunities=True, bad_counts=False, timing=True):
    spec = {
        "num_spec_steps": int(opportunities),
        "num_accepted_draft_tokens": int(opportunities),
        "num_draft_tokens": k if opportunities else 0,
        "acceptance_histogram": ([0, 1] + [0] * (k - 1))
        if opportunities
        else [0] * (k + 1),
    }
    if bad_counts:
        spec["num_accepted_draft_tokens"] = 99
    chunks = [
        {
            "id": "test",
            "prompt_token_ids": [10, 11],
            "choices": [{"delta": {"role": "assistant"}, "token_ids": []}],
        },
        {
            "choices": [
                {"delta": {"content": "B"}, **({"token_ids": [7]} if ids else {})}
            ]
        },
        {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    **({"token_ids": [2]} if ids else {}),
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            "metrics": {
                "time_to_first_token_ms": 1.0 if timing else None,
                "generation_time_ms": 2.0,
                "speculative_decoding": spec if k else None,
            },
        },
    ]
    return io.BytesIO(
        (
            "\n\n".join("data: " + json.dumps(x) for x in chunks)
            + "\n\ndata: [DONE]\n\n"
        ).encode()
    )


@pytest.fixture
def server(tmp_path):
    server = Server.__new__(Server)
    server.root, server.policy, server.k, server.eager = tmp_path, "P0", 2, False
    server.url, server.token, server.session = (
        "http://127.0.0.1:1",
        "test-only",
        "CPU_TEST_ONLY",
    )
    server.rpc = lambda method: [
        {"peak_allocated_bytes": 12, "peak_reserved_bytes": 16}
    ]
    (tmp_path / "image.png").write_bytes(b"test fixture, not an image")
    return server


def test_stream_keeps_original_token_ids_and_native_counts(server, monkeypatch):
    requests = []

    def open_request(req, **kwargs):
        requests.append(json.loads(req.data))
        return response()

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    case = {
        "image": "image.png",
        "prompt": "Question",
        "max_tokens": 16,
        "task": "mmstar",
    }
    result = server.generate(case)
    assert result["output_ids"] == [7, 2]
    assert result["prompt_ids"] == [10, 11]
    assert result["text"] == "B"
    assert result["speculative"]["num_accepted_draft_tokens"] == 1
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize(
    "variant,message",
    [
        ({"ids": False}, "token IDs"),
        ({"bad_counts": True}, "reconcile"),
        ({"timing": False}, "timing"),
    ],
)
def test_missing_or_invalid_measurements_fail_closed(
    server, monkeypatch, variant, message
):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: response(**variant))
    with pytest.raises(RuntimeError, match=message):
        server.generate(
            {"image": "image.png", "prompt": "Q", "max_tokens": 16, "task": "mmstar"}
        )


def test_no_draft_opportunities_is_valid_not_fake_zero_acceptance(server, monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **kw: response(opportunities=False)
    )
    result = server.generate(
        {"image": "image.png", "prompt": "Q", "max_tokens": 16, "task": "mmstar"}
    )
    assert result["speculative"]["num_spec_steps"] == 0


def test_continuation_uses_explicit_controlled_prefix(server, monkeypatch):
    sent = []

    def open_request(req, **kwargs):
        sent.append(json.loads(req.data))
        return response()

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    server.generate(
        {"image": "image.png", "prompt": "Q", "max_tokens": 16, "task": "mmstar"},
        prefix="The answer",
    )
    assert sent[0]["continue_final_message"] is True
    assert sent[0]["add_generation_prompt"] is False
    assert sent[0]["messages"][-1] == {"role": "assistant", "content": "The answer"}


def test_portable_lock_has_no_machine_specific_modelopt_path(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("bootstrap_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "tooling/modelopt"
    text = f"torch==2.13.0\nnvidia-modelopt @ {repo.as_uri()}\n"
    lock = module.canonical_lock(text, repo, "frozen-commit")
    assert "git+https://github.com/NVIDIA/Model-Optimizer.git@frozen-commit" in lock
    assert str(tmp_path) not in lock


def test_duplicate_input_filter_is_before_generation():
    from qspec_hybrid.data import retain_unique

    cases, seen, excluded = [], set(), []
    retain_unique(cases, {"id": "a", "image_sha256": "same"}, seen, excluded)
    retain_unique(cases, {"id": "b", "image_sha256": "same"}, seen, excluded)
    assert len(cases) == 1 and excluded[0]["id"] == "b"
