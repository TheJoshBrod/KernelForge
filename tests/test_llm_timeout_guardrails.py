from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm_tools
from src.llm_tools import GenModel
from src.optimizer.core import generator as opt_generator


def test_gen_model_chat_retries_retryable_timeout_then_succeeds(monkeypatch):
    calls: list[int] = []

    def fake_request(payload, provider_label):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return False, "Error calling Claude API: request timed out after 120 seconds"
        return True, "ok"

    monkeypatch.setattr(llm_tools, "_request_with_watchdog", fake_request)
    monkeypatch.setattr(llm_tools.settings, "llm_retry_limit", 1)
    monkeypatch.setattr(llm_tools.settings, "llm_retry_backoff_seconds", 0.0)

    model = GenModel("sys")
    response = model.chat("hello", "claude-sonnet-4-6")

    assert response == "ok"
    assert len(calls) == 2
    assert model.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ok"},
    ]


def test_gen_model_chat_does_not_retry_non_retryable_provider_error(monkeypatch):
    calls: list[int] = []

    def fake_request(payload, provider_label):
        calls.append(len(calls) + 1)
        return False, "Error calling Claude API: invalid api key"

    monkeypatch.setattr(llm_tools, "_request_with_watchdog", fake_request)
    monkeypatch.setattr(llm_tools.settings, "llm_retry_limit", 3)
    monkeypatch.setattr(llm_tools.settings, "llm_retry_backoff_seconds", 0.0)

    model = GenModel("sys")
    response = model.chat("hello", "claude-sonnet-4-6")

    assert response == "Error calling Claude API: invalid api key"
    assert len(calls) == 1
    assert model.history[-1] == {
        "role": "assistant",
        "content": "Error calling Claude API: invalid api key",
    }


def test_create_and_validate_reports_waiting_on_llm(monkeypatch, tmp_path: Path):
    status_updates: list[tuple[str, int]] = []

    class DummyBackend:
        kernel_extension = ".cu"

        def validate_kernel(self, code, paths, ssh_config=None):
            return True, ""

    llm = GenModel("sys")
    monkeypatch.setattr(
        llm,
        "chat",
        lambda msg, model: "// [START kernel.cu]\nextern \"C\" __global__ void k() {}\n// [END kernel.cu]",
    )

    feedback, is_valid, error = opt_generator.create_and_validate(
        DummyBackend(),
        llm,
        "prompt",
        "claude-sonnet-4-6",
        {
            "tmp_dir": tmp_path,
            "proj_dir": tmp_path,
            "iteration": 0,
            "attempt": 0,
        },
        status_callback=lambda step, attempt: status_updates.append((step, attempt)),
    )

    assert feedback is not None
    assert is_valid is True
    assert error == ""
    assert status_updates[0] == ("Waiting on LLM", 1)
    assert status_updates[-1] == ("Validating", 1)
