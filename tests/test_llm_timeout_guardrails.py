from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm_tools
from src.llm_tools import GenModel
from src.optimizer.core import generator as opt_generator


def test_anthropic_request_disables_sdk_retries(monkeypatch):
    captured: dict[str, object] = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text="// [START kernel.cu]\ncode\n// [END kernel.cu]")]
            )

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setattr(llm_tools.anthropic, "Anthropic", FakeAnthropic)

    response = llm_tools._anthropic_request(
        {
            "request_timeout": 12.0,
            "model": "claude-sonnet-4-6",
            "sys_prompt": "sys",
            "history": [{"role": "user", "content": "hello"}],
        }
    )

    assert response.startswith("// [START kernel.cu]")
    assert captured["client"]["timeout"] == 12.0
    assert captured["client"]["max_retries"] == 0


def test_gen_model_chat_retries_retryable_timeout_then_succeeds(monkeypatch):
    calls: list[int] = []

    def fake_request(payload, provider_label, **kwargs):
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

    def fake_request(payload, provider_label, **kwargs):
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


def test_gen_model_chat_enforces_overall_deadline(monkeypatch):
    calls: list[float] = []
    statuses: list[str] = []
    clock = {"now": 0.0}

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(seconds):
        clock["now"] += float(seconds)

    def fake_request(payload, provider_label, *, watchdog_timeout=None, status_callback=None):
        calls.append(float(watchdog_timeout))
        clock["now"] += float(watchdog_timeout)
        return False, "Error calling Claude API: request timed out after 120 seconds"

    monkeypatch.setattr(llm_tools.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(llm_tools.time, "sleep", fake_sleep)
    monkeypatch.setattr(llm_tools, "_request_with_watchdog", fake_request)
    monkeypatch.setattr(llm_tools.settings, "llm_watchdog_timeout_seconds", 120)
    monkeypatch.setattr(llm_tools.settings, "llm_total_timeout_seconds", 150)
    monkeypatch.setattr(llm_tools.settings, "llm_retry_limit", 2)
    monkeypatch.setattr(llm_tools.settings, "llm_retry_backoff_seconds", 2.0)

    model = GenModel("sys")
    response = model.chat("hello", "claude-sonnet-4-6", status_callback=statuses.append)

    assert response == "Error calling Claude API: overall request deadline exceeded after 150 seconds"
    assert calls == [120.0, 28.0]
    assert any("Claude attempt 1/3" in status for status in statuses)
    assert any("Claude retry backoff 1/2: 2.0s" in status for status in statuses)


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
        lambda msg, model, status_callback=None: "// [START kernel.cu]\nextern \"C\" __global__ void k() {}\n// [END kernel.cu]",
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
