from __future__ import annotations

import types

import src.llm_tools as llm_tools
from src.llm_tools import GenModel


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = types.SimpleNamespace(content="// kernel")
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=3),
        )
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeOpenAI:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.chat = types.SimpleNamespace(
            completions=completions,
        )


class _FakeAnthropicMessages:
    def create(self, **kwargs):
        usage = types.SimpleNamespace(input_tokens=13, output_tokens=8)
        content = [types.SimpleNamespace(text="// claude kernel")]
        return types.SimpleNamespace(content=content, usage=usage)


class _FakeAnthropic:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


class _FakeUsageLogger:
    def __init__(self) -> None:
        self.calls = []

    def log(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_openai_generation_defaults_to_medium_reasoning_effort(monkeypatch) -> None:
    completions = _FakeChatCompletions()
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("KFORGE_OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(llm_tools, "OpenAI", lambda **_: _FakeOpenAI(completions))

    response = GenModel("system").chat("build a kernel", "gpt-5")

    assert response == "// kernel"
    assert completions.kwargs["reasoning_effort"] == "medium"


def test_openai_generation_allows_reasoning_effort_override(monkeypatch) -> None:
    completions = _FakeChatCompletions()
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setattr(llm_tools, "OpenAI", lambda **_: _FakeOpenAI(completions))

    GenModel("system").chat("build a kernel", "gpt-5")

    assert completions.kwargs["reasoning_effort"] == "low"


def test_anthropic_generation_sets_last_usage_and_logs_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_tools.anthropic,
        "Anthropic",
        lambda **_: _FakeAnthropic(),
    )
    logger = _FakeUsageLogger()
    model = GenModel("system")
    model.set_usage_logger(logger)
    model.set_usage_context(step_type="generation", iteration=0, attempt=1)

    response = model.chat("build a kernel", "claude-opus-4-7")

    assert response == "// claude kernel"
    assert model.last_usage == {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "input_tokens": 13,
        "output_tokens": 8,
        "reasoning_tokens": 0,
    }
    assert logger.calls == [
        {
            "step_type": "generation",
            "iteration": 0,
            "attempt": 1,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "input_tokens": 13,
            "output_tokens": 8,
            "reasoning_tokens": 0,
        }
    ]
