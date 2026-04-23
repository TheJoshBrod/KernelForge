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
