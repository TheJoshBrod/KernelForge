from __future__ import annotations

import json
import os

from src.config import apply_llm_config


def test_apply_llm_config_uses_selected_anthropic_provider(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "api_keys": {
                    "openai": "openai-key",
                    "anthropic": "anthropic-key",
                    "google": "",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KFORGE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPTIMIZER_LLM_MODEL_NAME", raising=False)

    assert apply_llm_config()

    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-opus-4-7"
    assert os.environ["OPTIMIZER_LLM_MODEL_NAME"] == (
        "anthropic/claude-opus-4-7"
    )


def test_apply_llm_config_overwrites_stale_selected_provider_key(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "api_keys": {
                    "openai": "openai-key",
                    "anthropic": "saved-anthropic-key",
                    "google": "",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KFORGE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-key")

    assert apply_llm_config()

    assert os.environ["ANTHROPIC_API_KEY"] == "saved-anthropic-key"
