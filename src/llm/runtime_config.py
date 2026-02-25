"""Resolve runtime LLM environment from global and project config."""

from __future__ import annotations

from typing import Any

from .key_store import DEFAULT_API_KEYS, normalize_config
from .models import DEFAULT_PROJECT_MODELS, normalize_provider

_PROVIDER_ORDER = ("openai", "anthropic", "google")


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _infer_provider_from_model(model: str) -> str:
    value = (model or "").strip().lower()
    if not value:
        return ""
    if "claude" in value:
        return "anthropic"
    if "gemini" in value:
        return "google"
    if "gpt" in value or value.startswith(("o1", "o3", "o4")):
        return "openai"
    return ""


def _first_provider_with_key(api_keys: dict[str, str]) -> str:
    for provider in _PROVIDER_ORDER:
        if _as_str(api_keys.get(provider, "")):
            return provider
    return ""


def resolve_runtime_env(
    global_config: dict[str, Any] | None = None,
    project_config: dict[str, Any] | None = None,
    override_provider: str | None = None,
    override_model: str | None = None,
) -> dict[str, str]:
    normalized_global = normalize_config(global_config or {})
    api_keys = dict(DEFAULT_API_KEYS)
    raw_keys = normalized_global.get("api_keys")
    if isinstance(raw_keys, dict):
        for provider in api_keys:
            api_keys[provider] = _as_str(raw_keys.get(provider, ""))

    llm_cfg: dict[str, Any] = {}
    if isinstance(project_config, dict):
        maybe_llm = project_config.get("llm")
        if isinstance(maybe_llm, dict):
            llm_cfg = maybe_llm

    model = _as_str(override_model) or _as_str(llm_cfg.get("model", ""))
    provider = normalize_provider(_as_str(override_provider) or _as_str(llm_cfg.get("provider", "")))

    if not provider:
        provider = _infer_provider_from_model(model)
    if not provider:
        provider = _first_provider_with_key(api_keys)
    if provider not in DEFAULT_PROJECT_MODELS:
        return {}

    if not model:
        model = DEFAULT_PROJECT_MODELS[provider]
    api_key = _as_str(api_keys.get(provider, ""))

    env: dict[str, str] = {
        "LLM_PROVIDER": provider,
        "KFORGE_LLM_PROVIDER": provider,
        "KFORGE_LLM_MODEL": model,
    }

    if provider == "openai":
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env["OPENAI_MODEL"] = model
        # LiteLLM recognises OpenAI models by name; no prefix needed.
        litellm_model = model
    elif provider == "anthropic":
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        env["ANTHROPIC_MODEL"] = model
        # LiteLLM requires lowercase "anthropic/<model>" for newer models.
        litellm_model = model if model.startswith("anthropic/") else f"anthropic/{model}"
    elif provider == "google":
        if api_key:
            env["GOOGLE_API_KEY"] = api_key
            env["GEMINI_API_KEY"] = api_key
        env["GEMINI_MODEL"] = model
        litellm_model = model if model.startswith("gemini/") else f"gemini/{model}"
    else:
        litellm_model = model

    # OPTIMIZER_LLM_MODEL_NAME is read by byllm via pydantic-settings at subprocess
    # import time.  Supply the LiteLLM-formatted name so the provider is unambiguous.
    env["OPTIMIZER_LLM_MODEL_NAME"] = litellm_model

    return env

