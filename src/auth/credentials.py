from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from .codex_capabilities import detect_account_session
from .keychain_store import env_or_keychain_openai_key

AUTH_MODES = {"auto", "api_key", "account_session"}
PROVIDERS = {"openai", "anthropic", "gemini"}


@dataclass(slots=True)
class AuthStatus:
    mode_requested: str
    mode_effective: str
    provider: str
    model: str
    api_key_available: bool
    account_session_detected: bool
    reason: str
    in_container: bool
    checked_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_provider(value: str) -> str:
    provider = (value or "").strip().lower()
    if provider in {"gpt", "chatgpt"}:
        return "openai"
    if provider in PROVIDERS:
        return provider
    return ""


def _default_model(provider: str) -> str:
    if provider == "openai":
        return "gpt-5.2"
    if provider == "anthropic":
        return "claude-opus-4-5-20251101"
    if provider == "gemini":
        return "gemini-2.5-flash"
    return ""


def _provider_model_env(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_MODEL"
    if provider == "anthropic":
        return "ANTHROPIC_MODEL"
    if provider == "gemini":
        return "GEMINI_MODEL"
    return ""


def _provider_key_env(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "gemini":
        return "GEMINI_API_KEY"
    return ""


def _read_cfg_section(config: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    value = config.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _resolve_provider_and_model(config: dict[str, Any] | None, env: dict[str, str]) -> tuple[str, str]:
    auth_cfg = _read_cfg_section(config, "auth")
    llm_cfg = _read_cfg_section(config, "llm_info")

    provider = _normalize_provider(env.get("LLM_PROVIDER", ""))
    if not provider:
        provider = _normalize_provider(str(auth_cfg.get("provider", "")))
    if not provider:
        provider = _normalize_provider(str(llm_cfg.get("provider", "")))
    if not provider:
        if env.get("OPENAI_API_KEY"):
            provider = "openai"
        elif env.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
            provider = "gemini"

    model = ""
    model_env = _provider_model_env(provider) if provider else ""
    if model_env:
        model = (env.get(model_env) or "").strip()
    if not model:
        model = str(auth_cfg.get("model", "")).strip()
    if not model:
        model = str(llm_cfg.get("model", "")).strip()
    if not model and provider:
        model = _default_model(provider)

    return provider, model


def _resolve_mode_requested(config: dict[str, Any] | None, env: dict[str, str]) -> str:
    raw = (env.get("CGINS_AUTH_MODE") or "").strip().lower()
    if raw in AUTH_MODES:
        return raw
    auth_cfg = _read_cfg_section(config, "auth")
    raw = str(auth_cfg.get("mode", "")).strip().lower()
    if raw in AUTH_MODES:
        return raw
    return "auto"


def _legacy_key_from_config(config: dict[str, Any] | None) -> str:
    llm_cfg = _read_cfg_section(config, "llm_info")
    return str(llm_cfg.get("apikey", "")).strip()


def _provider_key_available(provider: str, config: dict[str, Any] | None, env: dict[str, str]) -> bool:
    if provider == "openai":
        if (env.get("OPENAI_API_KEY") or "").strip():
            return True
        if (env.get("CODEX_API_KEY") or "").strip():
            return True
        if env_or_keychain_openai_key():
            return True
        if _legacy_key_from_config(config):
            return True
        return False

    key_env = _provider_key_env(provider)
    if key_env and (env.get(key_env) or "").strip():
        return True
    if provider == "gemini" and (env.get("GOOGLE_API_KEY") or "").strip():
        return True
    if provider in {"anthropic", "gemini"} and _legacy_key_from_config(config):
        return True
    return False


def resolve_auth(
    config: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> AuthStatus:
    env_map = env if env is not None else dict(os.environ)
    ctx = runtime_context or {}
    in_container = bool(ctx.get("in_container")) or bool(env_map.get("CGINS_PROJECT_DIR"))

    provider, model = _resolve_provider_and_model(config, env_map)
    mode_requested = _resolve_mode_requested(config, env_map)

    if not provider:
        return AuthStatus(
            mode_requested=mode_requested,
            mode_effective="unconfigured",
            provider="",
            model="",
            api_key_available=False,
            account_session_detected=False,
            reason="provider_not_configured",
            in_container=in_container,
            checked_at=time.time(),
        )

    api_key_available = _provider_key_available(provider, config, env_map)
    account_session_detected = False
    if provider == "openai":
        account_session_detected, _ = detect_account_session(
            model=model or None,
            perform_live_probe=not in_container,
        )

    mode_effective = "unconfigured"
    reason = ""

    if mode_requested == "api_key":
        if api_key_available:
            mode_effective = "api_key"
            reason = "api_key_mode_with_key_available"
        else:
            reason = "api_key_mode_without_key"
    elif mode_requested == "account_session":
        if account_session_detected and not in_container:
            mode_effective = "account_session"
            reason = "account_session_detected"
        elif api_key_available:
            mode_effective = "api_key"
            reason = "account_session_fallback_to_api_key"
        else:
            reason = "account_session_unavailable_and_no_key_fallback"
    else:
        if account_session_detected and not in_container:
            mode_effective = "account_session"
            reason = "auto_prefers_account_session"
        elif api_key_available:
            mode_effective = "api_key"
            reason = "auto_fallback_to_api_key"
        else:
            reason = "auto_no_usable_auth"

    return AuthStatus(
        mode_requested=mode_requested,
        mode_effective=mode_effective,
        provider=provider,
        model=model,
        api_key_available=api_key_available,
        account_session_detected=account_session_detected,
        reason=reason,
        in_container=in_container,
        checked_at=time.time(),
    )


def apply_auth_env(status: AuthStatus, env: dict[str, str] | None = None) -> dict[str, str]:
    env_out = env if env is not None else os.environ

    if status.provider:
        env_out["LLM_PROVIDER"] = status.provider
    if status.model:
        model_env = _provider_model_env(status.provider)
        if model_env:
            env_out[model_env] = status.model

    env_out["CGINS_AUTH_MODE"] = status.mode_requested
    env_out["CGINS_AUTH_EFFECTIVE"] = status.mode_effective
    env_out["CGINS_AUTH_REASON"] = status.reason

    if status.mode_effective == "api_key":
        if status.provider == "openai":
            key = (
                (env_out.get("OPENAI_API_KEY") or "").strip()
                or (env_out.get("CODEX_API_KEY") or "").strip()
                or env_or_keychain_openai_key()
            )
            if key:
                env_out["OPENAI_API_KEY"] = key
                env_out["CODEX_API_KEY"] = key
        elif status.provider == "anthropic":
            # keep existing env key behavior
            pass
        elif status.provider == "gemini":
            if env_out.get("GOOGLE_API_KEY") and not env_out.get("GEMINI_API_KEY"):
                env_out["GEMINI_API_KEY"] = env_out["GOOGLE_API_KEY"]

    return env_out


def resolve_auth_for_current_process(
    config: dict[str, Any] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> AuthStatus:
    status = resolve_auth(config=config, env=dict(os.environ), runtime_context=runtime_context)
    apply_auth_env(status, os.environ)
    return status
