from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.auth.credentials import apply_auth_env, resolve_auth
from src.auth.keychain_store import (
    DEFAULT_OPENAI_ACCOUNT,
    DEFAULT_OPENAI_SERVICE,
    save_openai_api_key,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "llm_info": {"model": "", "apikey": "", "provider": ""},
    "auth": {
        "mode": "auto",
        "provider": "openai",
        "model": "",
        "api_key_saved": False,
        "account_session_detected": False,
        "last_checked_at": 0,
    },
    "providers": [],
}


def _find_config_path() -> Path | None:
    override = os.environ.get("CGINS_CONFIG_PATH")
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "frontend" / "config.json",
        repo_root / "config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def config_path_or_default() -> Path:
    existing = _find_config_path()
    if existing:
        return existing
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "frontend" / "config.json"


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULT_CONFIG)
    llm = dict(DEFAULT_CONFIG["llm_info"])
    llm.update(data.get("llm_info", {}) if isinstance(data.get("llm_info"), dict) else {})
    out["llm_info"] = llm

    auth = dict(DEFAULT_CONFIG["auth"])
    auth.update(data.get("auth", {}) if isinstance(data.get("auth"), dict) else {})
    out["auth"] = auth

    providers = data.get("providers")
    out["providers"] = providers if isinstance(providers, list) else list(DEFAULT_CONFIG["providers"])

    for key, value in data.items():
        if key not in out:
            out[key] = value
    return out


def load_config_data() -> tuple[dict[str, Any], Path]:
    path = config_path_or_default()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return _merge_defaults(data), path
        except Exception:
            pass
    return dict(DEFAULT_CONFIG), path


def save_config_data(cfg: dict[str, Any], *, strip_plaintext_apikey: bool = True) -> tuple[bool, str, Path]:
    path = config_path_or_default()
    data = _merge_defaults(cfg if isinstance(cfg, dict) else {})

    llm_info = data.get("llm_info", {})
    legacy_key = str(llm_info.get("apikey", "")).strip() if isinstance(llm_info, dict) else ""
    auth_cfg = data.get("auth", {})
    provider = str(auth_cfg.get("provider", "")).strip().lower() if isinstance(auth_cfg, dict) else ""
    if not provider and isinstance(llm_info, dict):
        provider = str(llm_info.get("provider", "")).strip().lower()

    # Migration path: if old plaintext key is present, store it in keychain.
    if legacy_key and provider == "openai":
        stored = save_openai_api_key(
            legacy_key,
            service=DEFAULT_OPENAI_SERVICE,
            account=DEFAULT_OPENAI_ACCOUNT,
        )
        if stored.ok and isinstance(auth_cfg, dict):
            auth_cfg["api_key_saved"] = True
            data["auth"] = auth_cfg

    if strip_plaintext_apikey and isinstance(llm_info, dict):
        llm_info["apikey"] = ""
        data["llm_info"] = llm_info

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        return False, str(exc), path
    return True, "saved", path


def apply_llm_config() -> bool:
    cfg, _ = load_config_data()
    status = resolve_auth(
        config=cfg,
        env=dict(os.environ),
        runtime_context={"in_container": bool(os.environ.get("CGINS_PROJECT_DIR"))},
    )
    apply_auth_env(status, os.environ)

    # Provider/model fallback from legacy llm_info for compatibility.
    llm_info = cfg.get("llm_info", {}) if isinstance(cfg, dict) else {}
    provider = str(llm_info.get("provider", "")).strip().lower() if isinstance(llm_info, dict) else ""
    model = str(llm_info.get("model", "")).strip() if isinstance(llm_info, dict) else ""
    if provider and "LLM_PROVIDER" not in os.environ:
        os.environ["LLM_PROVIDER"] = provider
    if provider == "openai" and model and "OPENAI_MODEL" not in os.environ:
        os.environ["OPENAI_MODEL"] = model
    if provider == "anthropic" and model and "ANTHROPIC_MODEL" not in os.environ:
        os.environ["ANTHROPIC_MODEL"] = model
    if provider == "gemini" and model and "GEMINI_MODEL" not in os.environ:
        os.environ["GEMINI_MODEL"] = model
    return True


def ensure_llm_config() -> str:
    """Ensure LLM provider/model env vars are set from config, keychain, or existing env vars."""
    apply_llm_config()
    provider = str(os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if provider:
        return provider

    # Infer provider from available keys.
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY"):
        os.environ["LLM_PROVIDER"] = "openai"
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        os.environ["LLM_PROVIDER"] = "gemini"
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["LLM_PROVIDER"] = "anthropic"
        return "anthropic"

    return ""
