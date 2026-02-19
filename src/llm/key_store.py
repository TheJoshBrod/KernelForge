"""Global API-key config helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .models import normalize_provider

DEFAULT_API_KEYS = {
    "openai": "",
    "anthropic": "",
    "google": "",
}

DEFAULT_CONFIG = {
    "api_keys": copy.deepcopy(DEFAULT_API_KEYS),
    "providers": [],
    "ssh_connections": [],
    "active_ssh_index": -1,
    "selected_gpu_index": -1,
    "selected_gpu_info": None,
}


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw = copy.deepcopy(cfg or {})
    out = copy.deepcopy(DEFAULT_CONFIG)
    out.update(raw)

    api_keys = copy.deepcopy(DEFAULT_API_KEYS)
    raw_api_keys = raw.get("api_keys")
    if isinstance(raw_api_keys, dict):
        for provider in api_keys:
            api_keys[provider] = _str_or_empty(raw_api_keys.get(provider, "")).strip()

    # One-release migration path from old llm_info shape.
    llm_info = raw.get("llm_info")
    if isinstance(llm_info, dict):
        legacy_provider = normalize_provider(_str_or_empty(llm_info.get("provider", "")))
        legacy_key = _str_or_empty(llm_info.get("apikey", "")).strip()
        if legacy_provider in api_keys and legacy_key and not api_keys[legacy_provider]:
            api_keys[legacy_provider] = legacy_key

    out["api_keys"] = api_keys
    out.pop("llm_info", None)
    return out


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        import json

        raw = json.loads(path.read_text())
        return normalize_config(raw)
    except Exception:
        return copy.deepcopy(DEFAULT_CONFIG)


def save_config(config_path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_config(cfg)
    path = Path(config_path)
    import json

    path.write_text(json.dumps(normalized, indent=2))
    return normalized

