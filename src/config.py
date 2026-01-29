import json
import os
from pathlib import Path


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


def apply_llm_config() -> bool:
    config_path = _find_config_path()
    if not config_path:
        return False

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    llm_info = data.get("llm_info")
    if not isinstance(llm_info, dict):
        return False

    provider = str(llm_info.get("provider", "")).strip().lower()
    model = str(llm_info.get("model", "")).strip()
    apikey = str(llm_info.get("apikey", "")).strip()

    if provider:
        os.environ["LLM_PROVIDER"] = provider

    if provider == "openai":
        if apikey:
            os.environ["OPENAI_API_KEY"] = apikey
        if model:
            os.environ["OPENAI_MODEL"] = model
    elif provider == "anthropic":
        if apikey:
            os.environ["ANTHROPIC_API_KEY"] = apikey
        if model:
            os.environ["ANTHROPIC_MODEL"] = model
    elif provider == "gemini":
        if apikey:
            os.environ["GOOGLE_API_KEY"] = apikey
            os.environ["GEMINI_API_KEY"] = apikey
        if model:
            os.environ["GEMINI_MODEL"] = model

    return True
