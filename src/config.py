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


def load_project_config(project_dir: Path | None) -> dict:
    if not project_dir:
        return {}
    config_path = project_dir / "config.json"
    if not config_path.exists():
        # Fallback: check if project_dir IS the config file
        if project_dir.suffix == ".json" and project_dir.exists():
           try:
              return json.loads(project_dir.read_text(encoding="utf-8"))
           except Exception:
              return {}
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def apply_llm_config() -> bool:
    config_path = _find_config_path()
    if not config_path:
        return False
    
    # config_path is the file path, so we pass its parent or handle it
    # _find_config_path returns the FILE path.
    # load_project_config expects a DIR (or we adjust it).
    # Let's just manually load here to avoid circular logic or path confusion, 
    # OR make load_project_config smarter.
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


def ensure_llm_config() -> str:
    """Ensure LLM provider/model env vars are set from config or existing keys."""
    apply_llm_config()

    provider = str(os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if provider:
        return provider

    # Infer provider from available API keys (prefer OpenAI)
    if os.environ.get("OPENAI_API_KEY"):
        os.environ["LLM_PROVIDER"] = "openai"
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        os.environ["LLM_PROVIDER"] = "gemini"
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["LLM_PROVIDER"] = "anthropic"
        return "anthropic"

    return ""
