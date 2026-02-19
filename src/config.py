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

    api_keys = data.get("api_keys")
    if not isinstance(api_keys, dict):
        # Fallback to old llm_info if api_keys missing (migration compatibility)
        llm_info = data.get("llm_info")
        if isinstance(llm_info, dict):
             provider = str(llm_info.get("provider", "")).strip().lower()
             model = str(llm_info.get("model", "")).strip()
             apikey = str(llm_info.get("apikey", "")).strip()
             if provider == "openai": api_keys = {"openai": apikey}
             elif provider == "anthropic": api_keys = {"anthropic": apikey}
             elif provider == "gemini": api_keys = {"gemini": apikey}
        
    if not isinstance(api_keys, dict):
        return False

    # Determine active provider/model from env
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip() or \
            os.environ.get("OPENAI_MODEL", "").strip() or \
            os.environ.get("ANTHROPIC_MODEL", "").strip() or \
            os.environ.get("GEMINI_MODEL", "").strip() or \
            os.environ.get("CGINS_MODEL", "").strip()

    # Infer provider from model if missing
    if not provider and model:
        if model.startswith("gpt"):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gemini"):
            provider = "gemini"

    # Set env vars based on provider
    if provider == "openai":
        key = api_keys.get("openai", "")
        if key: os.environ["OPENAI_API_KEY"] = key
        if model: os.environ["OPENAI_MODEL"] = model
    elif provider == "anthropic":
        key = api_keys.get("anthropic", "")
        if key: os.environ["ANTHROPIC_API_KEY"] = key
        if model: os.environ["ANTHROPIC_MODEL"] = model
    elif provider == "gemini":
        key = api_keys.get("gemini", "")
        if key: 
            os.environ["GOOGLE_API_KEY"] = key
            os.environ["GEMINI_API_KEY"] = key
        if model: os.environ["GEMINI_MODEL"] = model
    
    # If no provider set but keys exist, set defaults (fallback behavior)
    if not provider:
        if api_keys.get("openai"):
             os.environ["OPENAI_API_KEY"] = api_keys.get("openai")
        if api_keys.get("anthropic"):
             os.environ["ANTHROPIC_API_KEY"] = api_keys.get("anthropic")
        if api_keys.get("gemini"):
             k = api_keys.get("gemini")
             os.environ["GOOGLE_API_KEY"] = k
             os.environ["GEMINI_API_KEY"] = k

    return True

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
