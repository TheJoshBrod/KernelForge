import json
import os
from pathlib import Path

from src.llm.runtime_config import resolve_runtime_env


def _find_config_path() -> Path | None:
    override = os.environ.get("KFORGE_CONFIG_PATH")
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


def _provider_from_env() -> str:
    provider = (
        os.environ.get("LLM_PROVIDER")
        or os.environ.get("KFORGE_LLM_PROVIDER")
        or ""
    )
    provider = provider.strip().lower()
    if provider == "gemini":
        return "google"
    return provider


def _model_from_env(provider: str) -> str:
    kforge_model = os.environ.get("KFORGE_LLM_MODEL", "").strip()
    if kforge_model:
        return kforge_model
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "").strip()
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_MODEL", "").strip()
    if provider == "google":
        return (
            os.environ.get("GEMINI_MODEL", "").strip()
            or os.environ.get("GOOGLE_MODEL", "").strip()
        )
    return ""


def apply_llm_config() -> bool:
    config_path = _find_config_path()
    global_cfg: dict = {}
    try:
        if config_path and config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                global_cfg = data
    except Exception:
        global_cfg = {}

    project_cfg: dict = {}
    project_cfg_override = os.environ.get("KFORGE_PROJECT_CONFIG_PATH")
    if project_cfg_override:
        try:
            project_path = Path(project_cfg_override)
            if project_path.exists():
                loaded = json.loads(project_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    project_cfg = loaded
        except Exception:
            project_cfg = {}

    override_provider = _provider_from_env()
    override_model = _model_from_env(override_provider)
    env_map = resolve_runtime_env(
        global_config=global_cfg,
        project_config=project_cfg,
        override_provider=override_provider,
        override_model=override_model,
    )
    if not env_map:
        return False

    force_selected_env = bool(override_provider or override_model)
    for key, value in env_map.items():
        if value is not None and (force_selected_env or str(key) not in os.environ):
            os.environ[str(key)] = str(value)
    return True


def ensure_llm_config() -> str:
    """Ensure LLM provider/model env vars are set from config or existing keys."""
    apply_llm_config()

    provider = str(os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if provider == "gemini":
        provider = "google"
        os.environ["LLM_PROVIDER"] = "google"
    if provider:
        return provider

    # Infer provider from available API keys (prefer OpenAI)
    if os.environ.get("OPENAI_API_KEY"):
        os.environ["LLM_PROVIDER"] = "openai"
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["LLM_PROVIDER"] = "anthropic"
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        os.environ["LLM_PROVIDER"] = "google"
        return "google"

    return ""
