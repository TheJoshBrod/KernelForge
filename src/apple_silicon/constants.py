from __future__ import annotations

import os
from pathlib import Path

CGINS_APP_NAME = "cgins"

LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
# Pinned from upstream HEAD on 2026-02-09.
LLAMA_CPP_PINNED_COMMIT = "262364e31d1da43596fe84244fba44e94a0de64e"

DEFAULT_PROFILE_MODE = "both"
DEFAULT_GATE_MODE = "quick"

# v1 workload defaults
CHAT_PROFILE = {
    "name": "chat",
    "ctx": 8192,
    "prompt_tokens_target": 1024,
    "generate_tokens": 256,
    "repeats_quick": 2,
    "repeats_full": 5,
}

LONG_PROFILE = {
    "name": "long",
    "ctx": 32768,
    "prompt_tokens_target": 24576,
    "generate_tokens": 128,
    "repeats_quick": 1,
    "repeats_full": 3,
}

# Fast long-context proxy used for early-stage screening.
LONG_SMOKE_PROFILE = {
    "name": "long_smoke",
    "ctx": 16384,
    "prompt_tokens_target": 4096,
    "generate_tokens": 128,
    "repeats_quick": 1,
    "repeats_full": 2,
}

# Claim-grade long-context profile used in final ABBA confirmation.
LONG_CLAIM_PROFILE = {
    "name": "long_claim",
    "ctx": LONG_PROFILE["ctx"],
    "prompt_tokens_target": LONG_PROFILE["prompt_tokens_target"],
    "generate_tokens": LONG_PROFILE["generate_tokens"],
    "repeats_quick": LONG_PROFILE["repeats_quick"],
    "repeats_full": LONG_PROFILE["repeats_full"],
}

PASS_PRIMARY_UPLIFT_PCT = 15.0
PASS_MAX_REGRESSION_PCT = 5.0

# Pinned default tiny model for local validation.
DEFAULT_MODEL_NAME = "Qwen2.5-0.5B-Instruct Q4_K_M"
DEFAULT_MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    f"{DEFAULT_MODEL_FILENAME}"
)
# Retrieved from huggingface resolver response header x-linked-etag on 2026-02-09.
DEFAULT_MODEL_SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"

SUPPORTED_ARCHS = {
    "llama",
    "qwen2",
    "qwen2moe",
    "qwen3",
}

LLM_TUNING_ATTEMPTS_QUICK = 1
LLM_TUNING_ATTEMPTS_FULL = 3

def _resolve_cache_root_raw(cache_root_override: str | Path | None = None) -> Path:
    override = str(cache_root_override or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    explicit = str(os.environ.get("CGINS_CACHE_ROOT", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    xdg = str(os.environ.get("XDG_CACHE_HOME", "")).strip()
    if xdg:
        return (Path(xdg).expanduser().resolve() / CGINS_APP_NAME / "apple_silicon").resolve()

    return (Path.home() / ".cache" / CGINS_APP_NAME / "apple_silicon").resolve()


CACHE_ROOT = _resolve_cache_root_raw()
MODELS_CACHE_DIR = CACHE_ROOT / "models"
PACKS_CACHE_DIR = CACHE_ROOT / "packs"
ACTIVE_PACKS_PATH = CACHE_ROOT / "active_packs.json"

PROMPT_SUITE_PATH = Path(__file__).resolve().parent / "prompts" / "benchmark_prompts.json"
LLM_TUNER_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "llm_tuner_system_prompt.md"

GGML_METAL_RESOURCES_ENV = "GGML_METAL_PATH_RESOURCES"


def configure_cache_root(cache_root_override: str | Path | None = None) -> Path:
    global CACHE_ROOT, MODELS_CACHE_DIR, PACKS_CACHE_DIR, ACTIVE_PACKS_PATH
    CACHE_ROOT = _resolve_cache_root_raw(cache_root_override)
    MODELS_CACHE_DIR = CACHE_ROOT / "models"
    PACKS_CACHE_DIR = CACHE_ROOT / "packs"
    ACTIVE_PACKS_PATH = CACHE_ROOT / "active_packs.json"

    override = str(cache_root_override or "").strip()
    if override:
        os.environ["CGINS_CACHE_ROOT"] = str(CACHE_ROOT)
    return CACHE_ROOT


def current_cache_root() -> Path:
    return CACHE_ROOT


def env_default_model_sha256() -> str:
    value = os.environ.get("CGINS_DEFAULT_MODEL_SHA256", "").strip().lower()
    return value or DEFAULT_MODEL_SHA256
