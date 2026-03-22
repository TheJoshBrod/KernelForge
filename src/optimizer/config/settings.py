"""
Optimizer runtime settings.

Why this file exists:
- Many jobs (profiling, benchmarking, generation) run in subprocesses launched by the
  Jac server. Those subprocesses use whatever Python interpreter Jac is running under.

Design goal:
- Prefer `pydantic-settings` when available (typed env parsing, validation).
- Do not hard-crash if `pydantic-settings` is missing. A missing optional dependency
  should not prevent baseline benchmarking from running locally.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

MIN_CUDA_VERSION: tuple[int, int] = (11, 8)


def _detect_cuda_home() -> str:
    """Return the best available CUDA installation directory.

    Resolution order:
    1. CUDA_HOME env var (already set by the user or conda/venv activation)
    2. nvcc on PATH - derive home from the binary location
    3. /usr/local/cuda - canonical symlink on most Linux distros
    4. Any /usr/local/cuda-* directory, preferring the newest version
    5. Hardcoded fallback (original default)
    """
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]

    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc lives at <cuda_home>/bin/nvcc
        return str(Path(nvcc).resolve().parent.parent)

    canonical = Path("/usr/local/cuda")
    if canonical.exists():
        return str(canonical)

    def _ver(p: Path) -> list[int]:
        try:
            return [int(x) for x in p.name.split("-")[1].split(".")]
        except Exception:
            return [0]

    versioned = sorted(Path("/usr/local").glob("cuda-*"), key=_ver, reverse=True)
    if versioned:
        return str(versioned[0])

    return "/usr/local/cuda-12.1"


_DEFAULT_CUDA_HOME = _detect_cuda_home()


def _coerce_env_value(raw: str, typ: Any) -> Any:
    # `from __future__ import annotations` makes f.type a string, so compare by name.
    type_name = typ if isinstance(typ, str) else getattr(typ, "__name__", "")
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    if type_name == "bool":
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return raw


def _load_from_env(prefix: str, cls: Any) -> Any:
    kwargs = {}
    for f in fields(cls):
        key = prefix + f.name.upper()
        if key not in os.environ:
            continue
        raw = os.environ.get(key, "")
        try:
            kwargs[f.name] = _coerce_env_value(raw, f.type)
        except Exception:
            # If parsing fails, fall back to raw string rather than crashing.
            kwargs[f.name] = raw
    return cls(**kwargs)


try:
    # pydantic v2 settings package
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

    class PipelineConfig(BaseSettings):
        batch_size: int = 50
        verifier_timeout_seconds: int = 300
        mcts_c_constant: float = 1.0
        llm_model_name: str = "anthropic/claude-sonnet-4-6"
        llm_request_timeout_seconds: int = 90
        llm_watchdog_timeout_seconds: int = 120
        llm_total_timeout_seconds: int = 180
        llm_retry_limit: int = 2
        llm_retry_backoff_seconds: float = 2.0
        cuda_home: str = _DEFAULT_CUDA_HOME
        retry_limit: int = 3
        ancestor_code_depth: int = 3
        pw_initial_exponent: float = 0.5
        pw_final_exponent: float = 0.3
        pw_anneal_steps: int = 1000
        stagnation_threshold: int = 10

        model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    settings = PipelineConfig()

except Exception:
    # Minimal fallback that keeps the project runnable without `pydantic-settings`.
    @dataclass
    class PipelineConfig:
        batch_size: int = 50
        verifier_timeout_seconds: int = 300
        mcts_c_constant: float = 1.0
        llm_model_name: str = "anthropic/claude-sonnet-4-6"
        llm_request_timeout_seconds: int = 90
        llm_watchdog_timeout_seconds: int = 120
        llm_total_timeout_seconds: int = 180
        llm_retry_limit: int = 2
        llm_retry_backoff_seconds: float = 2.0
        cuda_home: str = _DEFAULT_CUDA_HOME
        retry_limit: int = 3
        ancestor_code_depth: int = 3
        pw_initial_exponent: float = 0.5
        pw_final_exponent: float = 0.3
        pw_anneal_steps: int = 1000
        stagnation_threshold: int = 10

    settings = _load_from_env("OPTIMIZER_", PipelineConfig)
