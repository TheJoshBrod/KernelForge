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
from dataclasses import dataclass, fields
from typing import Any


def _coerce_env_value(raw: str, typ: Any) -> Any:
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    if typ is bool:
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
        llm_model_name: str = "claude-sonnet-4-6"
        cuda_home: str = "/usr/local/cuda-12.1"
        retry_limit: int = 3
        ancestor_code_depth: int = 3

        model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    settings = PipelineConfig()

except Exception:
    # Minimal fallback that keeps the project runnable without `pydantic-settings`.
    @dataclass
    class PipelineConfig:
        batch_size: int = 50
        verifier_timeout_seconds: int = 300
        mcts_c_constant: float = 1.0
        llm_model_name: str = "claude-sonnet-4-6"
        cuda_home: str = "/usr/local/cuda-12.1"
        retry_limit: int = 3
        ancestor_code_depth: int = 3

    settings = _load_from_env("OPTIMIZER_", PipelineConfig)
