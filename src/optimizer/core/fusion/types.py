"""
Data models and validation constants for kernel fusion.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# Validation tolerances - loosened from PyTorch defaults for large tensors
FUSION_RTOL = 1e-3  # relative tolerance
FUSION_ATOL = 1e-5  # absolute tolerance
COSINE_THRESHOLD = 0.9999  # cosine similarity fallback threshold

# Retry settings
MAX_ATTEMPTS = 3


class FusionUIStatus(str, Enum):
    """Status from UI proposals."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FusionGenStatus(str, Enum):
    """Status of kernel generation."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class AttemptStatus(str, Enum):
    """Status of a single fusion attempt."""
    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"


class MemberOpContext(BaseModel):
    """Context for a single operation within a fusion group."""

    node_id: str  # e.g., "conv2d_4"
    op_type: str  # e.g., "conv2d"
    bench_op: str | None = None  # e.g., "torch_nn_functional_conv2d"
    io_dir: Path | None = None  # Path to entry_*.pt files
    existing_kernel_path: Path | None = None
    existing_kernel_code: str | None = None
    tensor_shapes: dict[str, list[int]] = Field(default_factory=dict)
    dtype: str | None = None
    pytorch_ms: float | None = None
    kernel_ms: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class FusionGroup(BaseModel):
    """A fusion group from fusion_groups.json or fusion.db."""

    id: str
    pattern_name: str
    name: str = ""
    members: list[str]  # Node IDs in execution order
    ui_status: FusionUIStatus = FusionUIStatus.PROPOSED
    gen_status: FusionGenStatus | None = None
    score: float = 0.0
    estimated_speedup: float = 1.0
    rationale: str = ""
    color_index: int = 0

    # Generation results
    baseline_ms: float | None = None
    fused_ms: float | None = None
    actual_speedup: float | None = None
    best_kernel_path: Path | None = None
    llm_model: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class FusionAttempt(BaseModel):
    """Record of a single fusion generation attempt."""

    id: int | None = None
    group_id: str
    attempt_num: int
    status: AttemptStatus
    kernel_path: Path | None = None
    error_message: str | None = None
    fused_ms: float | None = None
    llm_model: str | None = None
    timestamp: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class FusionResult(BaseModel):
    """Complete result of a fusion operation."""

    group_id: str
    status: FusionGenStatus
    kernel_path: Path | None = None
    baseline_ms: float | None = None
    fused_ms: float | None = None
    speedup: float | None = None
    error: str | None = None
    attempts: list[FusionAttempt] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
