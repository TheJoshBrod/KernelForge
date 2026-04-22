"""
Kernel Fusion Engine module.

Provides the FusionEngine class for generating fused CUDA/Triton kernels
from accepted fusion groups.
"""
from src.optimizer.core.fusion.engine import FusionEngine
from src.optimizer.core.fusion.types import (
    COSINE_THRESHOLD,
    FUSION_ATOL,
    FUSION_RTOL,
    MAX_ATTEMPTS,
    AttemptStatus,
    FusionAttempt,
    FusionGenStatus,
    FusionGroup,
    FusionResult,
    FusionUIStatus,
    MemberOpContext,
)

__all__ = [
    # Main class
    "FusionEngine",
    # Types
    "FusionGroup",
    "FusionResult",
    "FusionAttempt",
    "MemberOpContext",
    "FusionUIStatus",
    "FusionGenStatus",
    "AttemptStatus",
    # Constants
    "MAX_ATTEMPTS",
    "FUSION_RTOL",
    "FUSION_ATOL",
    "COSINE_THRESHOLD",
]
