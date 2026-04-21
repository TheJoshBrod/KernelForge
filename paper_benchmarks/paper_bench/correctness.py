from __future__ import annotations

from typing import Any

import torch

from .schema import CorrectnessStatus


def reference_correctness() -> tuple[CorrectnessStatus, str | None]:
    return CorrectnessStatus.reference, "Reference eager run."


def skipped_correctness(reason: str) -> tuple[CorrectnessStatus, str]:
    return CorrectnessStatus.skipped, reason


def compare_token_sequences(reference_tokens: list[int], candidate_tokens: list[int]) -> tuple[CorrectnessStatus, str | None]:
    if reference_tokens == candidate_tokens:
        return CorrectnessStatus.passed, None
    return CorrectnessStatus.failed, "Generated token sequence mismatch."


def compare_tensors(
    reference: Any,
    candidate: Any,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> tuple[CorrectnessStatus, str | None]:
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        return CorrectnessStatus.skipped, "Tensor comparison skipped; non-tensor outputs."
    if reference.shape != candidate.shape:
        return CorrectnessStatus.failed, f"Shape mismatch: {reference.shape} != {candidate.shape}"
    if torch.allclose(reference, candidate, atol=atol, rtol=rtol):
        return CorrectnessStatus.passed, None
    max_abs_diff = float((reference - candidate).abs().max().item())
    return CorrectnessStatus.failed, f"Tensor mismatch; max_abs_diff={max_abs_diff:.6g}"

