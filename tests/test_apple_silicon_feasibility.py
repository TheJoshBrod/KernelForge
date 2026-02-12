from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apple_silicon.device_probe import probe_device
from src.apple_silicon.feasibility import evaluate_candidate_feasibility


def test_feasibility_rejects_oversized_threadgroup() -> None:
    device = probe_device()
    record = evaluate_candidate_feasibility(
        device=device,
        template_mutations={"n_simdwidth": 32},
        kernel_overrides={"mul_mv_q4_k": {"threadgroup": 2048}},
    )
    assert record.attempted
    assert not record.success
    assert record.classification == "static_feasibility_reject"
    assert any("threadgroup_too_large" in reason for reason in record.reasons)


def test_feasibility_accepts_constrained_candidate() -> None:
    device = probe_device()
    record = evaluate_candidate_feasibility(
        device=device,
        template_mutations={"n_simdwidth": 32, "n_r0_q4_k": 3},
        kernel_overrides={"mul_mv_q4_k": {"threadgroup": 128, "tile": [16, 8]}},
    )
    assert record.attempted
    assert record.success
    assert record.classification == "feasible"
