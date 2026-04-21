from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_benchmarks.paper_bench.schema import validate_artifact_payload

from .helpers import build_common_payload


def test_kf_cast_requires_cast_hash_and_counts(sample_paths):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "benchmark_result",
            "variant": "kf_cast",
            "stage": "prefill",
            "cast_package_path": sample_paths["cast_path"],
            "cast_package_hash": None,
            "fallback_count": 0,
            "kernel_hit_count": 1,
            "latency_summary": {
                "count": 2,
                "mean_ms": 1.5,
                "median_ms": 1.5,
                "p95_ms": 1.95,
                "min_ms": 1.0,
                "max_ms": 2.0,
                "stddev_ms": 0.5,
            },
            "prompt_id": "p0",
            "prompt_hash": "abc123",
            "token_count": 2,
            "details": {},
        }
    )

    with pytest.raises(ValidationError):
        validate_artifact_payload(payload)


def test_model_path_hash_is_required(sample_paths):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "benchmark_result",
            "model_path_hash": None,
            "latency_summary": {
                "count": 2,
                "mean_ms": 1.5,
                "median_ms": 1.5,
                "p95_ms": 1.95,
                "min_ms": 1.0,
                "max_ms": 2.0,
                "stddev_ms": 0.5,
            },
            "prompt_id": "p0",
            "prompt_hash": "abc123",
            "token_count": 2,
            "details": {},
        }
    )

    with pytest.raises(ValidationError):
        validate_artifact_payload(payload)
