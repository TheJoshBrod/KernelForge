from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_benchmarks.paper_bench.schema import BenchmarkArtifact, validate_artifact_payload

from .helpers import build_common_payload


def test_benchmark_artifact_validation_accepts_valid_payload(sample_paths):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "benchmark_result",
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
            "details": {"note": "ok"},
        }
    )

    artifact = validate_artifact_payload(payload)
    assert isinstance(artifact, BenchmarkArtifact)
    assert artifact.stage.value == "prefill"


def test_missing_required_suite_hash_fails_validation(sample_paths):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "benchmark_result",
            "suite_hash": "",
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
