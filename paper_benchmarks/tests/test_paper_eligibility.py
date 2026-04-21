from __future__ import annotations

from paper_benchmarks.paper_bench.schema import SummaryArtifact, validate_artifact_payload

from .helpers import build_common_payload


def _benchmark_payload(sample_paths, **updates):
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
            "details": {},
        }
    )
    payload.update(updates)
    return payload


def test_missing_workload_hash_marks_non_paper(sample_paths):
    artifact = validate_artifact_payload(
        _benchmark_payload(sample_paths, workload_hash=None)
    )

    assert artifact.paper_eligible is False
    assert "workload hash missing" in artifact.paper_eligibility_issues


def test_synthetic_workload_marks_non_paper(sample_paths):
    artifact = validate_artifact_payload(
        _benchmark_payload(sample_paths, synthetic_workload=True)
    )

    assert artifact.paper_eligible is False
    assert "synthetic workload used" in artifact.paper_eligibility_issues


def test_missing_correctness_marks_non_paper(sample_paths):
    artifact = validate_artifact_payload(
        _benchmark_payload(sample_paths, correctness_status="skipped")
    )

    assert artifact.paper_eligible is False
    assert "correctness did not pass" in artifact.paper_eligibility_issues


def test_fallback_without_explicit_reporting_marks_non_paper(sample_paths):
    artifact = validate_artifact_payload(
        _benchmark_payload(
            sample_paths,
            variant="kf_cast",
            cast_package_path=sample_paths["cast_path"],
            cast_package_hash=sample_paths["cast_hash"],
            fallback_count=None,
            kernel_hit_count=None,
            correctness_status="passed",
        )
    )

    assert artifact.paper_eligible is False
    assert "fallback count missing" in artifact.paper_eligibility_issues
    assert "kernel hit count missing" in artifact.paper_eligibility_issues


def test_eager_only_comparison_cannot_support_paper_model_speedup_claim(sample_paths):
    common = build_common_payload(sample_paths)
    payload = {
        **common,
        "artifact_type": "summary_report",
        "variant": None,
        "stage": None,
        "latency_samples_ms": [],
        "correctness_status": "not_applicable",
        "rows": [
            {
                "variant": "eager",
                "stage": "prefill",
                "correctness_status": "reference",
                "sample_count": 2,
                "median_ms": 1.5,
                "mean_ms": 1.5,
                "speedup_vs_eager": 1.0,
                "paper_eligible": True,
                "claim_eligible": True,
                "fallback_count": None,
                "kernel_hit_count": None,
            }
        ],
        "summary_markdown_path": sample_paths["suite_path"],
    }

    artifact = validate_artifact_payload(payload)

    assert isinstance(artifact, SummaryArtifact)
    assert artifact.paper_eligible is False
    assert "missing torch_compile variant for model-level comparison" in artifact.paper_eligibility_issues
    assert "missing kf_cast variant for model-level comparison" in artifact.paper_eligibility_issues
    assert "no comparable model-level stage across eager, torch_compile, and kf_cast" in artifact.paper_eligibility_issues
