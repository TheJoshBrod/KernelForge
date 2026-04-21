from __future__ import annotations

from pathlib import Path

import pytest

from paper_benchmarks.paper_bench.cast_selection import NoEligibleCastKernelsError, select_fastest_valid_kernels


def _write_kernel(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _candidate(
    kernel_path: Path,
    *,
    candidate_id: str,
    evidence_tier: str = "deployment",
    median_latency_ms: float = 1.0,
    p95_latency_ms: float = 1.0,
    mean_latency_ms: float = 1.0,
    correctness_passed: bool = True,
    has_captured_inputs: bool = True,
    timing_real: bool = True,
    benchmark_source_path: Path | None = None,
    benchmark_source_hash: str | None = None,
    benchmark_errors: list[str] | None = None,
    runtime_audit_passed: bool | None = True,
    verified_timestamp: str = "2999-01-01T00:00:00Z",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "kernel_source_path": str(kernel_path),
        "benchmark_source_path": str(benchmark_source_path or kernel_path),
        "benchmark_source_hash": benchmark_source_hash,
        "benchmark_artifact_path": "benchmarks/op_benchmarks.json",
        "benchmark_row_ref": f"results[{candidate_id}]",
        "evidence_tier": evidence_tier,
        "median_latency_ms": median_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "mean_latency_ms": mean_latency_ms,
        "correctness_passed": correctness_passed,
        "has_captured_inputs": has_captured_inputs,
        "timing_real": timing_real,
        "benchmark_errors": benchmark_errors or [],
        "runtime_audit_passed": runtime_audit_passed,
        "verified_timestamp": verified_timestamp,
    }


def test_chooses_fastest_valid_kernel(tmp_path: Path):
    kernel_a = _write_kernel(tmp_path / "kernels" / "a.cu", "// a\n")
    kernel_b = _write_kernel(tmp_path / "kernels" / "b.cu", "// b\n")

    report = select_fastest_valid_kernels(
        {
            "torch_nn_functional_linear": [
                _candidate(kernel_a, candidate_id="a", median_latency_ms=1.2, p95_latency_ms=1.5, mean_latency_ms=1.3),
                _candidate(kernel_b, candidate_id="b", median_latency_ms=0.8, p95_latency_ms=1.1, mean_latency_ms=0.9),
            ]
        },
        project_root=tmp_path,
    )

    selected = report["selected_ops"]["torch_nn_functional_linear"]
    assert selected["candidate_id"] == "b"
    assert selected["kernel_source_path"] == str(kernel_b.resolve())


def test_rejects_correctness_failed_fastest_kernel(tmp_path: Path):
    fast_bad = _write_kernel(tmp_path / "kernels" / "fast_bad.cu", "// fast_bad\n")
    slow_good = _write_kernel(tmp_path / "kernels" / "slow_good.cu", "// slow_good\n")

    report = select_fastest_valid_kernels(
        {
            "torch_nn_functional_linear": [
                _candidate(fast_bad, candidate_id="fast_bad", median_latency_ms=0.5, correctness_passed=False),
                _candidate(slow_good, candidate_id="slow_good", median_latency_ms=0.9),
            ]
        },
        project_root=tmp_path,
    )

    selected = report["selected_ops"]["torch_nn_functional_linear"]
    rejected = report["rejected_candidates"]["torch_nn_functional_linear"]
    assert selected["candidate_id"] == "slow_good"
    assert any(item["candidate_id"] == "fast_bad" and "correctness failed" in item["rejection_reasons"] for item in rejected)


def test_rejects_missing_captured_inputs(tmp_path: Path):
    missing_inputs = _write_kernel(tmp_path / "kernels" / "missing_inputs.cu", "// missing_inputs\n")
    valid = _write_kernel(tmp_path / "kernels" / "valid.cu", "// valid\n")

    report = select_fastest_valid_kernels(
        {
            "torch_nn_functional_linear": [
                _candidate(missing_inputs, candidate_id="missing_inputs", median_latency_ms=0.4, has_captured_inputs=False),
                _candidate(valid, candidate_id="valid", median_latency_ms=0.8),
            ]
        },
        project_root=tmp_path,
    )

    assert report["selected_ops"]["torch_nn_functional_linear"]["candidate_id"] == "valid"
    rejected = report["rejected_candidates"]["torch_nn_functional_linear"]
    assert any(item["candidate_id"] == "missing_inputs" and "missing captured inputs" in item["rejection_reasons"] for item in rejected)


def test_rejects_stale_source_hash_mismatch(tmp_path: Path):
    stale = _write_kernel(tmp_path / "kernels" / "stale.cu", "// stale\n")
    valid = _write_kernel(tmp_path / "kernels" / "valid.cu", "// valid\n")

    report = select_fastest_valid_kernels(
        {
            "torch_nn_functional_linear": [
                _candidate(stale, candidate_id="stale", median_latency_ms=0.3, benchmark_source_hash="not-the-real-hash"),
                _candidate(valid, candidate_id="valid", median_latency_ms=0.9),
            ]
        },
        project_root=tmp_path,
    )

    assert report["selected_ops"]["torch_nn_functional_linear"]["candidate_id"] == "valid"
    rejected = report["rejected_candidates"]["torch_nn_functional_linear"]
    assert any(item["candidate_id"] == "stale" and "stale source hash mismatch" in item["rejection_reasons"] for item in rejected)


def test_prefers_deployment_timing_over_micro_timing(tmp_path: Path):
    deployment = _write_kernel(tmp_path / "kernels" / "deployment.cu", "// deployment\n")
    micro = _write_kernel(tmp_path / "kernels" / "micro.cu", "// micro\n")

    report = select_fastest_valid_kernels(
        {
            "torch_nn_functional_linear": [
                _candidate(deployment, candidate_id="deployment", evidence_tier="deployment", median_latency_ms=0.9),
                _candidate(micro, candidate_id="micro", evidence_tier="micro_only", median_latency_ms=0.3),
            ]
        },
        project_root=tmp_path,
        allow_micro_only=True,
    )

    assert report["selected_ops"]["torch_nn_functional_linear"]["candidate_id"] == "deployment"


def test_deterministic_tie_break(tmp_path: Path):
    a = _write_kernel(tmp_path / "kernels" / "a.cu", "// a\n")
    b = _write_kernel(tmp_path / "kernels" / "b.cu", "// b\n")
    c = _write_kernel(tmp_path / "kernels" / "c.cu", "// c\n")
    d = _write_kernel(tmp_path / "kernels" / "d.cu", "// d\n")

    report = select_fastest_valid_kernels(
        {
                "torch_nn_functional_linear": [
                    _candidate(a, candidate_id="a", median_latency_ms=1.0, p95_latency_ms=1.4, mean_latency_ms=1.2),
                    _candidate(b, candidate_id="b", median_latency_ms=1.0, p95_latency_ms=1.2, mean_latency_ms=1.1, verified_timestamp="2998-01-01T00:00:00Z"),
                    _candidate(c, candidate_id="c", median_latency_ms=1.0, p95_latency_ms=1.2, mean_latency_ms=1.0, verified_timestamp="2999-01-02T00:00:00Z"),
                    _candidate(d, candidate_id="d", median_latency_ms=1.0, p95_latency_ms=1.2, mean_latency_ms=1.0, verified_timestamp="2999-01-02T00:00:00Z"),
                ]
            },
            project_root=tmp_path,
        )

    assert report["selected_ops"]["torch_nn_functional_linear"]["candidate_id"] == "c"


def test_reports_rejected_candidates(tmp_path: Path):
    bad = _write_kernel(tmp_path / "kernels" / "bad.cu", "// bad\n")
    good = _write_kernel(tmp_path / "kernels" / "good.cu", "// good\n")

    report = select_fastest_valid_kernels(
        {
            "torch_nn_functional_linear": [
                _candidate(bad, candidate_id="bad", benchmark_errors=["runtime failure"]),
                _candidate(good, candidate_id="good"),
            ]
        },
        project_root=tmp_path,
    )

    rejected = report["rejected_candidates"]["torch_nn_functional_linear"]
    assert report["selected_ops"]["torch_nn_functional_linear"]["candidate_id"] == "good"
    assert any(item["candidate_id"] == "bad" and "benchmark errors" in item["rejection_reasons"] for item in rejected)


def test_does_not_silently_export_nothing(tmp_path: Path):
    bad = _write_kernel(tmp_path / "kernels" / "bad.cu", "// bad\n")

    with pytest.raises(NoEligibleCastKernelsError) as exc_info:
        select_fastest_valid_kernels(
            {
                "torch_nn_functional_linear": [
                    _candidate(bad, candidate_id="bad", correctness_passed=False),
                ]
            },
            project_root=tmp_path,
        )

    assert "No eligible kernels matched auto_best_fastest_valid" in str(exc_info.value)
    assert exc_info.value.report["selected_ops"] == {}
