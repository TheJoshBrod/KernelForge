from __future__ import annotations

from pathlib import Path

import yaml

from paper_benchmarks.paper_bench.provenance import sha256_path
from paper_benchmarks.paper_bench.registry import validate_model_suite_registry


def _write_workload(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id":"p0","prompt":"hello world"}\n', encoding="utf-8")
    return path


def _write_suite(path: Path, *, workload_path: Path, workload_type: str = "prompt_records") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite_id": path.stem,
        "benchmark_mode": "e2e_model",
        "description": "registry validator test suite",
        "workload_type": workload_type,
        "workload_path": str(workload_path),
        "synthetic_workload": False,
        "variants": ["eager", "torch_compile", "kf_cast"],
        "stages": ["load", "compile", "warmup", "prefill", "decode", "total_generate"],
        "batch_sizes": [1],
        "prompt_length_buckets": [{"bucket_id": "short", "min_tokens": 1, "max_tokens": 128}],
        "shape_or_length_buckets": [{"bucket_id": "short", "min_tokens": 1, "max_tokens": 128}],
        "max_new_tokens": 16,
        "warmup_runs": 1,
        "timed_runs": 2,
        "generation_mode": "greedy",
        "include_tokenization_in_timing": False,
        "measure_prefill_decode_separately": True,
        "device": "cpu",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_model(
    path: Path,
    *,
    suite_path: Path,
    workload_hash: str | None,
    paper_eligible: bool,
    baselines_required: list[str] | None = None,
    correctness_comparator: str | None = "exact_token_match_against_eager",
    benchmark_expectation: str = "candidate",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_file = path.parent / f"{path.stem}_model.bin"
    model_file.write_bytes(b"weights")
    config_file = path.parent / f"{path.stem}_config.json"
    config_file.write_text('{"hidden_size": 16}', encoding="utf-8")
    payload = {
        "model_id": path.stem,
        "category": "test-category",
        "loader_kind": "transformers_causal_lm",
        "model_path": str(model_file),
        "model_config_path": str(config_file),
        "task_type": "moe_decoder_llm",
        "dtype": "bf16",
        "device": "cpu",
        "validation_suite_path": str(suite_path),
        "workload_hash": workload_hash,
        "batch_sizes": [1],
        "shape_or_length_buckets": [{"bucket_id": "short", "min_tokens": 1, "max_tokens": 128}],
        "correctness_comparator": correctness_comparator,
        "baselines_required": baselines_required if baselines_required is not None else ["eager", "torch_compile", "kf_cast"],
        "deployment_artifact_path": None,
        "expected_memory_footprint_notes": ["Record peak memory before enabling paper mode."],
        "paper_eligible": paper_eligible,
        "benchmark_expectation": benchmark_expectation,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_registry(path: Path, entries: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "models": entries}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_missing_workload_fails(tmp_path: Path):
    suite_path = _write_suite(tmp_path / "suites" / "missing_workload_suite.yaml", workload_path=tmp_path / "data" / "missing.jsonl")
    model_path = _write_model(
        tmp_path / "models" / "missing_workload_model.yaml",
        suite_path=suite_path,
        workload_hash="deadbeef",
        paper_eligible=True,
    )
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "model_id": "missing_workload_model",
                "category": "MoE decoder LLM",
                "model_config": str(model_path),
                "suite_config": str(suite_path),
                "benchmark_expectation": "candidate",
            }
        ],
    )

    report = validate_model_suite_registry(registry_path)

    assert report.entries[0].paper_eligible_effective is False
    assert "workload file missing" in report.entries[0].validation_issues


def test_missing_baseline_requirement_fails(tmp_path: Path):
    workload_path = _write_workload(tmp_path / "data" / "prompts.jsonl")
    suite_path = _write_suite(tmp_path / "suites" / "baseline_suite.yaml", workload_path=workload_path)
    model_path = _write_model(
        tmp_path / "models" / "missing_baseline_model.yaml",
        suite_path=suite_path,
        workload_hash=sha256_path(workload_path),
        paper_eligible=True,
        baselines_required=["eager"],
    )
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "model_id": "missing_baseline_model",
                "category": "Dense decoder LLM",
                "model_config": str(model_path),
                "suite_config": str(suite_path),
                "benchmark_expectation": "candidate",
            }
        ],
    )

    report = validate_model_suite_registry(registry_path)

    assert report.entries[0].paper_eligible_effective is False
    assert "baselines_required missing torch_compile" in report.entries[0].validation_issues


def test_model_with_no_correctness_comparator_fails(tmp_path: Path):
    workload_path = _write_workload(tmp_path / "data" / "prompts.jsonl")
    suite_path = _write_suite(tmp_path / "suites" / "comparator_suite.yaml", workload_path=workload_path)
    model_path = _write_model(
        tmp_path / "models" / "missing_comparator_model.yaml",
        suite_path=suite_path,
        workload_hash=sha256_path(workload_path),
        paper_eligible=True,
        correctness_comparator=None,
    )
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "model_id": "missing_comparator_model",
                "category": "Transformer encoder model",
                "model_config": str(model_path),
                "suite_config": str(suite_path),
                "benchmark_expectation": "candidate",
            }
        ],
    )

    report = validate_model_suite_registry(registry_path)

    assert report.entries[0].paper_eligible_effective is False
    assert "correctness_comparator" in report.entries[0].missing_fields
    assert "correctness comparator missing" in report.entries[0].validation_issues


def test_neutral_and_regression_models_are_allowed_and_reported(tmp_path: Path):
    workload_a = _write_workload(tmp_path / "data" / "neutral.jsonl")
    workload_b = _write_workload(tmp_path / "data" / "regression.jsonl")
    suite_a = _write_suite(tmp_path / "suites" / "neutral_suite.yaml", workload_path=workload_a)
    suite_b = _write_suite(tmp_path / "suites" / "regression_suite.yaml", workload_path=workload_b)
    model_a = _write_model(
        tmp_path / "models" / "neutral_model.yaml",
        suite_path=suite_a,
        workload_hash=sha256_path(workload_a),
        paper_eligible=False,
        benchmark_expectation="neutral",
    )
    model_b = _write_model(
        tmp_path / "models" / "regression_model.yaml",
        suite_path=suite_b,
        workload_hash=sha256_path(workload_b),
        paper_eligible=False,
        benchmark_expectation="regression",
    )
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "model_id": "neutral_model",
                "category": "Vision transformer",
                "model_config": str(model_a),
                "suite_config": str(suite_a),
                "benchmark_expectation": "neutral",
            },
            {
                "model_id": "regression_model",
                "category": "Dynamic-shape workload",
                "model_config": str(model_b),
                "suite_config": str(suite_b),
                "benchmark_expectation": "regression",
            },
        ],
    )

    report = validate_model_suite_registry(registry_path)

    expectations = {entry.model_id: entry.benchmark_expectation for entry in report.entries}
    assert expectations["neutral_model"] == "neutral"
    assert expectations["regression_model"] == "regression"


def test_registry_validator_marks_incomplete_entries_non_paper_without_requiring_all_ten(tmp_path: Path):
    workload_path = _write_workload(tmp_path / "data" / "complete.jsonl")
    suite_complete = _write_suite(tmp_path / "suites" / "complete_suite.yaml", workload_path=workload_path)
    suite_incomplete = _write_suite(tmp_path / "suites" / "incomplete_suite.yaml", workload_path=tmp_path / "data" / "missing.jsonl")
    model_complete = _write_model(
        tmp_path / "models" / "complete_model.yaml",
        suite_path=suite_complete,
        workload_hash=sha256_path(workload_path),
        paper_eligible=False,
    )
    model_incomplete = _write_model(
        tmp_path / "models" / "incomplete_model.yaml",
        suite_path=suite_incomplete,
        workload_hash=None,
        paper_eligible=False,
        baselines_required=[],
    )
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "model_id": "complete_model",
                "category": "Batch-1 latency workload",
                "model_config": str(model_complete),
                "suite_config": str(suite_complete),
                "benchmark_expectation": "candidate",
            },
            {
                "model_id": "incomplete_model",
                "category": "Batched throughput workload",
                "model_config": str(model_incomplete),
                "suite_config": str(suite_incomplete),
                "benchmark_expectation": "candidate",
            },
        ],
    )

    report = validate_model_suite_registry(registry_path)

    assert report.model_count == 2
    assert report.incomplete_count == 2
    assert any(entry.model_id == "incomplete_model" and entry.paper_eligible_effective is False for entry in report.entries)
