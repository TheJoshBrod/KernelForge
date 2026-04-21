from __future__ import annotations

from pathlib import Path

import pytest

from paper_benchmarks.paper_bench.registry import SyntheticWorkloadError, enforce_workload_policy, load_suite_config


def test_synthetic_suite_is_forbidden_without_explicit_flag(tmp_path: Path):
    workload = tmp_path / "prompts.jsonl"
    workload.write_text('{"id":"p0","prompt":"demo"}\n', encoding="utf-8")
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suite_id: synthetic_demo",
                "benchmark_mode: e2e_model",
                "workload_type: prompt_jsonl",
                f"workload_path: {workload}",
                "synthetic_workload: true",
                "variants: [eager]",
                "stages: [prefill]",
                "warmup_count: 1",
                "timed_run_count: 1",
                "batch_size: 1",
                "max_new_tokens: 1",
                "device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    suite = load_suite_config(suite_path)
    with pytest.raises(SyntheticWorkloadError):
        enforce_workload_policy(suite, allow_synthetic_demo=False)

    assert enforce_workload_policy(suite, allow_synthetic_demo=True) is False

