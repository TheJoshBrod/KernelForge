from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_benchmarks.paper_bench.provenance import compute_suite_hash, sha256_path


@pytest.fixture
def sample_paths(tmp_path: Path) -> dict[str, str]:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model-weights")

    model_config_path = tmp_path / "config.json"
    model_config_path.write_text('{"hidden_size": 16}', encoding="utf-8")

    workload_path = tmp_path / "prompts.jsonl"
    workload_path.write_text('{"id":"p0","prompt":"hello world"}\n', encoding="utf-8")

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suite_id: test_suite",
                "benchmark_mode: e2e_model",
                "workload_type: prompt_jsonl",
                f"workload_path: {workload_path}",
                "synthetic_workload: false",
                "variants: [eager, torch_compile]",
                "stages: [prefill, decode, total_generate]",
                "warmup_count: 1",
                "timed_run_count: 2",
                "batch_size: 1",
                "max_new_tokens: 2",
                "device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cast_path = tmp_path / "model.cast"
    cast_path.write_bytes(b"cast-bundle")

    return {
        "model_path": str(model_path),
        "model_config_path": str(model_config_path),
        "workload_path": str(workload_path),
        "suite_path": str(suite_path),
        "cast_path": str(cast_path),
        "suite_file_hash": sha256_path(suite_path),
        "suite_hash": compute_suite_hash(suite_path, workload_path),
        "workload_hash": sha256_path(workload_path),
        "model_path_hash": sha256_path(model_path),
        "model_config_hash": sha256_path(model_config_path),
        "cast_hash": sha256_path(cast_path),
    }
