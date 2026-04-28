from __future__ import annotations

import json
from pathlib import Path

from paper_benchmarks.paper_bench.llm_runner import load_prompt_records
from paper_benchmarks.paper_bench.provenance import sha256_path
from paper_benchmarks.paper_bench.registry import load_model_config, load_suite_config


BENCH_DATASETS = Path("/home/gb10/Projects/Kernal-Forge/datacollection/extreme_benchmark/datasets")


def _assert_prompt_manifest(path: Path, *, expected_dataset: str, expected_license: str, expected_count: int) -> None:
    manifest_path = path.with_suffix(".manifest.json")
    assert path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_suite = load_prompt_records(path)
    prompt_ids = [record["id"] for record in prompt_suite.records]

    assert manifest["dataset"] == expected_dataset
    assert manifest["license"] == expected_license
    assert manifest["license_status"] == "verified"
    assert manifest["synthetic_workload"] is False
    assert manifest["prompt_text_may_be_stored_in_raw_artifacts"] is False
    assert manifest["prompt_file_hash"] == sha256_path(path)
    assert prompt_suite.manifest_path == str(manifest_path)
    assert prompt_suite.synthetic_workload is False
    assert len(prompt_ids) == expected_count
    assert len(prompt_ids) == len(set(prompt_ids))
    assert all(record.get("prompt_sha256") for record in prompt_suite.records)
    assert all(record.get("expected_answer") for record in prompt_suite.records)


def test_gsm8k_manifest_is_frozen_and_license_verified():
    _assert_prompt_manifest(
        BENCH_DATASETS / "gsm8k_test_manifest.jsonl",
        expected_dataset="openai/gsm8k",
        expected_license="MIT",
        expected_count=512,
    )


def test_arc_challenge_manifest_is_frozen_and_license_verified():
    _assert_prompt_manifest(
        BENCH_DATASETS / "arc_challenge_validation_manifest.jsonl",
        expected_dataset="allenai/ai2_arc",
        expected_license="CC-BY-SA-4.0",
        expected_count=299,
    )


def test_gemma_iiswc_candidate_suite_configs_parse():
    expectations = {
        "gemma_gsm8k_paper.yaml": ("gemma_gsm8k_paper", "gsm8k_test_manifest.jsonl", "MIT", 128, 5, 20),
        "gemma_gsm8k_smoke.yaml": ("gemma_gsm8k_smoke", "gsm8k_test_manifest.jsonl", "MIT", 16, 1, 2),
        "gemma_gsm8k_paper_no_cache.yaml": ("gemma_gsm8k_paper_no_cache", "gsm8k_test_manifest.jsonl", "MIT", 16, 5, 20),
        "gemma_gsm8k_smoke_no_cache.yaml": ("gemma_gsm8k_smoke_no_cache", "gsm8k_test_manifest.jsonl", "MIT", 16, 1, 2),
        "gemma_arc_challenge_paper.yaml": (
            "gemma_arc_challenge_paper",
            "arc_challenge_validation_manifest.jsonl",
            "CC-BY-SA-4.0",
            32,
            5,
            20,
        ),
        "gemma_arc_challenge_smoke.yaml": (
            "gemma_arc_challenge_smoke",
            "arc_challenge_validation_manifest.jsonl",
            "CC-BY-SA-4.0",
            16,
            1,
            2,
        ),
    }
    for filename, (suite_id, prompt_name, dataset_license, max_new_tokens, warmup, timed) in expectations.items():
        suite = load_suite_config(Path("paper_benchmarks/configs/suites") / filename)
        assert suite.suite_id == suite_id
        assert Path(suite.workload_path).name == prompt_name
        assert suite.dataset_license == dataset_license
        assert suite.synthetic_workload is False
        assert [variant.value for variant in suite.variants] == ["eager", "torch_compile", "kf_cast"]
        assert [stage.value for stage in suite.stages] == ["load", "compile", "warmup", "prefill", "decode", "total_generate"]
        expected_cache_mode = "kv_cache_off" if filename.endswith("_no_cache.yaml") else "kv_cache_on"
        assert suite.cache_mode == expected_cache_mode
        assert suite.max_new_tokens == max_new_tokens
        assert suite.warmup_count == warmup
        assert suite.timed_run_count == timed


def test_gemma_model_configs_point_to_gsm8k_primary_suite():
    prompt_hash = sha256_path(BENCH_DATASETS / "gsm8k_test_manifest.jsonl")
    for filename in ["gemma4_e2b_bf16.yaml", "gemma4_e2b_int4.yaml"]:
        model = load_model_config(Path("paper_benchmarks/configs/models") / filename)
        assert Path(model.validation_suite_path).name == "gemma_gsm8k_paper.yaml"
        assert model.workload_hash == prompt_hash
        assert model.benchmark_expectation == "external_gsm8k_primary_candidate"
