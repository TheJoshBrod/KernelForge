from __future__ import annotations

import json
from pathlib import Path

from paper_benchmarks.paper_bench.provenance import sha256_path
from paper_benchmarks.paper_bench.llm_runner import load_prompt_records
from paper_benchmarks.paper_bench.registry import load_model_config, load_suite_config


def _assert_real_or_explicitly_non_paper(
    raw_path: str | None,
    *,
    notes: list[str],
    paper_eligible: bool | None = None,
) -> None:
    assert raw_path, "expected a non-empty path"
    candidate = Path(str(raw_path))
    if candidate.exists():
        return

    marker_text = " ".join(notes).lower()
    path_text = str(raw_path).lower()
    placeholder_markers = ("/path/to/", "/abs/path/to/", "placeholder", "<fill-me-in>")
    assert (
        any(marker in path_text for marker in placeholder_markers)
        or "placeholder" in marker_text
        or "non-paper" in marker_text
        or "not a paper-result configuration" in marker_text
        or paper_eligible is False
    ), f"path does not exist and is not clearly marked as placeholder/non-paper: {raw_path}"


def test_qwen_model_config_parses_and_uses_real_or_explicitly_non_paper_paths():
    config_path = Path("paper_benchmarks/configs/models/qwen35a3b.yaml")
    model_spec = load_model_config(config_path)

    assert model_spec.model_id == "qwen35a3b"
    assert model_spec.loader_kind == "transformers_causal_lm"
    assert model_spec.torch_dtype == "bf16"
    assert model_spec.local_files_only is True
    assert model_spec.trust_remote_code is False
    assert model_spec.attn_implementation == "eager"
    assert model_spec.device_map == "auto"
    assert model_spec.device == "cuda"
    assert model_spec.validation_suite_path
    assert Path(model_spec.validation_suite_path).name == "qwen_llm_paper.yaml"
    assert model_spec.batch_sizes == [1]

    for raw_path in (
        model_spec.model_path,
        model_spec.tokenizer_path,
        model_spec.model_config_path,
        model_spec.cast_package_path,
        model_spec.deployment_artifact_path,
    ):
        _assert_real_or_explicitly_non_paper(
            raw_path,
            notes=model_spec.notes,
            paper_eligible=model_spec.paper_eligible,
        )

    if model_spec.model_config_path and Path(model_spec.model_config_path).exists():
        assert model_spec.expected_model_config_hash == sha256_path(model_spec.model_config_path)

    prompt_path = Path("paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.jsonl")
    if prompt_path.exists():
        assert model_spec.workload_hash == sha256_path(prompt_path)


def test_qwen_paper_suite_config_parses_with_conservative_defaults():
    config_path = Path("paper_benchmarks/configs/suites/qwen_llm_paper.yaml")
    suite = load_suite_config(config_path)

    assert suite.suite_id == "qwen_llm_paper"
    assert suite.workload_path
    _assert_real_or_explicitly_non_paper(suite.workload_path, notes=suite.notes)
    assert [variant.value for variant in suite.variants] == ["eager", "torch_compile", "kf_cast"]
    assert [stage.value for stage in suite.stages] == ["load", "compile", "warmup", "prefill", "decode", "total_generate"]
    assert Path(suite.workload_path).name == "qwen_paper_prompts_v1.jsonl"
    assert suite.batch_sizes == [1]
    assert suite.batch_size == 1
    assert suite.max_new_tokens == 128
    assert suite.warmup_count == 5
    assert suite.timed_run_count == 20
    assert suite.generation_mode == "greedy"
    assert suite.include_tokenization_in_timing is False
    assert suite.measure_prefill_decode_separately is True
    assert [bucket.bucket_id for bucket in suite.prompt_length_buckets] == ["short", "medium", "long"]


def test_qwen_smoke_suite_config_parses():
    config_path = Path("paper_benchmarks/configs/suites/qwen_llm_smoke.yaml")
    suite = load_suite_config(config_path)

    assert suite.suite_id == "qwen_llm_smoke"
    _assert_real_or_explicitly_non_paper(suite.workload_path, notes=suite.notes)
    assert [variant.value for variant in suite.variants] == ["eager", "torch_compile", "kf_cast"]
    assert Path(suite.workload_path).name == "qwen_paper_prompts_v1.jsonl"
    assert suite.batch_sizes == [1]
    assert suite.batch_size == 1
    assert suite.max_new_tokens == 16
    assert suite.warmup_count == 1
    assert suite.timed_run_count == 2
    assert suite.generation_mode == "greedy"
    assert suite.include_tokenization_in_timing is False
    assert suite.measure_prefill_decode_separately is True
    assert [bucket.bucket_id for bucket in suite.prompt_length_buckets] == ["short", "medium", "long"]


def test_frozen_qwen_prompt_suite_manifest_matches_prompt_file():
    prompt_path = Path("paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.jsonl")
    manifest_path = prompt_path.with_suffix(".manifest.json")

    assert prompt_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_suite = load_prompt_records(prompt_path)
    prompt_ids = [record["id"] for record in prompt_suite.records]

    assert manifest["prompt_file_hash"] == sha256_path(prompt_path)
    assert prompt_suite.manifest_path == str(manifest_path)
    assert prompt_suite.synthetic_workload is False
    assert manifest["synthetic_workload"] is False
    assert manifest["prompt_text_may_be_stored_in_raw_artifacts"] is False
    assert manifest["bucket_counts"] == {"short": 20, "medium": 20, "long": 20}
    assert len(prompt_ids) == 60
    assert len(prompt_ids) == len(set(prompt_ids))
    assert prompt_ids[:4] == [
        "mt_bench_q081_t1",
        "mt_bench_q081_t2",
        "mt_bench_q082_t1",
        "mt_bench_q082_t2",
    ]
