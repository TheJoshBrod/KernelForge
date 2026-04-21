from __future__ import annotations
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from paper_benchmarks.paper_bench.artifacts import create_run_layout_for_dir, load_json_artifact, write_json_artifact
from paper_benchmarks.paper_bench.cache import validate_reusable_artifact_payload
from paper_benchmarks.paper_bench.llm_runner import run_llm_benchmark
from paper_benchmarks.paper_bench.op_runner import run_operator_benchmark
from paper_benchmarks.paper_bench.report import summarize_run
from paper_benchmarks.paper_bench.schema import BenchmarkArtifact, Variant, validate_artifact_payload
from paper_benchmarks.paper_bench.stats import build_latency_summary

from .helpers import build_common_payload
from .test_cache import _clone_llm_context
from .test_llm_baselines import _make_compile_fn, _make_context, _toy_model_loader
from .test_op_runner import _make_fake_kf_loader, _make_op_context, _write_entry
from .test_report import _write_metric


def _llm_benchmark_payload(sample_paths, **updates):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "benchmark_result",
            "stage": "total_generate",
            "latency_summary": build_latency_summary(payload["latency_samples_ms"]).model_dump(mode="json"),
            "sample_records": [
                {
                    "sample_index": 0,
                    "prompt_ids": ["p0"],
                    "batch_size": 1,
                    "prompt_lengths": [4],
                    "generated_lengths": [3],
                    "generated_token_count": 3,
                    "decode_generated_token_count": 2,
                    "output_token_hashes": ["hash-0"],
                    "batch_output_hash": "batch-hash-0",
                },
                {
                    "sample_index": 1,
                    "prompt_ids": ["p1"],
                    "batch_size": 1,
                    "prompt_lengths": [4],
                    "generated_lengths": [3],
                    "generated_token_count": 3,
                    "decode_generated_token_count": 2,
                    "output_token_hashes": ["hash-1"],
                    "batch_output_hash": "batch-hash-1",
                },
            ],
            "prompt_id": None,
            "prompt_hash": None,
            "token_count": 6,
            "details": {
                "prompt_suite_hash": sample_paths["workload_hash"],
                "generation_settings": {
                    "do_sample": False,
                    "max_new_tokens": 3,
                    "pad_token_id": 0,
                    "eos_token_id": 2,
                    "batch_size": 1,
                },
                "selection_policy": {
                    "method": "frozen_input_order",
                    "post_hoc": False,
                    "prompt_bucket_id": "all",
                    "batch_size": 1,
                    "timed_run_count": 2,
                },
                "per_run_output_hash_verification": True,
                "correctness_checked_run_count": 2,
            },
        }
    )
    payload.update(updates)
    return payload


def test_attack_compare_only_against_eager_cannot_claim_torch_compile_win(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[7.0, 7.1, 7.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("missing torch_compile baseline" in claim for claim in summary.forbidden_claims)


def test_attack_hidden_kf_fallback_blocks_claim(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[8.0, 8.1, 8.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[6.0, 6.1, 6.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=None,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("fallback count was not reported" in claim for claim in summary.forbidden_claims)


def test_attack_token_mismatch_cannot_hide_behind_aggregate_metrics(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference", samples=[12.0, 12.5, 13.0])
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[9.0, 9.1, 9.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[4.0, 4.1, 4.2],
        correctness_status="failed",
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert any("Unsafe speedup; not a valid model-speedup result." in claim for claim in summary.paper_eligible_claims)
    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)


def test_attack_warmup_is_not_included_in_steady_state_latency(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.eager)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
    )

    warmup_artifact = load_json_artifact(layout.metrics_dir / "eager_warmup.json")
    total_artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert len(warmup_artifact.latency_samples_ms) == suite.warmup_count
    assert len(total_artifact.latency_samples_ms) == suite.timed_run_count
    assert total_artifact.latency_samples_ms != warmup_artifact.latency_samples_ms


def test_attack_compile_time_is_not_hidden_in_steady_state_latency(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.torch_compile)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=_make_compile_fn(),
    )

    compile_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_compile.json")
    total_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_total_generate.json")
    assert compile_artifact.compile_time_ms == compile_artifact.latency_samples_ms[0]
    assert all(sample != compile_artifact.compile_time_ms for sample in total_artifact.latency_samples_ms)


def test_attack_missing_prompt_suite_hash_marks_non_paper(sample_paths):
    artifact = validate_artifact_payload(_llm_benchmark_payload(sample_paths, workload_hash=None, details={"prompt_suite_hash": None}))
    assert artifact.paper_eligible is False
    assert "workload hash missing" in artifact.paper_eligibility_issues


def test_attack_missing_model_hash_or_config_fails_validation(sample_paths):
    with pytest.raises(ValidationError):
        validate_artifact_payload(_llm_benchmark_payload(sample_paths, model_config_hash=None))


def test_attack_synthetic_prompt_marks_paper_run_invalid(sample_paths):
    artifact = validate_artifact_payload(_llm_benchmark_payload(sample_paths, synthetic_workload=True))
    assert artifact.paper_eligible is False
    assert "synthetic workload used" in artifact.paper_eligibility_issues


def test_attack_cache_reuse_is_blocked_after_prompt_change(sample_paths, tmp_path: Path):
    source = _make_context(tmp_path / "shared", sample_paths, Variant.eager)
    run_llm_benchmark(
        layout=source[0],
        common_fields=source[1],
        env_artifact=source[2],
        manifest_artifact=source[3],
        model_spec=source[4],
        suite=source[5],
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        reuse_cache=False,
        cache_search_root=tmp_path,
    )
    target = _clone_llm_context(source, tmp_path / "changed_prompt")
    target[1]["workload_hash"] = "prompt-hash-changed"

    run_llm_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        model_spec=target[4],
        suite=target[5],
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(target[0].metrics_dir / "eager_total_generate.json")
    assert artifact.reused is False


def test_attack_cache_reuse_is_blocked_after_cast_change(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", tensor=torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", tensor=torch.tensor([[3.0, 4.0]]))

    source = _make_op_context(
        tmp_path / "source",
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "source.cast"),
    )
    run_operator_benchmark(
        layout=source[0],
        common_fields=source[1],
        env_artifact=source[2],
        manifest_artifact=source[3],
        suite=source[4],
        variant=Variant.kf_cast,
        kf_loader=_make_fake_kf_loader(),
        reuse_cache=False,
        cache_search_root=tmp_path,
    )

    target = _make_op_context(
        tmp_path / "target",
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "target.cast"),
    )
    run_operator_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        suite=target[4],
        variant=Variant.kf_cast,
        kf_loader=_make_fake_kf_loader(),
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(target[0].metrics_dir / "kf_cast_operator.json")
    assert artifact.reused is False


def test_attack_operator_result_cannot_be_presented_as_deployment(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 4.0]]))
    layout, common_fields, env, manifest, suite = _make_op_context(
        tmp_path,
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "softmax_kernel.cu"),
    )

    run_operator_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        suite=suite,
        variant=Variant.kf_cast,
        kf_loader=_make_fake_kf_loader(artifact_kind="direct_source"),
    )

    artifact = load_json_artifact(layout.metrics_dir / "kf_cast_operator.json")
    assert artifact.details["claim_scope"] == "micro_operator"
    assert artifact.details["deployment_comparable"] is False


def test_attack_micro_win_cannot_be_presented_as_model_win(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference", samples=[8.0, 8.1, 8.2])
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[6.0, 6.1, 6.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[7.0, 7.1, 7.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )
    _write_metric(layout, sample_paths, variant="eager", stage="operator", benchmark_mode="operator", comparison_group="aten.softmax", correctness_status="reference", samples=[5.0, 5.1, 5.2])
    _write_metric(layout, sample_paths, variant="torch_compile", stage="operator", benchmark_mode="operator", comparison_group="aten.softmax", samples=[4.0, 4.1, 4.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="operator",
        benchmark_mode="operator",
        comparison_group="aten.softmax",
        samples=[2.0, 2.1, 2.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=12,
    )

    summary = summarize_run(layout.run_dir)

    assert any("Operator wins did not translate into an end-to-end model win." in claim for claim in summary.paper_eligible_claims)
    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)


def test_attack_missing_raw_samples_cannot_be_reused(sample_paths):
    artifact = BenchmarkArtifact(
        **_llm_benchmark_payload(sample_paths, sample_records=[]),
    )

    with pytest.raises(ValueError, match="results with no raw samples"):
        validate_reusable_artifact_payload(artifact.model_dump(mode="json"))


def test_attack_missing_git_commit_marks_non_paper(sample_paths):
    artifact = validate_artifact_payload(_llm_benchmark_payload(sample_paths, git_commit="unknown"))
    assert artifact.paper_eligible is False
    assert "git commit unknown" in artifact.paper_eligibility_issues


def test_attack_non_paper_run_is_not_reported_as_paper_claim(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[8.0, 8.1, 8.2])
    kf_path = _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[5.0, 5.1, 5.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )
    kf_artifact = load_json_artifact(kf_path).model_copy(
        update={
            "paper_eligible": False,
            "paper_eligibility_issues": ["synthetic workload used"],
            "synthetic_workload": True,
        }
    )
    write_json_artifact(kf_path, kf_artifact)

    summary = summarize_run(layout.run_dir)

    assert summary.paper_eligible is False
    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)


def test_attack_kf_cast_source_or_package_hash_missing_fails_validation(sample_paths):
    with pytest.raises(ValidationError):
        validate_artifact_payload(
            _llm_benchmark_payload(
                sample_paths,
                variant="kf_cast",
                cast_package_path=None,
                cast_package_hash=None,
                kf_artifact_path=sample_paths["cast_path"],
                kf_artifact_hash=None,
                fallback_count=0,
                kernel_hit_count=1,
                details={
                    "prompt_suite_hash": sample_paths["workload_hash"],
                    "selection_policy": {"method": "frozen_input_order", "post_hoc": False},
                    "per_run_output_hash_verification": True,
                    "correctness_checked_run_count": 2,
                },
            )
        )


def test_attack_torch_compile_failure_is_not_silently_replaced_by_eager(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.torch_compile)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=_make_compile_fn(fail_on_prefill_run=1),
    )

    compile_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_compile.json")
    assert compile_artifact.details["execution_status"] == "failed"
    assert not (layout.metrics_dir / "torch_compile_total_generate.json").exists()


def test_attack_longest_prompt_only_post_hoc_selection_is_non_paper(sample_paths):
    artifact = validate_artifact_payload(
        _llm_benchmark_payload(
            sample_paths,
            details={
                "prompt_suite_hash": sample_paths["workload_hash"],
                "selection_policy": {
                    "method": "longest_prompt_only_post_hoc",
                    "post_hoc": True,
                },
                "per_run_output_hash_verification": True,
                "correctness_checked_run_count": 2,
            },
        )
    )
    assert artifact.paper_eligible is False
    assert "post-hoc longest-prompt-only selection is not paper eligible" in artifact.paper_eligibility_issues


def test_attack_per_run_output_hashes_must_be_checked(sample_paths):
    artifact = validate_artifact_payload(
        _llm_benchmark_payload(
            sample_paths,
            variant="torch_compile",
            correctness_status="passed",
            sample_records=[
                {
                    "sample_index": 0,
                    "prompt_ids": ["p0"],
                    "batch_size": 1,
                    "prompt_lengths": [4],
                    "generated_lengths": [3],
                    "generated_token_count": 3,
                    "decode_generated_token_count": 2,
                    "output_token_hashes": [],
                    "batch_output_hash": "batch-hash-0",
                },
                {
                    "sample_index": 1,
                    "prompt_ids": ["p1"],
                    "batch_size": 1,
                    "prompt_lengths": [4],
                    "generated_lengths": [3],
                    "generated_token_count": 3,
                    "decode_generated_token_count": 2,
                    "output_token_hashes": [],
                    "batch_output_hash": "batch-hash-1",
                },
            ],
            details={
                "prompt_suite_hash": sample_paths["workload_hash"],
                "selection_policy": {"method": "frozen_input_order", "post_hoc": False},
                "per_run_output_hash_verification": False,
                "correctness_checked_run_count": 1,
            },
        )
    )
    assert artifact.paper_eligible is False
    assert "per-run output token hashes missing" in artifact.paper_eligibility_issues
    assert "per-run output hash verification missing" in artifact.paper_eligibility_issues
    assert "per-run correctness check count mismatch" in artifact.paper_eligibility_issues


def test_attack_regressions_are_not_omitted_from_report(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference", samples=[8.0, 8.1, 8.2])
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[6.0, 6.1, 6.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[10.0, 10.1, 10.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert any("regressed against eager" in item for item in summary.failure_regressions)
    assert any("regressed against torch.compile" in item for item in summary.failure_regressions)
