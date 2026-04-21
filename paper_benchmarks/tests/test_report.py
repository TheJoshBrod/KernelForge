from __future__ import annotations

from pathlib import Path

from paper_benchmarks.paper_bench.artifacts import create_run_layout_for_dir, write_json_artifact
from paper_benchmarks.paper_bench.report import (
    SECTION_DEPLOYMENT,
    SECTION_EXPORT,
    SECTION_MODEL,
    SECTION_OPERATOR,
    summarize_run,
)
from paper_benchmarks.paper_bench.schema import BenchmarkArtifact, Variant
from paper_benchmarks.paper_bench.stats import build_latency_summary

from .helpers import build_common_payload


def _sample_records(sample_count: int) -> list[dict]:
    return [
        {
            "sample_index": index,
            "prompt_ids": [f"p{index}"],
            "batch_size": 1,
            "prompt_lengths": [32],
            "generated_lengths": [16],
            "generated_token_count": 16,
            "decode_generated_token_count": 15,
            "output_token_hashes": [f"hash-{index}"],
            "batch_output_hash": f"batch-hash-{index}",
        }
        for index in range(sample_count)
    ]


def _write_metric(
    layout,
    sample_paths,
    *,
    variant: str,
    stage: str,
    benchmark_mode: str = "e2e_model",
    comparison_group: str | None = "paper_group",
    samples: list[float] | None = None,
    correctness_status: str = "passed",
    details: dict | None = None,
    fallback_count: int | None = None,
    kernel_hit_count: int | None = None,
    cast: bool = False,
    kf_artifact_kind: str | None = None,
    model_id: str = "unit_test_model",
    workload_hash: str | None = None,
) -> Path:
    samples = list(samples or [10.0, 10.5, 11.0])
    common = build_common_payload(sample_paths)
    common.update(
        {
            "run_id": layout.run_id,
            "benchmark_mode": benchmark_mode,
            "variant": variant,
            "stage": stage,
            "comparison_group": comparison_group,
            "configured_batch_size": 1,
            "prompt_bucket_id": "bucket_32" if benchmark_mode != "operator" else None,
            "latency_samples_ms": samples,
            "latency_summary": build_latency_summary(samples).model_dump(mode="json"),
            "correctness_status": correctness_status,
            "correctness_message": None,
            "steady_state_time_ms": None if stage == "compile" else float(sum(samples) / len(samples)),
            "compile_time_ms": float(sum(samples) / len(samples)) if stage == "compile" else None,
            "fallback_count": fallback_count,
            "kernel_hit_count": kernel_hit_count,
            "timed_run_count": len(samples),
            "workload_hash": workload_hash or sample_paths["workload_hash"],
            "model_id": model_id,
        }
    )
    if cast:
        common["cast_package_path"] = sample_paths["cast_path"]
        common["cast_package_hash"] = sample_paths["cast_hash"]
        common["kf_artifact_path"] = sample_paths["cast_path"]
        common["kf_artifact_hash"] = sample_paths["cast_hash"]
        common["kf_artifact_kind"] = kf_artifact_kind or "cast"
    else:
        common["cast_package_path"] = None
        common["cast_package_hash"] = None
        common["kf_artifact_path"] = None
        common["kf_artifact_hash"] = None
        common["kf_artifact_kind"] = None
    metric_details = {
        "aggregate_metrics": {"stage_tokens_per_second": 1000.0 / max(common["steady_state_time_ms"] or 1.0, 1.0)},
        "generation_settings": {
            "do_sample": False,
            "max_new_tokens": 16,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "batch_size": 1,
        },
        "selected_prompt_ids": ["p0", "p1", "p2"],
        "selected_prompt_ids_hash": "selected-prompts-hash",
        "prompt_suite_hash": sample_paths["workload_hash"],
        "selection_policy": {
            "method": "frozen_input_order",
            "post_hoc": False,
            "prompt_bucket_id": "bucket_32" if benchmark_mode != "operator" else None,
            "batch_size": 1,
            "timed_run_count": len(samples),
        },
        "per_run_output_hash_verification": benchmark_mode != "operator",
        "correctness_checked_run_count": len(samples) if benchmark_mode != "operator" else 0,
        "runtime_patch_enabled": True if cast else None,
    }
    if cast:
        metric_details.update(
            {
                "cast_manifest": {
                    "project_ref": "project/test_qwen%20-%20NVIDIA%20GB10/",
                    "selection_policy": "auto_best_fastest_valid",
                    "selected_ops": ["torch_nn_functional_softmax"],
                    "selected_kernel_metadata": {
                        "torch_nn_functional_softmax": {
                            "candidate_id": "softmax:fastest_valid",
                            "kernel_source_path": "/tmp/softmax.cu",
                            "selected_source_hash": "softmax-source-hash",
                            "benchmark_reference": {
                                "artifact_path": "/tmp/op_benchmarks.json",
                                "row_ref": "torch_nn_functional_softmax#0",
                            },
                            "evidence_tier": "deployment",
                        }
                    },
                    "rejected_candidate_summary": {
                        "torch_nn_functional_softmax": {
                            "total": 2,
                            "reasons": {
                                "correctness failed": 1,
                                "missing captured inputs": 1,
                            },
                        }
                    },
                    "export_paper_eligible": True,
                },
                "project_ref": "project/test_qwen%20-%20NVIDIA%20GB10/",
                "loaded_kernels": ["torch_nn_functional_softmax"],
                "precompiled_vs_jit_path": {"torch_nn_functional_softmax": "jit"},
                "setup_time_ms": 3.25,
                "jit_compile_time_ms": 1.75,
                "precompiled_load_time_ms": 0.0,
                "exception_fallback_count": 0,
                "contiguous_copy_count": 0,
                "adaptation_count": 0,
            }
        )
    if benchmark_mode == "operator":
        metric_details.update(
            {
                "coverage": {
                    "entry_count": 2,
                    "unique_shape_count": 1,
                    "dtype_coverage": ["torch.float32"],
                },
                "claim_scope": "deployment_operator" if cast and kf_artifact_kind == "cast" else "micro_operator" if cast else "baseline",
            }
        )
    if details:
        metric_details.update(details)
    artifact = BenchmarkArtifact(
        **common,
        artifact_type="benchmark_result",
        sample_records=_sample_records(len(samples)),
        prompt_id=None,
        prompt_hash=None,
        token_count=48 if benchmark_mode != "operator" else None,
        details=metric_details,
    )
    file_name = f"{variant}_{stage}"
    if comparison_group:
        file_name = f"{file_name}__{comparison_group}"
    path = layout.metrics_dir / f"{file_name}.json"
    write_json_artifact(path, artifact)
    return path


def test_report_refuses_model_speedup_claim_without_torch_compile(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[8.0, 8.5, 9.0],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("missing torch_compile baseline" in claim for claim in summary.forbidden_claims)


def test_report_refuses_claim_when_correctness_fails(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[7.0, 7.5, 8.0])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[5.5, 6.0, 6.5],
        correctness_status="failed",
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert any("Unsafe speedup; not a valid model-speedup result." in claim for claim in summary.paper_eligible_claims)
    assert any("do not exactly match eager" in claim for claim in summary.forbidden_claims)
    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)


def test_report_refuses_claim_when_fallback_hidden_or_nonzero(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[7.0, 7.2, 7.4])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[6.0, 6.2, 6.4],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=2,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("fallback count is nonzero" in claim for claim in summary.forbidden_claims)


def test_report_distinguishes_eager_win_from_compile_win(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference", samples=[12.0, 12.5, 13.0])
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[8.0, 8.1, 8.2])
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

    assert any("Kernel Forge beats eager but does not beat torch.compile on this workload." in claim for claim in summary.paper_eligible_claims)
    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("kf_cast does not beat torch.compile" in claim for claim in summary.forbidden_claims)


def test_report_separates_operator_and_model_results(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[7.0, 7.2, 7.4])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[7.5, 7.7, 7.9],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )
    _write_metric(layout, sample_paths, variant="eager", stage="operator", benchmark_mode="operator", comparison_group="aten.softmax", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="operator", benchmark_mode="operator", comparison_group="aten.softmax", samples=[5.0, 5.1, 5.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="operator",
        benchmark_mode="operator",
        comparison_group="aten.softmax",
        samples=[4.0, 4.1, 4.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=12,
    )

    summary = summarize_run(layout.run_dir)

    assert summary.sections[SECTION_OPERATOR]["rows"]
    assert summary.sections[SECTION_MODEL]["rows"]
    assert all(row["stage"] == "operator" for row in summary.sections[SECTION_OPERATOR]["rows"])
    assert all(row["stage"] in {"prefill", "decode", "total_generate"} for row in summary.sections[SECTION_MODEL]["rows"])


def test_report_deployment_section_excludes_operator_only_rows(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[7.0, 7.2, 7.4])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[6.0, 6.1, 6.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="operator",
        benchmark_mode="operator",
        comparison_group="aten.softmax",
        samples=[4.0, 4.1, 4.2],
        cast=True,
        kf_artifact_kind="direct_source",
        fallback_count=0,
        kernel_hit_count=12,
    )

    summary = summarize_run(layout.run_dir)

    assert summary.sections[SECTION_DEPLOYMENT]["rows"]
    assert all(row["stage"] != "operator" for row in summary.sections[SECTION_DEPLOYMENT]["rows"])


def test_report_includes_regressions(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference", samples=[8.0, 8.1, 8.2])
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[6.0, 6.1, 6.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[9.0, 9.1, 9.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert any("regressed against eager" in item for item in summary.failure_regressions)
    assert any("regressed against torch.compile" in item for item in summary.failure_regressions)


def test_report_includes_raw_artifact_paths(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    metric_path = _write_metric(layout, sample_paths, variant="eager", stage="total_generate", correctness_status="reference")
    _write_metric(layout, sample_paths, variant="torch_compile", stage="total_generate", samples=[7.0, 7.1, 7.2])
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[6.0, 6.1, 6.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
    )

    summary = summarize_run(layout.run_dir)

    assert str(metric_path.resolve()) in summary.raw_artifact_paths
    assert all(Path(path).exists() for path in summary.raw_artifact_paths)
    assert all(row.artifact_path for row in summary.rows)
    assert Path(summary.summary_markdown_path).exists()
    assert Path(summary.summary_csv_path).exists()


def test_report_includes_export_selection_section(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(
        layout,
        sample_paths,
        variant="eager",
        stage="total_generate",
        correctness_status="reference",
        model_id="qwen35a3b",
    )
    _write_metric(
        layout,
        sample_paths,
        variant="torch_compile",
        stage="total_generate",
        samples=[7.0, 7.1, 7.2],
        model_id="qwen35a3b",
    )
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[6.0, 6.1, 6.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
        model_id="qwen35a3b",
    )

    summary = summarize_run(layout.run_dir)

    export_items = summary.sections[SECTION_EXPORT]["items"]
    assert export_items
    assert export_items[0]["selection_policy"] == "auto_best_fastest_valid"
    assert export_items[0]["project_ref"] == "project/test_qwen%20-%20NVIDIA%20GB10/"
    assert export_items[0]["selected_source_hashes"]["torch_nn_functional_softmax"] == "softmax-source-hash"
    assert export_items[0]["evidence_tiers"]["torch_nn_functional_softmax"] == "deployment"
    assert export_items[0]["rejected_export_candidate_summary"]["torch_nn_functional_softmax"]["reasons"]["correctness failed"] == 1


def test_report_refuses_qwen_claim_without_required_selection_policy(sample_paths, tmp_path: Path):
    layout = create_run_layout_for_dir(tmp_path / "run")
    _write_metric(
        layout,
        sample_paths,
        variant="eager",
        stage="total_generate",
        correctness_status="reference",
        model_id="qwen35a3b",
    )
    _write_metric(
        layout,
        sample_paths,
        variant="torch_compile",
        stage="total_generate",
        samples=[7.0, 7.1, 7.2],
        model_id="qwen35a3b",
    )
    _write_metric(
        layout,
        sample_paths,
        variant="kf_cast",
        stage="total_generate",
        samples=[6.0, 6.1, 6.2],
        cast=True,
        kf_artifact_kind="cast",
        fallback_count=0,
        kernel_hit_count=9,
        model_id="qwen35a3b",
        details={
            "cast_manifest": {
                "project_ref": "project/test_qwen%20-%20NVIDIA%20GB10/",
                "selection_policy": "manual_override",
                "selected_ops": ["torch_nn_functional_softmax"],
                "selected_kernel_metadata": {
                    "torch_nn_functional_softmax": {
                        "candidate_id": "softmax:manual",
                        "kernel_source_path": "/tmp/softmax_manual.cu",
                        "selected_source_hash": "softmax-manual-hash",
                        "evidence_tier": "manual_override",
                    }
                },
                "rejected_candidate_summary": {},
                "export_paper_eligible": False,
            },
        },
    )

    summary = summarize_run(layout.run_dir)

    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("export selection policy was not auto_best_fastest_valid" in claim for claim in summary.forbidden_claims)
