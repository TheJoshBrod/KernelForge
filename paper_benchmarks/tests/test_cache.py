from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from paper_benchmarks.paper_bench.artifacts import create_run_layout_for_dir, load_json_artifact
from paper_benchmarks.paper_bench.cache import (
    CacheValidationError,
    find_matching_reusable_artifact,
    make_cache_request,
    validate_reusable_artifact_payload,
)
from paper_benchmarks.paper_bench.llm_runner import run_llm_benchmark
from paper_benchmarks.paper_bench.op_runner import run_operator_benchmark
from paper_benchmarks.paper_bench.provenance import sha256_path
from paper_benchmarks.paper_bench.schema import BenchmarkArtifact, SummaryArtifact, Variant
from paper_benchmarks.tests.helpers import build_common_payload
from paper_benchmarks.tests.test_llm_baselines import _make_compile_fn, _make_context, _toy_model_loader
from paper_benchmarks.tests.test_kf_runtime import _configure_kf_context, _make_fake_cast_loader
from paper_benchmarks.tests.test_op_runner import _make_fake_kf_loader, _make_op_context, _write_entry


def _clone_llm_context(source_context, run_dir: Path):
    layout = create_run_layout_for_dir(run_dir)
    common_fields = dict(source_context[1])
    common_fields["run_id"] = layout.run_id
    env = source_context[2].model_copy(update={"run_id": layout.run_id})
    manifest = source_context[3].model_copy(update={"run_id": layout.run_id, "run_dir": str(layout.run_dir)})
    return layout, common_fields, env, manifest, source_context[4], source_context[5]


def test_matching_cache_is_reused_for_llm(sample_paths, tmp_path: Path):
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
    target = _clone_llm_context(source, tmp_path / "reused_run")
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
    reused = load_json_artifact(target[0].metrics_dir / "eager_total_generate.json")

    assert reused.reused is True
    assert reused.reused_from_artifact is not None
    assert reused.reused_from_artifact_hash == sha256_path(reused.reused_from_artifact)
    assert reused.reused_from_artifact.endswith("eager_total_generate.json")


def test_default_no_reuse(sample_paths, tmp_path: Path):
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
    target = _clone_llm_context(source, tmp_path / "fresh_run")
    run_llm_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        model_spec=target[4],
        suite=target[5],
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        reuse_cache=False,
        cache_search_root=tmp_path,
    )
    fresh = load_json_artifact(target[0].metrics_dir / "eager_total_generate.json")

    assert fresh.reused is False
    assert fresh.reused_from_artifact is None


def test_changed_prompt_hash_invalidates_cache(sample_paths, tmp_path: Path):
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
    layout, common_fields, env, manifest, model_spec, suite = _clone_llm_context(source, tmp_path / "changed_prompt_run")
    common_fields["workload_hash"] = "changed-prompt-hash"

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert artifact.reused is False


def test_changed_model_config_invalidates_cache(sample_paths, tmp_path: Path):
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
    layout, common_fields, env, manifest, model_spec, suite = _clone_llm_context(source, tmp_path / "changed_model_run")
    common_fields["model_config_hash"] = "changed-model-config-hash"

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert artifact.reused is False


def test_changed_compile_settings_invalidates_cache(sample_paths, tmp_path: Path):
    source = _make_context(tmp_path / "shared", sample_paths, Variant.torch_compile)
    run_llm_benchmark(
        layout=source[0],
        common_fields=source[1],
        env_artifact=source[2],
        manifest_artifact=source[3],
        model_spec=source[4],
        suite=source[5],
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=_make_compile_fn(),
        reuse_cache=False,
        cache_search_root=tmp_path,
    )

    target = _clone_llm_context(source, tmp_path / "changed_compile_run")
    target[1]["compile_settings"] = {
        **target[1]["compile_settings"],
        "dynamic": True,
    }
    run_llm_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        model_spec=target[4],
        suite=target[5],
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=_make_compile_fn(),
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(target[0].metrics_dir / "torch_compile_total_generate.json")
    assert artifact.reused is False


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("cuda_version", "99.0"),
        ("pytorch_version", "0.0.0-test"),
    ],
)
def test_changed_runtime_fields_invalidate_cache(sample_paths, tmp_path: Path, field_name: str, new_value: str):
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
    layout, common_fields, env, manifest, model_spec, suite = _clone_llm_context(source, tmp_path / "changed_runtime_run")
    common_fields[field_name] = new_value

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert artifact.reused is False


def test_changed_cast_hash_invalidates_cache(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", tensor=torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", tensor=torch.tensor([[3.0, 4.0]]))

    source = _make_op_context(
        tmp_path / "source",
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "source_cast" / "softmax.cast"),
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
        kernel_source_or_cast=str(tmp_path / "target_cast" / "softmax.cast"),
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


def test_changed_selected_kernel_hash_invalidates_cache(sample_paths, tmp_path: Path):
    source = _make_context(tmp_path / "source", sample_paths, Variant.kf_cast)
    _configure_kf_context(source[1], source[4], sample_paths)
    source[1]["kf_settings"]["project_ref"] = "project/test_qwen%20-%20NVIDIA%20GB10/"
    run_llm_benchmark(
        layout=source[0],
        common_fields=source[1],
        env_artifact=source[2],
        manifest_artifact=source[3],
        model_spec=source[4],
        suite=source[5],
        variant=Variant.kf_cast,
        model_loader=_toy_model_loader,
        cast_loader=_make_fake_cast_loader(
            selected_source_hash="selected-source-hash-a",
            project_ref="project/test_qwen%20-%20NVIDIA%20GB10/",
        ),
        reuse_cache=False,
        cache_search_root=tmp_path,
    )

    target = _clone_llm_context(source, tmp_path / "target")
    target[1]["kf_settings"]["project_ref"] = "project/test_qwen%20-%20NVIDIA%20GB10/"
    run_llm_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        model_spec=target[4],
        suite=target[5],
        variant=Variant.kf_cast,
        model_loader=_toy_model_loader,
        cast_loader=_make_fake_cast_loader(
            selected_source_hash="selected-source-hash-b",
            project_ref="project/test_qwen%20-%20NVIDIA%20GB10/",
        ),
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(target[0].metrics_dir / "kf_cast_total_generate.json")
    assert artifact.reused is False


def test_changed_project_ref_invalidates_cache(sample_paths, tmp_path: Path):
    source = _make_context(tmp_path / "source", sample_paths, Variant.kf_cast)
    _configure_kf_context(source[1], source[4], sample_paths)
    source[1]["kf_settings"]["project_ref"] = "project/test_qwen%20-%20NVIDIA%20GB10/"
    run_llm_benchmark(
        layout=source[0],
        common_fields=source[1],
        env_artifact=source[2],
        manifest_artifact=source[3],
        model_spec=source[4],
        suite=source[5],
        variant=Variant.kf_cast,
        model_loader=_toy_model_loader,
        cast_loader=_make_fake_cast_loader(
            selected_source_hash="selected-source-hash-a",
            project_ref="project/test_qwen%20-%20NVIDIA%20GB10/",
        ),
        reuse_cache=False,
        cache_search_root=tmp_path,
    )

    target = _clone_llm_context(source, tmp_path / "target")
    target[1]["kf_settings"]["project_ref"] = "project/other_qwen_project/"
    run_llm_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        model_spec=target[4],
        suite=target[5],
        variant=Variant.kf_cast,
        model_loader=_toy_model_loader,
        cast_loader=_make_fake_cast_loader(
            selected_source_hash="selected-source-hash-a",
            project_ref="project/other_qwen_project/",
        ),
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(target[0].metrics_dir / "kf_cast_total_generate.json")
    assert artifact.reused is False


def test_changed_kf_runtime_settings_invalidate_cache(sample_paths, tmp_path: Path):
    source = _make_context(tmp_path / "source", sample_paths, Variant.kf_cast)
    _configure_kf_context(source[1], source[4], sample_paths)
    source[1]["kf_settings"]["project_ref"] = "project/test_qwen%20-%20NVIDIA%20GB10/"
    run_llm_benchmark(
        layout=source[0],
        common_fields=source[1],
        env_artifact=source[2],
        manifest_artifact=source[3],
        model_spec=source[4],
        suite=source[5],
        variant=Variant.kf_cast,
        model_loader=_toy_model_loader,
        cast_loader=_make_fake_cast_loader(
            selected_source_hash="selected-source-hash-a",
            project_ref="project/test_qwen%20-%20NVIDIA%20GB10/",
        ),
        reuse_cache=False,
        cache_search_root=tmp_path,
    )

    target = _clone_llm_context(source, tmp_path / "target")
    target[1]["kf_settings"]["allow_jit"] = False
    run_llm_benchmark(
        layout=target[0],
        common_fields=target[1],
        env_artifact=target[2],
        manifest_artifact=target[3],
        model_spec=target[4],
        suite=target[5],
        variant=Variant.kf_cast,
        model_loader=_toy_model_loader,
        cast_loader=_make_fake_cast_loader(
            selected_source_hash="selected-source-hash-a",
            project_ref="project/test_qwen%20-%20NVIDIA%20GB10/",
        ),
        reuse_cache=True,
        cache_search_root=tmp_path,
    )

    artifact = load_json_artifact(target[0].metrics_dir / "kf_cast_total_generate.json")
    assert artifact.reused is False


def test_no_raw_samples_means_no_reuse(sample_paths, tmp_path: Path):
    payload = build_common_payload(sample_paths)
    artifact = BenchmarkArtifact(
        **payload,
        artifact_type="benchmark_result",
        latency_summary={"count": 2, "mean_ms": 1.5, "median_ms": 1.5},
        sample_records=[],
        details={},
    )
    metrics_path = tmp_path / "old_run" / "metrics" / "eager_prefill.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")

    request = make_cache_request(
        payload,
        variant=Variant.eager,
        stage=artifact.stage,
        sample_matrix=None,
    )
    assert find_matching_reusable_artifact(tmp_path, request) is None


def test_summary_only_results_cannot_be_reused(sample_paths):
    payload = build_common_payload(sample_paths)
    summary = SummaryArtifact(
        **{**payload, "variant": None, "stage": None, "latency_samples_ms": [], "correctness_status": "not_applicable"},
        artifact_type="summary_report",
        rows=[],
        summary_markdown_path=None,
    )

    with pytest.raises(CacheValidationError):
        validate_reusable_artifact_payload(summary.model_dump(mode="json"))


def test_legacy_numeric_results_cannot_be_reused():
    with pytest.raises(CacheValidationError):
        validate_reusable_artifact_payload(1.0)
