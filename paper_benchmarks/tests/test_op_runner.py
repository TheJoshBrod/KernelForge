from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from paper_benchmarks.paper_bench.artifacts import create_run_layout, load_json_artifact
from paper_benchmarks.paper_bench.op_runner import (
    load_operator_entries,
    resolve_project_operator_export_evidence,
    run_operator_benchmark,
)
from paper_benchmarks.paper_bench.provenance import build_environment_artifact_fields, collect_common_fields
from paper_benchmarks.paper_bench.report import summarize_run
from paper_benchmarks.paper_bench.schema import EnvironmentArtifact, RunManifestArtifact, Variant
from paper_benchmarks.paper_bench.provenance import safe_sha256_path


def _write_entry(root: Path, name: str, tensor: torch.Tensor, *, op_name: str = "aten.softmax") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    torch.save(
        {
            "op_name": op_name,
            "args": [tensor],
            "kwargs": {"dim": -1},
        },
        path,
    )
    return path


def _make_op_context(
    tmp_path: Path,
    sample_paths: dict[str, str],
    *,
    variant: Variant,
    entries_dir: Path,
    kernel_source_or_cast: str | None = None,
    project_ref: str | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries_dir.mkdir(parents=True, exist_ok=True)
    if kernel_source_or_cast:
        kernel_path = Path(kernel_source_or_cast)
        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        if not kernel_path.exists():
            if kernel_path.suffix == ".cu":
                kernel_path.write_text("// fake kernel\n", encoding="utf-8")
            else:
                kernel_path.write_bytes(b"fake-artifact")
    suite_path = tmp_path / "operator_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_id": "operator_softmax",
                "benchmark_mode": "operator",
                "workload_type": "operator_entries",
                "workload_path": str(entries_dir),
                "op_name": "aten.softmax",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common = collect_common_fields(
        repo_root=Path(__file__).resolve().parents[2],
        model_id="operator_softmax",
        model_path=kernel_source_or_cast or sample_paths["model_path"],
        model_config_path=sample_paths["model_config_path"],
        suite_id="operator_softmax",
        suite_path=str(suite_path),
        workload_path=str(entries_dir),
        command_line=["python", "-m", "paper_benchmarks.paper_bench.cli", "run-ops"],
        paper_eligible=True,
        synthetic_workload=False,
        cast_package_path=kernel_source_or_cast if kernel_source_or_cast and kernel_source_or_cast.endswith(".cast") else None,
        exported_kernel_paths=[kernel_source_or_cast] if kernel_source_or_cast and not kernel_source_or_cast.endswith(".cast") else None,
    )
    common["compile_settings"] = {
        "backend": "inductor",
        "mode": None,
        "fullgraph": False,
        "dynamic": False,
    }
    common["kf_settings"] = {
        "cast_package_path": kernel_source_or_cast if kernel_source_or_cast and kernel_source_or_cast.endswith(".cast") else None,
        "kernel_source_or_cast": kernel_source_or_cast,
        "require_precompiled": False,
        "allow_jit": True,
        "fail_on_fallback": True,
        "record_runtime_stats": True,
    }
    if project_ref:
        common["kf_settings"]["project_ref"] = project_ref
    layout = create_run_layout(tmp_path / "runs", common["timestamp_utc"], "operator_softmax", "operator_softmax")
    common["run_id"] = layout.run_id
    manifest = RunManifestArtifact(
        **common,
        artifact_type="run_manifest",
        benchmark_mode="operator",
        variant=None,
        stage=None,
        warmup_count=1,
        timed_run_count=2,
        latency_samples_ms=[],
        correctness_status="not_applicable",
        run_dir=str(layout.run_dir),
        variants_requested=["eager", "torch_compile", "kf_cast"],
        stages_requested=["load", "compile", "warmup", "operator"],
        description="operator paper benchmark test",
    )
    env = EnvironmentArtifact(
        **common,
        artifact_type="environment_snapshot",
        benchmark_mode="operator",
        variant=None,
        stage=None,
        warmup_count=1,
        timed_run_count=2,
        latency_samples_ms=[],
        correctness_status="not_applicable",
        **build_environment_artifact_fields(),
    )
    common_fields = manifest.model_dump(
        mode="json",
        exclude={"artifact_type", "run_dir", "variants_requested", "stages_requested", "description"},
    )
    suite = type(
        "Suite",
        (),
        {
            "suite_id": "operator_softmax",
            "benchmark_mode": "operator",
            "workload_type": "operator_entries",
            "workload_path": str(entries_dir),
            "synthetic_workload": False,
            "variants": [Variant.eager, Variant.torch_compile, Variant.kf_cast],
            "stages": ["load", "compile", "warmup", "operator"],
            "warmup_count": 1,
            "timed_run_count": 2,
            "device": "cpu",
            "callable_name": None,
            "op_name": "aten.softmax",
            "kernel_source_or_cast": kernel_source_or_cast,
        },
    )()
    return layout, common_fields, env, manifest, suite


def _compile_failure(_callable, _settings):
    raise RuntimeError("synthetic compile failure")


def _make_fake_kf_loader(*, mismatch: bool = False, artifact_kind: str = "cast", project_ref: str | None = None):
    def _loader(op_name, kernel_source_or_cast, *, reference_callable, device=None, layout=None, settings=None):
        def _invoke(*args, **kwargs):
            output = reference_callable(*args, **kwargs)
            return output + 1 if mismatch else output

        artifact_hash = hashlib.sha256(str(kernel_source_or_cast).encode("utf-8")).hexdigest()
        canonical_name = "torch_nn_functional_softmax"
        selected_kernel_metadata = {
            canonical_name: {
                "candidate_id": f"{canonical_name}:deployment" if artifact_kind == "cast" else f"{canonical_name}:direct_source",
                "kernel_source_path": str(kernel_source_or_cast),
                "selected_source_hash": artifact_hash,
                "evidence_tier": "deployment" if artifact_kind == "cast" else "micro_only",
                "benchmark_reference": {"artifact_path": "benchmarks/op_benchmarks.json", "row_ref": "results[0]"},
            }
        }
        return _invoke, {
            "load_time_ms": 3.0,
            "jit_compile_time_ms": 2.0,
            "compile_time_ms": 2.0,
            "runtime_load_time_ms": 3.0,
            "setup_time_ms": 3.0,
            "cast_package_path": str(kernel_source_or_cast) if artifact_kind == "cast" else None,
            "cast_package_hash": artifact_hash if artifact_kind == "cast" else None,
            "kf_artifact_path": str(kernel_source_or_cast),
            "kf_artifact_hash": artifact_hash,
            "kf_artifact_kind": artifact_kind,
            "claim_scope": "deployment_operator" if artifact_kind == "cast" else "micro_operator",
            "operator_runtime_label": "deployment/operator" if artifact_kind == "cast" else "micro/operator",
            "project_ref": project_ref,
            "selection_policy": "auto_best_fastest_valid" if artifact_kind == "cast" else None,
            "cast_manifest": {
                "project_ref": project_ref,
                "selection_policy": "auto_best_fastest_valid",
                "selected_kernel_metadata": selected_kernel_metadata,
            }
            if artifact_kind == "cast"
            else None,
            "selected_ops": [canonical_name],
            "selected_kernel_metadata": selected_kernel_metadata,
            "loaded_kernels": [{"op_name": op_name, "load_mode": "jit"}],
            "precompiled_vs_jit_path": {op_name: "jit"},
            "kernel_source_hashes": {"kernel.cu": artifact_hash},
            "selected_source_hashes": {canonical_name: artifact_hash},
            "kernel_hit_count": 0,
            "fallback_count": 0,
            "kernel_launches_attempted": 0,
            "kernel_launches_succeeded": 0,
            "kernel_launches_failed": 0,
            "exception_fallback_count": 0,
            "contiguous_copy_count": 0,
            "adaptation_count": 0,
        }

    return _loader


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_project_selection_fixture(
    project_root: Path,
    *,
    op_name: str = "torch_nn_functional_softmax",
    deployment_ms: float = 0.4,
    operator_ms: float = 0.2,
    operator_strict_pass: bool = False,
) -> dict[str, Path]:
    project_root.mkdir(parents=True, exist_ok=True)
    _write_json(project_root / "config.json", {"backend": "cuda"})
    (project_root / "model.py").write_text("class DemoModel:\n    pass\n", encoding="utf-8")
    entry_path = project_root / "io" / "individual_ops" / op_name / "entry_000000.pt"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_bytes(b"pt")

    operator_kernel = project_root / "kernels" / "generated" / "individual_op_kernels" / op_name / "kernel.cu"
    operator_kernel.parent.mkdir(parents=True, exist_ok=True)
    operator_kernel.write_text("// operator kernel\n", encoding="utf-8")

    deployment_kernel = project_root / "benchmarks" / "runtime_kernels" / op_name / "kernel" / "kernel.cu"
    deployment_kernel.parent.mkdir(parents=True, exist_ok=True)
    deployment_kernel.write_text("// deployment kernel\n", encoding="utf-8")

    _write_json(
        project_root / "benchmarks" / "op_benchmarks.json",
        {
            "project": project_root.name,
            "timestamp": "2026-04-21T12:00:00Z",
            "results": [
                {
                    "op": op_name,
                    "winner": "optimized",
                    "kernel_status": "ok",
                    "kernel_ms": operator_ms,
                    "kernel_entry_latencies_ms": [operator_ms, operator_ms + 0.01],
                    "kernel_estimated": False,
                    "kernel_source_path": str(operator_kernel),
                    "kernel_source_hash": safe_sha256_path(operator_kernel),
                    "kernel_correctness": {
                        "strict_pass": operator_strict_pass,
                        "errors": [] if operator_strict_pass else ["mismatch"],
                    },
                    "integrated_kernel_status": "ok",
                    "integrated_kernel_ms": deployment_ms,
                    "integrated_kernel_entry_latencies_ms": [deployment_ms, deployment_ms + 0.01],
                    "integrated_kernel_estimated": False,
                    "deployment_correctness": {
                        "strict_pass": True,
                        "errors": [],
                    },
                    "deployment_safe_winner": "optimized",
                    "deployment_source_path": str(deployment_kernel),
                    "deployment_source_hash": safe_sha256_path(deployment_kernel),
                    "benchmarked_entry_files": ["entry_000000.pt"],
                }
            ],
        },
    )
    _write_json(
        project_root / "benchmarks" / "qwen_tps_compare.json",
        {
            "timestamp": "2026-04-21T12:05:00Z",
            "forged": {
                "patch_sources": {op_name: str(deployment_kernel)},
                "patch_stats": {
                    op_name: {
                        "calls": 4,
                        "fallback": 0,
                        "kernel_success": 4,
                        "last_error": "",
                    }
                },
            },
        },
    )
    return {
        "project_root": project_root,
        "operator_kernel": operator_kernel,
        "deployment_kernel": deployment_kernel,
    }


def test_missing_captured_entries_fail(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "empty_entries"
    layout, common_fields, env, manifest, suite = _make_op_context(
        tmp_path,
        sample_paths,
        variant=Variant.eager,
        entries_dir=entries_dir,
    )

    try:
        run_operator_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env,
            manifest_artifact=manifest,
            suite=suite,
            variant=Variant.eager,
        )
    except ValueError as exc:
        assert "No captured operator entries found" in str(exc)
    else:
        raise AssertionError("Expected missing operator entries to fail")


def test_changed_entry_changes_entry_set_hash(tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 4.0]]))
    _, summary_before = load_operator_entries(entries_dir, requested_op_name="aten.softmax")

    _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 5.0]]))
    _, summary_after = load_operator_entries(entries_dir, requested_op_name="aten.softmax")

    assert summary_before["entry_set_hash"] != summary_after["entry_set_hash"]


def test_full_entry_set_hash_includes_all_entries(tmp_path: Path):
    entries_dir = tmp_path / "entries"
    entry_a = _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    entry_b = _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 4.0]]))
    entries, summary = load_operator_entries(entries_dir, requested_op_name="aten.softmax")

    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.entry_name):
        digest.update(entry.entry_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.entry_hash.encode("utf-8"))
        digest.update(b"\n")
    assert summary["entry_set_hash"] == digest.hexdigest()
    assert summary["entry_hashes"]["entry_000001.pt"] == load_operator_entries(entries_dir, requested_op_name="aten.softmax")[1]["entry_hashes"]["entry_000001.pt"]
    assert summary["entry_hashes"]["entry_000001.pt"] != summary["entry_hashes"]["entry_000002.pt"]
    assert entry_a.exists() and entry_b.exists()


def test_correctness_failure_prevents_safe_win(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 4.0]]))

    eager_layout, eager_common, eager_env, eager_manifest, eager_suite = _make_op_context(
        tmp_path / "eager",
        sample_paths,
        variant=Variant.eager,
        entries_dir=entries_dir,
    )
    run_operator_benchmark(
        layout=eager_layout,
        common_fields=eager_common,
        env_artifact=eager_env,
        manifest_artifact=eager_manifest,
        suite=eager_suite,
        variant=Variant.eager,
    )

    kf_layout, kf_common, kf_env, kf_manifest, kf_suite = _make_op_context(
        tmp_path / "kf",
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "softmax.cast"),
    )
    run_operator_benchmark(
        layout=kf_layout,
        common_fields=kf_common,
        env_artifact=kf_env,
        manifest_artifact=kf_manifest,
        suite=kf_suite,
        variant=Variant.kf_cast,
        kf_loader=_make_fake_kf_loader(mismatch=True, artifact_kind="cast"),
    )

    summary = summarize_run(kf_layout.run_dir)
    kf_rows = [row for row in summary.rows if row.variant == Variant.kf_cast and row.stage.value == "operator"]
    assert len(kf_rows) == 1
    assert kf_rows[0].claim_eligible is False
    artifact = load_json_artifact(kf_layout.metrics_dir / "kf_cast_operator.json")
    assert artifact.details["precheck_failures"] == 2
    assert artifact.timed_run_count == 4
    assert all(record["precheck_status"] == "failed" for record in artifact.sample_records)
    assert all(record["precheck_tensor_error_summary"] is not None for record in artifact.sample_records)


def test_direct_source_benchmark_cannot_be_labeled_deployment(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 4.0]]))
    layout, common_fields, env, manifest, suite = _make_op_context(
        tmp_path,
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "kernel.cu"),
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
    assert artifact.details["selected_kernel_metadata"]["torch_nn_functional_softmax"]["evidence_tier"] == "micro_only"


def test_torch_compile_failure_recorded_separately(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    _write_entry(entries_dir, "entry_000002.pt", torch.tensor([[3.0, 4.0]]))
    layout, common_fields, env, manifest, suite = _make_op_context(
        tmp_path,
        sample_paths,
        variant=Variant.torch_compile,
        entries_dir=entries_dir,
    )

    run_operator_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        suite=suite,
        variant=Variant.torch_compile,
        compile_model_fn=_compile_failure,
    )

    compile_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_compile.json")
    assert compile_artifact.correctness_status.value == "failed"
    assert compile_artifact.paper_eligible is False
    assert not (layout.metrics_dir / "torch_compile_operator.json").exists()


def test_project_ref_recorded(sample_paths, tmp_path: Path):
    entries_dir = tmp_path / "entries"
    _write_entry(entries_dir, "entry_000001.pt", torch.tensor([[1.0, 2.0]]))
    layout, common_fields, env, manifest, suite = _make_op_context(
        tmp_path,
        sample_paths,
        variant=Variant.kf_cast,
        entries_dir=entries_dir,
        kernel_source_or_cast=str(tmp_path / "softmax.cast"),
        project_ref="project/test_qwen%20-%20NVIDIA%20GB10/",
    )

    run_operator_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        suite=suite,
        variant=Variant.kf_cast,
        kf_loader=_make_fake_kf_loader(artifact_kind="cast", project_ref="project/test_qwen%20-%20NVIDIA%20GB10/"),
    )

    artifact = load_json_artifact(layout.metrics_dir / "kf_cast_operator.json")
    assert artifact.details["project_ref"] == "project/test_qwen%20-%20NVIDIA%20GB10/"
    summary = summarize_run(layout.run_dir)
    row = next(item for item in summary.rows if item.variant == Variant.kf_cast and item.stage.value == "operator")
    assert row.project_ref == "project/test_qwen%20-%20NVIDIA%20GB10/"
    assert row.export_selection_policy == "auto_best_fastest_valid"


def test_fastest_invalid_candidate_not_selected(tmp_path: Path, monkeypatch):
    project = _write_project_selection_fixture(tmp_path / "test_qwen - NVIDIA GB10")
    monkeypatch.setenv("KFORGE_DATA_DIR", str(tmp_path))

    evidence = resolve_project_operator_export_evidence(
        "project/test_qwen%20-%20NVIDIA%20GB10/",
        "aten.softmax",
        search_roots=[tmp_path],
    )

    assert evidence["selected_op_name"] == "torch_nn_functional_softmax"
    assert evidence["selected_candidate"]["candidate_id"] == "torch_nn_functional_softmax:deployment"
    assert evidence["selected_candidate"]["kernel_source_path"] == str(project["deployment_kernel"].resolve())
    rejected = evidence["rejected_candidates"]
    assert any(item["candidate_id"] == "torch_nn_functional_softmax:operator" for item in rejected)
