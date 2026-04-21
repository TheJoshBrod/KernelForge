from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from paper_benchmarks.paper_bench.artifacts import create_run_layout, load_json_artifact
from paper_benchmarks.paper_bench.op_runner import load_operator_entries, run_operator_benchmark
from paper_benchmarks.paper_bench.provenance import build_environment_artifact_fields, collect_common_fields
from paper_benchmarks.paper_bench.report import summarize_run
from paper_benchmarks.paper_bench.schema import EnvironmentArtifact, RunManifestArtifact, Variant


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


def _make_op_context(tmp_path: Path, sample_paths: dict[str, str], *, variant: Variant, entries_dir: Path, kernel_source_or_cast: str | None = None):
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


def _make_fake_kf_loader(*, mismatch: bool = False, artifact_kind: str = "cast"):
    def _loader(op_name, kernel_source_or_cast, *, reference_callable, device=None, layout=None, settings=None):
        def _invoke(*args, **kwargs):
            output = reference_callable(*args, **kwargs)
            return output + 1 if mismatch else output

        artifact_hash = hashlib.sha256(str(kernel_source_or_cast).encode("utf-8")).hexdigest()
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
            "selected_ops": [op_name],
            "loaded_kernels": [{"op_name": op_name, "load_mode": "jit"}],
            "precompiled_vs_jit_path": {op_name: "jit"},
            "kernel_source_hashes": {"kernel.cu": artifact_hash},
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
