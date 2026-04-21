from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_benchmarks.paper_bench.cast_export import (
    build_cast_manifest_metadata,
    copy_cast_artifact,
    export_cast_package,
    inspect_cast_package,
    resolve_cast_export_plan,
)
from paper_benchmarks.paper_bench.cast_selection import (
    NoEligibleCastKernelsError,
    POLICY_AUTO_BEST_FASTEST_VALID,
)
from paper_benchmarks.paper_bench.provenance import safe_sha256_path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_export_project(
    project_root: Path,
    *,
    op_name: str = "torch_nn_functional_linear",
    create_entry: bool = True,
    deployment_ms: float = 0.4,
    deployment_strict_pass: bool = True,
    deployment_winner: str = "optimized",
    operator_ms: float = 0.8,
    operator_strict_pass: bool = True,
    operator_winner: str = "optimized",
    runtime_audit_calls: int = 4,
    runtime_audit_fallback: int = 0,
    runtime_audit_error: str = "",
) -> dict[str, Path]:
    project_root.mkdir(parents=True, exist_ok=True)
    _write_json(project_root / "config.json", {"backend": "cuda"})
    (project_root / "model.py").write_text("class DemoModel:\n    pass\n", encoding="utf-8")

    entry_path = project_root / "io" / "individual_ops" / op_name / "entry_000000.pt"
    if create_entry:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_bytes(b"pt")

    operator_kernel = project_root / "kernels" / "generated" / "individual_op_kernels" / op_name / "kernel.cu"
    operator_kernel.parent.mkdir(parents=True, exist_ok=True)
    operator_kernel.write_text("// operator kernel\n", encoding="utf-8")

    deployment_kernel = project_root / "benchmarks" / "runtime_kernels" / op_name / "kernel" / "kernel.cu"
    deployment_kernel.parent.mkdir(parents=True, exist_ok=True)
    deployment_kernel.write_text("// deployment kernel\n", encoding="utf-8")

    manual_kernel = project_root / "kernels" / "generated" / "individual_op_kernels" / op_name / "manual_override.cu"
    manual_kernel.write_text("// manual override\n", encoding="utf-8")

    _write_json(
        project_root / "benchmarks" / "op_benchmarks.json",
        {
            "project": project_root.name,
            "timestamp": "2026-04-21T12:00:00Z",
            "results": [
                {
                    "op": op_name,
                    "winner": operator_winner,
                    "kernel_status": "ok",
                    "kernel_ms": operator_ms,
                    "kernel_entry_latencies_ms": [operator_ms, operator_ms + 0.01],
                    "kernel_estimated": False,
                    "kernel_source_path": str(operator_kernel),
                    "kernel_source_hash": safe_sha256_path(operator_kernel),
                    "kernel_correctness": {
                        "strict_pass": operator_strict_pass,
                        "errors": [],
                    },
                    "integrated_kernel_status": "ok",
                    "integrated_kernel_ms": deployment_ms,
                    "integrated_kernel_entry_latencies_ms": [deployment_ms, deployment_ms + 0.01],
                    "integrated_kernel_estimated": False,
                    "deployment_correctness": {
                        "strict_pass": deployment_strict_pass,
                        "errors": [],
                    },
                    "deployment_safe_winner": deployment_winner,
                    "deployment_source_path": str(deployment_kernel),
                    "deployment_source_hash": safe_sha256_path(deployment_kernel),
                    "benchmarked_entry_files": ["entry_000000.pt"] if create_entry else [],
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
                        "calls": runtime_audit_calls,
                        "fallback": runtime_audit_fallback,
                        "kernel_success": runtime_audit_calls - runtime_audit_fallback,
                        "last_error": runtime_audit_error,
                    }
                },
            },
        },
    )

    return {
        "project_root": project_root,
        "operator_kernel": operator_kernel,
        "deployment_kernel": deployment_kernel,
        "manual_kernel": manual_kernel,
    }


def test_default_download_cast_uses_auto_best_fastest_valid(tmp_path: Path):
    layout = _write_export_project(tmp_path / "test_qwen - NVIDIA GB10")

    plan = resolve_cast_export_plan(layout["project_root"], repo_root=tmp_path)

    assert plan["selection_policy"] == POLICY_AUTO_BEST_FASTEST_VALID
    assert plan["selected_op_count"] == 1


def test_manual_selection_still_works(tmp_path: Path):
    layout = _write_export_project(tmp_path / "test_qwen - NVIDIA GB10")

    plan = resolve_cast_export_plan(
        layout["project_root"],
        repo_root=tmp_path,
        selected_kernels={"torch_nn_functional_linear": str(layout["manual_kernel"])},
    )

    selected = plan["selected_ops"]["torch_nn_functional_linear"]
    assert selected["candidate_id"] == "torch_nn_functional_linear:manual_override"
    assert selected["manual_override"] is True
    assert plan["export_paper_eligible"] is False


def test_backend_rejects_invalid_fastest_candidate(tmp_path: Path):
    layout = _write_export_project(
        tmp_path / "test_qwen - NVIDIA GB10",
        deployment_ms=0.2,
        deployment_strict_pass=False,
        operator_ms=0.7,
        operator_strict_pass=True,
    )

    plan = resolve_cast_export_plan(layout["project_root"], repo_root=tmp_path)

    selected = plan["selected_ops"]["torch_nn_functional_linear"]
    rejected = plan["rejected_candidates"]["torch_nn_functional_linear"]
    assert selected["candidate_id"] == "torch_nn_functional_linear:operator"
    assert any(
        item["candidate_id"] == "torch_nn_functional_linear:deployment"
        and "correctness failed" in item["rejection_reasons"]
        for item in rejected
    )


def test_manifest_records_selection_policy_and_evidence_tier(tmp_path: Path):
    layout = _write_export_project(tmp_path / "test_qwen - NVIDIA GB10")

    plan = resolve_cast_export_plan(layout["project_root"], repo_root=tmp_path)
    manifest_meta = build_cast_manifest_metadata(plan)

    assert manifest_meta["selection_policy"] == POLICY_AUTO_BEST_FASTEST_VALID
    assert manifest_meta["project_root"] == str(layout["project_root"].resolve())
    assert manifest_meta["selected_kernel_metadata"]["torch_nn_functional_linear"]["evidence_tier"] == "deployment"
    assert manifest_meta["selected_kernel_metadata"]["torch_nn_functional_linear"]["selected_source_hash"] == safe_sha256_path(
        layout["deployment_kernel"]
    )


def test_export_from_fake_project_selects_fastest_valid_kernel(tmp_path: Path):
    layout = _write_export_project(
        tmp_path / "test_qwen - NVIDIA GB10",
        deployment_ms=0.35,
        deployment_strict_pass=True,
        operator_ms=0.6,
        operator_strict_pass=True,
    )

    plan = resolve_cast_export_plan(layout["project_root"], repo_root=tmp_path)

    selected = plan["selected_ops"]["torch_nn_functional_linear"]
    assert selected["candidate_id"] == "torch_nn_functional_linear:deployment"
    assert selected["kernel_source_path"] == str(layout["deployment_kernel"].resolve())


def test_export_from_project_with_no_valid_kernels_fails_loudly(tmp_path: Path):
    layout = _write_export_project(
        tmp_path / "test_qwen - NVIDIA GB10",
        create_entry=False,
        deployment_strict_pass=False,
        operator_strict_pass=False,
        deployment_winner="pytorch",
        operator_winner="pytorch",
        runtime_audit_calls=0,
    )

    with pytest.raises(NoEligibleCastKernelsError) as exc_info:
        resolve_cast_export_plan(layout["project_root"], repo_root=tmp_path)

    assert exc_info.value.report["selected_ops"] == {}
    skipped = exc_info.value.report["skipped_ops"]["torch_nn_functional_linear"]
    assert skipped["skip_reason"] == "no valid kernel matched auto_best_fastest_valid"
    assert "missing captured inputs" in skipped["rejected_reason_counts"]


def test_export_cast_package_and_inspect_round_trip(tmp_path: Path):
    layout = _write_export_project(tmp_path / "test_qwen - NVIDIA GB10")

    export_result = export_cast_package(layout["project_root"], repo_root=tmp_path)
    inspected = inspect_cast_package(export_result["export_path"])

    assert Path(export_result["export_path"]).exists()
    assert export_result["manifest"]["selection_policy"] == POLICY_AUTO_BEST_FASTEST_VALID
    assert inspected["checksum_verified"] is True
    assert inspected["selected_ops"] == ["torch_nn_functional_linear"]
    assert inspected["selected_kernel_metadata"]["torch_nn_functional_linear"]["evidence_tier"] == "deployment"


def test_copy_cast_artifact_preserves_export(tmp_path: Path):
    layout = _write_export_project(tmp_path / "test_qwen - NVIDIA GB10")

    export_result = export_cast_package(layout["project_root"], repo_root=tmp_path)
    copied = copy_cast_artifact(
        export_result["export_path"],
        destination_dir=tmp_path / "artifacts" / "qwen35a3b",
        filename="qwen35a3b_auto_best_fastest_valid_deadbee_2026-04-21.cast",
    )

    assert copied.exists()
    assert safe_sha256_path(copied) == export_result["cast_package_sha256"]
