from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from .schema import BenchmarkMode, RunManifestArtifact, Variant


MIN_PAPER_WARMUP_RUNS = 5
MIN_PAPER_TIMED_RUNS = 20
PUBLISHABLE_VARIANTS = {Variant.eager, Variant.torch_compile, Variant.kf_cast}


def _has_verified_terms(*values: str | None) -> bool:
    text = " ".join(str(value or "").strip().lower() for value in values if value)
    if not text:
        return False
    placeholders = (
        "must be verified",
        "to be verified",
        "tbd",
        "unknown",
        "local mirror",
    )
    return not any(marker in text for marker in placeholders)


def _cast_evidence_blockers(cast_inspection: dict[str, Any] | None) -> list[str]:
    if not cast_inspection:
        return ["CAST inspection missing for kf_cast plan"]
    blockers: list[str] = []
    if not cast_inspection.get("cast_package_sha256"):
        blockers.append("CAST package hash missing")
    if cast_inspection.get("checksum_verified") is False:
        blockers.append("CAST package checksum verification failed")
    if cast_inspection.get("loadability_blockers"):
        blockers.append("CAST loadability blockers present")
    if cast_inspection.get("export_paper_eligible") is not True:
        blockers.append("CAST export is not deployment-paper eligible")
    if cast_inspection.get("uses_non_deployment_evidence"):
        blockers.append("CAST uses non-deployment evidence")
    selected_metadata = cast_inspection.get("selected_kernel_metadata")
    if not isinstance(selected_metadata, dict) or not selected_metadata:
        blockers.append("CAST selected-kernel metadata missing")
        return blockers
    non_deployment = sorted(
        op_name
        for op_name, metadata in selected_metadata.items()
        if not isinstance(metadata, dict) or metadata.get("evidence_tier") != "deployment"
    )
    if non_deployment:
        blockers.append("selected kernels lack deployment-tier evidence: " + ", ".join(non_deployment))
    missing_refs = sorted(
        op_name
        for op_name, metadata in selected_metadata.items()
        if not isinstance(metadata, dict)
        or not (metadata.get("benchmark_artifact_path") or (metadata.get("benchmark_reference") or {}).get("artifact_path"))
    )
    if missing_refs:
        blockers.append("selected kernels lack benchmark artifact references: " + ", ".join(missing_refs))
    return blockers


def _expected_run_command(manifest: RunManifestArtifact) -> str:
    command = list(manifest.command_line)
    if command:
        command = list(command)
        if "plan-llm" in command:
            command[command.index("plan-llm")] = "run-llm"
        cleaned: list[str] = []
        skip_next = False
        for item in command:
            if skip_next:
                skip_next = False
                continue
            if item == "--write-plan":
                skip_next = True
                continue
            if item in {"--fail-if-not-paper-ready", "--skip-cast-inspection"}:
                continue
            cleaned.append(item)
        command = cleaned
    if not command:
        command = [sys.executable, "-m", "paper_benchmarks.paper_bench.cli", "run-llm"]
    return shlex.join(command)


def certify_llm_publication_plan(
    *,
    manifest: RunManifestArtifact,
    env_artifact,
    model_spec,
    suite,
    requested_variants: Sequence[Variant],
    cast_inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fatal_errors: list[str] = []
    paper_blockers: list[str] = []
    warnings: list[str] = []

    if not manifest.model_path_hash:
        fatal_errors.append("model path hash missing")
    if not manifest.model_config_hash:
        fatal_errors.append("model config hash missing")
    if not manifest.suite_hash:
        fatal_errors.append("suite/workload hash missing")
    if not manifest.workload_hash:
        fatal_errors.append("workload hash missing")
    if not manifest.command_line_text:
        fatal_errors.append("command line missing")
    if manifest.synthetic_workload:
        paper_blockers.append("synthetic workloads cannot support IISWC paper claims")
    if manifest.cache_mode not in {"kv_cache_on", "kv_cache_off"}:
        paper_blockers.append(f"cache mode {manifest.cache_mode!r} is not publishable in the current runner")
    if not manifest.placement_profile:
        paper_blockers.append("placement profile missing")
    if manifest.git_dirty:
        warnings.append("git tree is dirty; commit or archive the diff before final paper runs")
    if not _has_verified_terms(manifest.model_license, manifest.model_access_terms):
        paper_blockers.append("model license/access terms are missing or still marked unverified")
    if not _has_verified_terms(manifest.dataset_license, manifest.dataset_access_terms):
        paper_blockers.append("dataset license/access terms are missing or still marked unverified")
    if getattr(model_spec, "paper_eligible", False) is not True:
        paper_blockers.append("model config is not marked paper_eligible")
    if manifest.paper_eligibility_issues:
        paper_blockers.extend(str(issue) for issue in manifest.paper_eligibility_issues)
    if int(getattr(suite, "warmup_count", 0) or 0) < MIN_PAPER_WARMUP_RUNS:
        paper_blockers.append(
            f"warmup_count {getattr(suite, 'warmup_count', 0)} is below paper minimum {MIN_PAPER_WARMUP_RUNS}"
        )
    if int(getattr(suite, "timed_run_count", 0) or 0) < MIN_PAPER_TIMED_RUNS:
        paper_blockers.append(
            f"timed_run_count {getattr(suite, 'timed_run_count', 0)} is below paper minimum {MIN_PAPER_TIMED_RUNS}"
        )

    requested_set = set(requested_variants)
    if getattr(suite, "benchmark_mode", None) == BenchmarkMode.e2e_model:
        missing_variants = sorted(variant.value for variant in PUBLISHABLE_VARIANTS - requested_set)
        if missing_variants:
            paper_blockers.append("publishable model-level run must request variants: " + ", ".join(missing_variants))
    baseline_requirements = set(getattr(model_spec, "baselines_required", []) or [])
    missing_baselines = sorted(variant.value for variant in PUBLISHABLE_VARIANTS - baseline_requirements)
    if missing_baselines:
        paper_blockers.append("model config baselines_required missing: " + ", ".join(missing_baselines))

    if Variant.kf_cast in requested_set:
        if not manifest.cast_package_hash:
            fatal_errors.append("CAST package hash missing")
        paper_blockers.extend(_cast_evidence_blockers(cast_inspection))
        if manifest.kf_settings.get("fail_on_fallback") is not True:
            paper_blockers.append("kf_cast paper run must enable fail_on_fallback")
        if manifest.kf_settings.get("record_runtime_stats") is not True:
            paper_blockers.append("kf_cast paper run must record runtime stats")
        if manifest.kf_settings.get("require_precompiled") is not True:
            warnings.append("kf_cast paper run allows JIT; precompiled deployment artifacts are preferred for final runs")

    if getattr(env_artifact, "gpu_count", 0) <= 0:
        fatal_errors.append("no CUDA GPU detected")
    toolchain_status = manifest.toolchain_status or {}
    if Variant.kf_cast in requested_set and not bool(toolchain_status.get("jit_ready")):
        paper_blockers.append("CUDA JIT/precompile toolchain is not ready")

    fatal_errors = list(dict.fromkeys(fatal_errors))
    paper_blockers = list(dict.fromkeys(paper_blockers))
    warnings = list(dict.fromkeys(warnings))
    paper_ready = not fatal_errors and not paper_blockers
    return {
        "ok_to_run": not fatal_errors,
        "paper_ready": paper_ready,
        "claim_scope": "paper_eligible" if paper_ready else "exploratory_or_blocked",
        "fatal_errors": fatal_errors,
        "paper_blockers": paper_blockers,
        "warnings": warnings,
        "minimum_statistical_protocol": {
            "warmup_count": MIN_PAPER_WARMUP_RUNS,
            "timed_run_count": MIN_PAPER_TIMED_RUNS,
            "process_repetitions": "required before final paper figures",
            "outlier_policy": "must be declared before full matrix execution",
        },
    }


def build_llm_plan_payload(
    *,
    manifest: RunManifestArtifact,
    env_artifact,
    model_spec,
    suite,
    requested_variants: Sequence[Variant],
    cast_inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    certification = certify_llm_publication_plan(
        manifest=manifest,
        env_artifact=env_artifact,
        model_spec=model_spec,
        suite=suite,
        requested_variants=requested_variants,
        cast_inspection=cast_inspection,
    )
    return {
        "schema": "paper_bench_llm_publication_plan_v1",
        "run_id": manifest.run_id,
        "expected_run_dir": manifest.run_dir,
        "expected_run_command": _expected_run_command(manifest),
        "model": {
            "model_id": model_spec.model_id,
            "quantization": getattr(model_spec, "quantization", None),
            "model_path": model_spec.model_path,
            "model_path_hash": manifest.model_path_hash,
            "model_config_path": model_spec.model_config_path,
            "model_config_hash": manifest.model_config_hash,
            "placement_profile": manifest.placement_profile,
            "device_map": getattr(model_spec, "device_map", None),
            "paper_eligible_declared": bool(getattr(model_spec, "paper_eligible", False)),
            "benchmark_expectation": getattr(model_spec, "benchmark_expectation", None),
        },
        "suite": {
            "suite_id": suite.suite_id,
            "suite_path": manifest.suite_path,
            "suite_hash": manifest.suite_hash,
            "workload_path": manifest.workload_path,
            "workload_hash": manifest.workload_hash,
            "workload_slug": manifest.workload_slug,
            "cache_mode": manifest.cache_mode,
            "warmup_count": suite.warmup_count,
            "timed_run_count": suite.timed_run_count,
            "max_new_tokens": getattr(suite, "max_new_tokens", None),
            "variants": [variant.value for variant in requested_variants],
        },
        "cast": {
            "cast_package_path": manifest.cast_package_path,
            "cast_package_hash": manifest.cast_package_hash,
            "inspection": cast_inspection,
        },
        "baseline_dependencies": {
            "required_variants": sorted(variant.value for variant in PUBLISHABLE_VARIANTS),
            "requested_variants": [variant.value for variant in requested_variants],
            "model_baselines_required": [variant.value for variant in getattr(model_spec, "baselines_required", [])],
        },
        "certification": certification,
    }


def write_plan_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
