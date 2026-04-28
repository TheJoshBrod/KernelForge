from __future__ import annotations

from typing import Any

from .schema import BenchmarkArtifact, CorrectnessStatus, PERFORMANCE_STAGES, Stage, Variant


PAPER_CLAIM_ELIGIBLE = "paper_eligible"
PAPER_CLAIM_BLOCKED = "blocked"
PAPER_CLAIM_EXPLORATORY = "exploratory"
PAPER_CLAIM_INVALID = "invalid"

CLAIM_BASELINE = "baseline"
CLAIM_PURE_KF_SPEEDUP = "pure_kf_speedup"
CLAIM_HYBRID_KF_SPEEDUP = "hybrid_kf_speedup"
CLAIM_DEPLOYMENT_MEASUREMENT = "deployment_measurement"
CLAIM_EXPLORATORY = "exploratory"
CLAIM_INVALID = "invalid"


def _details(artifact: BenchmarkArtifact) -> dict[str, Any]:
    return artifact.details if isinstance(artifact.details, dict) else {}


def _cast_manifest(artifact: BenchmarkArtifact) -> dict[str, Any]:
    manifest = _details(artifact).get("cast_manifest")
    return dict(manifest) if isinstance(manifest, dict) else {}


def _selected_ops(artifact: BenchmarkArtifact) -> list[str]:
    details = _details(artifact)
    raw = details.get("selected_ops")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    manifest_ops = _cast_manifest(artifact).get("selected_ops")
    if isinstance(manifest_ops, list):
        return [str(item) for item in manifest_ops if str(item).strip()]
    return []


def _evidence_tiers(artifact: BenchmarkArtifact) -> dict[str, str]:
    metadata = _cast_manifest(artifact).get("selected_kernel_metadata")
    if not isinstance(metadata, dict):
        metadata = _details(artifact).get("selected_kernel_metadata")
    if not isinstance(metadata, dict):
        return {}
    tiers: dict[str, str] = {}
    for op_name, op_meta in metadata.items():
        if isinstance(op_meta, dict) and op_meta.get("evidence_tier") is not None:
            tiers[str(op_name)] = str(op_meta.get("evidence_tier"))
    return tiers


def _runtime_per_op_stats(artifact: BenchmarkArtifact) -> dict[str, dict[str, Any]]:
    per_op = _details(artifact).get("per_op_stats")
    if not isinstance(per_op, dict):
        return {}
    return {
        str(op_name): dict(stats)
        for op_name, stats in per_op.items()
        if isinstance(stats, dict)
    }


def per_op_launch_coverage(artifact: BenchmarkArtifact) -> dict[str, Any]:
    selected_ops = _selected_ops(artifact)
    per_op = _runtime_per_op_stats(artifact)
    coverage: dict[str, Any] = {}
    for op_name in selected_ops:
        stats = per_op.get(op_name, {})
        input_devices = stats.get("input_devices") if isinstance(stats.get("input_devices"), dict) else {}
        input_dtypes = stats.get("input_dtypes") if isinstance(stats.get("input_dtypes"), dict) else {}
        input_is_cuda = stats.get("input_is_cuda") if isinstance(stats.get("input_is_cuda"), dict) else {}
        coverage[op_name] = {
            "patched_calls": int(stats.get("patched_calls", 0) or 0),
            "kernel_launches_attempted": int(stats.get("kernel_launches_attempted", 0) or 0),
            "kernel_launches_succeeded": int(stats.get("kernel_launches_succeeded", 0) or 0),
            "fallbacks_to_original": int(stats.get("fallbacks_to_original", 0) or 0),
            "fallback_reasons": dict(stats.get("fallback_reasons", {}) if isinstance(stats.get("fallback_reasons"), dict) else {}),
            "input_devices": {str(key): int(value) for key, value in input_devices.items()},
            "input_dtypes": {str(key): int(value) for key, value in input_dtypes.items()},
            "input_is_cuda": {str(key): int(value) for key, value in input_is_cuda.items()},
        }
    return coverage


def _has_cpu_or_meta_selected_inputs(coverage: dict[str, Any]) -> list[str]:
    bad_ops: list[str] = []
    for op_name, op_coverage in coverage.items():
        input_devices = op_coverage.get("input_devices", {}) if isinstance(op_coverage, dict) else {}
        input_is_cuda = op_coverage.get("input_is_cuda", {}) if isinstance(op_coverage, dict) else {}
        has_bad_device = any(
            str(device).startswith("cpu") or str(device).startswith("meta")
            for device in input_devices
        )
        has_non_cuda_count = int(input_is_cuda.get("false", 0) or 0) > 0
        if has_bad_device or has_non_cuda_count:
            bad_ops.append(str(op_name))
    return bad_ops


def validate_benchmark_artifact(
    artifact: BenchmarkArtifact,
    *,
    eager_baseline: BenchmarkArtifact | None = None,
    torch_compile_baseline: BenchmarkArtifact | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    coverage = per_op_launch_coverage(artifact)
    details = _details(artifact)

    if artifact.correctness_status == CorrectnessStatus.failed:
        errors.append("correctness failure")
    if str(details.get("execution_status", "")).lower() == "failed":
        errors.append("execution failure")
    if artifact.stage in PERFORMANCE_STAGES and not artifact.latency_samples_ms:
        errors.append("timed latency samples missing")
    if artifact.stage in PERFORMANCE_STAGES and not artifact.workload_hash:
        errors.append("workload hash missing")
    if artifact.cache_mode and artifact.cache_mode not in {"kv_cache_on", "kv_cache_off", "operator"} and artifact.benchmark_mode is not None:
        warnings.append(f"cache mode {artifact.cache_mode} is not publishable in the current runner")
    if artifact.reused and artifact.cache_reuse_status != "signature_matched":
        errors.append("reused artifact lacks a matched cache signature")

    if artifact.variant == Variant.kf_cast and artifact.stage in PERFORMANCE_STAGES:
        selected_ops = _selected_ops(artifact)
        if not selected_ops:
            warnings.append("selected Forge op list missing")
        if artifact.kf_settings.get("record_runtime_stats") is True and not details.get("device_audit_artifact_path"):
            warnings.append("standalone device audit artifact missing")
        if artifact.fallback_count is None:
            errors.append("fallback count missing")
        if artifact.kernel_hit_count is None:
            errors.append("kernel launch count missing")
        elif artifact.kernel_hit_count <= 0:
            errors.append("zero selected-op kernel launches")
        if details.get("runtime_patch_enabled") is False:
            errors.append("CAST runtime patch was not enabled")
        bad_input_ops = _has_cpu_or_meta_selected_inputs(coverage)
        if bad_input_ops:
            errors.append("selected Forge op received non-CUDA inputs: " + ", ".join(sorted(bad_input_ops)))
        missing_runtime_coverage = [
            op_name
            for op_name, op_coverage in coverage.items()
            if int(op_coverage.get("patched_calls", 0) or 0) <= 0
        ]
        if selected_ops and not coverage:
            warnings.append("per-selected-op runtime coverage missing")
        elif missing_runtime_coverage:
            warnings.append("per-selected-op launch coverage incomplete: " + ", ".join(sorted(missing_runtime_coverage)))
        if artifact.kf_settings.get("fail_on_fallback", True) and (artifact.fallback_count or 0) > 0:
            errors.append("unexpected fallback while fail_on_fallback is enabled")
        if not artifact.cast_package_hash:
            errors.append("CAST package hash missing")
        manifest = _cast_manifest(artifact)
        if manifest and manifest.get("export_paper_eligible") is not True:
            warnings.append("CAST manifest is not marked export_paper_eligible")
        if manifest and manifest.get("posthoc_patches"):
            warnings.append("CAST manifest contains posthoc source patches")
        tiers = _evidence_tiers(artifact)
        non_deployment = sorted(op for op, tier in tiers.items() if tier != "deployment")
        if non_deployment:
            warnings.append("selected kernels lack deployment-tier evidence: " + ", ".join(non_deployment))
        toolchain_status = artifact.toolchain_status or details.get("toolchain_status") or {}
        if artifact.kf_settings.get("allow_jit", True) and isinstance(toolchain_status, dict) and not toolchain_status.get("jit_ready"):
            warnings.append("JIT/precompile toolchain is not recorded as ready")

    if artifact.variant == Variant.kf_cast and artifact.stage in PERFORMANCE_STAGES:
        if eager_baseline is None:
            warnings.append("missing eager baseline for comparison group")
        if torch_compile_baseline is None:
            warnings.append("missing torch_compile baseline for comparison group")
        for baseline_name, baseline in (("eager", eager_baseline), ("torch_compile", torch_compile_baseline)):
            if baseline is None:
                continue
            expected_statuses = {CorrectnessStatus.reference, CorrectnessStatus.passed}
            if baseline.correctness_status not in expected_statuses:
                errors.append(f"{baseline_name} baseline correctness is not usable: {baseline.correctness_status.value}")
            if baseline.steady_state_time_ms is None:
                errors.append(f"{baseline_name} baseline steady-state timing missing")
            if baseline.workload_hash != artifact.workload_hash:
                errors.append(f"{baseline_name} workload hash mismatch")
            if baseline.model_config_hash != artifact.model_config_hash:
                errors.append(f"{baseline_name} model config hash mismatch")
            if baseline.quantization != artifact.quantization:
                errors.append(f"{baseline_name} quantization mismatch")
            if baseline.quantization_config_hash != artifact.quantization_config_hash:
                errors.append(f"{baseline_name} quantization config hash mismatch")
            if baseline.placement_profile != artifact.placement_profile:
                errors.append(f"{baseline_name} placement profile mismatch")
            if baseline.cache_mode != artifact.cache_mode:
                errors.append(f"{baseline_name} cache mode mismatch")

    if artifact.stage in PERFORMANCE_STAGES:
        if not artifact.placement_profile and artifact.benchmark_mode is not None:
            warnings.append("placement profile missing")
        if not (artifact.model_license or artifact.model_access_terms):
            warnings.append("model license/access terms missing")
        if not (artifact.dataset_license or artifact.dataset_access_terms):
            warnings.append("dataset license/access terms missing")

    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    if errors:
        claim_status = PAPER_CLAIM_INVALID
        claim_category = CLAIM_INVALID
    elif warnings:
        claim_status = PAPER_CLAIM_EXPLORATORY
        claim_category = CLAIM_EXPLORATORY
        if artifact.variant == Variant.kf_cast and artifact.stage in PERFORMANCE_STAGES and artifact.cast_package_hash:
            claim_category = CLAIM_DEPLOYMENT_MEASUREMENT
    elif artifact.paper_eligible:
        claim_status = PAPER_CLAIM_ELIGIBLE
        if artifact.variant == Variant.kf_cast and artifact.stage in PERFORMANCE_STAGES:
            if artifact.fallback_count and artifact.fallback_count > 0:
                claim_category = CLAIM_HYBRID_KF_SPEEDUP
            elif not artifact.kf_settings.get("fail_on_fallback", True):
                claim_category = CLAIM_HYBRID_KF_SPEEDUP
            else:
                claim_category = CLAIM_PURE_KF_SPEEDUP
        elif artifact.variant in {Variant.eager, Variant.torch_compile}:
            claim_category = CLAIM_BASELINE
        else:
            claim_category = CLAIM_DEPLOYMENT_MEASUREMENT
    else:
        claim_status = PAPER_CLAIM_BLOCKED
        claim_category = CLAIM_EXPLORATORY

    return {
        "paper_claim_status": claim_status,
        "claim_category": claim_category,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "per_op_launch_coverage": coverage,
    }


def validated_artifact_update(
    artifact: BenchmarkArtifact,
    *,
    eager_baseline: BenchmarkArtifact | None = None,
    torch_compile_baseline: BenchmarkArtifact | None = None,
) -> BenchmarkArtifact:
    update = validate_benchmark_artifact(
        artifact,
        eager_baseline=eager_baseline,
        torch_compile_baseline=torch_compile_baseline,
    )
    artifact_update = {
        key: value
        for key, value in update.items()
        if key != "per_op_launch_coverage"
    }
    return BenchmarkArtifact.model_validate({**artifact.model_dump(mode="python"), **artifact_update})
