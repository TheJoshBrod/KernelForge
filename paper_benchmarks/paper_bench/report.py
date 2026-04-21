from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .artifacts import load_json_artifact, write_json_artifact
from .schema import (
    BenchmarkArtifact,
    BenchmarkMode,
    CorrectnessStatus,
    MODEL_LEVEL_STAGES,
    PERFORMANCE_STAGES,
    Stage,
    SummaryArtifact,
    SummaryRow,
    Variant,
)
from .stats import safe_speedup

SECTION_OPERATOR = "operator_benchmark"
SECTION_MODEL = "end_to_end_model_benchmark"
SECTION_DEPLOYMENT = "deployment_runtime_benchmark"
SECTION_OFFLINE = "offline_costs"
SECTION_CORRECTNESS = "correctness"
SECTION_COVERAGE = "coverage"
SECTION_EXPORT = "export_cast_selection"
SECTION_FAILURES = "failures_regressions"
SECTION_PAPER_CLAIMS = "paper_eligible_claims"
SECTION_FORBIDDEN = "forbidden_unsupported_claims"
REQUIRED_QWEN_PROJECT_REF = "project/test_qwen%20-%20NVIDIA%20GB10/"
REQUIRED_CAST_SELECTION_POLICY = "auto_best_fastest_valid"


def _claim_eligible(artifact: BenchmarkArtifact) -> bool:
    return (
        artifact.stage in PERFORMANCE_STAGES
        and artifact.paper_eligible
        and artifact.correctness_status in {
            CorrectnessStatus.reference,
            CorrectnessStatus.passed,
        }
    )


def _sorted_benchmark_artifacts(root: Path) -> list[tuple[Path, BenchmarkArtifact]]:
    benchmark_artifacts: list[tuple[Path, BenchmarkArtifact]] = []
    for path in sorted((root / "metrics").glob("*.json")):
        artifact = load_json_artifact(path)
        if isinstance(artifact, BenchmarkArtifact):
            benchmark_artifacts.append((path, artifact))
    if not benchmark_artifacts:
        raise ValueError(f"No benchmark artifacts found in {root / 'metrics'}")
    return benchmark_artifacts


def _artifact_prompt_count(artifact: BenchmarkArtifact) -> int | None:
    details = artifact.details or {}
    selected_prompt_ids = details.get("selected_prompt_ids")
    if isinstance(selected_prompt_ids, list):
        return len(selected_prompt_ids)
    prompt_ids: set[str] = set()
    for record in artifact.sample_records:
        raw_ids = record.get("prompt_ids")
        if isinstance(raw_ids, list):
            prompt_ids.update(str(item) for item in raw_ids)
    if prompt_ids:
        return len(prompt_ids)
    coverage = details.get("coverage")
    if isinstance(coverage, dict):
        entry_count = coverage.get("entry_count")
        if isinstance(entry_count, int):
            return entry_count
    return None


def _artifact_tps(artifact: BenchmarkArtifact) -> tuple[float | None, float | None, float | None]:
    aggregate = artifact.details.get("aggregate_metrics", {}) if isinstance(artifact.details, dict) else {}
    stage_tps = aggregate.get("stage_tokens_per_second")
    if artifact.stage == Stage.total_generate:
        return float(stage_tps) if stage_tps is not None else None, None, None
    if artifact.stage == Stage.prefill:
        return None, float(stage_tps) if stage_tps is not None else None, None
    if artifact.stage == Stage.decode:
        return None, None, float(stage_tps) if stage_tps is not None else None
    return None, None, None


def _generated_token_equality_status(artifact: BenchmarkArtifact) -> str | None:
    if artifact.stage not in MODEL_LEVEL_STAGES:
        return None
    if artifact.variant == Variant.eager:
        return "reference"
    if artifact.correctness_status == CorrectnessStatus.passed:
        return "exact_match"
    if artifact.correctness_status == CorrectnessStatus.failed:
        return "mismatch"
    return artifact.correctness_status.value


def _artifact_coverage(artifact: BenchmarkArtifact) -> dict[str, Any]:
    details = artifact.details or {}
    coverage = details.get("coverage")
    if isinstance(coverage, dict):
        return dict(coverage)
    prompt_count = _artifact_prompt_count(artifact)
    if artifact.stage in MODEL_LEVEL_STAGES:
        return {
            "prompt_count": prompt_count,
            "prompt_bucket_id": artifact.prompt_bucket_id,
            "comparison_group": artifact.comparison_group,
        }
    return {}


def _artifact_runtime_load_ms(artifact: BenchmarkArtifact) -> float | None:
    details = artifact.details or {}
    runtime_load_time_ms = details.get("runtime_load_time_ms")
    if runtime_load_time_ms is None and artifact.stage == Stage.load:
        return artifact.steady_state_time_ms
    return float(runtime_load_time_ms) if runtime_load_time_ms is not None else None


def _artifact_jit_compile_ms(artifact: BenchmarkArtifact) -> float | None:
    details = artifact.details or {}
    jit_compile_time_ms = details.get("jit_compile_time_ms")
    if jit_compile_time_ms is None and artifact.stage == Stage.compile:
        return artifact.compile_time_ms
    return float(jit_compile_time_ms) if jit_compile_time_ms is not None else None


def _artifact_setup_time_ms(artifact: BenchmarkArtifact) -> float | None:
    details = artifact.details or {}
    setup_time_ms = details.get("setup_time_ms")
    return float(setup_time_ms) if setup_time_ms is not None else None


def _artifact_precompiled_load_time_ms(artifact: BenchmarkArtifact) -> float | None:
    details = artifact.details or {}
    precompiled_load_time_ms = details.get("precompiled_load_time_ms")
    return float(precompiled_load_time_ms) if precompiled_load_time_ms is not None else None


def _artifact_cast_manifest(artifact: BenchmarkArtifact) -> dict[str, Any]:
    details = artifact.details or {}
    manifest = details.get("cast_manifest")
    return dict(manifest) if isinstance(manifest, dict) else {}


def _artifact_project_ref(artifact: BenchmarkArtifact) -> str | None:
    manifest = _artifact_cast_manifest(artifact)
    if manifest.get("project_ref"):
        return str(manifest.get("project_ref"))
    details = artifact.details or {}
    if details.get("project_ref"):
        return str(details.get("project_ref"))
    kf_settings = artifact.kf_settings if isinstance(artifact.kf_settings, dict) else {}
    if kf_settings.get("project_ref"):
        return str(kf_settings.get("project_ref"))
    return None


def _artifact_export_selection_policy(artifact: BenchmarkArtifact) -> str | None:
    manifest = _artifact_cast_manifest(artifact)
    policy = manifest.get("selection_policy")
    if policy is not None:
        return str(policy)
    details = artifact.details or {}
    policy = details.get("export_selection_policy")
    if policy is not None:
        return str(policy)
    return None


def _artifact_selected_kernel_metadata(artifact: BenchmarkArtifact) -> dict[str, dict[str, Any]]:
    manifest = _artifact_cast_manifest(artifact)
    metadata = manifest.get("selected_kernel_metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _artifact_selected_ops(artifact: BenchmarkArtifact) -> list[str]:
    manifest = _artifact_cast_manifest(artifact)
    selected_ops = manifest.get("selected_ops")
    if isinstance(selected_ops, list):
        return [str(item) for item in selected_ops if str(item).strip()]
    details = artifact.details or {}
    raw_selected_ops = details.get("selected_ops")
    if isinstance(raw_selected_ops, list):
        return [str(item) for item in raw_selected_ops if str(item).strip()]
    return []


def _artifact_selected_kernel_ids(artifact: BenchmarkArtifact) -> dict[str, str]:
    kernel_metadata = _artifact_selected_kernel_metadata(artifact)
    selected: dict[str, str] = {}
    for op_name, metadata in sorted(kernel_metadata.items()):
        if not isinstance(metadata, dict):
            continue
        candidate_id = metadata.get("candidate_id")
        if candidate_id is not None:
            selected[str(op_name)] = str(candidate_id)
    return selected


def _artifact_selected_kernel_paths(artifact: BenchmarkArtifact) -> dict[str, str]:
    kernel_metadata = _artifact_selected_kernel_metadata(artifact)
    selected: dict[str, str] = {}
    for op_name, metadata in sorted(kernel_metadata.items()):
        if not isinstance(metadata, dict):
            continue
        kernel_path = metadata.get("kernel_source_path")
        if kernel_path is not None:
            selected[str(op_name)] = str(kernel_path)
    return selected


def _artifact_selected_source_hashes(artifact: BenchmarkArtifact) -> dict[str, str]:
    kernel_metadata = _artifact_selected_kernel_metadata(artifact)
    selected: dict[str, str] = {}
    for op_name, metadata in sorted(kernel_metadata.items()):
        if not isinstance(metadata, dict):
            continue
        selected_hash = metadata.get("selected_source_hash")
        if selected_hash is not None:
            selected[str(op_name)] = str(selected_hash)
    if selected:
        return selected
    details = artifact.details or {}
    raw_selected_hashes = details.get("selected_source_hashes")
    if isinstance(raw_selected_hashes, dict):
        return {
            str(op_name): str(source_hash)
            for op_name, source_hash in sorted(raw_selected_hashes.items())
        }
    kf_settings = artifact.kf_settings if isinstance(artifact.kf_settings, dict) else {}
    raw_selected_hashes = kf_settings.get("selected_source_hashes")
    if isinstance(raw_selected_hashes, dict):
        return {
            str(op_name): str(source_hash)
            for op_name, source_hash in sorted(raw_selected_hashes.items())
        }
    return {}


def _artifact_evidence_tiers(artifact: BenchmarkArtifact) -> dict[str, str]:
    kernel_metadata = _artifact_selected_kernel_metadata(artifact)
    tiers: dict[str, str] = {}
    for op_name, metadata in sorted(kernel_metadata.items()):
        if not isinstance(metadata, dict):
            continue
        evidence_tier = metadata.get("evidence_tier")
        if evidence_tier is not None:
            tiers[str(op_name)] = str(evidence_tier)
    return tiers


def _artifact_benchmark_evidence_refs(artifact: BenchmarkArtifact) -> dict[str, Any]:
    kernel_metadata = _artifact_selected_kernel_metadata(artifact)
    refs: dict[str, Any] = {}
    for op_name, metadata in sorted(kernel_metadata.items()):
        if not isinstance(metadata, dict):
            continue
        reference = metadata.get("benchmark_reference")
        if isinstance(reference, dict):
            refs[str(op_name)] = dict(reference)
            continue
        artifact_path = metadata.get("benchmark_artifact_path")
        row_ref = metadata.get("benchmark_row_ref")
        if artifact_path is not None or row_ref is not None:
            refs[str(op_name)] = {
                "artifact_path": artifact_path,
                "row_ref": row_ref,
            }
    return refs


def _artifact_rejected_export_candidate_summary(artifact: BenchmarkArtifact) -> dict[str, Any]:
    manifest = _artifact_cast_manifest(artifact)
    summary = manifest.get("rejected_candidate_summary")
    return dict(summary) if isinstance(summary, dict) else {}


def _artifact_export_paper_eligible(artifact: BenchmarkArtifact) -> bool | None:
    manifest = _artifact_cast_manifest(artifact)
    if "export_paper_eligible" in manifest:
        return bool(manifest.get("export_paper_eligible"))
    return None


def _artifact_uses_non_deployment_evidence(artifact: BenchmarkArtifact) -> bool | None:
    tiers = _artifact_evidence_tiers(artifact)
    if not tiers:
        return None
    return any(tier != "deployment" for tier in tiers.values())


def _artifact_loaded_kernels(artifact: BenchmarkArtifact) -> list[str]:
    details = artifact.details or {}
    raw = details.get("loaded_kernels")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _artifact_precompiled_vs_jit_path(artifact: BenchmarkArtifact) -> dict[str, str]:
    details = artifact.details or {}
    raw = details.get("precompiled_vs_jit_path")
    if isinstance(raw, dict):
        return {str(op_name): str(mode) for op_name, mode in sorted(raw.items())}
    return {}


def _artifact_detail_int(artifact: BenchmarkArtifact, key: str) -> int | None:
    details = artifact.details or {}
    value = details.get(key)
    return int(value) if value is not None else None


def _required_project_ref(artifact: BenchmarkArtifact) -> str | None:
    if str(artifact.model_id).strip().lower() == "qwen35a3b":
        return REQUIRED_QWEN_PROJECT_REF
    return None


def _matching_workload_hash(items: tuple[BenchmarkArtifact, BenchmarkArtifact, BenchmarkArtifact]) -> bool:
    hashes = [str(item.workload_hash or "").strip() for item in items]
    return all(hashes) and len(set(hashes)) == 1


def _speedup_index(artifacts: list[tuple[Path, BenchmarkArtifact]]) -> tuple[dict[tuple[Stage, str | None], BenchmarkArtifact], dict[tuple[Stage, str | None], BenchmarkArtifact]]:
    eager = {
        (artifact.stage, artifact.comparison_group): artifact
        for _, artifact in artifacts
        if artifact.variant == Variant.eager
    }
    compiled = {
        (artifact.stage, artifact.comparison_group): artifact
        for _, artifact in artifacts
        if artifact.variant == Variant.torch_compile
    }
    return eager, compiled


def _build_row(
    path: Path,
    artifact: BenchmarkArtifact,
    *,
    eager_baselines: dict[tuple[Stage, str | None], BenchmarkArtifact],
    compile_baselines: dict[tuple[Stage, str | None], BenchmarkArtifact],
) -> SummaryRow:
    baseline = eager_baselines.get((artifact.stage, artifact.comparison_group))
    compile_baseline = compile_baselines.get((artifact.stage, artifact.comparison_group))
    speedup_vs_eager = safe_speedup(
        baseline.steady_state_time_ms if baseline is not None else None,
        artifact.steady_state_time_ms,
    )
    speedup_vs_compile = safe_speedup(
        compile_baseline.steady_state_time_ms if compile_baseline is not None else None,
        artifact.steady_state_time_ms,
    )
    total_tps, prefill_tps, decode_tps = _artifact_tps(artifact)
    return SummaryRow(
        variant=artifact.variant,
        stage=artifact.stage,
        comparison_group=artifact.comparison_group,
        configured_batch_size=artifact.configured_batch_size,
        prompt_bucket_id=artifact.prompt_bucket_id,
        artifact_path=str(path.resolve()),
        reused=artifact.reused,
        reused_from_artifact=artifact.reused_from_artifact,
        reused_from_artifact_hash=artifact.reused_from_artifact_hash,
        correctness_status=artifact.correctness_status,
        sample_count=len(artifact.latency_samples_ms),
        p05_ms=artifact.latency_summary.p05_ms,
        median_ms=artifact.latency_summary.median_ms,
        mean_ms=artifact.latency_summary.mean_ms,
        p95_ms=artifact.latency_summary.p95_ms,
        min_ms=artifact.latency_summary.min_ms,
        max_ms=artifact.latency_summary.max_ms,
        stddev_ms=artifact.latency_summary.stddev_ms,
        steady_state_time_ms=artifact.steady_state_time_ms,
        compile_time_ms=artifact.compile_time_ms,
        runtime_load_time_ms=_artifact_runtime_load_ms(artifact),
        setup_time_ms=_artifact_setup_time_ms(artifact),
        jit_compile_time_ms=_artifact_jit_compile_ms(artifact),
        precompiled_load_time_ms=_artifact_precompiled_load_time_ms(artifact),
        total_tokens_per_second=total_tps,
        prefill_tokens_per_second=prefill_tps,
        decode_tokens_per_second=decode_tps,
        prompt_count=_artifact_prompt_count(artifact),
        prompt_suite_hash=(artifact.details or {}).get("prompt_suite_hash", artifact.workload_hash),
        generated_token_equality_status=_generated_token_equality_status(artifact),
        speedup_vs_eager=speedup_vs_eager,
        speedup_vs_torch_compile=speedup_vs_compile,
        paper_eligible=artifact.paper_eligible,
        claim_eligible=_claim_eligible(artifact),
        fallback_count=artifact.fallback_count,
        kernel_hit_count=artifact.kernel_hit_count,
        exception_fallback_count=_artifact_detail_int(artifact, "exception_fallback_count"),
        contiguous_copy_count=_artifact_detail_int(artifact, "contiguous_copy_count"),
        adaptation_count=_artifact_detail_int(artifact, "adaptation_count"),
        cast_package_path=artifact.cast_package_path,
        cast_package_hash=artifact.cast_package_hash,
        project_ref=_artifact_project_ref(artifact),
        export_selection_policy=_artifact_export_selection_policy(artifact),
        loaded_kernels=_artifact_loaded_kernels(artifact),
        precompiled_vs_jit_path=_artifact_precompiled_vs_jit_path(artifact),
        selected_ops=_artifact_selected_ops(artifact),
        selected_kernel_ids=_artifact_selected_kernel_ids(artifact),
        selected_kernel_paths=_artifact_selected_kernel_paths(artifact),
        selected_source_hashes=_artifact_selected_source_hashes(artifact),
        evidence_tiers=_artifact_evidence_tiers(artifact),
        benchmark_evidence_refs=_artifact_benchmark_evidence_refs(artifact),
        rejected_export_candidate_summary=_artifact_rejected_export_candidate_summary(artifact),
        export_paper_eligible=_artifact_export_paper_eligible(artifact),
        uses_non_deployment_evidence=_artifact_uses_non_deployment_evidence(artifact),
        coverage=_artifact_coverage(artifact),
    )


def _serialize_section_rows(rows: list[SummaryRow]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in rows]


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _group_key(artifact: BenchmarkArtifact) -> tuple[str, str | None]:
    return artifact.stage.value, artifact.comparison_group


def _group_label(group: str | None, *, multi_group: bool) -> str:
    if multi_group and group:
        return f" [group={group}]"
    return ""


def _model_total_generate_groups(
    artifacts: list[tuple[Path, BenchmarkArtifact]],
) -> dict[str | None, dict[Variant, BenchmarkArtifact]]:
    grouped: dict[str | None, dict[Variant, BenchmarkArtifact]] = {}
    for _, artifact in artifacts:
        if artifact.stage != Stage.total_generate:
            continue
        if artifact.benchmark_mode not in {BenchmarkMode.e2e_model, BenchmarkMode.deployment}:
            continue
        grouped.setdefault(artifact.comparison_group, {})[artifact.variant] = artifact
    return grouped


def _operator_groups(
    artifacts: list[tuple[Path, BenchmarkArtifact]],
) -> dict[str | None, dict[Variant, BenchmarkArtifact]]:
    grouped: dict[str | None, dict[Variant, BenchmarkArtifact]] = {}
    for _, artifact in artifacts:
        if artifact.stage != Stage.operator:
            continue
        grouped.setdefault(artifact.comparison_group, {})[artifact.variant] = artifact
    return grouped


def _timed_samples_meet_minimum(artifact: BenchmarkArtifact) -> bool:
    return artifact.timed_run_count > 0 and len(artifact.latency_samples_ms) >= artifact.timed_run_count


def _deployment_path_used(artifact: BenchmarkArtifact) -> bool:
    details = artifact.details or {}
    return bool(
        artifact.cast_package_path
        and artifact.cast_package_hash
        and details.get("runtime_patch_enabled") is not False
    )


def _collect_claims_and_failures(
    artifacts: list[tuple[Path, BenchmarkArtifact]],
) -> tuple[list[str], list[str], list[str]]:
    paper_claims: list[str] = []
    forbidden_claims: list[str] = []
    failures: list[str] = []

    total_generate_groups = _model_total_generate_groups(artifacts)
    multi_group = len(total_generate_groups) > 1
    saw_model_win = False

    for comparison_group, variants in sorted(total_generate_groups.items(), key=lambda item: (str(item[0]))):
        eager = variants.get(Variant.eager)
        compiled = variants.get(Variant.torch_compile)
        kf = variants.get(Variant.kf_cast)
        group_suffix = _group_label(comparison_group, multi_group=multi_group)

        if eager is None or compiled is None or kf is None:
            if compiled is None:
                forbidden_claims.append(
                    f'Model-speedup claim unsupported{group_suffix}: missing torch_compile baseline.'
                )
            if eager is None:
                forbidden_claims.append(
                    f'Model-speedup claim unsupported{group_suffix}: missing eager baseline.'
                )
            if kf is None:
                forbidden_claims.append(
                    f'Model-speedup claim unsupported{group_suffix}: missing Kernel Forge deployment result.'
                )
            continue

        speedup_vs_eager = safe_speedup(eager.steady_state_time_ms, kf.steady_state_time_ms)
        speedup_vs_compile = safe_speedup(compiled.steady_state_time_ms, kf.steady_state_time_ms)
        beats_eager = bool(speedup_vs_eager is not None and speedup_vs_eager > 1.0)
        beats_compile = bool(speedup_vs_compile is not None and speedup_vs_compile > 1.0)
        exact_match = kf.correctness_status == CorrectnessStatus.passed
        no_synthetic = not any(item.synthetic_workload for item in (eager, compiled, kf))
        comparable = all(item.correctness_status in {CorrectnessStatus.reference, CorrectnessStatus.passed} for item in (eager, compiled, kf))
        timed_ok = all(_timed_samples_meet_minimum(item) for item in (eager, compiled, kf))
        prompt_frozen = _matching_workload_hash((eager, compiled, kf))
        fallback_reported_zero = (kf.fallback_count is not None) and (kf.fallback_count == 0)
        deployment_used = _deployment_path_used(kf)
        all_paper_eligible = all(item.paper_eligible for item in (eager, compiled, kf))
        all_successful = (
            eager.correctness_status in {CorrectnessStatus.reference, CorrectnessStatus.passed}
            and compiled.correctness_status == CorrectnessStatus.passed
            and kf.correctness_status == CorrectnessStatus.passed
        )
        export_selection_policy = _artifact_export_selection_policy(kf)
        selection_policy_ok = export_selection_policy == REQUIRED_CAST_SELECTION_POLICY
        project_ref = _artifact_project_ref(kf)
        required_project_ref = _required_project_ref(kf)
        required_project_ok = required_project_ref is None or project_ref == required_project_ref

        if kf.correctness_status == CorrectnessStatus.failed:
            paper_claims.append(f"Unsafe speedup; not a valid model-speedup result.{group_suffix}")
            forbidden_claims.append(
                f'Model-speedup claim unsupported{group_suffix}: Kernel Forge output tokens do not exactly match eager.'
            )
            failures.append(f"Correctness failure for Kernel Forge total_generate{group_suffix}.")
            continue

        if kf.fallback_count is None:
            forbidden_claims.append(
                f'Model-speedup claim unsupported{group_suffix}: Kernel Forge fallback count was not reported.'
            )
            failures.append(f"Hidden fallback state for Kernel Forge total_generate{group_suffix}.")
        elif kf.fallback_count > 0:
            forbidden_claims.append(
                f'Model-speedup claim unsupported{group_suffix}: Kernel Forge fallback count is nonzero ({kf.fallback_count}).'
            )
            failures.append(f"Kernel Forge fallback count was nonzero on total_generate{group_suffix}.")

        if not beats_compile and beats_eager and exact_match:
            paper_claims.append(
                f"Kernel Forge beats eager but does not beat torch.compile on this workload.{group_suffix}"
            )
        if not beats_compile and exact_match:
            failures.append(f"Kernel Forge regressed against torch.compile on total_generate{group_suffix}.")
        if not beats_eager and exact_match:
            failures.append(f"Kernel Forge regressed against eager on total_generate{group_suffix}.")

        gate_ok = all(
            [
                all_successful,
                exact_match,
                fallback_reported_zero,
                beats_compile,
                timed_ok,
                prompt_frozen,
                no_synthetic,
                deployment_used,
                selection_policy_ok,
                required_project_ok,
                all_paper_eligible,
                comparable,
            ]
        )

        if gate_ok:
            paper_claims.append(f"Kernel Forge improves model throughput on this workload.{group_suffix}")
            saw_model_win = True
        else:
            reasons: list[str] = []
            if not all_successful:
                reasons.append("not all variants ran successfully")
            if not exact_match:
                reasons.append("correctness did not exactly match eager")
            if not fallback_reported_zero:
                reasons.append("fallback reporting was hidden or nonzero")
            if not beats_compile:
                reasons.append("kf_cast does not beat torch.compile")
            if not timed_ok:
                reasons.append("timed sample count does not meet the suite minimum")
            if not prompt_frozen:
                reasons.append("prompt suite hash is missing or mismatched across variants")
            if not no_synthetic:
                reasons.append("synthetic data was used")
            if not deployment_used:
                reasons.append("deployment path was not used")
            if not selection_policy_ok:
                reasons.append(
                    f"export selection policy was not {REQUIRED_CAST_SELECTION_POLICY}"
                )
            if not required_project_ok and required_project_ref is not None:
                reasons.append(
                    f"cast package project_ref {project_ref or 'missing'} did not match required {required_project_ref}"
                )
            if not all_paper_eligible:
                reasons.append("one or more artifacts are not paper eligible")
            forbidden_claims.append(
                f'Model-speedup claim unsupported{group_suffix}: ' + "; ".join(dict.fromkeys(reasons))
            )

    operator_groups = _operator_groups(artifacts)
    any_operator_win = False
    for comparison_group, variants in sorted(operator_groups.items(), key=lambda item: str(item[0])):
        eager = variants.get(Variant.eager)
        kf = variants.get(Variant.kf_cast)
        if eager is None or kf is None:
            continue
        speedup_vs_eager = safe_speedup(eager.steady_state_time_ms, kf.steady_state_time_ms)
        if speedup_vs_eager is not None and speedup_vs_eager > 1.0 and kf.paper_eligible:
            any_operator_win = True

    if any_operator_win and total_generate_groups and not saw_model_win:
        paper_claims.append("Operator wins did not translate into an end-to-end model win.")

    for path, artifact in artifacts:
        details = artifact.details or {}
        execution_status = str(details.get("execution_status", "")).lower()
        if artifact.correctness_status == CorrectnessStatus.failed:
            failures.append(
                f"{artifact.variant.value} {artifact.stage.value} failed correctness or execution: {artifact.correctness_message or path.name}"
            )
        if execution_status == "failed":
            failures.append(
                f"{artifact.variant.value} {artifact.stage.value} reported execution failure at {path.resolve()}"
            )
        if artifact.variant == Variant.kf_cast and artifact.stage in PERFORMANCE_STAGES and artifact.fallback_count and artifact.fallback_count > 0:
            failures.append(
                f"Kernel Forge used {artifact.fallback_count} fallbacks during {artifact.stage.value}."
            )

    return (
        list(dict.fromkeys(paper_claims)),
        list(dict.fromkeys(forbidden_claims)),
        list(dict.fromkeys(failures)),
    )


def _export_selection_items(rows: list[SummaryRow]) -> list[dict[str, Any]]:
    preferred_rows = [
        row
        for row in rows
        if row.variant == Variant.kf_cast and row.stage == Stage.total_generate and row.cast_package_hash
    ]
    if not preferred_rows:
        preferred_rows = [
            row
            for row in rows
            if row.variant == Variant.kf_cast and row.cast_package_hash
        ]

    items: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for row in preferred_rows:
        key = (row.cast_package_hash, row.project_ref, row.comparison_group)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "artifact_path": row.artifact_path,
                "variant": row.variant.value,
                "stage": row.stage.value,
                "comparison_group": row.comparison_group,
                "project_ref": row.project_ref,
                "cast_package_path": row.cast_package_path,
                "cast_package_hash": row.cast_package_hash,
                "selection_policy": row.export_selection_policy,
                "selected_ops": row.selected_ops,
                "selected_kernel_ids": row.selected_kernel_ids,
                "selected_kernel_paths": row.selected_kernel_paths,
                "selected_source_hashes": row.selected_source_hashes,
                "benchmark_evidence_refs": row.benchmark_evidence_refs,
                "evidence_tiers": row.evidence_tiers,
                "rejected_export_candidate_summary": row.rejected_export_candidate_summary,
                "export_paper_eligible": row.export_paper_eligible,
                "uses_non_deployment_evidence": row.uses_non_deployment_evidence,
            }
        )
    return items


def _section_payloads(
    rows: list[SummaryRow],
    *,
    paper_claims: list[str],
    forbidden_claims: list[str],
    failure_regressions: list[str],
) -> dict[str, Any]:
    operator_rows = [row for row in rows if row.stage == Stage.operator]
    model_rows = [row for row in rows if row.stage in MODEL_LEVEL_STAGES]
    deployment_rows = [row for row in rows if row.variant == Variant.kf_cast and row.stage != Stage.operator]
    offline_rows = [row for row in rows if row.stage in {Stage.load, Stage.compile, Stage.warmup}]
    correctness_rows = [
        {
            "artifact_path": row.artifact_path,
            "variant": row.variant.value,
            "stage": row.stage.value,
            "comparison_group": row.comparison_group,
            "correctness_status": row.correctness_status.value,
            "generated_token_equality_status": row.generated_token_equality_status,
            "paper_eligible": row.paper_eligible,
            "fallback_count": row.fallback_count,
            "kernel_hit_count": row.kernel_hit_count,
        }
        for row in rows
        if row.stage in PERFORMANCE_STAGES
    ]
    coverage_items = [
        {
            "artifact_path": row.artifact_path,
            "variant": row.variant.value,
            "stage": row.stage.value,
            "comparison_group": row.comparison_group,
            "prompt_count": row.prompt_count,
            "prompt_suite_hash": row.prompt_suite_hash,
            "coverage": row.coverage,
        }
        for row in rows
        if row.coverage or row.prompt_count is not None or row.prompt_suite_hash
    ]
    export_items = _export_selection_items(rows)
    return {
        SECTION_OPERATOR: {"title": "Operator Benchmark", "rows": _serialize_section_rows(operator_rows)},
        SECTION_MODEL: {"title": "End-to-End Model Benchmark", "rows": _serialize_section_rows(model_rows)},
        SECTION_DEPLOYMENT: {"title": "Deployment/Runtime Benchmark", "rows": _serialize_section_rows(deployment_rows)},
        SECTION_OFFLINE: {"title": "Offline Costs", "rows": _serialize_section_rows(offline_rows)},
        SECTION_CORRECTNESS: {"title": "Correctness", "items": correctness_rows},
        SECTION_COVERAGE: {"title": "Coverage", "items": coverage_items},
        SECTION_EXPORT: {"title": "Export/CAST Selection", "items": export_items},
        SECTION_FAILURES: {"title": "Failures/Regressions", "items": list(failure_regressions)},
        SECTION_PAPER_CLAIMS: {"title": "Paper-Eligible Claims", "items": list(paper_claims)},
        SECTION_FORBIDDEN: {"title": "Forbidden/Unsupported Claims", "items": list(forbidden_claims)},
    }


def _markdown_table(
    rows: list[SummaryRow],
) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| Variant | Stage | Group | P05 ms | Median ms | Mean ms | P95 ms | Total TPS | Prefill TPS | Decode TPS | Speedup vs eager | Speedup vs torch.compile | Correctness | Prompt count | Prompt suite hash | Token equality | Fallbacks | Kernel hits | Artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.variant.value,
                    row.stage.value,
                    row.comparison_group or "-",
                    str(row.p05_ms) if row.p05_ms is not None else "n/a",
                    str(row.median_ms),
                    str(row.mean_ms) if row.mean_ms is not None else "n/a",
                    str(row.p95_ms),
                    f"{row.total_tokens_per_second:.4f}" if row.total_tokens_per_second is not None else "n/a",
                    f"{row.prefill_tokens_per_second:.4f}" if row.prefill_tokens_per_second is not None else "n/a",
                    f"{row.decode_tokens_per_second:.4f}" if row.decode_tokens_per_second is not None else "n/a",
                    f"{row.speedup_vs_eager:.4f}" if row.speedup_vs_eager is not None else "n/a",
                    f"{row.speedup_vs_torch_compile:.4f}" if row.speedup_vs_torch_compile is not None else "n/a",
                    row.correctness_status.value,
                    str(row.prompt_count) if row.prompt_count is not None else "-",
                    row.prompt_suite_hash or "-",
                    row.generated_token_equality_status or "-",
                    str(row.fallback_count) if row.fallback_count is not None else "-",
                    str(row.kernel_hit_count) if row.kernel_hit_count is not None else "-",
                    row.artifact_path or "-",
                ]
            )
            + " |"
        )
    return lines


def _markdown_list(items: list[Any]) -> list[str]:
    if not items:
        return ["No items."]
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- `{item.get('artifact_path', 'n/a')}`: {json.dumps(item, sort_keys=True)}")
        else:
            lines.append(f"- {item}")
    return lines


def _write_summary_csv(path: Path, rows: list[SummaryRow]) -> Path:
    fieldnames = list(SummaryRow.model_fields.keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.model_dump(mode="json")
            writer.writerow({key: _serialize_csv_value(payload.get(key)) for key in fieldnames})
    return path


def summarize_run(run_dir: str | Path) -> SummaryArtifact:
    root = Path(run_dir)
    benchmark_artifacts = _sorted_benchmark_artifacts(root)
    eager_baselines, compile_baselines = _speedup_index(benchmark_artifacts)
    rows = [
        _build_row(
            path,
            artifact,
            eager_baselines=eager_baselines,
            compile_baselines=compile_baselines,
        )
        for path, artifact in benchmark_artifacts
    ]
    paper_claims, forbidden_claims, failure_regressions = _collect_claims_and_failures(benchmark_artifacts)
    sections = _section_payloads(
        rows,
        paper_claims=paper_claims,
        forbidden_claims=forbidden_claims,
        failure_regressions=failure_regressions,
    )

    base = benchmark_artifacts[0][1]
    markdown_lines = [f"# Paper Benchmark Report for {base.run_id}", ""]
    for section_key in (
        SECTION_OPERATOR,
        SECTION_MODEL,
        SECTION_DEPLOYMENT,
        SECTION_OFFLINE,
        SECTION_CORRECTNESS,
        SECTION_COVERAGE,
        SECTION_EXPORT,
        SECTION_FAILURES,
        SECTION_PAPER_CLAIMS,
        SECTION_FORBIDDEN,
    ):
        section = sections[section_key]
        markdown_lines.append(f"## {section['title']}")
        markdown_lines.append("")
        if "rows" in section:
            if section_key == SECTION_OPERATOR:
                section_rows = [row for row in rows if row.stage == Stage.operator]
            elif section_key == SECTION_MODEL:
                section_rows = [row for row in rows if row.stage in MODEL_LEVEL_STAGES]
            elif section_key == SECTION_DEPLOYMENT:
                section_rows = [row for row in rows if row.variant == Variant.kf_cast and row.stage != Stage.operator]
            else:
                section_rows = [row for row in rows if row.stage in {Stage.load, Stage.compile, Stage.warmup}]
            markdown_lines.extend(_markdown_table(section_rows))
        else:
            markdown_lines.extend(_markdown_list(section.get("items", [])))
        markdown_lines.append("")

    markdown_path = root / "reports" / "summary.md"
    markdown_path.write_text("\n".join(markdown_lines).rstrip() + "\n", encoding="utf-8")
    csv_path = _write_summary_csv(root / "reports" / "summary.csv", rows)

    summary_common = base.model_dump(
        mode="json",
        exclude={
            "artifact_type",
            "variant",
            "stage",
            "comparison_group",
            "configured_batch_size",
            "prompt_bucket_id",
            "latency_samples_ms",
            "latency_summary",
            "sample_records",
            "correctness_status",
            "correctness_message",
            "fallback_count",
            "kernel_hit_count",
            "compile_time_ms",
            "steady_state_time_ms",
            "prompt_id",
            "prompt_hash",
            "token_count",
            "details",
        },
    )
    summary = SummaryArtifact(
        **summary_common,
        artifact_type="summary_report",
        variant=None,
        stage=None,
        comparison_group=None,
        configured_batch_size=None,
        prompt_bucket_id=None,
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        rows=rows,
        summary_markdown_path=str(markdown_path),
        summary_csv_path=str(csv_path),
        sections=sections,
        paper_eligible_claims=paper_claims,
        forbidden_claims=forbidden_claims,
        failure_regressions=failure_regressions,
        raw_artifact_paths=[str(path.resolve()) for path, _ in benchmark_artifacts],
    )
    write_json_artifact(root / "reports" / "summary.json", summary)
    return summary
