from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import load_json_artifact, write_json_artifact
from .provenance import safe_sha256_path
from .schema import BenchmarkArtifact, BenchmarkMode, Stage, Variant


class CacheValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CacheRequest:
    signature: dict[str, Any]
    target_path: Path


def _sorted_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sorted_jsonable(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_sorted_jsonable(item) for item in value]
    return value


def _device_name(artifact: BenchmarkArtifact | dict[str, Any]) -> str | None:
    gpu_names = getattr(artifact, "gpu_names", None)
    if gpu_names is None and isinstance(artifact, dict):
        gpu_names = artifact.get("gpu_names")
    if isinstance(gpu_names, list) and gpu_names:
        return str(gpu_names[0])
    return None


def _artifact_sample_matrix(artifact: BenchmarkArtifact) -> list[dict[str, Any]]:
    return stage_sample_matrix(artifact.stage, artifact.sample_records)


def _selected_source_hashes(payload: BenchmarkArtifact | dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        kf_settings = payload.get("kf_settings", {})
        exported_hashes = payload.get("exported_kernel_hashes", {})
    else:
        kf_settings = payload.kf_settings
        exported_hashes = payload.exported_kernel_hashes
    return (
        details.get("selected_source_hashes")
        or (kf_settings if isinstance(kf_settings, dict) else {}).get("selected_source_hashes")
        or exported_hashes
        or {}
    )


def _project_ref(payload: BenchmarkArtifact | dict[str, Any], details: dict[str, Any]) -> str | None:
    manifest = details.get("cast_manifest")
    if isinstance(manifest, dict) and manifest.get("project_ref"):
        return str(manifest.get("project_ref"))
    if details.get("project_ref"):
        return str(details.get("project_ref"))
    kf_settings = payload.get("kf_settings", {}) if isinstance(payload, dict) else payload.kf_settings
    if isinstance(kf_settings, dict) and kf_settings.get("project_ref"):
        return str(kf_settings.get("project_ref"))
    return None


def _kf_runtime_settings(payload: BenchmarkArtifact | dict[str, Any]) -> dict[str, Any]:
    kf_settings = payload.get("kf_settings", {}) if isinstance(payload, dict) else payload.kf_settings
    if not isinstance(kf_settings, dict):
        return {}
    return {
        "require_precompiled": bool(kf_settings.get("require_precompiled", False)),
        "allow_jit": bool(kf_settings.get("allow_jit", True)),
        "fail_on_fallback": bool(kf_settings.get("fail_on_fallback", True)),
        "record_runtime_stats": bool(kf_settings.get("record_runtime_stats", True)),
        "placement_profile": str(kf_settings.get("placement_profile")) if kf_settings.get("placement_profile") else None,
        "device_map": kf_settings.get("device_map"),
        "max_memory": kf_settings.get("max_memory") if isinstance(kf_settings.get("max_memory"), dict) else None,
    }


def _toolchain_signature(payload: BenchmarkArtifact | dict[str, Any]) -> dict[str, Any]:
    toolchain_status = payload.get("toolchain_status", {}) if isinstance(payload, dict) else payload.toolchain_status
    if not isinstance(toolchain_status, dict):
        toolchain_status = {}
    return {
        "jit_ready": toolchain_status.get("jit_ready"),
        "nvcc_path": toolchain_status.get("nvcc_path"),
        "ninja_path": toolchain_status.get("ninja_path"),
    }


def _is_kf_variant(value: Variant | str | None) -> bool:
    if isinstance(value, Variant):
        return value == Variant.kf_cast
    return str(value or "") == Variant.kf_cast.value


def _cache_variant(payload: BenchmarkArtifact | dict[str, Any], fallback: Variant | str | None = None) -> Variant | str | None:
    if isinstance(payload, dict):
        return payload.get("variant", fallback)
    return payload.variant if payload.variant is not None else fallback


def _kf_cache_fields(payload: BenchmarkArtifact | dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    variant = _cache_variant(payload)
    if not _is_kf_variant(variant):
        return {
            "cast_package_hash": None,
            "kernel_source_hashes": {},
            "selected_source_hashes": {},
            "kf_runtime_settings": {},
            "project_ref": None,
            "kf_artifact_hash": None,
            "kf_artifact_kind": None,
            "toolchain_status": {},
        }
    if isinstance(payload, dict):
        cast_package_hash = payload.get("cast_package_hash")
        kf_kernel_hashes = payload.get("exported_kernel_hashes", {}) or details.get("kernel_source_hashes") or {}
        kf_artifact_hash = payload.get("kf_artifact_hash")
        kf_artifact_kind = payload.get("kf_artifact_kind")
    else:
        cast_package_hash = payload.cast_package_hash
        kf_kernel_hashes = payload.exported_kernel_hashes or details.get("kernel_source_hashes") or {}
        kf_artifact_hash = payload.kf_artifact_hash
        kf_artifact_kind = payload.kf_artifact_kind
    return {
        "cast_package_hash": cast_package_hash,
        "kernel_source_hashes": kf_kernel_hashes,
        "selected_source_hashes": _selected_source_hashes(payload, details),
        "kf_runtime_settings": _kf_runtime_settings(payload),
        "project_ref": _project_ref(payload, details),
        "kf_artifact_hash": kf_artifact_hash,
        "kf_artifact_kind": kf_artifact_kind,
        "toolchain_status": _toolchain_signature(payload),
    }


def _normalize_reused_artifact_for_variant(artifact: BenchmarkArtifact) -> BenchmarkArtifact:
    if artifact.variant == Variant.kf_cast:
        return artifact
    return artifact.model_copy(
        update={
            "cast_package_path": None,
            "cast_package_hash": None,
            "kf_artifact_path": None,
            "kf_artifact_hash": None,
            "kf_artifact_kind": None,
            "exported_kernel_hashes": {},
            "kf_settings": {},
            "toolchain_status": {},
        }
    )


def stage_sample_matrix(stage: Stage, sample_records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for record in sample_records:
        if stage in {Stage.prefill, Stage.decode, Stage.total_generate, Stage.warmup, Stage.compile}:
            matrix.append(
                {
                    "prompt_ids": list(record.get("prompt_ids", [])),
                    "batch_size": record.get("batch_size"),
                    "prompt_lengths": list(record.get("prompt_lengths", [])),
                    "generated_lengths": list(record.get("generated_lengths", [])),
                    "generated_token_count": record.get("generated_token_count"),
                    "decode_generated_token_count": record.get("decode_generated_token_count"),
                }
            )
        elif stage == Stage.operator:
            matrix.append(
                {
                    "entry_name": record.get("entry_name"),
                    "entry_hash": record.get("entry_hash"),
                    "metadata": record.get("metadata"),
                }
            )
        elif stage == Stage.load:
            matrix.append(
                {
                    "sample_index": record.get("sample_index"),
                    "load_source": record.get("load_source"),
                    "latency_ms": record.get("latency_ms"),
                }
            )
    return matrix


def artifact_cache_signature(artifact: BenchmarkArtifact) -> dict[str, Any]:
    details = artifact.details or {}
    kf_fields = _kf_cache_fields(artifact, details)
    selection_policy = details.get("selection_policy")
    if selection_policy is None:
        selection_policy = {
            "comparison_group": artifact.comparison_group,
            "prompt_bucket_id": artifact.prompt_bucket_id,
            "selected_prompt_ids_hash": details.get("selected_prompt_ids_hash"),
            "entry_set_hash": details.get("entry_set_hash"),
        }
    return _sorted_jsonable(
        {
            "benchmark_harness_version": artifact.benchmark_harness_version,
            "git_available": artifact.git_available,
            "git_commit": artifact.git_commit,
            "git_dirty": artifact.git_dirty,
            "git_dirty_summary": artifact.git_dirty_summary,
            "git_untracked_summary": artifact.git_untracked_summary,
            "model_id": artifact.model_id,
            "model_config_hash": artifact.model_config_hash,
            "model_path_hash": artifact.model_path_hash or artifact.config_hashes.get("model_snapshot_id"),
            "quantization": artifact.quantization,
            "quantization_config_hash": artifact.quantization_config_hash,
            "placement_profile": artifact.placement_profile,
            "suite_hash": artifact.suite_hash,
            "workload_hash": artifact.workload_hash,
            "workload_slug": artifact.workload_slug,
            "cache_mode": artifact.cache_mode,
            "entry_set_hash": details.get("entry_set_hash") or artifact.config_hashes.get("entry_set"),
            "cast_package_hash": kf_fields["cast_package_hash"],
            "kernel_source_hashes": kf_fields["kernel_source_hashes"],
            "selected_source_hashes": kf_fields["selected_source_hashes"],
            "variant": artifact.variant.value,
            "stage": artifact.stage.value,
            "device_name": _device_name(artifact),
            "cuda_version": artifact.cuda_version,
            "pytorch_version": artifact.pytorch_version,
            "compile_settings": artifact.compile_settings,
            "kf_runtime_settings": kf_fields["kf_runtime_settings"],
            "project_ref": kf_fields["project_ref"],
            "generation_settings": details.get("generation_settings"),
            "warmup_count": artifact.warmup_count,
            "timed_run_count": artifact.timed_run_count,
            "configured_batch_size": artifact.configured_batch_size,
            "prompt_bucket_id": artifact.prompt_bucket_id,
            "comparison_group": artifact.comparison_group,
            "selection_policy": selection_policy,
            "sample_matrix": _artifact_sample_matrix(artifact),
            "benchmark_mode": artifact.benchmark_mode.value if artifact.benchmark_mode is not None else None,
            "kf_artifact_hash": kf_fields["kf_artifact_hash"],
            "kf_artifact_kind": kf_fields["kf_artifact_kind"],
            "toolchain_status": kf_fields["toolchain_status"],
        }
    )


def make_cache_request(
    common_fields: dict[str, Any],
    *,
    variant: Variant,
    stage: Stage,
    details: dict[str, Any] | None = None,
    configured_batch_size: int | None = None,
    prompt_bucket_id: str | None = None,
    comparison_group: str | None = None,
    sample_matrix: list[dict[str, Any]] | None = None,
    warmup_count: int | None = None,
    timed_run_count: int | None = None,
) -> dict[str, Any]:
    detail_payload = details or {}
    request_payload = {**common_fields, "variant": variant.value}
    kf_fields = _kf_cache_fields(request_payload, detail_payload)
    selection_policy = detail_payload.get("selection_policy")
    if selection_policy is None:
        selection_policy = {
            "comparison_group": comparison_group,
            "prompt_bucket_id": prompt_bucket_id,
            "selected_prompt_ids_hash": detail_payload.get("selected_prompt_ids_hash"),
            "entry_set_hash": detail_payload.get("entry_set_hash") or common_fields.get("config_hashes", {}).get("entry_set"),
        }
    return _sorted_jsonable(
        {
            "benchmark_harness_version": common_fields.get("benchmark_harness_version"),
            "git_available": common_fields.get("git_available"),
            "git_commit": common_fields.get("git_commit"),
            "git_dirty": common_fields.get("git_dirty"),
            "git_dirty_summary": common_fields.get("git_dirty_summary"),
            "git_untracked_summary": common_fields.get("git_untracked_summary"),
            "model_id": common_fields.get("model_id"),
            "model_config_hash": common_fields.get("model_config_hash"),
            "model_path_hash": common_fields.get("model_path_hash") or common_fields.get("config_hashes", {}).get("model_snapshot_id"),
            "quantization": common_fields.get("quantization"),
            "quantization_config_hash": common_fields.get("quantization_config_hash"),
            "placement_profile": common_fields.get("placement_profile"),
            "suite_hash": common_fields.get("suite_hash"),
            "workload_hash": common_fields.get("workload_hash"),
            "workload_slug": common_fields.get("workload_slug"),
            "cache_mode": common_fields.get("cache_mode"),
            "entry_set_hash": detail_payload.get("entry_set_hash") or common_fields.get("config_hashes", {}).get("entry_set"),
            "cast_package_hash": kf_fields["cast_package_hash"],
            "kernel_source_hashes": kf_fields["kernel_source_hashes"],
            "selected_source_hashes": kf_fields["selected_source_hashes"],
            "variant": variant.value,
            "stage": stage.value,
            "device_name": _device_name(common_fields),
            "cuda_version": common_fields.get("cuda_version"),
            "pytorch_version": common_fields.get("pytorch_version"),
            "compile_settings": common_fields.get("compile_settings") or {},
            "kf_runtime_settings": kf_fields["kf_runtime_settings"],
            "project_ref": kf_fields["project_ref"],
            "generation_settings": detail_payload.get("generation_settings"),
            "warmup_count": common_fields.get("warmup_count") if warmup_count is None else int(warmup_count),
            "timed_run_count": common_fields.get("timed_run_count") if timed_run_count is None else int(timed_run_count),
            "configured_batch_size": configured_batch_size,
            "prompt_bucket_id": prompt_bucket_id,
            "comparison_group": comparison_group,
            "selection_policy": selection_policy,
            "sample_matrix": sample_matrix,
            "benchmark_mode": common_fields.get("benchmark_mode"),
            "kf_artifact_hash": kf_fields["kf_artifact_hash"],
            "kf_artifact_kind": kf_fields["kf_artifact_kind"],
            "toolchain_status": kf_fields["toolchain_status"],
        }
    )


def validate_reusable_artifact_payload(payload: Any) -> BenchmarkArtifact:
    if not isinstance(payload, dict):
        raise CacheValidationError("legacy numeric-only benchmark results cannot be imported into paper runs")
    artifact_type = payload.get("artifact_type")
    if artifact_type == "summary_report":
        raise CacheValidationError("summary-only results cannot be imported into paper runs")
    if artifact_type != "benchmark_result":
        raise CacheValidationError("legacy numeric-only benchmark results cannot be imported into paper runs")
    artifact = load_json_artifact_from_payload(payload)
    if not isinstance(artifact, BenchmarkArtifact):
        raise CacheValidationError("only benchmark_result artifacts may be reused")
    if not artifact.latency_samples_ms:
        raise CacheValidationError("results with no raw samples cannot be imported into paper runs")
    if not artifact.sample_records:
        raise CacheValidationError("results with no raw samples cannot be imported into paper runs")
    return artifact


def load_json_artifact_from_payload(payload: dict[str, Any]) -> BenchmarkArtifact:
    artifact = load_json_artifact_from_dict(payload)
    if not isinstance(artifact, BenchmarkArtifact):
        raise CacheValidationError("only benchmark_result artifacts may be reused")
    return artifact


def load_json_artifact_from_dict(payload: dict[str, Any]):
    # Delayed import to keep this module free of circular write/load calls.
    from .schema import validate_artifact_payload

    return validate_artifact_payload(payload)


def validate_reusable_artifact_path(path: str | Path) -> BenchmarkArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_reusable_artifact_payload(payload)


def _signature_matches(candidate: Any, request: Any) -> bool:
    if request is None:
        return True
    if isinstance(request, dict):
        if not isinstance(candidate, dict):
            return False
        for key, value in request.items():
            if key not in candidate:
                return False
            if not _signature_matches(candidate[key], value):
                return False
        return True
    if isinstance(request, list):
        if not isinstance(candidate, list) or len(candidate) != len(request):
            return False
        return all(_signature_matches(c_item, r_item) for c_item, r_item in zip(candidate, request, strict=True))
    return candidate == request


def find_matching_reusable_artifact(
    search_root: str | Path,
    request_signature: dict[str, Any],
    *,
    exclude_run_dir: str | Path | None = None,
) -> tuple[Path, BenchmarkArtifact] | None:
    root = Path(search_root)
    if not root.exists():
        return None
    excluded = Path(exclude_run_dir).resolve() if exclude_run_dir else None
    for path in sorted(root.rglob("metrics/*.json")):
        if excluded is not None and excluded in path.resolve().parents:
            continue
        try:
            artifact = validate_reusable_artifact_path(path)
        except CacheValidationError:
            continue
        candidate_signature = artifact_cache_signature(artifact)
        if _signature_matches(candidate_signature, _sorted_jsonable(request_signature)):
            return path, artifact
    return None


def copy_reused_artifact(
    source_path: str | Path,
    artifact: BenchmarkArtifact,
    target_path: str | Path,
    *,
    run_id: str,
    timestamp_utc: str,
) -> Path:
    artifact = _normalize_reused_artifact_for_variant(artifact)
    reused_artifact = artifact.model_copy(
        update={
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "reused": True,
            "reused_from_artifact": str(Path(source_path).resolve()),
            "reused_from_artifact_hash": safe_sha256_path(source_path),
            "cache_reuse_status": "signature_matched",
        }
    )
    return write_json_artifact(target_path, reused_artifact)


def copy_reused_artifact_set(
    requests: Iterable[CacheRequest],
    *,
    search_root: str | Path,
    run_id: str,
    timestamp_utc: str,
    exclude_run_dir: str | Path | None = None,
) -> bool:
    resolved: list[tuple[Path, BenchmarkArtifact, Path]] = []
    for request in requests:
        match = find_matching_reusable_artifact(
            search_root,
            request.signature,
            exclude_run_dir=exclude_run_dir,
        )
        if match is None:
            return False
        source_path, artifact = match
        resolved.append((source_path, artifact, request.target_path))
    for source_path, artifact, target_path in resolved:
        copy_reused_artifact(
            source_path,
            artifact,
            target_path,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
        )
    return True
