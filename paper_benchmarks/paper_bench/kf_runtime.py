from __future__ import annotations

import copy
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .provenance import sha256_bytes, sha256_path

REQUIRED_RUNTIME_STATS_KEYS = (
    "patched_calls",
    "kernel_launches_attempted",
    "kernel_launches_succeeded",
    "kernel_launches_failed",
    "fallbacks_to_original",
    "exception_fallback_count",
    "contiguous_copy_count",
    "adaptation_count",
    "per_op",
)


@dataclass(frozen=True)
class KfRuntimeSettings:
    cast_package_path: str | None = None
    require_precompiled: bool = False
    allow_jit: bool = True
    fail_on_fallback: bool = True
    record_runtime_stats: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "cast_package_path": self.cast_package_path,
            "require_precompiled": self.require_precompiled,
            "allow_jit": self.allow_jit,
            "fail_on_fallback": self.fail_on_fallback,
            "record_runtime_stats": self.record_runtime_stats,
        }


def runtime_settings_from_dict(payload: dict[str, Any] | None) -> KfRuntimeSettings:
    if not payload:
        return KfRuntimeSettings()
    return KfRuntimeSettings(
        cast_package_path=str(payload["cast_package_path"]) if payload.get("cast_package_path") else None,
        require_precompiled=bool(payload.get("require_precompiled", False)),
        allow_jit=bool(payload.get("allow_jit", True)),
        fail_on_fallback=bool(payload.get("fail_on_fallback", True)),
        record_runtime_stats=bool(payload.get("record_runtime_stats", True)),
    )


def inspect_cast_package(cast_path: str | Path) -> dict[str, Any]:
    cast_file = Path(cast_path)
    if not cast_file.exists():
        raise FileNotFoundError(f"Cast package not found: {cast_file}")

    from kernelforge.run_cast import verify_checksums

    with zipfile.ZipFile(cast_file, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise RuntimeError(f"CAST package is missing manifest.json: {cast_file}")

        verify_checksums(archive)

        header = json.loads(archive.read("HEADER.json")) if "HEADER.json" in names else {}
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict):
            raise RuntimeError("CAST manifest.json must contain a JSON object")

        selection_manifest = (
            json.loads(archive.read("selection_manifest.json"))
            if "selection_manifest.json" in names
            else {}
        )
        if selection_manifest and not isinstance(selection_manifest, dict):
            raise RuntimeError("CAST selection_manifest.json must contain a JSON object")

        ops = manifest.get("ops", [])
        if not isinstance(ops, list):
            raise RuntimeError("CAST manifest ops must be a list")

        ops_by_name: dict[str, dict[str, Any]] = {}
        for index, op in enumerate(ops):
            if not isinstance(op, dict):
                raise RuntimeError(f"CAST manifest op at index {index} is not an object")
            op_name = str(op.get("name") or "").strip()
            if not op_name:
                raise RuntimeError(f"CAST manifest op at index {index} is missing name")
            if op_name in ops_by_name:
                raise RuntimeError(f"CAST manifest contains duplicate op entry for {op_name}")
            ops_by_name[op_name] = op

        selected_ops_raw = manifest.get("selected_ops")
        if isinstance(selected_ops_raw, list):
            selected_ops = [str(item) for item in selected_ops_raw if str(item).strip()]
        else:
            selected_ops = sorted(ops_by_name)

        selected_kernel_metadata = (
            manifest.get("selected_kernel_metadata", {})
            if isinstance(manifest.get("selected_kernel_metadata"), dict)
            else {}
        )

        kernel_source_hashes: dict[str, str] = {}
        selected_source_hashes: dict[str, str] = {}
        precompiled_binary_hashes: dict[str, str] = {}
        selected_op_reports: dict[str, dict[str, Any]] = {}

        for op_name in selected_ops:
            op = ops_by_name.get(op_name)
            if op is None:
                raise RuntimeError(f"CAST manifest selected op is missing from ops: {op_name}")

            cuda_source = str(op.get("cuda_source") or "").strip()
            if not cuda_source:
                raise RuntimeError(f"CAST manifest selected op {op_name} is missing cuda_source")
            if cuda_source not in names:
                raise RuntimeError(
                    f"CAST package missing selected kernel source for {op_name}: {cuda_source}"
                )

            source_hash = sha256_bytes(archive.read(cuda_source))
            kernel_source_hashes[cuda_source] = source_hash

            metadata = (
                selected_kernel_metadata.get(op_name)
                if isinstance(selected_kernel_metadata.get(op_name), dict)
                else {}
            )
            expected_hash = str(metadata.get("selected_source_hash") or "").strip()
            if expected_hash and expected_hash != source_hash:
                raise RuntimeError(
                    f"CAST selected source hash mismatch for {op_name}: "
                    f"manifest={expected_hash} archive={source_hash}"
                )
            selected_source_hashes[op_name] = expected_hash or source_hash

            precompiled = op.get("precompiled", {})
            if not isinstance(precompiled, dict):
                raise RuntimeError(f"CAST manifest precompiled entry for {op_name} must be an object")

            op_precompiled: dict[str, str] = {}
            for sm_name, relpath in sorted(precompiled.items()):
                relpath_str = str(relpath or "").strip()
                if not relpath_str:
                    raise RuntimeError(
                        f"CAST manifest precompiled path for {op_name} / {sm_name} is empty"
                    )
                if relpath_str not in names:
                    raise RuntimeError(
                        f"CAST package missing precompiled binary for {op_name} / {sm_name}: {relpath_str}"
                    )
                precompiled_binary_hashes[relpath_str] = sha256_bytes(archive.read(relpath_str))
                op_precompiled[str(sm_name)] = relpath_str

            selected_op_reports[op_name] = {
                "cuda_source": cuda_source,
                "kernel_source_hash": source_hash,
                "selected_source_hash": selected_source_hashes[op_name],
                "precompiled": op_precompiled,
            }

    loadability_blockers: list[str] = []
    archive_entries = sorted(names)
    model_entrypoints = (
        manifest.get("model_entrypoints", {})
        if isinstance(manifest.get("model_entrypoints"), dict)
        else {}
    )
    has_build_model = bool(model_entrypoints.get("build_model"))
    has_load_weights = bool(model_entrypoints.get("load_weights"))
    has_model_class = bool(str(manifest.get("model_class") or "").strip())
    weight_file = str(manifest.get("weight_file") or "").strip()
    model_init_args = manifest.get("model_init_args")

    if "model.py" not in names:
        loadability_blockers.append("model.py missing from package")
    if weight_file and weight_file not in names:
        loadability_blockers.append(f"weight_file missing from package: {weight_file}")
    if not has_model_class and not has_build_model and not has_load_weights:
        loadability_blockers.append(
            "manifest declares neither model_class nor build_model/load_weights entrypoints"
        )
    if (
        has_model_class
        and not weight_file
        and "model_config.json" not in names
        and not model_init_args
        and not has_build_model
        and not has_load_weights
    ):
        loadability_blockers.append(
            "model_class export missing weight_file, model_config.json, model_init_args, and fallback entrypoints"
        )

    return {
        "cast_path": str(cast_file),
        "cast_package_sha256": sha256_path(cast_file),
        "header": header if isinstance(header, dict) else {},
        "manifest": manifest,
        "selection_manifest": selection_manifest if isinstance(selection_manifest, dict) else {},
        "selected_ops": list(selected_ops),
        "selected_kernel_metadata": selected_kernel_metadata,
        "selected_source_hashes": dict(sorted(selected_source_hashes.items())),
        "selected_op_reports": dict(sorted(selected_op_reports.items())),
        "kernel_source_hashes": dict(sorted(kernel_source_hashes.items())),
        "precompiled_binary_hashes": dict(sorted(precompiled_binary_hashes.items())),
        "precompiled_binaries": sorted(precompiled_binary_hashes),
        "target_sm_versions": list(header.get("runtime", {}).get("target_sm_versions", []) or []),
        "gpu_name": header.get("runtime", {}).get("gpu_name"),
        "gpu_capability": header.get("runtime", {}).get("gpu_capability"),
        "selection_policy": manifest.get("selection_policy"),
        "export_paper_eligible": bool(manifest.get("export_paper_eligible")),
        "uses_non_deployment_evidence": any(
            str(meta.get("evidence_tier") or "") not in {"deployment", "manual_override"}
            for meta in selected_kernel_metadata.values()
            if isinstance(meta, dict)
        ),
        "loadability_blockers": loadability_blockers,
        "checksum_verified": True,
        "archive_entries": archive_entries,
    }


def _zip_member_hashes(cast_path: str | Path, manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    kernel_source_hashes: dict[str, str] = {}
    precompiled_hashes: dict[str, str] = {}
    with zipfile.ZipFile(cast_path, "r") as archive:
        for op in manifest.get("ops", []) if isinstance(manifest.get("ops"), list) else []:
            cuda_source = op.get("cuda_source")
            if isinstance(cuda_source, str):
                try:
                    kernel_source_hashes[cuda_source] = sha256_bytes(archive.read(cuda_source))
                except KeyError:
                    pass
            precompiled = op.get("precompiled", {})
            if isinstance(precompiled, dict):
                for relpath in precompiled.values():
                    if not isinstance(relpath, str):
                        continue
                    try:
                        precompiled_hashes[relpath] = sha256_bytes(archive.read(relpath))
                    except KeyError:
                        continue
    return dict(sorted(kernel_source_hashes.items())), dict(sorted(precompiled_hashes.items()))


def validate_runtime_stats_api(
    model: Any,
    *,
    stats_getter: Callable[[Any], dict[str, Any]] | None = None,
    stats_reset: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from kernelforge import get_runtime_stats as exported_get_runtime_stats
    from kernelforge import reset_runtime_stats as exported_reset_runtime_stats

    stats_before = get_cast_runtime_stats(model, stats_getter=stats_getter)
    missing_before = [key for key in REQUIRED_RUNTIME_STATS_KEYS if key not in stats_before]
    if missing_before:
        raise RuntimeError(
            "Runtime stats API is missing required keys before reset: "
            + ", ".join(missing_before)
        )

    stats_after_reset = reset_cast_runtime_stats(model, stats_reset=stats_reset)
    missing_after = [key for key in REQUIRED_RUNTIME_STATS_KEYS if key not in stats_after_reset]
    if missing_after:
        raise RuntimeError(
            "Runtime stats API is missing required keys after reset: "
            + ", ".join(missing_after)
        )

    return {
        "get_runtime_stats_exists": callable(exported_get_runtime_stats),
        "reset_runtime_stats_exists": callable(exported_reset_runtime_stats),
        "required_keys": list(REQUIRED_RUNTIME_STATS_KEYS),
        "stats_before_reset": stats_before,
        "stats_after_reset": stats_after_reset,
        "fallback_count_after_reset": int(stats_after_reset.get("fallbacks_to_original", 0) or 0),
        "per_op_stats_after_reset": stats_after_reset.get("per_op", {}),
    }


def get_cast_runtime_stats(model: Any, stats_getter: Callable[[Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    if stats_getter is not None:
        stats = stats_getter(model)
        return copy.deepcopy(stats) if isinstance(stats, dict) else {}
    from kernelforge.run_cast import get_runtime_stats

    return get_runtime_stats(model)


def reset_cast_runtime_stats(model: Any, stats_reset: Callable[[Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    if stats_reset is not None:
        return stats_reset(model)
    from kernelforge.run_cast import reset_runtime_stats

    return reset_runtime_stats(model)


def load_cast_model(
    cast_path: str | Path,
    *,
    device: str | None = None,
    settings: KfRuntimeSettings | dict[str, Any] | None = None,
    loader: Callable[..., Any] | None = None,
    stats_getter: Callable[[Any], dict[str, Any]] | None = None,
    stats_reset: Callable[[Any], dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    runtime_settings = runtime_settings_from_dict(settings if isinstance(settings, dict) else (
        settings.as_dict() if isinstance(settings, KfRuntimeSettings) else None
    ))
    effective_cast_path = runtime_settings.cast_package_path or str(cast_path)
    if not effective_cast_path:
        raise ValueError("Kernel Forge deployment benchmarking requires a cast package path.")

    cast_file = Path(effective_cast_path)
    if not cast_file.exists():
        raise FileNotFoundError(f"Cast package not found: {cast_file}")

    inspection = inspect_cast_package(cast_file)
    manifest = inspection["manifest"]
    kernel_source_hashes = inspection["kernel_source_hashes"]
    precompiled_binary_hashes = inspection["precompiled_binary_hashes"]

    if runtime_settings.require_precompiled:
        missing_precompiled = [
            op_name
            for op_name, report in inspection.get("selected_op_reports", {}).items()
            if not (isinstance(report, dict) and report.get("precompiled"))
        ]
        if missing_precompiled:
            raise RuntimeError(
                "Precompiled kernels were required, but the CAST package does not include "
                "precompiled binaries for selected ops: "
                + ", ".join(sorted(missing_precompiled))
            )

    if loader is None:
        from kernelforge.run_cast import load_cast as loader

    started = time.perf_counter()
    model = loader(
        str(cast_file),
        device=device,
        require_precompiled=runtime_settings.require_precompiled,
        allow_jit=runtime_settings.allow_jit,
        record_runtime_stats=runtime_settings.record_runtime_stats,
    )
    total_load_ms = (time.perf_counter() - started) * 1000.0

    runtime_report = getattr(model, "_kf_runtime_report", None)
    runtime_report = dict(runtime_report) if isinstance(runtime_report, dict) else {}
    runtime_stats = (
        get_cast_runtime_stats(model, stats_getter=stats_getter)
        if runtime_settings.record_runtime_stats
        else {}
    )
    runtime_stats_api = (
        validate_runtime_stats_api(model, stats_getter=stats_getter, stats_reset=stats_reset)
        if runtime_settings.record_runtime_stats
        else {}
    )
    setup_time_ms = float(runtime_report.get("setup_time_ms", total_load_ms))
    jit_compile_time_ms = float(runtime_report.get("jit_compile_time_ms", 0.0) or 0.0)
    runtime_load_time_ms = float(runtime_report.get("runtime_load_time_ms", total_load_ms))
    precompiled_load_time_ms = float(runtime_report.get("precompiled_load_time_ms", 0.0) or 0.0)

    selected_ops = runtime_report.get("selected_ops")
    loaded_kernels = runtime_report.get("loaded_kernels")
    load_modes = runtime_report.get("load_modes")

    if not isinstance(selected_ops, list):
        selected_ops = []
    if not isinstance(loaded_kernels, list):
        loaded_kernels = []
    if not isinstance(load_modes, dict):
        load_modes = {
            entry["op_name"]: entry.get("load_mode")
            for entry in runtime_report.get("op_reports", [])
            if isinstance(entry, dict) and "op_name" in entry
        }

    return model, {
        "cast_package_path": str(cast_file),
        "cast_package_hash": sha256_path(cast_file),
        "manifest": manifest,
        "cast_manifest": manifest,
        "project_ref": manifest.get("project_ref"),
        "cast_inspection": inspection,
        "kernel_source_hashes": kernel_source_hashes,
        "precompiled_binary_hashes": precompiled_binary_hashes,
        "selected_source_hashes": inspection.get("selected_source_hashes", {}),
        "runtime_report": runtime_report,
        "runtime_stats": runtime_stats,
        "runtime_stats_api": runtime_stats_api,
        "runtime_stats_after_reset": runtime_stats_api.get("stats_after_reset", {}),
        "runtime_load_time_ms": runtime_load_time_ms,
        "setup_time_ms": setup_time_ms,
        "load_time_ms": max(setup_time_ms, 0.0),
        "jit_compile_time_ms": jit_compile_time_ms,
        "precompiled_load_time_ms": precompiled_load_time_ms,
        "precompiled_vs_jit_path": dict(load_modes),
        "selected_ops": list(selected_ops),
        "loaded_kernels": list(loaded_kernels),
        "runtime_stats_enabled": bool(runtime_settings.record_runtime_stats),
        "runtime_patch_enabled": bool(runtime_report.get("runtime_patch_enabled", bool(getattr(model, "_cast_functional_patches", {})))),
        "patched_call_count": int(runtime_stats.get("patched_calls", 0) or 0),
        "kernel_hit_count": int(runtime_stats.get("kernel_launches_succeeded", 0) or 0),
        "kernel_launches_attempted": int(runtime_stats.get("kernel_launches_attempted", 0) or 0),
        "kernel_launches_succeeded": int(runtime_stats.get("kernel_launches_succeeded", 0) or 0),
        "kernel_launches_failed": int(runtime_stats.get("kernel_launches_failed", 0) or 0),
        "fallback_count": int(runtime_stats.get("fallbacks_to_original", 0) or 0),
        "exception_fallback_count": int(runtime_stats.get("exception_fallback_count", 0) or 0),
        "contiguous_copy_count": int(runtime_stats.get("contiguous_copy_count", 0) or 0),
        "adaptation_count": int(runtime_stats.get("adaptation_count", 0) or 0),
    }


def build_kf_runtime_details(
    runtime_meta: dict[str, Any],
    runtime_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = runtime_stats if runtime_stats is not None else runtime_meta.get("runtime_stats", {})
    return {
        "cast_manifest": runtime_meta.get("cast_manifest"),
        "project_ref": runtime_meta.get("project_ref"),
        "loaded_kernels": runtime_meta.get("loaded_kernels", []),
        "selected_ops": runtime_meta.get("selected_ops", []),
        "kernel_source_hashes": runtime_meta.get("kernel_source_hashes", {}),
        "selected_source_hashes": runtime_meta.get("selected_source_hashes", {}),
        "precompiled_binary_hashes": runtime_meta.get("precompiled_binary_hashes", {}),
        "runtime_load_time_ms": runtime_meta.get("runtime_load_time_ms"),
        "setup_time_ms": runtime_meta.get("setup_time_ms"),
        "jit_compile_time_ms": runtime_meta.get("jit_compile_time_ms"),
        "precompiled_load_time_ms": runtime_meta.get("precompiled_load_time_ms"),
        "precompiled_vs_jit_path": runtime_meta.get("precompiled_vs_jit_path", {}),
        "runtime_stats_enabled": runtime_meta.get("runtime_stats_enabled"),
        "runtime_stats_api": runtime_meta.get("runtime_stats_api", {}),
        "runtime_patch_enabled": runtime_meta.get("runtime_patch_enabled"),
        "patched_call_count": int(stats.get("patched_calls", runtime_meta.get("patched_call_count", 0)) or 0),
        "kernel_launches_attempted": int(stats.get("kernel_launches_attempted", runtime_meta.get("kernel_launches_attempted", 0)) or 0),
        "kernel_launches_succeeded": int(stats.get("kernel_launches_succeeded", runtime_meta.get("kernel_launches_succeeded", 0)) or 0),
        "kernel_launches_failed": int(stats.get("kernel_launches_failed", runtime_meta.get("kernel_launches_failed", 0)) or 0),
        "fallbacks_to_original": int(stats.get("fallbacks_to_original", runtime_meta.get("fallback_count", 0)) or 0),
        "exception_fallback_count": int(stats.get("exception_fallback_count", runtime_meta.get("exception_fallback_count", 0)) or 0),
        "contiguous_copy_count": int(stats.get("contiguous_copy_count", runtime_meta.get("contiguous_copy_count", 0)) or 0),
        "adaptation_count": int(stats.get("adaptation_count", runtime_meta.get("adaptation_count", 0)) or 0),
        "per_op_stats": stats.get("per_op", {}),
    }
