from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .cast_selection import (
    NoEligibleCastKernelsError,
    POLICY_AUTO_BEST_FASTEST_VALID,
    select_project_export_kernels,
)
from .provenance import collect_git_info, safe_sha256_path


class CastExportPlanError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _selected_kernel_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "selected_kernel_id": None,
            "selected_kernel_node_id": None,
            "selected_kernel_file": None,
            "selected_kernel_source_kind": "missing",
        }

    candidates = [path.stem, path.parent.name]
    kernel_id = None
    kernel_node_id = None
    for candidate in candidates:
        match = re.fullmatch(r"kernel_(\d+)", str(candidate))
        if match:
            kernel_id = str(candidate)
            kernel_node_id = int(match.group(1))
            break

    parts = set(path.parts)
    source_kind = "unknown"
    if "runtime_kernels" in parts:
        source_kind = "deployment_runtime"
    elif "trees" in parts and "kernels" in parts:
        source_kind = "search_tree"
    elif "generated" in parts and "individual_op_kernels" in parts:
        source_kind = "generated_root"

    return {
        "selected_kernel_id": kernel_id or path.name,
        "selected_kernel_node_id": kernel_node_id,
        "selected_kernel_file": path.name,
        "selected_kernel_source_kind": source_kind,
    }


def _resolve_kernel_path(raw_path: str, *, repo_root: Path, project_root: Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    for base in (repo_root, project_root):
        resolved = (base / candidate).resolve(strict=False)
        if resolved.exists():
            return resolved
    return (repo_root / candidate).resolve(strict=False)


def _default_project_ref(project_root: Path) -> str:
    return f"project/{quote(project_root.name, safe='-._')}/"


def _rejection_summary(rejected_candidates: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for op_name, candidates in sorted(rejected_candidates.items()):
        reason_counts: dict[str, int] = {}
        for candidate in candidates or []:
            for reason in candidate.get("rejection_reasons", []) or []:
                reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0)) + 1
        summary[op_name] = {
            "total": len(candidates or []),
            "reasons": dict(sorted(reason_counts.items())),
        }
    return summary


def _normalize_selected_entry(
    op_name: str,
    entry: dict[str, Any],
    *,
    repo_root: Path,
    manual_override: bool = False,
) -> dict[str, Any]:
    kernel_path = Path(str(entry.get("kernel_source_path") or "")).resolve(strict=False) if entry.get("kernel_source_path") else None
    selected_source_hash = str(entry.get("selected_source_hash") or "") or safe_sha256_path(kernel_path)
    benchmark_reference = entry.get("benchmark_reference")
    if not isinstance(benchmark_reference, dict):
        benchmark_reference = {
            "artifact_path": entry.get("benchmark_artifact_path"),
            "row_ref": entry.get("benchmark_row_ref"),
        }

    normalized = {
        **dict(entry),
        "op": op_name,
        "candidate_id": str(entry.get("candidate_id") or f"{op_name}:selected"),
        "kernel_source_path": str(kernel_path) if kernel_path is not None else None,
        "kernel_source_repo_relpath": (
            str(entry.get("kernel_source_repo_relpath") or "")
            or _relative_or_none(kernel_path, repo_root)
        ),
        "selected_source_hash": selected_source_hash,
        "benchmark_reference": benchmark_reference,
        "benchmark_artifact_path": (
            benchmark_reference.get("artifact_path")
            if benchmark_reference
            else entry.get("benchmark_artifact_path")
        ),
        "benchmark_row_ref": (
            benchmark_reference.get("row_ref")
            if benchmark_reference
            else entry.get("benchmark_row_ref")
        ),
        "manual_override": bool(manual_override),
        **_selected_kernel_identity(kernel_path),
    }
    return normalized


def resolve_cast_export_plan(
    project_root: str | Path,
    *,
    project_ref: str | None = None,
    selection_policy: str = POLICY_AUTO_BEST_FASTEST_VALID,
    selected_kernels: dict[str, str] | None = None,
    allow_operator_only: bool = True,
    allow_micro_only: bool = False,
    unsafe_override: bool = False,
    allow_native_package: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    if selection_policy != POLICY_AUTO_BEST_FASTEST_VALID:
        raise CastExportPlanError(
            f"Unsupported selection policy {selection_policy!r}. "
            f"Expected {POLICY_AUTO_BEST_FASTEST_VALID!r}."
        )

    root = Path(project_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root else _repo_root()
    timestamp = _timestamp_utc()
    git_info = collect_git_info(repo)

    selection_report = select_project_export_kernels(
        root,
        allow_operator_only=allow_operator_only,
        allow_micro_only=allow_micro_only,
        unsafe_override=unsafe_override,
        fail_if_empty=False,
    )

    selected_ops: dict[str, dict[str, Any]] = {}
    manual_override_ops: list[str] = []
    skipped_ops: dict[str, dict[str, Any]] = {}
    rejected_summary = _rejection_summary(selection_report.get("rejected_candidates", {}))

    auto_selected = selection_report.get("selected_ops", {})
    for op_name, entry in sorted(auto_selected.items()):
        if not isinstance(entry, dict):
            continue
        selected_ops[op_name] = _normalize_selected_entry(op_name, entry, repo_root=repo)

    for op_name in selection_report.get("unselected_ops", []) or []:
        rejected = rejected_summary.get(op_name, {"total": 0, "reasons": {}})
        skipped_ops[op_name] = {
            "skip_reason": "no valid kernel matched auto_best_fastest_valid",
            "rejected_candidate_count": rejected["total"],
            "rejected_reason_counts": rejected["reasons"],
        }

    for op_name, raw_selection in sorted((selected_kernels or {}).items()):
        selected_value = str(raw_selection or "").strip()
        if not selected_value:
            continue
        if selected_value == "__PYTORCH__":
            selected_ops.pop(op_name, None)
            skipped_ops[op_name] = {
                "skip_reason": "manual native/no-kernel selection",
                "rejected_candidate_count": rejected_summary.get(op_name, {}).get("total", 0),
                "rejected_reason_counts": rejected_summary.get(op_name, {}).get("reasons", {}),
            }
            continue

        resolved = _resolve_kernel_path(selected_value, repo_root=repo, project_root=root)
        if not resolved.exists():
            raise CastExportPlanError(f"Manual kernel selection for {op_name} does not exist: {selected_value}")

        auto_entry = auto_selected.get(op_name)
        auto_kernel = (
            Path(str(auto_entry.get("kernel_source_path"))).resolve(strict=False)
            if isinstance(auto_entry, dict) and auto_entry.get("kernel_source_path")
            else None
        )
        if auto_kernel is not None and auto_kernel == resolved:
            selected_ops[op_name] = _normalize_selected_entry(op_name, auto_entry, repo_root=repo)
            skipped_ops.pop(op_name, None)
            continue

        manual_override_ops.append(op_name)
        selected_ops[op_name] = _normalize_selected_entry(
            op_name,
            {
                "op": op_name,
                "candidate_id": f"{op_name}:manual_override",
                "kernel_source_path": str(resolved),
                "evidence_tier": "manual_override",
                "selection_reason": "manual override from export UI",
                "paper_eligible": False,
                "benchmark_reference": None,
                "benchmark_artifact_path": None,
                "benchmark_row_ref": None,
            },
            repo_root=repo,
            manual_override=True,
        )
        skipped_ops.pop(op_name, None)

    selected_op_list = sorted(selected_ops)
    exportable = bool(selected_op_list) or allow_native_package
    export_paper_eligible = (
        bool(selected_op_list)
        and not manual_override_ops
        and all(bool(selected_ops[op_name].get("paper_eligible")) for op_name in selected_op_list)
        and not any(
            skipped.get("skip_reason") == "no valid kernel matched auto_best_fastest_valid"
            for skipped in skipped_ops.values()
        )
    )

    plan = {
        "selection_policy": selection_policy,
        "selection_policy_details": {
            "policy_name": selection_policy,
            "allow_operator_only": bool(allow_operator_only),
            "allow_micro_only": bool(allow_micro_only),
            "unsafe_override": bool(unsafe_override),
            "allow_native_package": bool(allow_native_package),
            "manual_override_ops": manual_override_ops,
        },
        "project_ref": project_ref or _default_project_ref(root),
        "project_root": str(root),
        "project_id": root.name,
        "project_name": root.name,
        "selected_ops": selected_ops,
        "selected_op_list": selected_op_list,
        "selected_kernel_map": {
            op_name: str(selected_ops[op_name]["kernel_source_path"])
            for op_name in selected_op_list
            if selected_ops[op_name].get("kernel_source_path")
        },
        "selected_op_count": len(selected_op_list),
        "skipped_ops": skipped_ops,
        "unselected_ops": list(selection_report.get("unselected_ops", []) or []),
        "rejected_candidates": selection_report.get("rejected_candidates", {}),
        "rejected_candidate_summary": rejected_summary,
        "exportable": exportable,
        "export_paper_eligible": export_paper_eligible,
        "timestamp": timestamp,
        "git_commit": git_info.get("git_commit"),
        "git_branch": git_info.get("git_branch"),
        "git_dirty": bool(git_info.get("git_dirty")),
        "git_status_short": list(git_info.get("git_status_short", []) or []),
    }

    if not exportable:
        raise NoEligibleCastKernelsError(
            "No kernels satisfied auto_best_fastest_valid. Review rejected candidates.",
            report=plan,
        )

    return plan


def build_cast_manifest_metadata(export_plan: dict[str, Any]) -> dict[str, Any]:
    selected_ops = export_plan.get("selected_ops", {})
    selected_kernel_metadata: dict[str, dict[str, Any]] = {}
    for op_name, entry in sorted((selected_ops or {}).items()):
        if not isinstance(entry, dict):
            continue
        selected_kernel_metadata[op_name] = {
            "candidate_id": entry.get("candidate_id"),
            "selected_kernel_id": entry.get("selected_kernel_id"),
            "selected_kernel_node_id": entry.get("selected_kernel_node_id"),
            "selected_kernel_file": entry.get("selected_kernel_file"),
            "selected_kernel_source_kind": entry.get("selected_kernel_source_kind"),
            "kernel_source_path": entry.get("kernel_source_path"),
            "kernel_source_repo_relpath": entry.get("kernel_source_repo_relpath"),
            "selected_source_hash": entry.get("selected_source_hash"),
            "benchmark_reference": entry.get("benchmark_reference"),
            "benchmark_artifact_path": entry.get("benchmark_artifact_path"),
            "benchmark_row_ref": entry.get("benchmark_row_ref"),
            "evidence_tier": entry.get("evidence_tier"),
            "selection_reason": entry.get("selection_reason"),
        }

    return {
        "selection_policy": export_plan.get("selection_policy", POLICY_AUTO_BEST_FASTEST_VALID),
        "selection_policy_details": export_plan.get("selection_policy_details", {}),
        "project_ref": export_plan.get("project_ref"),
        "project_root": export_plan.get("project_root"),
        "project_id": export_plan.get("project_id"),
        "selected_ops": list(export_plan.get("selected_op_list", []) or []),
        "selected_op_count": int(export_plan.get("selected_op_count", len(selected_kernel_metadata))),
        "selected_kernel_metadata": selected_kernel_metadata,
        "selected_kernel_by_op": {
            op_name: {
                "selected_kernel_id": meta.get("selected_kernel_id"),
                "selected_kernel_node_id": meta.get("selected_kernel_node_id"),
                "selected_kernel_file": meta.get("selected_kernel_file"),
                "selected_kernel_source_kind": meta.get("selected_kernel_source_kind"),
                "kernel_source_repo_relpath": meta.get("kernel_source_repo_relpath"),
                "selected_source_hash": meta.get("selected_source_hash"),
                "evidence_tier": meta.get("evidence_tier"),
                "candidate_id": meta.get("candidate_id"),
            }
            for op_name, meta in selected_kernel_metadata.items()
        },
        "selected_kernel_map": dict(export_plan.get("selected_kernel_map", {}) or {}),
        "rejected_candidate_summary": dict(export_plan.get("rejected_candidate_summary", {}) or {}),
        "skipped_ops": dict(export_plan.get("skipped_ops", {}) or {}),
        "export_paper_eligible": bool(export_plan.get("export_paper_eligible")),
        "timestamp": export_plan.get("timestamp"),
        "git_commit": export_plan.get("git_commit"),
        "git_branch": export_plan.get("git_branch"),
        "git_dirty": bool(export_plan.get("git_dirty")),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _file_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _detect_sm_version() -> tuple[str | None, dict[str, Any]]:
    try:
        import torch
    except Exception:
        return None, {}

    if not torch.cuda.is_available():
        return None, {}

    capability = torch.cuda.get_device_capability(0)
    return (
        f"sm_{capability[0]}{capability[1]}",
        {
            "name": str(torch.cuda.get_device_name(0)),
            "capability": [int(capability[0]), int(capability[1])],
        },
    )


def _resolve_weight_artifact(project_root: Path) -> tuple[str, int, bytes]:
    weight_archive_path = ""
    weight_size = 0
    weight_bytes = b""
    cfg = _read_json(project_root / "config.json")
    artifacts = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts"), dict) else {}
    for item in artifacts.get("weights", []) or []:
        if not isinstance(item, dict):
            continue
        relpath = str(item.get("relpath") or "").strip()
        if not relpath:
            continue
        for candidate in (project_root / relpath, project_root / "weights.pt"):
            if not candidate.exists():
                continue
            weight_bytes = candidate.read_bytes()
            digest = _file_sha256_bytes(weight_bytes)
            weight_archive_path = f"weights/{digest}{candidate.suffix or '.pt'}"
            weight_size = len(weight_bytes)
            return weight_archive_path, weight_size, weight_bytes

    for candidate_name in ("weights.pt", "model.pt"):
        candidate = project_root / candidate_name
        if not candidate.exists():
            continue
        weight_bytes = candidate.read_bytes()
        digest = _file_sha256_bytes(weight_bytes)
        weight_archive_path = f"weights/{digest}{candidate.suffix or '.pt'}"
        weight_size = len(weight_bytes)
        return weight_archive_path, weight_size, weight_bytes

    return weight_archive_path, weight_size, weight_bytes


def _extract_model_metadata(project_root: Path) -> tuple[bytes | None, str, dict[str, bool]]:
    model_py = project_root / "model.py"
    if not model_py.exists():
        return None, "", {}

    model_bytes = model_py.read_bytes()
    model_class_name = ""
    entrypoints = {
        "build_model": False,
        "load_weights": False,
        "sample_inputs": False,
    }
    try:
        tree = ast.parse(model_bytes.decode("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not model_class_name:
                model_class_name = node.name
            if isinstance(node, ast.FunctionDef) and node.name in entrypoints:
                entrypoints[node.name] = True
    except Exception:
        pass
    return model_bytes, model_class_name, entrypoints


def _maybe_bundle_model_config(
    *,
    project_root: Path,
    file_map: dict[str, bytes],
    model_bytes: bytes | None,
    model_class_name: str,
    weight_bytes: bytes,
) -> None:
    model_cfg_src = project_root / "model_config.json"
    if model_cfg_src.exists():
        file_map["model_config.json"] = model_cfg_src.read_bytes()
        return

    if not model_bytes or not model_class_name:
        return

    try:
        import io
        import os
        import sys
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        try:
            temp_file.write(model_bytes)
            temp_file.close()
            spec = importlib.util.spec_from_file_location("_kf_export_model", temp_file.name)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules["_kf_export_model"] = module
            spec.loader.exec_module(module)
            model_cls = getattr(module, model_class_name, None)
            if not model_cls or not hasattr(model_cls, "config_class"):
                return
            cfg = model_cls.config_class()
            cfg_dict = cfg.to_dict()
            if weight_bytes:
                import torch

                state_dict = torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True)
                for key in (
                    "classifier.1.weight",
                    "classifier.weight",
                    "fc.weight",
                    "head.weight",
                    "heads.head.weight",
                ):
                    tensor = state_dict.get(key)
                    if tensor is not None and len(getattr(tensor, "shape", [])) == 2:
                        cfg_dict["num_labels"] = int(tensor.shape[0])
                        break
            file_map["model_config.json"] = json.dumps(cfg_dict, indent=2).encode("utf-8")
        finally:
            sys.modules.pop("_kf_export_model", None)
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
    except Exception:
        return


def export_cast_package(
    project_root: str | Path,
    *,
    project_ref: str | None = None,
    export_plan: dict[str, Any] | None = None,
    selection_policy: str = POLICY_AUTO_BEST_FASTEST_VALID,
    selected_kernels: dict[str, str] | None = None,
    allow_operator_only: bool = True,
    allow_micro_only: bool = False,
    unsafe_override: bool = False,
    allow_native_package: bool = False,
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root else _repo_root()
    plan = export_plan or resolve_cast_export_plan(
        root,
        project_ref=project_ref,
        selection_policy=selection_policy,
        selected_kernels=selected_kernels,
        allow_operator_only=allow_operator_only,
        allow_micro_only=allow_micro_only,
        unsafe_override=unsafe_override,
        allow_native_package=allow_native_package,
        repo_root=repo,
    )
    manifest_selection_meta = build_cast_manifest_metadata(plan)

    kernel_map = {
        op_name: str(source_path)
        for op_name, source_path in (plan.get("selected_kernel_map", {}) or {}).items()
        if source_path
    }
    if not kernel_map and not allow_native_package:
        raise NoEligibleCastKernelsError(
            "No kernels satisfied auto_best_fastest_valid. Review rejected candidates.",
            report=plan,
        )

    exported_at = _timestamp_utc()
    weight_archive_path, weight_size, weight_bytes = _resolve_weight_artifact(root)
    file_map: dict[str, bytes] = {}

    model_bytes, model_class_name, model_entrypoints = _extract_model_metadata(root)
    if model_bytes is not None:
        file_map["model.py"] = model_bytes
    _maybe_bundle_model_config(
        project_root=root,
        file_map=file_map,
        model_bytes=model_bytes,
        model_class_name=model_class_name,
        weight_bytes=weight_bytes,
    )
    if weight_archive_path:
        file_map[weight_archive_path] = weight_bytes

    sm_version, gpu_info = _detect_sm_version()
    ops_manifest: list[dict[str, Any]] = []
    selected_manifest_ops: dict[str, dict[str, Any]] = {}
    precompiled_paths: list[str] = []

    for op_name in sorted(kernel_map):
        kernel_path = Path(kernel_map[op_name]).resolve()
        if not kernel_path.exists():
            raise CastExportPlanError(f"Selected kernel for {op_name} does not exist: {kernel_path}")
        kernel_bytes = kernel_path.read_bytes()
        cu_path = f"kernels/{op_name}/kernel.cu"
        wrapper_path = f"kernels/{op_name}/wrapper.py"
        file_map[cu_path] = kernel_bytes
        file_map[wrapper_path] = (
            f"# Cast dispatch wrapper for {op_name}\n# Generated by KernelForge\n".encode("utf-8")
        )

        op_precompiled: dict[str, str] = {}
        if sm_version:
            so_file = kernel_path.parent / f"{op_name}.so"
            if so_file.exists():
                rel = f"compiled/{sm_version}/{op_name}.so"
                file_map[rel] = so_file.read_bytes()
                op_precompiled[sm_version] = rel
                precompiled_paths.append(rel)

        selection_entry = plan.get("selected_ops", {}).get(op_name, {})
        selected_manifest_ops[op_name] = selection_entry if isinstance(selection_entry, dict) else {}
        op_kernel_identity = _selected_kernel_identity(kernel_path)
        ops_manifest.append(
            {
                "name": op_name,
                "kernel_dir": f"kernels/{op_name}/",
                "cuda_source": cu_path,
                "wrapper": wrapper_path,
                "precompiled": op_precompiled,
                "selected_kernel_id": op_kernel_identity.get("selected_kernel_id"),
                "selected_kernel_node_id": op_kernel_identity.get("selected_kernel_node_id"),
                "selection_evidence": selected_manifest_ops[op_name],
            }
        )

    loader_stub = b"# Cast vendored runtime loader\n# pip install cast for the full runtime\n"
    file_map["loader.py"] = loader_stub

    manifest_obj = {
        "project_name": root.name,
        "project_id": plan.get("project_id", root.name),
        "project_root": str(root),
        "project_ref": plan.get("project_ref") or project_ref or _default_project_ref(root),
        "exported_at": exported_at,
        "timestamp": exported_at,
        "git_commit": plan.get("git_commit"),
        "model_class": model_class_name,
        "model_entrypoints": model_entrypoints,
        "model_init_args": {},
        "weight_file": weight_archive_path,
        "ops": ops_manifest,
        "selection_policy": plan.get("selection_policy", selection_policy),
        "selection_policy_details": manifest_selection_meta.get("selection_policy_details", {}),
        "selected_ops": manifest_selection_meta.get("selected_ops", list(kernel_map)),
        "selected_op_count": manifest_selection_meta.get("selected_op_count", len(kernel_map)),
        "selected_kernel_map": manifest_selection_meta.get("selected_kernel_map", dict(kernel_map)),
        "selected_kernel_metadata": manifest_selection_meta.get("selected_kernel_metadata", selected_manifest_ops),
        "selected_kernel_by_op": manifest_selection_meta.get("selected_kernel_by_op", {}),
        "export_paper_eligible": bool(plan.get("export_paper_eligible")),
        "rejected_candidate_summary": plan.get("rejected_candidate_summary", {}),
        "skipped_ops": plan.get("skipped_ops", {}),
    }
    file_map["manifest.json"] = json.dumps(manifest_obj, indent=2).encode("utf-8")
    file_map["selection_manifest.json"] = json.dumps(plan, indent=2).encode("utf-8")

    checksum_lines = [f"{_file_sha256_bytes(file_map[path])}  {path}" for path in file_map]
    checksums_bytes = "\n".join(checksum_lines).encode("utf-8")
    archive_checksum = _file_sha256_bytes(checksums_bytes)
    file_map["checksums.sha256"] = checksums_bytes

    header_obj = {
        "format_version": "1.0",
        "file_type": "kernelforge_inference",
        "project_name": root.name,
        "project_ref": manifest_obj["project_ref"],
        "exported_at": exported_at,
        "kernelforge_version": "0.1.0",
        "git_commit": plan.get("git_commit"),
        "runtime": {
            "min_cast_version": "0.1",
            "min_torch_version": "2.1.0",
            "min_cuda_version": "12.0",
            "target_sm_versions": sorted({sm for op in ops_manifest for sm in op["precompiled"].keys()}),
            "gpu_name": gpu_info.get("name"),
            "gpu_capability": gpu_info.get("capability"),
        },
        "contents": {
            "optimized_op_count": len(ops_manifest),
            "total_op_count": len(ops_manifest),
            "has_precompiled": bool(precompiled_paths),
            "precompiled_sm_versions": sorted({sm for op in ops_manifest for sm in op["precompiled"].keys()}),
            "weight_size_bytes": weight_size,
        },
        "archive_checksum": archive_checksum,
    }
    header_bytes = json.dumps(header_obj, indent=2).encode("utf-8")

    export_dir = root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    cast_path = Path(output_path).expanduser().resolve() if output_path else (export_dir / f"{root.name}.cast")
    cast_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cast_path, "w") as archive:
        archive.writestr("HEADER.json", header_bytes)
        for relpath, data in file_map.items():
            archive.writestr(relpath, data)

    return {
        "success": True,
        "name": cast_path.name,
        "export_path": str(cast_path),
        "size": cast_path.stat().st_size,
        "cast_package_sha256": safe_sha256_path(cast_path),
        "manifest": manifest_obj,
        "header": header_obj,
        "selection_manifest": plan,
        "precompiled_binaries": precompiled_paths,
        "target_sm": sm_version,
        "gpu_info": gpu_info,
    }


def inspect_cast_package(cast_path: str | Path) -> dict[str, Any]:
    from .kf_runtime import inspect_cast_package as inspect_runtime_cast_package

    return inspect_runtime_cast_package(cast_path)


def copy_cast_artifact(
    cast_path: str | Path,
    *,
    destination_dir: str | Path,
    filename: str,
) -> Path:
    source = Path(cast_path).expanduser().resolve()
    destination_root = Path(destination_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    target = destination_root / filename
    shutil.copy2(source, target)
    return target
