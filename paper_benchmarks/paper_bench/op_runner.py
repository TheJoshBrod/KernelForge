from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F

from kernelforge.run_cast import compile_kernel, verify_checksums
from src.optimizer.quantized import describe_tinygemm_qbits_tensor, prepare_tinygemm_linear_launch_args

from .artifacts import RunLayout, write_json_artifact
from .baselines import CompileSettings, compile_model, compile_settings_from_dict, sync_device
from .cache import CacheRequest, copy_reused_artifact_set, find_matching_reusable_artifact, make_cache_request
from .correctness import reference_correctness
from .provenance import hash_visible_paths, sha256_bytes, sha256_file, sha256_path
from .schema import BenchmarkArtifact, BenchmarkMode, CorrectnessStatus, EnvironmentArtifact, RunManifestArtifact, Stage, Variant
from .stats import build_latency_summary
from .validator import validated_artifact_update

_LAUNCH_SIGNATURE_RE = re.compile(r"torch::Tensor\s+launch\s*\(([^)]*)\)")
_F_PREFIX = "torch_nn_functional_"


@dataclass(frozen=True)
class OperatorEntry:
    entry_name: str
    entry_path: str
    entry_hash: str
    args: Any
    kwargs: dict[str, Any]
    metadata: dict[str, Any]
    synthetic: bool = False
    recorded_op_name: str | None = None


def _artifact_common_fields(common_fields: dict[str, Any]) -> dict[str, Any]:
    artifact_common = dict(common_fields)
    for key in (
        "artifact_type",
        "benchmark_mode",
        "variant",
        "stage",
        "warmup_count",
        "timed_run_count",
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
    ):
        artifact_common.pop(key, None)
    return artifact_common


def resolve_callable(callable_name: str):
    module_name, attr_name = callable_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _normalize_op_name(op_name: str) -> str:
    return str(op_name).strip()


def _functional_attr_from_op_name(op_name: str) -> str:
    normalized = _normalize_op_name(op_name)
    if normalized.startswith(_F_PREFIX):
        return normalized[len(_F_PREFIX):]
    if normalized.startswith("torch.nn.functional."):
        return normalized.rsplit(".", 1)[-1]
    if normalized.startswith("aten."):
        return normalized.split(".", 1)[1].split(".", 1)[0]
    return normalized.rsplit(".", 1)[-1].replace("/", "_")


def kf_operator_aliases(op_name: str) -> list[str]:
    normalized = _normalize_op_name(op_name)
    fn_attr = _functional_attr_from_op_name(normalized)
    aliases = [
        normalized,
        f"aten.{fn_attr}",
        f"torch.nn.functional.{fn_attr}",
        f"{_F_PREFIX}{fn_attr}",
    ]
    return list(dict.fromkeys(alias for alias in aliases if alias))


def resolve_operator_callable(op_name: str, callable_name: str | None = None):
    if callable_name:
        return resolve_callable(callable_name)

    fn_attr = _functional_attr_from_op_name(op_name)
    if hasattr(F, fn_attr):
        return getattr(F, fn_attr)

    normalized = _normalize_op_name(op_name)
    if normalized.startswith("aten."):
        target = torch.ops.aten
        for part in normalized.split(".", 1)[1].split("."):
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                break
        if hasattr(target, "default"):
            return target.default
        if callable(target):
            return target

    raise ValueError(f"Unable to resolve operator callable for {op_name!r}")


def _hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _tensor_metadata(tensor: torch.Tensor, *, path: str) -> dict[str, Any]:
    metadata = {
        "path": path,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "layout": str(tensor.layout),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
    }
    quantized_storage = describe_tinygemm_qbits_tensor(tensor)
    if quantized_storage:
        metadata["quantized_storage"] = quantized_storage
    return metadata


def _collect_tensor_metadata(value: Any, *, path: str = "root") -> list[dict[str, Any]]:
    if torch.is_tensor(value):
        return [_tensor_metadata(value, path=path)]
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            out.extend(_collect_tensor_metadata(item, path=f"{path}[{index}]"))
        return out
    if isinstance(value, tuple):
        out = []
        for index, item in enumerate(value):
            out.extend(_collect_tensor_metadata(item, path=f"{path}[{index}]"))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value.keys(), key=str):
            out.extend(_collect_tensor_metadata(value[key], path=f"{path}.{key}"))
        return out
    return []


def _entry_metadata(entry_name: str, op_name: str, args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    arg_tensors = _collect_tensor_metadata(args, path="args")
    kwarg_tensors = _collect_tensor_metadata(kwargs, path="kwargs")
    all_tensors = arg_tensors + kwarg_tensors
    shape_signatures = sorted(
        {
            "x".join(str(dim) for dim in tensor_meta["shape"])
            for tensor_meta in all_tensors
        }
    )
    dtype_coverage = sorted({tensor_meta["dtype"] for tensor_meta in all_tensors})
    return {
        "entry_name": entry_name,
        "op_name": op_name,
        "arg_tensor_count": len(arg_tensors),
        "kwarg_tensor_count": len(kwarg_tensors),
        "tensors": all_tensors,
        "shape_signatures": shape_signatures,
        "dtype_coverage": dtype_coverage,
    }


def _matches_requested_op(recorded_op_name: str | None, requested_op_name: str) -> bool:
    if not recorded_op_name:
        return True
    return _normalize_op_name(recorded_op_name) in set(kf_operator_aliases(requested_op_name))


def _stable_entry_set_hash(entries: list[OperatorEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.entry_name):
        digest.update(entry.entry_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.entry_hash.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_operator_entries(workload_path: str | Path, *, requested_op_name: str | None = None) -> tuple[list[OperatorEntry], dict[str, Any]]:
    root = Path(workload_path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(root.glob("entry_*.pt"))
    if not files:
        raise ValueError(f"No captured operator entries found in {workload_path}")

    entries: list[OperatorEntry] = []
    for file_path in files:
        try:
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to read captured entry {file_path}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict entry payload in {file_path}")

        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {}) or {}
        recorded_op_name = payload.get("op_name") or payload.get("op") or payload.get("name")
        synthetic = bool(payload.get("synthetic") or payload.get("synthetic_entry") or payload.get("synthetic_workload"))
        if requested_op_name and not _matches_requested_op(recorded_op_name, requested_op_name):
            raise ValueError(
                f"Captured entry {file_path.name} records op {recorded_op_name!r}, which does not match requested op {requested_op_name!r}"
            )
        op_name = requested_op_name or str(recorded_op_name or "unknown_op")
        entries.append(
            OperatorEntry(
                entry_name=file_path.name,
                entry_path=str(file_path),
                entry_hash=sha256_file(file_path),
                args=args,
                kwargs=dict(kwargs),
                metadata=_entry_metadata(file_path.name, op_name, args, kwargs),
                synthetic=synthetic,
                recorded_op_name=str(recorded_op_name) if recorded_op_name else None,
            )
        )

    shape_signatures = sorted(
        {
            shape
            for entry in entries
            for shape in entry.metadata.get("shape_signatures", [])
        }
    )
    dtype_coverage = sorted(
        {
            dtype
            for entry in entries
            for dtype in entry.metadata.get("dtype_coverage", [])
        }
    )
    summary = {
        "entry_count": len(entries),
        "entry_hashes": {entry.entry_name: entry.entry_hash for entry in entries},
        "entry_set_hash": _stable_entry_set_hash(entries),
        "synthetic_workload": any(entry.synthetic for entry in entries),
        "coverage": {
            "entry_count": len(entries),
            "unique_shape_count": len(shape_signatures),
            "unique_shapes": shape_signatures,
            "dtype_coverage": dtype_coverage,
        },
    }
    return entries, summary


def resolve_project_operator_entries_dir(project_root: str | Path, op_name: str) -> Path | None:
    root = Path(project_root).expanduser().resolve()
    entries_root = root / "io" / "individual_ops"
    if not entries_root.exists():
        return None
    aliases = {_normalize_op_name(alias) for alias in kf_operator_aliases(op_name)}
    matches = [
        child.resolve()
        for child in sorted(entries_root.iterdir())
        if child.is_dir() and _normalize_op_name(child.name) in aliases
    ]
    if not matches:
        return None
    if len(matches) > 1:
        match_text = ", ".join(str(match) for match in matches)
        raise ValueError(f"Ambiguous captured operator entry directories for {op_name!r}: {match_text}")
    return matches[0]


def resolve_project_operator_export_evidence(
    project_ref: str,
    op_name: str,
    *,
    search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    from .kf_project import find_project, load_project_export_candidates

    root = find_project(project_ref, search_roots=search_roots)
    selection = load_project_export_candidates(root).get("auto_best_fastest_valid", {})
    aliases = {_normalize_op_name(alias) for alias in kf_operator_aliases(op_name)}

    def _match_key(mapping: dict[str, Any]) -> str | None:
        matches = [key for key in sorted(mapping) if _normalize_op_name(key) in aliases]
        if not matches:
            return None
        if len(matches) > 1:
            joined = ", ".join(matches)
            raise ValueError(f"Ambiguous auto_best_fastest_valid operator match for {op_name!r}: {joined}")
        return matches[0]

    selected_ops = selection.get("selected_ops") if isinstance(selection.get("selected_ops"), dict) else {}
    rejected_candidates = (
        selection.get("rejected_candidates")
        if isinstance(selection.get("rejected_candidates"), dict)
        else {}
    )
    skipped_ops = selection.get("skipped_ops") if isinstance(selection.get("skipped_ops"), dict) else {}

    selected_key = _match_key(selected_ops)
    rejected_key = _match_key(rejected_candidates)
    skipped_key = _match_key(skipped_ops)
    selected_candidate = (
        dict(selected_ops[selected_key])
        if selected_key and isinstance(selected_ops.get(selected_key), dict)
        else None
    )
    return {
        "project_ref": str(project_ref),
        "project_root": str(root),
        "selection_policy": str(selection.get("policy_name") or ""),
        "export_paper_eligible": bool(selection.get("export_paper_eligible")),
        "selected_op_name": selected_key,
        "selected_candidate": selected_candidate,
        "rejected_candidates": list(rejected_candidates.get(rejected_key, [])) if rejected_key else [],
        "skipped_op": dict(skipped_ops.get(skipped_key, {})) if skipped_key and isinstance(skipped_ops.get(skipped_key), dict) else None,
    }


def _move_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _invoke(func, args: Any, kwargs: dict[str, Any]):
    if isinstance(args, tuple):
        return func(*args, **kwargs)
    if isinstance(args, list):
        return func(*args, **kwargs)
    return func(args, **kwargs)


def _timed_invoke(device: str, fn: Callable[[], Any]) -> tuple[float, Any, Exception | None]:
    sync_device(device)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            value = fn()
        error = None
    except Exception as exc:
        value = None
        error = exc
    sync_device(device)
    return (time.perf_counter() - start) * 1000.0, value, error


def _hash_tensor(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return sha256_bytes(buffer.getvalue())


def hash_output(value: Any) -> str:
    if torch.is_tensor(value):
        return _hash_json_payload(
            {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "hash": _hash_tensor(value),
            }
        )
    if isinstance(value, list):
        return _hash_json_payload(["list", [hash_output(item) for item in value]])
    if isinstance(value, tuple):
        return _hash_json_payload(["tuple", [hash_output(item) for item in value]])
    if isinstance(value, dict):
        return _hash_json_payload(["dict", {str(key): hash_output(value[key]) for key in sorted(value.keys(), key=str)}])
    return _hash_json_payload(["scalar", value])


def _tensor_error_summary(reference: Any, candidate: Any, *, prefix: str = "output") -> dict[str, Any] | None:
    if torch.is_tensor(reference) and torch.is_tensor(candidate):
        summary: dict[str, Any] = {
            "path": prefix,
            "kind": "tensor",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
        }
        if reference.shape != candidate.shape:
            summary["reason"] = "shape_mismatch"
            return summary
        if reference.dtype != candidate.dtype:
            summary["reason"] = "dtype_mismatch"
            return summary
        diff = (reference - candidate).detach()
        summary["reason"] = "tensor_mismatch"
        summary["max_abs_diff"] = float(diff.abs().max().item())
        summary["mean_abs_diff"] = float(diff.abs().mean().item())
        summary["reference_hash"] = _hash_tensor(reference)
        summary["candidate_hash"] = _hash_tensor(candidate)
        return summary
    if isinstance(reference, (list, tuple)) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            return {
                "path": prefix,
                "kind": type(reference).__name__,
                "reason": "length_mismatch",
                "reference_length": len(reference),
                "candidate_length": len(candidate),
            }
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate, strict=True)):
            nested = _tensor_error_summary(ref_item, cand_item, prefix=f"{prefix}[{index}]")
            if nested is not None:
                return nested
        return None
    if isinstance(reference, dict) and isinstance(candidate, dict):
        ref_keys = sorted(reference.keys(), key=str)
        cand_keys = sorted(candidate.keys(), key=str)
        if ref_keys != cand_keys:
            return {
                "path": prefix,
                "kind": "dict",
                "reason": "key_mismatch",
                "reference_keys": [str(item) for item in ref_keys],
                "candidate_keys": [str(item) for item in cand_keys],
            }
        for key in ref_keys:
            nested = _tensor_error_summary(reference[key], candidate[key], prefix=f"{prefix}.{key}")
            if nested is not None:
                return nested
        return None
    if reference != candidate:
        return {
            "path": prefix,
            "kind": "scalar",
            "reason": "scalar_mismatch",
            "reference": repr(reference),
            "candidate": repr(candidate),
        }
    return None


def _compare_outputs(reference: Any, candidate: Any, *, prefix: str = "output") -> tuple[CorrectnessStatus, str | None]:
    if torch.is_tensor(reference) and torch.is_tensor(candidate):
        if reference.shape != candidate.shape:
            return CorrectnessStatus.failed, f"{prefix} shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        if reference.dtype != candidate.dtype:
            return CorrectnessStatus.failed, f"{prefix} dtype mismatch: {reference.dtype} != {candidate.dtype}"
        if torch.allclose(reference, candidate, atol=1e-4, rtol=1e-3):
            return CorrectnessStatus.passed, None
        max_abs_diff = float((reference - candidate).abs().max().item())
        return CorrectnessStatus.failed, f"{prefix} tensor mismatch; max_abs_diff={max_abs_diff:.6g}"
    if isinstance(reference, (list, tuple)) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            return CorrectnessStatus.failed, f"{prefix} length mismatch: {len(reference)} != {len(candidate)}"
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate, strict=True)):
            status, message = _compare_outputs(ref_item, cand_item, prefix=f"{prefix}[{index}]")
            if status != CorrectnessStatus.passed:
                return status, message
        return CorrectnessStatus.passed, None
    if isinstance(reference, dict) and isinstance(candidate, dict):
        ref_keys = sorted(reference.keys(), key=str)
        cand_keys = sorted(candidate.keys(), key=str)
        if ref_keys != cand_keys:
            return CorrectnessStatus.failed, f"{prefix} key mismatch: {ref_keys} != {cand_keys}"
        for key in ref_keys:
            status, message = _compare_outputs(reference[key], candidate[key], prefix=f"{prefix}.{key}")
            if status != CorrectnessStatus.passed:
                return status, message
        return CorrectnessStatus.passed, None
    if reference == candidate:
        return CorrectnessStatus.passed, None
    return CorrectnessStatus.failed, f"{prefix} scalar mismatch: {reference!r} != {candidate!r}"


def _resolve_original_params(func: Callable[..., Any]) -> list[str] | None:
    try:
        return list(inspect.signature(func).parameters.keys())
    except Exception:
        return None


def _launch_arity(kernel_path: str | None, ext: Any) -> int | None:
    if kernel_path and os.path.exists(kernel_path):
        try:
            match = _LAUNCH_SIGNATURE_RE.search(Path(kernel_path).read_text(encoding="utf-8"))
            if match:
                params = [part.strip() for part in match.group(1).split(",") if part.strip()]
                return len(params)
        except Exception:
            pass
    try:
        return len(inspect.signature(ext.launch).parameters)
    except Exception:
        return None


def _resolve_launch_args(args: Any, kwargs: dict[str, Any], orig_params: list[str] | None, n_launch: int | None) -> list[Any]:
    if orig_params is not None:
        resolved = {orig_params[i]: value for i, value in enumerate(args) if i < len(orig_params)}
        resolved.update(kwargs)
        ordered = [resolved.get(name) for name in orig_params]
    elif isinstance(args, (tuple, list)):
        ordered = list(args)
    else:
        ordered = [args]
    limit = n_launch if n_launch is not None else len(ordered)
    return ordered[:limit]


def _make_ext_launch_callable(
    ext: Any,
    *,
    kernel_path: str | None,
    reference_callable: Callable[..., Any],
    function_name: str | None = None,
):
    n_launch = _launch_arity(kernel_path, ext)
    orig_params = _resolve_original_params(reference_callable)

    def _invoke_kernel(*args, **kwargs):
        ordered = _resolve_launch_args(args, kwargs, orig_params, n_launch)
        call_args = prepare_tinygemm_linear_launch_args(
            function_name,
            ordered,
            {},
            {"params": orig_params or []},
        )
        if call_args is None:
            call_args = [
                value.contiguous() if torch.is_tensor(value) and not value.is_contiguous() else value
                for value in ordered
            ]
        return ext.launch(*call_args)

    return _invoke_kernel


def _load_extension_from_shared_object(module_name: str, so_path: str):
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared object {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_direct_source_file(path: Path) -> tuple[str, str]:
    if path.is_file():
        if path.suffix == ".cu":
            return str(path), "source"
        if path.suffix == ".so":
            return str(path), "precompiled"
        raise ValueError(f"Unsupported Kernel Forge artifact {path}; expected .cu or .so")

    kernel_cu = path / "kernel.cu"
    if kernel_cu.exists():
        return str(kernel_cu), "source"
    shared_objects = sorted(path.glob("*.so"))
    if len(shared_objects) == 1:
        return str(shared_objects[0]), "precompiled"
    cu_files = sorted(path.glob("*.cu"))
    if len(cu_files) == 1:
        return str(cu_files[0]), "source"
    raise ValueError(f"Unable to resolve a single Kernel Forge source artifact under {path}")


def _default_kf_loader(
    op_name: str,
    kernel_source_or_cast: str,
    *,
    reference_callable: Callable[..., Any],
    device: str,
    layout: RunLayout,
    settings: dict[str, Any] | None = None,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    if device != "cuda":
        raise RuntimeError("Kernel Forge operator benchmarking currently requires device='cuda' for real kernels.")

    target_path = Path(kernel_source_or_cast)
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    kf_settings = settings or {}
    build_dir = layout.logs_dir / "kf_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if target_path.suffix == ".cast":
        load_started = time.perf_counter()
        cast_hash = sha256_file(target_path)
        cache_dir = Path.home() / ".cache" / "paper_benchmarks" / "operator_casts" / cast_hash
        with zipfile.ZipFile(target_path) as archive:
            verify_checksums(archive)
            manifest = json.loads(archive.read("manifest.json"))
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                archive.extractall(cache_dir)

        aliases = set(kf_operator_aliases(op_name))
        selected_op = None
        for op_record in manifest.get("ops", []):
            if str(op_record.get("name")) in aliases:
                selected_op = op_record
                break
        if selected_op is None:
            raise ValueError(f"Kernel Forge cast package {target_path} does not contain operator {op_name!r}")

        runtime_load_ms = (time.perf_counter() - load_started) * 1000.0
        kernel_rel = selected_op.get("cuda_source")
        if not kernel_rel:
            raise ValueError(f"Selected cast operator {selected_op.get('name')} is missing cuda_source")
        kernel_path = cache_dir / kernel_rel
        if not kernel_path.exists():
            raise FileNotFoundError(kernel_path)
        manifest_selection_meta = (
            manifest.get("selected_kernel_metadata", {})
            if isinstance(manifest.get("selected_kernel_metadata"), dict)
            else {}
        )
        op_selection_meta = (
            manifest_selection_meta.get(str(selected_op["name"]))
            if isinstance(manifest_selection_meta.get(str(selected_op["name"])), dict)
            else {}
        )
        source_hash = sha256_file(kernel_path)
        selected_source_hash = str(op_selection_meta.get("selected_source_hash") or source_hash)

        gpu_sm = "sm_{0}{1}".format(*torch.cuda.get_device_capability())
        load_mode = "jit"
        jit_compile_time_ms = 0.0
        precompiled_load_time_ms = 0.0
        ext = None
        precompiled = selected_op.get("precompiled", {})
        so_rel = precompiled.get(gpu_sm) if isinstance(precompiled, dict) else None
        if so_rel:
            so_path = cache_dir / so_rel
            if so_path.exists():
                precompiled_started = time.perf_counter()
                ext = _load_extension_from_shared_object(f"{selected_op['name']}_{cast_hash[:8]}", str(so_path))
                precompiled_load_time_ms = (time.perf_counter() - precompiled_started) * 1000.0
                load_mode = "precompiled"
        if ext is None:
            if kf_settings.get("require_precompiled", False):
                raise RuntimeError(f"Precompiled kernel required for {selected_op['name']} on {gpu_sm}, but none was available.")
            if not kf_settings.get("allow_jit", True):
                raise RuntimeError(f"JIT kernel loading is disabled for {selected_op['name']}.")
            jit_started = time.perf_counter()
            ext = compile_kernel(str(kernel_path), str(selected_op["name"]), str(build_dir))
            jit_compile_time_ms = (time.perf_counter() - jit_started) * 1000.0

        runner = _make_ext_launch_callable(
            ext,
            kernel_path=str(kernel_path),
            reference_callable=reference_callable,
            function_name=str(selected_op["name"]),
        )
        meta = {
            "load_time_ms": float(runtime_load_ms),
            "runtime_load_time_ms": float(runtime_load_ms),
            "setup_time_ms": float(runtime_load_ms),
            "jit_compile_time_ms": float(jit_compile_time_ms),
            "compile_time_ms": float(jit_compile_time_ms),
            "precompiled_load_time_ms": float(precompiled_load_time_ms),
            "cast_package_path": str(target_path),
            "cast_package_hash": cast_hash,
            "kf_artifact_path": str(target_path),
            "kf_artifact_hash": cast_hash,
            "kf_artifact_kind": "cast",
            "claim_scope": "deployment_operator",
            "operator_runtime_label": "deployment/operator",
            "runtime_patch_enabled": False,
            "project_ref": (
                str(manifest.get("project_ref"))
                if str(manifest.get("project_ref") or "").strip()
                else str(kf_settings.get("project_ref") or "")
            ) or None,
            "selection_policy": str(manifest.get("selection_policy") or "") or None,
            "cast_manifest": manifest,
            "selected_ops": [str(selected_op["name"])],
            "selected_kernel_metadata": {
                str(selected_op["name"]): (
                    dict(op_selection_meta)
                    if op_selection_meta
                    else {
                        "candidate_id": f"{selected_op['name']}:deployment",
                        "kernel_source_path": str(kernel_path),
                        "selected_source_hash": selected_source_hash,
                        "evidence_tier": "deployment",
                        "selection_reason": "cast manifest selected operator",
                        "benchmark_reference": {},
                    }
                )
            },
            "loaded_kernels": [
                {
                    "op_name": str(selected_op["name"]),
                    "load_mode": load_mode,
                    "kernel_source_path": str(kernel_path),
                    "kernel_source_hash": source_hash,
                }
            ],
            "kernel_source_hashes": {str(kernel_rel): source_hash},
            "selected_source_hashes": {str(selected_op["name"]): selected_source_hash},
            "precompiled_vs_jit_path": {str(selected_op["name"]): load_mode},
            "kernel_launches_attempted": 0,
            "kernel_launches_succeeded": 0,
            "kernel_launches_failed": 0,
            "fallback_count": 0,
            "exception_fallback_count": 0,
            "contiguous_copy_count": 0,
            "adaptation_count": 0,
        }
        return runner, meta

    resolved_artifact, resolved_kind = _resolve_direct_source_file(target_path)
    artifact_hash = sha256_path(target_path)
    resolved_source_hash = sha256_path(resolved_artifact)
    load_started = time.perf_counter()
    if resolved_kind == "precompiled":
        ext = _load_extension_from_shared_object(f"kf_direct_{target_path.stem}_{artifact_hash[:8]}", resolved_artifact)
        load_mode = "precompiled"
        jit_compile_time_ms = 0.0
        precompiled_load_time_ms = (time.perf_counter() - load_started) * 1000.0
    else:
        ext = compile_kernel(resolved_artifact, f"kf_direct_{_functional_attr_from_op_name(op_name)}", str(build_dir))
        load_mode = "jit"
        jit_compile_time_ms = (time.perf_counter() - load_started) * 1000.0
        precompiled_load_time_ms = 0.0
    runner = _make_ext_launch_callable(
        ext,
        kernel_path=resolved_artifact if resolved_kind == "source" else None,
        reference_callable=reference_callable,
        function_name=op_name,
    )
    alias_name = kf_operator_aliases(op_name)[-1]
    project_ref = str(kf_settings.get("project_ref") or "").strip() or None
    project_selected_kernel_metadata: dict[str, Any] = {}
    export_selection_match = False
    selection_policy = None
    if project_ref:
        project_evidence = resolve_project_operator_export_evidence(project_ref, op_name)
        selection_policy = project_evidence.get("selection_policy") or None
        selected_candidate = project_evidence.get("selected_candidate")
        selected_op_name = project_evidence.get("selected_op_name")
        if selected_candidate and selected_op_name:
            project_selected_kernel_metadata = {str(selected_op_name): dict(selected_candidate)}
            selected_path = str(selected_candidate.get("kernel_source_path") or "").strip()
            selected_hash = str(selected_candidate.get("selected_source_hash") or "").strip()
            if selected_path:
                export_selection_match = Path(selected_path).resolve(strict=False) == Path(resolved_artifact).resolve(strict=False)
            if not export_selection_match and selected_hash:
                export_selection_match = selected_hash == resolved_source_hash
    selected_kernel_metadata = (
        project_selected_kernel_metadata
        if export_selection_match and project_selected_kernel_metadata
        else {
            alias_name: {
                "candidate_id": f"{alias_name}:direct_source",
                "kernel_source_path": str(resolved_artifact),
                "selected_source_hash": resolved_source_hash,
                "evidence_tier": "micro_only",
                "selection_reason": "direct source operator replay",
                "benchmark_reference": {},
            }
        }
    )
    meta = {
        "load_time_ms": float(max(jit_compile_time_ms, precompiled_load_time_ms)),
        "runtime_load_time_ms": float(max(jit_compile_time_ms, precompiled_load_time_ms)),
        "setup_time_ms": float(max(jit_compile_time_ms, precompiled_load_time_ms)),
        "jit_compile_time_ms": float(jit_compile_time_ms),
        "compile_time_ms": float(jit_compile_time_ms),
        "precompiled_load_time_ms": float(precompiled_load_time_ms),
        "cast_package_path": None,
        "cast_package_hash": None,
        "kf_artifact_path": str(target_path),
        "kf_artifact_hash": artifact_hash,
        "kf_artifact_kind": "direct_source",
        "claim_scope": "micro_operator",
        "operator_runtime_label": "micro/operator",
        "runtime_patch_enabled": False,
        "project_ref": project_ref,
        "selection_policy": selection_policy,
        "selected_ops": [alias_name],
        "selected_kernel_metadata": selected_kernel_metadata,
        "project_selected_kernel_metadata": project_selected_kernel_metadata,
        "export_selection_match": export_selection_match,
        "loaded_kernels": [
            {
                "op_name": alias_name,
                "load_mode": load_mode,
                "kernel_source_path": resolved_artifact,
                "kernel_source_hash": resolved_source_hash,
            }
        ],
        "kernel_source_hashes": (
            hash_visible_paths([resolved_artifact]) if resolved_kind == "source" else {str(resolved_artifact): resolved_source_hash}
        ),
        "selected_source_hashes": {alias_name: resolved_source_hash},
        "precompiled_vs_jit_path": {alias_name: load_mode},
        "kernel_launches_attempted": 0,
        "kernel_launches_succeeded": 0,
        "kernel_launches_failed": 0,
        "fallback_count": 0,
        "exception_fallback_count": 0,
        "contiguous_copy_count": 0,
        "adaptation_count": 0,
    }
    return runner, meta


def _write_stage_artifact(
    layout: RunLayout,
    common_fields: dict[str, Any],
    *,
    variant: Variant,
    stage: Stage,
    samples_ms: list[float],
    warmup_count: int,
    timed_run_count: int,
    correctness_status: CorrectnessStatus,
    correctness_message: str | None,
    compile_time_ms: float | None,
    steady_state_time_ms: float | None,
    fallback_count: int | None,
    kernel_hit_count: int | None,
    details: dict[str, Any],
    sample_records: list[dict[str, Any]] | None = None,
) -> Path:
    artifact_common = _artifact_common_fields(common_fields)
    artifact = BenchmarkArtifact(
        **artifact_common,
        artifact_type="benchmark_result",
        benchmark_mode=BenchmarkMode.operator,
        variant=variant,
        stage=stage,
        warmup_count=warmup_count,
        timed_run_count=timed_run_count,
        latency_samples_ms=samples_ms,
        latency_summary=build_latency_summary(samples_ms),
        sample_records=sample_records or [],
        correctness_status=correctness_status,
        correctness_message=correctness_message,
        steady_state_time_ms=steady_state_time_ms,
        compile_time_ms=compile_time_ms,
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        details=details,
    )
    artifact = validated_artifact_update(artifact)
    return write_json_artifact(layout.metrics_dir / f"{variant.value}_{stage.value}.json", artifact)


def _metric_artifact_path(layout: RunLayout, variant: Variant, stage: Stage) -> Path:
    return layout.metrics_dir / f"{variant.value}_{stage.value}.json"


def _build_stage_common_details(
    suite,
    entry_summary: dict[str, Any],
    *,
    variant: Variant,
    kf_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = {
        "callable_name": getattr(suite, "callable_name", None),
        "operator_name": suite.op_name,
        "entry_set_hash": entry_summary["entry_set_hash"],
        "entry_hashes": entry_summary["entry_hashes"],
        "coverage": entry_summary["coverage"],
        "claim_scope": "baseline" if variant != Variant.kf_cast else (kf_meta or {}).get("claim_scope", "micro_operator"),
        "operator_runtime_label": None if variant != Variant.kf_cast else (kf_meta or {}).get("operator_runtime_label"),
        "deployment_comparable": bool(variant == Variant.kf_cast and (kf_meta or {}).get("kf_artifact_kind") == "cast"),
    }
    if kf_meta:
        details.update(
            {
                "selected_ops": kf_meta.get("selected_ops", []),
                "selected_kernel_metadata": kf_meta.get("selected_kernel_metadata", {}),
                "loaded_kernels": kf_meta.get("loaded_kernels", []),
                "precompiled_vs_jit_path": kf_meta.get("precompiled_vs_jit_path", {}),
                "kernel_source_hashes": kf_meta.get("kernel_source_hashes", {}),
                "selected_source_hashes": kf_meta.get("selected_source_hashes", {}),
                "project_ref": kf_meta.get("project_ref"),
                "project_selected_kernel_metadata": kf_meta.get("project_selected_kernel_metadata", {}),
                "export_selection_match": kf_meta.get("export_selection_match"),
                "export_selection_policy": kf_meta.get("selection_policy"),
                "cast_manifest": kf_meta.get("cast_manifest"),
            }
        )
    return details


def _entry_sample_matrix(entries: list[OperatorEntry]) -> list[dict[str, Any]]:
    return [
        {
            "entry_name": entry.entry_name,
            "entry_hash": entry.entry_hash,
            "metadata": entry.metadata,
        }
        for entry in entries
    ]


def run_operator_benchmark(
    *,
    layout: RunLayout,
    common_fields: dict[str, Any],
    env_artifact: EnvironmentArtifact,
    manifest_artifact: RunManifestArtifact,
    suite,
    variant: Variant,
    op_resolver: Callable[[str, str | None], Callable[..., Any]] = resolve_operator_callable,
    compile_model_fn: Callable[[Any, CompileSettings | dict[str, Any] | None], tuple[Any, float]] = compile_model,
    kf_loader: Callable[..., tuple[Callable[..., Any], dict[str, Any]]] = _default_kf_loader,
    reuse_cache: bool = False,
    cache_search_root: str | Path | None = None,
) -> RunLayout:
    entries, entry_summary = load_operator_entries(suite.workload_path, requested_op_name=suite.op_name)
    if entry_summary["synthetic_workload"] and not common_fields.get("synthetic_workload", False):
        raise RuntimeError("Synthetic operator entries are forbidden for paper runs. Pass --allow-synthetic-demo to proceed.")

    write_json_artifact(layout.run_dir / "manifest.json", manifest_artifact)
    write_json_artifact(layout.run_dir / "env.json", env_artifact)

    eager_func = op_resolver(suite.op_name, getattr(suite, "callable_name", None))
    device = suite.device
    compile_settings = compile_settings_from_dict(common_fields.get("compile_settings"))
    kf_settings = dict(common_fields.get("kf_settings") or {})
    common = dict(common_fields)

    compile_time_ms: float | None = None
    load_time_ms = 0.0
    fallback_count: int | None = None
    kernel_hit_count: int | None = None
    kf_meta: dict[str, Any] | None = None

    if variant == Variant.eager:
        candidate_func = eager_func
        stage_correctness_status, stage_correctness_message = reference_correctness()
    elif variant == Variant.torch_compile:
        candidate_func = eager_func
        stage_correctness_status = CorrectnessStatus.passed
        stage_correctness_message = None
    elif variant == Variant.kf_cast:
        kernel_source_or_cast = getattr(suite, "kernel_source_or_cast", None)
        if not kernel_source_or_cast:
            raise ValueError("Kernel Forge operator benchmarking requires --kernel-source-or-cast")
        candidate_func, kf_meta = kf_loader(
            suite.op_name,
            kernel_source_or_cast,
            reference_callable=eager_func,
            device=device,
            layout=layout,
            settings=kf_settings,
        )
        load_time_ms = float(kf_meta.get("load_time_ms", kf_meta.get("runtime_load_time_ms", 0.0)) or 0.0)
        compile_time_ms = float(kf_meta.get("compile_time_ms", kf_meta.get("jit_compile_time_ms", 0.0)) or 0.0)
        common["cast_package_path"] = kf_meta.get("cast_package_path")
        common["cast_package_hash"] = kf_meta.get("cast_package_hash")
        common["kf_artifact_path"] = kf_meta.get("kf_artifact_path")
        common["kf_artifact_hash"] = kf_meta.get("kf_artifact_hash")
        common["kf_artifact_kind"] = kf_meta.get("kf_artifact_kind")
        if kf_meta.get("kernel_source_hashes"):
            merged_hashes = dict(common.get("exported_kernel_hashes") or {})
            merged_hashes.update(kf_meta["kernel_source_hashes"])
            common["exported_kernel_hashes"] = dict(sorted(merged_hashes.items()))
        if kf_meta.get("selected_source_hashes"):
            kf_settings["selected_source_hashes"] = dict(kf_meta.get("selected_source_hashes") or {})
        if kf_meta.get("project_ref") and not kf_settings.get("project_ref"):
            kf_settings["project_ref"] = str(kf_meta.get("project_ref"))
        common["kf_settings"] = dict(kf_settings)
        fallback_count = int(kf_meta.get("fallback_count", 0) or 0)
        kernel_hit_count = int(kf_meta.get("kernel_hit_count", 0) or 0)
        stage_correctness_status = CorrectnessStatus.passed
        stage_correctness_message = None
    else:
        raise ValueError(f"Unsupported operator variant: {variant}")

    load_details = _build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta)
    cache_root = Path(cache_search_root) if cache_search_root else layout.run_dir.parent
    load_request = CacheRequest(
        signature=make_cache_request(
            common,
            variant=variant,
            stage=Stage.load,
            details=load_details,
            sample_matrix=[{"sample_index": 0, "load_source": load_details.get("claim_scope", "baseline")}],
            warmup_count=0,
            timed_run_count=1,
        ),
        target_path=_metric_artifact_path(layout, variant, Stage.load),
    )
    cache_requests: list[CacheRequest] = [load_request]
    compile_request: CacheRequest | None = None
    if variant == Variant.torch_compile:
        compile_request = CacheRequest(
            signature=make_cache_request(
                common,
                variant=variant,
                stage=Stage.compile,
                details=load_details,
                warmup_count=0,
                timed_run_count=1,
            ),
            target_path=_metric_artifact_path(layout, variant, Stage.compile),
        )
        cache_requests.append(compile_request)
    elif compile_time_ms and compile_time_ms > 0.0:
        cache_requests.append(
            CacheRequest(
                signature=make_cache_request(
                    common,
                    variant=variant,
                    stage=Stage.compile,
                    details=_build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta),
                    warmup_count=0,
                    timed_run_count=1,
                ),
                target_path=_metric_artifact_path(layout, variant, Stage.compile),
            )
        )
    if int(suite.warmup_count) > 0:
        cache_requests.append(
            CacheRequest(
                signature=make_cache_request(
                    common,
                    variant=variant,
                    stage=Stage.warmup,
                    details=_build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta),
                    sample_matrix=_entry_sample_matrix(entries),
                    warmup_count=int(suite.warmup_count),
                    timed_run_count=len(entries) * int(suite.warmup_count),
                ),
                target_path=_metric_artifact_path(layout, variant, Stage.warmup),
            )
        )
    cache_requests.append(
        CacheRequest(
            signature=make_cache_request(
                common,
                variant=variant,
                stage=Stage.operator,
                details=_build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta),
                sample_matrix=_entry_sample_matrix(entries),
                warmup_count=int(suite.warmup_count),
                timed_run_count=len(entries) * int(suite.timed_run_count),
            ),
            target_path=_metric_artifact_path(layout, variant, Stage.operator),
        )
    )
    if reuse_cache:
        if variant == Variant.torch_compile and compile_request is not None:
            compile_match = find_matching_reusable_artifact(
                cache_root,
                compile_request.signature,
                exclude_run_dir=layout.run_dir,
            )
            if compile_match is not None:
                _, compile_artifact = compile_match
                compile_failed = (
                    compile_artifact.correctness_status == CorrectnessStatus.failed
                    or str((compile_artifact.details or {}).get("execution_status", "")).lower() == "failed"
                )
                reusable_requests = cache_requests[:2] if compile_failed else cache_requests
                if copy_reused_artifact_set(
                    reusable_requests,
                    search_root=cache_root,
                    run_id=common["run_id"],
                    timestamp_utc=common["timestamp_utc"],
                    exclude_run_dir=layout.run_dir,
                ):
                    return layout
        elif copy_reused_artifact_set(
            cache_requests,
            search_root=cache_root,
            run_id=common["run_id"],
            timestamp_utc=common["timestamp_utc"],
            exclude_run_dir=layout.run_dir,
        ):
            return layout

    _write_stage_artifact(
        layout,
        common,
        variant=variant,
        stage=Stage.load,
        samples_ms=[float(load_time_ms)],
        warmup_count=0,
        timed_run_count=1,
        correctness_status=CorrectnessStatus.not_applicable if variant != Variant.eager else stage_correctness_status,
        correctness_message=None if variant != Variant.eager else stage_correctness_message,
        compile_time_ms=None,
        steady_state_time_ms=float(load_time_ms),
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        details=load_details,
        sample_records=[
            {
                "sample_index": 0,
                "load_source": load_details.get("claim_scope", "baseline"),
                "latency_ms": float(load_time_ms),
            }
        ],
    )

    if variant == Variant.torch_compile:
        compile_started = time.perf_counter()
        try:
            candidate_func, compile_time_ms = compile_model_fn(eager_func, compile_settings)
        except Exception as exc:
            compile_time_ms = (time.perf_counter() - compile_started) * 1000.0
            common["cast_package_path"] = None
            common["cast_package_hash"] = None
            common["kf_artifact_path"] = None
            common["kf_artifact_hash"] = None
            common["kf_artifact_kind"] = None
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.compile,
                samples_ms=[float(compile_time_ms)],
                warmup_count=0,
                timed_run_count=1,
                correctness_status=CorrectnessStatus.failed,
                correctness_message=f"torch.compile failed: {type(exc).__name__}: {exc}",
                compile_time_ms=float(compile_time_ms),
                steady_state_time_ms=None,
                fallback_count=None,
                kernel_hit_count=None,
                details={
                    **load_details,
                    "execution_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                sample_records=[
                    {
                        "sample_index": 0,
                        "latency_ms": float(compile_time_ms),
                        "execution_status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                ],
            )
            return layout

    if compile_time_ms and compile_time_ms > 0.0:
        _write_stage_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.compile,
            samples_ms=[float(compile_time_ms)],
            warmup_count=0,
            timed_run_count=1,
            correctness_status=CorrectnessStatus.not_applicable,
            correctness_message=None,
            compile_time_ms=float(compile_time_ms),
            steady_state_time_ms=None,
            fallback_count=fallback_count,
            kernel_hit_count=kernel_hit_count,
            details=_build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta),
            sample_records=[
                {
                    "sample_index": 0,
                    "latency_ms": float(compile_time_ms),
                    "execution_status": "ok",
                }
            ],
        )

    raw_rows: list[dict[str, Any]] = []
    operator_samples: list[float] = []
    warmup_samples: list[float] = []
    warmup_records: list[dict[str, Any]] = []
    per_entry_records: list[dict[str, Any]] = []
    failure_messages: list[str] = []
    error_count = 0
    correctness_pass_count = 0
    total_kernel_hits = 0

    for entry in entries:
        d_args = _move_to_device(entry.args, device)
        d_kwargs = _move_to_device(entry.kwargs, device)
        with torch.inference_mode():
            reference_output = _invoke(eager_func, d_args, d_kwargs)
        reference_output_hash = hash_output(reference_output)
        precheck_status = CorrectnessStatus.reference if variant == Variant.eager else CorrectnessStatus.passed
        precheck_message: str | None = None
        precheck_output_hash = reference_output_hash if variant == Variant.eager else None
        precheck_error_summary: dict[str, Any] | None = None
        if variant != Variant.eager:
            _, precheck_output, precheck_error = _timed_invoke(
                device,
                lambda bound_args=d_args, bound_kwargs=d_kwargs: _invoke(candidate_func, bound_args, bound_kwargs),
            )
            if precheck_error is not None:
                precheck_status = CorrectnessStatus.failed
                precheck_message = (
                    f"pre-timing correctness failed for {entry.entry_name}: "
                    f"{type(precheck_error).__name__}: {precheck_error}"
                )
                precheck_error_summary = {
                    "path": "output",
                    "kind": "exception",
                    "reason": "exception",
                    "error_type": type(precheck_error).__name__,
                    "error_message": str(precheck_error),
                }
            else:
                precheck_output_hash = hash_output(precheck_output)
                precheck_status, precheck_message = _compare_outputs(reference_output, precheck_output)
                if precheck_status != CorrectnessStatus.passed:
                    precheck_error_summary = _tensor_error_summary(reference_output, precheck_output)

        entry_samples: list[float] = []
        output_hashes: list[str | None] = []
        entry_failures: list[str] = []
        timed_results: list[dict[str, Any]] = []
        entry_warmups: list[float] = []
        per_entry_correct = True
        if precheck_status == CorrectnessStatus.failed:
            per_entry_correct = False
            error_count += 1
            failure_messages.append(precheck_message or f"pre-timing correctness failed for {entry.entry_name}")
            entry_failures.append(precheck_message or f"pre-timing correctness failed for {entry.entry_name}")
        for _ in range(int(suite.warmup_count)):
            elapsed_ms, _, _ = _timed_invoke(
                device,
                lambda bound_args=d_args, bound_kwargs=d_kwargs: _invoke(candidate_func, bound_args, bound_kwargs),
            )
            entry_warmups.append(float(elapsed_ms))
            warmup_samples.append(float(elapsed_ms))
        if entry_warmups:
            warmup_records.append(
                {
                    "entry_name": entry.entry_name,
                    "entry_hash": entry.entry_hash,
                    "latency_samples_ms": entry_warmups,
                    "metadata": entry.metadata,
                    "precheck_status": precheck_status.value,
                }
            )

        for run_index in range(int(suite.timed_run_count)):
            elapsed_ms, candidate_output, error = _timed_invoke(
                device,
                lambda bound_args=d_args, bound_kwargs=d_kwargs: _invoke(candidate_func, bound_args, bound_kwargs),
            )
            operator_samples.append(float(elapsed_ms))
            entry_samples.append(float(elapsed_ms))
            if error is not None:
                error_count += 1
                per_entry_correct = False
                output_hashes.append(None)
                message = f"timed run {run_index} failed for {entry.entry_name}: {type(error).__name__}: {error}"
                entry_failures.append(message)
                failure_messages.append(message)
                if variant == Variant.kf_cast and kf_settings.get("fail_on_fallback", True):
                    fallback_count = int((fallback_count or 0) + 1)
                if variant == Variant.kf_cast and kf_meta is not None:
                    kf_meta["kernel_launches_attempted"] = int(kf_meta.get("kernel_launches_attempted", 0) + 1)
                    kf_meta["kernel_launches_failed"] = int(kf_meta.get("kernel_launches_failed", 0) + 1)
                    kf_meta["exception_fallback_count"] = int(kf_meta.get("exception_fallback_count", 0) + 1)
                timed_results.append(
                    {
                        "run_index": run_index,
                        "latency_ms": float(elapsed_ms),
                        "output_hash": None,
                        "correctness_status": CorrectnessStatus.failed.value,
                        "correctness_message": message,
                        "tensor_error_summary": {
                            "path": "output",
                            "kind": "exception",
                            "reason": "exception",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    }
                )
                continue

            candidate_hash = hash_output(candidate_output)
            output_hashes.append(candidate_hash)
            tensor_error_summary = None
            if variant == Variant.eager:
                status, message = reference_correctness()
                correctness_pass_count += 1
            else:
                status, message = _compare_outputs(reference_output, candidate_output)
                if status == CorrectnessStatus.passed:
                    correctness_pass_count += 1
                    if variant == Variant.kf_cast:
                        total_kernel_hits += 1
                else:
                    per_entry_correct = False
                    error_count += 1
                    tensor_error_summary = _tensor_error_summary(reference_output, candidate_output)
                    failure_messages.append(
                        f"timed run {run_index} correctness mismatch for {entry.entry_name}: {message}"
                    )
                    entry_failures.append(
                        f"timed run {run_index} correctness mismatch for {entry.entry_name}: {message}"
                    )
            if variant == Variant.kf_cast and kf_meta is not None:
                kf_meta["kernel_launches_attempted"] = int(kf_meta.get("kernel_launches_attempted", 0) + 1)
                if status == CorrectnessStatus.passed:
                    kf_meta["kernel_launches_succeeded"] = int(kf_meta.get("kernel_launches_succeeded", 0) + 1)
                else:
                    kf_meta["kernel_launches_failed"] = int(kf_meta.get("kernel_launches_failed", 0) + 1)
            timed_results.append(
                {
                    "run_index": run_index,
                    "latency_ms": float(elapsed_ms),
                    "output_hash": candidate_hash,
                    "correctness_status": status.value,
                    "correctness_message": message,
                    "tensor_error_summary": tensor_error_summary,
                }
            )

        per_entry_record = {
            "entry_name": entry.entry_name,
            "entry_path": entry.entry_path,
            "entry_hash": entry.entry_hash,
            "recorded_op_name": entry.recorded_op_name,
            "metadata": entry.metadata,
            "precheck_status": precheck_status.value,
            "precheck_message": precheck_message,
            "precheck_output_hash": precheck_output_hash,
            "precheck_tensor_error_summary": precheck_error_summary,
            "warmup_latency_samples_ms": entry_warmups,
            "latency_samples_ms": entry_samples,
            "latency_summary": build_latency_summary(entry_samples).model_dump(mode="json"),
            "correctness_status": (
                CorrectnessStatus.reference.value
                if variant == Variant.eager
                else (CorrectnessStatus.passed.value if per_entry_correct else CorrectnessStatus.failed.value)
            ),
            "correctness_messages": entry_failures,
            "reference_output_hash": reference_output_hash,
            "output_hashes": output_hashes,
            "timed_results": timed_results,
        }
        per_entry_records.append(per_entry_record)
        raw_rows.append(per_entry_record)

    if warmup_samples:
        _write_stage_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.warmup,
            samples_ms=warmup_samples,
            warmup_count=int(suite.warmup_count),
            timed_run_count=len(warmup_samples),
            correctness_status=CorrectnessStatus.not_applicable if variant != Variant.eager else stage_correctness_status,
            correctness_message=None if variant != Variant.eager else stage_correctness_message,
            compile_time_ms=compile_time_ms,
            steady_state_time_ms=sum(warmup_samples) / len(warmup_samples),
            fallback_count=fallback_count,
            kernel_hit_count=kernel_hit_count,
            details=_build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta),
            sample_records=warmup_records,
        )

    raw_name = f"{variant.value}_operator_measurements.json"
    (layout.raw_dir / raw_name).write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")

    if variant == Variant.eager:
        correctness_status, correctness_message = reference_correctness()
    else:
        correctness_status = CorrectnessStatus.passed if not failure_messages else CorrectnessStatus.failed
        correctness_message = "; ".join(failure_messages[:8]) if failure_messages else None

    if variant == Variant.kf_cast:
        kernel_hit_count = total_kernel_hits
        if fallback_count is None:
            fallback_count = 0
        if kf_meta is not None:
            kf_meta["kernel_hit_count"] = kernel_hit_count
            kf_meta["fallback_count"] = fallback_count

    stage_details = _build_stage_common_details(suite, entry_summary, variant=variant, kf_meta=kf_meta)
    stage_details.update(
        {
            "error_count": error_count,
            "correctness_pass_count": correctness_pass_count,
            "precheck_required": variant != Variant.eager,
            "precheck_failures": sum(
                1 for record in per_entry_records if record.get("precheck_status") == CorrectnessStatus.failed.value
            ),
            "timed_run_count_per_entry": int(suite.timed_run_count),
            "warmup_count_per_entry": int(suite.warmup_count),
            "speedup_claim_safe": correctness_status in {CorrectnessStatus.reference, CorrectnessStatus.passed},
        }
    )
    if kf_meta:
        stage_details.update(
            {
                "kernel_launches_attempted": int(kf_meta.get("kernel_launches_attempted", 0)),
                "kernel_launches_succeeded": int(kf_meta.get("kernel_launches_succeeded", 0)),
                "kernel_launches_failed": int(kf_meta.get("kernel_launches_failed", 0)),
                "exception_fallback_count": int(kf_meta.get("exception_fallback_count", 0)),
            }
        )

    _write_stage_artifact(
        layout,
        common,
        variant=variant,
        stage=Stage.operator,
        samples_ms=operator_samples,
        warmup_count=int(suite.warmup_count),
        timed_run_count=len(operator_samples),
        correctness_status=correctness_status,
        correctness_message=correctness_message,
        compile_time_ms=compile_time_ms,
        steady_state_time_ms=(sum(operator_samples) / len(operator_samples)) if operator_samples else 0.0,
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        details=stage_details,
        sample_records=per_entry_records,
    )
    return layout
