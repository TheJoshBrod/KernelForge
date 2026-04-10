from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.progress import update_job_progress
from src.optimizer.tree_store import write_root_benchmark_metadata

from .harness import (
    DEFAULT_TIMED_RUNS,
    DEFAULT_WARMUP_RUNS,
    benchmark_entry_calls,
    sync_device as benchmark_sync_device,
)
from .integration import (
    empty_adapter_stats,
    invoke_kernel_launch,
    launch_params_for_runtime_kernel,
    merge_adapter_stats,
)
from .paths import find_latest_optimized_dir, project_dir_for_name
from .state import read_json_file, write_json_file

BenchmarkEntry = tuple[str, Any, dict[str, Any], Any]
BENCHMARK_SCHEMA_VERSION = 2

STRICT_CORRECTNESS_ATOL = 1e-4
STRICT_CORRECTNESS_RTOL = 1e-3
VERIFY_CORRECTNESS_ATOL = 1e-2
VERIFY_CORRECTNESS_RTOL = 1e-1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_local_toolchain_on_path() -> None:
    python_bin = str(Path(sys.executable).parent)
    current_path = os.environ.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    if python_bin not in path_parts:
        os.environ["PATH"] = (
            python_bin
            if not current_path
            else python_bin + os.pathsep + current_path
        )


def _resolve_device() -> str:
    target = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if target == "mps" and hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    if target in {"gpu", "cuda"} and torch.cuda.is_available():
        return "cuda"
    if target == "cpu":
        return "cpu"
    if hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _move_to_device(obj: Any, device: str) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _global_warmup(device: str) -> None:
    if device not in {"cuda", "mps"}:
        return
    try:
        with torch.no_grad():
            x = torch.randn((1024, 1024), device=device)
            y = torch.randn((1024, 1024), device=device)
            for _ in range(6):
                z = x @ y
                x = torch.relu(z)
        benchmark_sync_device(device)
    except Exception:
        pass


def _entry_signature(entries: list[BenchmarkEntry]) -> str:
    def _sig(v: Any) -> Any:
        if torch.is_tensor(v):
            return {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "requires_grad": bool(v.requires_grad),
            }
        if isinstance(v, list):
            return [_sig(x) for x in v[:3]]
        if isinstance(v, tuple):
            return tuple(_sig(x) for x in v[:3])
        if isinstance(v, dict):
            keys = sorted(list(v.keys()))[:5]
            return {k: _sig(v[k]) for k in keys}
        return type(v).__name__

    if not entries:
        return "empty"
    sample = entries[:3]
    payload = [
        (entry_file, _sig(args), _sig(kwargs), _sig(signature_meta))
        for entry_file, args, kwargs, signature_meta in sample
    ]
    return json.dumps(payload, sort_keys=True)


def _runtime_fingerprint(device: str) -> str:
    payload = {
        "device": device,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda or ""),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if device == "cuda" and torch.cuda.is_available():
        try:
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["device_capability"] = str(torch.cuda.get_device_capability(0))
            payload["device_count"] = int(torch.cuda.device_count())
        except Exception:
            pass
    if device == "mps" and hasattr(torch, "backends"):
        try:
            payload["mps_available"] = bool(torch.backends.mps.is_available())
            payload["mps_built"] = bool(torch.backends.mps.is_built())
        except Exception:
            pass
    return json.dumps(payload, sort_keys=True)


def _ops_from_csv(raw: str) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for part in str(raw).split(","):
        name = str(part).strip()
        if name:
            out.append(name)
    return out


def _load_entries(io_dir: Path, max_entries: int) -> list[BenchmarkEntry]:
    entries: list[BenchmarkEntry] = []
    files = sorted(io_dir.glob("entry_*.pt"))[:max_entries]
    for pt in files:
        try:
            payload = torch.load(pt, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(pt, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        signature_meta = payload.get("signature")
        if kwargs is None:
            kwargs = {}
        entries.append((pt.name, args, kwargs, signature_meta))
    return entries


def _get_pytorch_func(op_name: str):
    if op_name.startswith("torch_nn_functional_"):
        fn_name = op_name.replace("torch_nn_functional_", "", 1)
        if hasattr(F, fn_name):
            return getattr(F, fn_name)
    mapping = {
        "torch_nn_functional_relu": F.relu,
        "torch_nn_functional_linear": F.linear,
        "torch_nn_functional_layer_norm": F.layer_norm,
        "torch_nn_functional_embedding": F.embedding,
        "torch_nn_functional_dropout": F.dropout,
        "torch_nn_functional_batch_norm": F.batch_norm,
        "torch_nn_functional_gelu": F.gelu,
        "torch_nn_functional_scaled_dot_product_attention": F.scaled_dot_product_attention,
        "torch_nn_functional_softmax": F.softmax,
        "torch_nn_functional_adaptive_avg_pool1d": F.adaptive_avg_pool1d,
        "torch_nn_functional_adaptive_avg_pool2d": F.adaptive_avg_pool2d,
        "torch_nn_functional_max_pool2d": F.max_pool2d,
        "torch_nn_functional_pad": F.pad,
        "torch_nn_functional_conv2d": F.conv2d,
    }
    return mapping.get(op_name)


def _run_call(func, args: Any, kwargs: dict[str, Any]):
    if isinstance(args, tuple):
        return func(*args, **kwargs)
    if isinstance(args, list):
        return func(*args, **kwargs)
    return func(args, **kwargs)


def _measure_pytorch(
    func,
    entries: list[BenchmarkEntry],
    device: str,
) -> dict[str, Any]:
    if not entries:
        return benchmark_entry_calls([], device=device)

    entry_calls = []
    for entry_file, args, kwargs, _signature_meta in entries:
        d_args = _move_to_device(args, device)
        d_kwargs = _move_to_device(kwargs, device)

        def invoke(bound_args=d_args, bound_kwargs=d_kwargs):
            return _run_call(func, bound_args, bound_kwargs)

        entry_calls.append((entry_file, invoke))

    return benchmark_entry_calls(
        entry_calls,
        device=device,
        warmup_runs=DEFAULT_WARMUP_RUNS,
        timed_runs=DEFAULT_TIMED_RUNS,
    )


def _summarize_output_correctness(
    output_generated: Any,
    ground_truth: Any,
) -> dict[str, Any]:
    if torch.is_tensor(output_generated) and torch.is_tensor(ground_truth):
        target_ground_truth = ground_truth.to(output_generated.device)
        if not target_ground_truth.is_floating_point():
            is_equal = bool(torch.equal(output_generated, target_ground_truth))
            return {
                "strict_match": is_equal,
                "loose_match": is_equal,
                "max_abs_diff": None,
                "mean_abs_diff": None,
            }

        out_f = output_generated.float()
        gt_f = target_ground_truth.float()
        diff = (out_f - gt_f).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
        return {
            "strict_match": bool(
                torch.allclose(
                    output_generated,
                    target_ground_truth,
                    atol=STRICT_CORRECTNESS_ATOL,
                    rtol=STRICT_CORRECTNESS_RTOL,
                )
            ),
            "loose_match": bool(
                torch.allclose(
                    output_generated,
                    target_ground_truth,
                    atol=VERIFY_CORRECTNESS_ATOL,
                    rtol=VERIFY_CORRECTNESS_RTOL,
                )
            ),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        }

    if torch.is_tensor(output_generated) and output_generated.numel() == 1 and not torch.is_tensor(ground_truth):
        is_equal = bool(output_generated.detach().cpu().item() == ground_truth)
        return {
            "strict_match": is_equal,
            "loose_match": is_equal,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    if (not torch.is_tensor(output_generated)) and torch.is_tensor(ground_truth) and ground_truth.numel() == 1:
        is_equal = bool(output_generated == ground_truth.detach().cpu().item())
        return {
            "strict_match": is_equal,
            "loose_match": is_equal,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    is_equal = bool(output_generated == ground_truth)
    return {
        "strict_match": is_equal,
        "loose_match": is_equal,
        "max_abs_diff": None,
        "mean_abs_diff": None,
    }


def _measure_integrated_correctness(
    ext,
    *,
    launch_params: list[tuple[str, str]],
    op_name: str,
    func: Any,
    entries: list[BenchmarkEntry],
    device: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "entries_checked": 0,
        "strict_atol": STRICT_CORRECTNESS_ATOL,
        "strict_rtol": STRICT_CORRECTNESS_RTOL,
        "loose_atol": VERIFY_CORRECTNESS_ATOL,
        "loose_rtol": VERIFY_CORRECTNESS_RTOL,
        "strict_pass": True,
        "strict_mismatches": 0,
        "loose_pass": True,
        "loose_mismatches": 0,
        "worst_max_abs_diff": None,
        "worst_mean_abs_diff": None,
        "worst_entry": "",
        "errors": [],
    }
    if func is None or not entries:
        return summary

    for entry_file, args, kwargs, signature_meta in entries:
        try:
            d_args = _move_to_device(args, device)
            d_kwargs = _move_to_device(kwargs, device)
            ground_truth = _run_call(func, d_args, d_kwargs)
            output_generated = invoke_kernel_launch(
                ext,
                args=d_args,
                kwargs=d_kwargs,
                launch_params=launch_params,
                op_name=op_name,
                func=func,
                signature_meta=signature_meta,
                ensure_device=None,
                force_contiguous=False,
                adapter_stats=empty_adapter_stats(),
            )
            benchmark_sync_device(device)
            correctness = _summarize_output_correctness(output_generated, ground_truth)
        except Exception as exc:
            summary["errors"].append({"entry_file": str(entry_file), "error": str(exc)})
            summary["strict_mismatches"] += 1
            summary["loose_mismatches"] += 1
            summary["entries_checked"] += 1
            continue

        if not correctness["strict_match"]:
            summary["strict_mismatches"] += 1
        if not correctness["loose_match"]:
            summary["loose_mismatches"] += 1

        max_abs_diff = correctness.get("max_abs_diff")
        current_worst = summary.get("worst_max_abs_diff")
        if isinstance(max_abs_diff, (int, float)) and (
            current_worst is None or float(max_abs_diff) > float(current_worst)
        ):
            summary["worst_max_abs_diff"] = float(max_abs_diff)
            summary["worst_mean_abs_diff"] = correctness.get("mean_abs_diff")
            summary["worst_entry"] = str(entry_file)

        summary["entries_checked"] += 1

    summary["strict_pass"] = (
        summary["entries_checked"] > 0
        and summary["strict_mismatches"] == 0
        and not summary["errors"]
    )
    summary["loose_pass"] = (
        summary["entries_checked"] > 0
        and summary["loose_mismatches"] == 0
        and not summary["errors"]
    )
    return summary


def _coerce_cached_measurement(value: Any) -> dict[str, Any] | None:
    if isinstance(value, (int, float)):
        return {
            "mean_time_ms": float(value),
            "entry_files": [],
            "entry_latencies_ms": [],
            "entry_results": [],
            "entry_count": 0,
            "errors": [],
            "warmup_runs": DEFAULT_WARMUP_RUNS,
            "timed_runs": DEFAULT_TIMED_RUNS,
        }
    if not isinstance(value, dict):
        return None

    entry_files_raw = value.get("entry_files")
    entry_files = (
        [str(item) for item in entry_files_raw]
        if isinstance(entry_files_raw, list)
        else []
    )
    entry_latencies_raw = value.get("entry_latencies_ms")
    entry_latencies = []
    if isinstance(entry_latencies_raw, list):
        for item in entry_latencies_raw:
            try:
                entry_latencies.append(float(item))
            except Exception:
                continue

    mean_time_ms = value.get("mean_time_ms")
    if mean_time_ms is None and entry_latencies:
        mean_time_ms = sum(entry_latencies) / len(entry_latencies)
    try:
        parsed_mean = float(mean_time_ms) if mean_time_ms is not None else 0.0
    except Exception:
        parsed_mean = 0.0

    entry_count_raw = value.get("entry_count")
    try:
        entry_count = int(entry_count_raw) if entry_count_raw is not None else len(entry_files)
    except Exception:
        entry_count = len(entry_files)

    errors_raw = value.get("errors")
    errors = errors_raw if isinstance(errors_raw, list) else []

    entry_results = value.get("entry_results")
    if not isinstance(entry_results, list):
        entry_results = [
            {"entry_file": entry_file, "latency_ms": latency_ms}
            for entry_file, latency_ms in zip(entry_files, entry_latencies)
        ]

    return {
        "mean_time_ms": parsed_mean,
        "entry_files": entry_files,
        "entry_latencies_ms": entry_latencies,
        "entry_results": entry_results,
        "entry_count": entry_count,
        "errors": errors,
        "warmup_runs": int(value.get("warmup_runs", DEFAULT_WARMUP_RUNS)),
        "timed_runs": int(value.get("timed_runs", DEFAULT_TIMED_RUNS)),
    }


def _backend_for_suffix(suffix: str) -> str:
    ext = str(suffix or "").lower()
    if ext == ".cu":
        return "cuda"
    if ext == ".py":
        return "triton"
    if ext in {".metal", ".mps"}:
        return "mps"
    return ""


def _resolve_tree_kernel_source(
    optimized_root: Path | None,
    op_name: str,
) -> tuple[Path | None, str]:
    if optimized_root is None:
        return None, ""

    op_dir = optimized_root / op_name
    if not op_dir.exists():
        return None, ""

    backend = ""
    meta = op_dir / "generated_root.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            backend = str(payload.get("backend") or "")
        except Exception:
            pass

    db_file = op_dir / "nodes.db"
    if db_file.exists():
        try:
            import sqlite3 as _sqlite3

            with _sqlite3.connect(str(db_file)) as conn:
                row = conn.execute(
                    """
                    SELECT code
                    FROM nodes
                    WHERE code IS NOT NULL AND TRIM(code) != ''
                    ORDER BY
                        CASE WHEN value IS NULL OR value <= 0 THEN 1 ELSE 0 END,
                        value ASC,
                        id ASC
                    LIMIT 1
                    """
                ).fetchone()
            if row and row[0]:
                code_path = Path(str(row[0]))
                if not code_path.is_absolute():
                    code_path = op_dir.parent / code_path
                if code_path.exists():
                    return code_path, backend or _backend_for_suffix(code_path.suffix)
        except Exception:
            pass

    kernels_dir = op_dir / "kernels"
    if kernels_dir.exists():
        candidates = sorted(kernels_dir.glob("kernel_0.*"))
        if not candidates:
            candidates = sorted(kernels_dir.glob("kernel_*.*"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate, backend or _backend_for_suffix(candidate.suffix)

    return None, backend


def _profile_kernel_source_ms(
    source_path: Path | None,
    backend: str,
    io_op_dir: Path | None,
    benchmark_entry_files: list[Path] | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    if source_path is None or not source_path.exists():
        return None, "missing_kernel_source", backend
    if io_op_dir is None or not io_op_dir.exists():
        return None, "missing_io_entries", backend

    backend_name = str(backend or _backend_for_suffix(source_path.suffix))
    if backend_name not in {"cuda", "triton"}:
        return None, "unsupported_generated_backend", backend_name

    ext = ".cu" if backend_name == "cuda" else ".py"
    try:
        with tempfile.TemporaryDirectory(prefix="kforge-bench-") as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_dir = tmp_root / "bench_module"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            staged_kernel = tmp_dir / f"kernel{ext}"
            shutil.copy2(source_path, staged_kernel)
            if backend_name == "cuda":
                from src.optimizer.backends.cuda import CUDABackend

                stats = CUDABackend().profile_kernel(
                    {
                        "tmp_dir": tmp_dir,
                        "io_dir": io_op_dir,
                        "entry_files": benchmark_entry_files or [],
                    },
                    baseline=True,
                )
            else:
                from src.optimizer.backends.triton import TritonBackend

                stats = TritonBackend().profile_kernel(
                    {
                        "tmp_dir": tmp_dir,
                        "io_dir": io_op_dir,
                        "entry_files": benchmark_entry_files or [],
                    },
                    baseline=True,
                )
        ms = stats.get("mean_time_ms") if isinstance(stats, dict) else None
        if ms is None:
            return None, "generated_profile_missing", backend_name
        return stats, "ok", backend_name
    except Exception as e:
        msg = str(e).strip()
        if "Ninja is required to load C++ extensions" in msg:
            return None, "generated_profile_error_ninja", backend_name
        return None, "generated_profile_error", backend_name


def _profile_tree_kernel_ms(
    optimized_root: Path | None,
    op_name: str,
    io_op_dir: Path | None,
    benchmark_entry_files: list[Path] | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    source_path, backend = _resolve_tree_kernel_source(optimized_root, op_name)
    if source_path is None:
        return None, "missing_optimized_kernel", backend
    return _profile_kernel_source_ms(
        source_path,
        backend,
        io_op_dir,
        benchmark_entry_files=benchmark_entry_files,
    )


def _profile_generated_kernel_ms(
    project_dir: Path,
    op_name: str,
    io_op_dir: Path | None,
    benchmark_entry_files: list[Path] | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    generated_dir = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / op_name
    )
    if not generated_dir.exists():
        return None, "missing_generated", ""

    if (generated_dir / "success.cuda").exists():
        return _profile_kernel_source_ms(
            generated_dir / "kernel.cu",
            "cuda",
            io_op_dir,
            benchmark_entry_files=benchmark_entry_files,
        )

    if (generated_dir / "success.triton").exists():
        return _profile_kernel_source_ms(
            generated_dir / "kernel.py",
            "triton",
            io_op_dir,
            benchmark_entry_files=benchmark_entry_files,
        )
    if (generated_dir / "success.mps").exists():
        return None, "unsupported_generated_backend", "mps"
    if (generated_dir / "success.cpu").exists():
        return None, "unsupported_generated_backend", "cpu"
    return None, "missing_generated", ""


def _resolve_generated_kernel_source(
    project_dir: Path,
    op_name: str,
) -> tuple[Path | None, str]:
    generated_dir = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / op_name
    )
    if not generated_dir.exists():
        return None, ""
    if (generated_dir / "success.cuda").exists():
        kernel_path = generated_dir / "kernel.cu"
        return (kernel_path if kernel_path.exists() else None), "cuda"
    if (generated_dir / "success.triton").exists():
        kernel_path = generated_dir / "kernel.py"
        return (kernel_path if kernel_path.exists() else None), "triton"
    if (generated_dir / "success.mps").exists():
        kernel_path = generated_dir / "kernel.metal"
        return (kernel_path if kernel_path.exists() else None), "mps"
    return None, ""


def _stage_runtime_cuda_kernel(source_path: Path, op_name: str) -> tuple[Any, Path]:
    from src.optimizer.backends.cuda import loader

    runtime_root = Path(tempfile.mkdtemp(prefix="kforge-integrated-"))
    runtime_dir = runtime_root / "kernel"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_kernel = runtime_dir / "kernel.cu"
    shutil.copy2(source_path, runtime_kernel)
    module_suffix = "".join(
        char if (char.isalnum() or char == "_") else "_"
        for char in runtime_root.name
    )
    ext = loader.load_kernel(
        runtime_dir,
        name=f"kforge_integrated_{op_name}_{module_suffix}",
        build_dir=runtime_dir / ".build",
    )
    return ext, runtime_root


def _measure_integrated_kernel_source(
    source_path: Path | None,
    *,
    backend: str,
    op_name: str,
    func: Any,
    entries: list[BenchmarkEntry],
    device: str,
) -> tuple[dict[str, Any] | None, str]:
    if source_path is None or not source_path.exists():
        return None, "missing_kernel_source"
    if not entries:
        return None, "missing_io_entries"

    backend_name = str(backend or _backend_for_suffix(source_path.suffix))
    if backend_name != "cuda":
        return None, "unsupported_integrated_backend"
    if device != "cuda" or not torch.cuda.is_available():
        return None, "unsupported_integrated_device"

    runtime_root: Path | None = None
    try:
        ext, runtime_root = _stage_runtime_cuda_kernel(source_path, op_name)
        runtime_kernel = runtime_root / "kernel" / "kernel.cu"
        launch_params = launch_params_for_runtime_kernel(runtime_kernel, ext.launch)
        adapter_totals = empty_adapter_stats()
        entry_calls = []

        for entry_file, args, kwargs, signature_meta in entries:
            d_args = _move_to_device(args, device)
            d_kwargs = _move_to_device(kwargs, device)

            def invoke(
                bound_args=d_args,
                bound_kwargs=d_kwargs,
                bound_signature=signature_meta,
            ):
                call_adapter_stats = empty_adapter_stats()
                result = invoke_kernel_launch(
                    ext,
                    args=bound_args,
                    kwargs=bound_kwargs,
                    launch_params=launch_params,
                    op_name=op_name,
                    func=func,
                    signature_meta=bound_signature,
                    ensure_device=None,
                    force_contiguous=False,
                    adapter_stats=call_adapter_stats,
                )
                merge_adapter_stats(adapter_totals, call_adapter_stats)
                return result

            entry_calls.append((entry_file, invoke))

        measurement = benchmark_entry_calls(
            entry_calls,
            device=device,
            warmup_runs=DEFAULT_WARMUP_RUNS,
            timed_runs=DEFAULT_TIMED_RUNS,
        )
        measurement["correctness_summary"] = _measure_integrated_correctness(
            ext,
            launch_params=launch_params,
            op_name=op_name,
            func=func,
            entries=entries,
            device=device,
        )
        measurement["adapter_stats"] = adapter_totals
        measurement["adapter_mode"] = {
            "ensure_device": "",
            "force_contiguous": False,
        }
        return measurement, "ok"
    except Exception:
        return None, "integrated_profile_error"
    finally:
        if runtime_root is not None:
            shutil.rmtree(runtime_root, ignore_errors=True)


def _winner_from_measurements(
    *,
    pytorch_ms: float,
    candidate_status: str,
    candidate_ms: float | None,
    correctness_ok: bool = True,
) -> str:
    if (
        correctness_ok
        and candidate_status == "ok"
        and candidate_ms is not None
        and pytorch_ms > 0.0
        and candidate_ms < pytorch_ms
    ):
        return "optimized"
    return "pytorch"


def _safe_deployment_winner(
    *,
    pytorch_ms: float,
    candidate_status: str,
    candidate_ms: float | None,
    correctness_summary: dict[str, Any] | None,
) -> str:
    strict_ok = bool(
        isinstance(correctness_summary, dict)
        and correctness_summary.get("strict_pass") is True
    )
    return _winner_from_measurements(
        pytorch_ms=pytorch_ms,
        candidate_status=candidate_status,
        candidate_ms=candidate_ms,
        correctness_ok=strict_ok,
    )


def _normalize_op_dir_name(name: str) -> str:
    return str(name).replace(".", "_").replace("/", "_")


def _load_op_counts(summary_path: Path) -> dict[str, int]:
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = data.get("op_counts") if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for full_name, count in raw.items():
        op_dir_name = _normalize_op_dir_name(str(full_name))
        try:
            result[op_dir_name] = int(count)
        except Exception:
            result[op_dir_name] = 0
    return result


def _discover_captured_op_dirs(io_root: Path) -> dict[str, Path]:
    op_dirs: dict[str, Path] = {}
    if not io_root.exists():
        return op_dirs
    for d in io_root.iterdir():
        if not d.is_dir():
            continue
        has_entries = any(child.name.startswith("entry_") and child.suffix == ".pt" for child in d.iterdir())
        if has_entries:
            op_dirs[d.name] = d
    return op_dirs


def _select_candidate_ops(
    op_dirs: dict[str, Path],
    op_counts: dict[str, int],
    selected_ops: list[str],
) -> tuple[list[str], set[str]]:
    discovered_ops = sorted(op_dirs.keys())
    if discovered_ops:
        candidate_ops = list(discovered_ops)
        allowed_existing_ops = set(discovered_ops)
    else:
        candidate_ops = sorted(set(op_counts.keys()))
        allowed_existing_ops = set(candidate_ops)

    if selected_ops:
        selected_set = set(selected_ops)
        candidate_ops = [op for op in candidate_ops if op in selected_set]
    return candidate_ops, allowed_existing_ops


def _benchmark_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("benchmarks", "results"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _normalize_measurement_status(status: str) -> str:
    raw = str(status or "").strip().lower()
    if raw in {"ok", "ready"}:
        return "ready"
    if raw.startswith("missing"):
        return "missing"
    if "error" in raw:
        return "error"
    return raw or "unknown"


def _selection_reason(
    *,
    winner: str,
    deployment_winner: str,
    deployment_safe_winner: str,
    integrated_kernel_status: str,
    correctness_summary: dict[str, Any] | None,
) -> str:
    if deployment_safe_winner == "optimized":
        return "strict-pass + speedup + deployment replay success"
    strict_pass = bool(
        isinstance(correctness_summary, dict)
        and correctness_summary.get("strict_pass") is True
    )
    if deployment_winner == "optimized" and not strict_pass:
        return "faster in deployment replay but failed strict correctness"
    if deployment_winner == "optimized" and integrated_kernel_status != "ok":
        return "faster in replay candidate path but integrated benchmark did not complete cleanly"
    if winner == "optimized":
        return "microbenchmark win only; deployment-safe evidence insufficient"
    return "PyTorch remains the recommended backend"


def _build_benchmark_row(
    *,
    op_name: str,
    benchmarked_entry_count: int,
    available_entries: int,
    benchmarked_entry_files: list[str],
    pytorch_ms: float,
    pytorch_entry_latencies: list[float],
    kernel_ms: float | None,
    kernel_entry_latencies: list[float],
    kernel_status: str,
    winner: str,
    integrated_kernel_ms: float | None,
    integrated_kernel_entry_latencies: list[float],
    integrated_kernel_status: str,
    deployment_winner: str,
    deployment_safe_winner: str,
    baseline_source: str,
    backend: str,
    integrated_kernel_source: str,
    integrated_adapter_stats: dict[str, int],
    integrated_correctness_summary: dict[str, Any] | None,
    kernel_estimated: bool,
    kernel_benchmarked_entry_files: list[str],
    integrated_kernel_benchmarked_entry_files: list[str],
) -> dict[str, Any]:
    speedup = (
        float(pytorch_ms / kernel_ms)
        if pytorch_ms and kernel_ms and kernel_ms > 0
        else None
    )
    deployment_speedup = (
        float(pytorch_ms / integrated_kernel_ms)
        if pytorch_ms and integrated_kernel_ms and integrated_kernel_ms > 0
        else None
    )
    selection_backend = (
        "optimized" if deployment_safe_winner == "optimized" else "pytorch"
    )
    selection_reason = _selection_reason(
        winner=winner,
        deployment_winner=deployment_winner,
        deployment_safe_winner=deployment_safe_winner,
        integrated_kernel_status=integrated_kernel_status,
        correctness_summary=integrated_correctness_summary,
    )

    micro_payload: dict[str, Any] = {
        "status": _normalize_measurement_status(kernel_status),
        "sample_count": int(benchmarked_entry_count),
        "pytorch_ms": float(pytorch_ms),
        "kernel_ms": float(kernel_ms) if kernel_ms is not None else None,
        "speedup": speedup,
        "winner": winner,
        "entry_files": list(benchmarked_entry_files),
        "entry_latencies_ms": list(kernel_entry_latencies),
        "baseline_entry_latencies_ms": list(pytorch_entry_latencies),
    }
    if backend:
        micro_payload["backend"] = backend
    if kernel_estimated:
        micro_payload["estimated"] = True

    deployment_payload: dict[str, Any] = {
        "status": _normalize_measurement_status(integrated_kernel_status),
        "sample_count": int(benchmarked_entry_count),
        "replay_pytorch_ms": float(pytorch_ms),
        "replay_kernel_ms": (
            float(integrated_kernel_ms) if integrated_kernel_ms is not None else None
        ),
        "replay_speedup": deployment_speedup,
        "winner": deployment_winner,
        "safe": deployment_safe_winner == "optimized",
        "recommended_backend": selection_backend,
        "entry_files": list(integrated_kernel_benchmarked_entry_files or benchmarked_entry_files),
        "entry_latencies_ms": list(integrated_kernel_entry_latencies),
        "coverage": {
            "adapter_stats": dict(integrated_adapter_stats),
        },
        "fallback": {
            "supported": integrated_kernel_status == "ok",
            "observed": 0 if integrated_kernel_status == "ok" else None,
            "reasons": [] if integrated_kernel_status == "ok" else [integrated_kernel_status],
        },
    }
    if integrated_correctness_summary:
        deployment_payload["correctness"] = dict(integrated_correctness_summary)
    if integrated_kernel_source:
        deployment_payload["kernel_source"] = integrated_kernel_source
    if backend:
        deployment_payload["backend"] = backend

    row: dict[str, Any] = {
        "op": op_name,
        "selection": {
            "recommended_mode": "deployment",
            "recommended_backend": selection_backend,
            "recommended_reason": selection_reason,
            "export_allowed": selection_backend == "optimized",
            "unsafe_override_required": (
                winner == "optimized" and selection_backend != "optimized"
            ),
        },
        "micro": micro_payload,
        "deployment": deployment_payload,
        "stress": {
            "status": "not_run",
        },
        "e2e": {
            "status": "not_run",
        },
        # Legacy fields retained for readers that still expect the flat schema.
        "entries": benchmarked_entry_count,
        "available_entries": available_entries,
        "benchmarked_entry_count": benchmarked_entry_count,
        "benchmarked_entry_files": benchmarked_entry_files,
        "pytorch_ms": float(pytorch_ms),
        "pytorch_entry_latencies_ms": pytorch_entry_latencies,
        "kernel_ms": float(kernel_ms) if kernel_ms is not None else None,
        "kernel_entry_latencies_ms": kernel_entry_latencies,
        "kernel_status": kernel_status,
        "winner": winner,
        "integrated_kernel_ms": (
            float(integrated_kernel_ms) if integrated_kernel_ms is not None else None
        ),
        "integrated_kernel_entry_latencies_ms": integrated_kernel_entry_latencies,
        "integrated_kernel_status": integrated_kernel_status,
        "deployment_winner": deployment_winner,
        "deployment_safe_winner": deployment_safe_winner,
        "baseline_source": baseline_source,
    }
    if kernel_benchmarked_entry_files:
        row["kernel_benchmarked_entry_files"] = kernel_benchmarked_entry_files
    if integrated_kernel_benchmarked_entry_files:
        row["integrated_kernel_benchmarked_entry_files"] = (
            integrated_kernel_benchmarked_entry_files
        )
    if backend:
        row["backend"] = backend
    if integrated_kernel_source:
        row["deployment_kernel_source"] = integrated_kernel_source
    if integrated_adapter_stats:
        row["integration_adapter_stats"] = integrated_adapter_stats
    if integrated_correctness_summary:
        row["deployment_correctness"] = integrated_correctness_summary
    if kernel_estimated:
        row["kernel_estimated"] = True
    if speedup is not None:
        row["speedup"] = speedup
    if deployment_speedup is not None:
        row["deployment_speedup"] = deployment_speedup
    return row


def _build_output_payload(
    *,
    project: str,
    status: str,
    device: str,
    runtime_fingerprint: str,
    optimized_root: Path | None,
    rows: list[dict[str, Any]],
    errors: list[str],
    current: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "project": project,
        "timestamp": _now_iso(),
        "status": status,
        "device": device,
        "available_modes": ["micro", "deployment", "stress", "e2e"],
        "default_mode": "deployment",
        "selection_policy": "safe",
        "benchmark_protocol": {
            "warmup_runs": DEFAULT_WARMUP_RUNS,
            "timed_runs": DEFAULT_TIMED_RUNS,
            "notes": (
                "Direct kernel replay and deployment-style integrated replay; "
                "no confidence intervals."
            ),
        },
        "runtime_fingerprint": json.loads(runtime_fingerprint),
        "optimized_dir": str(optimized_root) if optimized_root else "",
        "benchmarks": rows,
        "results": rows,
        "errors": errors,
    }
    if current is not None and total is not None:
        payload["progress"] = {
            "current": int(current),
            "total": int(total),
            "percent": (float(current) / float(total)) if total else 0.0,
        }
    return payload


def _apply_generated_profile_result(
    *,
    op_name: str,
    pytorch_ms: float,
    kernel_status: str,
    kernel_ms: float | None,
    backend: str,
    kernel_estimated: bool,
    kernel_entry_latencies: list[float],
    kernel_benchmarked_entry_files: list[str],
    generated_stats: dict[str, Any] | None,
    generated_status: str,
    generated_backend: str,
    errors: list[str],
) -> tuple[str, float | None, str, bool, list[float], list[str]]:
    if generated_stats is not None:
        kernel_ms_raw = (
            generated_stats.get("mean_time_ms")
            if isinstance(generated_stats, dict)
            else None
        )
        kernel_ms = float(kernel_ms_raw) if kernel_ms_raw is not None else None
        kernel_status = "ok"
        kernel_entry_latencies = (
            generated_stats.get("entry_latencies_ms")
            if isinstance(generated_stats, dict)
            else []
        ) or []
        kernel_benchmarked_entry_files = (
            generated_stats.get("entry_files")
            if isinstance(generated_stats, dict)
            else []
        ) or []
        if generated_backend:
            backend = generated_backend
        return (
            kernel_status,
            kernel_ms,
            backend,
            kernel_estimated,
            kernel_entry_latencies,
            kernel_benchmarked_entry_files,
        )

    if kernel_status in {"missing", "missing_optimized_dir", "missing_optimized_kernel"}:
        kernel_status = generated_status
    if generated_backend and not backend:
        backend = generated_backend

    # Do not silently copy PyTorch latency into kernel_ms when direct kernel
    # profiling is unavailable. That produces fake 1.0x speedups.
    if generated_status == "generated_profile_error_ninja" and pytorch_ms > 0.0:
        msg = (
            f"{op_name}: generated kernel profiling unavailable "
            "(install ninja for direct kernel benchmarking)"
        )
        if msg not in errors:
            errors.append(msg)
        return (
            kernel_status,
            kernel_ms,
            backend,
            False,
            kernel_entry_latencies,
            kernel_benchmarked_entry_files,
        )

    if generated_status in {
        "generated_profile_error",
        "generated_profile_error_ninja",
    }:
        msg = f"{op_name}: generated kernel benchmark failed"
        if msg not in errors:
            errors.append(msg)

    return (
        kernel_status,
        kernel_ms,
        backend,
        kernel_estimated,
        kernel_entry_latencies,
        kernel_benchmarked_entry_files,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--max-entries", type=int, default=50)
    parser.add_argument("--ops", default="")
    args = parser.parse_args()
    _ensure_local_toolchain_on_path()

    project_dir = project_dir_for_name(args.project)
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    output_path = bench_dir / "op_benchmarks.json"
    cache_path = bench_dir / "torch_baseline_cache.json"

    try:
        io_root = project_dir / "io" / "individual_ops"
        summary_path = project_dir / "io" / "summary.json"
        op_counts = _load_op_counts(summary_path)
        if not io_root.exists() and not op_counts:
            write_json_file(
                output_path,
                {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "project": args.project,
                    "timestamp": _now_iso(),
                    "status": "empty",
                    "available_modes": ["micro", "deployment", "stress", "e2e"],
                    "default_mode": "deployment",
                    "selection_policy": "safe",
                    "benchmarks": [],
                    "results": [],
                    "errors": ["No profiling entries or summary found under io/"],
                },
            )
            return 0

        device = _resolve_device()
        _global_warmup(device)
        runtime_fingerprint = _runtime_fingerprint(device)

        optimized_root = find_latest_optimized_dir(args.project)

        cache_raw = read_json_file(cache_path, {})
        cache: dict[str, dict[str, Any]] = {}
        if isinstance(cache_raw, dict):
            for k, v in cache_raw.items():
                normalized = _coerce_cached_measurement(v)
                if normalized is not None:
                    cache[k] = normalized

        results_by_op: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        op_dirs = _discover_captured_op_dirs(io_root)
        selected_ops = _ops_from_csv(args.ops)
        candidate_ops, allowed_existing_ops = _select_candidate_ops(
            op_dirs, op_counts, selected_ops
        )

        existing_payload = read_json_file(output_path, {})
        if isinstance(existing_payload, dict):
            existing_results = _benchmark_rows_from_payload(existing_payload)
            for r in existing_results:
                if not (isinstance(r, dict) and r.get("op")):
                    continue
                op_name = str(r["op"])
                if allowed_existing_ops and op_name not in allowed_existing_ops:
                    continue
                results_by_op[op_name] = r
        if selected_ops:
            for op_name in selected_ops:
                results_by_op.pop(op_name, None)

        total_ops = len(candidate_ops)
        progress_offset = 0
        progress_total_override = 0
        try:
            progress_offset = int(os.environ.get("KFORGE_PROGRESS_OFFSET", "0") or "0")
        except Exception:
            progress_offset = 0
        try:
            progress_total_override = int(
                os.environ.get("KFORGE_PROGRESS_TOTAL", "0") or "0"
            )
        except Exception:
            progress_total_override = 0

        def _update_phase_progress(current_phase: int, message: str) -> None:
            absolute_total = (
                progress_total_override if progress_total_override > 0 else total_ops
            )
            absolute_current = progress_offset + current_phase
            if absolute_current < 0:
                absolute_current = 0
            if absolute_total > 0 and absolute_current > absolute_total:
                absolute_current = absolute_total
            update_job_progress(absolute_current, absolute_total, message)

        if total_ops <= 0:
            write_json_file(cache_path, cache)
            rows = [results_by_op[k] for k in sorted(results_by_op.keys())]
            write_json_file(
                output_path,
                _build_output_payload(
                    project=args.project,
                    status="empty",
                    device=device,
                    runtime_fingerprint=runtime_fingerprint,
                    optimized_root=optimized_root,
                    rows=rows,
                    errors=errors,
                ),
            )
            print(f"[benchmarking.benchmark_ops] Wrote {output_path}")
            return 0

        def _write_incremental(status: str, current: int, total: int) -> None:
            payload = _build_output_payload(
                project=args.project,
                status=status,
                device=device,
                runtime_fingerprint=runtime_fingerprint,
                optimized_root=optimized_root,
                rows=[results_by_op[k] for k in sorted(results_by_op.keys())],
                errors=errors,
                current=current,
                total=total,
            )
            write_json_file(output_path, payload)

        for idx, op_name in enumerate(candidate_ops):
            _update_phase_progress(
                idx,
                f"Benchmarking {op_name} ({idx + 1}/{total_ops})",
            )
            op_dir = op_dirs.get(op_name)
            entries = _load_entries(op_dir, args.max_entries) if op_dir else []
            func = _get_pytorch_func(op_name)

            entry_sig = _entry_signature(entries) if entries else "summary_only"
            cache_key = (
                f"{runtime_fingerprint}:{op_name}:{entry_sig}:"
                f"warmup={DEFAULT_WARMUP_RUNS},runs={DEFAULT_TIMED_RUNS}"
            )
            pytorch_measurement = cache.get(cache_key)
            baseline_source = "cache" if pytorch_measurement is not None else ""
            if pytorch_measurement is None:
                pytorch_measurement = {
                    "mean_time_ms": 0.0,
                    "entry_files": [entry_file for entry_file, _, _, _ in entries],
                    "entry_latencies_ms": [],
                    "entry_results": [],
                    "entry_count": len(entries),
                    "errors": [],
                    "warmup_runs": DEFAULT_WARMUP_RUNS,
                    "timed_runs": DEFAULT_TIMED_RUNS,
                }
                if func and entries:
                    try:
                        pytorch_measurement = _measure_pytorch(func, entries, device)
                        baseline_source = "measured"
                    except Exception as e:
                        errors.append(f"{op_name}: pytorch benchmark failed: {e}")
                        baseline_source = "error"
                else:
                    baseline_source = "unavailable"
                cache[cache_key] = pytorch_measurement

            pytorch_ms = float(pytorch_measurement.get("mean_time_ms") or 0.0)
            benchmarked_entry_files = pytorch_measurement.get("entry_files") or [
                entry_file for entry_file, _, _, _ in entries
            ]
            benchmarked_entry_count = int(
                pytorch_measurement.get("entry_count") or len(benchmarked_entry_files)
            )
            pytorch_entry_latencies = pytorch_measurement.get("entry_latencies_ms") or []

            count = op_counts.get(op_name, len(entries))
            try:
                count = int(count)
            except Exception:
                count = len(entries)

            kernel_ms = None
            kernel_status = "missing_optimized_dir" if not optimized_root else "missing_optimized_kernel"
            backend = ""
            kernel_estimated = False
            kernel_entry_latencies = []
            kernel_benchmarked_entry_files = []
            benchmark_entry_paths = (
                [op_dir / name for name in benchmarked_entry_files]
                if op_dir and benchmarked_entry_files
                else None
            )

            optimized_stats, optimized_status, optimized_backend = _profile_tree_kernel_ms(
                optimized_root,
                op_name,
                op_dir,
                benchmark_entry_files=benchmark_entry_paths,
            )
            (
                kernel_status,
                kernel_ms,
                backend,
                kernel_estimated,
                kernel_entry_latencies,
                kernel_benchmarked_entry_files,
            ) = _apply_generated_profile_result(
                op_name=op_name,
                pytorch_ms=pytorch_ms,
                kernel_status=kernel_status,
                kernel_ms=kernel_ms,
                backend=backend,
                kernel_estimated=kernel_estimated,
                kernel_entry_latencies=kernel_entry_latencies,
                kernel_benchmarked_entry_files=kernel_benchmarked_entry_files,
                generated_stats=optimized_stats,
                generated_status=optimized_status,
                generated_backend=optimized_backend,
                errors=errors,
            )

            if kernel_status != "ok":
                generated_stats, generated_status, generated_backend = _profile_generated_kernel_ms(
                    project_dir,
                    op_name,
                    op_dir,
                    benchmark_entry_files=benchmark_entry_paths,
                )
                (
                    kernel_status,
                    kernel_ms,
                    backend,
                    kernel_estimated,
                    kernel_entry_latencies,
                    kernel_benchmarked_entry_files,
                ) = _apply_generated_profile_result(
                    op_name=op_name,
                    pytorch_ms=pytorch_ms,
                    kernel_status=kernel_status,
                    kernel_ms=kernel_ms,
                    backend=backend,
                    kernel_estimated=kernel_estimated,
                    kernel_entry_latencies=kernel_entry_latencies,
                    kernel_benchmarked_entry_files=kernel_benchmarked_entry_files,
                    generated_stats=generated_stats,
                    generated_status=generated_status,
                    generated_backend=generated_backend,
                    errors=errors,
                )

            integrated_kernel_ms = None
            integrated_kernel_status = "missing_optimized_dir" if not optimized_root else "missing_optimized_kernel"
            integrated_kernel_entry_latencies: list[float] = []
            integrated_kernel_benchmarked_entry_files: list[str] = []
            integrated_kernel_source = ""
            integrated_adapter_stats: dict[str, int] = {}
            integrated_correctness_summary: dict[str, Any] | None = None

            integrated_candidates: list[tuple[str, Path | None, str, str]] = []
            optimized_source_path, optimized_source_backend = _resolve_tree_kernel_source(
                optimized_root,
                op_name,
            )
            integrated_candidates.append(
                (
                    "optimized_tree",
                    optimized_source_path,
                    optimized_source_backend,
                    "missing_optimized_dir" if not optimized_root else "missing_optimized_kernel",
                )
            )
            generated_source_path, generated_source_backend = _resolve_generated_kernel_source(
                project_dir,
                op_name,
            )
            integrated_candidates.append(
                (
                    "generated",
                    generated_source_path,
                    generated_source_backend,
                    "missing_generated",
                )
            )

            for source_kind, source_path, source_backend, missing_status in integrated_candidates:
                if source_path is None:
                    if integrated_kernel_status in {"missing_optimized_dir", "missing_optimized_kernel"}:
                        integrated_kernel_status = missing_status
                    continue
                integrated_measurement, candidate_status = _measure_integrated_kernel_source(
                    source_path,
                    backend=source_backend,
                    op_name=op_name,
                    func=func,
                    entries=entries,
                    device=device,
                )
                if integrated_measurement is None:
                    integrated_kernel_status = candidate_status
                    if candidate_status == "integrated_profile_error":
                        errors.append(
                            f"{op_name}: integrated kernel benchmark failed ({source_kind}: {candidate_status})"
                        )
                    continue

                integrated_kernel_status = candidate_status
                integrated_kernel_ms_raw = integrated_measurement.get("mean_time_ms")
                integrated_kernel_ms = (
                    float(integrated_kernel_ms_raw)
                    if integrated_kernel_ms_raw is not None
                    else None
                )
                integrated_kernel_entry_latencies = (
                    integrated_measurement.get("entry_latencies_ms") or []
                )
                integrated_kernel_benchmarked_entry_files = (
                    integrated_measurement.get("entry_files") or []
                )
                integrated_adapter_stats = (
                    integrated_measurement.get("adapter_stats")
                    if isinstance(integrated_measurement.get("adapter_stats"), dict)
                    else {}
                )
                integrated_correctness_summary = (
                    integrated_measurement.get("correctness_summary")
                    if isinstance(integrated_measurement.get("correctness_summary"), dict)
                    else None
                )
                integrated_kernel_source = source_kind
                break

            if count <= 0 and pytorch_ms <= 0.0 and kernel_status != "ok":
                _update_phase_progress(
                    idx + 1,
                    f"Benchmarked {idx + 1}/{total_ops} operators.",
                )
                _write_incremental(
                    "partial" if (idx + 1) < total_ops or errors else "ready",
                    idx + 1,
                    total_ops,
                )
                continue

            winner = _winner_from_measurements(
                pytorch_ms=pytorch_ms,
                candidate_status=kernel_status,
                candidate_ms=kernel_ms,
            )
            deployment_winner = _winner_from_measurements(
                pytorch_ms=pytorch_ms,
                candidate_status=integrated_kernel_status,
                candidate_ms=integrated_kernel_ms,
            )
            deployment_safe_winner = _safe_deployment_winner(
                pytorch_ms=pytorch_ms,
                candidate_status=integrated_kernel_status,
                candidate_ms=integrated_kernel_ms,
                correctness_summary=integrated_correctness_summary,
            )

            row = _build_benchmark_row(
                op_name=op_name,
                benchmarked_entry_count=benchmarked_entry_count,
                available_entries=count,
                benchmarked_entry_files=benchmarked_entry_files,
                pytorch_ms=pytorch_ms,
                pytorch_entry_latencies=pytorch_entry_latencies,
                kernel_ms=kernel_ms,
                kernel_entry_latencies=kernel_entry_latencies,
                kernel_status=kernel_status,
                winner=winner,
                integrated_kernel_ms=integrated_kernel_ms,
                integrated_kernel_entry_latencies=integrated_kernel_entry_latencies,
                integrated_kernel_status=integrated_kernel_status,
                deployment_winner=deployment_winner,
                deployment_safe_winner=deployment_safe_winner,
                baseline_source=baseline_source,
                backend=backend,
                integrated_kernel_source=integrated_kernel_source,
                integrated_adapter_stats=integrated_adapter_stats,
                integrated_correctness_summary=integrated_correctness_summary,
                kernel_estimated=kernel_estimated,
                kernel_benchmarked_entry_files=kernel_benchmarked_entry_files,
                integrated_kernel_benchmarked_entry_files=integrated_kernel_benchmarked_entry_files,
            )
            results_by_op[op_name] = row
            if kernel_status == "ok" or integrated_kernel_status == "ok":
                try:
                    write_root_benchmark_metadata(
                        project_dir,
                        op_name,
                        micro_kernel_ms=kernel_ms,
                        deployment_kernel_ms=integrated_kernel_ms,
                        backend=backend,
                        benchmark_source="benchmark_ops_refresh",
                    )
                except Exception as e:
                    errors.append(f"{op_name}: tree sync failed: {e}")
            _update_phase_progress(
                idx + 1,
                f"Benchmarked {idx + 1}/{total_ops} operators.",
            )
            _write_incremental(
                "partial" if (idx + 1) < total_ops or errors else "ready",
                idx + 1,
                total_ops,
            )

        write_json_file(cache_path, cache)
        results = [results_by_op[k] for k in sorted(results_by_op.keys())]
        status = "ready" if results else "empty"
        if errors and status == "ready":
            status = "partial"
        _write_incremental(status, total_ops, total_ops)
        print(f"[benchmarking.benchmark_ops] Wrote {output_path}")

        return 0
    except Exception as e:
        write_json_file(
            output_path,
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "project": args.project,
                "timestamp": _now_iso(),
                "status": "error",
                "available_modes": ["micro", "deployment", "stress", "e2e"],
                "default_mode": "deployment",
                "selection_policy": "safe",
                "benchmarks": [],
                "results": [],
                "errors": [str(e)],
            },
        )
        print(f"[benchmarking.benchmark_ops] Failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
