from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.progress import update_job_progress
from src.optimizer.tree_store import update_root_value

from .paths import find_latest_optimized_dir, project_dir_for_name
from .state import read_json_file, write_json_file


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


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
        _sync_device(device)
    except Exception:
        pass


def _entry_signature(entries: list[tuple[Any, dict[str, Any]]]) -> str:
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
    payload = [(_sig(args), _sig(kwargs)) for args, kwargs in sample]
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


def _load_entries(io_dir: Path, max_entries: int) -> list[tuple[Any, dict[str, Any]]]:
    entries: list[tuple[Any, dict[str, Any]]] = []
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
        if kwargs is None:
            kwargs = {}
        entries.append((args, kwargs))
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


def _measure_pytorch(func, entries: list[tuple[Any, dict[str, Any]]], device: str, repeats: int = 10) -> float:
    if not entries:
        return 0.0

    warmup_entries = entries[: min(3, len(entries))]
    for _ in range(3):
        for args, kwargs in warmup_entries:
            d_args = _move_to_device(args, device)
            d_kwargs = _move_to_device(kwargs, device)
            try:
                _run_call(func, d_args, d_kwargs)
            except Exception:
                pass
    _sync_device(device)

    timings: list[float] = []
    if device == "cuda":
        for args, kwargs in entries:
            d_args = _move_to_device(args, device)
            d_kwargs = _move_to_device(kwargs, device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ok = False
            for _ in range(repeats):
                try:
                    _run_call(func, d_args, d_kwargs)
                    ok = True
                except Exception:
                    break
            end.record()
            _sync_device(device)
            if ok:
                timings.append(start.elapsed_time(end) / repeats)
    else:
        for args, kwargs in entries:
            d_args = _move_to_device(args, device)
            d_kwargs = _move_to_device(kwargs, device)
            t0 = time.perf_counter()
            ok = False
            for _ in range(repeats):
                try:
                    _run_call(func, d_args, d_kwargs)
                    ok = True
                except Exception:
                    break
            _sync_device(device)
            if ok:
                timings.append(((time.perf_counter() - t0) * 1000.0) / repeats)

    if not timings:
        return 0.0
    return float(sum(timings) / len(timings))


def _read_best_kernel_ms(op_dir: Path) -> tuple[float | None, str, str]:
    log_file = op_dir / "improvement_log.json"
    if not log_file.exists():
        return None, "missing", ""
    try:
        data = json.loads(log_file.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return None, "missing", ""
        best = None
        best_ms = None
        for entry in data:
            results = entry.get("results") if isinstance(entry, dict) else None
            if not isinstance(results, dict):
                continue
            ms = results.get("min_time_ms")
            if ms is None:
                continue
            try:
                ms_val = float(ms)
            except Exception:
                continue
            if best_ms is None or ms_val < best_ms:
                best_ms = ms_val
                best = entry
        if best_ms is None:
            return None, "missing", ""
        backend = ""
        if isinstance(best, dict):
            backend = str(best.get("backend") or best.get("provider") or "")
        return best_ms, "ok", backend
    except Exception:
        return None, "error", ""


def _profile_generated_kernel_ms(
    project_dir: Path, op_name: str, io_op_dir: Path | None
) -> tuple[float | None, str, str]:
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
        if io_op_dir is None or not io_op_dir.exists():
            return None, "missing_io_entries", "cuda"
        if not (generated_dir / "kernel.cu").exists():
            return None, "missing_kernel_source", "cuda"
        try:
            from src.optimizer.backends.cuda import CUDABackend

            stats = CUDABackend().profile_kernel(
                {"tmp_dir": generated_dir, "io_dir": io_op_dir},
                baseline=True,
            )
            ms = stats.get("min_time_ms") if isinstance(stats, dict) else None
            if ms is None:
                return None, "generated_profile_missing", "cuda"
            return float(ms), "ok", "cuda"
        except Exception as e:
            msg = str(e).strip()
            if "Ninja is required to load C++ extensions" in msg:
                return None, "generated_profile_error_ninja", "cuda"
            return None, "generated_profile_error", "cuda"

    if (generated_dir / "success.triton").exists():
        return None, "unsupported_generated_backend", "triton"
    if (generated_dir / "success.mps").exists():
        return None, "unsupported_generated_backend", "mps"
    if (generated_dir / "success.cpu").exists():
        return None, "unsupported_generated_backend", "cpu"
    return None, "missing_generated", ""


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


def _load_op_profile_ms(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = data.get("op_profile_ms") if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for full_name, val in raw.items():
        op_dir_name = _normalize_op_dir_name(str(full_name))
        try:
            result[op_dir_name] = float(val)
        except Exception:
            continue
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--max-entries", type=int, default=50)
    parser.add_argument("--ops", default="")
    args = parser.parse_args()

    project_dir = project_dir_for_name(args.project)
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    output_path = bench_dir / "op_benchmarks.json"
    cache_path = bench_dir / "torch_baseline_cache.json"

    try:
        io_root = project_dir / "io" / "individual_ops"
        summary_path = project_dir / "io" / "summary.json"
        op_counts = _load_op_counts(summary_path)
        op_profile_ms = _load_op_profile_ms(summary_path)
        if not io_root.exists() and not op_counts and not op_profile_ms:
            write_json_file(
                output_path,
                {
                    "project": args.project,
                    "timestamp": _now_iso(),
                    "status": "empty",
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
        cache: dict[str, float] = {}
        if isinstance(cache_raw, dict):
            for k, v in cache_raw.items():
                try:
                    cache[k] = float(v)
                except Exception:
                    continue

        results_by_op: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        op_dirs: dict[str, Path] = {}
        if io_root.exists():
            for d in io_root.iterdir():
                if d.is_dir():
                    op_dirs[d.name] = d

        candidate_ops = sorted(set(op_dirs.keys()) | set(op_counts.keys()) | set(op_profile_ms.keys()))
        selected_ops = _ops_from_csv(args.ops)
        if selected_ops:
            selected_set = set(selected_ops)
            candidate_ops = [op for op in candidate_ops if op in selected_set]

        existing_payload = read_json_file(output_path, {})
        if isinstance(existing_payload, dict):
            existing_results = existing_payload.get("results")
            if isinstance(existing_results, list):
                for r in existing_results:
                    if isinstance(r, dict) and r.get("op"):
                        results_by_op[str(r["op"])] = r
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
            write_json_file(
                output_path,
                {
                    "project": args.project,
                    "timestamp": _now_iso(),
                    "status": "empty",
                    "device": device,
                    "runtime_fingerprint": json.loads(runtime_fingerprint),
                    "optimized_dir": str(optimized_root) if optimized_root else "",
                    "results": [results_by_op[k] for k in sorted(results_by_op.keys())],
                    "errors": errors,
                },
            )
            print(f"[benchmarking.benchmark_ops] Wrote {output_path}")
            return 0

        def _write_incremental(status: str, current: int, total: int) -> None:
            payload = {
                "project": args.project,
                "timestamp": _now_iso(),
                "status": status,
                "device": device,
                "runtime_fingerprint": json.loads(runtime_fingerprint),
                "optimized_dir": str(optimized_root) if optimized_root else "",
                "results": [results_by_op[k] for k in sorted(results_by_op.keys())],
                "errors": errors,
                "progress": {
                    "current": int(current),
                    "total": int(total),
                    "percent": (float(current) / float(total)) if total else 0.0,
                },
            }
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
            cache_key = f"{runtime_fingerprint}:{op_name}:{entry_sig}"
            pytorch_ms = cache.get(cache_key)
            baseline_source = "cache" if pytorch_ms is not None else ""
            if pytorch_ms is None:
                pytorch_ms = 0.0
                if func and entries:
                    try:
                        pytorch_ms = _measure_pytorch(func, entries, device)
                        baseline_source = "measured"
                    except Exception as e:
                        errors.append(f"{op_name}: pytorch benchmark failed: {e}")
                        pytorch_ms = 0.0
                        baseline_source = "error"
                elif op_name in op_profile_ms:
                    pytorch_ms = float(op_profile_ms[op_name])
                    baseline_source = "profile_summary"
                else:
                    baseline_source = "unavailable"
                cache[cache_key] = float(pytorch_ms)

            count = op_counts.get(op_name, len(entries))
            try:
                count = int(count)
            except Exception:
                count = len(entries)

            kernel_ms = None
            kernel_status = "missing"
            backend = ""
            kernel_estimated = False
            if optimized_root:
                kernel_ms, kernel_status, backend = _read_best_kernel_ms(optimized_root / op_name)
            else:
                kernel_status = "missing_optimized_dir"

            if kernel_status != "ok":
                generated_ms, generated_status, generated_backend = _profile_generated_kernel_ms(
                    project_dir, op_name, op_dir
                )
                if generated_ms is not None:
                    kernel_ms = generated_ms
                    kernel_status = "ok"
                    if generated_backend:
                        backend = generated_backend
                elif generated_status == "generated_profile_error_ninja" and pytorch_ms > 0.0:
                    kernel_ms = float(pytorch_ms)
                    kernel_status = "ok"
                    kernel_estimated = True
                    if generated_backend and not backend:
                        backend = generated_backend
                    errors.append(
                        f"{op_name}: generated kernel profiling unavailable (install ninja for direct kernel benchmarking)"
                    )
                else:
                    if kernel_status in {"missing", "missing_optimized_dir"}:
                        kernel_status = generated_status
                    if generated_backend and not backend:
                        backend = generated_backend
                    if generated_status in {
                        "generated_profile_error",
                        "generated_profile_error_ninja",
                    }:
                        errors.append(f"{op_name}: generated kernel benchmark failed")

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

            winner = "pytorch"
            if kernel_status == "ok" and kernel_ms is not None and pytorch_ms and kernel_ms < pytorch_ms:
                winner = "optimized"

            row = {
                "op": op_name,
                "entries": count,
                "pytorch_ms": float(pytorch_ms),
                "kernel_ms": float(kernel_ms) if kernel_ms is not None else None,
                "kernel_status": kernel_status,
                "winner": winner,
                "baseline_source": baseline_source,
            }
            if backend:
                row["backend"] = backend
            if kernel_estimated:
                row["kernel_estimated"] = True
            if pytorch_ms and kernel_ms:
                row["speedup"] = float(pytorch_ms / kernel_ms) if kernel_ms > 0 else None
            results_by_op[op_name] = row
            if kernel_status == "ok" and kernel_ms is not None:
                try:
                    update_root_value(
                        project_dir,
                        op_name,
                        kernel_ms,
                        description="Generated baseline kernel (benchmarked)",
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
                "project": args.project,
                "timestamp": _now_iso(),
                "status": "error",
                "results": [],
                "errors": [str(e)],
            },
        )
        print(f"[benchmarking.benchmark_ops] Failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
