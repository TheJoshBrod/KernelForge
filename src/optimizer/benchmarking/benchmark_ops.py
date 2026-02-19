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

from .paths import find_latest_optimized_dir, project_dir_for_name
from .state import read_json_file, write_json_file


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_device() -> str:
    target = os.environ.get("CGINS_TARGET_DEVICE", "").strip().lower()
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
            ms = results.get("mean_time_ms")
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

        results = []
        errors: list[str] = []
        op_dirs: dict[str, Path] = {}
        if io_root.exists():
            for d in io_root.iterdir():
                if d.is_dir():
                    op_dirs[d.name] = d

        candidate_ops = sorted(set(op_dirs.keys()) | set(op_counts.keys()) | set(op_profile_ms.keys()))
        for op_name in candidate_ops:
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
            if optimized_root:
                kernel_ms, kernel_status, backend = _read_best_kernel_ms(optimized_root / op_name)
            else:
                kernel_status = "missing_optimized_dir"

            if count <= 0 and pytorch_ms <= 0.0 and kernel_status != "ok":
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
            if pytorch_ms and kernel_ms:
                row["speedup"] = float(pytorch_ms / kernel_ms) if kernel_ms > 0 else None
            results.append(row)

        write_json_file(cache_path, cache)
        status = "ready" if results else "empty"
        if errors and status == "ready":
            status = "partial"
        payload = {
            "project": args.project,
            "timestamp": _now_iso(),
            "status": status,
            "device": device,
            "runtime_fingerprint": json.loads(runtime_fingerprint),
            "optimized_dir": str(optimized_root) if optimized_root else "",
            "results": results,
            "errors": errors,
        }
        write_json_file(output_path, payload)
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
