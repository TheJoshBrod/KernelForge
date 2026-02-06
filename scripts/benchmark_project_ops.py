import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.optimizer.components.hardware.profiler import load_batch, get_module
from src.progress import update_job_progress, wait_if_paused, check_cancelled


def find_optimized_dir(project_name: str) -> Path | None:
    base_dir = Path("kernels/optimized")
    if not base_dir.exists():
        return None
    suffix = f"_{project_name}"
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_torch_func(op_name: str):
    prefix = "torch_nn_functional_"
    if op_name.startswith(prefix):
        func_name = op_name[len(prefix):]
    else:
        func_name = op_name
    return getattr(F, func_name, None)


def _normalize_op_name(op_name: str) -> str:
    return str(op_name).strip().replace(".", "_").replace("/", "_")


def _resolve_device() -> str:
    value = os.environ.get("CGINS_TARGET_DEVICE", "").strip().lower()
    if value == "mps":
        return "mps"
    if value in {"gpu", "cuda"}:
        return "cuda"
    if value == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def measure_pytorch(func, inputs, repeats: int = 10) -> float:
    timings = []
    device = _resolve_device()

    # Warmup
    for args, kwargs in inputs:
        try:
            func(*args, **kwargs)
        except Exception:
            pass
    _sync_device(device)

    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for args, kwargs in inputs:
            start.record()
            for _ in range(repeats):
                try:
                    func(*args, **kwargs)
                except Exception:
                    pass
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / repeats)
    else:
        for args, kwargs in inputs:
            start_time = time.perf_counter()
            for _ in range(repeats):
                try:
                    func(*args, **kwargs)
                except Exception:
                    pass
            _sync_device(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            timings.append(elapsed_ms / repeats)

    return float(np.mean(timings)) if timings else 0.0


def measure_kernel(kernel_dir: Path, inputs, repeats: int = 10) -> float:
    module = get_module(kernel_dir, baseline=True)
    timings = []
    device = _resolve_device()

    # Warmup
    for args, kwargs in inputs:
        try:
            module.launch(*args, **kwargs)
        except TypeError:
            module.launch(*args)
    _sync_device(device)

    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for args, kwargs in inputs:
            start.record()
            for _ in range(repeats):
                try:
                    module.launch(*args, **kwargs)
                except TypeError:
                    module.launch(*args)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / repeats)
    else:
        for args, kwargs in inputs:
            start_time = time.perf_counter()
            for _ in range(repeats):
                try:
                    module.launch(*args, **kwargs)
                except TypeError:
                    module.launch(*args)
            _sync_device(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            timings.append(elapsed_ms / repeats)

    return float(np.mean(timings)) if timings else 0.0


def _short_error(msg: str, limit: int = 240) -> str:
    if not msg:
        return ""
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark per-op kernels for a project.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--only-ops", default="")
    args = parser.parse_args()

    only_ops = set()
    if args.only_ops:
        for item in str(args.only_ops).split(","):
            norm = _normalize_op_name(item)
            if norm:
                only_ops.add(norm)

    target_device = _resolve_device()
    if target_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; benchmark requires a GPU.")
        return 1
    if target_device == "mps" and not (hasattr(torch, "backends") and torch.backends.mps.is_available()):
        print("MPS not available; benchmark requires Apple Silicon.")
        return 1

    project_dir = Path("projects") / args.project
    io_root = project_dir / "io" / "individual_ops"
    if not io_root.exists():
        print(f"No profiling data at {io_root}")
        return 1

    optimized_root = find_optimized_dir(args.project)
    generated_root = project_dir / "kernels" / "generated" / "individual_op_kernels"

    results = []
    candidates = []
    for op_dir in sorted(io_root.iterdir()):
        if not op_dir.is_dir():
            continue
        op_name = op_dir.name
        if only_ops and op_name not in only_ops:
            continue
        func = get_torch_func(op_name)
        if func is None:
            continue

        entry_files = sorted(op_dir.glob("entry_*.pt"))[: args.max_files]
        if not entry_files:
            continue

        kernel_dir = None
        source = "missing"
        kernel_status = "missing"
        if optimized_root and (optimized_root / op_name / "kernel.cu").exists():
            kernel_dir = optimized_root / op_name
            source = "optimized"
            kernel_status = "candidate"
        elif (generated_root / op_name / "kernel.cu").exists():
            kernel_dir = generated_root / op_name
            source = "generated"
            kernel_status = "candidate"

        candidates.append(
            (op_name, func, entry_files, kernel_dir, source, kernel_status)
        )

    total = len(candidates)
    update_job_progress(0, total, "Starting benchmark")

    for idx, (op_name, func, entry_files, kernel_dir, source, kernel_status) in enumerate(
        candidates, start=1
    ):
        if not wait_if_paused():
            print("Benchmark cancelled.")
            return 1
        if check_cancelled():
            print("Benchmark cancelled.")
            return 1
        update_job_progress(idx - 1, total, f"Benchmarking {op_name}")

        inputs = load_batch([str(p) for p in entry_files])
        if not inputs:
            results.append(
                {
                    "op": op_name,
                    "entries": len(entry_files),
                    "pytorch_ms": 0.0,
                    "kernel_ms": None,
                    "speedup": 0.0,
                    "winner": False,
                    "source": source,
                    "kernel_dir": os.path.relpath(kernel_dir, project_dir)
                    if kernel_dir
                    else None,
                    "kernel_status": "skipped",
                    "kernel_error": "No input entries loaded",
                    "pytorch_error": "",
                }
            )
            update_job_progress(idx, total, f"Skipped {op_name}")
            continue

        pytorch_error = ""
        try:
            pytorch_ms = measure_pytorch(func, inputs)
        except Exception as e:
            print(f"Failed PyTorch benchmark for {op_name}: {e}")
            pytorch_ms = 0.0
            pytorch_error = _short_error(str(e))

        kernel_ms = None
        kernel_error = ""
        if kernel_dir:
            try:
                kernel_ms = measure_kernel(kernel_dir, inputs)
                kernel_status = "ok"
            except Exception as e:
                print(f"Failed kernel benchmark for {op_name}: {e}")
                kernel_error = _short_error(str(e))
                kernel_status = "error"

        speedup = (pytorch_ms / kernel_ms) if (pytorch_ms and kernel_ms) else 0.0
        winner = speedup > 1.0 if kernel_status == "ok" else False

        results.append(
            {
                "op": op_name,
                "entries": len(entry_files),
                "pytorch_ms": pytorch_ms,
                "kernel_ms": kernel_ms,
                "speedup": speedup,
                "winner": winner,
                "source": source,
                "kernel_dir": os.path.relpath(kernel_dir, project_dir)
                if kernel_dir
                else None,
                "kernel_status": kernel_status,
                "kernel_error": kernel_error,
                "pytorch_error": pytorch_error,
            }
        )
        update_job_progress(idx, total, f"Finished {op_name}")

    update_job_progress(total, total, "Benchmark complete")

    out_dir = project_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "op_benchmarks.json"
    payload = {
        "project": args.project,
        "device": torch.cuda.get_device_name(0)
        if target_device == "cuda" and torch.cuda.is_available()
        else ("Apple MPS" if target_device == "mps" else "CPU"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
