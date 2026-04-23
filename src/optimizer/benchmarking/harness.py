from __future__ import annotations

import time
from statistics import fmean, median, pstdev
from typing import Any, Callable, Sequence

import torch

DEFAULT_WARMUP_RUNS = 25
DEFAULT_TIMED_RUNS = 100


def sync_device(device: str) -> None:
    target = (device or "").strip().lower()
    if target == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif target == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def summarize_entry_results(
    entry_results: Sequence[dict[str, Any]],
    *,
    errors: Sequence[dict[str, str]] | None = None,
    device: str,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    timed_runs: int = DEFAULT_TIMED_RUNS,
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in entry_results if "latency_ms" in row]
    summary: dict[str, Any] = {
        "device": device,
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "entry_count": len(entry_results),
        "entry_files": [str(row["entry_file"]) for row in entry_results if "entry_file" in row],
        "entry_latencies_ms": latencies,
        "entry_results": list(entry_results),
        "errors": list(errors or []),
        "median_time_ms": None,
        "mean_time_ms": None,
        "std_time_ms": None,
        "min_time_ms": None,
        "max_time_ms": None,
    }
    if latencies:
        summary["median_time_ms"] = float(median(latencies))
        summary["mean_time_ms"] = float(fmean(latencies))
        summary["std_time_ms"] = float(pstdev(latencies)) if len(latencies) > 1 else 0.0
        summary["min_time_ms"] = float(min(latencies))
        summary["max_time_ms"] = float(max(latencies))
    return summary


def benchmark_entry_calls(
    entry_calls: Sequence[tuple[str, Callable[[], Any]]],
    *,
    device: str,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    timed_runs: int = DEFAULT_TIMED_RUNS,
) -> dict[str, Any]:
    if timed_runs <= 0:
        raise ValueError("timed_runs must be > 0")

    target = (device or "cpu").strip().lower()
    use_cuda_events = target == "cuda" and torch.cuda.is_available()

    entry_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for entry_file, invoke in entry_calls:
        try:
            for _ in range(int(warmup_runs)):
                invoke()
            sync_device(target)

            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(int(timed_runs)):
                    invoke()
                end.record()
                sync_device(target)
                latency_ms = float(start.elapsed_time(end)) / float(timed_runs)
            else:
                start_time = time.perf_counter()
                for _ in range(int(timed_runs)):
                    invoke()
                sync_device(target)
                latency_ms = ((time.perf_counter() - start_time) * 1000.0) / float(
                    timed_runs
                )

            entry_results.append(
                {
                    "entry_file": str(entry_file),
                    "latency_ms": float(latency_ms),
                }
            )
        except Exception as exc:
            errors.append({"entry_file": str(entry_file), "error": str(exc)})

    return summarize_entry_results(
        entry_results,
        errors=errors,
        device=target,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
