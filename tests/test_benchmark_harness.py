from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.benchmarking.harness import benchmark_entry_calls


def test_benchmark_entry_calls_collects_entry_latencies_on_cpu():
    def make_call(offset: int):
        def _invoke():
            total = 0
            for value in range(200):
                total += value + offset
            return total

        return _invoke

    summary = benchmark_entry_calls(
        [
            ("entry_000001.pt", make_call(1)),
            ("entry_000002.pt", make_call(2)),
        ],
        device="cpu",
        warmup_runs=1,
        timed_runs=3,
    )

    assert summary["entry_count"] == 2
    assert summary["entry_files"] == ["entry_000001.pt", "entry_000002.pt"]
    assert len(summary["entry_latencies_ms"]) == 2
    assert all(latency >= 0.0 for latency in summary["entry_latencies_ms"])
    assert summary["mean_time_ms"] is not None
    assert summary["warmup_runs"] == 1
    assert summary["timed_runs"] == 3


def test_benchmark_entry_calls_records_failed_entries():
    def _good():
        return 1

    def _bad():
        raise RuntimeError("boom")

    summary = benchmark_entry_calls(
        [
            ("entry_good.pt", _good),
            ("entry_bad.pt", _bad),
        ],
        device="cpu",
        warmup_runs=1,
        timed_runs=2,
    )

    assert summary["entry_files"] == ["entry_good.pt"]
    assert summary["entry_count"] == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["entry_file"] == "entry_bad.pt"
