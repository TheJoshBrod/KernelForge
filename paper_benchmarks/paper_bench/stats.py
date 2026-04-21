from __future__ import annotations

import math
from statistics import mean, median, pstdev

from .schema import LatencySummary


def percentile(samples: list[float], pct: float) -> float | None:
    if not samples:
        return None
    if pct <= 0:
        return min(samples)
    if pct >= 100:
        return max(samples)
    ordered = sorted(samples)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[lower]
    fraction = idx - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def build_latency_summary(samples_ms: list[float]) -> LatencySummary:
    if not samples_ms:
        return LatencySummary(count=0)
    return LatencySummary(
        count=len(samples_ms),
        mean_ms=float(mean(samples_ms)),
        median_ms=float(median(samples_ms)),
        p05_ms=float(percentile(samples_ms, 5) or 0.0),
        p95_ms=float(percentile(samples_ms, 95) or 0.0),
        min_ms=float(min(samples_ms)),
        max_ms=float(max(samples_ms)),
        stddev_ms=float(pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    )


def safe_speedup(baseline_ms: float | None, candidate_ms: float | None) -> float | None:
    if baseline_ms is None or candidate_ms is None:
        return None
    if baseline_ms <= 0 or candidate_ms <= 0:
        return None
    return float(baseline_ms / candidate_ms)
