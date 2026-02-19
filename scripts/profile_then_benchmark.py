#!/usr/bin/env python3
"""Run project profiling and then benchmark (best-effort benchmark step)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"[profile_then_benchmark] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    profile_script = repo_root / "benchmarks" / "profiler" / "profile_project.py"
    benchmark_script = repo_root / "scripts" / "benchmark_project_ops.py"

    if not profile_script.exists():
        print(f"[profile_then_benchmark] Missing profiler script: {profile_script}")
        return 1

    profile_cmd = [sys.executable, str(profile_script), "--project", args.project]
    profile_rc = _run(profile_cmd, repo_root)
    if profile_rc != 0:
        print(
            f"[profile_then_benchmark] Profiling failed with exit code {profile_rc}; "
            "continuing as best-effort so project creation is not blocked."
        )
        return 0

    if benchmark_script.exists():
        bench_cmd = [sys.executable, str(benchmark_script), "--project", args.project]
        bench_rc = _run(bench_cmd, repo_root)
        if bench_rc != 0:
            print(
                f"[profile_then_benchmark] Benchmark returned {bench_rc}; "
                "continuing without failing the full pipeline."
            )
    else:
        print("[profile_then_benchmark] Benchmark script not found; skipping benchmark step.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
