#!/usr/bin/env python3
"""Benchmark project operators and emit dashboard benchmark artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.benchmark_ops",
        "--project",
        args.project,
    ]
    print(f"[benchmark_project_ops] Running benchmark module for project={args.project}")
    return subprocess.run(cmd, cwd=str(repo_root)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

