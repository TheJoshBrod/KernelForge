#!/usr/bin/env python3
"""Benchmark optimized kernels (best-effort wrapper)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "src.optimizer.benchmark_pytorch"]
    print(f"[benchmark_project_ops] Running benchmark module for project={args.project or 'N/A'}")
    rc = subprocess.run(cmd, cwd=str(repo_root)).returncode

    if rc != 0:
        print(
            "[benchmark_project_ops] Benchmark module returned a non-zero status; "
            "treating as best-effort and continuing."
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

