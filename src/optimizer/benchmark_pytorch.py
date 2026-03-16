"""Compatibility entrypoint that forwards to the unified benchmark module."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from src.projects.paths import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("KFORGE_PROJECT_NAME", ""))
    args = parser.parse_args()

    if not args.project:
        print("[benchmark_pytorch] --project is required")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.benchmark_ops",
        "--project",
        args.project,
    ]
    return subprocess.run(cmd, cwd=str(repo_root())).returncode


if __name__ == "__main__":
    raise SystemExit(main())
