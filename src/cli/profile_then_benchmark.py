#!/usr/bin/env python3
"""Run project profiling and then benchmark in one tracked pipeline."""

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
    parser.add_argument("--weights-b64-path", default="")
    parser.add_argument("--validation-b64-path", default="")
    parser.add_argument("--validation-name-path", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.pipeline",
        "--project",
        args.project,
    ]
    if args.weights_b64_path:
        cmd += ["--weights-b64-path", args.weights_b64_path]
    if args.validation_b64_path:
        cmd += ["--validation-b64-path", args.validation_b64_path]
    if args.validation_name_path:
        cmd += ["--validation-name-path", args.validation_name_path]
    return _run(cmd, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())

