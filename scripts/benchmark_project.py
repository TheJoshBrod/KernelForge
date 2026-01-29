import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_optimized_dir(project_name: str) -> Path | None:
    base_dir = Path("kernels/optimized")
    if not base_dir.exists():
        return None
    suffix = f"_{project_name}"
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and benchmark project kernels.")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    optimized_dir = _find_optimized_dir(args.project)
    if optimized_dir is None:
        print("No optimized kernels found for project.")
        return 1

    compile_cmd = [
        sys.executable,
        "benchmarks/runner/collect_kernels.py",
        "--optimized",
        "--source",
        str(optimized_dir),
    ]
    subprocess.check_call(compile_cmd)

    bench_cmd = [sys.executable, "benchmarks/runner/runner.py", "--optimized"]
    subprocess.check_call(bench_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
