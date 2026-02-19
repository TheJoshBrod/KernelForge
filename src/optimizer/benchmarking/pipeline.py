from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .assets import prepare_uploaded_assets
from .paths import project_dir_for_name, repo_root
from .state import update_job_state


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"[benchmarking.pipeline] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--weights-b64-path", default="")
    parser.add_argument("--validation-b64-path", default="")
    parser.add_argument("--validation-name-path", default="")
    args = parser.parse_args()

    root = repo_root()
    project_dir = project_dir_for_name(args.project, create=True)
    state_path = project_dir / "state.json"

    try:
        update_job_state(
            state_path,
            "profile",
            {
                "status": "running",
                "control": "running",
                "active": True,
                "phase": "preparing",
                "progress_percent": 0.05,
                "message": "Preparing assets",
            },
        )
        prepare_uploaded_assets(
            project_dir,
            weights_b64_path=args.weights_b64_path,
            validation_b64_path=args.validation_b64_path,
            validation_name_path=args.validation_name_path,
        )

        update_job_state(
            state_path,
            "profile",
            {
                "status": "running",
                "control": "running",
                "active": True,
                "phase": "profiling",
                "progress_percent": 0.35,
                "message": "Profiling operators",
            },
        )
        profile_cmd = [
            sys.executable,
            "-m",
            "src.optimizer.benchmarking.profile_project",
            "--project",
            args.project,
        ]
        profile_rc = _run(profile_cmd, root)
        if profile_rc != 0:
            update_job_state(
                state_path,
                "profile",
                {
                    "status": "error",
                    "control": "idle",
                    "active": False,
                    "phase": "error",
                    "progress_percent": 1.0,
                    "message": f"Profiling failed with exit code {profile_rc}",
                },
            )
            print(f"[benchmarking.pipeline] Profiling failed ({profile_rc}).")
            return profile_rc

        update_job_state(
            state_path,
            "profile",
            {
                "status": "running",
                "control": "running",
                "active": True,
                "phase": "benchmarking",
                "progress_percent": 0.75,
                "message": "Benchmarking operators",
            },
        )
        bench_cmd = [
            sys.executable,
            "-m",
            "src.optimizer.benchmarking.benchmark_ops",
            "--project",
            args.project,
        ]
        bench_rc = _run(bench_cmd, root)
        if bench_rc != 0:
            update_job_state(
                state_path,
                "profile",
                {
                    "status": "error",
                    "control": "idle",
                    "active": False,
                    "phase": "error",
                    "progress_percent": 1.0,
                    "message": f"Benchmark failed with exit code {bench_rc}",
                },
            )
            print(f"[benchmarking.pipeline] Benchmark failed ({bench_rc}).")
            return bench_rc
        update_job_state(
            state_path,
            "profile",
            {
                "status": "running",
                "control": "running",
                "active": True,
                "phase": "finalizing",
                "progress_percent": 0.95,
                "message": "Finalizing results",
            },
        )
    except Exception as e:
        update_job_state(
            state_path,
            "profile",
            {
                "status": "error",
                "control": "idle",
                "active": False,
                "phase": "error",
                "progress_percent": 1.0,
                "message": f"Pipeline error: {e}",
            },
        )
        print(f"[benchmarking.pipeline] Pipeline error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
