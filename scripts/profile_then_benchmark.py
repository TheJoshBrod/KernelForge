import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _update_state(state_path: Path, job_key: str, updates: dict) -> None:
    state = _load_state(state_path)
    job_state = dict(state.get(job_key, {}))
    job_state.update(updates)
    state[job_key] = job_state
    _save_state(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile a project then benchmark its PyTorch baseline timings."
    )
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    project_dir = repo_root / "projects" / args.project
    if not project_dir.exists():
        print(f"Project not found: {project_dir}")
        return 1

    state_path_env = os.environ.get("CGINS_STATE_PATH", "").strip()
    state_path = Path(state_path_env).resolve() if state_path_env else (project_dir / "state.json")

    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    bench_log = logs_dir / "benchmark.log"

    # Surface "baseline benchmark pending" in the UI while profiling runs.
    _update_state(
        state_path,
        "benchmark",
        {
            "status": "queued",
            "log": str(bench_log),
            "message": "Waiting for profiling to finish",
        },
    )

    # 1) Profile: writes projects/<name>/io/individual_ops/**/entry_*.pt and io/summary.json
    profile_cmd = [
        sys.executable,
        str(repo_root / "benchmarks" / "profiler" / "profile_project.py"),
        "--project",
        args.project,
    ]
    prof = subprocess.run(profile_cmd, cwd=str(repo_root))
    if prof.returncode != 0:
        print(f"Profile failed with exit code {prof.returncode}")
        _update_state(
            state_path,
            "benchmark",
            {
                "status": "error",
                "finished_at": time.time(),
                "return_code": int(prof.returncode),
                "message": "Profile failed; benchmark skipped",
            },
        )
        return int(prof.returncode)

    # 2) Benchmark: produces projects/<name>/benchmarks/op_benchmarks.json
    started_at = time.time()
    _update_state(
        state_path,
        "benchmark",
        {
            "status": "running",
            "started_at": started_at,
            "log": str(bench_log),
            "message": "Benchmarking PyTorch baseline",
        },
    )

    env = os.environ.copy()
    env["CGINS_STATE_PATH"] = str(state_path)
    env["CGINS_JOB_KEY"] = "benchmark"
    project_tmp = project_dir / ".tmp"
    try:
        project_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(project_tmp)
        env["TMP"] = str(project_tmp)
        env["TEMP"] = str(project_tmp)
    except Exception:
        pass

    bench_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "benchmark_project_ops.py"),
        "--project",
        args.project,
    ]
    with bench_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            bench_cmd,
            cwd=str(repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return_code = proc.wait()

    finished_at = time.time()
    status = "success" if return_code == 0 else "error"
    final_message = "Benchmark complete" if return_code == 0 else "Benchmark failed (see logs)"
    _update_state(
        state_path,
        "benchmark",
        {
            "status": status,
            "finished_at": finished_at,
            "return_code": int(return_code),
            "message": final_message,
        },
    )
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
