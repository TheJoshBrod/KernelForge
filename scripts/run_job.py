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
    parser = argparse.ArgumentParser(description="Run a job and update state.json")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--cwd", default=None)
    args, cmd = parser.parse_known_args()

    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    state_path = Path(args.state_path)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    _update_state(
        state_path,
        args.job_key,
        {
            "status": "running",
            "started_at": started_at,
            "log": str(log_path),
        },
    )

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=args.cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            return_code = proc.wait()
    except Exception as exc:
        _update_state(
            state_path,
            args.job_key,
            {
                "status": "error",
                "finished_at": time.time(),
                "error": str(exc),
            },
        )
        return 1

    status = "success" if return_code == 0 else "error"
    _update_state(
        state_path,
        args.job_key,
        {
            "status": status,
            "finished_at": time.time(),
            "return_code": return_code,
        },
    )
    return return_code


if __name__ == "__main__":
    sys.exit(main())
