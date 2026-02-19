#!/usr/bin/env python3
"""Lightweight async job runner used by frontend walkers.

This module executes a command, streams stdout/stderr to a log file,
and updates project state.json for job lifecycle tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.optimizer.benchmarking.state import (
        read_json_file as _locked_read_json_file,
        update_job_state as _locked_update_job_state,
    )
except Exception:
    _locked_read_json_file = None
    _locked_update_job_state = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_state(state_path: Path) -> dict[str, Any]:
    if _locked_read_json_file is not None:
        data = _locked_read_json_file(state_path, {})
        return data if isinstance(data, dict) else {}
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_path)


def _update_state(state_path: Path, job_key: str, updates: dict[str, Any]) -> dict[str, Any]:
    if _locked_update_job_state is not None:
        return _locked_update_job_state(state_path, job_key, updates)
    state = _read_state(state_path)
    job_state = dict(state.get(job_key, {}))
    job_state.update(updates)
    state[job_key] = job_state
    _write_state(state_path, state)
    return job_state


def _current_control(state_path: Path, job_key: str) -> str:
    state = _read_state(state_path)
    job_state = state.get(job_key, {})
    return str(job_state.get("control", "running")).lower().strip()


def _append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _normalize_command(raw_cmd: list[str]) -> list[str]:
    cmd = list(raw_cmd)
    if cmd and cmd[0] == "--":
        return cmd[1:]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a job and track state/log output.")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--use-container", action="store_true")
    parser.add_argument("--container-image", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    state_path = Path(args.state_path).resolve()
    log_path = Path(args.log_path).resolve()
    cwd_path = Path(args.cwd).resolve()
    cmd = _normalize_command(args.command)

    if not cmd:
        _append_log(log_path, "[run_job] No command provided.")
        _update_state(
            state_path,
            args.job_key,
            {
                "status": "error",
                "control": "running",
                "message": "No command provided",
                "finished_at": _now_iso(),
            },
        )
        return 1

    if args.use_container:
        _append_log(
            log_path,
            (
                "[run_job] Container mode requested"
                + (f" ({args.container_image})" if args.container_image else "")
                + " - executing directly in host environment."
            ),
        )

    env = dict(os.environ)
    env["CGINS_STATE_PATH"] = str(state_path)
    env["CGINS_JOB_KEY"] = args.job_key
    env["PYTHONUNBUFFERED"] = "1"

    _append_log(log_path, f"[run_job] Started at {_now_iso()}")
    _append_log(log_path, f"[run_job] CWD: {cwd_path}")
    _append_log(log_path, f"[run_job] Command: {cmd}")
    _update_state(
        state_path,
        args.job_key,
        {
            "status": "running",
            "control": "running",
            "started_at": _now_iso(),
            "log": str(log_path),
            "command": cmd,
        },
    )

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd_path),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    _update_state(state_path, args.job_key, {"pid": proc.pid})

    paused = False
    cancelled = False
    while True:
        ret = proc.poll()
        if ret is not None:
            break

        control = _current_control(state_path, args.job_key)
        if control == "cancelled":
            cancelled = True
            _append_log(log_path, "[run_job] Cancellation requested.")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            break

        if control == "paused" and not paused:
            try:
                os.kill(proc.pid, signal.SIGSTOP)
                paused = True
                _update_state(state_path, args.job_key, {"status": "paused"})
                _append_log(log_path, "[run_job] Paused.")
            except Exception:
                pass
        elif control != "paused" and paused:
            try:
                os.kill(proc.pid, signal.SIGCONT)
                paused = False
                _update_state(state_path, args.job_key, {"status": "running"})
                _append_log(log_path, "[run_job] Resumed.")
            except Exception:
                pass

        time.sleep(0.5)

    returncode = proc.poll()
    if cancelled:
        _update_state(
            state_path,
            args.job_key,
            {
                "status": "cancelled",
                "message": "Job cancelled",
                "finished_at": _now_iso(),
            },
        )
        _append_log(log_path, f"[run_job] Cancelled at {_now_iso()}")
        return 0

    if returncode == 0:
        _update_state(
            state_path,
            args.job_key,
            {
                "status": "completed",
                "message": "Job completed",
                "finished_at": _now_iso(),
            },
        )
        _append_log(log_path, f"[run_job] Completed at {_now_iso()}")
        return 0

    _update_state(
        state_path,
        args.job_key,
        {
            "status": "error",
            "message": f"Job failed with exit code {returncode}",
            "finished_at": _now_iso(),
        },
    )
    _append_log(log_path, f"[run_job] Failed at {_now_iso()} with exit code {returncode}")
    return int(returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())

