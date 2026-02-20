from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .locks import file_lock


def _atomic_write_json_unlocked(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json_unlocked(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return data


def read_json_file(path: Path, default: Any):
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path):
        return _read_json_unlocked(path, default)


def write_json_file(path: Path, payload: Any) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path):
        _atomic_write_json_unlocked(path, payload)


def update_job_state(state_path: Path, job_key: str, updates: dict[str, Any]) -> dict[str, Any]:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with file_lock(lock_path):
        state = _read_json_unlocked(state_path, {})
        if not isinstance(state, dict):
            state = {}
        job_state = dict(state.get(job_key, {}))
        job_state.update(updates)
        state[job_key] = job_state
        _atomic_write_json_unlocked(state_path, state)
        return job_state

