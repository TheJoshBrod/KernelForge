import os
import time
from pathlib import Path

from src.optimizer.benchmarking.state import (
    read_json_file,
    update_job_state as update_locked_job_state,
)


def _load_state(state_path: Path) -> dict:
    state = read_json_file(state_path, {})
    return state if isinstance(state, dict) else {}


def update_job_progress(current: int, total: int, message: str | None = None) -> None:
    state_path = os.environ.get("KFORGE_STATE_PATH")
    job_key = os.environ.get("KFORGE_JOB_KEY")
    if not state_path or not job_key:
        return

    try:
        state_file = Path(state_path)
        progress = {
            "current": int(current),
            "total": int(total),
            "percent": (float(current) / float(total)) if total else 0.0,
            "updated_at": time.time(),
        }
        updates = {"progress": progress}
        if message:
            updates["message"] = message
        update_locked_job_state(state_file, job_key, updates)
    except Exception:
        return


def update_job_usage(usage: dict) -> None:
    state_path = os.environ.get("KFORGE_STATE_PATH")
    job_key = os.environ.get("KFORGE_JOB_KEY")
    if not state_path or not job_key:
        return

    try:
        state_file = Path(state_path)
        update_locked_job_state(state_file, job_key, {"usage": usage or {}})
    except Exception:
        return


def _get_job_state() -> dict:
    state_path = os.environ.get("KFORGE_STATE_PATH")
    job_key = os.environ.get("KFORGE_JOB_KEY")
    if not state_path or not job_key:
        return {}
    try:
        state_file = Path(state_path)
        state = _load_state(state_file)
        return dict(state.get(job_key, {}))
    except Exception:
        return {}


def get_job_control() -> str:
    job_state = _get_job_state()
    return str(job_state.get("control", "")).lower()


def check_cancelled() -> bool:
    return get_job_control() == "cancelled"


def wait_if_paused(poll_seconds: float = 2.0) -> bool:
    while True:
        control = get_job_control()
        if control == "paused":
            time.sleep(poll_seconds)
            continue
        if control == "cancelled":
            return False
        return True
