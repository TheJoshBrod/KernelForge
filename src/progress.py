import json
import os
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


def update_job_progress(current: int, total: int, message: str | None = None) -> None:
    state_path = os.environ.get("CGINS_STATE_PATH")
    job_key = os.environ.get("CGINS_JOB_KEY")
    if not state_path or not job_key:
        return

    try:
        state_file = Path(state_path)
        state = _load_state(state_file)
        job_state = dict(state.get(job_key, {}))
        progress = {
            "current": int(current),
            "total": int(total),
            "percent": (float(current) / float(total)) if total else 0.0,
            "updated_at": time.time(),
        }
        job_state["progress"] = progress
        if message:
            job_state["message"] = message
        state[job_key] = job_state
        _save_state(state_file, state)
    except Exception:
        return


def update_job_usage(usage: dict) -> None:
    state_path = os.environ.get("CGINS_STATE_PATH")
    job_key = os.environ.get("CGINS_JOB_KEY")
    if not state_path or not job_key:
        return

    try:
        state_file = Path(state_path)
        state = _load_state(state_file)
        job_state = dict(state.get(job_key, {}))
        job_state["usage"] = usage or {}
        state[job_key] = job_state
        _save_state(state_file, state)
    except Exception:
        return


def _get_job_state() -> dict:
    state_path = os.environ.get("CGINS_STATE_PATH")
    job_key = os.environ.get("CGINS_JOB_KEY")
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
