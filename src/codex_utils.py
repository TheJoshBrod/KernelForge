from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.auth.codex_runner import run_codex_prompt


def op_enabled(global_flag_env: str, list_env: str, op_key: str) -> bool:
    global_flag = (os.environ.get(global_flag_env, "") or "").strip().lower()
    if global_flag in {"1", "true", "yes", "on"}:
        return True

    allowed = (os.environ.get(list_env, "") or "").strip()
    if not allowed:
        return False
    normalized = {item.strip() for item in allowed.split(",") if item.strip()}
    return op_key in normalized


def prepare_work_dir(work_dir: Path, kernel_code: str, *, task: str | None = None) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "kernel.cu").write_text(kernel_code or "", encoding="utf-8")
    if task:
        (work_dir / "TASK.md").write_text(task, encoding="utf-8")
    attempts = work_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)


def run_codex(work_dir: Path, prompt: str, *, model: str | None = None, sandbox: str | None = None) -> tuple[bool, str]:
    result: dict[str, Any] = run_codex_prompt(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        sandbox=sandbox or "workspace-write",
        timeout_sec=600,
        env=os.environ.copy(),
    )
    if int(result.get("exit_code", 1)) == 0:
        return True, ""
    if result.get("auth_error"):
        return False, "Codex authentication failed. Configure CGinS auth (api_key or account_session)."
    stderr = str(result.get("stderr", "")).strip()
    stdout = str(result.get("stdout", "")).strip()
    return False, stderr or stdout or "Codex execution failed"
