from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .codex_capabilities import inspect_codex_cli


def _auth_error(text: str) -> bool:
    blob = (text or "").lower()
    return any(
        token in blob
        for token in [
            "not logged in",
            "login",
            "unauthorized",
            "forbidden",
            "api key",
            "authentication",
        ]
    )


def _approval_mode_for_sandbox(sandbox: str | None) -> str:
    value = (sandbox or "workspace-write").strip().lower()
    if value in {"read-only", "readonly"}:
        return "suggest"
    if value in {"workspace-write", "auto-edit"}:
        return "auto-edit"
    if value in {"danger-full-access", "full-auto"}:
        return "full-auto"
    return "auto-edit"


def run_codex_prompt(
    *,
    work_dir: Path,
    prompt: str,
    model: str | None = None,
    sandbox: str | None = "workspace-write",
    timeout_sec: int = 300,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    caps = inspect_codex_cli()
    if not caps.binary_found:
        return {
            "exit_code": 127,
            "stdout": "",
            "stderr": caps.reason,
            "auth_error": False,
            "command_used": "",
            "runner_mode": "missing",
        }

    cmd: list[str]
    use_stdin = False
    if caps.supports_exec_subcommand:
        cmd = ["codex", "exec", "--sandbox", sandbox or "workspace-write"]
        if model:
            cmd += ["--model", model]
        cmd += ["-"]
        use_stdin = True
        runner_mode = "exec"
    else:
        cmd = ["codex", "-q", "-a", _approval_mode_for_sandbox(sandbox)]
        if model:
            cmd += ["-m", model]
        cmd += [prompt]
        runner_mode = "prompt"

    try:
        proc = subprocess.run(
            cmd,
            input=prompt if use_stdin else None,
            text=True,
            capture_output=True,
            check=False,
            cwd=str(work_dir),
            env=env,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "codex prompt timed out",
            "auth_error": False,
            "command_used": " ".join(shlex.quote(c) for c in cmd),
            "runner_mode": runner_mode,
        }

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    merged = (stdout + "\n" + stderr).strip()
    return {
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "auth_error": _auth_error(merged),
        "command_used": " ".join(shlex.quote(c) for c in cmd),
        "runner_mode": runner_mode,
    }
