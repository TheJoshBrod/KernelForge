from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

AUTH_ERROR_PATTERNS = [
    r"not\s+logged\s+in",
    r"login",
    r"unauthorized",
    r"forbidden",
    r"api\s*key",
    r"authentication",
]

SESSION_HINT_FILES = {
    "auth.json",
    "credentials.json",
    "session.json",
    "token.json",
    "tokens.json",
}


@dataclass(slots=True)
class CodexCapabilities:
    binary_found: bool
    version: str
    supports_exec_subcommand: bool
    supports_login_subcommand: bool
    supports_prompt_mode: bool
    help_text: str
    reason: str = ""


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _has_keyword(text: str, keyword: str) -> bool:
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text or "", flags=re.IGNORECASE))


def inspect_codex_cli() -> CodexCapabilities:
    if shutil.which("codex") is None:
        return CodexCapabilities(
            binary_found=False,
            version="",
            supports_exec_subcommand=False,
            supports_login_subcommand=False,
            supports_prompt_mode=False,
            help_text="",
            reason="codex binary not found on PATH",
        )

    help_proc = _run(["codex", "--help"])
    help_text = ((help_proc.stdout or "") + "\n" + (help_proc.stderr or "")).strip()
    version_proc = _run(["codex", "--version"])
    version = (version_proc.stdout or version_proc.stderr or "").strip()

    supports_prompt_mode = _has_keyword(help_text, "Usage") and _has_keyword(help_text, "prompt")
    supports_exec = _has_keyword(help_text, "exec")
    supports_login = _has_keyword(help_text, "login")

    return CodexCapabilities(
        binary_found=True,
        version=version,
        supports_exec_subcommand=supports_exec,
        supports_login_subcommand=supports_login,
        supports_prompt_mode=supports_prompt_mode,
        help_text=help_text,
        reason="ok",
    )


def _looks_like_auth_error(text: str) -> bool:
    blob = (text or "").lower()
    return any(re.search(pattern, blob) for pattern in AUTH_ERROR_PATTERNS)


def _candidate_session_dirs() -> list[Path]:
    paths: list[Path] = []
    codex_home = (os.environ.get("CODEX_HOME") or "").strip()
    if codex_home:
        paths.append(Path(codex_home).expanduser())
    paths.extend(
        [
            Path.home() / ".codex",
            Path.home() / ".config" / "codex",
            Path.home() / ".local" / "share" / "codex",
        ]
    )
    uniq: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def _detect_session_files() -> tuple[bool, str]:
    for root in _candidate_session_dirs():
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            name = item.name.lower()
            if name in SESSION_HINT_FILES or "auth" in name or "token" in name:
                return True, f"session_file:{item}"
    return False, "no_session_files"


def detect_account_session(*, model: str | None = None, perform_live_probe: bool = True) -> tuple[bool, str]:
    caps = inspect_codex_cli()
    if not caps.binary_found:
        return False, caps.reason

    session_file_found, file_reason = _detect_session_files()
    if not perform_live_probe:
        return session_file_found, file_reason

    if not caps.supports_prompt_mode and not caps.supports_exec_subcommand:
        return session_file_found, "codex_has_no_supported_prompt_mode"

    cmd = ["codex", "-q"]
    if model:
        cmd += ["-m", model]
    cmd += ["Respond with the single word: ok"]
    proc = _run(cmd, timeout=20)
    merged = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()

    if proc.returncode == 0 and not _looks_like_auth_error(merged):
        return True, "live_probe_ok"
    if _looks_like_auth_error(merged):
        return False, "live_probe_auth_error"
    if session_file_found:
        return True, f"session_file_hint_with_probe_rc_{proc.returncode}"
    return False, f"live_probe_failed_rc_{proc.returncode}"
