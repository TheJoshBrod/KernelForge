from __future__ import annotations

import subprocess
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.auth import codex_capabilities as cc
from src.auth import codex_runner as cr
from src.auth import credentials as creds
from src.auth import keychain_store as ks


def _proc(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def test_keychain_store_load_delete_macos(monkeypatch) -> None:
    monkeypatch.setattr(ks.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ks.shutil, "which", lambda _name: "/usr/bin/security")

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output=True, text=True, check=False):  # noqa: ANN001
        calls.append(list(cmd))
        if "add-generic-password" in cmd:
            return _proc(0)
        if "find-generic-password" in cmd:
            return _proc(0, stdout="sk-test-123\n")
        if "delete-generic-password" in cmd:
            return _proc(1, stderr="item not found")
        return _proc(1, stderr="unexpected")

    monkeypatch.setattr(ks.subprocess, "run", fake_run)

    save_res = ks.save_openai_api_key("sk-test-123")
    assert save_res.ok

    load_res = ks.load_openai_api_key()
    assert load_res.ok
    assert load_res.value == "sk-test-123"

    del_res = ks.delete_openai_api_key()
    assert del_res.ok
    assert del_res.reason == "already_missing"

    assert any("add-generic-password" in c for c in calls)
    assert any("find-generic-password" in c for c in calls)
    assert any("delete-generic-password" in c for c in calls)


def test_resolve_auth_auto_prefers_account_session(monkeypatch) -> None:
    monkeypatch.setattr(creds, "detect_account_session", lambda **_kwargs: (True, "ok"))
    monkeypatch.setattr(creds, "env_or_keychain_openai_key", lambda: "")

    status = creds.resolve_auth(
        config={"auth": {"mode": "auto", "provider": "openai", "model": "gpt-5.2"}},
        env={},
        runtime_context={},
    )
    assert status.mode_effective == "account_session"
    assert status.reason == "auto_prefers_account_session"


def test_resolve_auth_auto_falls_back_to_api_key(monkeypatch) -> None:
    monkeypatch.setattr(creds, "detect_account_session", lambda **_kwargs: (False, "no_session"))
    monkeypatch.setattr(creds, "env_or_keychain_openai_key", lambda: "sk-fallback")

    status = creds.resolve_auth(
        config={"auth": {"mode": "auto", "provider": "openai", "model": "gpt-5.2"}},
        env={},
        runtime_context={},
    )
    assert status.mode_effective == "api_key"
    assert status.reason == "auto_fallback_to_api_key"


def test_resolve_auth_account_mode_no_session_no_key(monkeypatch) -> None:
    monkeypatch.setattr(creds, "detect_account_session", lambda **_kwargs: (False, "no_session"))
    monkeypatch.setattr(creds, "env_or_keychain_openai_key", lambda: "")

    status = creds.resolve_auth(
        config={"auth": {"mode": "account_session", "provider": "openai", "model": "gpt-5.2"}},
        env={},
        runtime_context={},
    )
    assert status.mode_effective == "unconfigured"
    assert status.reason == "account_session_unavailable_and_no_key_fallback"


def test_resolve_auth_container_account_mode_falls_back_key(monkeypatch) -> None:
    monkeypatch.setattr(creds, "detect_account_session", lambda **_kwargs: (True, "session"))
    monkeypatch.setattr(creds, "env_or_keychain_openai_key", lambda: "sk-container")

    status = creds.resolve_auth(
        config={"auth": {"mode": "account_session", "provider": "openai", "model": "gpt-5.2"}},
        env={},
        runtime_context={"in_container": True},
    )
    assert status.mode_effective == "api_key"
    assert status.reason == "account_session_fallback_to_api_key"


def test_inspect_codex_cli_help_parser(monkeypatch) -> None:
    monkeypatch.setattr(cc.shutil, "which", lambda _name: "/usr/bin/codex")

    def fake_run(cmd, *, cwd=None, timeout=10):  # noqa: ANN001, ARG001
        if "--help" in cmd:
            return _proc(
                0,
                stdout="Usage: codex [OPTIONS] prompt\nCommands:\n  exec\n  login\n  prompt\n",
            )
        if "--version" in cmd:
            return _proc(0, stdout="codex 1.2.3")
        return _proc(1, stderr="bad cmd")

    monkeypatch.setattr(cc, "_run", fake_run)

    caps = cc.inspect_codex_cli()
    assert caps.binary_found
    assert caps.supports_exec_subcommand
    assert caps.supports_login_subcommand
    assert caps.supports_prompt_mode
    assert caps.version == "codex 1.2.3"


def test_codex_runner_selects_exec_when_available(monkeypatch, tmp_path: Path) -> None:
    caps = cc.CodexCapabilities(
        binary_found=True,
        version="codex 1.2.3",
        supports_exec_subcommand=True,
        supports_login_subcommand=False,
        supports_prompt_mode=True,
        help_text="",
        reason="ok",
    )
    monkeypatch.setattr(cr, "inspect_codex_cli", lambda: caps)

    seen: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        seen["cmd"] = list(cmd)
        return _proc(0, stdout="ok")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)

    result = cr.run_codex_prompt(
        work_dir=tmp_path,
        prompt="Say ok",
        model="gpt-5.2",
        sandbox="workspace-write",
        timeout_sec=5,
    )
    assert result["exit_code"] == 0
    assert result["runner_mode"] == "exec"
    assert seen["cmd"][:3] == ["codex", "exec", "--sandbox"]


def test_codex_runner_prompt_mode_detects_auth_error(monkeypatch, tmp_path: Path) -> None:
    caps = cc.CodexCapabilities(
        binary_found=True,
        version="codex 1.2.3",
        supports_exec_subcommand=False,
        supports_login_subcommand=False,
        supports_prompt_mode=True,
        help_text="",
        reason="ok",
    )
    monkeypatch.setattr(cr, "inspect_codex_cli", lambda: caps)

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ARG001
        return _proc(1, stderr="not logged in")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)

    result = cr.run_codex_prompt(
        work_dir=tmp_path,
        prompt="Say ok",
        model="gpt-5.2",
        sandbox="workspace-write",
        timeout_sec=5,
    )
    assert result["exit_code"] == 1
    assert result["runner_mode"] == "prompt"
    assert result["auth_error"] is True
    assert "codex -q" in result["command_used"]
