from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

DEFAULT_OPENAI_SERVICE = "cgins.openai"
DEFAULT_OPENAI_ACCOUNT = "default"


@dataclass(slots=True)
class KeychainResult:
    ok: bool
    value: str = ""
    reason: str = ""


def _supported() -> tuple[bool, str]:
    if platform.system().lower() != "darwin":
        return False, "macOS keychain is only supported on Darwin"
    if shutil.which("security") is None:
        return False, "security CLI not found on PATH"
    return True, ""


def save_openai_api_key(
    api_key: str,
    *,
    service: str = DEFAULT_OPENAI_SERVICE,
    account: str = DEFAULT_OPENAI_ACCOUNT,
) -> KeychainResult:
    supported, reason = _supported()
    if not supported:
        return KeychainResult(ok=False, reason=reason)

    value = (api_key or "").strip()
    if not value:
        return KeychainResult(ok=False, reason="API key is empty")

    cmd = [
        "security",
        "add-generic-password",
        "-U",
        "-s",
        service,
        "-a",
        account,
        "-w",
        value,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        return KeychainResult(ok=False, reason=stderr or "failed to store key in keychain")
    return KeychainResult(ok=True, reason="stored")


def load_openai_api_key(
    *,
    service: str = DEFAULT_OPENAI_SERVICE,
    account: str = DEFAULT_OPENAI_ACCOUNT,
) -> KeychainResult:
    supported, reason = _supported()
    if not supported:
        return KeychainResult(ok=False, reason=reason)

    cmd = [
        "security",
        "find-generic-password",
        "-s",
        service,
        "-a",
        account,
        "-w",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").lower()
        if "could not be found" in stderr or "item not found" in stderr:
            return KeychainResult(ok=False, reason="not_found")
        return KeychainResult(ok=False, reason=(proc.stderr or "").strip() or "failed to read keychain")
    return KeychainResult(ok=True, value=(proc.stdout or "").strip(), reason="loaded")


def delete_openai_api_key(
    *,
    service: str = DEFAULT_OPENAI_SERVICE,
    account: str = DEFAULT_OPENAI_ACCOUNT,
) -> KeychainResult:
    supported, reason = _supported()
    if not supported:
        return KeychainResult(ok=False, reason=reason)

    cmd = [
        "security",
        "delete-generic-password",
        "-s",
        service,
        "-a",
        account,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").lower()
        if "could not be found" in stderr or "item not found" in stderr:
            return KeychainResult(ok=True, reason="already_missing")
        return KeychainResult(ok=False, reason=(proc.stderr or "").strip() or "failed to delete keychain item")
    return KeychainResult(ok=True, reason="deleted")


def env_or_keychain_openai_key() -> str:
    env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key
    loaded = load_openai_api_key()
    if loaded.ok and loaded.value:
        return loaded.value
    return ""
