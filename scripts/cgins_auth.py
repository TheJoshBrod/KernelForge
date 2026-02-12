#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.auth.codex_capabilities import detect_account_session, inspect_codex_cli
from src.auth.credentials import apply_auth_env, resolve_auth
from src.auth.keychain_store import (
    DEFAULT_OPENAI_ACCOUNT,
    DEFAULT_OPENAI_SERVICE,
    delete_openai_api_key,
    save_openai_api_key,
)
from src.config import load_config_data, save_config_data


def _status_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    status = resolve_auth(
        config=cfg,
        env=dict(os.environ),
        runtime_context={"in_container": bool(os.environ.get("CGINS_PROJECT_DIR"))},
    )
    apply_auth_env(status, os.environ)
    codex_caps = inspect_codex_cli()
    session_ok, session_reason = detect_account_session(model=status.model or None, perform_live_probe=False)
    return {
        "success": True,
        "auth": status.to_dict(),
        "codex_capabilities": {
            "binary_found": codex_caps.binary_found,
            "version": codex_caps.version,
            "supports_exec_subcommand": codex_caps.supports_exec_subcommand,
            "supports_login_subcommand": codex_caps.supports_login_subcommand,
            "supports_prompt_mode": codex_caps.supports_prompt_mode,
            "reason": codex_caps.reason,
        },
        "account_session": {
            "detected": session_ok,
            "reason": session_reason,
        },
    }


def _update_auth_cfg(cfg: dict[str, Any], *, mode: str, provider: str, model: str) -> dict[str, Any]:
    out = dict(cfg or {})
    auth = out.get("auth") if isinstance(out.get("auth"), dict) else {}
    llm = out.get("llm_info") if isinstance(out.get("llm_info"), dict) else {}

    auth["mode"] = mode
    auth["provider"] = provider
    auth["model"] = model
    auth["last_checked_at"] = int(time.time())
    out["auth"] = auth

    llm["provider"] = provider
    llm["model"] = model
    llm["apikey"] = ""
    out["llm_info"] = llm
    return out


def cmd_status(_args: argparse.Namespace) -> int:
    cfg, _ = load_config_data()
    print(json.dumps(_status_payload(cfg), indent=2))
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    cfg, _ = load_config_data()
    provider = (args.provider or "openai").strip().lower()
    model = (args.model or "").strip()
    if provider != "openai":
        print(json.dumps({"success": False, "error": "v1 auth login currently supports provider=openai only"}, indent=2))
        return 2

    if args.method == "api_key":
        api_key = (args.api_key or "").strip() or (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            print(json.dumps({"success": False, "error": "Missing API key. Pass --api-key or set OPENAI_API_KEY."}, indent=2))
            return 2
        stored = save_openai_api_key(
            api_key,
            service=DEFAULT_OPENAI_SERVICE,
            account=DEFAULT_OPENAI_ACCOUNT,
        )
        if not stored.ok:
            print(json.dumps({"success": False, "error": stored.reason}, indent=2))
            return 1

        cfg = _update_auth_cfg(cfg, mode="api_key", provider=provider, model=model)
        auth_cfg = cfg.get("auth", {})
        if isinstance(auth_cfg, dict):
            auth_cfg["api_key_saved"] = True
            cfg["auth"] = auth_cfg
        ok, reason, path = save_config_data(cfg, strip_plaintext_apikey=True)
        if not ok:
            print(json.dumps({"success": False, "error": reason}, indent=2))
            return 1
        print(
            json.dumps(
                {
                    "success": True,
                    "method": "api_key",
                    "config_path": str(path),
                    "message": "API key saved to macOS keychain.",
                },
                indent=2,
            )
        )
        return 0

    detected, detect_reason = detect_account_session(model=model or None, perform_live_probe=True)
    caps = inspect_codex_cli()
    cfg = _update_auth_cfg(cfg, mode="account_session", provider=provider, model=model)
    auth_cfg = cfg.get("auth", {})
    if isinstance(auth_cfg, dict):
        auth_cfg["account_session_detected"] = bool(detected)
        cfg["auth"] = auth_cfg
    ok, reason, path = save_config_data(cfg, strip_plaintext_apikey=True)
    if not ok:
        print(json.dumps({"success": False, "error": reason}, indent=2))
        return 1

    if detected:
        print(
            json.dumps(
                {
                    "success": True,
                    "method": "account_session",
                    "config_path": str(path),
                    "detected": True,
                    "reason": detect_reason,
                },
                indent=2,
            )
        )
        return 0

    next_step = "Run `codex -q \"Respond with the single word: ok\"` after authenticating your Codex CLI, then rerun this command."
    if caps.supports_login_subcommand:
        next_step = "Run `codex login` and complete auth, then rerun this command."
    print(
        json.dumps(
            {
                "success": False,
                "method": "account_session",
                "detected": False,
                "reason": detect_reason,
                "next_step": next_step,
                "config_path": str(path),
            },
            indent=2,
        )
    )
    return 1


def cmd_logout(args: argparse.Namespace) -> int:
    cfg, _ = load_config_data()
    method = (args.method or "all").strip().lower()
    payload: dict[str, Any] = {"success": True, "method": method}

    if method in {"api_key", "all"}:
        deleted = delete_openai_api_key(
            service=DEFAULT_OPENAI_SERVICE,
            account=DEFAULT_OPENAI_ACCOUNT,
        )
        payload["api_key_deleted"] = bool(deleted.ok)
        if not deleted.ok:
            payload["api_key_delete_reason"] = deleted.reason

    auth_cfg = cfg.get("auth") if isinstance(cfg.get("auth"), dict) else {}
    if method in {"account_session", "all"}:
        auth_cfg["account_session_detected"] = False
    if method == "all":
        auth_cfg["mode"] = "auto"
        auth_cfg["api_key_saved"] = False
    cfg["auth"] = auth_cfg

    ok, reason, path = save_config_data(cfg, strip_plaintext_apikey=True)
    payload["config_path"] = str(path)
    if not ok:
        print(json.dumps({"success": False, "error": reason}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cgins-auth", description="CGinS authentication manager")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show effective auth status")

    login = sub.add_parser("login", help="Authenticate CGinS")
    login.add_argument("--method", choices=["api_key", "account_session"], required=True)
    login.add_argument("--provider", default="openai")
    login.add_argument("--model", default="gpt-5.2")
    login.add_argument("--api-key", default="")

    logout = sub.add_parser("logout", help="Clear stored credentials or session preference")
    logout.add_argument("--method", choices=["api_key", "account_session", "all"], default="all")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args)
    if args.command == "login":
        return cmd_login(args)
    if args.command == "logout":
        return cmd_logout(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
