#!/usr/bin/env python3
"""
Run codex exec in a workspace and capture transcript, diff, and metrics.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Tuple
import difflib

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.auth.codex_runner import run_codex_prompt


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    return re.sub(r"sk-[A-Za-z0-9]{8,}", "sk-REDACTED", text)


def _last_int(pattern: str, text: str) -> int | None:
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except Exception:
        return None


def _extract_usage(text: str) -> dict:
    if not text:
        return {}

    usage: dict = {}

    input_tokens = _last_int(r"input[_ ]tokens[^0-9]*(\d+)", text)
    if input_tokens is None:
        input_tokens = _last_int(r"prompt[_ ]tokens[^0-9]*(\d+)", text)
    output_tokens = _last_int(r"output[_ ]tokens[^0-9]*(\d+)", text)
    if output_tokens is None:
        output_tokens = _last_int(r"completion[_ ]tokens[^0-9]*(\d+)", text)
    reasoning_tokens = _last_int(r"reasoning[_ ]tokens[^0-9]*(\d+)", text)

    cached_tokens = _last_int(r"cached[_ ]tokens[^0-9]*(\d+)", text)
    if cached_tokens is None:
        cache_read = _last_int(r"cache[_ ]read[_ ]tokens[^0-9]*(\d+)", text)
        cache_write = _last_int(r"cache[_ ]write[_ ]tokens[^0-9]*(\d+)", text)
        cached_input = _last_int(r"cached[_ ]input[_ ]tokens[^0-9]*(\d+)", text)
        cached_tokens = (cache_read or 0) + (cache_write or 0) + (cached_input or 0)
        if cached_tokens == 0:
            cached_tokens = None

    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if reasoning_tokens is not None:
        usage["reasoning_tokens"] = reasoning_tokens
    if cached_tokens is not None:
        usage["cached_tokens"] = cached_tokens

    return usage


def _next_attempt(attempts_dir: Path) -> int:
    existing = sorted(attempts_dir.glob("codex-*.transcript.log"))
    numbers = []
    for path in existing:
        match = re.search(r"codex-(\d+)\.transcript\.log", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return (max(numbers) + 1) if numbers else 1


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_diff(before: str, after: str, diff_path: Path) -> Tuple[int, int]:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="kernel.cu.before",
            tofile="kernel.cu.after",
        )
    )
    diff_path.write_text("".join(diff), encoding="utf-8")
    return len(before_lines), len(after_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run codex exec with logging")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    attempts_dir = work_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    kernel_path = work_dir / "kernel.cu"
    if not kernel_path.exists():
        raise SystemExit("kernel.cu not found in work dir")

    prompt = args.prompt
    if args.prompt_file:
        prompt = _read_text(Path(args.prompt_file))
    if not prompt:
        raise SystemExit("Provide --prompt or --prompt-file")

    attempt = args.attempt or _next_attempt(attempts_dir)
    transcript_path = attempts_dir / f"codex-{attempt}.transcript.log"
    diff_path = attempts_dir / f"codex-{attempt}.diff.patch"
    metrics_path = attempts_dir / f"codex-{attempt}.metrics.json"
    prompt_path = attempts_dir / f"codex-{attempt}.prompt.txt"
    before_path = attempts_dir / f"codex-{attempt}.kernel.before.cu"
    after_path = attempts_dir / f"codex-{attempt}.kernel.after.cu"

    before_code = _read_text(kernel_path)
    before_path.write_text(before_code, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    started = time.time()
    run_result = run_codex_prompt(
        work_dir=work_dir,
        prompt=prompt,
        model=args.model,
        sandbox=args.sandbox,
        timeout_sec=900,
    )
    transcript_raw = (run_result.get("stdout", "") or "") + ("\n" + (run_result.get("stderr", "") or ""))
    proc_rc = int(run_result.get("exit_code", 1))
    finished = time.time()

    transcript = _redact_secrets(transcript_raw)
    transcript_path.write_text(transcript, encoding="utf-8")

    after_code = _read_text(kernel_path)
    after_path.write_text(after_code, encoding="utf-8")

    before_lines, after_lines = _write_diff(before_code, after_code, diff_path)
    usage = _extract_usage(transcript_raw)

    metrics = {
        "attempt": attempt,
        "exit_code": proc_rc,
        "started_at": started,
        "finished_at": finished,
        "duration_sec": round(finished - started, 4),
        "model": args.model,
        "sandbox": args.sandbox,
        "kernel_before_lines": before_lines,
        "kernel_after_lines": after_lines,
        "kernel_changed": before_code != after_code,
        "transcript_path": str(transcript_path),
        "diff_path": str(diff_path),
        "usage": usage,
        "auth_error": bool(run_result.get("auth_error", False)),
        "command_used": str(run_result.get("command_used", "")),
        "runner_mode": str(run_result.get("runner_mode", "")),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return proc_rc


if __name__ == "__main__":
    raise SystemExit(main())
