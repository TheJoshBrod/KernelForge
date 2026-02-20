from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def _short_error(msg: str, limit: int = 2000) -> str:
    if not msg:
        return ""
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."


def format_verifier_output(
    traceback_error: str,
    kernel_code: str,
    log_file_path: Path | None,
    input_and_output: dict,
    summarizer: Callable[[str, str, dict], str] | None = None,
) -> str:
    """
    Format verifier output with raw error and optional LLM-generated feedback.
    """
    disable_summary = os.environ.get("KFORGE_DISABLE_SUMMARY", "").lower() in {
        "1",
        "true",
        "yes",
    }
    raw_error = _short_error(traceback_error)
    feedback = raw_error

    if summarizer is not None and not disable_summary:
        try:
            io_copy = input_and_output.copy() if input_and_output else {}
            if "correct-output" not in io_copy and "output" in io_copy:
                io_copy["correct-output"] = io_copy["output"]
                del io_copy["output"]
            feedback = summarizer(traceback_error, kernel_code, io_copy)
        except Exception as e:
            feedback = f"{raw_error}\n[LLM Feedback Error: {str(e)}]"

    output = (
        f"[Raw Error]:\n{raw_error}\n\n"
        f"[LLM Generated Feedback]:\n{feedback}"
    )

    if log_file_path:
        try:
            with open(log_file_path, "w") as f:
                f.write(output)
        except Exception:
            pass

    return output
