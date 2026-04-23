"""Conversation lineage tracker.

Persists one `history.json` per node/op capturing the full chain of
(prompt -> code -> error -> fix-prompt -> ...) across correction retries.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


_COMPILE_MARKER = "[Compilation Failed]"
_MISMATCH_MARKER = "[Output Mismatch"
_RUNTIME_MARKER = "[Runtime Error"


def classify_error(error_details: Optional[str], stage_hint: Optional[str] = None) -> Optional[str]:
    """Classify an error string into a coarse category.

    Uses the same markers already emitted by
    `src/optimizer/backends/cuda/verifier.py`. `stage_hint` comes from
    the generator's `last_stage` (e.g. "llm_api", "setup").
    """
    if stage_hint:
        if stage_hint == "llm_api":
            return "LLMApiError"
        if stage_hint == "setup":
            return "SetupError"

    if not error_details:
        return None

    text = str(error_details)
    if _COMPILE_MARKER in text:
        return "CompilationFailed"
    if _MISMATCH_MARKER in text:
        return "NumericalMismatch"
    if _RUNTIME_MARKER in text:
        return "RuntimeError"
    if "Failed to extract code" in text:
        return "ExtractFailed"
    return "Unknown"


def _truncate_middle(text: Optional[str], cap: int = 8192) -> Optional[str]:
    """Truncate the *middle* of a long string, preserving head and tail.

    Compiler dumps are tens of KB of repeated template-instantiation noise;
    the useful signal is at the top (error marker) and the bottom (final
    line number / root cause).
    """
    if text is None:
        return None
    text = str(text)
    if len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head
    dropped = len(text) - cap
    return f"{text[:head]}\n...[truncated {dropped} bytes]...\n{text[-tail:]}"


def _sha1(s: str) -> str:
    return "sha1:" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


class LineageRecorder:
    """Accumulates a correction chain and writes `history.json` atomically."""

    def __init__(
        self,
        out_dir: Path,
        phase: str,
        *,
        node_id: Optional[int] = None,
        op_name: Optional[str] = None,
        system_prompt: str = "",
        filename: Optional[str] = None,
        error_cap_bytes: int = 8192,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.phase = phase
        self.node_id = node_id
        self.op_name = op_name
        self.system_prompt = system_prompt or ""
        self.system_prompt_digest = _sha1(self.system_prompt) if self.system_prompt else None
        self.error_cap_bytes = error_cap_bytes
        self.chain: list[dict[str, Any]] = []
        self._finalized = False

        if filename is None:
            if node_id is not None:
                filename = f"history_{node_id}.json"
            else:
                filename = "history.json"
        self.filename = filename

    def record_attempt(
        self,
        *,
        iteration: int,
        role: str,
        prompt: str,
        llm_response_code: Optional[str],
        is_valid: bool,
        error_details: Optional[str],
        error_summary: Optional[str] = None,
        stage_hint: Optional[str] = None,
    ) -> None:
        if role not in ("initial", "fix"):
            raise ValueError(f"role must be 'initial' or 'fix', got {role!r}")
        entry = {
            "iteration": iteration,
            "role": role,
            "prompt": prompt if prompt is not None else "",
            "system_prompt_digest": self.system_prompt_digest,
            "llm_response_code": llm_response_code,
            "is_valid": bool(is_valid),
            "error_type": classify_error(error_details, stage_hint) if not is_valid else None,
            "error_details": _truncate_middle(error_details, self.error_cap_bytes) if not is_valid else None,
            "error_summary": error_summary,
        }
        self.chain.append(entry)

    def finalize(self, final_outcome: str, attempts_to_correct: int) -> Optional[Path]:
        if self._finalized:
            return None
        self._finalized = True

        payload = {
            "node_id": self.node_id,
            "phase": self.phase,
            "op_name": self.op_name,
            "final_outcome": final_outcome,
            "attempts_to_correct": attempts_to_correct,
            "chain": self.chain,
            "system_prompt": self.system_prompt,
            "system_prompt_digest": self.system_prompt_digest,
        }

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[lineage] failed to create out_dir {self.out_dir}: {exc}")
            return None

        target = self.out_dir / self.filename
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".history_", suffix=".json.tmp", dir=str(self.out_dir)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, target)
            return target
        except Exception as exc:
            print(f"[lineage] failed to write {target}: {exc}")
            return None

    def __repr__(self) -> str:
        last = self.chain[-1] if self.chain else None
        last_state = "empty" if last is None else ("valid" if last.get("is_valid") else "failed")
        node_bit = f" node={self.node_id}" if self.node_id is not None else ""
        op_bit = f" op={self.op_name}" if self.op_name else ""
        return (
            f"<LineageRecorder phase={self.phase}{node_bit}{op_bit} "
            f"attempts={len(self.chain)} last={last_state}>"
        )
