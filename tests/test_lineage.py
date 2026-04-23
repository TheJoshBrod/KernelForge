from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.lineage import LineageRecorder, _truncate_middle, classify_error


def test_classify_error_markers():
    assert classify_error("[Compilation Failed] expected ';'") == "CompilationFailed"
    assert classify_error("[Output Mismatch entry_0] max_abs_diff=0.5") == "NumericalMismatch"
    assert classify_error("[Runtime Error entry_0] CUDA illegal memory") == "RuntimeError"
    assert classify_error("Failed to extract code from LLM response") == "ExtractFailed"
    assert classify_error(None) is None
    assert classify_error("", stage_hint="llm_api") == "LLMApiError"
    assert classify_error("", stage_hint="setup") == "SetupError"
    assert classify_error("some weird thing") == "Unknown"


def test_truncate_middle_preserves_head_and_tail():
    head = "HEAD_MARKER[Compilation Failed]"
    tail = "TAIL_LINE:42: error final"
    big = head + ("x" * 20000) + tail
    out = _truncate_middle(big, cap=200)
    assert out.startswith("HEAD_MARKER")
    assert out.endswith("final")
    assert "truncated" in out
    assert len(out) < len(big)


def test_truncate_middle_short_unchanged():
    s = "short string"
    assert _truncate_middle(s, cap=100) == s


def test_recorder_writes_expected_shape(tmp_path):
    rec = LineageRecorder(
        out_dir=tmp_path,
        phase="optimizer",
        node_id=7,
        op_name="mul",
        system_prompt="SYS",
    )

    rec.record_attempt(
        iteration=0,
        role="initial",
        prompt="initial user prompt",
        llm_response_code="__global__ void k(){}",
        is_valid=False,
        error_details="[Compilation Failed] expected ';'",
    )
    rec.record_attempt(
        iteration=1,
        role="fix",
        prompt="[Compilation Failed] expected ';'",
        llm_response_code="__global__ void k(){ ; }",
        is_valid=True,
        error_details=None,
    )

    out = rec.finalize(final_outcome="success", attempts_to_correct=2)
    assert out is not None
    assert out.name == "history_7.json"

    data = json.loads(out.read_text())
    assert data["node_id"] == 7
    assert data["phase"] == "optimizer"
    assert data["op_name"] == "mul"
    assert data["final_outcome"] == "success"
    assert data["attempts_to_correct"] == 2
    assert data["system_prompt"] == "SYS"
    assert data["system_prompt_digest"].startswith("sha1:")

    chain = data["chain"]
    assert len(chain) == 2
    assert chain[0]["role"] == "initial"
    assert chain[0]["error_type"] == "CompilationFailed"
    assert chain[0]["is_valid"] is False

    assert chain[1]["role"] == "fix"
    assert chain[1]["is_valid"] is True
    assert chain[1]["error_type"] is None
    assert chain[1]["error_details"] is None

    # Lineage linkage: fix prompt equals prior iteration's raw error
    assert chain[1]["prompt"] == chain[0]["error_details"]


def test_recorder_finalize_is_idempotent(tmp_path):
    rec = LineageRecorder(tmp_path, phase="generator", op_name="add", system_prompt="s")
    rec.record_attempt(
        iteration=0, role="initial", prompt="p", llm_response_code="c",
        is_valid=True, error_details=None,
    )
    first = rec.finalize("success", attempts_to_correct=1)
    second = rec.finalize("success", attempts_to_correct=1)
    assert first is not None
    assert second is None  # second call short-circuits


def test_recorder_rejects_bad_role(tmp_path):
    rec = LineageRecorder(tmp_path, phase="generator", system_prompt="")
    try:
        rec.record_attempt(
            iteration=0, role="bogus", prompt="p", llm_response_code=None,
            is_valid=False, error_details="x",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_recorder_atomic_no_partial_file(tmp_path, monkeypatch):
    """If the write fails mid-stream, no history.json is left behind."""
    rec = LineageRecorder(tmp_path, phase="generator", system_prompt="")
    rec.record_attempt(
        iteration=0, role="initial", prompt="p", llm_response_code="c",
        is_valid=True, error_details=None,
    )

    import src.common.lineage as lineage_mod

    def boom(*a, **kw):
        raise IOError("disk full")

    monkeypatch.setattr(lineage_mod.os, "replace", boom)
    result = rec.finalize("success", attempts_to_correct=1)
    assert result is None
    assert not (tmp_path / "history.json").exists()
    leftovers = list(tmp_path.glob(".history_*.json.tmp"))
    # tmp file might linger on failure; that's acceptable (no corrupt target)
    # but the real target must not exist
    assert not (tmp_path / "history.json").exists()


def test_repr_reflects_state(tmp_path):
    rec = LineageRecorder(tmp_path, phase="optimizer", node_id=3, system_prompt="")
    assert "attempts=0" in repr(rec)
    assert "last=empty" in repr(rec)
    rec.record_attempt(
        iteration=0, role="initial", prompt="p", llm_response_code="c",
        is_valid=False, error_details="[Compilation Failed]",
    )
    assert "attempts=1" in repr(rec)
    assert "last=failed" in repr(rec)
