"""Tests for the LLM usage tracking pipeline:
- src/llm/pricing.py  (compute_cost)
- src/llm/usage_db.py (log_llm_call + aggregate reads)
- src/llm/litellm_callback.py (LiteLLM success-callback extraction)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.pricing import compute_cost
from src.llm.usage_db import (
    log_llm_call,
    get_project_totals,
    get_operator_totals,
    get_recent_calls,
)
from src.llm.litellm_callback import _extract_usage, register_worker_usage_callback


# ---------- pricing ----------

def test_compute_cost_gpt54():
    in_cost, out_cost, total = compute_cost("gpt-5.4", 1_000_000, 1_000_000)
    assert in_cost == 1.25
    assert out_cost == 10.0
    assert abs(total - 11.25) < 1e-9


def test_compute_cost_litellm_prefix():
    # "openai/gpt-4o-mini" should resolve to gpt-4o-mini pricing.
    in_cost, out_cost, total = compute_cost("openai/gpt-4o-mini", 2_000_000, 1_000_000)
    assert abs(in_cost - 0.30) < 1e-9   # 2M * 0.15 / 1M
    assert abs(out_cost - 0.60) < 1e-9  # 1M * 0.60 / 1M
    assert abs(total - 0.90) < 1e-9


def test_compute_cost_unknown_model_zero():
    assert compute_cost("totally-made-up-model", 1000, 1000) == (0.0, 0.0, 0.0)


def test_compute_cost_longest_prefix():
    # gpt-4.1-mini must beat gpt-4.1 (mini is cheaper).
    in_cost, _, _ = compute_cost("gpt-4.1-mini-2026-01-01", 1_000_000, 0)
    assert abs(in_cost - 0.40) < 1e-9


# ---------- usage_db round-trip ----------

def test_log_and_aggregate(tmp_path: Path):
    proj = tmp_path / "proj"
    # Two operators, three calls with different step types.
    log_llm_call(
        proj,
        {"provider": "openai", "model": "gpt-5.4",
         "input_tokens": 1000, "output_tokens": 500, "reasoning_tokens": 200},
        step_type="generate", job_key="job-a", operator="op_matmul", attempt=1,
    )
    log_llm_call(
        proj,
        {"provider": "openai", "model": "gpt-5.4",
         "input_tokens": 2000, "output_tokens": 1000, "reasoning_tokens": 400},
        step_type="optimize", job_key="job-a", operator="op_matmul",
        iteration=1, attempt=2,
    )
    log_llm_call(
        proj,
        {"provider": "openai", "model": "gpt-4o-mini",
         "input_tokens": 500, "output_tokens": 200, "reasoning_tokens": 0},
        step_type="verifier_summary", job_key="job-a", operator="op_conv",
    )

    totals = get_project_totals(proj)
    assert totals["calls"] == 3
    assert totals["input_tokens"] == 3500
    assert totals["output_tokens"] == 1700
    assert totals["reasoning_tokens"] == 600

    # Cost math: gpt-5.4 ((1000+2000)*1.25 + (500+1000)*10)/1e6 = 0.00375 + 0.015 = 0.01875
    # gpt-4o-mini (500*0.15 + 200*0.60)/1e6 = 0.000075 + 0.00012 = 0.000195
    expected = 0.01875 + 0.000195
    assert abs(totals["total_cost_usd"] - expected) < 1e-9

    per_op = get_operator_totals(proj)
    assert len(per_op) == 2
    # Sorted by cost desc: matmul (0.01875) before conv (0.000195)
    assert per_op[0]["operator"] == "op_matmul"
    assert per_op[0]["calls"] == 2
    assert per_op[1]["operator"] == "op_conv"
    assert per_op[1]["calls"] == 1

    recent = get_recent_calls(proj, limit=10)
    assert len(recent) == 3
    # Most-recent first
    assert recent[0]["step_type"] == "verifier_summary"
    assert recent[0]["model"] == "gpt-4o-mini"


def test_log_empty_usage_is_noop(tmp_path: Path):
    proj = tmp_path / "proj"
    log_llm_call(proj, {}, step_type="generate")
    # DB file should not even be created for an empty payload.
    assert not (proj / "llm_usage.db").exists()
    assert get_project_totals(proj)["calls"] == 0


def test_log_missing_provider_is_skipped(tmp_path: Path):
    proj = tmp_path / "proj"
    log_llm_call(
        proj,
        {"provider": "", "model": "gpt-5.4",
         "input_tokens": 100, "output_tokens": 100},
        step_type="generate",
    )
    assert get_project_totals(proj)["calls"] == 0


def test_unknown_model_logs_tokens_zero_cost(tmp_path: Path):
    proj = tmp_path / "proj"
    log_llm_call(
        proj,
        {"provider": "openai", "model": "future-model-9000",
         "input_tokens": 10, "output_tokens": 20},
        step_type="generate",
    )
    totals = get_project_totals(proj)
    assert totals["calls"] == 1
    assert totals["input_tokens"] == 10
    assert totals["output_tokens"] == 20
    assert totals["total_cost_usd"] == 0.0


# ---------- litellm callback extraction ----------

def _fake_response(model, prompt, completion, reasoning=0):
    """Build a SimpleNamespace that looks enough like a LiteLLM ModelResponse."""
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return SimpleNamespace(model=model, usage=usage)


def test_extract_usage_openai_with_reasoning():
    resp = _fake_response("gpt-5.4", prompt=1234, completion=567, reasoning=120)
    usage = _extract_usage(resp)
    assert usage == {
        "provider": "openai",
        "model": "gpt-5.4",
        "input_tokens": 1234,
        "output_tokens": 567,
        "reasoning_tokens": 120,
    }


def test_extract_usage_dict_shape():
    resp = {
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    # _extract_usage reads .usage via getattr first; dict fallback kicks in.
    # Our helper uses [] fallback, but model is read via getattr only — so
    # test the typical LiteLLM object case too.
    class R:
        pass
    r = R()
    r.model = "gpt-4o-mini"
    r.usage = resp["usage"]
    usage = _extract_usage(r)
    assert usage["provider"] == "openai"
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20


def test_extract_usage_provider_inference():
    assert _extract_usage(_fake_response("claude-sonnet-4-6", 1, 1))["provider"] == "anthropic"
    assert _extract_usage(_fake_response("gemini-2.5-pro", 1, 1))["provider"] == "google"
    assert _extract_usage(_fake_response("anthropic/claude-opus", 1, 1))["provider"] == "anthropic"


def test_register_callback_end_to_end(tmp_path: Path, monkeypatch):
    """Simulate a LiteLLM success callback firing — it should write a row
    tagged step_type='verifier_summary' with the right operator/iteration/attempt."""
    import litellm

    # Start from a clean callback list (other tests may have registered).
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)

    proj = tmp_path / "proj"
    register_worker_usage_callback(
        proj, job_key="job-x", operator="op_reduce",
        iteration=3, attempt=7,
    )
    assert len(litellm.success_callback) == 1

    # Invoke the callback as LiteLLM would.
    resp = _fake_response("gpt-5.4", prompt=800, completion=400, reasoning=100)
    litellm.success_callback[0]({}, resp, 0.0, 0.1)

    recent = get_recent_calls(proj, limit=5)
    assert len(recent) == 1
    row = recent[0]
    assert row["step_type"] == "verifier_summary"
    assert row["operator"] == "op_reduce"
    assert row["iteration"] == 3
    assert row["attempt"] == 7
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-5.4"
    assert row["input_tokens"] == 800
    assert row["output_tokens"] == 400
    assert row["reasoning_tokens"] == 100
    # (800*1.25 + 400*10)/1e6 = 0.001 + 0.004 = 0.005
    assert abs(row["total_cost_usd"] - 0.005) < 1e-9


def test_register_callback_replaces_prior_kforge_callback(tmp_path: Path, monkeypatch):
    """Re-registering with a new context should replace the prior KernelForge
    callback rather than stacking — otherwise each job would emit N rows."""
    import litellm
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)

    proj = tmp_path / "proj"
    register_worker_usage_callback(proj, "job-1", "op_a", 1, 1)
    register_worker_usage_callback(proj, "job-2", "op_b", 2, 2)
    assert len(litellm.success_callback) == 1  # not 2

    resp = _fake_response("gpt-4o-mini", prompt=100, completion=50)
    litellm.success_callback[0]({}, resp, 0.0, 0.1)

    recent = get_recent_calls(proj, limit=5)
    assert len(recent) == 1
    # Latest registration (op_b/job-2) wins.
    assert recent[0]["operator"] == "op_b"
    assert recent[0]["iteration"] == 2
    assert recent[0]["attempt"] == 2
