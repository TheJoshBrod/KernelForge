"""
src/llm/pricing.py
Per-model USD pricing for LLM calls, keyed by longest-prefix match on model name.

Prices are USD per 1,000,000 tokens. Reasoning tokens are billed at the output
rate for OpenAI reasoning models, and OpenAI's `completion_tokens` already
includes reasoning tokens — so the consistent formula is simply:

    input_cost  = input_tokens  * input_per_mtok  / 1e6
    output_cost = output_tokens * output_per_mtok / 1e6     # covers reasoning
"""

from __future__ import annotations

from typing import Tuple


# USD per 1M tokens. Longest-prefix match wins.
_PRICE_TABLE: dict[str, dict[str, float]] = {
    # OpenAI GPT-5 family (medium-reasoning tier)
    "gpt-5.4": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5":   {"input_per_mtok": 1.25, "output_per_mtok": 10.00},

    # OpenAI GPT-4.1 family
    "gpt-4.1-mini": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
    "gpt-4.1-nano": {"input_per_mtok": 0.10, "output_per_mtok": 0.40},
    "gpt-4.1":      {"input_per_mtok": 2.00, "output_per_mtok": 8.00},

    # OpenAI GPT-4o family
    "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    "gpt-4o":      {"input_per_mtok": 2.50, "output_per_mtok": 10.00},

    # OpenAI o-series reasoning models
    "o4-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o3-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o3":      {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    "o1-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o1":      {"input_per_mtok": 15.00, "output_per_mtok": 60.00},

    # Anthropic Claude API standard pricing
    "claude-opus-4-7": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-opus-4-6": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-opus-4-5": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-opus-4-1": {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
    "claude-opus-4":   {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
    "claude-3-opus":   {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
    "claude-sonnet-4-6": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-sonnet-4-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-sonnet-4":   {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-sonnet-3-7": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-3-7-sonnet": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "claude-haiku-3-5": {"input_per_mtok": 0.80, "output_per_mtok": 4.00},
    "claude-3-5-haiku": {"input_per_mtok": 0.80, "output_per_mtok": 4.00},
    "claude-haiku-3":   {"input_per_mtok": 0.25, "output_per_mtok": 1.25},
    "claude-3-haiku":   {"input_per_mtok": 0.25, "output_per_mtok": 1.25},
}


def _lookup(model: str) -> dict[str, float] | None:
    if not model:
        return None
    m = model.strip().lower()
    # LiteLLM-formatted names like "openai/gpt-5.4" — strip known provider prefix.
    if "/" in m:
        m = m.split("/", 1)[1]
    best_key = ""
    for key in _PRICE_TABLE:
        if m.startswith(key) and len(key) > len(best_key):
            best_key = key
    return _PRICE_TABLE.get(best_key) if best_key else None


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> Tuple[float, float, float]:
    """Return (input_cost_usd, output_cost_usd, total_cost_usd). Unknown model → 0s."""
    prices = _lookup(model)
    if prices is None:
        return 0.0, 0.0, 0.0
    in_cost = (input_tokens or 0) * prices["input_per_mtok"] / 1_000_000
    out_cost = (output_tokens or 0) * prices["output_per_mtok"] / 1_000_000
    return in_cost, out_cost, in_cost + out_cost
