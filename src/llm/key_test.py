"""Fast per-provider API key validation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import anthropic
from google import genai
from openai import OpenAI

from .models import DEFAULT_TEST_MODELS, normalize_provider


def _error_payload(provider: str, started: float, error: str, code: str = "Error") -> dict[str, Any]:
    return {
        "provider": provider,
        "success": False,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "error_code": code,
        "error": error,
    }


def test_api_key(provider: str, api_key: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    started = time.perf_counter()
    provider_norm = normalize_provider(provider)
    key = (api_key or "").strip()

    if not provider_norm:
        return _error_payload(provider_norm, started, "Missing provider", "MissingProvider")
    if provider_norm not in DEFAULT_TEST_MODELS:
        return _error_payload(provider_norm, started, f"Unsupported provider: {provider_norm}", "UnsupportedProvider")
    if not key:
        return _error_payload(provider_norm, started, "Missing API key", "MissingApiKey")

    model = DEFAULT_TEST_MODELS[provider_norm]
    try:
        if provider_norm == "openai":
            client = OpenAI(api_key=key, timeout=timeout_seconds)
            client.responses.create(
                model=model,
                input="ping",
                max_output_tokens=1,
            )
        elif provider_norm == "anthropic":
            client = anthropic.Anthropic(api_key=key, timeout=timeout_seconds)
            client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        elif provider_norm == "google":
            client = genai.Client(api_key=key)
            client.models.generate_content(
                model=model,
                contents="ping",
                config={"max_output_tokens": 1},
            )
        else:
            return _error_payload(provider_norm, started, f"Unsupported provider: {provider_norm}", "UnsupportedProvider")
    except Exception as exc:
        return _error_payload(provider_norm, started, str(exc), exc.__class__.__name__)

    return {
        "provider": provider_norm,
        "success": True,
        "tested_model": model,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def test_all_api_keys(api_keys: dict[str, str] | None, timeout_seconds: float = 5.0) -> dict[str, Any]:
    keys = api_keys or {}
    providers = ["openai", "anthropic", "google"]
    results: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {
            pool.submit(test_api_key, provider, keys.get(provider, ""), timeout_seconds): provider
            for provider in providers
        }
        for fut in as_completed(future_map):
            provider = future_map[fut]
            try:
                results[provider] = fut.result()
            except Exception as exc:
                results[provider] = {
                    "provider": provider,
                    "success": False,
                    "error_code": exc.__class__.__name__,
                    "error": str(exc),
                }

    all_success = True
    for provider in providers:
        if not results.get(provider, {}).get("success", False):
            all_success = False
            break

    return {
        "success": all_success,
        "results": results,
    }

