"""Shared model defaults for providers."""

from __future__ import annotations


PROVIDER_ALIASES = {
    "gemini": "google",
    "google-genai": "google",
}

DEFAULT_TEST_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google": "gemini-2.0-flash",
}

DEFAULT_PROJECT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-2.0-flash",
}


def normalize_provider(provider: str | None) -> str:
    value = (provider or "").strip().lower()
    return PROVIDER_ALIASES.get(value, value)

