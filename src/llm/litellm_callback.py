"""
src/llm/litellm_callback.py
Register a LiteLLM success callback that logs every LLM call made via byllm
(inside the verifier worker subprocess) to the project-scoped usage DB.

We deliberately bypass byllm's abstraction for observability — byllm's
@by(llm) decorators stay intact for the actual verifier logic, but usage
is captured directly from the LiteLLM response so we get the raw
completion_tokens_details.reasoning_tokens without depending on byllm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm.usage_db import log_llm_call


_REGISTERED_KEY = "_kforge_usage_callback_registered"


def _extract_usage(completion_response: Any) -> dict[str, Any]:
    """Pull provider/model/token counts out of a LiteLLM ModelResponse."""
    usage_obj = getattr(completion_response, "usage", None)
    if usage_obj is None:
        # Some LiteLLM responses are dict-like.
        try:
            usage_obj = completion_response["usage"]
        except Exception:
            return {}

    def _field(obj: Any, name: str, default: Any = 0) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    input_tokens = int(_field(usage_obj, "prompt_tokens", 0) or 0)
    output_tokens = int(_field(usage_obj, "completion_tokens", 0) or 0)
    details = _field(usage_obj, "completion_tokens_details", None)
    reasoning_tokens = int(_field(details, "reasoning_tokens", 0) or 0)

    model = str(_field(completion_response, "model", "") or "")
    provider = "openai"
    m = model.lower()
    if "/" in m:
        provider = m.split("/", 1)[0]
    elif "claude" in m:
        provider = "anthropic"
    elif "gemini" in m:
        provider = "google"

    return {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def register_worker_usage_callback(
    proj_dir: Path | str,
    job_key: str | None,
    operator: str | None,
    iteration: int | None,
    attempt: int | None,
) -> None:
    """Append a success callback to litellm.success_callback that logs to
    <proj_dir>/llm_usage.db with step_type='verifier_summary'. Safe to call
    repeatedly within the same worker — only the most recent context wins,
    and we don't stack duplicate callbacks.
    """
    try:
        import litellm
    except Exception:
        return

    proj_dir = Path(proj_dir)

    def on_success(kwargs, completion_response, start_time, end_time):
        try:
            usage = _extract_usage(completion_response)
            if not usage or (usage.get("input_tokens", 0) == 0 and usage.get("output_tokens", 0) == 0):
                return
            log_llm_call(
                proj_dir,
                usage,
                step_type="verifier_summary",
                job_key=job_key,
                operator=operator,
                iteration=iteration,
                attempt=attempt,
            )
        except Exception:
            return

    # Replace any previously-registered KernelForge callback so the context
    # tags (operator/iteration/attempt) always match the current job.
    existing = getattr(litellm, "success_callback", None) or []
    new_callbacks = [cb for cb in existing if not getattr(cb, _REGISTERED_KEY, False)]
    setattr(on_success, _REGISTERED_KEY, True)
    new_callbacks.append(on_success)
    litellm.success_callback = new_callbacks
