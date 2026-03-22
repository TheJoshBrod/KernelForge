"""
src/llm_tools.py
Generalized LLM tooling for handling model agnostic conversations and tooling.
"""
from __future__ import annotations

import multiprocessing
import queue
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import anthropic
from google import genai
from openai import OpenAI

from src.optimizer.config.settings import settings


_RETRYABLE_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "connection",
    "connect",
    "read error",
    "api connection",
    "502",
    "503",
    "504",
    "429",
)


def _provider_for_model(model: str) -> tuple[str, str]:
    lowered = str(model or "").strip().lower()
    if "claude" in lowered:
        return "anthropic", "Claude"
    if "gemini" in lowered:
        return "google", "Gemini"
    if "gpt" in lowered:
        return "openai", "OpenAI"
    return "", ""


def _provider_payload(
    provider: str,
    model: str,
    sys_prompt: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "sys_prompt": sys_prompt,
        "history": history,
        "request_timeout": float(getattr(settings, "llm_request_timeout_seconds", 90)),
    }


def _anthropic_request(payload: dict[str, Any]) -> str:
    client = anthropic.Anthropic(
        timeout=payload["request_timeout"],
        max_retries=0,
    )
    message = client.messages.create(
        model=payload["model"],
        max_tokens=8196,
        system=payload["sys_prompt"],
        messages=payload["history"],
    )
    response_text = ""
    for block in message.content:
        if hasattr(block, "text"):
            response_text += block.text
    return response_text


def _gemini_request(payload: dict[str, Any]) -> str:
    client = genai.Client()
    history = payload["history"]
    if len(history) == 1:
        response = client.models.generate_content(
            model=payload["model"],
            contents=history[0]["content"],
            config={
                "system_instruction": payload["sys_prompt"],
            },
        )
        return response.text

    chat_history = []
    for msg in history[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        chat_history.append(
            {
                "role": role,
                "parts": [{"text": msg["content"]}],
            }
        )

    chat = client.chats.create(
        model=payload["model"],
        config={
            "system_instruction": payload["sys_prompt"],
        },
        history=chat_history,
    )
    latest_user_msg = history[-1]["content"]
    response = chat.send_message(latest_user_msg)
    return response.text


def _openai_request(payload: dict[str, Any]) -> str:
    client = OpenAI(
        timeout=payload["request_timeout"],
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=payload["model"],
        messages=[{"role": "system", "content": payload["sys_prompt"]}] + payload["history"],
        max_completion_tokens=4096,
    )
    content = response.choices[0].message.content
    return content if content is not None else ""


def _provider_worker(payload: dict[str, Any], result_queue) -> None:
    try:
        provider = payload["provider"]
        if provider == "anthropic":
            text = _anthropic_request(payload)
        elif provider == "google":
            text = _gemini_request(payload)
        elif provider == "openai":
            text = _openai_request(payload)
        else:
            raise RuntimeError("Unsupported llm model/provider")
        result_queue.put({"ok": True, "text": text})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def _close_queue_handle(q) -> None:
    if q is None:
        return
    try:
        q.close()
    except Exception:
        pass
    try:
        q.join_thread()
    except Exception:
        pass


def _stop_process(proc, grace_seconds: float = 1.0) -> None:
    if proc is None:
        return
    try:
        proc.join(timeout=grace_seconds)
    except Exception:
        pass
    if proc.is_alive():
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=grace_seconds)
        except Exception:
            pass
    if proc.is_alive() and hasattr(proc, "kill"):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.join(timeout=grace_seconds)
        except Exception:
            pass


def _emit_status(
    status_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if status_callback is None:
        return
    text = str(message or "").strip()
    if not text:
        return
    try:
        status_callback(text)
    except Exception:
        pass


def _request_with_watchdog(
    payload: dict[str, Any],
    provider_label: str,
    *,
    watchdog_timeout: float | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    watchdog_timeout = float(
        watchdog_timeout
        if watchdog_timeout is not None
        else getattr(settings, "llm_watchdog_timeout_seconds", 120)
    )
    heartbeat_seconds = max(1.0, min(5.0, watchdog_timeout / 6.0 if watchdog_timeout > 0 else 5.0))
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_provider_worker, args=(payload, result_queue))
    proc.daemon = True
    proc.start()
    start_time = time.monotonic()
    _emit_status(status_callback, f"Calling {provider_label} API")

    try:
        deadline = start_time + watchdog_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return (
                    False,
                    f"Error calling {provider_label} API: request timed out after {int(watchdog_timeout)} seconds",
                )
            try:
                result = result_queue.get(timeout=min(heartbeat_seconds, remaining))
                break
            except queue.Empty:
                if not proc.is_alive():
                    try:
                        result = result_queue.get_nowait()
                        break
                    except queue.Empty:
                        return False, f"Error calling {provider_label} API: worker exited without response"
                elapsed = int(max(0.0, time.monotonic() - start_time))
                timeout_display = int(max(1.0, round(watchdog_timeout)))
                _emit_status(
                    status_callback,
                    f"Waiting on {provider_label} API ({elapsed}s/{timeout_display}s)",
                )

        if not isinstance(result, dict):
            return False, f"Error calling {provider_label} API: invalid worker response"
        if result.get("ok"):
            return True, str(result.get("text", ""))
        error_text = str(result.get("error", "")).strip() or "Unknown provider error"
        return False, f"Error calling {provider_label} API: {error_text}"
    finally:
        _stop_process(proc)
        _close_queue_handle(result_queue)


def _should_retry(error_text: str) -> bool:
    lowered = str(error_text or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _RETRYABLE_ERROR_MARKERS)


class GenModel:
    """
    Provider-agnostic chat history.

    - sys_prompt is stored separately
    - history contains only conversational turns w/ tool calls
    - tools are possible tools model can call
    """

    def __init__(self, sys_prompt: str):
        self.sys_prompt = sys_prompt
        self.history: List[Dict[str, Any]] = []
        self.tools: Dict[str, callable] = {}

    def chat(
        self,
        user_msg: str,
        model: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        if not user_msg or not model:
            return ""

        provider, provider_label = _provider_for_model(model)
        if not provider:
            return "Unsupported llm model/provider"

        self.__user(user_msg)
        payload = _provider_payload(provider, model, self.sys_prompt, list(self.history))

        retry_limit = max(0, int(getattr(settings, "llm_retry_limit", 2)))
        total_attempts = retry_limit + 1
        backoff_seconds = float(getattr(settings, "llm_retry_backoff_seconds", 2.0))
        total_timeout_seconds = float(getattr(settings, "llm_total_timeout_seconds", 180))
        total_deadline = (
            time.monotonic() + total_timeout_seconds
            if total_timeout_seconds > 0
            else None
        )

        final_error = ""
        for attempt in range(total_attempts):
            if total_deadline is not None:
                remaining_total = total_deadline - time.monotonic()
                if remaining_total <= 0:
                    final_error = (
                        f"Error calling {provider_label} API: overall request deadline exceeded "
                        f"after {int(total_timeout_seconds)} seconds"
                    )
                    break
                attempt_watchdog = min(
                    float(getattr(settings, "llm_watchdog_timeout_seconds", 120)),
                    remaining_total,
                )
            else:
                attempt_watchdog = float(getattr(settings, "llm_watchdog_timeout_seconds", 120))

            _emit_status(
                status_callback,
                f"{provider_label} attempt {attempt + 1}/{total_attempts}",
            )
            ok, response = _request_with_watchdog(
                payload,
                provider_label,
                watchdog_timeout=attempt_watchdog,
                status_callback=status_callback,
            )
            if ok:
                self.__assistant(response)
                return response

            final_error = response
            if attempt + 1 >= total_attempts or not _should_retry(response):
                break

            sleep_seconds = backoff_seconds * (2 ** attempt)
            if total_deadline is not None:
                remaining_total = total_deadline - time.monotonic()
                if remaining_total <= 0:
                    final_error = (
                        f"Error calling {provider_label} API: overall request deadline exceeded "
                        f"after {int(total_timeout_seconds)} seconds"
                    )
                    break
                sleep_seconds = min(sleep_seconds, remaining_total)
            if sleep_seconds > 0:
                _emit_status(
                    status_callback,
                    f"{provider_label} retry backoff {attempt + 1}/{retry_limit}: {sleep_seconds:.1f}s",
                )
                time.sleep(sleep_seconds)

        self.__assistant(final_error)
        return final_error

    def set_sys_prompt(self, sys_prompt):
        self.sys_prompt = sys_prompt

    def set_tools(self, tools: Dict[str, callable]):
        self.tools = tools

    def to_json(self, **kwargs) -> str:
        import json

        return json.dumps(self.history, **kwargs)

    def __repr__(self) -> str:
        return f"ChatHistory(turns={len(self.history)})"

    def __user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def __assistant(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def __tool(self, name: str, content: str) -> None:
        self.history.append(
            {
                "role": "tool",
                "name": name,
                "content": content,
            }
        )
