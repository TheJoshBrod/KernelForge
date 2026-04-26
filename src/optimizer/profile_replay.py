"""Utilities for replaying profiled PyTorch operator calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_LEGACY_KEYWORD_ONLY_PARAMS: dict[str, set[str]] = {
    "torch.nn.functional.grouped_mm": {"offs", "bias", "out_dtype"},
    "torch._grouped_mm": {"offs", "bias", "out_dtype"},
}


def _kind_name(kind: Any) -> str:
    if hasattr(kind, "name"):
        return str(kind.name)
    text = str(kind)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def _is_keyword_only(
    function_name: str | None,
    param_name: str,
    signature: Mapping[str, Any] | None,
) -> bool:
    signature = signature or {}
    kinds = signature.get("kinds") or signature.get("param_kinds") or {}
    if isinstance(kinds, Mapping):
        kind = kinds.get(param_name)
        if kind is not None and _kind_name(kind) == "KEYWORD_ONLY":
            return True

    keyword_only = signature.get("keyword_only") or signature.get("kwonlyargs") or []
    if param_name in keyword_only:
        return True

    if function_name in _LEGACY_KEYWORD_ONLY_PARAMS:
        return param_name in _LEGACY_KEYWORD_ONLY_PARAMS[function_name]
    return False


def normalize_args_kwargs(
    args: list[Any] | tuple[Any, ...] | None,
    kwargs: Mapping[str, Any] | None,
    signature: Mapping[str, Any] | None,
    *,
    function_name: str | None = None,
    preserve_keyword_only: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Normalize a recorded call using profile signature metadata.

    ``preserve_keyword_only=True`` is for replaying the original PyTorch API.
    ``False`` preserves the historic generated-kernel launch ABI, where all
    signature parameters are supplied positionally.
    """

    normalized = list(args or [])
    remaining = dict(kwargs or {})
    signature = signature or {}
    params = list(signature.get("params") or [])
    defaults = signature.get("defaults") or {}

    if not params:
        return normalized, remaining

    for name in params[len(normalized):]:
        if preserve_keyword_only and _is_keyword_only(function_name, name, signature):
            if name not in remaining and isinstance(defaults, Mapping) and name in defaults:
                remaining[name] = defaults[name]
            continue

        if name in remaining:
            normalized.append(remaining.pop(name))
        elif isinstance(defaults, Mapping) and name in defaults:
            normalized.append(defaults[name])
        else:
            break

    return normalized, remaining


def normalize_profile_call_args(
    payload: Mapping[str, Any],
    *,
    preserve_keyword_only: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    function_name = (
        payload.get("function_name")
        or payload.get("op_name")
        or payload.get("op")
    )
    return normalize_args_kwargs(
        payload.get("args") or [],
        payload.get("kwargs") or {},
        payload.get("signature") or {},
        function_name=str(function_name) if function_name else None,
        preserve_keyword_only=preserve_keyword_only,
    )
