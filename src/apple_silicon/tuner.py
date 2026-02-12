from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, TYPE_CHECKING

from src.auth.credentials import apply_auth_env, resolve_auth
from src.config import ensure_llm_config, load_config_data

from .feasibility import repair_candidate_params
from .constants import LLM_TUNER_SYSTEM_PROMPT_PATH
from .kernel_patch import TEMPLATE_MUTATION_SPECS
from .runtime_args import sanitize_runtime_args
from .types import AllowedParams, DeviceProfile, ModelProfile, TuningCandidate

if TYPE_CHECKING:
    from src.llm_tools import GenModel

HOTSPOT_OPS = [
    "mul_mv_q4_k_decode",
    "mul_mv_q5_k_decode",
    "mul_mv_ext_decode",
    "softmax_long_ctx",
    "rms_norm_decode",
    "long_vector_schedule",
]
ALLOWED_TEMPLATE_MUTATIONS = sorted(TEMPLATE_MUTATION_SPECS.keys())

JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class LlmTuningError(RuntimeError):
    pass


def _provider_model(provider: str) -> tuple[str, str, str]:
    provider = provider.strip().lower()
    if provider in {"openai", "gpt", "chatgpt"}:
        return "openai", "OPENAI_MODEL", "gpt-5.2"
    if provider == "anthropic":
        return "anthropic", "ANTHROPIC_MODEL", "claude-opus-4-5-20251101"
    if provider == "gemini":
        return "gemini", "GEMINI_MODEL", "gemini-2.5-flash"
    raise LlmTuningError(
        f"Unsupported LLM provider '{provider}'. Use openai, anthropic, or gemini."
    )


def _load_system_prompt() -> str:
    if LLM_TUNER_SYSTEM_PROMPT_PATH.exists():
        return LLM_TUNER_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a low-level Apple Silicon/llama.cpp tuning agent. "
        "Return strict JSON with kernel_overrides and runtime_args."
    )


def _extract_json_payload(response: str) -> dict[str, Any]:
    text = (response or "").strip()
    if not text:
        raise LlmTuningError("LLM returned an empty response.")

    match = JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise LlmTuningError(f"LLM response did not contain JSON: {response[:300]}")
        text = text[start : end + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlmTuningError(f"Failed to parse tuner JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LlmTuningError("Tuner payload must be a JSON object.")
    return payload


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _clamp(value: int, *, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _coerce_allowed_params(
    allowed_params: AllowedParams | dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(allowed_params, AllowedParams):
        return asdict(allowed_params)
    if isinstance(allowed_params, dict):
        return dict(allowed_params)
    return {}


def _normalize_kernel_overrides(
    payload: Any,
    *,
    allowed_params: AllowedParams | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise LlmTuningError("kernel_overrides must be an object.")

    allowed = _coerce_allowed_params(allowed_params)
    tew = _as_int(allowed.get("thread_execution_width")) or 32
    max_threads = _as_int(allowed.get("max_threads_per_threadgroup")) or 1024
    max_threads = max(1, max_threads)
    tew = max(1, tew)

    normalized: dict[str, Any] = {}
    for op, config in payload.items():
        op_name = str(op).strip()
        if not op_name:
            continue
        if not isinstance(config, dict):
            continue

        item: dict[str, Any] = {}
        variant_name = str(config.get("variant_name", "")).strip()
        if variant_name:
            item["variant_name"] = variant_name

        threadgroup = _as_int(config.get("threadgroup"))
        if threadgroup is not None:
            fixed = _clamp(threadgroup, lo=tew, hi=max_threads)
            fixed = max(tew, (fixed // tew) * tew)
            item["threadgroup"] = fixed

        tile = config.get("tile")
        if isinstance(tile, list) and len(tile) == 2:
            tile_x = _as_int(tile[0])
            tile_y = _as_int(tile[1])
            if tile_x is not None and tile_y is not None:
                tile_x = max(1, tile_x)
                tile_y = max(1, tile_y)
                if tile_x * tile_y > max_threads:
                    tile_y = max(1, max_threads // tile_x)
                    if tile_x * tile_y > max_threads:
                        tile_x = max(1, max_threads // max(1, tile_y))
                item["tile"] = [tile_x, tile_y]

        use_simdgroup = config.get("use_simdgroup")
        if isinstance(use_simdgroup, bool):
            item["use_simdgroup"] = use_simdgroup

        notes = str(config.get("notes", "")).strip()
        if notes:
            item["notes"] = notes

        vector_width = _as_int(config.get("vector_width"))
        if vector_width is not None and vector_width > 0:
            item["vector_width"] = vector_width
        k_dim = _as_int(config.get("k_dim"))
        if k_dim is not None and k_dim > 0:
            item["k_dim"] = k_dim
        has_tail_path = config.get("has_tail_path")
        if isinstance(has_tail_path, bool):
            item["has_tail_path"] = has_tail_path

        if item:
            normalized[op_name] = item

    return normalized


def _normalize_template_mutations(
    payload: Any,
    *,
    allowed_params: AllowedParams | dict[str, Any] | None = None,
) -> dict[str, int]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise LlmTuningError("template_mutations must be an object.")
    allowed = _coerce_allowed_params(allowed_params)
    allowed_simd = {
        int(v)
        for v in (allowed.get("allowed_simd_widths") or [])
        if _as_int(v) is not None and int(v) > 0
    }
    out: dict[str, int] = {}
    for key, raw_value in payload.items():
        name = str(key).strip().lower()
        if not name or name not in ALLOWED_TEMPLATE_MUTATIONS:
            continue
        try:
            value = int(raw_value)
        except Exception:
            continue
        spec = TEMPLATE_MUTATION_SPECS.get(name) or {}
        min_v = int(spec.get("min", value))
        max_v = int(spec.get("max", value))
        value = _clamp(value, lo=min_v, hi=max_v)
        if name == "n_simdwidth" and allowed_simd:
            if value not in allowed_simd:
                value = min(allowed_simd, key=lambda candidate: (abs(candidate - value), candidate))
        out[name] = value
    return out


def _normalize_source_patches(payload: Any) -> list[dict[str, Any]]:
    # Keep generation constrained to template/function-constant space by default.
    if os.environ.get("CGINS_ALLOW_SOURCE_PATCHES", "").strip() != "1":
        return []
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise LlmTuningError("source_patches must be a list.")
    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file", "")).strip()
        find = str(item.get("find", ""))
        replace = str(item.get("replace", ""))
        patch_id = str(item.get("patch_id", "")).strip() or "patch"
        if not file_name or not find:
            continue
        out.append(
            {
                "patch_id": patch_id,
                "file": file_name,
                "find": find,
                "replace": replace,
            }
        )
    return out


def _summarize_attempts(previous_attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for attempt in previous_attempts[-4:]:
        summary.append(
            {
                "candidate_name": attempt.get("candidate_name", ""),
                "score": attempt.get("score"),
                "pass_gate": attempt.get("pass_gate"),
                "delta": attempt.get("delta", {}),
            }
        )
    return summary


def _build_tuning_prompt(
    *,
    device: DeviceProfile,
    model: ModelProfile,
    profile_mode: str,
    gate_mode: str,
    baseline: list[dict[str, Any]],
    previous_attempts: list[dict[str, Any]],
    allowed_params: AllowedParams | dict[str, Any] | None = None,
) -> str:
    allowed = _coerce_allowed_params(allowed_params)
    allowed_simd = [
        int(v)
        for v in (allowed.get("allowed_simd_widths") or [])
        if _as_int(v) is not None and int(v) > 0
    ]
    max_threads = _as_int(allowed.get("max_threads_per_threadgroup")) or 1024
    tew = _as_int(allowed.get("thread_execution_width")) or 32
    payload = {
        "task": (
            "Propose ONE new llama.cpp Apple Silicon tuning candidate for Qwen/Llama GGUF "
            "that can outperform baseline on this machine. Prioritize robust decode uplift across both chat and long profiles."
        ),
        "scope": {
            "runtime": "llama.cpp",
            "target_profiles": profile_mode,
            "gate_mode": gate_mode,
            "supported_hotspots": HOTSPOT_OPS,
            "allowed_template_mutations": ALLOWED_TEMPLATE_MUTATIONS,
            "patch_files": ["ggml-metal.metal", "ggml-metal-impl.h"],
        },
        "device": asdict(device),
        "model": {
            "name": model.name,
            "architecture": model.architecture,
            "quant": model.quant,
            "sha256": model.sha256,
        },
        "baseline": baseline,
        "recent_attempts": _summarize_attempts(previous_attempts),
        "output_schema": {
            "candidate_name": "short-name",
            "rationale": "one-paragraph rationale",
            "kernel_overrides": {
                "<hotspot_op_name>": {
                    "variant_name": "string",
                    "threadgroup": 128,
                    "tile": [32, 16],
                    "use_simdgroup": True,
                    "notes": "short explanation",
                }
            },
            "runtime_args": ["-fa", "-ub", "512", "--cgins-long-vector-schedule", "on"],
                "template_mutations": {
                    "n_r0_q4_k": 3,
                    "n_r0_q5_k": 3,
                    "n_simdwidth": tew,
                },
                "source_patches": [
                    {
                        "patch_id": "softmax-guard",
                    "file": "ggml-metal.metal",
                    "find": "if (tptg.x > N_SIMDWIDTH) {",
                    "replace": "if (tptg.x > N_SIMDWIDTH) {",
                }
            ],
        },
        "allowed_params": allowed,
        "constraints": [
            "Return valid JSON only (no markdown).",
            "kernel_overrides can be empty when template_mutations/source_patches are provided.",
            "runtime_args can only include: -fa, --flash-attn, -b, --batch-size, -ub, --ubatch-size, -ngl, --n-gpu-layers, -t, --threads, --threads-batch, --cgins-long-vector-schedule.",
            "template_mutations keys must be from allowed_template_mutations and values must be integers.",
            f"n_simdwidth must be one of {allowed_simd or [tew]}.",
            f"threadgroup must be a multiple of thread_execution_width ({tew}) and <= {max_threads}.",
            "if tile is provided, tile[0] * tile[1] must be <= max_threads_per_threadgroup.",
            "Prefer template_mutations and runtime_args; avoid source_patches unless explicitly required.",
            "If source_patches are used, they must only touch patch_files and include file/find/replace.",
            "Use integer values for numeric runtime args.",
            "Bias towards candidates that improve the worse of chat/long decode, not just one profile.",
        ],
    }
    return json.dumps(payload, indent=2)


def propose_llm_candidate(
    *,
    device: DeviceProfile,
    model: ModelProfile,
    profile_mode: str,
    gate_mode: str,
    baseline: list[dict[str, Any]],
    previous_attempts: list[dict[str, Any]],
    allowed_params: AllowedParams | dict[str, Any] | None = None,
) -> tuple[TuningCandidate, dict[str, Any]]:
    cfg, _ = load_config_data()
    status = resolve_auth(config=cfg, env=dict(os.environ), runtime_context={"in_container": False})
    apply_auth_env(status, os.environ)
    provider = ensure_llm_config().strip().lower()
    if not provider or status.mode_effective == "unconfigured":
        raise LlmTuningError(
            f"No usable auth configured for LLM tuning ({status.reason}). "
            "Set API key or account session in Settings."
        )

    provider_name, model_env, default_model = _provider_model(provider)
    model_name = os.environ.get(model_env, default_model).strip() or default_model

    from src.llm_tools import GenModel  # lazy import so unit tests don't require provider SDK deps

    llm = GenModel(_load_system_prompt())
    prompt = _build_tuning_prompt(
        device=device,
        model=model,
        profile_mode=profile_mode,
        gate_mode=gate_mode,
        baseline=baseline,
        previous_attempts=previous_attempts,
        allowed_params=allowed_params,
    )
    response = llm.chat(prompt, model_name)
    if response.lower().startswith("error calling"):
        raise LlmTuningError(response)

    payload = _extract_json_payload(response)
    overrides = _normalize_kernel_overrides(
        payload.get("kernel_overrides", {}),
        allowed_params=allowed_params,
    )
    runtime_args = sanitize_runtime_args(payload.get("runtime_args", []))
    template_mutations = _normalize_template_mutations(
        payload.get("template_mutations", {}),
        allowed_params=allowed_params,
    )
    source_patches = _normalize_source_patches(payload.get("source_patches", []))
    repaired_mutations, repaired_overrides, repairs = repair_candidate_params(
        template_mutations=template_mutations,
        kernel_overrides=overrides,
        allowed_params=allowed_params,
        device=device,
    )

    candidate = TuningCandidate(
        candidate_name=str(payload.get("candidate_name", "")).strip() or "llm-candidate",
        rationale=str(payload.get("rationale", "")).strip() or "No rationale provided.",
        kernel_overrides=repaired_overrides,
        runtime_args=runtime_args,
        template_mutations=repaired_mutations,
        source_patches=source_patches,
        raw_response=response,
    )
    meta = {
        "provider": provider_name,
        "model": model_name,
        "constraint_repairs": repairs,
        "allowed_params": _coerce_allowed_params(allowed_params),
    }
    return candidate, meta
