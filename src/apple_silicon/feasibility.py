from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import AllowedParams, DeviceProfile, FeasibilityCheckRecord

MAX_THREADS_PER_THREADGROUP_DEFAULT = 1024
MAX_THREADGROUP_MEMORY_BYTES_DEFAULT = 32 * 1024
SUPPORTED_SIMD_WIDTHS = {16, 32, 64}
UNROLL_LIMITS = {
    "n_r0_q4_k": (1, 8),
    "n_r0_q5_k": (1, 8),
    "n_r0_q6_k": (1, 8),
}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _nearest_allowed(value: int, allowed: list[int]) -> int:
    if not allowed:
        return value
    return min(allowed, key=lambda candidate: (abs(candidate - value), candidate))


def derive_allowed_params(device: DeviceProfile) -> AllowedParams:
    tew = _as_int(device.metal_thread_execution_width) or 32
    if tew <= 0:
        tew = 32
    max_threads = _as_int(device.metal_max_threads_per_threadgroup) or MAX_THREADS_PER_THREADGROUP_DEFAULT
    if max_threads <= 0:
        max_threads = MAX_THREADS_PER_THREADGROUP_DEFAULT
    max_tg_mem = _as_int(device.metal_max_threadgroup_memory_bytes) or MAX_THREADGROUP_MEMORY_BYTES_DEFAULT
    if max_tg_mem <= 0:
        max_tg_mem = MAX_THREADGROUP_MEMORY_BYTES_DEFAULT

    allowed_simd = [tew] if tew in SUPPORTED_SIMD_WIDTHS else [_nearest_allowed(tew, sorted(SUPPORTED_SIMD_WIDTHS))]
    source = "device_probe" if device.metal_thread_execution_width is not None else "default_fallback"
    return AllowedParams(
        thread_execution_width=int(tew),
        allowed_simd_widths=[int(v) for v in sorted(set(allowed_simd))],
        max_threads_per_threadgroup=int(max_threads),
        max_threadgroup_memory_bytes=int(max_tg_mem),
        source=source,
    )


def _coerce_allowed_params(
    *,
    device: DeviceProfile,
    allowed_params: AllowedParams | dict[str, Any] | None = None,
) -> AllowedParams:
    if isinstance(allowed_params, AllowedParams):
        return allowed_params
    if isinstance(allowed_params, dict):
        base = derive_allowed_params(device)
        tew = _as_int(allowed_params.get("thread_execution_width")) or base.thread_execution_width
        max_threads = _as_int(allowed_params.get("max_threads_per_threadgroup")) or base.max_threads_per_threadgroup
        max_mem = _as_int(allowed_params.get("max_threadgroup_memory_bytes")) or base.max_threadgroup_memory_bytes
        simd_raw = allowed_params.get("allowed_simd_widths")
        simd_values = []
        if isinstance(simd_raw, list):
            for item in simd_raw:
                value = _as_int(item)
                if isinstance(value, int) and value > 0:
                    simd_values.append(value)
        if not simd_values:
            simd_values = list(base.allowed_simd_widths)
        return AllowedParams(
            thread_execution_width=int(max(1, tew)),
            allowed_simd_widths=sorted(set(int(v) for v in simd_values)),
            max_threads_per_threadgroup=int(max(1, max_threads)),
            max_threadgroup_memory_bytes=int(max(1024, max_mem)),
            source=str(allowed_params.get("source") or base.source),
        )
    return derive_allowed_params(device)


def repair_candidate_params(
    *,
    template_mutations: dict[str, int] | None,
    kernel_overrides: dict[str, Any] | None,
    allowed_params: AllowedParams | dict[str, Any] | None,
    device: DeviceProfile,
) -> tuple[dict[str, int], dict[str, Any], list[str]]:
    allowed = _coerce_allowed_params(device=device, allowed_params=allowed_params)
    repairs: list[str] = []

    mutations = dict(template_mutations or {})
    overrides: dict[str, Any] = {
        str(op): dict(cfg) if isinstance(cfg, dict) else cfg
        for op, cfg in (kernel_overrides or {}).items()
    }

    simd = _as_int(mutations.get("n_simdwidth"))
    if simd is not None and allowed.allowed_simd_widths and simd not in set(allowed.allowed_simd_widths):
        fixed = _nearest_allowed(simd, list(allowed.allowed_simd_widths))
        mutations["n_simdwidth"] = fixed
        repairs.append(f"clamp_simdwidth:{simd}->{fixed}")

    for key, (min_v, max_v) in UNROLL_LIMITS.items():
        value = _as_int(mutations.get(key))
        if value is None:
            continue
        fixed = max(min_v, min(max_v, value))
        if fixed != value:
            mutations[key] = fixed
            repairs.append(f"clamp_{key}:{value}->{fixed}")

    tew = max(1, int(allowed.thread_execution_width))
    max_threads = max(1, int(allowed.max_threads_per_threadgroup))
    max_tg_mem = max(1024, int(allowed.max_threadgroup_memory_bytes))

    for op_name, cfg in list(overrides.items()):
        if not isinstance(cfg, dict):
            continue

        threadgroup = _as_int(cfg.get("threadgroup"))
        if threadgroup is not None:
            fixed = max(tew, min(max_threads, threadgroup))
            fixed = max(tew, (fixed // tew) * tew)
            if fixed != threadgroup:
                cfg["threadgroup"] = fixed
                repairs.append(f"clamp_threadgroup:{op_name}:{threadgroup}->{fixed}")

        tile = cfg.get("tile")
        if isinstance(tile, list) and len(tile) == 2:
            tile_x = _as_int(tile[0]) or 1
            tile_y = _as_int(tile[1]) or 1
            tile_x = max(1, tile_x)
            tile_y = max(1, tile_y)
            if tile_x * tile_y > max_threads:
                # Deterministically shrink Y first, then X.
                tile_y = max(1, max_threads // tile_x)
                if tile_x * tile_y > max_threads:
                    tile_x = max(1, max_threads // max(1, tile_y))
                repairs.append(f"clamp_tile_threads:{op_name}->{tile_x}x{tile_y}")
            shared_bytes_est = tile_x * tile_y * 4 * 2
            if shared_bytes_est > max_tg_mem:
                target_threads = max(1, max_tg_mem // 8)
                tile_y = max(1, min(tile_y, target_threads // max(1, tile_x)))
                shared_bytes_est = tile_x * tile_y * 8
                repairs.append(f"clamp_tile_shared_mem:{op_name}:{shared_bytes_est}")
            fixed_tile = [int(tile_x), int(tile_y)]
            if fixed_tile != tile:
                cfg["tile"] = fixed_tile

        vector_width = _as_int(cfg.get("vector_width"))
        k_dim = _as_int(cfg.get("k_dim"))
        has_tail_path = bool(cfg.get("has_tail_path", False))
        if vector_width is not None and k_dim is not None and vector_width > 0:
            if (k_dim % vector_width) != 0 and not has_tail_path:
                cfg["has_tail_path"] = True
                repairs.append(f"enable_tail_path:{op_name}:k={k_dim}:vw={vector_width}")

        overrides[op_name] = cfg

    return mutations, overrides, repairs


def evaluate_candidate_feasibility(
    *,
    device: DeviceProfile,
    template_mutations: dict[str, int] | None,
    kernel_overrides: dict[str, Any] | None,
    allowed_params: AllowedParams | dict[str, Any] | None = None,
) -> FeasibilityCheckRecord:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    allowed = _coerce_allowed_params(device=device, allowed_params=allowed_params)
    limits = {
        "device_chip": device.chip,
        "allowed_params": asdict(allowed),
    }

    mutations = dict(template_mutations or {})
    overrides = dict(kernel_overrides or {})

    simd = _as_int(mutations.get("n_simdwidth"))
    if simd is not None and allowed.allowed_simd_widths and simd not in set(allowed.allowed_simd_widths):
        reasons.append(f"unsupported_simd_width:{simd}")

    for key, (min_v, max_v) in UNROLL_LIMITS.items():
        value = _as_int(mutations.get(key))
        if value is None:
            continue
        if value < min_v or value > max_v:
            reasons.append(f"unroll_out_of_range:{key}={value}")

    tew = max(1, int(allowed.thread_execution_width))
    max_threads = max(1, int(allowed.max_threads_per_threadgroup))
    max_tg_mem = max(1024, int(allowed.max_threadgroup_memory_bytes))

    for op_name, cfg in overrides.items():
        if not isinstance(cfg, dict):
            continue
        threadgroup = _as_int(cfg.get("threadgroup"))
        tile = cfg.get("tile")

        if threadgroup is not None:
            if threadgroup <= 0:
                reasons.append(f"threadgroup_non_positive:{op_name}={threadgroup}")
            if threadgroup > max_threads:
                reasons.append(f"threadgroup_too_large:{op_name}={threadgroup}")
            if threadgroup < tew:
                reasons.append(f"threadgroup_too_small:{op_name}={threadgroup}")
            if threadgroup % tew != 0:
                reasons.append(f"threadgroup_not_simd_aligned:{op_name}={threadgroup}:tew={tew}")

        tile_x = tile_y = None
        if isinstance(tile, list) and len(tile) == 2:
            tile_x = _as_int(tile[0])
            tile_y = _as_int(tile[1])
            if tile_x is None or tile_y is None or tile_x <= 0 or tile_y <= 0:
                reasons.append(f"invalid_tile:{op_name}={tile}")
            else:
                threads_est = tile_x * tile_y
                if threads_est > max_threads:
                    reasons.append(f"tile_threads_too_large:{op_name}={threads_est}")
                if threads_est % tew != 0:
                    reasons.append(f"tile_threads_not_simd_aligned:{op_name}={threads_est}:tew={tew}")
                # Conservative shared-memory proxy for fused reductions.
                shared_bytes_est = tile_x * tile_y * 4 * 2
                details[f"{op_name}_shared_bytes_est"] = shared_bytes_est
                if shared_bytes_est > max_tg_mem:
                    reasons.append(f"tile_shared_mem_too_large:{op_name}={shared_bytes_est}")

        vector_width = _as_int(cfg.get("vector_width"))
        k_dim = _as_int(cfg.get("k_dim"))
        has_tail_path = bool(cfg.get("has_tail_path", False))
        if vector_width is not None and k_dim is not None and vector_width > 0:
            if (k_dim % vector_width) != 0 and not has_tail_path:
                reasons.append(
                    f"vector_divisibility_requires_tail_path:{op_name}:k={k_dim}:vw={vector_width}"
                )

    if reasons:
        return FeasibilityCheckRecord(
            attempted=True,
            success=False,
            classification="static_feasibility_reject",
            reasons=reasons,
            limits=limits,
            details=details,
        )

    return FeasibilityCheckRecord(
        attempted=True,
        success=True,
        classification="feasible",
        reasons=[],
        limits=limits,
        details=details,
    )

