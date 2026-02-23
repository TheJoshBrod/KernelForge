"""Unified profiling orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.optimizer.core.types import GPUSpecs

from .cache import get_cache
from .collector_registry import get_collectors
from .schemas import GPUProfilePayload, GPURecord
from .utils import utc_now_iso

FAST_TTL_SECONDS = 10
DEEP_TTL_SECONDS = 90


def _sanitize_mode(mode: str) -> str:
    return "deep" if str(mode).lower() == "deep" else "fast"


def _record_key(rec: GPURecord) -> str:
    if rec.uuid:
        return rec.uuid
    return f"{rec.vendor}:{rec.backend_hint}:{rec.device_id}:{rec.name}"


def _dedupe(records: List[GPURecord]) -> List[GPURecord]:
    seen: Dict[str, GPURecord] = {}
    for rec in records:
        key = _record_key(rec)
        if key not in seen:
            seen[key] = rec
            continue
        prev = seen[key]
        # Prefer richer record (more non-null fields)
        prev_score = sum(1 for v in prev.model_dump().values() if v not in (None, "", []))
        rec_score = sum(1 for v in rec.model_dump().values() if v not in (None, "", []))
        if rec_score > prev_score:
            seen[key] = rec
    return list(seen.values())


def _run_collectors(mode: str) -> Tuple[List[GPURecord], List[str]]:
    all_records: List[GPURecord] = []
    errors: List[str] = []
    for collector in get_collectors():
        try:
            records, collector_errors = collector(mode)
            all_records.extend(records)
            errors.extend(collector_errors)
        except Exception as exc:
            errors.append(f"{collector.__module__}: {exc}")
    return _dedupe(all_records), errors


def get_profile(mode: str = "fast", use_cache: bool = True) -> Dict[str, Any]:
    profile_mode = _sanitize_mode(mode)
    ttl = DEEP_TTL_SECONDS if profile_mode == "deep" else FAST_TTL_SECONDS
    cache_key = f"gpu-profile:{profile_mode}"
    cache = get_cache()

    if use_cache:
        cached = cache.get(cache_key, ttl)
        if cached is not None:
            return cached

    records, errors = _run_collectors(profile_mode)
    payload = GPUProfilePayload(
        available=bool(records),
        source="local",
        gpus=records,
        timestamp=utc_now_iso(),
        stale=False,
        errors=errors,
    ).model_dump()

    cache.set(cache_key, payload)
    return payload


def _memory_label(memory_total_mb: Any) -> str:
    try:
        mb = float(memory_total_mb)
        if mb <= 0:
            return "Unknown"
        return f"{mb / 1024.0:.1f} GB"
    except Exception:
        return "Unknown"


def _safe_cuda_version() -> str:
    try:
        import torch

        return str(torch.version.cuda or "")
    except Exception:
        return ""


def _safe_mps_built() -> bool:
    try:
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        return bool(mps_backend and mps_backend.is_built())
    except Exception:
        return False


def get_frontend_payload(mode: str = "fast", use_cache: bool = True) -> Dict[str, Any]:
    payload = get_profile(mode=mode, use_cache=use_cache)
    gpus_raw = payload.get("gpus", [])

    gpus: List[Dict[str, Any]] = []
    for item in gpus_raw:
        gpu = dict(item)
        cc = gpu.get("compute_capability") or gpu.get("architecture") or "unknown"
        gpus.append(
            {
                # Legacy fields expected by frontend pages
                "name": gpu.get("name", "Unknown GPU"),
                "connection": gpu.get("source", "local"),
                "memory": _memory_label(gpu.get("memory_total_mb")),
                "compute_capability": str(cc),
                "sm_count": gpu.get("num_sms", "N/A"),
                # Canonical fields for new UI logic
                "vendor": gpu.get("vendor", "unknown"),
                "backend_hint": gpu.get("backend_hint", "cpu"),
                "device_id": gpu.get("device_id", "0"),
                "uuid": gpu.get("uuid"),
                "source": gpu.get("source", "local"),
                "memory_total_mb": gpu.get("memory_total_mb"),
                "memory_used_mb": gpu.get("memory_used_mb"),
                "utilization_percent": gpu.get("utilization_percent"),
                "temperature_c": gpu.get("temperature_c"),
                "power_watts": gpu.get("power_watts"),
                "clock_mhz": gpu.get("clock_mhz"),
                "driver_version": gpu.get("driver_version"),
                "error": gpu.get("error"),
            }
        )

    cuda_available = any(g.get("backend_hint") in {"cuda", "rocm"} for g in gpus)
    mps_available = any(g.get("backend_hint") == "metal" for g in gpus)
    xpu_available = any(g.get("backend_hint") == "xpu" for g in gpus)

    preferred_target = "cpu"
    if mps_available:
        preferred_target = "mps"
    elif cuda_available:
        preferred_target = "cuda"
    elif xpu_available:
        preferred_target = "xpu"

    return {
        "available": bool(gpus),
        "source": payload.get("source", "local"),
        "gpus": gpus,
        "timestamp": payload.get("timestamp"),
        "stale": payload.get("stale", False),
        "errors": payload.get("errors", []),
        # Backward-compatible fields
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "xpu_available": xpu_available,
        "device_count": len(gpus),
        "device_name": gpus[0]["name"] if gpus else "",
        "cuda_version": _safe_cuda_version() if cuda_available else "",
        "mps_built": _safe_mps_built(),
        # New convenience field
        "preferred_target": preferred_target,
    }


def get_device_specs(device_index: int = 0, mode: str = "fast") -> GPUSpecs:
    payload = get_profile(mode=mode, use_cache=True)
    gpus = payload.get("gpus", [])
    if not gpus:
        return GPUSpecs()

    index = min(max(device_index, 0), len(gpus) - 1)
    gpu = gpus[index]

    vendor = str(gpu.get("vendor", "unknown"))
    compute_capability = str(gpu.get("compute_capability") or gpu.get("architecture") or "0.0")
    total_mb = gpu.get("memory_total_mb")
    total_gb = float(total_mb) / 1024.0 if total_mb not in (None, 0, "0") else 0.0
    warp_size = int(gpu.get("warp_size") or (64 if vendor == "amd" else 32))
    num_sms = int(gpu.get("num_sms") or 0)

    tensor_cores_available = False
    if vendor == "nvidia":
        try:
            tensor_cores_available = int(float(compute_capability)) >= 7
        except Exception:
            tensor_cores_available = False
    elif vendor == "amd":
        cc_l = compute_capability.lower()
        tensor_cores_available = ("gfx9" in cc_l) or ("gfx11" in cc_l) or ("cdna" in cc_l)

    max_threads_per_block = int(gpu.get("max_threads_per_block") or 1024)
    max_threads_per_sm = int(gpu.get("max_threads_per_sm") or (num_sms * 2048 if num_sms else 0))
    if vendor == "intel":
        max_threads_per_block = min(max_threads_per_block, 512)

    # Hardware detail fields (from collector or fallback to 0)
    regs_per_block = int(gpu.get("regs_per_block") or 0)
    regs_per_sm = int(gpu.get("regs_per_sm") or 0)
    shared_mem_per_block_bytes = int(gpu.get("shared_mem_per_block_bytes") or 0)
    shared_mem_per_sm_bytes = int(gpu.get("shared_mem_per_sm_bytes") or 0)
    l2_cache_bytes = int(gpu.get("l2_cache_bytes") or 0)
    mem_bus_width_bits = int(gpu.get("mem_bus_width_bits") or 0)
    mem_clock_mhz = int(gpu.get("mem_clock_mhz") or gpu.get("clock_mhz") or 0)

    # Compute peak memory bandwidth: 2 * mem_clock_MHz * bus_width_bits / 8 / 1000 = GB/s
    peak_bw = 0.0
    if mem_clock_mhz and mem_bus_width_bits:
        peak_bw = 2.0 * mem_clock_mhz * mem_bus_width_bits / 8.0 / 1000.0

    return GPUSpecs(
        gpu_name=str(gpu.get("name", "Unknown GPU")),
        vendor=vendor,
        backend_hint=str(gpu.get("backend_hint", "cpu")),
        device_id=str(gpu.get("device_id", index)),
        uuid=gpu.get("uuid"),
        source=str(gpu.get("source", "local")),
        compute_capability=compute_capability,
        total_memory_gb=total_gb,
        sm_clock_mhz=int(gpu.get("clock_mhz") or 0),
        mem_clock_mhz=mem_clock_mhz,
        power_limit_watts=gpu.get("power_watts"),
        num_sms=num_sms,
        warp_size=warp_size,
        max_threads_per_block=max_threads_per_block,
        max_threads_per_sm=max_threads_per_sm,
        max_blocks_per_sm="unknown",
        registers_per_sm=regs_per_sm,
        registers_per_block=regs_per_block,
        shared_mem_per_sm_kb=shared_mem_per_sm_bytes // 1024 if shared_mem_per_sm_bytes else 0,
        shared_mem_per_block_kb=shared_mem_per_block_bytes // 1024 if shared_mem_per_block_bytes else 0,
        l2_cache_kb=l2_cache_bytes // 1024 if l2_cache_bytes else 0,
        memory_bus_width_bits=mem_bus_width_bits,
        peak_memory_bandwidth_gbps=round(peak_bw, 1),
        warps_per_sm=(max_threads_per_sm // warp_size) if warp_size and max_threads_per_sm else 0,
        tensor_cores_available=tensor_cores_available,
        memory_total_mb=gpu.get("memory_total_mb"),
        memory_used_mb=gpu.get("memory_used_mb"),
        utilization_percent=gpu.get("utilization_percent"),
        temperature_c=gpu.get("temperature_c"),
        power_watts=gpu.get("power_watts"),
        clock_mhz=gpu.get("clock_mhz"),
        driver_version=gpu.get("driver_version"),
        error=gpu.get("error"),
    )

