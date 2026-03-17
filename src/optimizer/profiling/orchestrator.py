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


def _runtime_backend_flags() -> Dict[str, bool]:
    flags = {
        "cuda": False,
        "rocm": False,
        "mps": False,
        "xpu": False,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        is_rocm = bool(getattr(torch.version, "hip", None))
        if not is_rocm:
            try:
                cfg = torch.__config__.show()
                is_rocm = "USE_ROCM=ON" in str(cfg)
            except Exception:
                is_rocm = False

        if is_rocm:
            flags["rocm"] = cuda_available
        else:
            flags["cuda"] = cuda_available

        mps_backend = getattr(torch.backends, "mps", None)
        flags["mps"] = bool(mps_backend and mps_backend.is_available())

        xpu_backend = getattr(torch, "xpu", None)
        flags["xpu"] = bool(xpu_backend and xpu_backend.is_available())
    except Exception:
        pass
    return flags


def _runtime_usable_for_backend(backend_hint: str, runtime_flags: Dict[str, bool]) -> bool:
    backend = str(backend_hint or "cpu").lower()
    if backend == "cuda":
        return runtime_flags["cuda"]
    if backend == "rocm":
        return runtime_flags["rocm"]
    if backend == "metal":
        return runtime_flags["mps"]
    if backend == "xpu":
        return runtime_flags["xpu"]
    return False


def get_frontend_payload(mode: str = "fast", use_cache: bool = True) -> Dict[str, Any]:
    payload = get_profile(mode=mode, use_cache=use_cache)
    gpus_raw = payload.get("gpus", [])
    runtime_flags = _runtime_backend_flags()

    gpus: List[Dict[str, Any]] = []
    for item in gpus_raw:
        gpu = dict(item)
        cc = gpu.get("compute_capability") or gpu.get("architecture") or "unknown"
        runtime_usable = _runtime_usable_for_backend(gpu.get("backend_hint", "cpu"), runtime_flags)
        runtime_reason = ""
        if not runtime_usable and gpu.get("backend_hint") in {"cuda", "rocm", "metal", "xpu"}:
            runtime_reason = (
                "Hardware detected, but the current packaged runtime does not expose this backend."
            )
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
                # Runtime capability fields
                "runtime_usable": runtime_usable,
                "runtime_reason": runtime_reason,
            }
        )

    cuda_available = runtime_flags["cuda"] or runtime_flags["rocm"]
    mps_available = runtime_flags["mps"]
    xpu_available = runtime_flags["xpu"]

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
        "gpu_present": bool(gpus),
        "runtime_acceleration_available": preferred_target != "cpu",
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

    # Collected directly from torch device properties
    regs_per_sm = int(gpu.get("regs_per_sm") or 0)
    max_threads_per_sm = int(gpu.get("max_threads_per_sm") or 0)
    l2_cache_kb = int(gpu.get("l2_cache_kb") or 0)

    # Fallback: derive max_threads_per_sm from SM count if not collected
    if not max_threads_per_sm and num_sms:
        max_threads_per_sm = num_sms * 2048

    max_threads_per_block = 1024
    if vendor == "intel":
        max_threads_per_block = 512

    # Registers per block == registers per SM on all current NVIDIA architectures
    registers_per_block = regs_per_sm

    # Shared memory per SM and per block by compute capability.
    # These are the hardware defaults (not the configurable maximum).
    # Sources: CUDA Programming Guide, Appendix G.
    _smem_per_sm_kb = {
        "9.0": 228, "8.9": 100, "8.7": 100, "8.6": 100, "8.0": 164,
        "7.5": 64,  "7.0": 96,  "6.1": 96,  "6.0": 64,
        "5.3": 64,  "5.2": 96,  "5.0": 64,
    }
    shared_mem_per_sm_kb = _smem_per_sm_kb.get(compute_capability, 48)
    # Default max per block is 48 KB across all CUDA archs (48 KB is the safe default;
    # higher values require cudaFuncSetAttribute at runtime).
    shared_mem_per_block_kb = min(shared_mem_per_sm_kb, 48)

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
        mem_clock_mhz=int(gpu.get("clock_mhz") or 0),
        power_limit_watts=gpu.get("power_watts"),
        num_sms=num_sms,
        warp_size=warp_size,
        max_threads_per_block=max_threads_per_block,
        max_threads_per_sm=max_threads_per_sm,
        max_blocks_per_sm="unknown",
        registers_per_sm=regs_per_sm,
        registers_per_block=registers_per_block,
        shared_mem_per_sm_kb=shared_mem_per_sm_kb,
        shared_mem_per_block_kb=shared_mem_per_block_kb,
        l2_cache_kb=l2_cache_kb,
        memory_bus_width_bits=0,
        peak_memory_bandwidth_gbps=0.0,
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
