from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from dataclasses import asdict
from typing import Any

from .types import DeviceProfile


def _run_text(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return ""
    return ""


def _extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_int_any(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _maybe_call(value: Any) -> Any:
    try:
        return value() if callable(value) else value
    except Exception:
        return value


def _probe_metal_limits() -> dict[str, Any]:
    """
    Probe Metal runtime limits when PyObjC Metal bindings are available.
    Falls back silently when unavailable.
    """
    try:
        import Metal  # type: ignore
    except Exception:
        return {"source": "unavailable"}

    try:
        create_fn = getattr(Metal, "MTLCreateSystemDefaultDevice", None)
        device = create_fn() if callable(create_fn) else None
    except Exception:
        device = None
    if device is None:
        return {"source": "unavailable"}

    thread_execution_width: int | None = None
    max_threads_per_threadgroup: int | None = None
    max_threadgroup_memory_bytes: int | None = None

    try:
        tew_raw = _maybe_call(getattr(device, "threadExecutionWidth", None))
        thread_execution_width = _to_int_any(tew_raw)
    except Exception:
        thread_execution_width = None

    try:
        tg_raw = _maybe_call(getattr(device, "maxThreadsPerThreadgroup", None))
        if tg_raw is not None:
            width = _to_int_any(_maybe_call(getattr(tg_raw, "width", None)))
            height = _to_int_any(_maybe_call(getattr(tg_raw, "height", None)))
            depth = _to_int_any(_maybe_call(getattr(tg_raw, "depth", None)))
            if width is not None and height is not None and depth is not None:
                max_threads_per_threadgroup = max(1, width) * max(1, height) * max(1, depth)
            else:
                max_threads_per_threadgroup = _to_int_any(tg_raw)
    except Exception:
        max_threads_per_threadgroup = None

    try:
        tg_mem_raw = _maybe_call(getattr(device, "maxThreadgroupMemoryLength", None))
        max_threadgroup_memory_bytes = _to_int_any(tg_mem_raw)
    except Exception:
        max_threadgroup_memory_bytes = None

    return {
        "source": "pyobjc_metal",
        "thread_execution_width": thread_execution_width,
        "max_threads_per_threadgroup": max_threads_per_threadgroup,
        "max_threadgroup_memory_bytes": max_threadgroup_memory_bytes,
    }


def probe_device() -> DeviceProfile:
    system = platform.system().lower()
    arch = platform.machine().lower()
    macos_version = platform.mac_ver()[0] or ""

    hardware_text = _run_text(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])
    sw_text = _run_text(["sw_vers"])

    if not macos_version:
        macos_version = _extract_first(r"ProductVersion:\s*([^\n]+)", sw_text)

    chip = _extract_first(r"^\s*Chip:\s*([^\n]+)", hardware_text)
    gpu_cores = _to_int(_extract_first(r"Total Number of Cores:\s*(\d+)\s*$", hardware_text))
    cpu_cores = _to_int(
        _extract_first(r"Total Number of Cores:\s*(\d+)\s*\([^)]+\)", hardware_text)
    )
    if cpu_cores is None:
        # fallback when display section appears before hardware section
        cpu_cores = _to_int(_extract_first(r"\bTotal Number of Cores:\s*(\d+)", hardware_text))

    memory_gb = _to_float(_extract_first(r"Memory:\s*([\d.]+)\s*GB", hardware_text))
    metal_line = _extract_first(r"Metal:\s*([^\n]+)", hardware_text)
    if not metal_line:
        metal_line = _extract_first(r"Metal Support:\s*([^\n]+)", hardware_text)
    metal_text = metal_line.lower()
    metal_supported = ("supported" in metal_text) or ("metal" in metal_text)

    is_apple_silicon = system == "darwin" and arch == "arm64"
    metal_limits = _probe_metal_limits()
    thread_execution_width = _to_int_any(metal_limits.get("thread_execution_width"))
    max_threads_per_threadgroup = _to_int_any(metal_limits.get("max_threads_per_threadgroup"))
    max_threadgroup_memory_bytes = _to_int_any(metal_limits.get("max_threadgroup_memory_bytes"))

    fp_input = "|".join(
        [
            system,
            arch,
            macos_version,
            chip,
            str(gpu_cores),
            str(cpu_cores),
            str(memory_gb),
            metal_line,
            str(thread_execution_width),
            str(max_threads_per_threadgroup),
            str(max_threadgroup_memory_bytes),
        ]
    )
    fingerprint = hashlib.sha256(fp_input.encode("utf-8")).hexdigest()[:16]

    return DeviceProfile(
        platform=system,
        arch=arch,
        macos_version=macos_version,
        is_apple_silicon=is_apple_silicon,
        chip=chip or "Unknown",
        gpu_cores=gpu_cores,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        metal_supported=metal_supported,
        metal_feature_set=metal_line,
        fingerprint=fingerprint,
        metal_thread_execution_width=thread_execution_width,
        metal_max_threads_per_threadgroup=max_threads_per_threadgroup,
        metal_max_threadgroup_memory_bytes=max_threadgroup_memory_bytes,
    )


def probe_device_dict() -> dict:
    return asdict(probe_device())
