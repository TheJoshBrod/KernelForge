"""NVIDIA collector."""

from __future__ import annotations

from typing import List, Tuple

from ..schemas import GPURecord
from ..utils import command_exists, is_rocm_runtime, run_command, to_float, to_int


def _is_nvidia_name(name: str) -> bool:
    value = (name or "").lower()
    return any(
        token in value
        for token in (
            "nvidia",
            "geforce",
            "tesla",
            "quadro",
            "rtx",
            "gtx",
            "a100",
            "h100",
            "l40",
            "t4",
        )
    )


def collect(mode: str = "fast") -> Tuple[List[GPURecord], List[str]]:
    records: List[GPURecord] = []
    errors: List[str] = []
    torch = None
    try:
        import torch as _torch

        torch = _torch
    except Exception:
        torch = None

    if torch is not None and not is_rocm_runtime():
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            errors.append(f"torch.cuda availability probe failed: {exc}")

        if cuda_available:
            try:
                count = torch.cuda.device_count()
            except Exception as exc:
                count = 0
                errors.append(f"torch.cuda probe failed: {exc}")

            for idx in range(count):
                try:
                    name = torch.cuda.get_device_name(idx)
                    if not _is_nvidia_name(name):
                        continue
                    props = torch.cuda.get_device_properties(idx)
                    major, minor = torch.cuda.get_device_capability(idx)
                    total_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
                    total_mb = int(total_bytes / (1024 * 1024)) if total_bytes else None
                    num_sms = int(getattr(props, "multi_processor_count", 0) or 0)
                    warp_size = int(getattr(props, "warp_size", 32) or 32)
                    regs_per_sm = int(getattr(props, "regs_per_multiprocessor", 0) or 0) or None
                    max_threads_per_sm = int(
                        getattr(props, "max_threads_per_multi_processor", 0) or 0
                    ) or None
                    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
                    l2_cache_kb = int(l2_bytes / 1024) if l2_bytes else None

                    used_mb = None
                    try:
                        with torch.cuda.device(idx):
                            free_b, total_b = torch.cuda.mem_get_info()
                        used_mb = int((total_b - free_b) / (1024 * 1024))
                    except Exception:
                        pass

                    records.append(
                        GPURecord(
                            vendor="nvidia",
                            backend_hint="cuda",
                            name=name,
                            device_id=str(idx),
                            source="local",
                            compute_capability=f"{major}.{minor}",
                            memory_total_mb=total_mb,
                            memory_used_mb=used_mb,
                            num_sms=num_sms,
                            warp_size=warp_size,
                            regs_per_sm=regs_per_sm,
                            max_threads_per_sm=max_threads_per_sm,
                            l2_cache_kb=l2_cache_kb,
                        )
                    )
                except Exception as exc:
                    errors.append(f"nvidia torch probe failed for index {idx}: {exc}")

    if not command_exists("nvidia-smi"):
        return records, errors

    query = (
        "index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,"
        "power.draw,clocks.sm,driver_version,compute_cap"
    )
    rc, out, err = run_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        timeout=2.0 if mode == "deep" else 1.0,
    )
    if rc != 0:
        if err:
            errors.append(f"nvidia-smi failed: {err}")
        return records, errors

    by_index = {rec.device_id: rec for rec in records}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 11:
            continue
        idx = parts[0]
        rec = by_index.get(idx)
        if not rec:
            name = parts[1]
            if not _is_nvidia_name(name):
                continue
            rec = GPURecord(
                vendor="nvidia",
                backend_hint="cuda",
                name=name,
                device_id=idx or str(len(records)),
                source="local",
            )
            records.append(rec)
            by_index[rec.device_id] = rec

        rec.uuid = parts[2] or rec.uuid
        rec.memory_total_mb = to_int(parts[3], rec.memory_total_mb)
        rec.memory_used_mb = to_int(parts[4], rec.memory_used_mb)
        rec.utilization_percent = to_float(parts[5], rec.utilization_percent)
        rec.temperature_c = to_float(parts[6], rec.temperature_c)
        rec.power_watts = to_float(parts[7], rec.power_watts)
        rec.clock_mhz = to_int(parts[8], rec.clock_mhz)
        rec.driver_version = parts[9] or rec.driver_version
        rec.compute_capability = parts[10] or rec.compute_capability

    return records, errors
