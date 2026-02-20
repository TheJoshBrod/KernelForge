"""AMD ROCm collector."""

from __future__ import annotations

import json
from typing import List, Tuple

from ..schemas import GPURecord
from ..utils import command_exists, is_rocm_runtime, run_command, to_float, to_int


def _is_amd_name(name: str) -> bool:
    value = (name or "").lower()
    return any(token in value for token in ("amd", "radeon", "instinct", "mi", "gfx"))


def collect(mode: str = "fast") -> Tuple[List[GPURecord], List[str]]:
    records: List[GPURecord] = []
    errors: List[str] = []

    try:
        import torch
    except Exception:
        return records, errors

    if not torch.cuda.is_available():
        return records, errors

    rocm = is_rocm_runtime()
    try:
        count = torch.cuda.device_count()
    except Exception as exc:
        return records, [f"torch.cuda probe failed: {exc}"]

    for idx in range(count):
        try:
            name = torch.cuda.get_device_name(idx)
            if not rocm and not _is_amd_name(name):
                continue

            props = torch.cuda.get_device_properties(idx)
            total_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
            total_mb = int(total_bytes / (1024 * 1024)) if total_bytes else None
            num_sms = int(getattr(props, "multi_processor_count", 0) or 0)
            arch = getattr(props, "gcnArchName", None)
            if not arch:
                try:
                    cc_major, cc_minor = torch.cuda.get_device_capability(idx)
                    arch = f"{cc_major}.{cc_minor}"
                except Exception:
                    arch = "rocm"

            used_mb = None
            try:
                with torch.cuda.device(idx):
                    free_b, total_b = torch.cuda.mem_get_info()
                used_mb = int((total_b - free_b) / (1024 * 1024))
            except Exception:
                pass

            records.append(
                GPURecord(
                    vendor="amd",
                    backend_hint="rocm",
                    name=name,
                    device_id=str(idx),
                    source="local",
                    compute_capability=str(arch),
                    architecture=str(arch),
                    memory_total_mb=total_mb,
                    memory_used_mb=used_mb,
                    num_sms=num_sms,
                    warp_size=64,
                )
            )
        except Exception as exc:
            errors.append(f"amd torch probe failed for index {idx}: {exc}")

    if mode != "deep" or (not records) or (not command_exists("rocm-smi")):
        return records, errors

    rc, out, err = run_command(
        ["rocm-smi", "--showuse", "--showtemp", "--showpower", "--json"],
        timeout=2.5,
    )
    if rc != 0:
        if err:
            errors.append(f"rocm-smi failed: {err}")
        return records, errors

    try:
        data = json.loads(out)
    except Exception as exc:
        errors.append(f"rocm-smi JSON parse failed: {exc}")
        return records, errors

    for key, card in data.items():
        if not isinstance(card, dict):
            continue
        idx = key.replace("card", "").strip()
        rec = None
        for item in records:
            if item.device_id == idx:
                rec = item
                break
        if rec is None:
            continue

        util = card.get("GPU use (%)") or card.get("GPU use")
        temp = card.get("Temperature (Sensor edge) (C)") or card.get("Temperature (Sensor edge)")
        power = card.get("Average Graphics Package Power (W)") or card.get("Average Graphics Package Power")

        rec.utilization_percent = to_float(util, rec.utilization_percent)
        rec.temperature_c = to_float(temp, rec.temperature_c)
        rec.power_watts = to_float(power, rec.power_watts)

        used_bytes = to_int(card.get("VRAM Total Used Memory (B)"))
        if used_bytes is not None:
            rec.memory_used_mb = int(used_bytes / (1024 * 1024))

    return records, errors
