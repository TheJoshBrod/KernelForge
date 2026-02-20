"""Intel integrated/XPU collector."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

from ..schemas import GPURecord


def _sysfs_intel_records() -> List[GPURecord]:
    records: List[GPURecord] = []
    vendor_files = sorted(glob.glob("/sys/class/drm/card*/device/vendor"))
    for vf in vendor_files:
        try:
            vendor = Path(vf).read_text().strip().lower()
            if vendor != "0x8086":
                continue
            card_path = Path(vf).parent.parent
            card_name = card_path.name
            device_file = card_path / "device" / "device"
            device_id = device_file.read_text().strip() if device_file.exists() else card_name
            records.append(
                GPURecord(
                    vendor="intel",
                    backend_hint="xpu",
                    name=f"Intel GPU {device_id}",
                    device_id=card_name.replace("card", ""),
                    source="local",
                    architecture="integrated",
                )
            )
        except Exception:
            continue
    return records


def collect(mode: str = "fast") -> Tuple[List[GPURecord], List[str]]:
    del mode
    records: List[GPURecord] = []
    errors: List[str] = []

    try:
        import torch
    except Exception:
        return _sysfs_intel_records(), errors

    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            if xpu.is_available():
                count = xpu.device_count()
                for idx in range(count):
                    try:
                        name = xpu.get_device_name(idx) if hasattr(xpu, "get_device_name") else f"Intel XPU {idx}"
                    except Exception:
                        name = f"Intel XPU {idx}"
                    records.append(
                        GPURecord(
                            vendor="intel",
                            backend_hint="xpu",
                            name=name,
                            device_id=str(idx),
                            source="local",
                            architecture="xpu",
                        )
                    )
        except Exception as exc:
            errors.append(f"torch.xpu probe failed: {exc}")

    if records:
        return records, errors
    return _sysfs_intel_records(), errors

