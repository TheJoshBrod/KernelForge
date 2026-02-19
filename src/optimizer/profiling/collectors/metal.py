"""Apple Metal / MPS collector."""

from __future__ import annotations

import platform
from typing import List, Tuple

from ..schemas import GPURecord
from ..utils import run_command, to_int


def collect(mode: str = "fast") -> Tuple[List[GPURecord], List[str]]:
    del mode
    records: List[GPURecord] = []
    errors: List[str] = []

    if platform.system().lower() != "darwin":
        return records, errors

    try:
        import torch
    except Exception:
        return records, errors

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    mps_built = bool(mps_backend and mps_backend.is_built())

    if not mps_available and not mps_built:
        return records, errors

    total_mb = None
    rc, out, _ = run_command(["sysctl", "-n", "hw.memsize"], timeout=0.8)
    if rc == 0 and out:
        mem_bytes = to_int(out.strip())
        if mem_bytes is not None:
            total_mb = int(mem_bytes / (1024 * 1024))

    name = "Apple Silicon GPU" if mps_available else "Apple GPU (MPS unavailable)"
    rec = GPURecord(
        vendor="apple",
        backend_hint="metal",
        name=name,
        device_id="0",
        source="local",
        architecture=platform.machine(),
        memory_total_mb=total_mb,
    )
    if not mps_available:
        rec.error = "MPS backend is built but unavailable"

    records.append(rec)
    return records, errors

