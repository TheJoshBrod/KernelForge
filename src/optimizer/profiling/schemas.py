"""Pydantic schemas for profiling payloads."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GPURecord(BaseModel):
    vendor: str = "unknown"
    backend_hint: str = "cpu"  # cuda | rocm | xpu | metal | cpu
    name: str = "Unknown GPU"
    device_id: str = "0"
    uuid: Optional[str] = None
    source: str = "local"  # local | remote

    compute_capability: Optional[str] = None
    architecture: Optional[str] = None
    driver_version: Optional[str] = None

    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    utilization_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None
    clock_mhz: Optional[int] = None

    num_sms: Optional[int] = None
    warp_size: Optional[int] = None
    regs_per_sm: Optional[int] = None
    max_threads_per_sm: Optional[int] = None
    l2_cache_kb: Optional[int] = None

    error: Optional[str] = None


class GPUProfilePayload(BaseModel):
    available: bool = False
    source: str = "local"  # local | remote | mixed
    gpus: List[GPURecord] = Field(default_factory=list)
    timestamp: str
    stale: bool = False
    errors: List[str] = Field(default_factory=list)

