"""Collector registration."""

from __future__ import annotations

from typing import Callable, List, Tuple

from .collectors.amd import collect as collect_amd
from .collectors.intel import collect as collect_intel
from .collectors.metal import collect as collect_metal
from .collectors.nvidia import collect as collect_nvidia
from .schemas import GPURecord

CollectorFn = Callable[[str], Tuple[List[GPURecord], List[str]]]


def get_collectors() -> List[CollectorFn]:
    # Order matters for overlap: nvidia/amd first, then apple/intel.
    return [collect_nvidia, collect_amd, collect_metal, collect_intel]

