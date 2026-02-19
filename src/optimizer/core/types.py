import math
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class GPUSpecs(BaseModel):
    # Canonical cross-vendor fields
    gpu_name: str = "Unknown GPU"
    vendor: str = "unknown"
    backend_hint: str = "cpu"
    device_id: str = "0"
    uuid: Optional[str] = None
    source: str = "local"

    # Legacy CUDA-centric fields (kept for compatibility)
    nvml_architecture: int = 0
    total_memory_gb: float = 0.0
    sm_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    power_limit_watts: Optional[float] = None
    compute_capability: str = "0.0"
    num_sms: int = 0
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 0
    max_blocks_per_sm: Any = "unknown"  # Can be "unknown" string or int
    registers_per_sm: int = 0
    registers_per_block: int = 0
    shared_mem_per_sm_kb: int = 0
    shared_mem_per_block_kb: int = 0
    l2_cache_kb: int = 0
    memory_bus_width_bits: int = 0
    peak_memory_bandwidth_gbps: float = 0.0
    warps_per_sm: int = 0
    tensor_cores_available: bool = False

    # Optional telemetry
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    utilization_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None
    clock_mhz: Optional[int] = None
    driver_version: Optional[str] = None
    error: Optional[str] = None

class KernelNode(BaseModel):
    id: int
    parent_id: Optional[int] = Field(default=None, alias="parent")
    children_ids: List[int] = Field(default_factory=list, alias="children")
    visits: int = 1
    value: Optional[float] = None
    best_subtree_value: Optional[float] = None
    code: Optional[str] = None
    improvement_description: Optional[str] = None
    speedup_vs_parent: Optional[float] = None
    
    class Config:
        populate_by_name = True

    def uct_score(self, parent_node: 'KernelNode', C: float = 1.0) -> float:
        """
        Compute UCT score for minimization.
        Lower score = better.
        """
        exploitation = self.best_subtree_value if self.best_subtree_value is not None else self.value
        exploration = C * math.sqrt(math.log(parent_node.visits) / self.visits)

        return exploitation - exploration
