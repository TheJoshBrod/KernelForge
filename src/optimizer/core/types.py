import math
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class GPUSpecs(BaseModel):
    gpu_name: str
    nvml_architecture: int
    total_memory_gb: float
    sm_clock_mhz: int
    mem_clock_mhz: int
    power_limit_watts: Optional[float]
    compute_capability: str
    num_sms: int
    warp_size: int
    max_threads_per_block: int
    max_threads_per_sm: int
    max_blocks_per_sm: Any  # Can be "unknown" string or int
    registers_per_sm: int
    registers_per_block: int
    shared_mem_per_sm_kb: int
    shared_mem_per_block_kb: int
    l2_cache_kb: int
    memory_bus_width_bits: int
    peak_memory_bandwidth_gbps: float
    warps_per_sm: int
    tensor_cores_available: bool

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
