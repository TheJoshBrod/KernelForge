"""
Abstract base class for optimizer backends.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Optional, List
from src.optimizer.core.types import GPUSpecs

class Backend(ABC):
    """
    Abstract base class for optimization backends (CUDA, Metal, Triton, etc.).
    """

    @abstractmethod
    def get_device_specs(self, device_index: int = 0, ssh_config: dict = None) -> GPUSpecs:
        """
        Retrieves hardware specifications for the target device.
        """
        pass

    @abstractmethod
    def generate_optimization_prompt(self, 
                                   gpu_specs: GPUSpecs, 
                                   current_code: str, 
                                   improvement_log: List[str],
                                   ancestor_codes: Optional[List[Tuple[int, str]]] = None) -> str:
        """
        Generates the LLM prompt for optimizing kernel code.
        """
        pass

    @abstractmethod
    def validate_kernel(self, 
                       code: str, 
                       paths: dict[str, Path], 
                       ssh_config: dict = None) -> Tuple[bool, str]:
        """
        Validates the generated kernel code (compilation + correctness).
        """
        pass

    @abstractmethod
    def profile_kernel(self, 
                      paths: dict[str, Path], 
                      baseline: bool = False, 
                      device_index: int = 0, 
                      previous_stats: dict = None,
                      ssh_config: dict = None) -> dict:
        """
        Profiles the kernel performance.
        """
        pass
