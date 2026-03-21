"""
Triton Backend Implementation.
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

from src.optimizer.core.backend import Backend
from src.optimizer.core.types import GPUSpecs
import src.optimizer.backends.triton.profiler as profiler
import src.optimizer.backends.triton.prompts as prompts
import src.optimizer.backends.triton.verifier as verifier


class TritonBackend(Backend):
    """
    Backend for OpenAI Triton (supports NVIDIA CUDA and AMD ROCm GPUs).
    """

    @property
    def kernel_extension(self) -> str:
        return ".py"

    def get_device_specs(self, device_index: int = 0, ssh_config: dict = None) -> GPUSpecs:
        return profiler.get_gpu_specs(device_index, ssh_config)

    def get_sys_prompt(self) -> str:
        return prompts.get_sys_prompt()

    def generate_optimization_prompt(self,
                                   gpu_specs: GPUSpecs,
                                   kernel_code: str,
                                   improvement_log: List[dict],
                                   ancestor_codes: Optional[List[Tuple[int, str]]] = None,
                                   failed_siblings: Optional[List[str]] = None) -> str:

        return prompts.generate_gpu_optimization_prompt(
            gpu_info=gpu_specs.model_dump(),
            kernel_code=kernel_code,
            improvement_log=improvement_log,
            ancestor_codes=ancestor_codes,
            failed_siblings=failed_siblings
        )

    def validate_kernel(self,
                       code: str,
                       paths: Dict[str, Path],
                       ssh_config: dict = None) -> Tuple[bool, str]:
        if ssh_config:
            return verifier.validate_remote_kernel(ssh_config, code, paths)
        return verifier.validate_kernel(code, paths)

    def profile_kernel(self,
                      paths: Dict[str, Path],
                      baseline: bool = False,
                      device_index: int = 0,
                      previous_stats: dict = None,
                      ssh_config: dict = None) -> dict:
        if ssh_config:
            stats, _ = profiler.profile_remote_kernel(ssh_config, paths, baseline=baseline)
            return stats

        stats, _ = profiler.profile_kernel(
            paths,
            baseline=baseline,
            device_index=device_index,
            previous_stats=previous_stats
        )
        return stats
