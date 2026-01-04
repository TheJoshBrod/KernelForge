import pycuda.driver as cuda
import pycuda.autoinit
from pynvml import *

# ******************
#  HELPER FUNCTIONS
# ******************

def _nvml_safe(fn, default=None):
    try:
        return fn()
    except NVMLError_NotSupported:
        return default
    except NVMLError:
        return default

def _to_str(x):
        return x.decode() if isinstance(x, bytes) else x

# ******************
# PROFILER FUNCTIONS
# ******************


def get_gpu_specs(device_index: int = 0):
    """Retrieves GPU architecture information as context for LLM interpretation

    Args:
        device_index (int, optional): Chooses connected NVIDIA GPU via index. Defaults to 0.

    Returns:
        dict: _description_
    """
    # -------------------------
    # NVML: Physical hardware
    # -------------------------
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)

    power_limit_mw = _nvml_safe(
        lambda: nvmlDeviceGetPowerManagementLimit(handle),
        default=None
    )


    nvml_info = {
        "gpu_name": _to_str(nvmlDeviceGetName(handle)),
        "nvml_architecture": nvmlDeviceGetArchitecture(handle),  # enum, map below
        "total_memory_gb": nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3),
        "sm_clock_mhz": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM),
        "mem_clock_mhz": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM),
        "power_limit_watts": None if power_limit_mw is None else power_limit_mw / 1000,
    }

    nvmlShutdown()

    # -------------------------
    # CUDA Runtime: Execution limits
    # -------------------------
    dev = cuda.Device(device_index)
    attrs = dev.get_attributes()
    cc_major, cc_minor = dev.compute_capability()

    cuda_info = {
        "compute_capability": f"{cc_major}.{cc_minor}",
        "num_sms": attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT],
        "warp_size": attrs[cuda.device_attribute.WARP_SIZE],
        "max_threads_per_block": attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK],
        "max_threads_per_sm": attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR],
        "max_blocks_per_sm": attrs[cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR],
        "registers_per_sm": attrs[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR],
        "registers_per_block": attrs[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK],
        "shared_mem_per_sm_kb": attrs[
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        ] // 1024,
        "shared_mem_per_block_kb": attrs[
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        ] // 1024,
        "l2_cache_kb": attrs[cuda.device_attribute.L2_CACHE_SIZE] // 1024,
        "memory_bus_width_bits": attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH],
    }

    # -------------------------
    # Derived metrics
    # -------------------------
    memory_bandwidth_gbps = (
        nvml_info["mem_clock_mhz"] * 1e6 *
        cuda_info["memory_bus_width_bits"] / 8 * 2
    ) / 1e9 

    derived = {
        "peak_memory_bandwidth_gbps": memory_bandwidth_gbps,
        "warps_per_sm": cuda_info["max_threads_per_sm"] // cuda_info["warp_size"],
        "tensor_cores_available": cc_major >= 7,
    }

    # -------------------------
    # Unified output
    # -------------------------
    gpu_spec = {
        **nvml_info,
        **cuda_info,
        **derived,
    }

    return gpu_spec

def profile_kernel(kernel_path: str, *, device_index: int = 0, baseline: bool = False, previous_stats: dict = None):
    
    pass