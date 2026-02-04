"""
src/optimizer/components/hardware/profiler.py
Uses pynvml, pycuda, and torch to analyze GPU diagnostic statistics and architecture information.  
"""
import glob
import os
import time
from pathlib import Path

import numpy as np
try:
    import pycuda.driver as cuda
except Exception:
    cuda = None
import torch
try:
    from pynvml import *  # type: ignore
    from pynvml import NVMLError_NotSupported  # type: ignore
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

    class NVMLError(Exception):
        pass

    class NVMLError_NotSupported(Exception):
        pass
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
# GPU Architecture Info
# Kernel Profiling


from src.optimizer.config.settings import settings
from src.optimizer.core.types import GPUSpecs

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


def get_gpu_specs(device_index: int = 0) -> GPUSpecs:
    """Retrieves GPU architecture information as context for LLM interpretation

    Args:
        device_index (int, optional): Chooses connected NVIDIA GPU via index. Defaults to 0.

    Returns:
        GPUSpecs: Pydantic model containing GPU specs
    """
    # -------------------------
    # NVML: Physical hardware
    # -------------------------
    if cuda is None or not NVML_AVAILABLE:
        return {
            "gpu_name": "Unknown",
            "compute_capability": "unknown",
            "cuda_available": False,
            "notes": "CUDA/NVML not available (likely non-CUDA device)",
        }

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)

    power_limit_mw = _nvml_safe(
        lambda: nvmlDeviceGetPowerManagementLimit(handle),
        default=None
    )

    nvml_info = {
        "gpu_name": _to_str(nvmlDeviceGetName(handle)),
        # enum, map below
        "nvml_architecture": nvmlDeviceGetArchitecture(handle),
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
        "max_blocks_per_sm": attrs.get(
            getattr(cuda.device_attribute,
                    'MAX_BLOCKS_PER_MULTIPROCESSOR', None),
            "unknown"
        ) if hasattr(cuda.device_attribute, 'MAX_BLOCKS_PER_MULTIPROCESSOR') else "unknown",
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
    gpu_spec_data = {
        **nvml_info,
        **cuda_info,
        **derived,
    }
    
    gpu_spec = GPUSpecs(**gpu_spec_data)

    # Optimization: Set the CUDA Architecture to the specific device to speed up JIT compilation
    os.environ["TORCH_CUDA_ARCH_LIST"] = gpu_spec.compute_capability

    return gpu_spec


def get_module(kernel_path: Path, baseline: bool):
    """Retrieves a module object from compiled CUDA kernel
    
    Args:
        kernel_path (Path): Directory containing compiled CUDA kernel
        baseline (bool): Is this the performance of the initial correctly generated kernel in `kernels/generated/*`

    Raises:
        FileNotFoundError: If directory does not contain compiled CUDA kernel AND NOT initial baseline (should never happen)

    Returns:
        module: module object to run load_inline CUDA kernel in python
    """

    # Sanity check to see if generated and validated (you're welcome future me.)
    so_files = list(kernel_path.glob("*.so"))

    cuda_source = (kernel_path / "kernel.cu").read_text()

    # Extract function signature from CUDA code (same as in validator)
    import re
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", cuda_source)
    if not match:
        raise ValueError(
            "Could not find 'launch' function signature in kernel.cu")

    cpp_source = match.group(1) + ";"

    if not so_files:
        if not baseline:
            raise FileNotFoundError(
                "Somehow passed correctness validation (generator.py), but lost .so file (GPUprofiler.py)")

        # Baseline case: compile the kernel now
        # Use a consistent module name based on the directory
        module_name = kernel_path.name
    else:
        # Already compiled case: load existing module
        module_name = so_files[0].stem

    target_device = os.environ.get("CGINS_TARGET_DEVICE", "").strip().lower()
    if target_device in {"gpu", "cuda"} or target_device == "":
        module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['launch'],
            build_directory=str(kernel_path),
            verbose=False,
            with_cuda=True
        )
    else:
        module = load_inline(
            name=module_name,
            cpp_sources=cuda_source,
            functions=['launch'],
            build_directory=str(kernel_path),
            verbose=False,
            with_cuda=False
        )
    return module


def normalize_args_kwargs(args: list, kwargs: dict, params: list, defaults: dict) -> tuple[list, dict]:
    if not params:
        return args, kwargs

    normalized = list(args)
    remaining_kwargs = dict(kwargs)

    for i in range(len(normalized), len(params)):
        param_name = params[i]

        if param_name in remaining_kwargs:
            normalized.append(remaining_kwargs.pop(param_name))
        elif param_name in defaults:
            normalized.append(defaults[param_name])
        else:
            break

    return normalized, remaining_kwargs


def get_input_files(io_dir: Path) -> list:
    """Retrieves list of all input file paths for a given profiled pytorch op"""
    pt_files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))

    if not pt_files:
        raise ValueError(f"No entry_*.pt files found in {io_dir}")
    return pt_files


def _target_device() -> str:
    value = os.environ.get("CGINS_TARGET_DEVICE", "").strip().lower()
    if value in {"gpu", "cuda"}:
        return "cuda"
    if value == "mps":
        return "mps"
    if value == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def load_batch(pt_files: list) -> list[tuple[list[any], dict[str, any]]]:
    """Loads a batch of .pt files into GPU memory"""
    inputs = []
    device = _target_device()
    for pt_file in pt_files:
        try:
            entry = torch.load(pt_file, map_location='cpu')

            # Move to target device
            args = [
                (arg.to("mps") if device == "mps" else (arg.cuda() if device == "cuda" else arg.cpu()))
                if isinstance(arg, torch.Tensor) else arg
                for arg in entry['args']
            ]

            kwargs = {
                k: (v.to("mps") if device == "mps" else (v.cuda() if device == "cuda" else v.cpu()))
                if isinstance(v, torch.Tensor) else v
                for k, v in entry['kwargs'].items()
            }

            # Normalize using signature
            if 'signature' in entry:
                sig = entry['signature']
                params = sig.get('params', [])
                defaults = sig.get('defaults', {})

                if params:
                    args, kwargs = normalize_args_kwargs(
                        args, kwargs, params, defaults)

            inputs.append((args, kwargs))

        except Exception as e:
            print(f"Warning: Failed to load {pt_file}: {e}")
            continue

    return inputs


def profile_kernel(paths: dict[str, Path], *, baseline=False, device_index: int = 0, previous_stats: dict = None):

    kernel_path = paths["tmp_dir"]
    module = get_module(kernel_path, baseline)

    input_dir = paths["io_dir"]
    target_device = _target_device()
    if target_device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device_index)

    # Batching logic to prevent OOM
    all_files = get_input_files(input_dir)
    BATCH_SIZE = settings.batch_size
    timings = []

    # We profile everything using one profiler context
    # UPDATE: Removed global profiler context as it accumulates too much RAM (OOM on batch 4)
    # The consumer (optimize_ops.py) does not use the chrome trace 'prof' object, so we return None.
    prof = None

    for i in range(0, len(all_files), BATCH_SIZE):
        batch_files = all_files[i: i + BATCH_SIZE]
        inputs = load_batch(batch_files)

        # Warmup (only for this batch)
        for args, kwargs in inputs:
            try:
                module.launch(*args, **kwargs)
            except TypeError:
                module.launch(*args)
        _sync_device(target_device)

        # Measure
        if target_device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for args, kwargs in inputs:
                start.record()
                for _ in range(10):
                    try:
                        module.launch(*args, **kwargs)
                    except TypeError:
                        module.launch(*args)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end) / 10
                timings.append(elapsed_ms)
        else:
            for args, kwargs in inputs:
                start_time = time.perf_counter()
                for _ in range(10):
                    try:
                        module.launch(*args, **kwargs)
                    except TypeError:
                        module.launch(*args)
                _sync_device(target_device)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / 10
                timings.append(elapsed_ms)

        # Profile run (for detailed metrics)
        for args, kwargs in inputs:
            try:
                module.launch(*args, **kwargs)
            except TypeError:
                module.launch(*args)
            _sync_device(target_device)

        # Cleanup VRAM
        del inputs
        if target_device == "cuda":
            torch.cuda.empty_cache()
        elif target_device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    stats = {
        'mean_time_ms': float(np.mean(timings)),
        'std_time_ms':  float(np.std(timings)),
        'min_time_ms':  float(np.min(timings)),
        'max_time_ms':  float(np.max(timings)),
    }

    # Compare with baseline if provided
    if previous_stats:
        speedup = previous_stats['mean_time_ms'] / stats['mean_time_ms']
        print(f"Speedup: {speedup:.2f}x")

    return stats, prof
