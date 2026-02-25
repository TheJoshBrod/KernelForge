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
from src.optimizer.profiling import get_device_specs as get_profiled_device_specs

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
    """Retrieve GPU specs via the unified profiling orchestrator."""
    import re as _re
    gpu_spec = get_profiled_device_specs(device_index=device_index, mode="deep")

    # Only set TORCH_CUDA_ARCH_LIST when compute_capability is a valid numeric
    # version string like "7.5" or "9.0".  Guard against strings such as
    # "NVIDIA GeForce GTX 1660 Ti" that can leak in when the ROCm-runtime
    # detection misfires.
    cc = str(gpu_spec.compute_capability or "").strip()
    if cc and _re.match(r"^\d+\.\d+$", cc):
        os.environ["TORCH_CUDA_ARCH_LIST"] = cc

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

    target_device = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
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
    value = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
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


# ******************
# REMOTE PROFILER
# ******************

def get_remote_gpu_specs(ssh_config: dict) -> GPUSpecs:
    """
    Retrieves GPU specs from a remote server via SSH.
    """
    from src.optimizer.core.ssh_client import RemoteWorkerClient
    import json
    
    # worker path: src/optimizer/backends/cuda/remote_worker.py
    worker_path = Path(__file__).parent / "remote_worker.py"
    loader_path = Path(__file__).parent / "loader.py"
    
    try:
        worker = RemoteWorkerClient(ssh_config, worker_path, {str(loader_path): "loader.py"})
        
        result = worker.send_task("get_specs")
        worker.close()
        
        if "error" in result:
             raise RuntimeError(f"Remote specs retrieval failed: {result['error']}")
             
        return GPUSpecs(**result)

    except Exception as e:
        print(f"Remote Specs Error: {e}")
        # Return fallback/empty specs or re-raise?
        # Re-raising seems appropriate as we need specs for optimization
        raise e


def profile_remote_kernel(ssh_config: dict, paths: dict[str, Path], baseline: bool = False) -> tuple[dict, any]:
    """
    Profiles a kernel on a remote server using the persistent worker.
    """
    from src.optimizer.core.ssh_client import RemoteWorkerClient, upload_files
    
    try:
        worker_path = Path(__file__).parent / "remote_worker.py"
        loader_path = Path(__file__).parent / "loader.py"
        
        worker = RemoteWorkerClient(ssh_config, worker_path, {str(loader_path): "loader.py"})
        
        # 1. Prepare Code
        kernel_path = paths["tmp_dir"] / "kernel.cu"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel code not found at {kernel_path}")
        
        kernel_code = kernel_path.read_text()
        
        # 2. Upload IO files to shared cache
        io_dir = paths["io_dir"]
        io_files = sorted(list(io_dir.glob("*.pt")))
        file_map = {str(f): f.name for f in io_files}
        
        remote_io_dir = "kforge_workspace/io_cache/" + io_dir.name
        upload_files(worker.client, file_map, remote_io_dir)
        
        # 3. Send profile task
        payload = {
            "code": kernel_code,
            "io_dir": remote_io_dir,
            "batch_size": settings.batch_size
        }
        
        result = worker.send_task("profile", payload)
        worker.close()
        
        if "error" in result:
            print(f"Remote Profiling Error: {result['error']}")
            raise RuntimeError(f"Remote profiling failed: {result['error']}")
            
        return result, None

    except Exception as e:
        print(f"Remote Profiling Exception: {e}")
        raise e
