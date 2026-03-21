"""
src/optimizer/components/hardware/profiler.py
Uses pynvml, pycuda, and torch to analyze GPU diagnostic statistics and architecture information.  
"""
import glob
import hashlib
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
from src.optimizer.benchmarking.harness import (
    DEFAULT_TIMED_RUNS,
    DEFAULT_WARMUP_RUNS,
    benchmark_entry_calls,
    summarize_entry_results,
)
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


_CRITICAL_GPU_FIELDS: dict = {
    "compute_capability": lambda v: v != "0.0",
    "num_sms": lambda v: v > 0,
    "registers_per_block": lambda v: v > 0,
}


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
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", cc)

    missing = [
        f for f, ok in _CRITICAL_GPU_FIELDS.items()
        if not ok(getattr(gpu_spec, f))
    ]
    if missing:
        import sys
        print(
            f"[KernelForge] WARNING: GPU detection returned zeroed fields: {missing}. "
            "LLM optimization prompts will use degraded hardware context "
            "(wrong arch advice, invalid occupancy estimates). "
            "Check pycuda/NVML availability if this is unexpected. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

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

    cuda_source = (kernel_path / "kernel.cu").read_text()

    # Invalidate cached .so if kernel source has changed since it was built.
    source_hash = hashlib.md5(cuda_source.encode()).hexdigest()[:16]
    hash_file = kernel_path / ".source_hash"
    so_files = list(kernel_path.glob("*.so"))
    if so_files and hash_file.exists() and hash_file.read_text().strip() != source_hash:
        for so in so_files:
            so.unlink(missing_ok=True)
        so_files = []

    # Extract function signature from CUDA code (same as in validator)
    import re
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", cuda_source)
    if not match:
        raise ValueError(
            "Could not find 'launch' function signature in kernel.cu")

    cpp_source = match.group(1) + ";"

    if not so_files:
        # No .so found — compile now (covers baseline and cases where the
        # pre-compiled .so lives in a different cache directory).
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
    hash_file.write_text(source_hash)
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


def get_input_files(io_dir: Path, selected_files: list[Path] | list[str] | None = None) -> list:
    """Retrieves list of all input file paths for a given profiled pytorch op"""
    if selected_files:
        pt_files = [str(Path(item)) for item in selected_files]
    else:
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


def load_batch(pt_files: list) -> list[tuple[str, list[any], dict[str, any]]]:
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

            inputs.append((Path(pt_file).name, args, kwargs))

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
    selected_files = paths.get("entry_files")
    all_files = get_input_files(input_dir, selected_files)
    BATCH_SIZE = settings.batch_size
    entry_results = []
    timing_errors = []

    # We profile everything using one profiler context
    # UPDATE: Removed global profiler context as it accumulates too much RAM (OOM on batch 4)
    # The consumer (optimize_ops.py) does not use the chrome trace 'prof' object, so we return None.
    prof = None

    for i in range(0, len(all_files), BATCH_SIZE):
        batch_files = all_files[i: i + BATCH_SIZE]
        inputs = load_batch(batch_files)

        entry_calls = []
        for entry_file, args, kwargs in inputs:
            try:
                module.launch(*args, **kwargs)
            except TypeError:
                module.launch(*args)
            _sync_device(target_device)

            def invoke(bound_args=args, bound_kwargs=kwargs):
                try:
                    return module.launch(*bound_args, **bound_kwargs)
                except TypeError:
                    return module.launch(*bound_args)

            entry_calls.append((entry_file, invoke))

        batch_stats = benchmark_entry_calls(
            entry_calls,
            device=target_device,
            warmup_runs=DEFAULT_WARMUP_RUNS,
            timed_runs=DEFAULT_TIMED_RUNS,
        )
        entry_results.extend(batch_stats.get("entry_results") or [])
        timing_errors.extend(batch_stats.get("errors") or [])

        # Profile run (for detailed metrics)
        for _, args, kwargs in inputs:
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

    stats = summarize_entry_results(
        entry_results,
        errors=timing_errors,
        device=target_device,
        warmup_runs=DEFAULT_WARMUP_RUNS,
        timed_runs=DEFAULT_TIMED_RUNS,
    )

    # Compare with baseline if provided
    if previous_stats:
        prev_mean = previous_stats.get('mean_time_ms') if isinstance(previous_stats, dict) else None
        curr_mean = stats.get('mean_time_ms')
        if prev_mean and curr_mean:
            speedup = prev_mean / curr_mean
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
