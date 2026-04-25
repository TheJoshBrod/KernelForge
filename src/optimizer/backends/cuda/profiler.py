"""
src/optimizer/components/hardware/profiler.py
Uses pynvml, pycuda, and torch to analyze GPU diagnostic statistics and architecture information.  
"""
import glob
import hashlib
import os
import time
from pathlib import Path

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
from src.optimizer.benchmarking.harness import (
    DEFAULT_TIMED_RUNS,
    DEFAULT_WARMUP_RUNS,
    summarize_entry_results,
)
from src.optimizer.quantized import prepare_tinygemm_linear_launch_args

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


def _canonical_signature(entries: list[dict]) -> dict:
    for entry in entries:
        sig = entry.get("signature", {}) if isinstance(entry, dict) else {}
        if sig and sig.get("params"):
            return sig

    if not entries:
        return {"params": [], "defaults": {}}

    first_args = entries[0].get("args", []) or []
    params = [f"arg{i}" for i in range(len(first_args))]
    seen = set(params)
    for entry in entries:
        kwargs = entry.get("kwargs", {}) or {}
        if not isinstance(kwargs, dict):
            continue
        for key in kwargs:
            if key in seen:
                continue
            params.append(key)
            seen.add(key)
    return {"params": params, "defaults": {}}


def _call_launch(module, args: list, kwargs: dict):
    try:
        return module.launch(*args, **kwargs)
    except TypeError:
        if kwargs:
            return module.launch(*args, *list(kwargs.values()))
        return module.launch(*args)


def get_input_files(io_dir: Path, selected_files: list | None = None) -> list:
    """Retrieves list of all input file paths for a given profiled pytorch op"""
    if selected_files:
        pt_files = []
        seen = set()
        for item in selected_files:
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = Path(io_dir) / candidate
            candidate = candidate.resolve()
            if candidate.exists() and candidate.name.startswith("entry_") and candidate.suffix == ".pt":
                text = str(candidate)
                if text not in seen:
                    pt_files.append(text)
                    seen.add(text)
        pt_files = sorted(pt_files)
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


def _move_to_device(item, device: str):
    if torch.is_tensor(item):
        if device == "mps":
            return item.to("mps")
        if device == "cuda":
            return item.cuda()
        return item.cpu()
    if isinstance(item, (list, tuple)):
        return type(item)(_move_to_device(x, device) for x in item)
    if isinstance(item, dict):
        return {k: _move_to_device(v, device) for k, v in item.items()}
    return item


def _canonical_signature_from_files(pt_files: list) -> dict:
    param_order: list[str] | None = None
    seen: set[str] = set()

    for pt_file in pt_files:
        try:
            entry = torch.load(pt_file, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Warning: Failed to inspect {pt_file}: {e}")
            continue
        try:
            sig = entry.get("signature", {}) if isinstance(entry, dict) else {}
            if sig and sig.get("params"):
                return sig

            if param_order is None:
                first_args = entry.get("args", []) or []
                param_order = [f"arg{i}" for i in range(len(first_args))]
                seen = set(param_order)
            kwargs = entry.get("kwargs", {}) or {}
            if isinstance(kwargs, dict):
                for key in kwargs:
                    if key not in seen:
                        param_order.append(key)
                        seen.add(key)
        finally:
            del entry

    return {"params": param_order or [], "defaults": {}}


def load_batch(pt_files: list, signature: dict | None = None) -> list[tuple[str, list[any], dict[str, any]]]:
    """Loads a batch of .pt files into GPU memory"""
    inputs = []
    device = _target_device()
    signature = signature or _canonical_signature_from_files(pt_files)
    params = signature.get("params", [])
    defaults = signature.get("defaults", {})

    for pt_file in pt_files:
        try:
            entry = torch.load(pt_file, map_location='cpu', weights_only=False)

            args = list(entry.get('args') or [])
            kwargs = dict(entry.get('kwargs') or {})

            # Normalize using signature
            if params:
                args, kwargs = normalize_args_kwargs(
                    args, kwargs, params, defaults)

            function_name = entry.get("function_name") or entry.get("op_name") or entry.get("op")
            special_args = prepare_tinygemm_linear_launch_args(
                function_name,
                args,
                kwargs,
                signature,
                move_to_device=lambda item: _move_to_device(item, device),
            )
            if special_args is not None:
                args = special_args
                kwargs = {}
            else:
                args = [_move_to_device(arg, device) for arg in args]
                kwargs = {k: _move_to_device(v, device) for k, v in kwargs.items()}

            inputs.append((Path(pt_file).name, args, kwargs))
            del entry

        except Exception as e:
            print(f"Warning: Failed to normalize profiler input {pt_file}: {e}")
            continue

    return inputs


def profile_kernel(paths: dict[str, Path], *, baseline=False, device_index: int = 0, previous_stats: dict = None):

    kernel_path = paths["tmp_dir"]
    module = get_module(kernel_path, baseline)

    input_dir = paths["io_dir"]
    target_device = _target_device()
    if target_device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device_index)

    # Batching logic to prevent OOM.  Generated-kernel benchmarking passes a
    # narrowed entry_files list; use a conservative batch there because a single
    # embedding table can be multiple GiB.
    selected_files = paths.get("entry_files")
    all_files = get_input_files(input_dir, selected_files)
    batch_size_raw = os.environ.get("KFORGE_CUDA_PROFILE_BATCH_SIZE", "").strip()
    try:
        BATCH_SIZE = int(batch_size_raw) if batch_size_raw else int(settings.batch_size)
    except Exception:
        BATCH_SIZE = int(settings.batch_size)
    if selected_files:
        selected_batch_raw = os.environ.get("KFORGE_CUDA_PROFILE_SELECTED_BATCH_SIZE", "1").strip()
        try:
            selected_batch = int(selected_batch_raw)
        except Exception:
            selected_batch = 1
        BATCH_SIZE = min(BATCH_SIZE, max(1, selected_batch))
    BATCH_SIZE = max(1, BATCH_SIZE)
    entry_results = []
    timing_errors = []

    # We profile everything using one profiler context
    # UPDATE: Removed global profiler context as it accumulates too much RAM (OOM on batch 4)
    # The consumer (optimize_ops.py) does not use the chrome trace 'prof' object, so we return None.
    prof = None

    for i in range(0, len(all_files), BATCH_SIZE):
        batch_files = all_files[i: i + BATCH_SIZE]
        signature = _canonical_signature_from_files(batch_files)
        inputs = load_batch(batch_files, signature=signature)

        # Warmup (only for this batch, discarded)
        for entry_file, args, kwargs in inputs:
            try:
                for _ in range(DEFAULT_WARMUP_RUNS):
                    _call_launch(module, args, kwargs)
                _sync_device(target_device)

                # Measure
                if target_device == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(DEFAULT_TIMED_RUNS):
                        _call_launch(module, args, kwargs)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end) / DEFAULT_TIMED_RUNS
                else:
                    start_time = time.perf_counter()
                    for _ in range(DEFAULT_TIMED_RUNS):
                        _call_launch(module, args, kwargs)
                    _sync_device(target_device)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / DEFAULT_TIMED_RUNS
                entry_results.append({"entry_file": entry_file, "latency_ms": float(elapsed_ms)})

                # Profile run (for detailed metrics)
                _call_launch(module, args, kwargs)
                _sync_device(target_device)
            except Exception as e:
                timing_errors.append({"entry_file": entry_file, "error": str(e)})

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
        prev_val = (previous_stats.get('median_time_ms') or previous_stats.get('mean_time_ms')) if isinstance(previous_stats, dict) else None
        curr_val = stats.get('median_time_ms')
        if prev_val and curr_val:
            speedup = prev_val / curr_val
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
    quantized_path = Path(__file__).parents[2] / "quantized.py"
    
    try:
        worker = RemoteWorkerClient(
            ssh_config,
            worker_path,
            {str(loader_path): "loader.py", str(quantized_path): "quantized.py"},
        )
        
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
        quantized_path = Path(__file__).parents[2] / "quantized.py"
        
        worker = RemoteWorkerClient(
            ssh_config,
            worker_path,
            {str(loader_path): "loader.py", str(quantized_path): "quantized.py"},
        )
        
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
            "batch_size": settings.batch_size,
            "op_name": paths.get("op_name") or io_dir.name,
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
