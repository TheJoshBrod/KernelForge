"""
Triton Backend Profiler.
Handles GPU specs detection (NVIDIA + AMD) and kernel profiling.
"""

import os
import glob
import importlib
import importlib.util
import subprocess
from pathlib import Path

import torch

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from src.optimizer.config.settings import settings
from src.optimizer.core.types import GPUSpecs
from src.optimizer.profiling import get_device_specs as get_profiled_device_specs
from src.optimizer.benchmarking.harness import (
    DEFAULT_TIMED_RUNS,
    DEFAULT_WARMUP_RUNS,
    benchmark_entry_calls,
    summarize_entry_results,
    sync_device,
)


# ******************
#  GPU SPECS
# ******************

def get_gpu_specs(device_index: int = 0, ssh_config: dict = None) -> GPUSpecs:
    """Retrieve GPU specs via unified profiling; keep remote SSH path."""
    if ssh_config:
        return get_remote_gpu_specs(ssh_config)
    return get_profiled_device_specs(device_index=device_index, mode="fast")


def _is_rocm() -> bool:
    """Check if PyTorch was built with ROCm support."""
    try:
        config = torch.__config__.show()
        return 'rocm' in config.lower() or 'hip' in config.lower()
    except Exception:
        return False


def _get_torch_cuda_specs(device_index: int = 0) -> GPUSpecs:
    """
    Lightweight GPU specs using only torch.cuda (works on both NVIDIA and AMD/ROCm).
    Provides fewer details than the full pynvml/pycuda query.
    """
    props = torch.cuda.get_device_properties(device_index)
    cc_major, cc_minor = torch.cuda.get_device_capability(device_index)

    return GPUSpecs(
        gpu_name=props.name,
        compute_capability=f"{cc_major}.{cc_minor}",
        total_memory_gb=getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 3),
        num_sms=props.multi_processor_count,
        max_threads_per_block=props.max_threads_per_block if hasattr(props, 'max_threads_per_block') else 1024,
        warp_size=props.warp_size if hasattr(props, 'warp_size') else 32,
        # Fields we can't get from torch.cuda alone — set reasonable defaults
        sm_clock_mhz=0,
        mem_clock_mhz=0,
        power_limit_watts=None,
        nvml_architecture=0,
        max_threads_per_sm=0,
        max_blocks_per_sm="unknown",
        registers_per_sm=0,
        registers_per_block=0,
        shared_mem_per_sm_kb=0,
        shared_mem_per_block_kb=props.max_threads_per_block // 1024 if hasattr(props, 'max_threads_per_block') else 48,
        l2_cache_kb=0,
        memory_bus_width_bits=0,
        peak_memory_bandwidth_gbps=0.0,
        warps_per_sm=0,
    )


def _get_rocm_specs(device_index: int = 0) -> GPUSpecs:
    """
    GPU specs for AMD ROCm GPUs. Uses torch.cuda (which works under ROCm)
    and rocm-smi CLI for additional info.
    """
    props = torch.cuda.get_device_properties(device_index)

    # Try to get additional info from rocm-smi
    power_watts = None
    clock_mhz = 0
    mem_clock_mhz = 0
    try:
        result = subprocess.run(
            ['rocm-smi', '--showpower', '--showclocks', '--json'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            # Parse rocm-smi JSON output (format varies by version)
            for card_key, card_data in data.items():
                if isinstance(card_data, dict):
                    power_str = card_data.get('Average Graphics Package Power (W)', '')
                    if power_str:
                        try:
                            power_watts = float(power_str)
                        except (ValueError, TypeError):
                            pass
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass  # rocm-smi not available

    # ROCm uses GCN/CDNA arch strings like "gfx90a" instead of SM numbers
    arch_str = getattr(props, 'gcnArchName', '') if hasattr(props, 'gcnArchName') else ''

    return GPUSpecs(
        gpu_name=props.name,
        compute_capability=arch_str if arch_str else "rocm",
        total_memory_gb=getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 3),
        num_sms=props.multi_processor_count,
        max_threads_per_block=1024,  # ROCm standard
        warp_size=64,  # AMD wavefront size
        sm_clock_mhz=clock_mhz,
        mem_clock_mhz=mem_clock_mhz,
        power_limit_watts=power_watts,
        nvml_architecture=0,
        max_threads_per_sm=0,
        max_blocks_per_sm="unknown",
        registers_per_sm=0,
        registers_per_block=0,
        shared_mem_per_sm_kb=0,
        shared_mem_per_block_kb=64,  # AMD LDS standard
        l2_cache_kb=0,
        memory_bus_width_bits=0,
        peak_memory_bandwidth_gbps=0.0,
        warps_per_sm=0,
    )


# ******************
#  KERNEL LOADING
# ******************

def load_triton_module(kernel_path: Path):
    """Load a Triton kernel module from a .py file."""
    module_name = kernel_path.stem + f"_{id(kernel_path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(kernel_path))
    if spec is None:
        raise ImportError(f"Cannot create module spec from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    """Retrieves list of all input file paths for a given profiled pytorch op."""
    if selected_files:
        pt_files = [str(Path(item)) for item in selected_files]
    else:
        pt_files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))
    if not pt_files:
        raise ValueError(f"No entry_*.pt files found in {io_dir}")
    return pt_files


def move_to_cuda(item):
    """Recursively move tensors to CUDA."""
    if torch.is_tensor(item):
        return item.cuda()
    elif isinstance(item, (list, tuple)):
        return type(item)(move_to_cuda(x) for x in item)
    elif isinstance(item, dict):
        return {k: move_to_cuda(v) for k, v in item.items()}
    return item


def load_batch(pt_files: list) -> list[tuple[str, list, dict]]:
    """Loads a batch of .pt files into GPU memory."""
    inputs = []
    for pt_file in pt_files:
        try:
            entry = torch.load(pt_file, map_location='cpu')

            args = [
                arg.cuda() if isinstance(arg, torch.Tensor) else arg
                for arg in entry['args']
            ]

            kwargs = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
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


# ******************
#  KERNEL PROFILING
# ******************

def profile_kernel(paths: dict[str, Path], *, baseline=False, device_index: int = 0, previous_stats: dict = None):
    """
    Profile a Triton kernel with the shared internal benchmark harness.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")

    kernel_path = paths["tmp_dir"] / "kernel.py"
    if not kernel_path.exists():
        # Try loading from the directory (baseline case)
        kernel_path = paths["tmp_dir"] / "kernel.py"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel file not found at {kernel_path}")

    module = load_triton_module(kernel_path)

    if not hasattr(module, 'launch'):
        raise ValueError("Triton kernel module has no 'launch()' function")

    input_dir = paths["io_dir"]
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)

    # Batching logic
    selected_files = paths.get("entry_files")
    all_files = get_input_files(input_dir, selected_files)
    BATCH_SIZE = settings.batch_size
    entry_results = []
    timing_errors = []
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
            sync_device("cuda")

            def invoke(bound_args=args, bound_kwargs=kwargs):
                try:
                    return module.launch(*bound_args, **bound_kwargs)
                except TypeError:
                    return module.launch(*bound_args)

            entry_calls.append((entry_file, invoke))

        batch_stats = benchmark_entry_calls(
            entry_calls,
            device="cuda",
            warmup_runs=DEFAULT_WARMUP_RUNS,
            timed_runs=DEFAULT_TIMED_RUNS,
        )
        entry_results.extend(batch_stats.get("entry_results") or [])
        timing_errors.extend(batch_stats.get("errors") or [])

        # Cleanup VRAM
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats = summarize_entry_results(
        entry_results,
        errors=timing_errors,
        device="cuda",
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
#  REMOTE PROFILER
# ******************

def get_remote_gpu_specs(ssh_config: dict) -> GPUSpecs:
    """
    Retrieves GPU specs from a remote server via SSH.
    """
    from src.optimizer.core.ssh_client import connect_ssh, execute_remote_command, ensure_remote_setup
    import json
    import tempfile

    client = connect_ssh(ssh_config)
    try:
        if not ensure_remote_setup(client):
            raise RuntimeError("Remote dependencies check failed")

        script_content = r'''
import json
import sys
try:
    import torch
except ImportError:
    print(json.dumps({"error": "PyTorch not installed"}))
    sys.exit(1)

def get_specs():
    try:
        if not torch.cuda.is_available():
            print(json.dumps({"error": "No GPU available"}))
            return

        props = torch.cuda.get_device_properties(0)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)

        specs = {
            "gpu_name": props.name,
            "compute_capability": f"{cc_major}.{cc_minor}",
            "total_memory_gb": props.total_mem / (1024 ** 3),
            "num_sms": props.multi_processor_count,
            "max_threads_per_block": 1024,
            "warp_size": 32,
            "sm_clock_mhz": 0,
            "mem_clock_mhz": 0,
            "power_limit_watts": None,
            "nvml_architecture": 0,
            "max_threads_per_sm": 0,
            "max_blocks_per_sm": "unknown",
            "registers_per_sm": 0,
            "registers_per_block": 0,
            "shared_mem_per_sm_kb": 0,
            "shared_mem_per_block_kb": 48,
            "l2_cache_kb": 0,
            "memory_bus_width_bits": 0,
            "peak_memory_bandwidth_gbps": 0.0,
            "warps_per_sm": 0,
        }

        # Try pynvml for richer data
        try:
            from pynvml import *
            import pycuda.driver as cuda
            import pycuda.autoinit

            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            specs["gpu_name"] = name
            specs["total_memory_gb"] = nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)
            specs["sm_clock_mhz"] = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
            specs["mem_clock_mhz"] = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)

            dev = cuda.Device(0)
            attrs = dev.get_attributes()
            specs["num_sms"] = attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]
            specs["warp_size"] = attrs[cuda.device_attribute.WARP_SIZE]
            specs["max_threads_per_block"] = attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
            specs["max_threads_per_sm"] = attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR]
            specs["registers_per_sm"] = attrs[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR]
            specs["registers_per_block"] = attrs[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK]
            specs["shared_mem_per_sm_kb"] = attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR] // 1024
            specs["shared_mem_per_block_kb"] = attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] // 1024
            specs["l2_cache_kb"] = attrs[cuda.device_attribute.L2_CACHE_SIZE] // 1024
            specs["memory_bus_width_bits"] = attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]

            bw = (specs["mem_clock_mhz"] * 1e6 * specs["memory_bus_width_bits"] / 8 * 2) / 1e9
            specs["peak_memory_bandwidth_gbps"] = bw
            specs["warps_per_sm"] = specs["max_threads_per_sm"] // specs["warp_size"]

            nvmlShutdown()
        except ImportError:
            pass  # pynvml/pycuda not available, use torch.cuda fallback

        print(json.dumps(specs))

    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e) + "\n" + traceback.format_exc()}))

if __name__ == "__main__":
    get_specs()
'''
        script_name = "triton_specs_fetcher.py"
        remote_workspace = "kforge_workspace"

        # Upload script
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(script_content)
            tmp_path = tmp.name

        from src.optimizer.core.ssh_client import upload_files
        upload_files(client, {tmp_path: script_name}, remote_workspace)
        os.unlink(tmp_path)

        # Execute
        cmd = f"~/kforge_workspace/venv/bin/python3 {remote_workspace}/{script_name}"
        exit_code, out, err = execute_remote_command(client, cmd)

        # Cleanup
        execute_remote_command(client, f"rm {remote_workspace}/{script_name}")

        if exit_code != 0:
            print(f"Remote profiling error: {out} {err}")
            raise RuntimeError(f"Remote profiling failed: {err}")

        try:
            import json
            stats = json.loads(out)
            if "error" in stats:
                raise RuntimeError(stats["error"])
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON from remote: {out}")

        return GPUSpecs(**stats)

    finally:
        client.close()


def profile_remote_kernel(ssh_config: dict, paths: dict[str, Path], baseline: bool = False) -> tuple[dict, any]:
    """
    Profiles a Triton kernel on a remote server via SSH.
    """
    from src.optimizer.core.ssh_client import RemoteWorkerClient, upload_files

    try:
        worker = RemoteWorkerClient(ssh_config)

        # 1. Prepare Code
        kernel_path = paths["tmp_dir"] / "kernel.py"
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
