"""
Triton Remote Worker.
Runs on a remote server via SSH. Handles verify, profile, and get_specs commands
for Triton kernels using a binary protocol over stdin/stdout.
"""

import os
import sys
import struct
import pickle
import traceback
import json
import glob
import time
import importlib
import importlib.util
import tempfile
from pathlib import Path

import numpy as np


def configure_remote_env():
    """
    Verify Triton is available on the remote machine.
    Unlike CUDA's remote worker, Triton doesn't need nvcc discovery —
    it compiles kernels JIT through Python.
    """
    print("DEBUG: Configuring Triton remote environment...", flush=True)

    try:
        import triton
        print(f"DEBUG: Triton {triton.__version__} available", flush=True)
    except ImportError:
        print("WARNING: Triton not installed. Install with: pip install triton", flush=True)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"DEBUG: CUDA available, device: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("WARNING: No CUDA device found", flush=True)
    except ImportError:
        print("WARNING: PyTorch not installed", flush=True)


# Configure before imports
configure_remote_env()

import torch

try:
    import triton
    import triton.testing
except ImportError:
    triton = None


# --- Helper Functions ---

def move_to_cuda(item):
    """Recursively move tensors to CUDA."""
    if torch.is_tensor(item):
        return item.cuda()
    elif isinstance(item, (list, tuple)):
        return type(item)(move_to_cuda(x) for x in item)
    elif isinstance(item, dict):
        return {k: move_to_cuda(v) for k, v in item.items()}
    return item


def normalize_args_kwargs(args, kwargs, signature_info):
    """Normalize args and kwargs into a complete positional argument list."""
    params = signature_info.get("params", [])
    defaults = signature_info.get("defaults", {})

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


def load_triton_module(code: str, tmpdir: str):
    """Write Triton kernel code to a temp file and import it."""
    kernel_path = os.path.join(tmpdir, "kernel.py")
    with open(kernel_path, 'w', encoding='utf-8') as f:
        f.write(code)

    module_name = f"triton_kernel_{int(time.time())}"
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handle_verify(data):
    """
    Imports and verifies the Triton kernel against IO files.
    """
    try:
        kernel_code = data['code']
        io_dir = data['io_dir']

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                module = load_triton_module(kernel_code, tmpdir)
            except Exception as e:
                return {"valid": False, "log": f"Import Error: {str(e)}"}

            if not hasattr(module, 'launch'):
                return {"valid": False, "log": "No 'launch()' function found in kernel"}

            # Load IO files
            entry_files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))
            if not entry_files:
                return {"valid": False, "log": "No entry files found in io_dir"}

            all_valid = True
            error_logs = []

            for entry_file in entry_files:
                try:
                    entry = torch.load(entry_file, map_location='cpu', weights_only=False)
                    args = entry.get("args", [])
                    kwargs = entry.get("kwargs", {})
                    signature_info = entry.get("signature", {"params": [], "defaults": {}})

                    normalized_args, _ = normalize_args_kwargs(args, kwargs, signature_info)
                    cuda_args = [move_to_cuda(item) for item in normalized_args]

                    # Launch
                    output_generated = module.launch(*cuda_args)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if torch.is_tensor(output_generated) and not output_generated.is_cuda:
                        output_generated = output_generated.cuda()

                    ground_truth = entry["output"]
                    if torch.is_tensor(ground_truth):
                        ground_truth = ground_truth.to(output_generated.device)

                    is_correct = torch.allclose(output_generated, ground_truth, atol=1e-2, rtol=1e-1)

                    if not is_correct:
                        all_valid = False
                        diff = torch.abs(output_generated - ground_truth)
                        error_logs.append(f"[{os.path.basename(entry_file)}] Max diff: {diff.max().item():.6f}")

                except Exception as e:
                    all_valid = False
                    error_logs.append(f"[{os.path.basename(entry_file)}] Runtime Error: {str(e)}")

            if all_valid:
                return {"valid": True, "log": f"Passed {len(entry_files)} tests."}
            else:
                return {"valid": False, "log": "\n".join(error_logs)}

    except Exception as e:
        return {"valid": False, "log": f"Worker Error: {str(e)}"}


def handle_profile(data):
    """
    Imports and profiles the Triton kernel using triton.testing.do_bench.
    """
    if triton is None:
        return {"error": "Triton not installed on remote"}

    try:
        kernel_code = data['code']
        io_dir = data['io_dir']
        batch_size = data.get('batch_size', 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                module = load_triton_module(kernel_code, tmpdir)
            except Exception as e:
                return {"error": f"Import Error: {str(e)}"}

            if not hasattr(module, 'launch'):
                return {"error": "No 'launch()' function found"}

            # Load inputs
            files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))
            if not files:
                return {"error": "No input files found"}

            timings = []

            # Batch processing
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                inputs = []

                for f in batch_files:
                    try:
                        entry = torch.load(f, map_location='cpu', weights_only=False)
                        args = entry.get('args', [])
                        kwargs = entry.get('kwargs', {})
                        sig = entry.get('signature', {})
                        norm_args, _ = normalize_args_kwargs(args, kwargs, sig)
                        cuda_args = [move_to_cuda(x) for x in norm_args]
                        inputs.append(cuda_args)
                    except Exception:
                        continue

                if not inputs:
                    continue

                # Measure each input with triton.testing.do_bench
                for args in inputs:
                    ms = triton.testing.do_bench(
                        lambda a=args: module.launch(*a),
                        warmup=25,
                        rep=100,
                    )
                    timings.append(ms)

                # Cleanup
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not timings:
                return {"error": "Profiling failed (no valid timings)"}

            return {
                "median_time_ms": float(np.median(timings)),
                "mean_time_ms": float(np.mean(timings)),
                "std_time_ms": float(np.std(timings)),
                "min_time_ms": float(np.min(timings)),
                "max_time_ms": float(np.max(timings))
            }

    except Exception as e:
        return {"error": f"Profiling Error: {str(e)}", "trace": traceback.format_exc()}


def main():
    # Signal readiness
    print("READY", flush=True)

    while True:
        try:
            # 1. Read message length (4 bytes)
            raw_len = sys.stdin.buffer.read(4)
            if not raw_len:
                break  # EOF

            msg_len = struct.unpack('>I', raw_len)[0]

            # 2. Read payload
            payload = sys.stdin.buffer.read(msg_len)
            request = pickle.loads(payload)

            # 3. Process
            cmd = request.get('command')
            result = {}

            if cmd == 'verify':
                result = handle_verify(request.get('data'))

            elif cmd == 'profile':
                result = handle_profile(request.get('data'))

            elif cmd == 'get_specs':
                # Use torch.cuda for basic specs
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    cc_major, cc_minor = torch.cuda.get_device_capability(0)
                    result = {
                        "gpu_name": props.name,
                        "compute_capability": f"{cc_major}.{cc_minor}",
                        "total_memory_gb": props.total_mem / (1024 ** 3),
                        "num_sms": props.multi_processor_count,
                        "max_threads_per_block": 1024,
                        "warp_size": 32,
                    }

                    # Try to enrich with pynvml/pycuda
                    try:
                        import pynvml
                        import pycuda.driver as drv
                        import pycuda.autoinit

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')

                        dev = drv.Device(0)
                        attrs = dev.get_attributes()

                        result.update({
                            "gpu_name": name,
                            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                            "num_sms": attrs[drv.device_attribute.MULTIPROCESSOR_COUNT],
                            "warp_size": attrs[drv.device_attribute.WARP_SIZE],
                            "max_threads_per_block": attrs[drv.device_attribute.MAX_THREADS_PER_BLOCK],
                        })
                        pynvml.nvmlShutdown()
                    except ImportError:
                        pass  # Just use basic torch.cuda info
                else:
                    result = {"error": "No GPU available"}

            else:
                result = {"error": "Unknown command"}

            # 4. Send Response
            resp_bytes = pickle.dumps(result)
            sys.stdout.buffer.write(struct.pack('>I', len(resp_bytes)))
            sys.stdout.buffer.write(resp_bytes)
            sys.stdout.buffer.flush()

        except Exception as e:
            err = {"error": str(e), "trace": traceback.format_exc()}
            resp_bytes = pickle.dumps(err)
            sys.stdout.buffer.write(struct.pack('>I', len(resp_bytes)))
            sys.stdout.buffer.write(resp_bytes)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
