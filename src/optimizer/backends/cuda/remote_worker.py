import os
import sys
import struct
import pickle
import traceback
import json
import shutil
import glob
import time
import subprocess
from pathlib import Path
import re
import numpy as np

def configure_remote_env():
    """
    Dynamically configures CUDA environment on the remote worker.
    Searches for the newest available nvcc and sets CUDA_HOME/PATH.
    Disables Ninja if nvcc is too old (< 11.0).
    """
    print("DEBUG: Configuring remote environment...", flush=True)
    
    # Potential nvcc locations
    candidates = glob.glob("/usr/local/cuda-*/bin/nvcc")
    if os.path.exists("/usr/bin/nvcc"):
        candidates.append("/usr/bin/nvcc")
        
    best_version = (0, 0)
    best_path = None
    best_home = None
    
    for nvcc_path in candidates:
        try:
            # Check version: "Cuda compilation tools, release 11.0, V11.0.194"
            out = subprocess.check_output([nvcc_path, "--version"]).decode()
            match = re.search(r"release (\d+\.\d+)", out)
            if match:
                ver_str = match.group(1)
                ver_major, ver_minor = map(int, ver_str.split('.'))
                ver_tuple = (ver_major, ver_minor)
                
                if ver_tuple > best_version:
                    best_version = ver_tuple
                    best_path = nvcc_path
                    # /usr/local/cuda-11.0/bin/nvcc -> /usr/local/cuda-11.0
                    if "/usr/local/cuda-" in nvcc_path:
                         best_home = str(Path(nvcc_path).parent.parent)
                    else:
                         # For /usr/bin/nvcc, try to guess if it's a symlink or system install
                         # Often /usr/bin/nvcc is just a symlink to /usr/local/cuda/bin/nvcc
                         # Let's not set CUDA_HOME if it's /usr/bin to avoid messing up system paths
                         best_home = None 
        except Exception:
            continue
            
    if best_path:
        print(f"DEBUG: Found best nvcc at {best_path} (v{best_version[0]}.{best_version[1]})", flush=True)
        
        # Update PATH
        bin_dir = str(Path(best_path).parent)
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        
        # Update CUDA_HOME
        if best_home:
            os.environ["CUDA_HOME"] = best_home
            print(f"DEBUG: Set CUDA_HOME to {best_home}", flush=True)
            
        # Ninja compatibility check
        if best_version < (11, 0):
            print("DEBUG: nvcc < 11.0 detected, disabling Ninja build system.", flush=True)
            os.environ["USE_NINJA"] = "0"
    else:
        print("DEBUG: No nvcc found. Relying on existing environment.", flush=True)

# Configure before importing torch/loader
configure_remote_env()

# Assuming loader.py is uploaded to the same directory
import loader
import torch

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

def handle_verify(data):
    """
    Compiles and verifies the kernel against IO files.
    """
    try:
        kernel_code = data['code']
        io_dir = data['io_dir']
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Use loader to compile
                module = loader.compile_code_string(
                    code=kernel_code,
                    name=f"kernel_{int(time.time())}",
                    build_dir=tmpdir,
                    verbose=False
                )
            except Exception as e:
                 return {"valid": False, "log": f"Compilation Error: {str(e)}"}

            # Load IO files
            entry_files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))
            if not entry_files:
                return {"valid": False, "log": "No entry files found in io_dir"}

            all_valid = True
            error_logs = []

            for entry_file in entry_files:
                try:
                    entry = torch.load(entry_file, map_location='cpu')
                    args = entry.get("args", [])
                    kwargs = entry.get("kwargs", {})
                    signature_info = entry.get("signature", {"params": [], "defaults": {}})

                    normalized_args, _ = normalize_args_kwargs(args, kwargs, signature_info)
                    cuda_args = [move_to_cuda(item) for item in normalized_args]

                    # Launch
                    output_generated = module.launch(*cuda_args)
                    torch.cuda.synchronize()

                    if not output_generated.is_cuda:
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
    Compiles and profiles the kernel.
    """
    try:
        kernel_code = data['code']
        io_dir = data['io_dir']
        batch_size = data.get('batch_size', 5)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                module = loader.compile_code_string(
                    code=kernel_code,
                    name=f"kernel_{int(time.time())}",
                    build_dir=tmpdir,
                    verbose=False
                )
            except Exception as e:
                 return {"error": f"Compilation Error: {str(e)}"}
            
            # Load inputs
            files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))
            if not files:
                 return {"error": "No input files found"}
            
            timings = []
            
            # Batch processing
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i+batch_size]
                inputs = []
                
                # Load batch
                for f in batch_files:
                    try:
                         entry = torch.load(f, map_location='cpu')
                         args = entry.get('args', [])
                         kwargs = entry.get('kwargs', {})
                         sig = entry.get('signature', {})
                         norm_args, _ = normalize_args_kwargs(args, kwargs, sig)
                         cuda_args = [move_to_cuda(x) for x in norm_args]
                         inputs.append(cuda_args)
                    except:
                        continue
                
                if not inputs: continue

                # Warmup
                for args in inputs:
                     module.launch(*args)
                torch.cuda.synchronize()
                
                # Measure
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                for args in inputs:
                    start.record()
                    for _ in range(10): 
                         module.launch(*args)
                    end.record()
                    torch.cuda.synchronize()
                    timings.append(start.elapsed_time(end) / 10.0)
                
                # Cleanup
                del inputs
                torch.cuda.empty_cache()

            if not timings:
                 return {"error": "Profiling failed (no valid timings)"}
            
            return {
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
            if not raw_len: break # EOF
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
                # Simplified info gathering
                # Full GPUSpecs retrieval logic similar to profiler.py
                import pynvml
                import pycuda.driver as cuda
                import pycuda.autoinit # Ensure context

                def _nvml_safe(fn, default=None):
                    try:
                        return fn()
                    except:
                        return default

                def _to_str(x):
                    return x.decode() if isinstance(x, bytes) else x

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                power_limit_mw = _nvml_safe(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle))
                
                nvml_info = {
                    "gpu_name": _to_str(pynvml.nvmlDeviceGetName(handle)),
                    "nvml_architecture": pynvml.nvmlDeviceGetArchitecture(handle),
                    "total_memory_gb": pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3),
                    "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                    "power_limit_watts": None if power_limit_mw is None else power_limit_mw / 1000,
                }

                pynvml.nvmlShutdown()
                
                dev = cuda.Device(0)
                attrs = dev.get_attributes()
                cc_major, cc_minor = dev.compute_capability()
                
                cuda_info = {
                    "compute_capability": f"{cc_major}.{cc_minor}",
                    "num_sms": attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT],
                    "warp_size": attrs[cuda.device_attribute.WARP_SIZE],
                    "max_threads_per_block": attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK],
                    "max_threads_per_sm": attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR],
                    "max_blocks_per_sm": attrs.get(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR, "unknown") if hasattr(cuda.device_attribute, 'MAX_BLOCKS_PER_MULTIPROCESSOR') else "unknown",
                    "registers_per_sm": attrs[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR],
                    "registers_per_block": attrs[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK],
                    "shared_mem_per_sm_kb": attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR] // 1024,
                    "shared_mem_per_block_kb": attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] // 1024,
                    "l2_cache_kb": attrs[cuda.device_attribute.L2_CACHE_SIZE] // 1024,
                    "memory_bus_width_bits": attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
                }
                
                # Derived
                memory_bandwidth_gbps = (
                    nvml_info["mem_clock_mhz"] * 1e6 *
                    cuda_info["memory_bus_width_bits"] / 8 * 2
                ) / 1e9
                
                derived = {
                    "peak_memory_bandwidth_gbps": memory_bandwidth_gbps,
                    "warps_per_sm": cuda_info["max_threads_per_sm"] // cuda_info["warp_size"],
                }
                
                result = {**nvml_info, **cuda_info, **derived}

            else:
                result = {"error": "Unknown command"}

            # 4. Send Response (Length + Payload)
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
