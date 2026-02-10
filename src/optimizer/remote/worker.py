import sys
import struct
import pickle
import traceback
import json
import os
import shutil
import glob
import time
import subprocess
from pathlib import Path
import re

import glob
import time
import subprocess
from pathlib import Path
import re
import sys

# --- Environment Setup (Must be before torch import for safety in some envs) ---
def setup_environment():
    """Finds a suitable nvcc and sets CUDA_HOME."""
    
    # 1. Check if we have nvcc in the current venv (best match for torch)
    venv_bin = os.path.dirname(sys.executable)
    venv_nvcc = os.path.join(venv_bin, "nvcc")
    if os.path.exists(venv_nvcc):
        print(f"DTO: Found VENV nvcc at {venv_nvcc}", file=sys.stderr)
        os.environ["CUDA_HOME"] = os.path.dirname(venv_bin)
        os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"
        return

    # 2. Check standard locations for a NEWER cuda
    # The default /usr/bin/nvcc might be old (e.g. 9.x or 10.x) which causes the dependency flag error.
    # We prefer /usr/local/cuda-12.x or 11.x
    candidates = sorted(glob.glob("/usr/local/cuda*"), reverse=True) # Try newest first
    
    found_good_cuda = False
    for path in candidates:
        nvcc_path = os.path.join(path, "bin", "nvcc")
        if os.path.exists(nvcc_path):
             # Simple check: avoid "cuda-10" if we want newer? 
             # For now, just take the newest one found.
             print(f"DTO: Found NVCC candidate at {nvcc_path}", file=sys.stderr)
             os.environ["CUDA_HOME"] = path
             os.environ["PATH"] = f"{path}/bin:{os.environ['PATH']}"
             found_good_cuda = True
             break
    
    if found_good_cuda:
        return

    # 3. Fallback to `which nvcc`
    try:
        which_nvcc = subprocess.check_output(["which", "nvcc"]).decode().strip()
        print(f"DTO: Falling back to system nvcc at {which_nvcc}", file=sys.stderr)
        # We don't change CUDA_HOME if it's just /usr/bin, usually implies /usr is home
        # But if it's /usr/bin/nvcc, setting CUDA_HOME=/usr might be wrong or right?
        # PyTorch defaults to finding it.
    except:
        print("DTO: Warning - Could not find any nvcc.", file=sys.stderr)

setup_environment()

import torch
import numpy as np
from torch.utils.cpp_extension import load_inline

# Initialize CUDA context if available
try:
    if cuda:
        cuda.init()
        DEVICE_COUNT = cuda.Device.count()
    else:
        DEVICE_COUNT = 0
except Exception as e:
    DEVICE_COUNT = 0
    print(f"CUDA Init failed: {e}", file=sys.stderr)

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

def get_module(kernel_code, tmp_dir):
    """Compiles the kernel code using load_inline."""
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", kernel_code)
    if not match:
        raise ValueError("Could not find 'launch' function signature in kernel code.")
    
    cpp_source = match.group(1) + ";"
    
    # Use a unique name to avoid collision/caching issues if checking multiple attempts
    # mechanism: load_inline caches by source content hashing, so it should be fine.
    # But we can force it by valid name.
    module_name = f"kernel_{int(time.time())}"
    
    return load_inline(
        name=module_name,
        cpp_sources=cpp_source,
        cuda_sources=kernel_code,
        functions=['launch'],
        build_directory=str(tmp_dir),
        verbose=False,
        with_cuda=True
    )


def handle_verify(data):
    """
    Compiles and verifies the kernel against IO files.
    data = {
        'code': str,       # Kernel CUDA code
        'io_dir': str      # Path to directory containing .pt files on remote
    }
    """
    try:
        kernel_code = data['code']
        io_dir = data['io_dir']
        
        # Create a temp dir for compilation artifacts
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                module = get_module(kernel_code, tmpdir)
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
    data = {
        'code': str,
        'io_dir': str,
        'batch_size': int (optional)
    }
    """
    try:
        kernel_code = data['code']
        io_dir = data['io_dir']
        batch_size = data.get('batch_size', 5) # Default batch size

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                module = get_module(kernel_code, tmpdir)
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
                    for _ in range(10): # Profile 10 runs per input
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
            
            # Allow executing arbitrary python code payload for maximum flexibility?
            # Ideally the client sends a function and args.
            if cmd == 'execute_script':
                # Client sends a script string and expected entry point?
                # This seems risky if we want to carry state, but checks strict separation.
                # But for performance we want to persist imports.
                
                # Let's try to support the specific commands requested: verify, profile
                # But we need to implement the actual logic in here.
                pass
            
            if cmd == 'verify':
                result = handle_verify(request.get('data'))
                
            elif cmd == 'profile':
                 result = handle_profile(request.get('data'))
                 
            elif cmd == 'get_specs':
                # Simplified info gathering
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_names = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    gpu_names.append(name)
                    
                import pycuda.driver as drv
                drv.init()
                dev = drv.Device(0) # Assume dev 0 for now or handle list
                attributes = dev.get_attributes()
                arch = f"{dev.compute_capability()[0]}.{dev.compute_capability()[1]}"
                
                result = {
                    "gpu_names": gpu_names,
                    "arch": arch,
                    "mem": pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total
                }

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
