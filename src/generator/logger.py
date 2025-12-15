"""Handles all functions that log statistics"""
import torch
import json
import time
from pathlib import Path
import tempfile
import os
import glob
from typing import Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import load

def handle_input_pytorch(input_path: Path):
    inputs = torch.load(input_path)

    args = inputs["args"]
    kwargs = inputs["kwargs"]

    def move(item):
        if torch.is_tensor(item):
            return item.cuda()
        return item  # Keep all other values

    cuda_args = [move(a) for a in args]
    cuda_kwargs = {k: move(v) for k, v in kwargs.items()}

    return cuda_args, cuda_kwargs

def handle_input_custom(input_path):
    inputs = torch.load(input_path)
    if isinstance(inputs, dict) and "args" in inputs and "kwargs" in inputs:
        args = inputs["args"]
        kwargs = inputs["kwargs"]
        
        # Move tensors to CUDA, keep scalars as-is
        cuda_args = []
        for item in args:
            if torch.is_tensor(item):
                cuda_args.append(item.cuda())
            elif isinstance(item, (int, float, bool, str)):
                cuda_args.append(item)
            # Skip other types
        
        cuda_kwargs = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                cuda_kwargs[k] = v.cuda()
            elif isinstance(v, (int, float, bool, str)):
                cuda_kwargs[k] = v
            # Skip other types
    return cuda_args, cuda_kwargs

@dataclass
class PerformanceMetrics:
    mean_time_ms: float
    median_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_gflops: float = None

    
    def speedup_over(self, other: 'PerformanceMetrics') -> float:
        return other.mean_time_ms / self.mean_time_ms

    def to_dict(self) -> float:
        return {
            "mean_time_ms": self.mean_time_ms,
            "median_time_ms": self.median_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "throughput_gflops": self.throughput_gflops,
        }


def run_custom_kernel(module, all_inputs, warmup=5, iterations=100) -> PerformanceMetrics:
    """Profile custom CUDA kernel across all inputs together."""
    try:
        # Warmup
        for _ in range(warmup):
            for cuda_args, cuda_kwargs in all_inputs:
                module.launch(*cuda_args, **cuda_kwargs)
        torch.cuda.synchronize()
        
        # Timing runs
        times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for cuda_args, cuda_kwargs in all_inputs:
                module.launch(*cuda_args, **cuda_kwargs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = torch.tensor(times)
        return PerformanceMetrics(
            mean_time_ms=times.mean().item(),
            median_time_ms=times.median().item(),
            std_time_ms=times.std().item(),
            min_time_ms=times.min().item(),
            max_time_ms=times.max().item()
        )
    except Exception as e:
        print(f"Custom kernel failed: {e}")
        return None

def run_pytorch_version(exec_str, all_inputs, warmup=5, iterations=100) -> PerformanceMetrics:
    """Profile PyTorch reference implementation across all inputs together."""
    try:
        torch_module = __import__("torch")
        
        # Warmup
        for _ in range(warmup):
            for cuda_args, cuda_kwargs in all_inputs:        
                context = {
                    "torch": torch_module,
                    "args": cuda_args,
                    "kwargs": cuda_kwargs,
                }
                exec(exec_str, context)
        torch.cuda.synchronize()
        
        # Timing runs
        times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for cuda_args, cuda_kwargs in all_inputs:
                context = {
                    "torch": torch_module,
                    "args": cuda_args,
                    "kwargs": cuda_kwargs,
                }
                exec(exec_str, context)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = torch.tensor(times)
        return PerformanceMetrics(
            mean_time_ms=times.mean().item(),
            median_time_ms=times.median().item(),
            std_time_ms=times.std().item(),
            min_time_ms=times.min().item(),
            max_time_ms=times.max().item()
        )
    except Exception as e:
        print(f"PyTorch version failed: {e}")
        return None

def create_log_file(custom_metrics: PerformanceMetrics, pytorch_metrics: PerformanceMetrics, output_dir: Path):
    
    log = {}
    
    log["metadata"] = {
        "time": time.time(),
        "pytorch": torch.__version__,
        "GPU": torch.cuda.get_device_name(0),
        "CUDA Version": torch.version.cuda,
        "cuDNN Version": torch.backends.cudnn.version()
    }

    log["custom"] = custom_metrics.to_dict()
    log["pytorch"] = pytorch_metrics.to_dict()
    
    with open(output_dir / "performance.json", "w") as f:
        json.dump(log,f)

def compare_kernel_to_pytorch(output_dir: Path, function_name: str, exec_str: str, warmup=10, iterations=100):
    
    cu_path = output_dir / "kernel.cu"

    

    # Load Compiled Module
    compile_path = output_dir / "compiled"
    os.makedirs(compile_path, exist_ok=True)
    module_name = f"generated_module_{os.path.basename(str(compile_path))}"

    module = None
    try:
    # Try loading an existing compiled module
        module = load(
            name=module_name,
            sources=[cu_path],
            build_directory=compile_path,
            verbose=False,
            extra_cflags=[],
            extra_cuda_cflags=[],
            with_cuda=True,
            is_python_module=False
        )
    except Exception as load_err:
        # If compiled module does not exist, create compile
        print("No compiled module found compiling now...")
        try:
            # Compile from source
            module = load(
                name=module_name,
                sources=[cu_path],
                build_directory=compile_path,
                verbose=True,
            )

        except Exception as compile_err:
            print("Compilation failed:", compile_err)
            return

    print("Compiled/Loaded compiled version...")

    input_paths = glob.glob(str(output_dir / "input*.pt"))
    
    # Handle all inputs
    all_pytorch_inputs = []
    all_custom_inputs = []
    for input_path in input_paths:
        cuda_args, cuda_kwargs = handle_input_pytorch(input_path)
        all_pytorch_inputs.append((cuda_args, cuda_kwargs))

        cuda_args, cuda_kwargs = handle_input_custom(input_path)
        all_custom_inputs.append((cuda_args, cuda_kwargs))




    # Profile compiled version
    custom_metrics = run_custom_kernel(module, all_custom_inputs, warmup, iterations)

    # Profile pytorch version
    pytorch_metrics = run_pytorch_version(exec_str, all_pytorch_inputs, warmup, iterations)
    
    create_log_file(custom_metrics, pytorch_metrics, output_dir)