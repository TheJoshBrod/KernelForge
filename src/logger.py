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

def save_statistics_csv(benchmark_name: str, benchmark_stats: list[dict], OUTPUT_BASE_DIR: Path):
    """
    Save benchmark statistics to a CSV file.
    Includes median and mean attempts for successful runs, and total success rate.
    Also includes performance metrics (time and memory).
    """
    if not benchmark_stats:
        print("No statistics to save.")
        return
    
    # Calculate metrics
    successful_runs = [s for s in benchmark_stats if s['success']]
    total_runs = len(benchmark_stats)
    success_count = len(successful_runs)
    success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0
    
    # Calculate median and mean attempts for successful runs only
    if successful_runs:
        attempts_list = [s['attempts'] for s in successful_runs]
        median_attempts = median(attempts_list)
        mean_attempts = mean(attempts_list)
        
        # Calculate mean performance metrics for successful runs
        exec_times = [s['execution_time_ms'] for s in successful_runs if s['execution_time_ms'] is not None]
        mem_alloc = [s['memory_allocated_mb'] for s in successful_runs if s['memory_allocated_mb'] is not None]
        peak_mem = [s['peak_memory_mb'] for s in successful_runs if s['peak_memory_mb'] is not None]
        
        mean_exec_time = mean(exec_times) if exec_times else 0
        mean_mem_alloc = mean(mem_alloc) if mem_alloc else 0
        mean_peak_mem = mean(peak_mem) if peak_mem else 0
    else:
        median_attempts = 0
        mean_attempts = 0
        mean_exec_time = 0
        mean_mem_alloc = 0
        mean_peak_mem = 0
    
    # Save summary CSV
    summary_path = OUTPUT_BASE_DIR / benchmark_name / "summary_statistics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Operations', total_runs])
        writer.writerow(['Successful Operations', success_count])
        writer.writerow(['Success Rate (%)', f'{success_rate:.2f}'])
        writer.writerow(['Median Attempts (Success Only)', f'{median_attempts:.2f}'])
        writer.writerow(['Mean Attempts (Success Only)', f'{mean_attempts:.2f}'])
        writer.writerow(['Mean Execution Time (ms)', f'{mean_exec_time:.3f}'])
        writer.writerow(['Mean Memory Allocated (MB)', f'{mean_mem_alloc:.3f}'])
        writer.writerow(['Mean Peak Memory (MB)', f'{mean_peak_mem:.3f}'])
    
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"  Total Operations: {total_runs}")
    print(f"  Successful: {success_count}")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Median Attempts (successful): {median_attempts:.2f}")
    print(f"  Mean Attempts (successful): {mean_attempts:.2f}")
    print(f"  Mean Execution Time: {mean_exec_time:.3f} ms")
    print(f"  Mean Memory Allocated: {mean_mem_alloc:.3f} MB")
    print(f"  Mean Peak Memory: {mean_peak_mem:.3f} MB")
    print(f"  Saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    # Save detailed CSV with per-operation stats
    detail_path = OUTPUT_BASE_DIR / benchmark_name / "detailed_statistics.csv"
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Test Name', 
            'Operation', 
            'Success', 
            'Attempts',
            'Execution Time (ms)',
            'Memory Allocated (MB)',
            'Memory Reserved (MB)',
            'Peak Memory (MB)'
        ])
        for stat in benchmark_stats:
            attempts_str = str(stat['attempts']) if stat['attempts'] != -1 else 'Failed'
            exec_time = f"{stat['execution_time_ms']:.3f}" if stat['execution_time_ms'] is not None else 'N/A'
            mem_alloc = f"{stat['memory_allocated_mb']:.3f}" if stat['memory_allocated_mb'] is not None else 'N/A'
            mem_reserved = f"{stat['memory_reserved_mb']:.3f}" if stat['memory_reserved_mb'] is not None else 'N/A'
            peak_mem = f"{stat['peak_memory_mb']:.3f}" if stat['peak_memory_mb'] is not None else 'N/A'
            
            writer.writerow([
                stat['test_name'],
                stat['operation'],
                'Yes' if stat['success'] else 'No',
                attempts_str,
                exec_time,
                mem_alloc,
                mem_reserved,
                peak_mem
            ])
    
    print(f"Detailed statistics saved to: {detail_path}\n")

def get_arg_types(all_args: list[list[Any]], all_kwargs: list[dict[str, Any]]):

    print("args")
    for args in all_args:
        print([
            "tensor" if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ])

    print("kwargs")
    for kwargs in all_kwargs:
        print([
            f"{key}: tensor" if isinstance(value, torch.Tensor) else f"{key}: {value}"
            for key, value in kwargs.items()
        ])

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

def run_pytorch_version(function_name, exec_str, all_inputs, warmup=5, iterations=100) -> PerformanceMetrics:
    """Profile PyTorch reference implementation across all inputs together."""
    try:
        torch_module = __import__("torch")
        
        # Warmup
        for i in range(warmup):
            counter = 0
            for cuda_args, cuda_kwargs in all_inputs:        
                counter += 1
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
    
    tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")

    cu_path = output_dir / "kernel.cu"

    # Compile
    try:
        module = load(
                    name=f"generated_module_{os.path.basename(tmpdir)}",
                    sources=[cu_path],
                    build_directory=output_dir,
                    verbose=True, 
                )
    except Exception as e:
        return
    
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
    pytorch_metrics = run_pytorch_version(function_name, exec_str, all_pytorch_inputs, warmup, iterations)
    
    create_log_file(custom_metrics, pytorch_metrics, output_dir)