"""Handles all functions that log statistics"""
import csv
from typing import Any
from pathlib import Path
from statistics import mean, median

import torch

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
