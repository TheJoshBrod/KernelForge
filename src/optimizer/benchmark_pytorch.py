
import glob
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Import load_batch from GPUprofiler
# Assuming run as: python3 -m src.optimizer.benchmark_pytorch
from src.optimizer.components.hardware.profiler import load_batch, get_gpu_specs


def _resolve_device() -> str:
    value = os.environ.get("CGINS_TARGET_DEVICE", "").strip().lower()
    if value == "mps" and hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    if value in {"gpu", "cuda"} and torch.cuda.is_available():
        return "cuda"
    if value == "cpu":
        return "cpu"
    if hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

def get_optimized_dir_path():
    """Finds the optimization directory for coverage."""
    # Try to find the specific directory we've been working with
    base_path = Path("kernels/optimized")
    if not base_path.exists():
        print(f"Error: {base_path} not found.")
        return None

    # Heuristic: Pick the most recently modified directory or the one matching current GPU
    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dirs:
        print("No optimization directories found.")
        return None

    # Just take the first one or the one matching known string if possible
    # In this specific env, we know it's NVIDIA_GeForce_RTX_3050_Laptop_GPU_MywUC7tnvC
    target = "NVIDIA_GeForce_RTX_3050_Laptop_GPU_MywUC7tnvC"
    for d in dirs:
        if target in d.name:
            return d

    return dirs[0]


def get_pytorch_func(op_name):
    """Maps folder name to torch function."""
    mapping = {
        "torch_nn_functional_relu": F.relu,
        "torch_nn_functional_linear": F.linear,
        "torch_nn_functional_layer_norm": F.layer_norm,
        "torch_nn_functional_embedding": F.embedding,
        "torch_nn_functional_dropout": F.dropout,
        "torch_nn_functional_batch_norm": F.batch_norm,
        "torch_nn_functional_gelu": F.gelu,
        "torch_nn_functional_scaled_dot_product_attention": F.scaled_dot_product_attention,
        "torch_nn_functional_softmax": F.softmax,
        "torch_nn_functional_adaptive_avg_pool1d": F.adaptive_avg_pool1d,
        "torch_nn_functional_adaptive_avg_pool2d": F.adaptive_avg_pool2d,
        "torch_nn_functional_max_pool2d": F.max_pool2d,
        "torch_nn_functional_pad": F.pad,
        "torch_nn_functional_conv2d": F.conv2d,
    }
    return mapping.get(op_name)

def measure_pytorch(func, inputs):
    """Measures execution time of pytorch function."""
    timings = []
    device = _resolve_device()
    
    # Warmup
    for args, kwargs in inputs:
        try:
            func(*args, **kwargs)
        except Exception as e:
            pass # Ignore warmup errors
    _sync_device(device)
    
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        for args, kwargs in inputs:
            start.record()
            for _ in range(10):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    pass
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / 10)
    else:
        for args, kwargs in inputs:
            start_time = time.perf_counter()
            for _ in range(10):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    pass
            _sync_device(device)
            timings.append(((time.perf_counter() - start_time) * 1000.0) / 10)
        
    return float(np.mean(timings))


def main():
    print(f"{'='*80}")
    print(f"{'OPERATOR':<40} | {'PYTORCH (ms)':<12} | {'OPTIMIZED (ms)':<14} | {'SPEEDUP':<10}")
    print(f"{'-'*80}")

    opt_base_dir = get_optimized_dir_path()
    if not opt_base_dir:
        return

    # Location of input files
    bench_data_root = Path("benchmarks/profiler/individual_ops")

    operators = [d.name for d in opt_base_dir.iterdir() if d.is_dir()]
    operators.sort()

    for op_name in operators:
        # 1. Get PyTorch Function
        func = get_pytorch_func(op_name)
        if not func:
            continue

        # 2. Load Inputs
        io_dir = bench_data_root / op_name
        if not io_dir.exists():
            continue

        try:
            pt_files = sorted(glob.glob(os.path.join(io_dir, "entry_*.pt")))
            # Limit to small batch for speed
            pt_files = pt_files[:50]
            inputs = load_batch(pt_files)
        except Exception as e:
            print(f"Failed to load inputs for {op_name}: {e}")
            continue

        if not inputs:
            continue

        # 3. Measure PyTorch Time
        try:
            pytorch_time = measure_pytorch(func, inputs)
        except Exception as e:
            print(f"Error measuring {op_name}: {e}")
            pytorch_time = 0.0

        # 4. Get Optimized Time
        log_file = opt_base_dir / op_name / "improvement_log.json"
        optimized_time = 0.0
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    # Find the best time (is_best=True)
                    best_entry = next(
                        (item for item in logs if item.get('is_best')), None)
                    # If multiple is_best (shouldn't happen but fallback to last or min)
                    if not best_entry:
                        # fallback to minimum mean_time_ms
                        best_entry = min(
                            logs, key=lambda x: x['results']['mean_time_ms'])

                    optimized_time = best_entry['results']['mean_time_ms']
            except Exception as e:
                pass

        # 5. Report
        speedup_str = "-"
        if optimized_time > 0 and pytorch_time > 0:
            speedup = pytorch_time / optimized_time
            speedup_str = f"{speedup:.2f}x"

        print(
            f"{op_name:<40} | {pytorch_time:<12.4f} | {optimized_time:<14.4f} | {speedup_str:<10}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
