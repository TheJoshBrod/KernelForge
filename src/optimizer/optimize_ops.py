"""
Optimizes individual Ops for target GPU architecture
"""
import tempfile
from pathlib import Path

import src.optimizer.GPUprofiler as gpu
import src.optimizer.generator as generator


def optimization_loop(gpu_specs: dict, op_dir: Path, io_dir: Path):
    """Optimizes target kernel

    Args:
        op_dir (str): Directory of op
    """

    # Baseline measure performance
    baseline_stats = gpu.profile_kernel(op_dir, baseline=True)

    # Iterative refinement loop
    # Step 1. Generate:
    #   Generate kernel: Via LLM generate new kernel using the gpu specs & previously found best kernel
    #   Verification loop: Check if new kernel can handle expected input/output pairs, if not attempt to fix (max 3 times) 
    # Step 2. Profile:
    #   Profile Kernel: Run kernel and measure inference time and memory management
    #   Log kernel: If kernel performs better, have LLM attempt to explain what this improvement did better and give extra context (pass that to next iteration)

    improvement_log = []
    for _ in range(10):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path: Path = Path(tmpdir)
        
            generator.generate(gpu_specs, op_dir, improvement_log, tmpdir_path, io_dir)
            baseline_stats, feedback = gpu.profile_kernel(op_dir, baseline=True, previous_stats=baseline_stats)
            improvement_log.append(feedback)
    
    return

def get_io_dir(op_dir: Path):
    io_dir = Path(f"benchmarks/profiler/individual_ops/{op_dir.name}")
    return io_dir 

def main():
    """Calls optimization pipeline on each kernel."""

    # Collect GPU specs
    gpu_specs = gpu.get_gpu_specs()

    # Optimize each individual op
    op_dirs = list(Path("kernels/generated/individual_op_kernels").glob("*"))
    for op_dir in op_dirs:
        io_dir = get_io_dir(op_dir)
        optimization_loop(gpu_specs, op_dir, io_dir)


if __name__ == "__main__":
    main() 