"""
Optimizes individual Ops for target GPU architecture
"""
import json
import random
import string
import sys
import tempfile
from pathlib import Path

import src.optimizer.generator as generator
import src.optimizer.GPUprofiler as gpu


def get_project_dir(gpu_name: str):

    # Determine project name (random or user requested)
    letters = string.ascii_letters + string.digits
    proj_name = ''.join(random.choices(letters, k=10))
    if len(sys.argv) >= 3:
        proj_name = sys.argv[2]
    
    # Sanitize GPU name
    clean_gpu_name = gpu_name.replace(" ", "_").replace(":", "").replace("-", "_")
    
    full_name = f"{clean_gpu_name}_{proj_name}"

    print(f"Beginning optimizing on project {full_name}...")

    proj_dir = Path(f"kernels/optimized/{full_name}")
    try:
        proj_dir.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        print(
            f"Error: Project {full_name} name already exists, please pick a new name")

    return proj_dir


def optimization_loop(gpu_specs: dict, paths: dict[str, Path]):
    """Optimizes target kernel

    Args:
        op_dir (str): Directory of op
    """

    # Baseline measure performance
    print("\nMeasuring baseline...")
    with tempfile.TemporaryDirectory() as tmpdir:
        paths["tmp_dir"] = Path(tmpdir)
        import shutil
        shutil.copy(paths["op_dir"] / "kernel.cu", Path(tmpdir) / "kernel.cu")
        baseline_stats, profiler = gpu.profile_kernel(paths, baseline=True)
        best_stats = baseline_stats.copy()
        best_kernel_code = (paths["op_dir"] / "kernel.cu").read_text()
    print("Finished baseline.")
    
    # Create attempts directory
    (paths["proj_dir"] / "attempts").mkdir(parents=True, exist_ok=True)

    # Iterative refinement loop
    # Step 1. Generate:
    #   Generate kernel: Via LLM generate new kernel using the gpu specs & previously found best kernel
    #   Verification loop: Check if new kernel can handle expected input/output pairs, if not attempt to fix (max 3 times)
    # Step 2. Profile:
    #   Profile Kernel: Run kernel and measure inference time and memory management
    #   Log kernel: If kernel performs better, have LLM attempt to explain what this improvement did better and give extra context (pass that to next iteration)

    improvement_log = []
    for iteration in range(10):
        print(f"\nIteration {iteration}:")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths["tmp_dir"] = Path(tmpdir)

            print("\tBeginning generation...")
            improvement_description, is_valid = generator.generate(
                best_kernel_code, gpu_specs, improvement_log, paths)
            print("\tFinished generation.")
            print(f"\t\t- Desc: {improvement_description}")
            print(f"\t\t- Status: {is_valid}")

            if is_valid:
                # Save this attempt
                current_kernel_code = (paths["tmp_dir"] / "kernel.cu").read_text()
                with open(paths["proj_dir"] / "attempts" / f"kernel_{iteration}.cu", "w") as f:
                    f.write(current_kernel_code)

                # Log the attempt with results
                print("\tBeginning Profiler...")
                current_stats, profiler = gpu.profile_kernel(paths)
                print("\tFinished Profiler.")
                log_entry = {
                    "iteration": iteration,
                    "attempted": improvement_description,
                    "results": current_stats,
                    "speedup_vs_baseline": baseline_stats['mean_time_ms'] / current_stats['mean_time_ms'],
                    "speedup_vs_best": best_stats['mean_time_ms'] / current_stats['mean_time_ms']
                }
                print(f"\t\t- stats: {log_entry['speedup_vs_best']}")
                if current_stats['mean_time_ms'] < best_stats['mean_time_ms']:
                    log_entry["is_best"] = True
                    best_stats = current_stats.copy()
                    best_kernel_code = current_kernel_code
                else:
                    log_entry["is_best"] = False
                improvement_log.append(log_entry)

    with open(paths["proj_dir"] / "improvement_log.json", 'w') as f:
        json.dump(improvement_log, f, indent=2)

    # Save the final best kernel
    with open(paths["proj_dir"] / "kernel.cu", "w") as f:
        f.write(best_kernel_code)

    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_stats['mean_time_ms']:.3f} ms")
    print(f"Best:     {best_stats['mean_time_ms']:.3f} ms")
    print(
        f"Speedup:  {baseline_stats['mean_time_ms'] / best_stats['mean_time_ms']:.2f}x")


def main():
    """Calls optimization pipeline on each kernel."""

    # Directory of Torch files containing formatted input/output pairs (for specific model preferably)
    if len(sys.argv) < 2:
        print("Missing input/output torch dir")
        print(
            "`python3 -m src.optimizer.optimize_ops <io_directory> <optional_project_name>`")
        sys.exit(1)
    io_parent_dir = Path(sys.argv[1])

    # Collect GPU specs first to get name
    gpu_specs = gpu.get_gpu_specs()

    # Optimization: Set the CUDA Architecture to the specific device to speed up JIT compilation
    import os
    os.environ["TORCH_CUDA_ARCH_LIST"] = gpu_specs["compute_capability"]

    # Output directory
    proj_dir = get_project_dir(gpu_specs["gpu_name"])

    # Directory containing initial wave of correct, but unoptimized kernels
    op_dirs = list(Path("kernels/generated/individual_op_kernels").glob("*"))
    # Prioritize attention as requested
    op_dirs.sort()

    # For each kernel, run optimization loop
    for op_dir in op_dirs:

        io_dir = io_parent_dir / op_dir.name
        if not io_dir.exists():
            print(f"{op_dir.name} has no i/o")
            continue

        # Check if the kernel was successfully generated
        if not (op_dir / "success").exists():
            print(f"Skipping {op_dir.name}: No 'success' marker found (generation failed).")
            continue
        proj_op_dir = proj_dir / op_dir.name
        proj_op_dir.mkdir(parents=True, exist_ok=False)
        paths = {
            "proj_dir": proj_op_dir,
            "io_dir": io_dir,
            "op_dir": op_dir
        }
        optimization_loop(gpu_specs, paths)


if __name__ == "__main__":
    main()
