import os
import sys
import json
import math
import random
import string
import tempfile
from pathlib import Path

import src.optimizer.components.llm.generator as generator
import src.optimizer.components.hardware.profiler as gpu
import src.optimizer.core.mcts as mcts
from src.optimizer.core.types import KernelNode, GPUSpecs
from src.optimizer.config.settings import settings


def get_project_dir(gpu_name: str):

    # Determine project name (random or user requested)
    letters = string.ascii_letters + string.digits
    proj_name = ''.join(random.choices(letters, k=10))
    if len(sys.argv) >= 3:
        proj_name = sys.argv[2]
    
    # Sanitize GPU name
    clean_gpu_name = gpu_name.replace(
        " ", "_").replace(":", "").replace("-", "_")

    full_name = f"{clean_gpu_name}_{proj_name}"

    print(f"Beginning optimizing on project {full_name}...")

    proj_dir = Path(f"kernels/optimized/{full_name}")
    try:
        proj_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(
            f"Error: Project {full_name} creation failed: {e}")

    return proj_dir


def save_iteration(paths: dict, parent_info: KernelNode, improvement_description: str, best_kernel_code: str):
    """Profiles iteration and records performance results

    Args:
        paths (dict): path objects to various filepaths
        parent_info (KernelNode): 
        improvement_description (str): _description_
        best_kernel_code (str): _description_

    Returns:
        _type_: _description_
    """
    next_id = len(list(paths["proj_dir"].glob("nodes/*.json")))

    # Export attempt's code
    current_kernel_code = (paths["tmp_dir"] / "kernel.cu").read_text()
    with open(paths["proj_dir"] / "attempts" / f"kernel_{next_id}.cu", "w") as f:
        f.write(current_kernel_code)

    # Log the attempt with results
    print("\tBeginning Profiler...")
    current_stats, profiler = gpu.profile_kernel(paths)
    print("\tFinished Profiler.")
    log_entry = {
        "iteration": next_id,
        "attempted": improvement_description,
        "results": current_stats,
        "speedup_vs_parent": parent_info.value / current_stats['mean_time_ms'],
    }

    # Export attempt as json
    with open(paths["proj_dir"] / "nodes" / f"{next_id}.json", "w") as f:
        node_val = {
            "id": next_id,
            "value": current_stats['mean_time_ms'],
            "speedup_vs_parent": log_entry['speedup_vs_parent'],
            "improvement_description": improvement_description,
            "parent": parent_info.id,
            "code": str(paths["proj_dir"] / "attempts" / f"kernel_{next_id}.cu"),
            "visits": 1
        }
        # Validate and dump with Pydantic
        node_obj = KernelNode.model_validate(node_val)
        f.write(node_obj.model_dump_json(indent=4, by_alias=True))

    next_id += 1
    return log_entry


def optimize(gpu_specs: GPUSpecs, paths: dict[str, Path], parent_node: KernelNode):
    """Optimizes target kernel

    Args:
        gpu_specs (GPUSpecs): GPU specs
        paths (dict[str, Path]): paths to directories
        parent_node (KernelNode): node to optimize off of
    """

    # Create attempts directory
    (paths["proj_dir"] / "attempts").mkdir(parents=True, exist_ok=True)
    (paths["proj_dir"] / "nodes").mkdir(parents=True, exist_ok=True)

    # Iterative refinement loop
    # Step 1. Generate:
    #   Generate kernel: Via LLM generate new kernel using the gpu specs & previously found best kernel
    #   Verification loop: Check if new kernel can handle expected input/output pairs, if not attempt to fix (max 3 times)
    # Step 2. Profile:
    #   Profile Kernel: Run kernel and measure inference time and memory management
    #   Log kernel: If kernel performs better, have LLM attempt to explain what this improvement did better and give extra context (max 3 times)

    # TODO: Add improvement log
    improvement_log = []

    with tempfile.TemporaryDirectory() as tmpdir:
        paths["tmp_dir"] = Path(tmpdir)

        # Kernel Generation
        print("\tBeginning generation...")
        improvement_description, is_valid = generator.generate(
            parent_node.code, gpu_specs, improvement_log, paths)
        print("\tFinished generation.")
        print(f"\t\t- Status: {is_valid}")

        # If its valid, log it
        if is_valid:
            log_entry = save_iteration(
                paths, parent_node, improvement_description, str(paths["proj_dir"] / "attempts" / f"kernel_{parent_node.id}.cu"))


def create_project(gpu_specs: GPUSpecs, io_parent_dir: Path):
    """Creates a new optimization project for each individual operator kernel.

    Args:
        gpu_specs (GPUSpecs): GPU specs
        io_parent_dir (Path): Path to directory containing input/output torch files

    Returns:
        Path: Path to project directory
    """
    # Output directory (access via dot notation now)
    proj_dir = get_project_dir(gpu_specs.gpu_name)

    # Directory containing initial wave of correct, but unoptimized kernels
    op_dirs = list(Path("kernels/generated/individual_op_kernels").glob("*"))

    # Profile each kernel in op_dirs and save results as a *.json node in their respective directories
    for op_dir in op_dirs:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            io_dir = io_parent_dir / op_dir.name

            if not io_dir.exists():
                print(f"{op_dir.name} has no i/o")
                continue

            # Check if the kernel has been previously generated
            if not (op_dir / "success").exists():
                print(
                    f"Skipping {op_dir.name}: No 'success' marker found (generation failed).")
                continue

            # Prepare project op directory
            proj_op_dir = proj_dir / op_dir.name
            proj_op_dir.mkdir(parents=True, exist_ok=True)
            (proj_op_dir / "attempts").mkdir(parents=True, exist_ok=True)
            (proj_op_dir / "nodes").mkdir(parents=True, exist_ok=True)

            # Skip if already initialized
            if (proj_op_dir / "nodes" / "0.json").exists():
                print(
                    f"{op_dir.name} already initialized, skipping baseline profile.")
                continue

            paths = {
                "tmp_dir": tmp_path,
                "io_dir": io_dir,
                "proj_dir": proj_op_dir
            }

            # Copy kernel to tmp_dir for profiling
            (tmp_path / "kernel.cu").write_text((op_dir / "kernel.cu").read_text())

            # Profile kernel
            current_stats, profiler = gpu.profile_kernel(paths, baseline=True)

            # Log kernel
            node_data = {
                "id": 0,
                "value": current_stats['mean_time_ms'],
                "speedup_vs_parent": 1.0,
                "improvement_description": "Initial",
                "parent": -1,
                "code": str(proj_op_dir / "attempts" / "kernel_0.cu"),
                "visits": 1
            }
            node = KernelNode.model_validate(node_data)
            
            # Save root node manually to ensure parent is -1
            with open(paths["proj_dir"] / "nodes" / "0.json", "w") as f:
                f.write(node.model_dump_json(indent=4, by_alias=True))
                
            # Copy kernel to attempts manually as well
            (paths["proj_dir"] / "attempts").mkdir(parents=True, exist_ok=True)
            with open(paths["proj_dir"] / "attempts" / "kernel_0.cu", "w") as f:
                f.write((op_dir / "kernel.cu").read_text())

    return proj_dir


def main():
    """Calls optimization pipeline on each kernel."""
    global next_id

    # Directory of Torch files containing formatted input/output pairs (for specific model preferably)
    if len(sys.argv) < 2:
        print("Missing input/output torch dir")
        print(
            "`python3 -m src.optimizer.pipeline <io_directory> <optional_project_name>`")
        sys.exit(1)
    io_parent_dir = Path(sys.argv[1])

    # Collect GPU specs first to get name
    gpu_specs = gpu.get_gpu_specs()

    # Create project (or resume if exists/provided)
    proj_dir = create_project(gpu_specs, io_parent_dir)

    for op_dir_path in proj_dir.iterdir():
        if not op_dir_path.is_dir():
            continue

        op_name = op_dir_path.name
        print(f"Optimizing {op_name}...")

        # Check if IO exists
        io_dir = io_parent_dir / op_name
        if not io_dir.exists():
            print(f"  Skipping {op_name}: IO directory not found")
            continue

        # Paths for optimization
        paths = {
            "proj_dir": op_dir_path,
            "io_dir": io_dir,
            "op_dir": Path("kernels/generated/individual_op_kernels") / op_name,
        }

        # Select parent node then optimize off of it
        # mcts.choose_optimization might need C constant which is default in settings now.
        parent_node = mcts.choose_optimization(paths)
        optimize(gpu_specs, paths, parent_node)
        
        # Pass paths to update_tree (it handles dict now)
        mcts.update_tree(paths, parent_node)

if __name__ == "__main__":
    main()
