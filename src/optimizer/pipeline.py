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



def get_project_dir(gpu_name: str, optional_name: str = None):
    letters = string.ascii_letters + string.digits
    proj_name = ''.join(random.choices(letters, k=10))
    if optional_name:
        proj_name = optional_name
    
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


def save_iteration(paths: dict, parent_info: KernelNode, improvement_description: str, best_kernel_code: str, ssh_config: dict = None):
    """Profiles iteration and records performance results
    """
    next_id = len(list(paths["proj_dir"].glob("nodes/*.json")))

    # Export attempt's code
    current_kernel_code = (paths["tmp_dir"] / "kernel.cu").read_text()
    with open(paths["proj_dir"] / "attempts" / f"kernel_{next_id}.cu", "w") as f:
        f.write(current_kernel_code)

    # Log the attempt with results
    print("\tBeginning Profiler...")
    if ssh_config:
        current_stats, profiler = gpu.profile_remote_kernel(ssh_config, paths)
    else:
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
            "code": str(Path(paths["proj_dir"].name) / "attempts" / f"kernel_{next_id}.cu"),
            "visits": 1
        }
        # Validate and dump with Pydantic
        node_obj = KernelNode.model_validate(node_val)
        f.write(node_obj.model_dump_json(indent=4, by_alias=True))

    next_id += 1
    return log_entry


def optimize(gpu_specs: GPUSpecs, paths: dict[str, Path], parent_node: KernelNode, ssh_config: dict = None):
    """Optimizes target kernel
    """

    # Create attempts directory
    (paths["proj_dir"] / "attempts").mkdir(parents=True, exist_ok=True)
    (paths["proj_dir"] / "nodes").mkdir(parents=True, exist_ok=True)

    # Collect improvement history from tree (walk from parent to root)
    improvement_log, ancestor_codes = mcts.collect_ancestry(
        paths, 
        parent_node, 
        code_depth=settings.ancestor_code_depth
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths["tmp_dir"] = Path(tmpdir)

        # Read the actual kernel code from file path
        kernel_code_path = Path(parent_node.code)
        if not kernel_code_path.is_absolute():
            kernel_code_path = paths["proj_dir"].parent / kernel_code_path

        if kernel_code_path.exists():
            kernel_code = kernel_code_path.read_text()
        else:
            print(f"\t\tError: Kernel code file not found: {kernel_code_path}")
            return None

        # Kernel Generation
        print(f"\tBeginning generation (history: {len(improvement_log)} entries)...")
        improvement_description, is_valid = generator.generate(
            kernel_code, gpu_specs, improvement_log, paths, ancestor_codes=ancestor_codes, ssh_config=ssh_config)
        print("\tFinished generation.")
        print(f"\t\t- Status: {is_valid}")

        # If its valid, log it and return the new node
        if is_valid:
            log_entry = save_iteration(
                paths, parent_node, improvement_description, str(paths["proj_dir"] / "attempts" / f"kernel_{parent_node.id}.cu"), ssh_config=ssh_config)
            
            # Load and return the newly created node
            new_node_id = len(list((paths["proj_dir"] / "nodes").glob("*.json"))) - 1
            new_node_path = paths["proj_dir"] / "nodes" / f"{new_node_id}.json"
            if new_node_path.exists():
                with open(new_node_path, 'r') as f:
                    return mcts.KernelNode.model_validate(json.load(f))
    
    return None


def create_project(gpu_specs: GPUSpecs, io_parent_dir: Path, optional_proj_name: str = None, ssh_config: dict = None):
    """Creates a new optimization project for each individual operator kernel.
    """
    # Output directory (access via dot notation now)
    proj_dir = get_project_dir(gpu_specs.gpu_name, optional_proj_name)

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
            if ssh_config:
                 current_stats, profiler = gpu.profile_remote_kernel(ssh_config, paths, baseline=True)
            else:
                 current_stats, profiler = gpu.profile_kernel(paths, baseline=True)

            # Log kernel
            node_data = {
                "id": 0,
                "value": current_stats['mean_time_ms'],
                "speedup_vs_parent": 1.0,
                "improvement_description": "Initial",
                "parent": -1,
                "code": str(Path(proj_op_dir.name) / "attempts" / "kernel_0.cu"),
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
    
    # Needs argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="CGinS Optimization Pipeline")
    parser.add_argument("io_dir", type=str, help="Directory containing IO pairs")
    parser.add_argument("proj_name", type=str, nargs="?", default=None, help="Optional project name")
    parser.add_argument("--remote", type=str, help="Path to configuration JSON for remote remote execution")
    
    args = parser.parse_args()
    
    io_parent_dir = Path(args.io_dir)
    optional_proj_name = args.proj_name
    
    ssh_config = None
    
    if args.remote:
        config_path = Path(args.remote)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
            
        with open(config_path, "r") as f:
            remote_config = json.load(f)
            
        # Get active SSH config
        connections = remote_config.get("ssh_connections", [])
        active_idx = remote_config.get("active_ssh_index", -1)
        
        if 0 <= active_idx < len(connections):
            ssh_config = connections[active_idx]
            print(f"remote Mode: Using SSH connection to {ssh_config.get('host')}")
        else:
            print("Error: Invalid active_ssh_index in remote config.")
            sys.exit(1)
            
        # Collect GPU specs remotely
        print("Retrieving remote GPU specs...")
        gpu_specs = gpu.get_remote_gpu_specs(ssh_config)
    else:
        # Collect GPU specs locally
        gpu_specs = gpu.get_gpu_specs()

    # Create project (or resume if exists/provided)
    proj_dir = create_project(gpu_specs, io_parent_dir, optional_proj_name, ssh_config)

    for op_dir_path in proj_dir.iterdir():
        if not op_dir_path.is_dir() or "max" not in str(op_dir_path):
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

        mcts._NODE_CACHE.clear()

        for _ in range(100):
            # Select parent node then optimize off of it
            parent_node = mcts.choose_optimization(paths)
            new_node = optimize(gpu_specs, paths, parent_node, ssh_config)
            
            # Update tree with the new node (if optimization succeeded)
            if new_node:
                mcts.update_tree(paths, new_node)

if __name__ == "__main__":
    main()
