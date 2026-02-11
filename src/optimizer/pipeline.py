import sys
import json
import random
import string
import argparse
import tempfile
import queue
from pathlib import Path

import src.optimizer.components.llm.generator as generator
import src.optimizer.components.llm.prompts as opt_prompts
import src.optimizer.components.hardware.profiler as gpu
import src.optimizer.core.mcts as mcts
import src.generator.prompts.prompts as gen_prompts
from src.optimizer.core.types import KernelNode, GPUSpecs
from src.optimizer.config.settings import settings
import torch



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
            kernel_code_path = Path.cwd() / kernel_code_path

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


def create_new_root(gpu_specs: GPUSpecs, paths: dict[str, Path]) -> KernelNode:
    """Generate a fresh kernel as an independent root node.
    
    Creates a new optimization tree separate from existing ones by generating
    a kernel from scratch with a different approach.
    
    Args:
        gpu_specs: GPU architecture specifications
        paths: Dictionary containing project paths
        
    Returns:
        The newly created root KernelNode, or None if generation failed
    """
    from src.llm_tools import GenModel
    
    # Get next available node ID
    next_id = len(list((paths["proj_dir"] / "nodes").glob("*.json")))
    
    # Get existing roots to show LLM for diversity
    existing_roots = mcts.get_existing_roots(paths)
    print(f"\tFound {len(existing_roots)} existing root(s)")
    
    # Load operator specification from entry files (same as generator)
    op_name = paths["proj_dir"].name
    entry_files = sorted(paths["io_dir"].glob("entry_*.pt"))
    
    if entry_files:
        # Load all entry files to get function calls (same as generator main.py)
        call_list = []
        for entry_file in entry_files:
            try:
                entry = torch.load(entry_file, map_location='cpu', weights_only=False)
                call_list.append(entry)
            except Exception as e:
                print(f"\t\tWarning: Error loading {entry_file}: {e}")
                continue
        
        if call_list:
            # Get function name from first entry
            function_name = call_list[0].get("function_name", op_name)
            operator_spec = gen_prompts.generate_function_spec_from_calls(call_list, function_name)
        else:
            operator_spec = {"function_name": op_name, "parameters": [], "num_calls": 0}
    else:
        # Fallback minimal spec
        operator_spec = {"function_name": op_name, "parameters": [], "num_calls": 0}
    
    # Generate prompt with existing roots for diversity guidance
    prompt = opt_prompts.generate_new_root_prompt(
        operator_spec,
        existing_roots
    )
    
    # Get generator system prompt
    sys_prompt = gen_prompts.get_system_prompt()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths["tmp_dir"] = Path(tmpdir)
        
        # DEBUG: Save prompt for inspection
        (paths["proj_dir"] / "attempts").mkdir(parents=True, exist_ok=True)
        prompt_path = paths["proj_dir"] / "attempts" / f"new_root_prompt_{next_id}.md"
        with open(prompt_path, "w") as f:
            f.write("# System Prompt\n\n")
            f.write(sys_prompt)
            f.write("\n\n---\n\n# User Message\n\n")
            f.write(prompt)
        print(f"\t\tSaved prompt to: {prompt_path}")
        
        # Generate kernel using LLM with retry logic
        llm = GenModel(sys_prompt)
        
        # Initial attempt
        feedback, code = generator.extract_feedback_and_code(
            llm.chat(prompt, settings.llm_model_name)
        )
        
        if code is None:
            print("\t\tError: Failed to extract code from LLM response")
            return None
        
        # Write to tmp for validation
        (paths["tmp_dir"] / "kernel.cu").write_text(code)
        
        # Validate the kernel with retries
        import src.optimizer.components.worker.verifier as verifier
        is_valid, error = verifier.validate_kernel(code, paths)
        
        # Retry loop on validation failure
        attempt = 0
        while not is_valid and attempt < settings.retry_limit:
            attempt += 1
            print(f"\t\tValidation failed, retry {attempt}/{settings.retry_limit}...")
            
            # Save failed attempt to garbage dump
            dump_dir = paths["proj_dir"] / "garbage_dump"
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / f"new_root_{next_id}_attempt{attempt}.cu").write_text(code)
            
            # Send error back to LLM for correction
            feedback, code = generator.extract_feedback_and_code(
                llm.chat(error, settings.llm_model_name)
            )
            
            if code is None:
                print(f"\t\tRetry {attempt}: Failed to extract code")
                continue
            
            # Validate the new attempt
            (paths["tmp_dir"] / "kernel.cu").write_text(code)
            is_valid, error = verifier.validate_kernel(code, paths)
        
        if not is_valid:
            print(f"\t\tValidation failed after {settings.retry_limit} retries: {error}")
            dump_dir = paths["proj_dir"] / "garbage_dump"
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / f"new_root_{next_id}_final_failed.cu").write_text(code)
            return None
        
        # Profile the kernel
        print("\t\tProfiling new root kernel...")
        current_stats, _ = gpu.profile_kernel(paths)
        
        # Create root node (parent = -1)
        node_data = {
            "id": next_id,
            "parent": -1,  # Root marker
            "value": current_stats['mean_time_ms'],
            "speedup_vs_parent": 1.0,
            "improvement_description": "Initial",
            "code": f"{paths['proj_dir'].name}/attempts/kernel_{next_id}.cu",
            "visits": 1
        }
        node = KernelNode.model_validate(node_data)
        
        # Save node
        (paths["proj_dir"] / "nodes").mkdir(parents=True, exist_ok=True)
        with open(paths["proj_dir"] / "nodes" / f"{next_id}.json", "w") as f:
            f.write(node.model_dump_json(indent=4, by_alias=True))
        
        # Save kernel code
        (paths["proj_dir"] / "attempts" / f"kernel_{next_id}.cu").write_text(code)
        
        print(f"\t\tCreated new root: Node {next_id} ({current_stats['mean_time_ms']:.4f} ms)")
        return node


def create_project(gpu_specs: GPUSpecs, io_parent_dir: Path):
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

def run_parallel_optimization(gpu_specs: GPUSpecs, paths: dict, n_workers: int = 4, max_iterations: int = 100):
    """Run parallel MCTS optimization using multiprocessing.
    
    Dispatches nodes to workers as they become available, stops after max_iterations.
    
    Args:
        gpu_specs: GPU specifications
        paths: Dictionary containing project paths
        n_workers: Number of worker processes
        max_iterations: Total number of nodes to process before stopping
    """
    import multiprocessing as mp
    from multiprocessing import Process
    import time
    from src.optimizer.components.worker.parallel_worker import worker_routine
    
    # CUDA requires 'spawn' start method (default 'fork' doesn't work with CUDA)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print(f"\n{'='*60}")
    print(f"PARALLEL MCTS OPTIMIZATION")
    print(f"  Workers: {n_workers}")
    print(f"  Max Iterations: {max_iterations}")
    print(f"  Project: {paths['proj_dir']}")
    print(f"{'='*60}\n")
    
    # Shared state via Manager
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    gpu_lock = manager.Lock()
    
    # Shared counter for sequential node IDs (start from current node count)
    initial_count = len(list((paths["proj_dir"] / "nodes").glob("*.json")))
    node_counter = mp.Value('i', initial_count)  # Atomic integer counter
    
    in_flight_ids = set()
    nodes_dispatched = 0
    nodes_completed = 0
    nodes_failed = 0
    
    # Start persistent workers
    print(f"[INIT] Starting {n_workers} workers...")
    workers = []
    for i in range(n_workers):
        p = Process(target=worker_routine, args=(task_queue, result_queue, gpu_lock, node_counter, paths))
        p.start()
        workers.append(p)
    
    # Ensure tree is loaded
    mcts.load_tree_once(paths)
    print(f"[INIT] Loaded {len(mcts._NODE_CACHE)} nodes from tree")
    
    # Main loop: dispatch until limit, then drain remaining
    print(f"\n[LOOP] Starting optimization loop...")
    
    start_time = time.time()
    dispatch_blocked = False
    
    while nodes_dispatched < max_iterations or in_flight_ids:
        
        # === DISPATCH (only if under limit and not blocked) ===
        dispatched_this_round = False
        while len(in_flight_ids) < n_workers and nodes_dispatched < max_iterations:
            try:
                node = mcts.choose_optimization(paths, exclude_ids=in_flight_ids)
            except ValueError:
                if not dispatch_blocked:
                    print(f"[DISPATCH] Tree exhausted, waiting for workers...")
                    dispatch_blocked = True
                break
            
            if node is None or node.id in in_flight_ids:
                if not dispatch_blocked:
                    print(f"[DISPATCH] Waiting for capacity...")
                    dispatch_blocked = True
                break
            
            # Build context for worker
            history, codes = mcts.collect_ancestry(paths, node)
            context = {
                "history": history,
                "codes": codes,
                "gpu_specs": gpu_specs,
                "paths": paths,
            }
            
            task_queue.put((node, context))
            in_flight_ids.add(node.id)
            nodes_dispatched += 1
            dispatched_this_round = True
            dispatch_blocked = False
            
            elapsed = time.time() - start_time
            print(f"[DISPATCH] Node {node.id} ({nodes_dispatched}/{max_iterations}) | In-flight: {len(in_flight_ids)} | {elapsed:.0f}s")
        
        # === COLLECT completed results ===
        results_collected = False
        while True:
            try:
                node_id, result_data, status = result_queue.get(timeout=0.2)
            except queue.Empty:
                break
            except Exception as e:
                print(f"[ERROR] Result collection: {e}")
                break
            
            in_flight_ids.discard(node_id)
            results_collected = True
            dispatch_blocked = False
            
            if status == "success" and result_data is not None:
                nodes_completed += 1
                runtime_ms = result_data["runtime_ms"]
                kernel_id = result_data["kernel_id"]
                
                parent_node = mcts._NODE_CACHE.get(node_id)
                if parent_node:
                    new_node = KernelNode(
                        id=kernel_id,
                        parent_id=node_id,
                        children_ids=[],
                        visits=1,
                        value=runtime_ms,
                        best_subtree_value=runtime_ms,
                        speedup_vs_parent=parent_node.value / runtime_ms if runtime_ms > 0 else 1.0,
                        improvement_description=result_data.get("feedback", "Parallel optimization"),
                        code=result_data["code_path"]
                    )
                    mcts.update_tree(paths, new_node)
                    
                    speedup = parent_node.value / runtime_ms if runtime_ms > 0 else 1.0
                    print(f"[SUCCESS] Node {node_id} -> {kernel_id} | {runtime_ms:.4f}ms | {speedup:.2f}x | Done: {nodes_completed}")
            else:
                nodes_failed += 1
                print(f"[FAILED] Node {node_id}: {status}")
        
        # If nothing happened this round and we have in-flight work, wait
        if not dispatched_this_round and not results_collected and in_flight_ids:
            time.sleep(0.5)
        elif not in_flight_ids and nodes_dispatched >= max_iterations:
            break  # All done
    
    # Cleanup: send sentinel to each worker for clean exit
    print(f"\n[CLEANUP] Sending exit signals to workers...")
    for _ in workers:
        task_queue.put(None)
    
    for i, w in enumerate(workers):
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  Nodes Dispatched: {nodes_dispatched}")
    print(f"  Nodes Completed: {nodes_completed}")
    print(f"  Nodes Failed: {nodes_failed}")
    print(f"  Success Rate: {nodes_completed/(nodes_dispatched or 1)*100:.1f}%")
    print(f"{'='*60}\n")

def main():
    """Calls optimization pipeline on each kernel."""
    global next_id

    parser = argparse.ArgumentParser(
        description="CUDA Kernel Optimizer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal optimization run
  python3 -m src.optimizer.pipeline benchmarks/profiler/individual_ops project_name
  
  # Create a new independent root for a specific operator
  python3 -m src.optimizer.pipeline benchmarks/profiler/individual_ops project_name --new-root torch_nn_functional_embedding

  # Run using remote SSH configuration
  python3 -m src.optimizer.pipeline benchmarks/profiler/individual_ops project_name --remote config.json
"""
    )

    parser.add_argument(
        "io_dir",
        type=Path,
        help="Directory containing input/output torch files"
    )

    parser.add_argument(
        "project_name",
        nargs="?",
        default=None,
        help="Optional project name"
    )

    parser.add_argument(
        "--new-root",
        type=str,
        metavar="OP_NAME",
        help="Create a new independent root for the specified operator (no optimization)"
    )

    parser.add_argument(
        "--remote",
        type=Path,
        help="Path to configuration JSON for remote execution"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel optimization mode"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel mode (default: 4)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of nodes to process (default: 100)"
    )

    parser.add_argument(
        "--op",
        type=str,
        metavar="OP_NAME",
        help="Optimize only a specific operator"
    )
    
    args = parser.parse_args()
    io_parent_dir = args.io_dir
    optional_proj_name = args.project_name

    ssh_config = None

    # -----------------------
    # GPU SPEC COLLECTION
    # -----------------------
    if args.remote:
        config_path = args.remote

        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, "r") as f:
            remote_config = json.load(f)

        connections = remote_config.get("ssh_connections", [])
        active_idx = remote_config.get("active_ssh_index", -1)

        if 0 <= active_idx < len(connections):
            ssh_config = connections[active_idx]
            print(f"Remote mode: Using SSH connection to {ssh_config.get('host')}")
        else:
            print("Error: Invalid active_ssh_index in remote config.")
            sys.exit(1)

        print("Retrieving remote GPU specs...")
        gpu_specs = gpu.get_remote_gpu_specs(ssh_config)

    else:
        gpu_specs = gpu.get_gpu_specs()

    # -----------------------
    # PROJECT CREATION
    # -----------------------

    # Maintain compatibility if create_project expects sys.argv layout
    if optional_proj_name:
        sys.argv = [sys.argv[0], str(io_parent_dir), optional_proj_name]

    # -----------------------
    # NEW ROOT HANDLING
    # -----------------------

    if args.new_root:
        op_name = args.new_root
        proj_dir = get_project_dir(gpu_specs.gpu_name)
        op_dir_path = proj_dir / op_name
        
        if not op_dir_path.exists():
            print(f"Error: Operator '{op_name}' not found in project.")
            print(f"Expected path: {op_dir_path}")
            sys.exit(1)
        
        # Check if operator has at least node 0 (baseline)
        if not (op_dir_path / "nodes" / "0.json").exists():
            print(f"Error: Operator '{op_name}' has no baseline node. Run normal optimization first.")
            sys.exit(1)
        
        io_dir = io_parent_dir / op_name
        if not io_dir.exists():
            print(f"Error: IO directory not found for '{op_name}'")
            sys.exit(1)
        
        paths = {
            "proj_dir": op_dir_path,
            "io_dir": io_dir,
            "op_dir": Path("kernels/generated/individual_op_kernels") / op_name,
        }
        
        print(f"Creating new root for {op_name}...")
        new_root = create_new_root(gpu_specs, paths)
        
        if new_root:
            print(f"\nSuccess! Created new root: Node {new_root.id}")
            print(f"  Runtime: {new_root.value:.4f} ms")
        else:
            print("\nFailed to create new root.")
            sys.exit(1)
        
        sys.exit(0)

    # Normal operation: Create project (or resume if exists/provided)
    proj_dir = create_project(
        gpu_specs,
        io_parent_dir,
        optional_proj_name,
        ssh_config
    )

    # Build list of operators to process
    if args.op:
        # Single operator mode
        op_dir_path = proj_dir / args.op
        if not op_dir_path.exists():
            print(f"Error: Operator '{args.op}' not found in project.")
            print(f"Available operators: {[d.name for d in proj_dir.iterdir() if d.is_dir()]}")
            sys.exit(1)
        operators_to_process = [op_dir_path]
        print(f"Single operator mode: {args.op}")
    else:
        # All operators
        operators_to_process = [d for d in proj_dir.iterdir() if d.is_dir()]
        print(f"Processing all operators ({len(operators_to_process)} found)")

    # Process each operator
    for op_dir_path in operators_to_process:
        op_name = op_dir_path.name
        print(f"\n{'='*40}")
        print(f"Optimizing: {op_name}")
        print(f"{'='*40}")

        # Check if IO exists
        io_dir = io_parent_dir / op_name
        if not io_dir.exists():
            print(f"  Skipping {op_name}: IO directory not found at {io_dir}")
            continue

        # Paths for optimization
        paths = {
            "proj_dir": op_dir_path,
            "io_dir": io_dir,
            "op_dir": Path("kernels/generated/individual_op_kernels") / op_name,
        }

        mcts._NODE_CACHE.clear()

        if args.parallel:
            # === PARALLEL MODE ===
            run_parallel_optimization(
                gpu_specs=gpu_specs,
                paths=paths,
                n_workers=args.workers,
                max_iterations=args.max_iterations
            )
        else:
            # === SEQUENTIAL MODE (original) ===
            for i in range(args.max_iterations):
                # Select parent node then optimize off of it
                parent_node = mcts.choose_optimization(paths)
                new_node = optimize(gpu_specs, paths, parent_node)
                
                # Update tree with the new node (if optimization succeeded)
                if new_node:
                    mcts.update_tree(paths, new_node)
                
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{args.max_iterations} iterations")

if __name__ == "__main__":
    main()