import sys
import json
import random
import shutil
import string
import argparse
import tempfile
import queue
import os
from datetime import datetime, timezone
import time
from pathlib import Path

import torch

from src.config import ensure_llm_config, load_project_config

import src.optimizer.core.generator as generator
import src.optimizer.core.mcts as mcts
from src.optimizer.core.types import KernelNode, GPUSpecs
from src.optimizer.config.settings import settings
from src.optimizer.core.backend import Backend
from src.optimizer.backends.cuda import CUDABackend
from src.optimizer.backends.metal import MetalBackend
from src.optimizer.backends.triton import TritonBackend


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def update_queue_state(proj_base_dir: Path, updates: dict):
    from src.optimizer.benchmarking.locks import file_lock
    queue_path = proj_base_dir / "queue.json"
    lock_path = queue_path.with_suffix(".json.lock")
    with file_lock(lock_path):
        if queue_path.exists():
            try:
                state = json.loads(queue_path.read_text())
            except:
                state = {"active_tasks": {}, "benchmark_slot": {"now": None, "pending": []}, "pending_operators": [], "current_operator": "", "completed_tasks": []}
        else:
            state = {"active_tasks": {}, "benchmark_slot": {"now": None, "pending": []}, "pending_operators": [], "current_operator": "", "completed_tasks": []}

        if "completed_tasks" not in state:
            state["completed_tasks"] = []

        if "active_tasks" in updates:
            for k, v in updates["active_tasks"].items():
                if k not in state["active_tasks"]:
                    state["active_tasks"][k] = {}
                state["active_tasks"][k].update(v)
        if "remove_tasks" in updates:
            for k in updates["remove_tasks"]:
                removed = state["active_tasks"].pop(str(k), None)
                if removed:
                    removed["id"] = str(k)
                    removed["completed_at"] = int(time.time() * 1000)
                    state["completed_tasks"].append(removed)
        if "benchmark_slot" in updates:
            state["benchmark_slot"].update(updates["benchmark_slot"])
        if "pending_operators" in updates:
            state["pending_operators"] = updates["pending_operators"]
        if "current_operator" in updates:
            state["current_operator"] = updates["current_operator"]

        # Auto-archive Done/Failed tasks from active_tasks
        # Note: "Iter Complete" and "Iter Failed" are NOT archived — they are
        # intermediate states used during multi-iteration optimization loops.
        done_keys = []
        for k, v in state["active_tasks"].items():
            step = v.get("current_step", "")
            if step in ("Done", "Failed"):
                done_keys.append(k)
        for k in done_keys:
            archived = state["active_tasks"].pop(k)
            archived["id"] = k
            archived["completed_at"] = int(time.time() * 1000)
            state["completed_tasks"].append(archived)
            # Remove completed op from pending_operators
            completed_op = archived.get("op_name", "")
            if completed_op and "pending_operators" in state:
                state["pending_operators"] = [
                    p for p in state["pending_operators"] if p != completed_op
                ]
        # Cap completed list at 200 entries
        if len(state["completed_tasks"]) > 200:
            state["completed_tasks"] = state["completed_tasks"][-200:]

        queue_path.write_text(json.dumps(state, indent=2))



def _projects_base_dir() -> Path:
    base = _repo_root() / "kernels" / "projects"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _generated_kernels_root(optional_proj_name: str | None = None) -> Path:
    if optional_proj_name:
        per_project = (
            _projects_base_dir()
            / optional_proj_name
            / "kernels"
            / "generated"
            / "individual_op_kernels"
        )
        if per_project.exists():
            return per_project
    return _repo_root() / "kernels" / "generated" / "individual_op_kernels"


def _compact_reason(reason: str, limit: int = 320) -> str:
    text = str(reason or "").replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return ""
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def get_project_dir(gpu_name: str, optional_name: str = None, backend_name: str = "cuda"):
    letters = string.ascii_letters + string.digits
    proj_name = ''.join(random.choices(letters, k=10))
    clean_gpu_name = gpu_name.replace(" ", "_").replace(":", "").replace("-", "_")
    if optional_name:
        full_name = f"{optional_name}/trees"
    else:
        full_name = f"{clean_gpu_name}_{proj_name}-{backend_name}/trees"

    print(f"Beginning optimizing on project {full_name}...")

    proj_dir = _projects_base_dir() / full_name
    try:
        proj_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(
            f"Error: Project {full_name} creation failed: {e}")

    return proj_dir


def save_iteration(backend: Backend, paths: dict, parent_info: KernelNode, improvement_description: str, best_kernel_code: str, ssh_config: dict = None):
    """Profiles iteration and records performance results
    """
    next_id = mcts.get_next_node_id(paths)

    # Export attempt's code — use backend-appropriate extension
    ext = backend.kernel_extension
    current_kernel_code = (paths["tmp_dir"] / f"kernel{ext}").read_text()
    with open(paths["proj_dir"] / "kernels" / f"kernel_{next_id}{ext}", "w") as f:
        f.write(current_kernel_code)

    # Log the attempt with results
    print("\tBeginning Profiler...")
    current_stats = backend.profile_kernel(paths, ssh_config=ssh_config)
    print("\tFinished Profiler.")
    log_entry = {
        "iteration": next_id,
        "attempted": improvement_description,
        "results": current_stats,
        "speedup_vs_parent": parent_info.value / current_stats['mean_time_ms'] if current_stats['mean_time_ms'] > 0 else 1.0,
    }

    # Save node to DB
    node_val = {
        "id": next_id,
        "value": current_stats['mean_time_ms'],
        "speedup_vs_parent": log_entry['speedup_vs_parent'],
        "improvement_description": improvement_description,
        "parent": parent_info.id,
        "code": str(Path(paths["proj_dir"].name) / "kernels" / f"kernel_{next_id}{ext}"),
        "visits": 1
    }
    # Validate and dump with Pydantic
    node_obj = KernelNode.model_validate(node_val)
    mcts.save_node(paths, node_obj)

    next_id += 1
    return log_entry, node_obj


def optimize(
    backend: Backend,
    gpu_specs: GPUSpecs,
    paths: dict[str, Path],
    parent_node: KernelNode,
    ssh_config: dict = None,
    model: str = None,
    _proj_base_dir: Path = None,
    _task_key: str = None,
) -> tuple[KernelNode | None, str]:
    """Optimizes target kernel
    """

    # Create kernels directory
    (paths["proj_dir"] / "kernels").mkdir(parents=True, exist_ok=True)

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
            return None, f"Kernel code file not found: {kernel_code_path}"

        # Kernel Generation
        print(f"\tBeginning generation (history: {len(improvement_log)} entries)...")
        improvement_description, is_valid, failure_reason = generator.generate(
            backend, kernel_code, gpu_specs, improvement_log, paths, model=model, ancestor_codes=ancestor_codes, ssh_config=ssh_config)
        print("\tFinished generation.")
        print(f"\t\t- Status: {is_valid}")

        # If valid, profile and save normally
        if is_valid:
            if _proj_base_dir and _task_key:
                update_queue_state(_proj_base_dir, {"active_tasks": {_task_key: {"current_step": "Profiling"}}})
            log_entry, new_node = save_iteration(
                backend, paths, parent_node, improvement_description, str(paths["proj_dir"] / "kernels" / f"kernel_{parent_node.id}{backend.kernel_extension}"), ssh_config=ssh_config)
            return new_node, ""

        # Failed/invalid kernel — still save to tree so MCTS can track it.
        # value=None is treated as inf by choose_optimization(), keeping UCT
        # from revisiting this branch while still recording the attempt.
        next_id = mcts.get_next_node_id(paths)
        (paths["proj_dir"] / "kernels").mkdir(parents=True, exist_ok=True)
        kernel_tmp = paths["tmp_dir"] / f"kernel{backend.kernel_extension}"
        if kernel_tmp.exists():
            dest = paths["proj_dir"] / "kernels" / f"kernel_{next_id}{backend.kernel_extension}"
            shutil.copy(str(kernel_tmp), str(dest))
            code_path = str(Path(paths["proj_dir"].name) / "kernels" / f"kernel_{next_id}{backend.kernel_extension}")
        else:
            code_path = parent_node.code
        node_val = {
            "id": next_id,
            "value": None,
            "speedup_vs_parent": 0.0,
            "improvement_description": f"[FAILED] {failure_reason or 'validation failed'}",
            "parent": parent_node.id,
            "code": code_path,
            "visits": 1,
        }
        node_obj = KernelNode.model_validate(node_val)
        mcts.save_node(paths, node_obj)
        return node_obj, ""


def create_new_root(backend: Backend, gpu_specs: GPUSpecs, paths: dict[str, Path], model: str = None) -> KernelNode:
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
    import src.generator.prompts.prompts as gen_prompts

    # Select backend-specific prompts
    if backend.kernel_extension == ".py":
        import src.optimizer.backends.triton.prompts as opt_prompts
    else:
        import src.optimizer.backends.cuda.prompts as opt_prompts
    
    # Get next available node ID
    # Get next available node ID
    next_id = mcts.get_next_node_id(paths)
    
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
        (paths["proj_dir"] / "kernels").mkdir(parents=True, exist_ok=True)
        prompt_path = paths["proj_dir"] / "kernels" / f"new_root_prompt_{next_id}.md"
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
            llm.chat(prompt, model or settings.llm_model_name)
        )
        
        if code is None:
            print("\t\tError: Failed to extract code from LLM response")
            return None
        
        # Write to tmp for validation
        (paths["tmp_dir"] / f"kernel{backend.kernel_extension}").write_text(code)
        
        # Validate the kernel with retries
        is_valid, error = backend.validate_kernel(code, paths)
        
        # Retry loop on validation failure
        attempt = 0
        while not is_valid and attempt < settings.retry_limit:
            attempt += 1
            print(f"\t\tValidation failed, retry {attempt}/{settings.retry_limit}...")
            
            # Save failed attempt to garbage dump
            dump_dir = paths["proj_dir"] / "garbage_dump"
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / f"new_root_{next_id}_attempt{attempt}{backend.kernel_extension}").write_text(code)
            
            # Send error back to LLM for correction
            feedback, code = generator.extract_feedback_and_code(
                llm.chat(error, model or settings.llm_model_name)
            )
            
            if code is None:
                print(f"\t\tRetry {attempt}: Failed to extract code")
                continue
            
            # Validate the new attempt
            (paths["tmp_dir"] / f"kernel{backend.kernel_extension}").write_text(code)
            is_valid, error = backend.validate_kernel(code, paths)
        
        if not is_valid:
            print(f"\t\tValidation failed after {settings.retry_limit} retries: {error}")
            dump_dir = paths["proj_dir"] / "garbage_dump"
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / f"new_root_{next_id}_final_failed{backend.kernel_extension}").write_text(code)
            return None
        
        # Profile the kernel
        print("\t\tProfiling new root kernel...")
        current_stats = backend.profile_kernel(paths)
        
        # Create root node (parent = -1)
        node_data = {
            "id": next_id,
            "parent": -1,  # Root marker
            "value": current_stats['mean_time_ms'],
            "speedup_vs_parent": 1.0,
            "improvement_description": "Initial",
            "code": f"{paths['proj_dir'].name}/kernels/kernel_{next_id}{backend.kernel_extension}",
            "visits": 1
        }
        node = KernelNode.model_validate(node_data)
        
        # Save node
        # Save node
        mcts.save_node(paths, node)
        
        # Save kernel code
        (paths["proj_dir"] / "kernels" / f"kernel_{next_id}{backend.kernel_extension}").write_text(code)
        
        print(f"\t\tCreated new root: Node {next_id} ({current_stats['mean_time_ms']:.4f} ms)")
        return node


def create_project(backend: Backend, gpu_specs: GPUSpecs, io_parent_dir: Path, optional_proj_name: str = None, ssh_config: dict = None):
    """Creates a new optimization project for each individual operator kernel.
    """
    # Derive backend name from extension (e.g. .cu -> cuda, .py -> triton)
    ext_to_name = {".cu": "cuda", ".py": "triton", ".metal": "metal"}
    bk_name = ext_to_name.get(backend.kernel_extension, "cuda")
    proj_dir = get_project_dir(gpu_specs.gpu_name, optional_proj_name, backend_name=bk_name)

    # Directory containing initial wave of correct, but unoptimized kernels
    generated_root = _generated_kernels_root(optional_proj_name)
    op_dirs = list(generated_root.glob("*"))

    # Profile each kernel in op_dirs and save results as a *.json node in their respective directories
    for op_dir in op_dirs:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            io_dir = io_parent_dir / op_dir.name

            if not io_dir.exists():
                print(f"{op_dir.name} has no i/o")
                continue

            # Check if the kernel has been previously generated
            # Map backend extension to success marker name (e.g. .cu -> success.cuda, .py -> success.triton)
            ext_to_device = {".cu": "cuda", ".py": "triton", ".metal": "metal"}
            device_name = ext_to_device.get(backend.kernel_extension, "cuda")
            success_marker = op_dir / f"success.{device_name}"
            if not success_marker.exists():
                print(
                    f"Skipping {op_dir.name}: No 'success.{device_name}' marker found (generation failed).")
                continue

            # Prepare project op directory
            proj_op_dir = proj_dir / op_dir.name
            proj_op_dir.mkdir(parents=True, exist_ok=True)
            (proj_op_dir / "kernels").mkdir(parents=True, exist_ok=True)

            paths = {
                "tmp_dir": tmp_path,
                "io_dir": io_dir,
                "proj_dir": proj_op_dir
            }

            # Skip if already initialized
            if mcts.node_exists(paths, 0):
                print(
                    f"{op_dir.name} already initialized, skipping baseline profile.")
                continue

            # Copy kernel to tmp_dir for profiling — prefer backend's own extension
            if (op_dir / f"kernel{backend.kernel_extension}").exists():
                src_ext = backend.kernel_extension
            elif (op_dir / "kernel.cu").exists():
                src_ext = ".cu"
            else:
                src_ext = ".py"
            dst_ext = backend.kernel_extension
            (tmp_path / f"kernel{dst_ext}").write_text((op_dir / f"kernel{src_ext}").read_text())

            # Profile kernel
            current_stats = backend.profile_kernel(paths, baseline=True, ssh_config=ssh_config)

            # Log kernel
            node_data = {
                "id": 0,
                "value": current_stats['mean_time_ms'],
                "speedup_vs_parent": 1.0,
                "improvement_description": "Initial",
                "parent": -1,
                "code": str(Path(proj_op_dir.name) / "kernels" / f"kernel_0{backend.kernel_extension}"),
                "visits": 1
            }
            node = KernelNode.model_validate(node_data)
            
            # Save root node
            mcts.save_node(paths, node)
                
            # Copy kernel to kernels manually as well
            (paths["proj_dir"] / "kernels").mkdir(parents=True, exist_ok=True)
            with open(paths["proj_dir"] / "kernels" / f"kernel_0{backend.kernel_extension}", "w") as f:
                f.write((op_dir / f"kernel{src_ext}").read_text())

    return proj_dir

def run_parallel_optimization(backend: Backend, gpu_specs: GPUSpecs, paths: dict, n_workers: int = 4, max_iterations: int = 100, ssh_config: dict = None, model: str = None):
    """Run parallel MCTS optimization using multiprocessing.
    
    Dispatches nodes to workers as they become available, stops after max_iterations.
    
    Args:
        backend: The backend instance (CUDABackend, TritonBackend, etc.)
        gpu_specs: GPU specifications
        paths: Dictionary containing project paths
        n_workers: Number of worker processes
        max_iterations: Total number of nodes to process before stopping
        ssh_config: Optional SSH configuration for remote execution
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
    initial_count = mcts.get_next_node_id(paths)
    node_counter = mp.Value('i', initial_count)  # Atomic integer counter
    
    in_flight_ids = set()
    nodes_dispatched = 0
    nodes_completed = 0
    nodes_failed = 0
    
    # Start persistent workers
    print(f"[INIT] Starting {n_workers} workers...")
    workers = []
    
    # helper to get backend class name or type
    backend_type = "cuda"
    if isinstance(backend, TritonBackend):
        backend_type = "triton"
    elif isinstance(backend, MetalBackend):
        backend_type = "metal"

    for i in range(n_workers):
        p = Process(target=worker_routine, args=(task_queue, result_queue, gpu_lock, node_counter, paths, backend_type, model))
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
            
            proj_base_dir = paths["proj_dir"].parent.parent
            queue_entry = {
                "tag": "[OPT]" if node.id > 0 else "[GEN]",
                "op_name": paths["io_dir"].name,
                "current_step": "Monitoring" if node.id == 0 else "Generating",
                "attempt_current": 1,
                "attempt_max": settings.retry_limit if hasattr(settings, 'retry_limit') else 3,
                "parent_ref": f"kernel_{node.id}" if node.id > 0 else "",
                "status": "In Progress"
            }
            update_queue_state(proj_base_dir, {"active_tasks": {str(node.id): queue_entry}})
            
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
            
            proj_base_dir = paths["proj_dir"].parent.parent
            if status == "success" and result_data is not None:
                nodes_completed += 1
                runtime_ms = result_data["runtime_ms"]
                kernel_id = result_data["kernel_id"]
                
                parent_node = mcts._NODE_CACHE.get(node_id)
                speedup = 1.0
                if parent_node:
                    speedup = parent_node.value / runtime_ms if runtime_ms > 0 else 1.0
                    new_node = KernelNode(
                        id=kernel_id,
                        parent_id=node_id,
                        children_ids=[],
                        visits=1,
                        value=runtime_ms,
                        best_subtree_value=runtime_ms,
                        speedup_vs_parent=speedup,
                        improvement_description=result_data.get("feedback", "Parallel optimization"),
                        code=result_data["code_path"]
                    )
                    mcts.update_tree(paths, new_node)
                    print(f"[SUCCESS] Node {node_id} -> {kernel_id} | {runtime_ms:.4f}ms | {speedup:.2f}x | Done: {nodes_completed}")
                
                update_queue_state(proj_base_dir, {
                    "active_tasks": {str(node_id): {"current_step": "Done", "result": f"{speedup:.2f}x", "status": "Done"}},
                    "benchmark_slot": {"now": None, "pending": []}
                })
            elif status == "status_update":
                step_name = result_data.get("step")
                update_dict = {"current_step": step_name, "attempt_current": result_data.get("attempt", 1)}
                bm_slot = {}
                if step_name == "Benchmarking":
                    bm_slot = {"benchmark_slot": {"now": f"{paths['io_dir'].name} · kernel_{node_id}", "pending": []}}
                update_queue_state(proj_base_dir, {"active_tasks": {str(node_id): update_dict}, **bm_slot})
            else:
                nodes_failed += 1
                error_label = status[:50] if isinstance(status, str) else "Failed"
                update_queue_state(proj_base_dir, {
                    "active_tasks": {str(node_id): {"current_step": "Failed", "result": error_label, "status": "Failed"}},
                    "benchmark_slot": {"now": None, "pending": []}
                })
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
    
    parser = argparse.ArgumentParser(
        description="CUDA Kernel Optimizer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal optimization run
  python3 -m src.optimizer.pipeline kernels/projects/<project>/io/individual_ops project_name
  
  # Create a new independent root for a specific operator
  python3 -m src.optimizer.pipeline kernels/projects/<project>/io/individual_ops project_name --new-root torch_nn_functional_embedding

  # Run using remote SSH configuration
  python3 -m src.optimizer.pipeline kernels/projects/<project>/io/individual_ops project_name --remote config.json
  
  # Use a different backend
  python3 -m src.optimizer.pipeline kernels/projects/<project>/io/individual_ops project_name --backend metal
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
        "--backend",
        type=str,
        choices=["cuda", "metal", "triton"],
        default="cuda",
        help="Backend to use for optimization (default: cuda)"
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

    # --- Load Project Config (LLM Settings) ---
    # Try to find config.json relative to IO dir (e.g. projects/Test-project/io -> projects/Test-project/config.json)
    current_path = io_parent_dir.absolute()
    project_config_path = None
    for _ in range(3): # Check up to 3 levels up
        candidate = current_path / "config.json"
        if candidate.exists():
            project_config_path = candidate
            break
        current_path = current_path.parent
    
    if project_config_path:
        print(f"Loading project config from: {project_config_path}")
        os.environ["KFORGE_PROJECT_CONFIG_PATH"] = str(project_config_path)
    
    # Initialize LLM config from environment/file
    ensure_llm_config()

    # Export active model name to environment so settings.py can pick it up
    # ensure_llm_config sets LLM_PROVIDER and specific model vars (e.g. ANTHROPIC_MODEL)
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    model_name = ""
    if provider == "openai":
        model_name = os.environ.get("OPENAI_MODEL", "")
    elif provider == "anthropic":
        model_name = os.environ.get("ANTHROPIC_MODEL", "")
    elif provider == "google" or provider == "gemini":
        model_name = os.environ.get("GEMINI_MODEL", "")
    
    if model_name:
        print(f"Using LLM Model: {model_name} (Provider: {provider})")

    ssh_config = None
    
    # Instantiate Backend based on arguments
    if args.backend == "cuda":
        backend = CUDABackend()
    elif args.backend == "metal":
        backend = MetalBackend()
    elif args.backend == "triton":
        backend = TritonBackend()
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

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
        gpu_specs = backend.get_device_specs(ssh_config=ssh_config)
    else:
        # Collect GPU specs locally
        gpu_specs = backend.get_device_specs()

    # Create project (or resume if exists/provided)
    proj_dir = create_project(backend, gpu_specs, io_parent_dir, optional_proj_name, ssh_config)

    # -----------------------
    # NEW ROOT HANDLING
    # -----------------------

    if args.new_root:
        op_name = args.new_root
        op_dir_path = proj_dir / op_name
        
        if not op_dir_path.exists():
            print(f"Error: Operator '{op_name}' not found in project.")
            print(f"Expected path: {op_dir_path}")
            sys.exit(1)
        
        # Check if operator has at least node 0 (baseline)
        paths_check = {"proj_dir": op_dir_path}
        if not mcts.node_exists(paths_check, 0):
            print(f"Error: Operator '{op_name}' has no baseline node. Run normal optimization first.")
            sys.exit(1)
        
        io_dir = io_parent_dir / op_name
        if not io_dir.exists():
            print(f"Error: IO directory not found for '{op_name}'")
            sys.exit(1)
        
        paths = {
            "proj_dir": op_dir_path,
            "io_dir": io_dir,
            "op_dir": _generated_kernels_root(optional_proj_name) / op_name,
        }
        
        print(f"Creating new root for {op_name}...")
        new_root = create_new_root(backend, gpu_specs, paths, model=model_name)
        
        if new_root:
            print(f"\nSuccess! Created new root: Node {new_root.id}")
            print(f"  Runtime: {new_root.value:.4f} ms")
        else:
            print("\nFailed to create new root.")
            sys.exit(1)
        
        sys.exit(0)



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

    # Initialize queue status — preserve existing active_tasks so we don't
    # wipe entries written by _write_initial_queue_state before pipeline.py starts.
    proj_base_dir = proj_dir.parent
    update_queue_state(proj_base_dir, {
        "pending_operators": [d.name for d in operators_to_process],
        "benchmark_slot": {"now": None, "pending": []},
        "current_operator": ""
    })

    # Process each operator
    for op_dir_path in operators_to_process:
        op_name = op_dir_path.name
        print(f"\n{'='*40}")
        print(f"Optimizing: {op_name}")
        print(f"{'='*40}")

        # Check if IO exists
        io_dir = io_parent_dir / op_name
        if not io_dir.exists():
            reason = f"IO directory not found at {io_dir}"
            print(f"  Skipping {op_name}: {reason}")
            print(
                f"[optimize-result] op={op_name} status=hard_error "
                f"new_nodes=0 last_reason={json.dumps(_compact_reason(reason))}"
            )
            continue
            
        update_queue_state(proj_base_dir, {
            "current_operator": op_name
        })

        # Paths for optimization
        paths = {
            "proj_dir": op_dir_path,
            "io_dir": io_dir,
            "op_dir": _generated_kernels_root(optional_proj_name) / op_name,
        }

        mcts._NODE_CACHE.clear()
        if args.parallel:
            # === PARALLEL MODE ===
            run_parallel_optimization(
                backend=backend,
                gpu_specs=gpu_specs,
                paths=paths,
                ssh_config=ssh_config,
                n_workers=args.workers,
                max_iterations=args.max_iterations,
                model=model_name
            )
            print(
                f"[optimize-result] op={op_name} status=unknown "
                "new_nodes=0 last_reason="
                f"{json.dumps('parallel_mode_result_not_tracked')}"
            )
        else:
            # === SEQUENTIAL MODE (original) ===
            op_new_nodes = 0
            op_last_reason = ""
            task_key = "seq_opt"
            retry_limit = getattr(settings, 'retry_limit', 3)
            total_iters = args.max_iterations
            for i in range(total_iters):
                is_last = (i == total_iters - 1)
                # Select parent node then optimize off of it
                parent_node = mcts.choose_optimization(paths)
                update_queue_state(proj_base_dir, {"active_tasks": {task_key: {
                    "tag": "[OPT]",
                    "op_name": op_name,
                    "current_step": "Generating",
                    "attempt_current": 1,
                    "attempt_max": retry_limit,
                    "parent_ref": f"kernel_{parent_node.id}",
                    "status": "In Progress",
                    "iter_current": i + 1,
                    "iter_max": total_iters,
                }}})
                new_node, failure_reason = optimize(
                    backend, gpu_specs, paths, parent_node, ssh_config, model=model_name,
                    _proj_base_dir=proj_base_dir, _task_key=task_key,
                )

                # Update tree with the new node (always saved now)
                if new_node:
                    mcts.update_tree(paths, new_node)
                    # Only count as an improvement if the kernel was valid/profiled
                    if new_node.value is not None:
                        op_new_nodes += 1
                        op_last_reason = ""
                        speedup = parent_node.value / new_node.value if new_node.value > 0 else 1.0
                        # Use "Done" only on last iteration to trigger archiving;
                        # otherwise use "Iter Complete" which is NOT auto-archived.
                        step_name = "Done" if is_last else "Iter Complete"
                        update_queue_state(proj_base_dir, {"active_tasks": {task_key: {
                            "current_step": step_name,
                            "result": f"{speedup:.2f}x",
                            "status": step_name,
                        }}})
                    else:
                        step_name = "Failed" if is_last else "Iter Failed"
                        display_step = step_name if is_last else "Validating"
                        update_queue_state(proj_base_dir, {"active_tasks": {task_key: {
                            "current_step": display_step,
                            "result": "validation failed",
                            "status": step_name,
                        }}})
                else:
                    step_name = "Failed" if is_last else "Iter Failed"
                    update_queue_state(proj_base_dir, {"active_tasks": {task_key: {
                        "current_step": step_name,
                        "result": "no kernel produced",
                        "status": step_name,
                    }}})

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{args.max_iterations} iterations")
            if op_new_nodes > 0:
                print(
                    f"[optimize-result] op={op_name} status=improved "
                    f"new_nodes={op_new_nodes} last_reason={json.dumps('')}"
                )
            else:
                if not op_last_reason:
                    op_last_reason = "No valid candidates were produced."
                print(
                    f"[optimize-result] op={op_name} status=no_improvement "
                    f"new_nodes=0 last_reason={json.dumps(_compact_reason(op_last_reason))}"
                )
    
if __name__ == "__main__":
    main()
