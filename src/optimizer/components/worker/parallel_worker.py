"""
src/optimizer/components/worker/parallel_worker.py
Worker routine for parallel MCTS optimization using multiprocessing.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

from src.optimizer.core.types import KernelNode


def worker_routine(task_queue, result_queue, gpu_lock, node_counter, paths_template: dict, backend_type: str = "cuda", model: str = None):
    """Persistent worker process for parallel optimization.
    
    Pulls tasks from task_queue, generates/compiles/validates kernels,
    and pushes results to result_queue. Exits on None sentinel.
    """
    from src.optimizer.config.settings import settings
    from src.llm_tools import GenModel
    import src.optimizer.core.generator as generator
    import time
    
    # Instantiate backend
    if backend_type == "cuda":
        from src.optimizer.backends.cuda import CUDABackend
        backend = CUDABackend()
    elif backend_type == "triton":
        from src.optimizer.backends.triton import TritonBackend
        backend = TritonBackend()
    elif backend_type == "metal":
        from src.optimizer.backends.metal import MetalBackend
        backend = MetalBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    worker_pid = os.getpid()
    
    while True:
        try:
            task = task_queue.get(timeout=30)
            if task is None:
                break  # Sentinel received — exit cleanly
            parent_node, context = task
        except Exception:
            continue  # Timeout, check again
        
        node_id = parent_node.id
        tmp_dir = None
        
        try:
            # Build paths for this task
            paths = context.get("paths", paths_template).copy()
            paths["node_counter"] = node_counter
            paths["gpu_lock"] = gpu_lock
            
            # Each worker needs its own temp directory for compilation
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"worker_{worker_pid}_"))
            paths["tmp_dir"] = tmp_dir
            
            # io_dir logic (same as original, ensure it exists or derive it)
            if "io_dir" not in paths:
                proj_dir = Path(paths["proj_dir"])
                if (proj_dir / "io").exists():
                    paths["io_dir"] = proj_dir / "io"
                else:
                    paths["io_dir"] = proj_dir
            
            gpu_specs = context.get("gpu_specs")
            history = context.get("history", [])
            codes = context.get("codes", [])
            paths["iteration"] = parent_node.id
            
            # Get parent's kernel code
            parent_code_path = paths["proj_dir"] / "nodes" / f"{parent_node.id}.json"
            best_kernel_code = ""

            # Try loading from DB node object first (context has partial node info, but maybe not code path?)
            # Actually parent_node is a KernelNode object but code might be relative.
            if parent_node.code:
                code_p = Path(parent_node.code)
                if not code_p.is_absolute():
                     code_p = paths["proj_dir"] / parent_node.code
                if code_p.exists():
                    best_kernel_code = code_p.read_text()
            
            # Fallback for root or if code path invalid
            if not best_kernel_code:
                 # Try typical paths
                 p1 = paths["proj_dir"] / "kernels" / f"kernel_{parent_node.id}{backend.kernel_extension}"
                 if p1.exists():
                     best_kernel_code = p1.read_text()
            
            if not best_kernel_code:
                result_queue.put((node_id, None, "no_parent_code"))
                continue
            
            # === GENERATION PHASE (CPU/Network bound) ===
            
            # Reserve ID early for prompt logging
            with node_counter.get_lock():
                kernel_id = node_counter.value
                node_counter.value += 1

            # 1. Generate Prompt
            prompt = backend.generate_optimization_prompt(
                gpu_specs, best_kernel_code, history, codes
            )
            
            # Save prompt to file
            prompt_path = paths["proj_dir"] / "kernels" / f"prompt_{kernel_id}.md"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(prompt_path, "w") as f:
                    f.write(f"# System Prompt\n\n{backend.get_sys_prompt()}\n\n---\n\n# User Message\n\n{prompt}")
                print(f"[WORKER {worker_pid}] Saved prompt to: {prompt_path}")
            except Exception as e:
                print(f"[WORKER {worker_pid}] Failed to save prompt: {e}")

            # 2. Call LLM
            sys_prompt = backend.get_sys_prompt()
            llm = GenModel(sys_prompt)
            
            result_queue.put((node_id, {"step": "Generating", "attempt": 1}, "status_update"))
            
            # Retry loop variables
            current_prompt = prompt
            is_valid = False
            validation_error = ""
            metrics = {} # To store Code, feedback etc
            
            retry_limit = settings.retry_limit if hasattr(settings, 'retry_limit') else 3
            
            for attempt in range(retry_limit + 1):
                # 3. Chat & Extract Code
                # For first attempt, current_prompt is the full prompt
                # For retries, current_prompt is the error message (GenModel keeps history)
                
                try:
                    response = llm.chat(current_prompt, model or settings.llm_model_name)
                except Exception as e:
                    result_queue.put((node_id, None, f"llm_error: {e}"))
                    break

                feedback, code = generator.extract_feedback_and_code(response)

                if code is None:
                    # If we can't extract code, treat as validation error and retry if possible
                    validation_error = "Failed to extract code from LLM response"
                    if attempt < retry_limit:
                        current_prompt = f"Error: {validation_error}. Please provide the full kernel code in a code block."
                        continue
                    else:
                        break

                # Write to tmp
                (paths["tmp_dir"] / f"kernel{backend.kernel_extension}").write_text(code)

                # 4. Validate
                result_queue.put((node_id, {"step": "Validating", "attempt": attempt + 1}, "status_update"))
                is_valid, validation_error = backend.validate_kernel(code, paths)
                
                if is_valid:
                    break
                
                # If invalid, prepare for next attempt
                if attempt < retry_limit:
                    print(f"[WORKER {worker_pid}] Attempt {attempt+1} failed: {validation_error[:100]}... Retrying.")
                    current_prompt = f"Compilation/Validation failed with error:\n{validation_error}\n\nPlease fix the code."
            
            # check final status
            if not is_valid:
                result_queue.put((node_id, None, f"validation_failed_after_retries: {validation_error[:200]}"))
                continue

            # 5. Profile (Exclusive GPU access)
            runtime_ms = float('inf')
            
            result_queue.put((node_id, {"step": "Profiling", "attempt": attempt + 1 if 'attempt' in locals() else 1}, "status_update"))
            
            # Use lock if provided (for strict serialization of GPU kernels)
            if gpu_lock:
                 with gpu_lock:
                      stats = backend.profile_kernel(paths)
                      runtime_ms = stats.get('mean_time_ms', float('inf'))
            else:
                 stats = backend.profile_kernel(paths)
                 runtime_ms = stats.get('mean_time_ms', float('inf'))

            if runtime_ms == float('inf'):
                 result_queue.put((node_id, None, f"profiling_failed"))
                 continue

            # 6. Save Kernel
            # We already have kernel_id reserved
            
            attempts_dir = paths["proj_dir"] / "kernels"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            kernel_filename = f"kernel_{kernel_id}{backend.kernel_extension}"
            kernel_path = attempts_dir / kernel_filename
            
            import shutil
            shutil.copy(paths["tmp_dir"] / f"kernel{backend.kernel_extension}", kernel_path)
            
            # Return success
            result_queue.put((
                node_id,
                {
                    "runtime_ms": runtime_ms,
                    "kernel_id": kernel_id,
                    "code_path": str(Path(paths["proj_dir"].name) / "kernels" / kernel_filename),
                    "feedback": feedback
                },
                "success"
            ))
            
            # Cleanup temp directory
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir = None
            
        except Exception as e:
            import traceback
            print(f"[WORKER {worker_pid}] ERROR: {e}")
            traceback.print_exc()
            result_queue.put((node_id, None, f"worker_error: {e}"))
        finally:
            # Always cleanup temp directory
            if tmp_dir and tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
