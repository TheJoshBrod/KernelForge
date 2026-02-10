"""
src/optimizer/components/worker/verifier.py
Validates a generated CUDA kernel by compiling it as a PyTorch C++ extension
and comparing its tensor output against a ground-truth tensor.
"""

import os
import re
import multiprocessing
import queue
from pathlib import Path

import torch
from byllm.lib import by  # type: ignore
from byllm.lib import Model  # type: ignore
from torch.utils.cpp_extension import load_inline
from src.optimizer.config.settings import settings

llm = Model(model_name=settings.llm_model_name)


@by(llm)
def summarize_issue_with_traceback(
    traceback_error: str,
    cu_code: str,
    input_and_output: dict
) -> str:
    """
    Analyze CUDA kernel compilation or runtime errors and provide actionable fix suggestions.

    Important:
    - The caller-side argument formatting is already correct and MUST NOT be modified.
    - The issue will always be internal to the CUDA kernel code (argument order, typing,
        indexing, launch signature, or parameter handling).

    The input_and_output dict contains:
    - args: List of positional arguments (normalized from kwargs if applicable)
    - signature: Dict with 'params' (parameter names in order) and 'defaults'
    - output or correct-output: Expected output tensor

    The CUDA kernel's launch() function MUST accept arguments in exactly this order:
    {', '.join(input_and_output.get('signature', {}).get('params', ['arg0', 'arg1', '...']))}

    Provide specific recommendations for:
    1. Correct argument ordering and types in the kernel launch() signature
    2. Correct parameter use inside the CUDA code
    3. Indexing, pointer arithmetic, and shape-related issues

    Do NOT suggest modifying Python call sites, pybind11 bindings, or argument conversion logic.
    Only CUDA-side fixes are relevant.
    """


def normalize_args_kwargs(args: list, kwargs: dict, signature_info: dict) -> tuple[list, dict]:
    """
    Normalize args and kwargs into a complete positional argument list.

    Args:
        args: Positional arguments as captured
        kwargs: Keyword arguments as captured
        signature_info: Dict with 'params' (list of param names) and 'defaults' (dict of defaults)

    Returns:
        tuple: (normalized_args, remaining_kwargs)
            - normalized_args: Complete list of positional args with defaults filled
            - remaining_kwargs: Any kwargs that couldn't be mapped (for error reporting)
    """
    params = signature_info.get("params", [])
    defaults = signature_info.get("defaults", {})

    if not params:
        # No signature info available, return as-is
        return args, kwargs

    # Start with provided positional args
    normalized = list(args)
    remaining_kwargs = dict(kwargs)

    # Fill in any kwargs that should be positional
    for i in range(len(normalized), len(params)):
        param_name = params[i]

        if param_name in remaining_kwargs:
            # Use the provided kwarg
            normalized.append(remaining_kwargs.pop(param_name))
        elif param_name in defaults:
            # Use the default value
            normalized.append(defaults[param_name])
        else:
            # No value available - this might cause issues
            # Log a warning but continue
            print(
                f"Warning: No value for parameter '{param_name}' at position {i}")
            break

    return normalized, remaining_kwargs


def move_to_cuda(item):
    """Recursively move tensors to CUDA."""
    if torch.is_tensor(item):
        return item.cuda()
    elif isinstance(item, (list, tuple)):
        return type(item)(move_to_cuda(x) for x in item)
    elif isinstance(item, dict):
        return {k: move_to_cuda(v) for k, v in item.items()}
    return item


# --- Persistent Worker Globals ---
_WORKER_PROCESS = None
_WORKER_Q_IN = None
_WORKER_Q_OUT = None


def _validate_worker_loop(q_in, q_out):
    """
    Persistent worker loop.
    Waits for (generated_cu_code, tmpdir, io_dir) tuples.
    Sends back (is_valid, log_message).
    """
    # Set environment variables ONCE
    import sys
    python_bin_dir = os.path.dirname(sys.executable)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
    os.environ["PATH"] = f"{python_bin_dir}:/usr/local/cuda-12.1/bin:{os.environ['PATH']}"

    # Pre-import torch to warm up (already imported at top level, but ensure context is ready)
    if torch.cuda.is_available():
        torch.cuda.init()

    while True:
        try:
            job = q_in.get()
            if job is None:
                break  # Sentinel to exit

            generated_cu_code, tmpdir, io_dir = job

            # Run the validation logic (refactored from original _validate_worker)
            try:
                # Compile
                match = re.search(
                    r"(torch::Tensor\s+launch\s*\([^)]*\))", generated_cu_code)
                if not match:
                    raise ValueError(
                        "Could not find 'launch' function signature.")

                cpp_source = match.group(1) + ";"

                # Check if module already loaded (optimization?)
                # load_inline uses a cache based on sources, but since we change tmpdir every time,
                # we force a recompile/realloc. Ideally we should reuse tmpdir if code is same,
                # but 'tmpdir' is provided by caller.

                module = load_inline(
                    name=f"generated_module_{os.path.basename(tmpdir)}",
                    cpp_sources=cpp_source,
                    cuda_sources=generated_cu_code,
                    functions=['launch'],
                    build_directory=tmpdir,
                    verbose=False,  # Reduce spam
                    with_cuda=True
                )

                # Execution Test
                entry_files = sorted(Path(io_dir).glob("entry_*.pt"))
                if not entry_files:
                    q_out.put(
                        (False, "[Error] No entry files found in io_dir"))
                    continue

                all_valid = True
                error_logs = []

                for entry_file in entry_files:
                    try:
                        entry = torch.load(entry_file)
                        args = entry.get("args", [])
                        kwargs = entry.get("kwargs", {})
                        signature_info = entry.get(
                            "signature", {"params": [], "defaults": {}})

                        normalized_args, remaining_kwargs = normalize_args_kwargs(
                            args, kwargs, signature_info)
                        cuda_args = [move_to_cuda(item)
                                     for item in normalized_args]

                        output_generated = module.launch(*cuda_args)
                        torch.cuda.synchronize()

                        if not output_generated.is_cuda:
                            output_generated = output_generated.cuda()

                        ground_truth = entry["output"]
                        if torch.is_tensor(ground_truth):
                            ground_truth = ground_truth.to(
                                output_generated.device)

                        is_correct = torch.allclose(
                            output_generated, ground_truth, atol=1e-2, rtol=1e-1)

                        if not is_correct:
                            all_valid = False
                            diff = torch.abs(output_generated - ground_truth)
                            error_logs.append(
                                f"[Output Mismatch {entry_file.name}] Max diff: {diff.max().item():.6f}")

                    except Exception as e:
                        all_valid = False
                        error_logs.append(
                            f"[Runtime Error {entry_file.name}] {str(e)}")

                if all_valid:
                    q_out.put(
                        (True, f"[Success] All {len(entry_files)} tests passed"))
                else:
                    q_out.put((False, "\n".join(error_logs)))

            except Exception as e:
                # Compilation or other top-level error
                q_out.put((False, f"[System Error] {str(e)}"))

        except KeyboardInterrupt:
            break
        except Exception as e:
            # Fatal worker error
            try:
                q_out.put((False, f"[Worker Crash] {str(e)}"))
            except:
                pass


def _ensure_worker_alive():
    global _WORKER_PROCESS, _WORKER_Q_IN, _WORKER_Q_OUT
    if _WORKER_PROCESS is None or not _WORKER_PROCESS.is_alive():
        if _WORKER_PROCESS is not None:
            print("Verifier: Restarting worker process...")

        # Use spawn context consistently for both Queues and Process
        ctx = multiprocessing.get_context('spawn')

        # Reset queues using the context
        _WORKER_Q_IN = ctx.Queue()
        _WORKER_Q_OUT = ctx.Queue()

        _WORKER_PROCESS = ctx.Process(
            target=_validate_worker_loop, args=(_WORKER_Q_IN, _WORKER_Q_OUT))
        _WORKER_PROCESS.daemon = True  # Kill if parent dies
        _WORKER_PROCESS.start()


def _kill_worker():
    global _WORKER_PROCESS
    if _WORKER_PROCESS:
        _WORKER_PROCESS.terminate()
        _WORKER_PROCESS.join()
        _WORKER_PROCESS = None


def validate_kernel(generated_cu_code: str, paths: dict[str, Path]) -> tuple[bool, str]:
    """
    Validates kernel using the persistent worker.
    """
    tmpdir = paths["tmp_dir"]
    io_dir = paths["io_dir"]

    # 1. Write Code (needed for load_inline debugging/artifacts, though we pass string to worker)
    # The worker also writes/compiles, but writing here ensures artifacts exist for user inspection
    cu_path = os.path.join(tmpdir, "kernel.cu")
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(generated_cu_code)

    # 2. Send to Worker
    _ensure_worker_alive()

    # We pass paths as STRINGS to be safe
    _WORKER_Q_IN.put((generated_cu_code, str(tmpdir), str(io_dir)))

    # 3. Wait for result with timeout
    import time
    import queue
    start_time = time.time()
    TIMEOUT = settings.verifier_timeout_seconds

    while time.time() - start_time < TIMEOUT:
        if not _WORKER_PROCESS.is_alive():
            return False, "[Process Error] Worker process crashed unexpectedly."

        try:
            return _WORKER_Q_OUT.get(timeout=0.5)
        except queue.Empty:
            continue

    # Timeout occurred
    print(f"Warning: Validation timed out. Killing worker.")
    _kill_worker()
    return False, "[Timeout Error] Validation timed out (Infinite loop detected)."



def validate_remote_kernel(ssh_config: dict, generated_cu_code: str, paths: dict[str, Path]) -> tuple[bool, str]:
    """
    Validates a kernel on a remote server using the persistent worker.
    """
    from src.optimizer.core.ssh_client import RemoteWorkerClient, upload_files
    
    try:
        worker = RemoteWorkerClient(ssh_config)
        
        # Upload IO files to shared cache
        io_dir = paths["io_dir"]
        io_files = list(io_dir.glob("*.pt"))
        file_map = {str(f): f.name for f in io_files}
        
        remote_io_dir = "cgins_workspace/io_cache/" + io_dir.name
        upload_files(worker.client, file_map, remote_io_dir)
        
        # Send verify task
        payload = {
            "code": generated_cu_code,
            "io_dir": remote_io_dir
        }
        
        result = worker.send_task("verify", payload)
        worker.close()
        
        if result.get("valid"):
            return True, "Validation Successful"
        else:
            return False, result.get("log", "Unknown Validation Error")
            
    except Exception as e:
        return False, f"Remote Validation Exception: {e}"
