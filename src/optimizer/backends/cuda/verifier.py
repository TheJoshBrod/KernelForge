"""
src/optimizer/components/worker/verifier.py
Validates a generated CUDA kernel by compiling it as a PyTorch C++ extension
and comparing its tensor output against a ground-truth tensor.
"""

import os
import re
import hashlib
import tempfile
import multiprocessing
import queue
import traceback
from pathlib import Path

import torch

from byllm.lib import by, Model
from torch.utils.cpp_extension import load_inline
from src.optimizer.config.settings import settings
from src.optimizer.backends.error_utils import format_verifier_output
import src.optimizer.backends.cuda.loader as loader

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

        When the error contains Output Mismatch diagnostics:
        - If output contains Inf values: first check numeric overflow before assuming algorithmic error.
          Reduced-precision types (fp16, bf16) saturate at finite limits; accumulating in the same
          type causes overflow. Fix: use a float or double accumulator and cast back at the end.
        - If output contains NaN values: check for uninitialized shared memory, division by zero,
          or operations on uninitialized accumulators.
        - Compare Input dtype vs Output dtype: a mismatch often indicates a missing cast.
        - Do not assume a wrong algorithm until precision and casting issues are ruled out.
        """


def handle_output(traceback_error: str, cu_code: str, log_file_path: Path, input_and_output: dict) -> str:
    summarizer = summarize_issue_with_traceback if settings.llm_model_name else None
    return format_verifier_output(
        traceback_error=traceback_error,
        kernel_code=cu_code,
        log_file_path=log_file_path,
        input_and_output=input_and_output,
        summarizer=summarizer,
    )


def normalize_args_kwargs(args: list, kwargs: dict, signature_info: dict) -> tuple[list, dict]:
    """
    Normalize args and kwargs into a complete positional argument list.
    """
    params = signature_info.get("params", [])
    defaults = signature_info.get("defaults", {})

    if not params:
        return args, kwargs

    normalized = list(args)
    remaining_kwargs = dict(kwargs)

    for i in range(len(normalized), len(params)):
        param_name = params[i]
        if param_name in remaining_kwargs:
            normalized.append(remaining_kwargs.pop(param_name))
        elif param_name in defaults:
            normalized.append(defaults[param_name])
        else:
            print(f"Warning: No value for parameter '{param_name}' at position {i}")
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


def move_to_target(item):
    target = loader.target_device()
    if target == "cpu":
        if torch.is_tensor(item):
            return item.cpu()
        if isinstance(item, (list, tuple)):
            return type(item)(move_to_target(x) for x in item)
        if isinstance(item, dict):
            return {k: move_to_target(v) for k, v in item.items()}
        return item
    if target == "mps":
        if torch.is_tensor(item):
            return item.to("mps")
        if isinstance(item, (list, tuple)):
            return type(item)(move_to_target(x) for x in item)
        if isinstance(item, dict):
            return {k: move_to_target(v) for k, v in item.items()}
        return item
    return move_to_cuda(item)


# --- Persistent Worker Globals ---
_WORKER_PROCESS = None
_WORKER_Q_IN = None
_WORKER_Q_OUT = None


def _validate_worker_loop(q_in, q_out):
    """
    Persistent worker loop.
    Waits for job tuples (see validate_kernel for layout).
    Sends back (is_valid, log_message).
    """
    # Set environment variables ONCE
    import sys

    # Delegate environment setup to loader
    if loader.target_device() == "cuda":
        loader.ensure_cuda_env()

    # Pre-import torch to warm up (already imported at top level, but ensure context is ready)
    if loader.target_device() == "cuda" and torch.cuda.is_available():
        torch.cuda.init()

    while True:
        try:
            job = q_in.get()
            if job is None:
                break  # Sentinel to exit

            (generated_cu_code, tmpdir, io_dir, cache_name, cache_dir,
             proj_root, job_key, operator, iteration, attempt) = job

            # Register LiteLLM success callback so any LLM call made by the
            # verifier's byllm-decorated summarizer gets logged to the project
            # usage DB with step_type='verifier_summary'.
            if proj_root is not None:
                try:
                    from src.llm.litellm_callback import register_worker_usage_callback
                    register_worker_usage_callback(
                        proj_root, job_key, operator, iteration, attempt
                    )
                except Exception:
                    pass

            # Run the validation logic
            try:
                # Load the pre-compiled module from the cache dir populated by the main process.
                # compile_code_string finds the existing .so and skips nvcc entirely.
                try:
                    module = loader.compile_code_string(
                        code=generated_cu_code,
                        name=cache_name,
                        build_dir=cache_dir,
                        verbose=False
                    )
                except Exception as e:
                    # Compilation Error
                    tb = traceback.format_exc()
                    # We don't have a specific entry yet, so provide a dummy one for LLM context
                    dummy_entry = {"input": "Unknown", "output": "Unknown"}
                    log_msg = handle_output(tb, generated_cu_code, None, dummy_entry)
                    q_out.put((False, f"[Compilation Failed]\n{log_msg}"))
                    continue

                # Execution Test
                entry_files = sorted(Path(io_dir).glob("entry_*.pt"))
                if not entry_files:
                    q_out.put(
                        (False, "[Error] No entry files found in io_dir"))
                    continue

                # Determine canonical param order once for all entries, matching
                # _infer_param_order in prompts.py: prefer an explicit recorded
                # signature; otherwise build from the first entry's positional args
                # plus the union of all kwargs keys seen across every entry (in
                # first-seen order).  Using a per-entry order would give different
                # arities to module.launch() for entries with different kwargs
                # structures (e.g. some passing stride=2, others relying on defaults).
                entries = []
                canonical_signature = None
                for f in entry_files:
                    e = torch.load(f)
                    entries.append(e)
                    if canonical_signature is None:
                        sig = e.get("signature", {})
                        if sig and sig.get("params"):
                            canonical_signature = sig

                if canonical_signature is None and entries:
                    first_args = entries[0].get("args", []) or []
                    param_order = [f"arg{i}" for i in range(len(first_args))]
                    seen = set(param_order)
                    for entry in entries:
                        kw = entry.get("kwargs", {}) or {}
                        if isinstance(kw, dict):
                            for k in kw:
                                if k not in seen:
                                    param_order.append(k)
                                    seen.add(k)
                    canonical_signature = {"params": param_order, "defaults": {}}

                all_valid = True
                error_logs = []
                llm_analysis_count = 0
                MAX_LLM_ANALYSIS = 1

                for entry_file, entry in zip(entry_files, entries):
                    try:
                        args = entry.get("args", [])
                        kwargs = entry.get("kwargs", {})

                        normalized_args, remaining_kwargs = normalize_args_kwargs(args, kwargs, canonical_signature)
                        cuda_args = [move_to_target(item) for item in normalized_args]

                        output_generated = module.launch(*cuda_args)
                        if loader.target_device() == "cuda":
                            torch.cuda.synchronize()
                        elif loader.target_device() == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                            torch.mps.synchronize()

                        if loader.target_device() == "cuda" and torch.is_tensor(output_generated) and not output_generated.is_cuda:
                            output_generated = output_generated.cuda()
                        elif loader.target_device() == "mps" and torch.is_tensor(output_generated):
                            output_generated = output_generated.to("mps")

                        ground_truth = entry["output"]

                        if torch.is_tensor(output_generated) and torch.is_tensor(ground_truth):
                            ground_truth = ground_truth.to(output_generated.device)
                            if not ground_truth.is_floating_point():
                                is_correct = torch.equal(output_generated, ground_truth)
                            else:
                                is_correct = torch.allclose(
                                    output_generated, ground_truth, atol=1e-2, rtol=1e-1
                                )
                        elif torch.is_tensor(output_generated) and output_generated.numel() == 1 and not torch.is_tensor(ground_truth):
                            is_correct = output_generated.detach().cpu().item() == ground_truth
                        elif (not torch.is_tensor(output_generated)) and torch.is_tensor(ground_truth) and ground_truth.numel() == 1:
                            is_correct = output_generated == ground_truth.detach().cpu().item()
                        else:
                            is_correct = output_generated == ground_truth

                        if not is_correct:
                            all_valid = False
                            if torch.is_tensor(output_generated) and torch.is_tensor(ground_truth):
                                out_f = output_generated.float()
                                gt_f = ground_truth.float()

                                inf_count = torch.isinf(output_generated).sum().item()
                                nan_count = torch.isnan(output_generated).sum().item()

                                diag = [f"[Output Mismatch {entry_file.name}]"]
                                diag.append(f"Output: dtype={output_generated.dtype}, shape={list(output_generated.shape)}")
                                diag.append(f"Expected: dtype={ground_truth.dtype}, shape={list(ground_truth.shape)}")

                                if inf_count > 0:
                                    diag.append(f"WARNING: {int(inf_count)} Inf values in output - possible numeric overflow")
                                    finite_out = out_f[torch.isfinite(out_f)]
                                    if finite_out.numel() > 0:
                                        diag.append(f"  Finite output range: [{finite_out.min().item():.4g}, {finite_out.max().item():.4g}]")
                                    diag.append(f"  Expected range: [{gt_f.min().item():.4g}, {gt_f.max().item():.4g}]")
                                elif nan_count > 0:
                                    diag.append(f"WARNING: {int(nan_count)} NaN values in output")
                                else:
                                    diff = torch.abs(out_f - gt_f)
                                    diag.append(f"Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
                                    diag.append(f"Output range: [{out_f.min().item():.4g}, {out_f.max().item():.4g}]")

                                for i, arg in enumerate(normalized_args):
                                    if torch.is_tensor(arg):
                                        diag.append(f"Input[{i}]: dtype={arg.dtype}, shape={list(arg.shape)}")

                                scalar_kw = {k: v for k, v in (entry.get("kwargs") or {}).items()
                                             if not torch.is_tensor(v)}
                                if scalar_kw:
                                    diag.append(f"Kwargs: {scalar_kw}")

                                analysis_msg = "\n".join(diag) + "\n"
                            else:
                                analysis_msg = (
                                    f"[Output Mismatch {entry_file.name}]\n"
                                    f"Generated: {output_generated}\n"
                                    f"Expected: {ground_truth}\n"
                                )
                            
                            if llm_analysis_count < MAX_LLM_ANALYSIS:
                                entry["generated-incorrect-output"] = output_generated
                                log_msg = handle_output(analysis_msg, generated_cu_code, None, entry)
                                llm_analysis_count += 1
                            else:
                                log_msg = f"{analysis_msg}\n(LLM Analysis skipped)"
                                
                            error_logs.append(log_msg)

                    except Exception as e:
                        all_valid = False
                        tb = traceback.format_exc()
                        if llm_analysis_count < MAX_LLM_ANALYSIS:
                            log_msg = handle_output(tb, generated_cu_code, None, entry)
                            llm_analysis_count += 1
                        else:
                             log_msg = f"[Runtime Error {entry_file.name}]\n{str(e)}\n(LLM Analysis skipped)"
                        error_logs.append(log_msg)

                if all_valid:
                    q_out.put(
                        (True, f"[Success] All {len(entry_files)} tests passed"))
                else:
                    q_out.put((False, "\n\n".join(error_logs)))

            except Exception as e:
                # Other top-level error
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

    # Write kernel code for artifacts/debugging
    cu_path = os.path.join(tmpdir, "kernel.cu")
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(generated_cu_code)

    # Pre-compile in the main process so the worker subprocess only needs to
    # load the already-built .so — avoids nvcc hanging inside a spawned process.
    code_hash = hashlib.md5(generated_cu_code.encode()).hexdigest()[:16]
    cache_name = f"kforge_{code_hash}"
    cache_dir = os.path.join(tempfile.gettempdir(), "kforge_build_cache", cache_name)
    try:
        loader.compile_code_string(
            code=generated_cu_code,
            name=cache_name,
            build_dir=cache_dir,
            verbose=False,
        )
    except Exception as compile_err:
        return False, f"[Compilation Failed]\n{compile_err}"

    # Derive usage-tracking context from paths (project root, operator, etc.)
    op_proj_dir = paths.get("proj_dir") if isinstance(paths, dict) else None
    proj_root = None
    if op_proj_dir is not None:
        try:
            from src.llm.usage_db import project_usage_dir_from_op_dir
            proj_root = project_usage_dir_from_op_dir(op_proj_dir)
        except Exception:
            proj_root = op_proj_dir.parent
    operator = op_proj_dir.name if op_proj_dir is not None else None
    iteration = paths.get("iteration") if isinstance(paths, dict) else None
    attempt = paths.get("attempt") if isinstance(paths, dict) else None
    job_key = os.environ.get("KFORGE_JOB_KEY")

    # Send to worker (cache_name + cache_dir so it loads the pre-built .so)
    _ensure_worker_alive()
    _WORKER_Q_IN.put((
        generated_cu_code, str(tmpdir), str(io_dir), cache_name, cache_dir,
        proj_root, job_key, operator, iteration, attempt,
    ))

    # Wait for result with timeout
    import time
    import queue
    start_time = time.time()
    TIMEOUT = settings.verifier_timeout_seconds

    while time.time() - start_time < TIMEOUT:
        if not _WORKER_PROCESS.is_alive():
            return False, "[Process Error] Worker process crashed unexpectedly."

        try:
            result = _WORKER_Q_OUT.get(timeout=0.5)
            return result
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
    if "tmp_dir" in paths:
        cu_path = os.path.join(paths["tmp_dir"], "kernel.cu")
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(generated_cu_code)

    from src.optimizer.core.ssh_client import RemoteWorkerClient, upload_files
    
    try:
        worker_path = Path(__file__).parent / "remote_worker.py"
        loader_path = Path(__file__).parent / "loader.py"
        
        worker = RemoteWorkerClient(ssh_config, worker_path, {str(loader_path): "loader.py"})
        
        # Upload IO files to shared cache
        io_dir = paths["io_dir"]
        io_files = list(io_dir.glob("*.pt"))
        file_map = {str(f): f.name for f in io_files}
        
        remote_io_dir = "kforge_workspace/io_cache/" + io_dir.name
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
