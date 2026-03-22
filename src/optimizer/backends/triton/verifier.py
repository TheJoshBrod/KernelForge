"""
Triton Backend Verifier.
Validates generated Triton kernels by dynamically importing them and comparing 
their outputs against ground-truth tensors.
"""

import os
import sys
import importlib
import importlib.util
import time
import traceback
from pathlib import Path

import torch

from byllm.lib import by, Model
from src.optimizer.config.settings import settings
from src.optimizer.backends.error_utils import format_verifier_output

llm = Model(model_name=settings.llm_model_name)


@by(llm)
def summarize_issue_with_traceback(
        traceback_error: str,
        triton_code: str,
        input_and_output: dict
    ) -> str:
        """
        Analyze Triton kernel compilation or runtime errors and provide actionable fix suggestions.

        Important:
        - The caller-side argument formatting is already correct and MUST NOT be modified.
        - The issue will always be internal to the Triton kernel code (kernel logic,
            block sizes, masking, pointer arithmetic, or launch configuration).

        The input_and_output dict contains:
        - args: List of positional arguments (normalized from kwargs if applicable)
        - signature: Dict with 'params' (parameter names in order) and 'defaults'
        - output or correct-output: Expected output tensor

        The Triton kernel's launch() function MUST accept arguments in exactly this order:
        {', '.join(input_and_output.get('signature', {}).get('params', ['arg0', 'arg1', '...']))}

        Provide specific recommendations for:
        1. Correct argument ordering and types in the launch() function
        2. Correct use of tl.load/tl.store with proper masking
        3. Block size, grid calculation, and pointer arithmetic issues
        4. Missing imports or decorator issues

        Do NOT suggest modifying Python call sites or argument conversion logic.
        Only Triton-side fixes are relevant.

        Do NOT suggest using:
        - `.data_ptr()` when launching the kernel — tensors must be passed directly, NEVER as
          raw pointers via .data_ptr(). Triton infers pointer types from the tensor dtype.
          Passing int64 from .data_ptr() causes "Unsupported ptr type" errors.
        - `tl.pointer`, `tl.float32_ptr`, `tl.uint64` as type annotations on pointer params —
          none of these exist. Do NOT annotate pointer parameters at all. Only `tl.constexpr`.
        - `tl.broadcast(tensor, (A, B))` with a tuple shape — this causes `'tuple_type' has no
          attribute 'is_block'`. For 2D indexing use `t[:, None]` and `t[None, :]` instead.
        - `continue` or `break` inside the kernel (not supported in Triton JIT)
        - `tl.any()` or `tl.all()` (do not exist in tl namespace)
        - `while` loops (not supported — use `for _ in range(N)`)
        - Data-dependent `if` over tensor values (use `tl.where` instead)
        For early-exit patterns, always recommend removing the exit and using masked operations.
        For any/all checks, recommend `tl.reduce_or(mask)` or restructuring with tl.where.
        """


def handle_output(traceback_error: str, triton_code: str, log_file_path: Path, input_and_output: dict) -> str:
    summarizer = summarize_issue_with_traceback if settings.llm_model_name else None
    return format_verifier_output(
        traceback_error=traceback_error,
        kernel_code=triton_code,
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
            # print(f"Warning: No value for parameter '{param_name}' at position {i}")
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


def _load_triton_module(kernel_path: Path):
    """
    Dynamically import a Triton kernel module from a .py file.
    """
    module_name = kernel_path.stem + f"_{id(kernel_path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(kernel_path))
    if spec is None:
        raise ImportError(f"Cannot create module spec from {kernel_path}")
    module = importlib.util.module_from_spec(spec)

    # Don't pollute sys.modules with temporary modules 
    # (allows reimporting changed files)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to import Triton kernel: {e}") from e

    return module


def validate_kernel(generated_py_code: str, paths: dict[str, Path]) -> tuple[bool, str]:
    """
    Validates a Triton kernel by importing it and comparing outputs.
    """
    tmpdir = paths["tmp_dir"]
    io_dir = paths["io_dir"]

    # Write kernel code to file
    kernel_path = Path(tmpdir) / "kernel.py"
    kernel_path.write_text(generated_py_code, encoding="utf-8")

    try:
        # Import the module
        try:
            module = _load_triton_module(kernel_path)
        except Exception as e:
            # Import/Compilation Error
            tb = traceback.format_exc()
            dummy_entry = {"input": "Unknown", "output": "Unknown"}
            log_msg = handle_output(tb, generated_py_code, None, dummy_entry)
            return False, f"[Import/Compilation Failed]\n{log_msg}"

        if not hasattr(module, 'launch'):
            return False, "[Error] Triton kernel module has no 'launch()' function"

        # Run against test entries
        selected_entry_files = paths.get("entry_files") or []
        if selected_entry_files:
            entry_files = []
            for raw_path in selected_entry_files:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = Path(io_dir) / candidate.name
                if candidate.exists():
                    entry_files.append(candidate)
            entry_files = sorted(entry_files)
        else:
            entry_files = sorted(Path(io_dir).glob("entry_*.pt"))
        if not entry_files:
            return False, "[Error] No entry files found in io_dir"

        all_valid = True
        error_logs = []
        llm_analysis_count = 0 
        MAX_LLM_ANALYSIS = 1

        for entry_file in entry_files:
            try:
                entry = torch.load(entry_file)
                args = entry.get("args", [])
                kwargs = entry.get("kwargs", {})
                signature_info = entry.get("signature", {"params": [], "defaults": {}})

                normalized_args, remaining_kwargs = normalize_args_kwargs(args, kwargs, signature_info)
                # Triton uses CUDA logic for tensors usually
                cuda_args = [move_to_cuda(item) for item in normalized_args]

                output_generated = module.launch(*cuda_args)

                # Synchronize GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Ensure output is on GPU
                if torch.is_tensor(output_generated) and not output_generated.is_cuda:
                    output_generated = output_generated.cuda()

                ground_truth = entry["output"]
                if torch.is_tensor(output_generated) and torch.is_tensor(ground_truth):
                    ground_truth = ground_truth.to(output_generated.device)

                if torch.is_tensor(output_generated) and torch.is_tensor(ground_truth):
                    if not ground_truth.is_floating_point():
                        is_correct = torch.equal(output_generated, ground_truth)
                    else:
                        is_correct = torch.allclose(
                            output_generated, ground_truth, atol=1e-2, rtol=1e-1)
                elif torch.is_tensor(output_generated) and output_generated.numel() == 1 and not torch.is_tensor(ground_truth):
                    is_correct = output_generated.detach().cpu().item() == ground_truth
                elif (not torch.is_tensor(output_generated)) and torch.is_tensor(ground_truth) and ground_truth.numel() == 1:
                    is_correct = output_generated == ground_truth.detach().cpu().item()
                else:
                    is_correct = output_generated == ground_truth

                if not is_correct:
                    all_valid = False
                    if torch.is_tensor(output_generated) and torch.is_tensor(ground_truth):
                        diff = torch.abs(output_generated - ground_truth)
                        analysis_msg = (
                            f"[Output Mismatch {entry_file.name}]\n"
                            f"Max diff: {diff.max().item():.6f}\n"
                            f"Mean diff: {diff.mean().item():.6f}\n"
                        )
                    else:
                        analysis_msg = (
                            f"[Output Mismatch {entry_file.name}]\n"
                            f"Generated: {output_generated}\n"
                            f"Expected: {ground_truth}\n"
                        )
                    
                    # Generate detailed analysis only for the first few failures
                    if llm_analysis_count < MAX_LLM_ANALYSIS:
                        entry["generated-incorrect-output"] = output_generated
                        log_msg = handle_output(analysis_msg, generated_py_code, None, entry)
                        llm_analysis_count += 1
                    else:
                        log_msg = f"{analysis_msg}\n(LLM Analysis skipped for subsequent errors)"
                        
                    error_logs.append(log_msg)

            except Exception as e:
                all_valid = False
                tb = traceback.format_exc()
                if llm_analysis_count < MAX_LLM_ANALYSIS:
                    log_msg = handle_output(tb, generated_py_code, None, entry)
                    llm_analysis_count += 1
                else:
                    log_msg = f"[Runtime Error {entry_file.name}]\n{str(e)}\n(LLM Analysis skipped)"
                error_logs.append(log_msg)

        if all_valid:
            return True, f"[Success] All {len(entry_files)} tests passed"
        else:
            return False, "\n\n".join(error_logs)

    except Exception as e:
        return False, f"[System Error] {str(e)}"


def validate_remote_kernel(ssh_config: dict, generated_py_code: str, paths: dict[str, Path]) -> tuple[bool, str]:
    """
    Validates a Triton kernel on a remote server via SSH.
    """
    from src.optimizer.core.ssh_client import RemoteWorkerClient, upload_files

    try:
        worker = RemoteWorkerClient(ssh_config)

        # Upload IO files to shared cache
        io_dir = paths["io_dir"]
        selected_entry_files = paths.get("entry_files") or []
        io_files = (
            [Path(entry_file) for entry_file in selected_entry_files]
            if selected_entry_files
            else list(io_dir.glob("*.pt"))
        )
        file_map = {str(f): f.name for f in io_files}

        remote_io_dir = "kforge_workspace/io_cache/" + io_dir.name
        upload_files(worker.client, file_map, remote_io_dir)

        # Send verify task
        payload = {
            "code": generated_py_code,
            "io_dir": remote_io_dir,
            "entry_files": [Path(entry_file).name for entry_file in selected_entry_files],
        }

        result = worker.send_task("verify", payload)
        worker.close()

        if result.get("valid"):
            return True, "Validation Successful"
        else:
            return False, result.get("log", "Unknown Validation Error")

    except Exception as e:
        return False, f"Remote Validation Exception: {e}"
