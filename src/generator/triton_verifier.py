"""
src/generator/triton_verifier.py
Validates a generated Triton kernel by importing it via importlib
and comparing its tensor output against a ground-truth tensor.
"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

# Re-use helpers from the CUDA verifier
from src.generator.verifier import (
    normalize_args_kwargs,
    move_to_target,
    handle_output,
    _run_extra_validation,
    _short_error,
    _target_device,
)


def _load_triton_module(kernel_path: Path, tmpdir: Path):
    """Import kernel.py as a Python module using importlib."""
    module_name = f"triton_kernel_{os.path.basename(tmpdir)}"
    spec = importlib.util.spec_from_file_location(module_name, str(kernel_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_kernel(
    generated_code: str,
    entry_file: str,
    log_file_path: Path,
    tmpdir: Path,
) -> tuple[bool, bool, str]:
    """
    Validates a Triton kernel by writing kernel.py, importing it,
    calling launch(), and comparing against ground truth.

    Returns:
        tuple: (call_success, exec_success, log_message)
    """
    log_message = ""
    call_success = False
    exec_success = False
    runtime_success = False

    # --- 1. Write Code to File ---
    kernel_path = Path(tmpdir) / "kernel.py"
    with open(kernel_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    # --- 2. Import / Compile Check ---
    try:
        module = _load_triton_module(kernel_path, Path(tmpdir))

        # Ensure launch() exists
        if not hasattr(module, "launch"):
            raise AttributeError(
                "kernel.py must define a 'launch' function, but none was found."
            )
        call_success = True

    except Exception as e:
        call_success = False
        exec_success = False
        tb = traceback.format_exc()
        try:
            entry = torch.load(entry_file)
        except Exception:
            entry = {"input": "Unknown", "output": "Unknown"}

        log_message = (
            f"[Import Failed]\nSummarized traceback:\n"
            f"{handle_output(tb, generated_code, log_file_path, entry)}"
        )
        return call_success, exec_success, log_message

    # --- 3. Execution Check ---
    try:
        entry = torch.load(entry_file)

        args = entry.get("args", [])
        kwargs = entry.get("kwargs", {})
        signature_info = entry.get("signature", {"params": [], "defaults": {}})

        # Normalize to positional arguments
        normalized_args, remaining_kwargs = normalize_args_kwargs(
            args, kwargs, signature_info
        )

        if remaining_kwargs:
            print(f"Warning: Unmapped kwargs: {list(remaining_kwargs.keys())}")

        # Move tensors to target device
        target_args = [move_to_target(item) for item in normalized_args]

        # Call the launch function
        output_generated = module.launch(*target_args)

        # Synchronize
        target = _target_device()
        if target in {"cuda", "triton"}:
            torch.cuda.synchronize()
        elif target == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

        # Ensure output is on the right device
        if torch.is_tensor(output_generated):
            if target in {"cuda", "triton"} and not output_generated.is_cuda:
                output_generated = output_generated.cuda()
            elif target == "cpu":
                output_generated = output_generated.cpu()

        runtime_success = True

    except Exception as e:
        runtime_success = False
        exec_success = False
        tb = traceback.format_exc()

        input_info = {
            "original_args_count": len(args) if "args" in locals() else 0,
            "original_kwargs": list(kwargs.keys()) if "kwargs" in locals() else [],
            "normalized_args_count": len(target_args) if "target_args" in locals() else 0,
            "signature_params": signature_info.get("params", []) if "signature_info" in locals() else [],
            "args": [
                {
                    "index": idx,
                    "dtype": str(a.dtype) if torch.is_tensor(a) else None,
                    "shape": list(a.shape) if torch.is_tensor(a) else None,
                    "type": type(a).__name__,
                }
                for idx, a in enumerate(target_args)
            ] if "target_args" in locals() else [],
        }

        log_message = (
            "[Kernel Runtime Error]\n"
            f"Summarized traceback:\n{handle_output(tb, generated_code, log_file_path, entry)}\n\n"
            f"Input metadata:\n{input_info}"
        )
        return call_success, exec_success, log_message

    # --- 4. Output Comparison ---
    if runtime_success:
        try:
            ground_truth = entry["output"]
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.to(output_generated.device)

            if torch.is_tensor(ground_truth):
                if output_generated.shape != ground_truth.shape:
                    raise ValueError(
                        f"Output shape mismatch: got {list(output_generated.shape)}, "
                        f"expected {list(ground_truth.shape)}"
                    )
                if output_generated.dtype != ground_truth.dtype:
                    raise ValueError(
                        f"Output dtype mismatch: got {output_generated.dtype}, "
                        f"expected {ground_truth.dtype}"
                    )

            if torch.is_tensor(ground_truth) and not ground_truth.is_floating_point():
                is_correct = torch.equal(output_generated, ground_truth)
            else:
                is_correct = torch.allclose(
                    output_generated, ground_truth, atol=1e-2, rtol=1e-1
                )

            if is_correct:
                exec_success = True
                log_message += "Validation Successful: Outputs match.\n"
            else:
                exec_success = False
                diff = torch.abs(output_generated - ground_truth)
                analysis = (
                    "[Output Mismatch]\n"
                    f"- Expected shape: {list(ground_truth.shape)}\n"
                    f"- Output shape:   {list(output_generated.shape)}\n"
                    f"- Max difference: {diff.max().item():.6f}\n"
                    f"- Mean difference:{diff.mean().item():.6f}\n"
                    "Likely causes: incorrect masking, block size, or pointer arithmetic.\n"
                )
                entry["generated-incorrect-output"] = output_generated
                log_message += handle_output(
                    analysis, generated_code, log_file_path, entry
                )
                with open(log_file_path, "w") as f:
                    f.write(f"[Incorrect Output]:\n{log_message}")

        except Exception as e:
            exec_success = False
            log_message += f"Output Comparison Error (Exec Status=False):\n{e}"

    # --- 5. Extra Validation (optional) ---
    if exec_success:
        extra_cases_env = os.environ.get("CGINS_EXTRA_VALIDATION_CASES", "0")
        try:
            extra_cases = int(extra_cases_env)
        except Exception:
            extra_cases = 0
        if extra_cases > 0:
            function_name = entry.get("function_name", "")
            signature_info = entry.get("signature", {"params": [], "defaults": {}})
            ok, extra_msg = _run_extra_validation(
                function_name, module, entry, signature_info, extra_cases
            )
            if not ok:
                exec_success = False
                log_message += f"\n[Extra Validation Failed]\n{extra_msg}"

    return call_success, exec_success, log_message
