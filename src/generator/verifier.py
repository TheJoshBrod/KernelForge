"""
src/generator/verifier.py
Validates a generated CUDA kernel by compiling it as a PyTorch C++ extension
and comparing its tensor output against a ground-truth tensor.
"""
import os
import re
from pathlib import Path

import torch
from byllm.lib import by  # type: ignore
from byllm.lib import Model  # type: ignore
from torch.utils.cpp_extension import load_inline
from src.config import ensure_llm_config

_SUMMARY_SYSTEM_PROMPT = """
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
""".strip()


def _short_error(msg: str, limit: int = 2000) -> str:
    if not msg:
        return ""
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."

def _get_provider() -> str:
    provider = ensure_llm_config().strip().lower()
    if provider:
        return provider
    return os.environ.get("LLM_PROVIDER", "").strip().lower()


def _byllm_model_name(provider: str) -> str:
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini/gemini-2.0-flash-exp")
    return os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")


def _summarize_with_byllm(
    traceback_error: str,
    cu_code: str,
    input_and_output: dict,
    model_name: str,
) -> str:
    llm = Model(model_name=model_name)

    @by(llm)
    def _summarize(
        traceback_error: str,
        cu_code: str,
        input_and_output: dict,
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

    return _summarize(traceback_error, cu_code, input_and_output)


def summarize_issue_with_traceback(
    traceback_error: str,
    cu_code: str,
    input_and_output: dict,
) -> str:
    provider = _get_provider()
    if not provider or provider in {"openai", "gpt", "chatgpt"}:
        return traceback_error

    model_name = _byllm_model_name(provider)
    return _summarize_with_byllm(traceback_error, cu_code, input_and_output, model_name)


def handle_output(traceback_error: str, cu_code: str, log_file_path: Path, input_and_output: dict) -> str:

    input_and_output["correct-output"] = input_and_output["output"]
    del input_and_output["output"]

<<<<<<< HEAD
    raw_error = _short_error(traceback_error)

    provider = _get_provider()
    disable_summary = os.environ.get("CGINS_DISABLE_SUMMARY", "").lower() in {"1", "true", "yes"}

    if disable_summary or not provider:
        feedback = raw_error
    elif provider in {"openai", "gpt", "chatgpt"}:
        feedback = raw_error
    else:
        try:
            feedback = summarize_issue_with_traceback(
                traceback_error, cu_code, input_and_output
            )
        except Exception:
            feedback = traceback_error

    output = (
        f"[Raw Error]:\n{raw_error}\n\n"
        f"[LLM Generated Feedback]:\n{feedback}"
    )
    with open(log_file_path, "w") as f:
        f.write(output)

    return output


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


def _resolve_function(function_name: str):
    if not function_name:
        return None
    if function_name.startswith("torch.nn.functional."):
        name = function_name.split(".")[-1]
        return getattr(torch.nn.functional, name, None)
    if function_name.startswith("torch."):
        name = function_name.split(".")[-1]
        return getattr(torch, name, None)
    return None


def _is_safe_for_extra_tests(function_name: str, args: list, kwargs: dict) -> bool:
    safe_ops = {
        "torch.nn.functional.linear",
        "torch.nn.functional.gelu",
        "torch.nn.functional.layer_norm",
    }
    if function_name not in safe_ops:
        return False
    for item in list(args) + list(kwargs.values()):
        if torch.is_tensor(item) and not item.is_floating_point():
            return False
    return True


def _random_like_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        return torch.randn_like(t)
    if t.dtype == torch.int64:
        return torch.randint(low=0, high=10, size=t.shape, dtype=t.dtype)
    return t.clone()


def _generate_random_inputs(entry: dict):
    args = entry.get("args", [])
    kwargs = entry.get("kwargs", {})
    new_args = []
    for a in args:
        if torch.is_tensor(a):
            new_args.append(_random_like_tensor(a))
        else:
            new_args.append(a)
    new_kwargs = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            new_kwargs[k] = _random_like_tensor(v)
        else:
            new_kwargs[k] = v
    return new_args, new_kwargs


def _run_extra_validation(
    function_name: str,
    module,
    entry: dict,
    signature_info: dict,
    extra_cases: int,
) -> tuple[bool, str]:
    func = _resolve_function(function_name)
    if not func:
        return True, ""
    if not _is_safe_for_extra_tests(function_name, entry.get("args", []), entry.get("kwargs", {})):
        return True, ""

    for _ in range(extra_cases):
        args, kwargs = _generate_random_inputs(entry)
        normalized_args, remaining_kwargs = normalize_args_kwargs(
            args, kwargs, signature_info
        )
        if remaining_kwargs:
            return False, f"Extra validation failed: unmapped kwargs {list(remaining_kwargs.keys())}"
        cuda_args = [move_to_cuda(item) for item in normalized_args]
        try:
            with torch.no_grad():
                expected = func(*cuda_args)
        except Exception as e:
            return False, f"Extra validation failed calling PyTorch op: {e}"

        try:
            got = module.launch(*cuda_args)
        except Exception as e:
            return False, f"Extra validation failed calling kernel: {e}"

        if expected.shape != got.shape:
            return False, "Extra validation failed: output shape mismatch"
        if expected.dtype != got.dtype:
            return False, "Extra validation failed: output dtype mismatch"
        if torch.is_tensor(expected):
            if expected.is_floating_point():
                ok = torch.allclose(got, expected, atol=1e-2, rtol=1e-1)
            else:
                ok = torch.equal(got, expected)
            if not ok:
                return False, "Extra validation failed: output mismatch"

    return True, ""


def validate_kernel(
    generated_cu_code: str,
    entry_file: str,
    log_file_path: Path,
    tmpdir: Path
) -> tuple[bool, bool, str]:
    """
    Validates a single-file CUDA kernel using the PyTorch C++ extension API.

    Returns:
        tuple: (call_success, exec_success, log_message)
    """

    log_message = ""

    call_success = False
    exec_success = False
    runtime_success = False

    # --- 1. Stage 1: Write Code to File ---
    cu_path = os.path.join(tmpdir, "kernel.cu")

    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(generated_cu_code)

    # --- 2. Stage 1: Call Status (Compilation) ---
    try:
        # Set environment variables for correct CUDA version
        # Also add python executable dir to PATH to ensure ninja is found
        import sys
        python_bin_dir = os.path.dirname(sys.executable)
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
        os.environ["PATH"] = f"{python_bin_dir}:/usr/local/cuda-12.1/bin:{os.environ['PATH']}"

        # Extract function signature for load_inline
        # Looking for: torch::Tensor launch(...)
        match = re.search(
            r"(torch::Tensor\s+launch\s*\([^)]*\))", generated_cu_code)
        if not match:
            raise ValueError(
                "Could not find 'launch' function signature in generated code.")

        cpp_source = match.group(1) + ";"

        module = load_inline(
            name=f"generated_module_{os.path.basename(tmpdir)}",
            cpp_sources=cpp_source,
            cuda_sources=generated_cu_code,
            functions=['launch'],
            build_directory=tmpdir,
            verbose=True,
            with_cuda=True
        )
        call_success = True

    except Exception as e:
        # --- Handle compilation failure ---
        call_success = False
        exec_success = False
        # Need to load entry to pass to handle_output if possible, or pass empty dict
        try:
            entry = torch.load(entry_file)
        except:
            entry = {"input": "Unknown", "output": "Unknown"}

        log_message = f"[Compilation Failed]\nSummarized traceback:\n {handle_output(str(e), generated_cu_code, log_file_path, entry)}"
        return call_success, exec_success, log_message

    # --- 3. Stage 2: Execution Status (Correctness) ---
    try:
        # Load ground truth and inputs
        entry = torch.load(entry_file)

        # Extract args, kwargs, and signature info
        args = entry.get("args", [])
        kwargs = entry.get("kwargs", {})
        signature_info = entry.get("signature", {"params": [], "defaults": {}})

        # Normalize to positional arguments
        normalized_args, remaining_kwargs = normalize_args_kwargs(
            args, kwargs, signature_info)

        # Warn if there are kwargs that couldn't be mapped
        if remaining_kwargs:
            print(f"Warning: Unmapped kwargs: {list(remaining_kwargs.keys())}")

        # Move tensors to CUDA, keep scalars as-is
        cuda_args = [move_to_cuda(item) for item in normalized_args]

        # Call with ONLY positional args (load_inline doesn't support kwargs)
        output_generated = module.launch(*cuda_args)

        # Ensure all CUDA operations complete
        torch.cuda.synchronize()

        # Move to same device as ground truth if needed
        if not output_generated.is_cuda:
            output_generated = output_generated.cuda()

        # Kernel executed without a runtime error
        runtime_success = True

    except Exception as e:
        # --- Handle runtime errors ---
        runtime_success = False
        exec_success = False

        # Build detailed input info including normalization details
        input_info = {
            "original_args_count": len(args) if "args" in locals() else 0,
            "original_kwargs": list(kwargs.keys()) if "kwargs" in locals() else [],
            "normalized_args_count": len(cuda_args) if "cuda_args" in locals() else 0,
            "signature_params": signature_info.get("params", []) if "signature_info" in locals() else [],
            "args": [
                {
                    "index": idx,
                    "dtype": str(a.dtype) if torch.is_tensor(a) else None,
                    "shape": list(a.shape) if torch.is_tensor(a) else None,
                    "type": type(a).__name__,
                }
                for idx, a in enumerate(cuda_args) if "cuda_args" in locals()
            ],
        }

        log_message = (
            "[Kernel Runtime Error]\n"
            f"Summarized traceback:\n{handle_output(str(e), generated_cu_code, log_file_path, entry)}\n\n"
            f"Input metadata:\n{input_info}"
        )

        return call_success, exec_success, log_message

    # --- 4. Final Comparison ---
    if runtime_success:
        try:
            # Use numerical tolerance checking
            ground_truth = entry["output"]
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.to(output_generated.device)

            if torch.is_tensor(ground_truth):
                if output_generated.shape != ground_truth.shape:
                    raise ValueError(
                        f"Output shape mismatch: got {list(output_generated.shape)}, expected {list(ground_truth.shape)}"
                    )
                if output_generated.dtype != ground_truth.dtype:
                    raise ValueError(
                        f"Output dtype mismatch: got {output_generated.dtype}, expected {ground_truth.dtype}"
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
                output_shape = list(output_generated.shape)
                expected_shape = list(ground_truth.shape)

                analysis = (
                    "[Output Mismatch]\n"
                    f"- Expected shape: {expected_shape}\n"
                    f"- Output shape:   {output_shape}\n"
                    f"- Max difference: {diff.max().item():.6f}\n"
                    f"- Mean difference:{diff.mean().item():.6f}\n"
                    "Likely causes: boundary indexing errors, thread grid size mismatch, "
                    "or incorrect memory writes.\n"
                )

                entry["generated-incorrect-output"] = output_generated
                log_message += handle_output(
                    analysis, generated_cu_code, log_file_path, entry
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

    # Final return
    return call_success, exec_success, log_message
