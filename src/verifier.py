"""
src/verifier.py
Validates a generated CUDA kernel by compiling it as a PyTorch C++ extension
and comparing its tensor output against a ground-truth tensor.
"""

import torch
import os
import time

from pathlib import Path
from torch.utils.cpp_extension import load

from byllm.lib import Model, by

llm = Model(model_name="gpt-5")



@by(llm)
def summarize_issue_with_traceback(traceback_error: str, cu_code: str) -> str: ...
"""Summarize the issue of the CUDA kernel within the traceback.
Ignoring warnings, focuses on actual errors.
How to fix CUDA kernel?"""


def handle_output(traceback_error: str, cu_code: str, log_file_path: Path) -> str:
    
    feedback = summarize_issue_with_traceback(traceback_error, cu_code)
    
    output = f"[Traceback Error]:\n{traceback_error}\n\n[LLM Generated Feedback]:\n{feedback}"
    with open(log_file_path, "w") as f:
        f.write(output)

    return feedback
    

def validate_kernel(
    generated_cu_code: str,
    input_path: str,
    ground_truth_path: str,
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
        module = load(
            name=f"generated_module_{os.path.basename(tmpdir)}",
            sources=[cu_path],
            build_directory=tmpdir,
            verbose=True, 
        )
        call_success = True

    except Exception as e:
        # --- Handle compilation failure ---
        call_success = False
        exec_success = False
        log_message = f"[Compilation Failed]\nSummarized traceback:\n {handle_output(str(e), generated_cu_code, log_file_path)}"
        return call_success, exec_success, log_message
    

    # --- 3. Stage 2: Execution Status (Correctness) ---
    try:
        # Load ground truth and inputs
        inputs = torch.load(input_path)
        ground_truth = torch.load(ground_truth_path).cuda()
        
        # Check if inputs contain separate args and kwargs
        if isinstance(inputs, dict) and "args" in inputs and "kwargs" in inputs:
            args = inputs["args"]
            kwargs = inputs["kwargs"]
            
            # Move tensors to CUDA, keep scalars as-is
            cuda_args = []
            for item in args:
                if torch.is_tensor(item):
                    cuda_args.append(item.cuda())
                elif isinstance(item, (int, float, bool, str)):
                    cuda_args.append(item)
                # Skip other types
            
            cuda_kwargs = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    cuda_kwargs[k] = v.cuda()
                elif isinstance(v, (int, float, bool, str)):
                    cuda_kwargs[k] = v
                # Skip other types
            
            # Call with both args and kwargs
            output_generated = module.launch(*cuda_args, **cuda_kwargs)
        
        else:
            # Original behavior: inputs is a list/tuple/tensor/scalar
            if isinstance(inputs, (list, tuple)):
                cuda_inputs = []
                for item in inputs:
                    if torch.is_tensor(item):
                        cuda_inputs.append(item.cuda())
                    elif isinstance(item, (int, float, bool, str)):
                        cuda_inputs.append(item)
                    else:
                        continue
            elif torch.is_tensor(inputs):
                cuda_inputs = [inputs.cuda()]
            else:
                # Single scalar input
                if isinstance(inputs, (int, float, bool, str)):
                    cuda_inputs = [inputs]
                else:
                    exec_success = False
                    print("No valid inputs found")
                    exit(1)
                    return call_success, exec_success, log_message
            
            # Call the compiled kernel
            output_generated = module.launch(*cuda_inputs)
        
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
        # Extract useful metadata about inputs
        input_info = {
            "input_type": type(inputs).__name__,
            "args": [
                {
                    "index": idx,
                    "dtype": str(a.dtype) if torch.is_tensor(a) else None,
                    "shape": list(a.shape) if torch.is_tensor(a) else None,
                    "type": type(a).__name__,
                }
                for idx, a in enumerate(cuda_args) if "cuda_args" in locals()
            ],
            "kwargs": [
                {
                    "name": k,
                    "dtype": str(v.dtype) if torch.is_tensor(v) else None,
                    "shape": list(v.shape) if torch.is_tensor(v) else None,
                    "type": type(v).__name__,
                }
                for k, v in (cuda_kwargs.items() if "cuda_kwargs" in locals() else [])
            ]
        }


        log_message = (
            "[Kernel Runtime Error]\n"
            f"Summarized traceback:\n{handle_output(str(e), generated_cu_code, log_file_path)}\n\n"
            f"Input metadata:\n{input_info}"
        )
        
        return call_success, exec_success, log_message
    
    # --- 4. Final Comparison ---
    if runtime_success:
        try:
            # Use numerical tolerance checking
            is_correct = torch.allclose(output_generated, ground_truth, atol=1e-2, rtol=1e-1)
            
            if is_correct:
                exec_success = True
                log_message += "Validation Successful: Outputs match.\n"
            else:
                exec_success = False
                diff = torch.abs(output_generated - ground_truth)
                output_shape = list(output_generated.shape)
                expected_shape = list(ground_truth.shape)

                log_message += (
                    "[Output Mismatch]\n"
                    f"- Expected shape: {expected_shape}\n"
                    f"- Output shape:   {output_shape}\n"
                    f"- Max difference: {diff.max().item():.6f}\n"
                    f"- Mean difference:{diff.mean().item():.6f}\n"
                    "Likely causes: boundary indexing errors, thread grid size mismatch, "
                    "or incorrect memory writes.\n"
                )

                with open(log_file_path, "w") as f:
                    f.write(f"[Incorrect Output]:\n{log_message}")
        except Exception as e:
            exec_success = False
            log_message += f"Output Comparison Error (Exec Status=False):\n{e}"
    
    # Final return
    return call_success, exec_success, log_message
