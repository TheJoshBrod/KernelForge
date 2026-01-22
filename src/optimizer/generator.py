"""
src/optimizer/generator.py
Uses LLM to generate CUDA kernels that is model-agnostic.
"""
import re
from pathlib import Path
from typing import Optional
from typing import Tuple

import src.optimizer.prompts as prompts
import src.optimizer.verifier as verifier
from src.llm_tools import GenModel

# Global variables
sys_prompt = prompts.get_sys_prompt()


def extract_feedback_and_code(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract feedback and code sections from a formatted string.

    Args:
        content: The input string containing feedback and code sections

    Returns:
        A tuple of (feedback, code) where each is None if not found
    """

    # Extract feedback section (tolerant to spacing)
    feedback_pattern = r'//\s*\[START FEEDBACK\](.*?)//\s*\[END FEEDBACK\]'
    feedback_match = re.search(feedback_pattern, content, re.DOTALL | re.IGNORECASE)
    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided"

    # Extract code section (tolerant to spacing)
    # 1. Try strict tags
    code_pattern = r'//\s*\[START kernel\.cu\](.*?)//\s*\[END kernel\.cu\]'
    code_match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if code_match:
        code = code_match.group(1).strip()
    else:
        # 2. Markdown fallback
        # This handles ```cpp\n ... ``` or ```cuda\n ... ```
        fallback_pattern = r"```(?:C\+\+|cpp|cuda|c)?\s*\n(.*?)```"
        fallback_match = re.search(fallback_pattern, content, re.DOTALL | re.IGNORECASE)
        code = fallback_match.group(1).strip() if fallback_match else None

    return feedback, code


def create_and_validate(llm: GenModel, msg: str, model: str, paths: dict[Path]) -> Tuple[str, bool, str]:
    """Generates a new kernel then validates it for correctness

    Args:
        llm (GenModel): LLM abstraction class with chat history
        msg (str): User message for LLM
        model (str): Name of LLM model
        paths (dict[Path]): Data structure holding different filepaths

    Returns:
        Tuple[str, bool, str]: _description_
    """
    response = llm.chat(msg, model)
    feedback, cu_code = extract_feedback_and_code(response)

    if cu_code is None:
        print("Error: Could not extract code from LLM response.")
        print(f"Raw response:\n{response}")
        return feedback, False, "Failed to extract code"

    is_valid, error = verifier.validate_kernel(cu_code, paths)
    return feedback, is_valid, error


def generate(best_kernel_code: str, gpu_specs: dict, improvement_log: list, paths: dict[str, Path], model: str = "claude-opus-4-5-20251101") -> Tuple[str, bool]:
    """Generates and validates CUDA kernels 

    Args:
        gpu_specs (dict): Specs of specific GPU architecture
        op_dir (str): Directory of previously generated CUDA kernel  
        improvement_log (list): "Chat History" of why LLM thinks it made an improvement over past attempts
        temp_dir (Path): Path of directory to compile kernel into (used later by profiler)
        io_dir (Path): Path of file that contains all input/output pairs recorded of this op
        model (str, optional): LLM that will generate kernels. Defaults to None (will use env var or default).
    """
    if model is None or model == "claude-3-5-sonnet-20240620":
        import os
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        if provider == "gemini":
            model = "gemini-2.0-flash-exp"
        else:
            model = "claude-3-5-sonnet-20240620"

    # Attempt initial CUDA code generation
    llm: GenModel = GenModel(sys_prompt)
    msg = prompts.generate_gpu_optimization_prompt(
        gpu_specs, best_kernel_code, improvement_log)
    feedback, is_valid, error = create_and_validate(llm, msg, model, paths)
    if is_valid:
        return feedback, True
    print("\t\tInitial gen failed...")
    # On failure attempt fix 3 times before giving up
    for i in range(3):
        print(f"\t\t\tReattempt {i}")
        _, is_valid, error = create_and_validate(llm, error, model, paths)
        if is_valid:
            return feedback, True

    return "", False
