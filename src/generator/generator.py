"""
src/generator/generator.py
Uses LLM to generate CUDA kernels that is semi-model-agnostic.
"""
import re

from src.llm_tools import GenModel
import src.generator.prompts.prompts


def cleanup_mkdown(input: str) -> str:
    """Extract code using strict tags (preferred) or markdown code blocks (fallback)."""

    # 1. Try Strict Tags (Recommended)
    # Supports both CUDA (// [START kernel.cu]) and Triton (# [START kernel.py])
    tag_pattern = r'(?://|#)\s*\[START kernel\.(?:cu|py)\](.*?)(?://|#)\s*\[END kernel\.(?:cu|py)\]'
    match = re.search(tag_pattern, input, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Try Markdown with Language Specifier (CUDA or Python/Triton)
    pattern = r"```(?:C\+\+|cpp|cuda|c|python|triton)\s*\n(.*?)```"
    match = re.search(pattern, input, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # 3. Try Generic Markdown
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, input, re.DOTALL)

    if match:
        return match.group(1).strip()

    # 4. No format found, return as-is
    return input.strip()


def generate(gen_model: GenModel, msg: str, model: str) -> str:
    """
    Generate kernel code using the provided GenModel instance.
    
    Args:
        gen_model: Initialized GenModel with system prompt
        msg: User message / prompt
        model: Model name string (e.g. "gpt-4o", "gemini-1.5-pro", etc.)
        
    Returns:
        Cleaned up kernel code string
    """
    print(f"Generating code with {model}...")
    
    response = gen_model.chat(msg, model)
    
    code = cleanup_mkdown(response)
    print("Code generated...")
    return code

