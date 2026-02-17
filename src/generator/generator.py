"""
src/generator/generator.py
Uses LLM to generate CUDA kernels that is semi-model-agnostic.
"""
import re

from google import genai
import ollama as ol
from anthropic import Anthropic
from openai import OpenAI

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


def ollama_generator(msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR

    Args:
        msg (str): Context for LLM to generate Kernel/IR
        model (str, optional): Which Ollama model to use for LLM. Defaults to "llama3.2:latest".
        outputIR (str, optional): What is the desired output IR type. Defaults to "CUDA".

    Returns:
        str: kernel_code
    """
    print("Generating code...")
    sys_prompt = src.generator.prompts.prompts.get_system_prompt()
    response = ol.chat(model=model, messages=[
                       {"role": "system", "content": sys_prompt}, {"role": "user", "content": msg}])

    cu_code = response['message']['content']

    print("Code generated...")
    return cleanup_mkdown(cu_code)


def convert_chatgpt_to_gemini(chatgpt_history: list) -> list:
    gemini_history = []

    for msg in chatgpt_history:
        role = msg["role"]

        # Gemini uses "model" instead of "assistant"
        if role == "assistant":
            role = "model"

        # Gemini supports only "user" and "model" inside the messages list
        if role == "system":
            # Skip it here (it goes to system_instruction)
            continue

        content = msg["content"]
        gemini_history.append({
            "role": role,
            "parts": [{"text": content}]
        })

    return gemini_history


def gemini_generator(conversation_history: list, model: str = "gemini-2.5-flash", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR using Gemini.

    Returns:
        str: kernel_code
    """
    print("Generating code...")
    sys_prompt = src.generator.prompts.prompts.get_system_prompt()

    client = genai.Client()

    gemini_history = convert_chatgpt_to_gemini(conversation_history)
    response = client.models.generate_content(
        model=model,
        contents=gemini_history,
        config={
            "system_instruction": sys_prompt
        }
    )

    cu_code = cleanup_mkdown(response.text)

    print("Code generated...")
    return cu_code


def chatgpt_generator(conversation_history: list, model: str = "gpt-4o", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR using OpenAI.

    Returns:
        str: kernel_code
    """

    client = OpenAI()

    print("Generating code...")
    sys_prompt = src.generator.prompts.prompts.get_system_prompt()

    conversation_history.insert(0, {"role": "system", "content": sys_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history
    )

    cu_code = cleanup_mkdown(response.choices[0].message.content)

    print("Code generated...")
    return cu_code


def convert_chatgpt_to_anthropic(chatgpt_history: list) -> list:
    anthropic_history = []
    for msg in chatgpt_history:
        role = msg["role"]
        if role == "system":
            continue  # Handle separately
        if role == "assistant":
            role = "assistant"
        elif role == "user":
            role = "user"

        content = msg["content"]
        anthropic_history.append({
            "role": role,
            "content": content
        })
    return anthropic_history


def anthropic_generator(conversation_history: list,
                        model: str = "claude-opus-4-5-20251101") -> str:
    """Initial generation of kernel/IR using Anthropic Claude API."""
    print("Generating code with Claude...")

    from anthropic import Anthropic

    anthropic_history = convert_chatgpt_to_anthropic(conversation_history)

    client = Anthropic()

    # Build the request parameters
    params = {
        "model": model,
        "max_tokens": 4096,
        "system": src.generator.prompts.prompts.get_system_prompt(),
        "messages": anthropic_history
    }

    response = client.messages.create(**params)

    # Extract the generated content
    code = cleanup_mkdown(response.content[0].text)

    print("Code generated…")
    return code
