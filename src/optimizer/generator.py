import re
from pathlib import Path

import ollama as ol
from openai import OpenAI
from anthropic import Anthropic
import google.genai as genai

import src.optimizer.verifier as verifier


def cleanup_mkdown(input: str) -> str:
    """Extract code from markdown code blocks using regex."""

    # Try to match code blocks with language specifiers (C++, cpp, cuda, c)
    pattern = r"```(?:C\+\+|cpp|cuda|c)\s*\n(.*?)```"
    match = re.search(pattern, input, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Try generic code block without language specifier
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, input, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # No markdown found, return as-is
    return input.strip()

# ************************
# Model Specific Functions
# ************************

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
    response = ol.chat(model=model, messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": msg}])
    
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
            "parts": [content]
        })

    return gemini_history

def gemini_generator(conversation_history: list, model: str = "gemini-2.5-flash", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR using Gemini.

    Returns:
        str: kernel_code
    """
    print("Generating code...")
    sys_prompt = src.generator.prompts.prompts.get_system_prompt()

    
    chat = genai.GenerativeModel(
        model_name=model,
        system_instruction=sys_prompt
    )
    
    gemini_history = convert_chatgpt_to_gemini(conversation_history)
    response = chat.generate_content(gemini_history)

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
    
    conversation_history.insert(0, sys_prompt)
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


# **********************
# Verification Functions
# **********************


def generate(gpu_specs: dict, op_dir: str, improvement_log: list, temp_dir: Path, io_dir: Path, model: str = "claude-opus-4-5-20251101"):
    """Generates and validates CUDA kernels 

    Args:
        gpu_specs (dict): Specs of specific GPU architecture
        op_dir (str): Directory of previously generated CUDA kernel  
        improvement_log (list): "Chat History" of why LLM thinks it made an improvement over past attempts
        temp_dir (Path): Path of directory to compile kernel into (used later by profiler)
        io_dir (Path): Path of file that contains all input/output pairs recorded of this op
        model (str, optional): LLM that will generate kernels. Defaults to "claude-opus-4-5-20251101".
    """

    error_msgs = []
    for _ in range(3):
        
        # Generate
        cu_code = generate() # TODO

        # Verify
        is_valid, error = verifier.validate_kernel(cu_code, io_dir, temp_dir)
        if is_valid:
            break
        error_msgs.append(error)

