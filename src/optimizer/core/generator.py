"""
src/optimizer/components/llm/generator.py
Uses LLM to generate CUDA kernels that is model-agnostic.
"""
import re
from pathlib import Path
from typing import Optional
from typing import Tuple

from src.llm_tools import GenModel
from src.config import ensure_llm_config
from src.optimizer.core.types import GPUSpecs
from src.optimizer.config.settings import settings
from src.optimizer.core.backend import Backend
import src.optimizer.core.mcts as mcts


def _log(msg: str):
    print(f"[Generator] {msg}")


def _format_reason(reason: str, max_len: int = 240) -> str:
    text = str(reason or "").replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return ""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _current_op_name(paths: dict[Path]) -> str:
    proj_dir = paths["proj_dir"] if "proj_dir" in paths else None
    if proj_dir is None:
        return "unknown"
    try:
        return str(proj_dir.name)
    except Exception:
        return "unknown"


def _log_attempt_result(paths: dict[Path], status: str, reason: str = "") -> None:
    op_name = _current_op_name(paths)
    iteration = paths["iteration"] if "iteration" in paths else "unknown"
    attempt = paths["attempt"] if "attempt" in paths else "unknown"
    line = (
        f"\t\t[optimize-attempt] op={op_name} iteration={iteration} "
        f"attempt={attempt} status={status}"
    )
    short_reason = _format_reason(reason)
    if short_reason:
        line += f" reason={short_reason}"
    print(line)


def _dump_failed_llm_response(paths: dict[Path], response: str, tag: str) -> None:
    proj_dir = paths["proj_dir"] if "proj_dir" in paths else None
    if not proj_dir:
        return
    try:
        dump_dir = proj_dir / "garbage_dump"
        dump_dir.mkdir(parents=True, exist_ok=True)
        iteration = paths["iteration"] if "iteration" in paths else "unknown"
        attempt = paths["attempt"] if "attempt" in paths else "unknown"
        dump_path = dump_dir / f"llm_response_iter{iteration}_attempt{attempt}_{tag}.txt"
        dump_path.write_text(str(response or ""), encoding="utf-8")
        print(f"\t\t- Saved failed LLM response to: {dump_path}")
    except Exception as e:
        print(f"\t\t- Failed to save LLM response dump: {e}")


def extract_feedback_and_code(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract feedback and code sections from a formatted string.

    Args:
        content: The input string containing feedback and code sections

    Returns:
        A tuple of (feedback, code) where each is None if not found
    """

    # Extract feedback section (tolerant to spacing)
    feedback_pattern = r'(?://|#)\s*\[START FEEDBACK\](.*?)(?://|#)\s*\[END FEEDBACK\]'
    feedback_match = re.search(
        feedback_pattern, content, re.DOTALL | re.IGNORECASE)
    feedback = feedback_match.group(1).strip(
    ) if feedback_match else "No feedback provided"

    # Extract code section (tolerant to spacing)
    # 1. Try strict tags
    code_pattern = r'(?://|#)\s*\[START kernel\.(?:cu|py)\](.*?)(?://|#)\s*\[END kernel\.(?:cu|py)\]'
    code_match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)

    if code_match:
        code = code_match.group(1).strip()
    else:
        # 2. Markdown fallback
        # This handles ```cpp\n ... ``` or ```cuda\n ... ```
        fallback_pattern = r"```(?:C\+\+|cpp|cuda|c|python|triton)?\s*\n(.*?)```"
        fallback_match = re.search(
            fallback_pattern, content, re.DOTALL | re.IGNORECASE)
        code = fallback_match.group(1).strip() if fallback_match else None

    return feedback, code




def create_and_validate(backend: Backend, llm: GenModel, msg: str, model: str, paths: dict[Path], ssh_config: dict = None) -> Tuple[str, bool, str]:
    """Generates a new kernel then validates it for correctness

    Args:
        backend (Backend): Backend instance specific to the GPU architecture for validation
        llm (GenModel): LLM abstraction class with chat history
        msg (str): User message for LLM
        model (str): Name of LLM model
        paths (dict[Path]): Data structure holding different filepaths
        ssh_config (dict, optional): SSH configuration for remote validation.

    Returns:
        Tuple[str, bool, str]: _description_
    """
    response = llm.chat(msg, model)
    feedback, cu_code = extract_feedback_and_code(response)

    if cu_code is None:
        reason = "Failed to extract code from LLM response"
        _dump_failed_llm_response(paths, response, "extract_failed")
        _log_attempt_result(paths, "failed", reason)
        return feedback, False, reason
    
    # Serialize validation (compilation + execution) if GPU lock is available
    gpu_lock = paths.get("gpu_lock")
    if gpu_lock:
        with gpu_lock:
            _log("GPU lock acquired, running validation...")
            is_valid, error = backend.validate_kernel(cu_code, paths, ssh_config)
    else:
        is_valid, error = backend.validate_kernel(cu_code, paths, ssh_config)
    
    _log(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")
    if not is_valid:
        _log_attempt_result(paths, "failed", str(error))
        # Save to garbage dump
        proj_dir = paths.get("proj_dir")
        if proj_dir:
            dump_dir = proj_dir / "garbage_dump"
            dump_dir.mkdir(parents=True, exist_ok=True)

            iteration = paths.get("iteration", "unknown")
            attempt = paths.get("attempt", "unknown")

            ext = ".py" if "@triton.jit" in cu_code else ".cu"
            filename = f"kernel_iter{iteration}_attempt{attempt}{ext}"
            dump_path = dump_dir / filename

            try:
                dump_path.write_text(cu_code)
                print(f"\t\t- Saved failed kernel to: {dump_path}")
            except Exception as e:
                print(f"\t\t- Failed to save garbage kernel: {e}")

    _log_attempt_result(paths, "success", "")
    return feedback, is_valid, error


def generate(backend: Backend, best_kernel_code: str, gpu_specs: GPUSpecs, improvement_log: list, paths: dict[str, Path], model: str = None, ancestor_codes: list[tuple[int, str]] = None, ssh_config: dict = None) -> Tuple[str, bool, str]:
    """Generates and validates CUDA kernels 

    Args:
        backend (Backend): Backend instance specific to the GPU architecture
        gpu_specs (GPUSpecs): Specs of specific GPU architecture
        best_kernel_code (str): Current best kernel code to optimize
        improvement_log (list): History of optimization attempts from tree ancestors
        paths (dict[str, Path]): Paths to directories
        model (str, optional): LLM model name. Defaults to settings value.
        ancestor_codes (list[tuple[int, str]], optional): List of (iteration_id, code) tuples from ancestors
        ssh_config (dict, optional): SSH configuration for remote validation.
    """
    if not model:
        ensure_llm_config()
        model = model or settings.llm_model_name

    # Attempt initial CUDA code generation
    sys_prompt = backend.get_sys_prompt()
    llm: GenModel = GenModel(sys_prompt)
    msg = backend.generate_optimization_prompt(
        gpu_specs, best_kernel_code, improvement_log, ancestor_codes)

    # DEBUG: Save full prompt alongside each generation
    # Use shared counter if available (parallel mode), else count files (sequential mode)
    if "node_counter" in paths:
        # Parallel mode: use atomic shared counter
        with paths["node_counter"].get_lock():
            next_node_id = paths["node_counter"].value
            paths["node_counter"].value += 1
    else:
        # Sequential mode: count existing nodes
        next_node_id = mcts.get_next_node_id(paths)
    
    prompt_dump_path = paths["proj_dir"] / "kernels" / f"prompt_{next_node_id}.md"
    prompt_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_dump_path, "w") as f:
        f.write("# System Prompt\n\n")
        f.write(sys_prompt)
        f.write("\n\n---\n\n# User Message\n\n")
        f.write(msg)
    print(f"\t\tSaved prompt to: {prompt_dump_path}")

    paths["attempt"] = 0
    feedback, is_valid, error = create_and_validate(backend, llm, msg, model, paths, ssh_config)
    if is_valid:
        return feedback, True, ""
    print("\t\tInitial gen failed...")
    last_error = str(error) if error else ""
    # On failure attempt fix before giving up
    for i in range(settings.retry_limit):
        print(f"\t\t\tReattempt {i}")
        paths["attempt"] = i + 1
        retry_feedback, is_valid, error = create_and_validate(backend, llm, error, model, paths, ssh_config)
        if is_valid:
            return retry_feedback, True, ""
        if error:
            last_error = str(error)

    if not last_error:
        last_error = "No valid kernel produced after retries."
    return "", False, last_error
