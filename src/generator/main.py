"""
Main pipeline for CUDA kernel generation and validation.

Walks through each operation in a benchmark to:
1. Monitor kernel and ATen calls
2. Generate CUDA kernel code
3. Validate correctness through iterative refinement
"""
import glob
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

import src.generator.generator as generator
import src.generator.monitor as monitor
import src.generator.prompts.prompts as prompts
import src.generator.verifier as verify

# Configuration
MAX_ATTEMPTS = 5
OUTPUT_BASE_DIR = Path("kernels/generated")


def validate_with_retries(output_dir: Path, entry_files: list[str], conversation_history: list) -> bool:
    """
    Attempt to validate and fix kernel code up to MAX_ATTEMPTS times.

    Args:
        output_dir: Directory to save kernel outputs
        entry_files: List of paths to entry_*.pt files containing inputs/outputs
        conversation_history: LLM conversation history

    Returns:
        bool: is the final kernel successful
    """

    # Iterative kernel folder
    kernel_dir = output_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Try n times to go through entire test suite
    for attempt in range(MAX_ATTEMPTS + 1):

        # Generate kernel
        try:
            cu_code = generator.anthropic_generator(conversation_history)
            conversation_history.append(
                {"role": "assistant", "content": cu_code})
        except Exception as e:
            print(f"Failed on attempt {attempt}\n{e}")
            return False

        tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")

        # Output newest version of kernel
        with open(output_dir / "kernel.cu", "w", encoding="utf-8") as f:
            f.write(cu_code)

        # For each generated kernel validate ALL input/output
        is_valid = True
        for i, entry_file in enumerate(tqdm(entry_files, desc="Input Tests")):

            # Validate current kernel with entry file
            log_file_loc = output_dir / "attempts" / f"log-{attempt}-{i}.txt"
            os.makedirs(log_file_loc.parent, exist_ok=True)
            call_success, exec_success, feedback = verify.validate_kernel(
                cu_code, entry_file, log_file_loc, tmpdir
            )

            print(feedback)

            # If failed on a testcase regenerate
            is_valid = is_valid and call_success and exec_success
            if not is_valid:
                # Save kernel
                with open(kernel_dir / f"kernel-{attempt}-{i}.cu", "w") as f:
                    f.write(cu_code)

                conversation_history.append(
                    {"role": "user", "content": feedback})
                break

        # Delete tmp directory before next generation
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        # If all testcases passed, escape
        if is_valid:
            print(f"SUCCESSFUL on {attempt + 1}")
            # Save kernel
            with open(kernel_dir / f"kernel-{attempt}-g.cu", "w") as f:
                f.write(cu_code)

            return True

    return False


def process_function(directory_name: str, entry_files: list[str], op_dir: Path):
    """
    Process all profiled calls for a given function.

    Args:
        directory_name: Name of the directory that is based on the PyTorch API function (e.g. "torch-nn-functional-relu" -> "torch.nn.functional.relu")
        entry_files: List of paths to entry_*.pt files
        op_dir: Output directory for this operation
    """

    # Load first call to set up context for profiling
    first_call = torch.load(
        entry_files[0], map_location='cpu', weights_only=False)
    first_args = first_call.get("args", [])
    first_kwargs = first_call.get("kwargs", {})

    # Extract function name out of directory name
    function_name = first_call.get("function_name")
    if not function_name:
        print(f"Skipping {directory_name}: no function_name stored")
        return False

    context = {
        "torch": torch,
        "F": torch.nn.functional,
        "args": first_args,
        "kwargs": first_kwargs,
    }
    print(function_name)
    exec_str = f"{function_name}(*args, **kwargs)"

    # Set up conversation history
    conversation_history = []

    # Profile operation
    try:
        op_details = monitor.profile_single_op(context, exec_str)
    except Exception as e:
        print(e)
        return False

    # Load all calls for prompt generation
    call_list = []
    for entry_file in entry_files:
        try:
            entry = torch.load(
                entry_file, map_location='cpu', weights_only=False)
            call_list.append(entry)
        except Exception as e:
            print(f"Error loading {entry_file}: {e}")
            continue

    if not call_list:
        print(f"Failed to load any entries for {function_name}")
        return False

    prompt = prompts.generate_full_llm_prompt(
        call_list, function_name, op_details)
    conversation_history.append({"role": "user", "content": prompt})

    call_list.clear()

    # Validate loop - pass entry files directly
    success = validate_with_retries(
        op_dir, entry_files, conversation_history
    )

    # Track performance
    if success:
        success_file = op_dir / "success"
        with open(success_file, "w") as f:
            f.write("passed")

    return success


def main():
    """Main entry point: load benchmarks and process each one."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <benchmark_dir>")
        sys.exit(1)

    # Loop over all function directories
    function_dirs = sorted(glob.glob(os.path.join(sys.argv[1], "*")))

    for func_dir in tqdm(function_dirs, desc="Processing functions"):
        if not os.path.isdir(func_dir):
            continue

        function_name = os.path.basename(func_dir).replace("_", ".")
        print(function_name)

        # Get all entry files
        entry_files = sorted(glob.glob(os.path.join(func_dir, "entry_*.pt")))

        if not entry_files:
            print(f"No entry files found for {function_name}, skipping...")
            continue

        op_dir = OUTPUT_BASE_DIR / "individual_op_kernels" / \
            function_name.replace(".", "_")
        op_dir.mkdir(parents=True, exist_ok=True)

        performance_file = op_dir / "success"
        if performance_file.exists():
            continue

        # Pass entry file paths directly
        process_function(function_name, entry_files, op_dir)


if __name__ == "__main__":
    main()
