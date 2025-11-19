"""
Main pipeline for CUDA kernel generation and validation.

Walks through each operation in a benchmark to:
1. Monitor kernel and ATen calls
2. Generate CUDA kernel code
3. Validate correctness through iterative refinement
"""

import os
import sys
import glob
import shutil
import torch
import tempfile
from pathlib import Path
from tqdm import tqdm

import src.monitor
import src.generator
import src.verifier
import src.logger

# Configuration
MAX_ATTEMPTS = 5
OUTPUT_BASE_DIR = Path("generated_kernels")

def validate_with_retries(output_dir: Path, validation_size: int, conversation_history: list) -> bool:
    """
    Attempt to validate and fix kernel code up to MAX_ATTEMPTS times.
    
    Returns:
        bool: is the final kernel successful
    """

    # Try n times to go through entire test suite
    for attempt in range(MAX_ATTEMPTS + 1):
        
        # Generate kernel
        try:
            cu_code = src.generator.anthropic_generator(conversation_history)
            conversation_history.append({"role": "assistant", "content": cu_code})
        except Exception as e:
            print(f"Failed on attempt {attempt}\n{e}")
            return False, "", -1
        
        tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")

        # Output newest version of kernel
        with open(output_dir / f"kernel.cu", "w") as f:
            f.write(cu_code)

        # For each generated kernel validate ALL input/output
        is_valid = True
        for i in tqdm(range(validation_size), desc="Input Tests"):

            input_path = output_dir / f"input{i}.pt"      
            gold_path = output_dir / f"gold{i}.pt"

            # Validate current kernel
            log_file_loc = output_dir / "attempts" / f"log-{attempt}-{i}.txt"
            os.makedirs(log_file_loc.parent, exist_ok=True)
            call_success, exec_success, feedback = src.verifier.validate_kernel(
                cu_code, input_path, gold_path, log_file_loc, tmpdir
            )
                        
            # If failed on a testcase regenerate
            is_valid = is_valid and call_success and exec_success
            if not is_valid:        
                # Save kernel
                with open(output_dir / f"kernel-{attempt}-{i}.cu", "w") as f:
                    f.write(cu_code)
                
                conversation_history.append({"role": "user", "content": feedback})
                break
        
        # Delete tmp directory before next generation 
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        # If all testcases passed, escape
        if is_valid:
            print(f"SUCCESSFUL on {attempt + 1}")
            # Save kernel
            with open(output_dir / f"kernel-{attempt}-g.cu", "w") as f:
                f.write(cu_code)
                
            return True
        
    return False


def process_function(function_name: str, call_list: list[dict], index: int, op_dir: Path):
    """
    Process all profiled calls for a given function.

    Args:
        function_name: Name of the PyTorch API function (e.g. "torch.nn.functional.relu")
        call_list: List of recorded calls, each containing 'args', 'kwargs', and 'output'
        index: Index of this function within the benchmark file
    """

    
    first_call = call_list[0]

    first_args = first_call.get("args", [])
    first_kwargs = first_call.get("kwargs", {})

    context = {
        "torch": __import__("torch"),
        "args": first_args,
        "kwargs": first_kwargs,
    }

    exec_str = f"{function_name}(*args, **kwargs)"

    
    # Set up conversation history
    conversation_history = []
    

    # Profile operation
    try:
        op_details = src.monitor.profile_single_op(context, exec_str)
    except Exception as e:
        print(e)
        return False


    # Define validation set 
    all_args   = [call.get("args", []) for call in call_list]
    all_kwargs = [call.get("kwargs", {}) for call in call_list]
    all_output = [call.get("output", None) for call in call_list]
    all_iterations = [all_args, all_kwargs, all_output]

    prompt = op_details
    conversation_history.append({"role": "user", "content": prompt})
    
    # Save input/output for verification
    for i, _ in enumerate(all_iterations[0]):

        input_path = op_dir / f"input{i}.pt"
        gold_path = op_dir / f"gold{i}.pt"

        args = all_iterations[0][i]
        kwargs = all_iterations[1][i]
        
        inputs = {"args": args, "kwargs": kwargs}
        ground_truth = all_iterations[2][i]

        torch.save(inputs, input_path)
        torch.save(ground_truth, gold_path)

    # Validate loop
    success = validate_with_retries(
        op_dir, len(all_args), conversation_history
    )

    # Track performance
    if success:
        src.logger.compare_kernel_to_pytorch(op_dir, function_name)


    # Erase pt files
    for i, _ in enumerate(all_iterations[0]):
        input_path = op_dir / f"input{i}.pt"
        gold_path = op_dir / f"gold{i}.pt"

        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(gold_path):
            os.remove(gold_path)


def main():
    """Main entry point: load benchmarks and process each one."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <benchmark_file.pt>")
        sys.exit(1)
    
    # Loop over all operations in the benchmark
    for i, file_name in enumerate(tqdm(glob.glob(sys.argv[1] + "/*.pt"), desc="Processing functions")):

        file = Path(file_name)
        function_name = file.stem
        call_list = torch.load(file)[function_name]

        op_dir = OUTPUT_BASE_DIR / "PyTorchFunctions" / function_name.replace(".", "_")
        op_dir.mkdir(parents=True, exist_ok=True)

        performance_file = op_dir / "performance.txt" 
        if performance_file.exists():
            continue

        process_function(function_name, call_list, i, op_dir)
    

if __name__ == "__main__":
    main()
