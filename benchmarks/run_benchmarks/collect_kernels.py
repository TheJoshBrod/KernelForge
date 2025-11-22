#!/usr/bin/env python3
import os
import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import hashlib
import re

# Root directory where generated kernels live
SOURCE_ROOT = Path("generated_kernels/PyTorchFunctions")

# Output directory for compiled modules
OUTPUT_ROOT = Path("benchmarks/run_benchmarks/compiled_kernels")

# NVCC optimization flags
EXTRA_CUDA_CFLAGS = ["-O3", "--use_fast_math", "-lineinfo"]
EXTRA_CFLAGS = ["-O3"]


def find_kernel_files(root: Path):
    """Yield all kernel.cu files in directory tree."""
    for path in root.rglob("kernel.cu"):
        yield path


def read_cuda_kernel(kernel_path: Path):
    """Read the CUDA kernel source code."""
    with open(kernel_path, 'r') as f:
        return f.read()


def extract_launch_signature(cuda_source: str):
    """Extract the launch function signature from CUDA source."""
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", cuda_source)
    if not match:
        raise ValueError("Could not find 'launch' function signature in generated code.")
    return match.group(1) + ";"


def compile_kernel(kernel_path: Path):
    """Compile a kernel using PyTorch's load_inline."""
    parent_name = kernel_path.parent.name

    # Create output directory for this kernel
    output_dir = OUTPUT_ROOT / parent_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CUDA source
    cuda_source = read_cuda_kernel(kernel_path)
    
    # Extract launch function signature
    try:
        cpp_source = extract_launch_signature(cuda_source)
    except ValueError as e:
        print(f"✗ Skipping {kernel_path}: {e}")
        return None

    # Create a unique module name based on parent directory
    module_hash = hashlib.md5(parent_name.encode()).hexdigest()[:8]
    module_name = f"kernel_{parent_name}_{module_hash}"

    print(f"\nCompiling: {kernel_path}")
    print(f"Module name: {module_name}")
    print(f"Build directory: {output_dir}")

    try:
        # Set environment for correct CUDA version
        import sys
        python_bin_dir = os.path.dirname(sys.executable)
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
        os.environ["PATH"] = f"{python_bin_dir}:/usr/local/cuda-12.1/bin:{os.environ['PATH']}"
        
        module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['launch'],
            build_directory=str(output_dir),
            extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
            extra_cflags=EXTRA_CFLAGS,
            verbose=True,
            with_cuda=True
        )
        print(f"✓ Success - Module compiled and loaded")
        return module
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return None


def main():
    # Ensure PyTorch with CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in PyTorch")
        return

    kernels = list(find_kernel_files(SOURCE_ROOT))

    if not kernels:
        print("No kernel.cu files found.")
        return

    print(f"Found {len(kernels)} kernel.cu files.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    compiled_modules = {}
    success_count = 0

    for kernel in kernels:
        module = compile_kernel(kernel)
        if module is not None:
            parent_name = kernel.parent.name
            compiled_modules[parent_name] = module
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Compilation complete: {success_count}/{len(kernels)} succeeded")
    print(f"{'='*60}")

    return compiled_modules


if __name__ == "__main__":
    main()