# CUDA Backend (`src/optimizer/backends/cuda/`)

This directory contains the NVIDIA CUDA implementation of the `Backend` interface. It is responsible for all CUDA-specific operations, including profiling, verification, and code generation.

## File Breakdown

### `__init__.py`
**Class**: `CUDABackend`
- Implementing the `Backend` abstract base class.
- Serves as the main entry point for CUDA optimization.
- Delegates tasks to the specialized modules below (`profiler`, `verifier`, `prompts`).
- Handles both local and remote (SSH) execution contexts transparently.

### `loader.py` (Formerly `runtime_loader.py`)
**Purpose**: JIT Compilation & Loading
- Compiles CUDA C++ code into Python-callable modules using `torch.utils.cpp_extension`.
- Manages the CUDA environment (finding `nvcc`, setting `CUDA_HOME`).
- Provides `compile_code_string` for in-memory compilation and `load_kernel` for loading from disk.
- Used by both `verifier.py` (locally) and `remote_worker.py` (remotely).

### `profiler.py`
**Purpose**: Performance Measurement & Hardware Specs
- **Hardware Specs**: Fetches GPU details (Name, VRAM, Compute Capability) via `pynvml` and `pycuda`.
- **Profiling**: Runs kernels to measure execution time (mean, std, min, max).
- Handles warmup runs and synchronizing CUDA events for accurate timing.
- Supports both local profiling and remote profiling (via `ssh_client`).

### `verifier.py`
**Purpose**: Correctness Checking
- Validates generated kernels by compiling them and running them against test inputs.
- Compares kernel output with "Ground Truth" (PyTorch eager implementation) tensors.
- Uses a **persistent worker process** to isolate potential CUDA crashes (segfaults) from the main optimizer process.
- Returns pass/fail status and detailed error logs (precision mismatches, runtime errors).

### `prompts.py`
**Purpose**: LLM Prompt Engineering
- Contains the specific System Prompts and Instruction Templates for generating CUDA C++.
- `get_sys_prompt()`: Defines the "Expert CUDA Engineer" persona.
- `generate_gpu_optimization_prompt()`: Constructs the prompt with current code, errors, and profiling data to guide the LLM.

### `remote_worker.py`
**Purpose**: Remote Execution Script
- A standalone script uploaded to remote SSH servers.
- Acts as an agent to perform `verify` and `profile` tasks on the remote GPU.
- Imports `loader.py` (also uploaded) to compile kernels on the remote machine.
- Communicates back to `ssh_client` via standard I/O (stdin/stdout) using pickles.

## Usage Flow
1. **Pipeline** creates `CUDABackend`.
2. **Backend** calls `profiler` to get `GPUSpecs`.
3. **Backend** calls `prompts` to get instructions for LLM.
4. **LLM** generates code.
5. **Backend** calls `verifier` (which uses `loader`) to check correctness.
6. **Backend** calls `profiler` (which uses `loader`) to measure speed.
