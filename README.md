# CGinS: CUDA Ghost in the Shell

**CGinS** (CUDA Ghost in the Shell) is an automated system for generating and validating CUDA kernels from PyTorch operations. It leverages Large Language Models (LLMs) to iteratively convert high-level PyTorch operations into optimized CUDA kernels, achieving comparable performance to standard PyTorch eager mode execution.

## Motivation

PyTorch's eager mode offers great flexibility but misses out on global graph-level optimizations available to compiled backends. Consequently, performance relies heavily on individual kernel efficiency. Creating these high-performance kernels manually is tedious and requires deep hardware expertise.

**CGinS** addresses this by using LLMs as a just-in-time compiler. Unlike static compilers, it uses **runtime profiling** to capture real-world execution contexts (tensor shapes, types, arguments) and prompts LLMs to generate hardware-aware kernels.

**Key Results:**
- **Within 20% of PyTorch Eager Mode** on standard workloads (ResNet-50, DistilBERT, Swin-Base, ConvNeXt-Base).
- **Iterative Refinement**: Uses a validation-guided feedback loop where compilation and runtime errors help the model self-correct, typically converging within 2-4 iterations.
- **Hardware Optimization**: A dedicated optimization pass targets specific GPU constraints (e.g., memory hierarchy, tiling) for maximum throughput.

## How It Works

The system operates in two main phases:

### 1. Generator (Correctness)
The **Generator** pipeline focuses on producing *correct* CUDA kernels.
- **Profiling**: Intercepts PyTorch operator calls to record input data and attributes.
- **LLM Generation**: Prompts an LLM (Claude, Gemini, OpenAI) to write a CUDA kernel for the specific operation.
- **Validation Loop**: Compiles and runs the kernel against the original PyTorch output. Errors are fed back to the LLM for correction.
- **Output**: A set of validated, functionally correct kernels.

### 2. Optimizer (Performance) (WIP)
The **Optimizer** pipeline refines the validated kernels for specific hardware.
- **Analysis**: Takes valid kernels and applies hardware-specific optimization strategies (e.g., loop unrolling, shared memory usage).
- **Tuning**: Can optimize for different metrics (inference latency, GPU utilization).
- **Benchmarking**: Verifies that the new kernel is not only correct but faster than the baseline.

## Installation

### Prerequisites
- **Python** ≥ 3.12
- **PyTorch** (CUDA-enabled)
- **NVIDIA GPU** with CUDA support (NVCC ≥ 12.1 recommended)
- **LLM API Key** (Gemini, OpenAI, or Anthropic)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd GinS
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Add your LLM provider's API key to your environment.
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   # OR
   export OPENAI_API_KEY="your-api-key"
   # OR
   export ANTHROPIC_API_KEY="your-api-key"
   ```

## Usage

### Generate Input/Output Ground Truth
To generate initial dataset


### Generating Kernels
To run the generation pipeline (profiling → generation → validation):
```bash
./generate.sh
```
This will:
1. Scan for individual all pt files in `benchmarks/profiler/individual_ops/` for input/outputs.
2. Generate kernels in `kernels/generated/`.
3. Validate them against PyTorch reference outputs.

### Optimizing Kernels
To run the optimization pipeline (takes generated kernels → optimizes for hardware):
```bash
./optimize.sh
```
This will:
1. Read valid kernels from `kernels/generated/`.
2. Apply hardware-specific optimizations.
3. Save results to `kernels/optimized/`.

## Project Structure

```
GinS/
├── benchmarks/         # Data and scripts for profiling models
├── kernels/            # Output directory for generated CUDA code
├── src/
│   ├── generator/      # Pipeline for ensuring kernel correctness
│   └── optimizer/      # Pipeline for performance tuning
├── generate.sh         # Entry point for generation
├── optimize.sh         # Entry point for optimization
└── requirements.txt    # Python dependencies
```