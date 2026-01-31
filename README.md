# CGinS: CUDA Ghost in the Shell

**CGinS (CUDA Ghost in the Shell)** is an autonomous, multi-agent system that enables model engineers to achieve highly optimized CUDA kernel performance without requiring deep hardware expertise. The system employs LLM-based agents to abstract away individual CUDA kernels by iteratively generate, validate, and optimize CUDA kernels from PyTorch operations through autonomous decision-making and self-improvement mechanisms.

## Motivation

Modern AI infrastructure faces a critical challenge: achieving peak performance on specialized hardware requires extensive CUDA expertise that most model engineers lack. While PyTorch's eager mode provides flexibility, manually optimizing kernels for production workloads is both time-intensive and demands deep hardware knowledge.

CGinS pushes LLMs beyond information retrieval into autonomous technical decision-making. The system captures operator-level input-output pairs from your models during runtime profiling, then employs agents that reason about complex performance tradeoffs and autonomously converge toward optimal solutions, without human intervention.

The core innovation is a persistent learning architecture where agents maintain an "improvement log tree" that tracks which optimization strategies yield actual speedups. This enables the system to learn from prior attempts and progressively refine its approach without falling into a local minima.

**Key Results:**
- Up to 4x optimization speedups over baseline generation through autonomous iterative refinement
- Performance within 10% of native PyTorch implementations on production workloads
- Automated correctness validation with precision threshold of 1e-5 per element
- Two-tiered feedback mechanism:
   - Generation-verification cycle with up to 3 self-debugging attempts per kernel
   - Performance profiling loop measuring real hardware metrics to guide optimization
- Self-improving agents that autonomously explore optimization strategies, learn from failures, and converge toward optimal solutions through iterative reasoning

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
- **Logging**: The LLM provides context of what it tried differently and stores the performance of the new kernel to be passed along

## Installation

### Prerequisites
- **Python** ≥ 3.12
- **PyTorch** (CUDA-enabled)
- **NVIDIA GPU** with CUDA support (NVCC ≥ 12.1 recommended)
- **LLM API Key** (Gemini, OpenAI, or Anthropic)

### LLM Provider Configuration
Set the provider and model via environment variables or the UI settings gear.

- LLM_PROVIDER=openai (or gemini, nthropic)
- OPENAI_API_KEY=...
- OPENAI_MODEL=gpt-5.2 (or your preferred OpenAI model)
- Optional: OPENAI_USE_RESPONSES=1 (force Responses API). By default, gpt-5 models use Responses.
- Optional: OPENAI_MAX_OUTPUT_TOKENS or OPENAI_MAX_TOKENS

The frontend Settings gear writes rontend/config.json. The backend will
auto-load that file and set the matching environment variables on startup.

Validate your provider/model/key from the UI:
- Settings > Test API Connection

Or via CLI:
```bash
python scripts/test_llm_connection.py --provider openai --model gpt-5.2 --apikey <key>
```

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd GinS
   ```

2. **Backend Python dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Frontend Jac dependencies (npm)**
   The front end was written in Jac due to its "one-language" capabilites of using modern web tools (React, npm/bun, etc.) with Python compability 
   ```bash
   cd frontend
   jac install
   ```

## Usage

To run the frontend make sure jac is installed (setup steps 2-3) and then run

```bash
jac start main.jac
```

Once the project has been bundled, open [localhost:8000](localhost:8000) to access the CGinS tool.

Routes are client-side; projects load at:
http://localhost:8000/project/<project_name>

### Profiling an uploaded project

After uploading `model.py` and `weights.pt` in the UI, profile your project with:

```bash
python benchmarks/profiler/profile_project.py --project <project_name>
```

This writes per‑op input/output pairs under `projects/<project_name>/io/individual_ops`,
which the UI reads to display real operator lists.

#### Profiling filters (skipped ops)
By default, profiling skips ops that are nondeterministic/training-only or pure metadata:
- RNG / training-only: dropout*, rand*, bernoulli, multinomial, etc.
- View/meta: view, reshape, permute, transpose, squeeze, unsqueeze, expand, as_strided
- Shape queries: size, stride, numel
- Copy/cast/alloc: to/_to_copy/contiguous/clone/copy_/empty/zeros/ones/full/arange

Override per project in `projects/<project_name>/config.json`:
```json
{
  "profile": {
    "allow_ops": [],
    "skip_ops": [],
    "skip_prefixes": []
  }
}
```

### Full pipeline (CLI)

You can run the full pipeline without the UI:

`ash
# 1) Profile
python benchmarks/profiler/profile_project.py --project <project_name>

# 2) Generate kernels (project-scoped)
python -m src.generator.main --io-dir projects/<project_name>/io/individual_ops --out-dir projects/<project_name>/kernels/generated

# 3) Optimize kernels
python -m src.optimizer.optimize_ops projects/<project_name>/io/individual_ops <project_name> --kernel-dir projects/<project_name>/kernels/generated/individual_op_kernels

# 4) Benchmark optimized kernels (requires CUDA)
python scripts/benchmark_project_ops.py --project <project_name>
`

### Sample model + weights

Use the included CGinS mini model for testing:

- Model code: enchmarks/models/cgins_mini.py
- Weights: enchmarks/models/cgins_mini_weights.pt

The model exposes uild_model() and make_example_input() so profiling works out of the box.

#### Quick test project (no UI)
Create a ready-to-run project from the sample model:
```bash
python scripts/create_test_project.py --name pipeline_test
```

### Testing checklist
See `TEST_PLAN.md` for a thorough end-to-end test plan and failure mode checks.

### Automated smoke tests
Run a quick automated test suite (profile + optional LLM + optional benchmark):
```bash
python scripts/run_smoke_tests.py --project pipeline_test
```

With LLM generation:
```bash
python scripts/run_smoke_tests.py --project pipeline_test --with-llm --require-llm
```

With benchmarks (CUDA required):
```bash
python scripts/run_smoke_tests.py --project pipeline_test --with-benchmark
```


## Project Structure

```
GinS/
├── benchmarks/         # Data and scripts for profiling models
├── frontend/           # Frontend tool to interact with tool
├── kernels/            # Output directory for generated CUDA code
├── src/
│   ├── generator/      # Pipeline for ensuring kernel correctness
│   └── optimizer/      # Pipeline for performance tuning
└── requirements.txt    # Python dependencies
```
