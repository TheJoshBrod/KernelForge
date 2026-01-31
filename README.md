# CGinS: CUDA Ghost in the Shell

**CGinS (CUDA Ghost in the Shell)** is an autonomous, multi-agent system that enables model engineers to achieve highly optimized CUDA kernel performance without requiring deep hardware expertise. The system employs LLM-based agents to abstract away individual CUDA kernels by iteratively generate, validate, and optimize CUDA kernels from PyTorch operations through autonomous decision-making and self-improvement mechanisms.

**Key Results:**
- Up to 4x optimization speedups over baseline generation through autonomous iterative refinement
- Performance up to 33% faster than native PyTorch implementations on production workloads
- Automated correctness validation with precision threshold of 1e-5 per element
- Two-tiered feedback mechanism:
  - Generation-verification cycle with up to 3 self-debugging attempts per kernel
  - Performance profiling loop measuring real hardware metrics to guide optimization
- Self-improving agents that autonomously explore optimization strategies, learn from failures, and converge toward optimal solutions through iterative reasoning

## References

- [System Architecture](docs/system-architecture.md)
- [Motivation](docs/motivation.md)
- [Paper](docs/CGinS-Paper.pdf)

## Installation

### Prerequisites
- **Python** >= 3.12
- **PyTorch** (CUDA-enabled)
- **NVIDIA GPU** with CUDA support (NVCC >= 12.1 recommended)
- **LLM API Key** (Gemini, OpenAI, or Anthropic)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CGinS
   ```

2. **Backend Python dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Frontend Jac dependencies (npm)**
   The front end was written in Jac due to its one-language capabilities of using modern web tools (React, npm/bun, etc.) with Python compatibility.
   ```bash
   cd frontend
   jac install
   ```

### LLM Provider Configuration
Set the provider and model via environment variables or the UI settings gear.

- `LLM_PROVIDER=openai` (or `gemini`, `anthropic`)
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-5.2` (or your preferred OpenAI model)
- Optional: `OPENAI_USE_RESPONSES=1` (force Responses API). By default, gpt-5 models use Responses.
- Optional: `OPENAI_MAX_OUTPUT_TOKENS` or `OPENAI_MAX_TOKENS`

The frontend Settings gear writes `frontend/config.json`. The backend auto-loads that file and sets matching environment variables on startup. You can override the config location with `CGINS_CONFIG_PATH`.

Validate your provider/model/key from the UI:
- Settings > Test API Connection

Or via CLI:
```bash
python scripts/test_llm_connection.py --provider openai --model gpt-5.2 --apikey <key>
```

## Usage

To run the frontend make sure Jac is installed (setup steps 2-3) and then run:

```bash
jac start main.jac
```

Once the project has been bundled, open http://localhost:8000 to access the CGinS tool.

Routes are client-side; projects load at:
```
http://localhost:8000/project/<project_name>
```

### Profiling an uploaded project

After uploading `model.py` and `weights.pt` in the UI, profile your project with:

```bash
python benchmarks/profiler/profile_project.py --project <project_name>
```

This writes per-op input/output pairs under `projects/<project_name>/io/individual_ops`, which the UI reads to display real operator lists.

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

```bash
# 1) Profile
python benchmarks/profiler/profile_project.py --project <project_name>

# 2) Generate kernels (project-scoped)
python -m src.generator.main --io-dir projects/<project_name>/io/individual_ops --out-dir projects/<project_name>/kernels/generated

# 3) Optimize kernels
python -m src.optimizer.optimize_ops projects/<project_name>/io/individual_ops <project_name> --kernel-dir projects/<project_name>/kernels/generated/individual_op_kernels

# 4) Benchmark optimized kernels (requires CUDA)
python scripts/benchmark_project_ops.py --project <project_name>
```

### Sample model + weights

Use the included CGinS mini model for testing:
- Model code: `benchmarks/models/cgins_mini.py`
- Weights: `benchmarks/models/cgins_mini_weights.pt`

The model exposes `build_model()` and `make_example_input()` so profiling works out of the box.

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
CGinS/
├── benchmarks/         # Data and scripts for profiling models
├── frontend/           # Frontend tool to interact with tool
├── kernels/            # Output directory for generated CUDA code
├── docs/               # Documentation
├── src/
│   ├── generator/      # Pipeline for ensuring kernel correctness
│   └── optimizer/      # Pipeline for performance tuning
└── requirements.txt    # Python dependencies
```
