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