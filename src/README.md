# CGinS Source (`src`)

This directory contains the core Python logic for the **Generator** (Correctness) and **Optimizer** (Performance) pipelines.

## Structure

```
src/
├── generator/          # Correctness Pipeline
│   ├── generator.py        # LLM interaction & kernel generation
│   ├── verifier.py         # JIT compilation & output validation
│   ├── main.py             # Entry point for generator pipeline
│   └── prompts/            # System prompts for LLMs
└── optimizer/          # Performance Pipeline
    ├── pipeline.py         # Main optimization loop & MCTS driver
    ├── core/               # MCTS & Tree logic
    │   ├── mcts.py
    │   └── types.py
    └── components/
        └── hardware/       # Profiling & Hardware specs
            └── profiler.py
```

## 1. Generator Pipeline
**Location:** `src/generator/`

Responsible for producing an initial set of *correct* CUDA kernels for a given operator. It uses an iterative LLM-driven loop to generate code, compile it, and verify its output against PyTorch ground truth.

**Usage:**
```bash
python3 -m src.generator.main <input_torch_dir>
```

## 2. Optimizer Pipeline
**Location:** `src/optimizer/`

Responsible for taking a correct kernel and optimizing it for specific hardware. It uses Monte Carlo Tree Search (MCTS) to explore optimization strategies (tiling, unrolling, etc.) and benchmarks them on the target GPU.

**Usage:**
```bash
python3 -m src.optimizer.pipeline <input_torch_dir> <optional_project_name>
```

## Key Components

-   **`verifier.py`**: Handles dynamic C++ extension compilation (`torch.utils.cpp_extension`) to validate generated CUDA code.
-   **`profiler.py`**: Benchmarks kernel execution time (`mean_time_ms`) to guide the optimization search.
-   **`mcts.py`**: Implements the search algorithm to navigate the space of possible kernel optimizations.
