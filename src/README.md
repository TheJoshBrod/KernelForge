# Kernel Forge Source (`src`)

This directory contains the backend generation, optimization, and benchmarking logic.

## Structure

```text
src/
├── generator/                  # Kernel generation + correctness validation
│   ├── main.py                 # Generator entrypoint (operator-level)
│   ├── generator.py            # LLM interaction and code extraction
│   └── prompts/                # Prompt templates/system prompts
├── optimizer/
│   ├── workflow.py             # Canonical orchestration CLI (profile/generate/optimize/benchmark)
│   ├── pipeline.py             # Optimization pipeline (MCTS-driven)
│   ├── benchmarking/           # Baseline + optimized benchmarking pipeline
│   ├── core/                   # MCTS core types/logic
│   └── backends/               # CUDA and Triton backend abstractions (Metal is a skeleton)
└── progress.py                 # Job progress + pause/cancel helpers via state.json
```

## Canonical entrypoints

### Orchestration (preferred)

```bash
python -m src.optimizer.workflow <action> [flags]
```

Actions:

- `profile`
- `generate`
- `optimize`
- `benchmark`

### Generator direct

`src.generator.main` is called by workflow for generation.

### Optimizer direct

`src.optimizer.pipeline` is called by workflow for optimization.

## Current execution model

- `profile`: captures operator data and baseline benchmark artifacts.
- `generate`: runs per-operator generation, verifies success markers, then optionally optimize+benchmark.
- `optimize`: runs optimization by operator, optionally benchmark.
- `benchmark`: rebuilds `op_benchmarks.json` from current project outputs.

## Project artifact location

`kernels/projects/<project_name>/...`

Key files:

- `state.json`
- `io/summary.json`
- `kernels/generated/individual_op_kernels/<op>/`
- `trees/<op>/`
- `benchmarks/op_benchmarks.json`
