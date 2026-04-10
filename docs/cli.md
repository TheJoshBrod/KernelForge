# CLI Reference

Use `src.optimizer.workflow` for headless or scripted runs.

## Commands

### 1. Profile

```bash
python -m src.optimizer.workflow profile --project <project_name>
```

Captures operator I/O tensors and baseline benchmark data.

### 2. Generate kernels

```bash
# All operators
python -m src.optimizer.workflow generate \
  --project <project_name> \
  --target-device cuda

# Selected operators only
python -m src.optimizer.workflow generate \
  --project <project_name> \
  --ops torch_nn_functional_conv2d,torch_nn_functional_linear \
  --target-device cuda
```

### 3. Optimize + benchmark (end-to-end)

```bash
python -m src.optimizer.workflow generate \
  --project <project_name> \
  --target-device cuda \
  --optimize \
  --benchmark \
  --iterations 5
```

### 4. Optimize existing kernels

```bash
python -m src.optimizer.workflow optimize \
  --project <project_name> \
  --ops torch_nn_functional_conv2d \
  --target-device cuda \
  --iterations 5 \
  --benchmark
```

### 5. Benchmark only

```bash
python -m src.optimizer.workflow benchmark --project <project_name>
```

Mode-aware benchmarking:

```bash
python -m src.optimizer.workflow benchmark \
  --project <project_name> \
  --mode deployment \
  --selection-policy safe
```

Benchmark mode meanings:

- `micro`: fast per-op benchmark used for search-oriented measurements
- `deployment`: integrated replay benchmark used for safe recommendation
- `stress`: reserved for robustness-focused coverage checks
- `e2e`: reserved for harness-backed end-to-end benchmarking

Selection policy meanings:

- `safe`: deployment-safe recommendations only
- `mixed`: allow faster unsafe candidates alongside safe ones
- `fastest`: prefer raw microbenchmark winners
- `custom_only`: exclude wrapper-backed kernels where metadata supports it

## Project artifact layout

```
kernels/projects/<project_name>/
├── state.json                          # job state (progress, pause/cancel)
├── io/
│   ├── summary.json
│   └── individual_ops/                 # captured tensor I/O per operator
├── kernels/
│   └── generated/individual_op_kernels/<op>/
├── trees/<op>/                         # MCTS nodes and kernel source per attempt
├── benchmarks/op_benchmarks.json      # schema v2: selection + micro + deployment + stress + e2e
└── logs/
```
