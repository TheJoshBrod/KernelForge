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
├── benchmarks/op_benchmarks.json
└── logs/
```
