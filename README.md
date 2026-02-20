# CGinS: CUDA Ghost in the Shell

CGinS is an operator-level kernel generation and optimization system for PyTorch workloads.
It profiles model operators, generates correct kernels, optimizes them, and benchmarks results against PyTorch baselines.

## What is current in this repo

- Frontend orchestration via Jac/React (`frontend/`)
- Backend orchestration via walkers (`frontend/walkers/project.jac`)
- Canonical CLI workflow entrypoint: `src/optimizer/workflow.py`
- Project artifacts under: `kernels/projects/<project_name>/...`

## Setup

### Prerequisites

- Python 3.12+
- CUDA-capable PyTorch install (or CPU/MPS fallback)
- `jac` installed for frontend runtime

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
jac install
```

## Run the UI

```bash
cd frontend
jac start main.jac
```

Open `http://localhost:8000`.

Project routes:

- `http://localhost:8000/project/<project_name>`
- `http://localhost:8000/project/<project_name>/operator-workbench`

## UI forge modes

- `Automatic`: runs generation for all discovered operators in the project.
- `Manual`: runs generation only for operators selected in the dashboard.

In both modes, Start Forge runs profile/baseline as needed, then generate/optimize/benchmark.
Speed comparison updates as operators complete.

## Canonical CLI workflow

Use `src.optimizer.workflow` as the entrypoint.

### 1) Profile

```bash
python -m src.optimizer.workflow profile --project <project_name>
```

Optional upload-path flags (used by UI project creation flow):

- `--weights-b64-path`
- `--validation-b64-path`
- `--validation-name-path`

### 2) Generate kernels

Generate all discovered operators:

```bash
python -m src.optimizer.workflow generate \
  --project <project_name> \
  --target-device cuda
```

Generate selected operators only:

```bash
python -m src.optimizer.workflow generate \
  --project <project_name> \
  --ops torch_nn_functional_conv2d,torch_nn_functional_relu \
  --target-device cuda
```

Generate + optimize + benchmark (common end-to-end run):

```bash
python -m src.optimizer.workflow generate \
  --project <project_name> \
  --ops torch_nn_functional_conv2d,torch_nn_functional_relu \
  --target-device cuda \
  --optimize \
  --benchmark \
  --iterations 5
```

### 3) Optimize existing kernels

```bash
python -m src.optimizer.workflow optimize \
  --project <project_name> \
  --ops torch_nn_functional_conv2d \
  --target-device cuda \
  --iterations 5 \
  --benchmark
```

### 4) Benchmark only

```bash
python -m src.optimizer.workflow benchmark --project <project_name>
```

## Runtime artifact layout

All project state lives under:

- `kernels/projects/<project_name>/state.json`
- `kernels/projects/<project_name>/io/individual_ops/`
- `kernels/projects/<project_name>/io/summary.json`
- `kernels/projects/<project_name>/kernels/generated/individual_op_kernels/`
- `kernels/projects/<project_name>/trees/`
- `kernels/projects/<project_name>/benchmarks/op_benchmarks.json`
- `kernels/projects/<project_name>/logs/`

## References

- `docs/system-architecture.md`
- `docs/profiling/api.md`
- `docs/profiling/architecture.md`
- `src/README.md`
- `frontend/README.md`
