# CGinS Frontend

Frontend is built with Jac + React and orchestrates project lifecycle through walkers.

## Main capabilities

- Create/load projects
- Start Forge from dashboard
- `Automatic` mode: run all discovered operators
- `Manual` mode: select specific operators to run
- Live dashboard updates:
  - pipeline/job state
  - speed comparison (baseline vs optimized)
  - operator time consumption
  - MCTS nodes explored
- Operator Workbench for tree inspection and per-operator actions

## Run locally

```bash
cd frontend
jac install
jac start main.jac
```

Open `http://localhost:8000`.

## Backend integration

- Walkers are defined in `frontend/walkers/project.jac`
- Key calls used by dashboard/workbench:
  - `StartProfile`
  - `StartGenerate`
  - `StartOptimize`
  - `GetProjectStatus`
  - `GetDashboardCharts`
  - `GetProjectOps`
  - `GetProjectMctsSummary`
  - `GetMctsTree`

## Notes

- Generation uses standard `generate` job state (no active advanced-generate pathway).
- Dashboard polling is designed to keep charts and MCTS summary updated while runs are active.
