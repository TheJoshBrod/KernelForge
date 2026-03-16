# Kernel Forge Frontend

Built with Jaseci ecosystem using Jac-client as a frontend. The frontend is a multi-page app that manages project lifecycle, launches backend jobs, and displays optimization results in real time.

---

## Project structure

```
frontend/
├── main.jac                        # App entry point: routing, imports, ConfigProvider
├── context/
│   └── ConfigContext.cl.jac        # Global config/hardware state (React Context + useConfig hook)
├── app/
│   ├── home/                       # Landing page: project list, navigation
│   ├── create/                     # New project form
│   ├── settings/                   # API keys, GPU selection, SSH, about
│   └── project/
│       ├── ProjectLayout.cl.jac    # Parent layout for all project routes
│       ├── dashboard/              # Operator results table, job status, KPI charts
│       ├── operator_workbench/     # Code editor, kernel tree, logs, metrics inspector
│       │   └── workbench/
│       │       └── menus/          # GenerateModal, OptimizeModal, UploadModal
│       ├── shared/
│       │   └── MetricsAdapter.cl.jac  # Normalises backend metrics for display
│       └── export/                 # ANVIL / CAST export UI
├── components/
│   ├── ui/                         # Button, GlowButton, Modal, Icons, SearchableSelect, etc.
│   ├── editor/
│   │   └── ForgePythonEditor.cl.jac  # Monaco-based Python/CUDA editor
│   ├── backdrops/                  # Animated backgrounds (Ember, Matrix, GameOfLife, Wave)
│   └── settings/                  # SettingsMenu, ApiKeysSettings, ApiKeysTestResults
└── walkers/                        # Backend JAC services (see below)
```

---

## Routing

Defined in `main.jac`:

| Route | Component |
|---|---|
| `/` | Home |
| `/new-project` | NewProject |
| `/settings/*` | SettingsPage, ApiKeysPage, GpuPage, SshPage, AboutSettings |
| `/project/:projectName` | ProjectLayout |
| `/project/:projectName/` | Dashboard |
| `/project/:projectName/operator-workbench` | OperatorWorkbench (op selector) |
| `/project/:projectName/operator-workbench/:opName` | OperatorWorkbenchView |
| `/project/:projectName/export` | Export |

---

## Global state: ConfigContext

`context/ConfigContext.cl.jac` provides a `ConfigProvider` and `useConfig` hook available to all components.

Holds:
- `config`: API keys, SSH connections, GPU selection (mirrors `frontend/config.json` on disk)
- `hwInfo`: detected hardware (GPUs, CUDA/ROCm/MPS availability)
- `saveConfig(cfg)`: persists config changes via `SaveConfig` walker

Fetched on mount via `GetConfig` and `DetectHardware` walkers.

---

## Backend calls

All backend communication goes through JAC walkers using `__jacSpawn`:

```
resp = await __jacSpawn("WalkerName", "", {"key": val});
data = resp.reports[0] if resp and resp.reports else {};
```

Walkers are defined in `frontend/walkers/`:

| File | Responsibility | Key walkers |
|---|---|---|
| `globals.jac` | Global constants, config path, default schema | (none public) |
| `config.jac` | Read/write config.json | `GetConfig`, `SaveConfig` |
| `system.jac` | Hardware detection, SSH, API key testing | `DetectHardware`, `CheckSshConnection`, `TestApiKey`, `TestAllApiKeys`, `GetSystemInfo` |
| `llm.jac` | LLM model listing | `GetOpenLLMModels`, `TestLLMConnection` |
| `project_admin.jac` | Project CRUD, operator listing, benchmarks, export | `CreateProject`, `DeleteProject`, `GetProjects`, `GetProjectOps`, `GetProjectStatus`, `GetProjectBenchmarks`, `ExportProject`, `CloneProjectForGpu` |
| `kernel_job_runners.jac` | Job execution and control | `StartProfile`, `StartGenerate`, `StartOptimize`, `StartBenchmark`, `SetJobControl`, `ClearProjectQueue`, `GetJobLog` |
| `optimization_results.jac` | Results, MCTS data, metrics, kernel import/export | `GetDashboardCharts`, `GetMctsTree`, `GetProjectMctsSummary`, `GetOpDetails`, `GetProjectMetrics`, `ImportKernel`, `DownloadCast` |
| `catalog_db.jac` | SQLite project catalog persistence | (none public) |
| `job_supervisor.jac` | Job queue management, state persistence, hardware/LLM env setup | (none public) |

---

## Run locally

```bash
cd frontend
jac install
jac start main.jac
```

Open `http://localhost:8000`.
