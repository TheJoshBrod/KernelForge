# CGinS Test Plan (Manual + CLI)

This plan is designed to quickly surface what is broken across the full CGinS pipeline. It is organized from fastest/lowest-dependency checks to full end-to-end validation. Use the sample model/weights for reproducibility, then repeat with a real model.

## 0) Preconditions / Environment
- Python >= 3.12, CUDA-capable GPU, NVCC available.
- `pip install -r requirements.txt`
- `jac` installed and frontend deps installed (`cd frontend && jac install`).
- For LLM-backed generation: valid API key and provider/model set in Settings or env vars.

## 1) Smoke Tests (No API Key Required)
Goal: ensure UI, routing, and backend walkers work without touching LLMs.

1. Start UI
   - `cd frontend`
   - `jac format . --fix`
   - `jac check .`
   - `jac start main.jac`
   - Open `http://localhost:8000/`
   - Expect: Home page loads, Settings page loads, routes `/project/<name>` work.

2. Create a project (UI)
   - Create a new project with any name.
   - Upload `benchmarks/models/cgins_mini.py` and `benchmarks/models/cgins_mini_weights.pt`.
   - Expect: project appears in Home list and navigates to project page.

3. Backend walkers
   - From UI:
     - `GetProjects` populates list.
     - `GetConfig` works without error.
   - Expect: no 500s in server log; no red error toast.

## 2) LLM Connectivity (API Key Required)
Goal: verify the provider/model/key are valid and reachable.

### UI
- Settings → “Test API Connection”
- Expect: Success message with latency and “ok” response.
- Failure should return a clear error (missing key, invalid model, 401, etc.).

### CLI (optional)
- `python scripts/test_llm_connection.py --provider openai --model gpt-5.2 --apikey <key>`
- Expect JSON: `{"success": true, "latency_ms": ..., "response": "ok"}`.

## 3) Profiling (No API Key Required)
Goal: verify per-op artifacts are created and skip filters apply.

1. CLI profiling
   - `python benchmarks/profiler/profile_project.py --project <project_name>`
   - Expect:
     - `projects/<project>/io/individual_ops/<op>/entry_*.pt` created.
     - `projects/<project>/io/summary.json` created.
     - Dropout/RNG/meta ops are *skipped* by default (not written).

2. UI operators list
   - Project page → “Operators” list.
   - Expect: op counts match `summary.json` and skip list.

## 4) Generation (API Key Required)
Goal: validate kernel generation, retries, and failure reporting.

1. Start generation from UI (or CLI):
   - UI: “Generate Kernels”
   - CLI: `python -m src.generator.main --io-dir projects/<project>/io/individual_ops --out-dir projects/<project>/kernels/generated`
   - Expect:
     - For each op: `projects/<project>/kernels/generated/individual_op_kernels/<op>/kernel.cu` on success.
     - `attempts/failure.json` if generation fails.
     - `state.json` progress updates (percent + message) during run.

2. Pause/Resume/Cancel
   - Pause generation in UI; confirm process stops advancing (progress timestamps stop).
   - Resume and verify progress continues.
   - Cancel and confirm job status becomes `cancelled` and exits.

## 5) Optimize (API Key Required)
Goal: ensure optimized kernels are created and tracked.

- UI: “Optimize Kernels”
- Expect:
  - output in `projects/<project>/kernels/optimized/<op>/`
  - state updates in `projects/<project>/state.json`

## 6) Benchmark (CUDA Required)
Goal: compare PyTorch vs kernel performance and emit per-op results.

- UI: “Benchmark”
- CLI: `python scripts/benchmark_project_ops.py --project <project_name>`
- Expect:
  - `projects/<project>/benchmarks/op_benchmarks.json`
  - UI shows PyTorch and kernel timings per op.
  - For ops without kernels, kernel fields show `missing` status (not 0 ms).

## 7) Export (No API Key Required)
Goal: ensure .cgins export contains only selected kernels.

- UI: select ops → “Export .cgins”
- Expect:
  - `projects/<project>/exports/<project>.cgins.zip`
  - ZIP contains `manifest.json` and selected kernel files.

## 8) Failure Mode Tests (Intentional Breakage)
Goal: verify errors are surfaced clearly (no silent failures).

1. Missing/invalid API key
   - Clear API key, run generation.
   - Expect: `failure.json` with `stage: llm_api`.

2. Bad weights or wrong model
   - Upload mismatched weights.
   - Expect: profiling fails with clear error and UI warning.

3. CUDA compiler failure
   - Inject invalid kernel code into `kernel.cu`.
   - Benchmark should show kernel failure with error in logs, not crash UI.

## 9) Regression Tests (Recommended for every change)
Run these after each set of changes:
- `jac format . --fix` and `jac check .`
- Full UI load and routing check.
- CLI profiling + generation + benchmark on sample model.
- Verify `failure.json` is produced on forced LLM failure.

## 10) What “100% working” means (Definition of Done)
- Profile produces per-op entries with skip filters applied.
- Generate produces kernels or `failure.json` for every op.
- Optimize produces at least one optimized candidate per op (if enabled).
- Benchmark shows both PyTorch and kernel results (or clear missing/error status).
- Export produces a valid `.cgins.zip` with selected ops.
- UI shows accurate status + progress for profile/generate/optimize/benchmark.
- Settings “Test API Connection” returns success with valid credentials.
