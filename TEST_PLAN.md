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

## 11) Apple Silicon (llama.cpp) v1
Goal: validate the Apple Silicon workflow for llama.cpp wrapper optimization.

1. Toolchain bootstrap:
   - `python scripts/apple_silicon/bootstrap.py`
   - Expect `build/bin/llama-cli` exists under `.vendor/llama.cpp`.
2. Environment doctor:
   - `python scripts/apple_silicon/cgins_as.py doctor`
   - Expect `device.is_apple_silicon = true` and Metal support reported on M-series Macs.
3. Quick optimize:
   - Ensure LLM provider/model/API key are configured in Settings (or env vars).
   - `python scripts/apple_silicon/cgins_as.py optimize --profile both --quick --project <project_name>`
   - Expect report at `projects/<project_name>/benchmarks/apple_silicon_report.json`.
4. Kernel-focused optimize:
   - `python scripts/apple_silicon/cgins_as.py optimize-kernels --model <gguf> --profile both --budget 240 --stage full --kernel-mode iterative --strict-parity --attempt-log /tmp/as_kernel_attempts.jsonl`
   - Expect candidate cache + attempt log to include compile/correctness records and selected best candidate.
4. Pack export / disable:
   - `python scripts/apple_silicon/cgins_as.py export-pack --model <gguf> --out /tmp/test-pack.cginspack`
   - `python scripts/apple_silicon/cgins_as.py disable-pack --model <gguf>`
   - Expect exported pack file and successful disable response.
5. Build reusable pack from candidate:
   - `python scripts/apple_silicon/cgins_as.py build-pack --model <gguf> --from-candidate <candidate_dir_or_resources> --reuse-policy chip_family+os_minor --activate`
   - Expect manifest compatibility policy fields and active-pack update.
5. Frontend:
   - Open project page and use **Apple Silicon (llama.cpp)** panel.
   - Run Doctor, Quick/Full optimize, Export, Disable.
   - Expect status row `apple_silicon_optimize` and report deltas rendered.
6. PyTorch MPS bridge:
   - `python scripts/apple_silicon/cgins_as.py torch-optimize --project <project_name>`
   - Expect optimizer executes with `CGINS_TARGET_DEVICE=mps`.

7. Academic validation study:
   - Prepare matrix JSON with checksums:
     - `python scripts/apple_silicon/prepare_study_matrix.py --matrix benchmarks/studies/study_matrix.template.json --out benchmarks/studies/study_matrix.json`
   - `python scripts/apple_silicon/cgins_as.py validate-study --matrix benchmarks/studies/study_matrix.json --profiles chat,long --arms baseline,flash,oneshot_kernel,iterative_kernel --kernel-mode iterative --attempt-log /tmp/apple_silicon_attempts.jsonl --gate-mode full --abba-cycles 8 --warmup-blocks 2 --strict-parity --strict-power --decode-claim-threshold-pct 30 --out benchmarks/studies/<run_id>`
   - Expect artifacts:
     - `study_manifest.json`
     - `runs_raw.jsonl`
     - `attempts.jsonl`
     - `claim_decisions.json`
     - `hotspots.json`
     - `op_profiles.json`
     - `exclusions.csv`
     - `summary.json`
     - `methods_note.md`
     - `metrics_by_block.csv`
     - `paired_deltas.csv`
     - `ci_results.csv`
     - `pvalues_corrected.csv`
     - `plots/*.svg` and `plots/*.png`
   - `python scripts/apple_silicon/render_study_report.py --study-dir benchmarks/studies/<run_id>` regenerates plot bundle.

8. Dispatch-audit canary (authoritative backend evidence):
   - `python scripts/apple_silicon/cgins_as.py validate-study --matrix benchmarks/studies/study_matrix.json --profiles chat --arms baseline,oneshot_kernel --kernel-mode oneshot --kernel-total-budget 1 --gate-mode quick --abba-cycles 1 --warmup-blocks 0 --parity-stage numeric --out benchmarks/studies/<canary_run_id>`
   - Expect:
     - `<canary_run_id>/dispatch_audit/*.json` exists.
     - For kernel attempts with candidate resources expected: `dispatch_audit_status == ok` and `candidate_resources_used == true`.
     - `throughput_report.json` contains audit counters (`dispatch_audit_status_counts`, `candidate_resources_used_count`, `candidate_resources_used_rate`).
