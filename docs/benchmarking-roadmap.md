# Benchmarking Roadmap For Kernel Forge

This document turns the benchmarking investigation into a concrete
implementation roadmap for Kernel Forge itself. It covers:

- schema changes
- CLI and API design
- default-selection policy
- UI and GUI flows
- the effect on kernel benchmarking and MCTS
- a commit-by-commit rollout plan

This roadmap assumes work starts from the current `demobranch`.

## 1. Problem Statement

Kernel Forge currently mixes three different objectives:

1. `search objective`
   Fast per-op benchmarking used to guide kernel generation and optimization.
2. `deployment objective`
   Strict correctness and integration checks used to decide whether a kernel is
   safe to recommend or export.
3. `reporting objective`
   Project-level and end-to-end metrics used in the UI, docs, and external
   claims.

Today, the product mostly treats the raw microbenchmark result as if it also
answered the deployment and reporting questions. That creates four classes of
errors:

- unsafe kernels can become default recommendations
- project speedup estimates can overstate real end-to-end gains
- benchmark refresh can overwrite tree values used by search
- open-source users do not get a clear distinction between "fast in isolation"
  and "safe to deploy"

## 2. Design Goals

The target design should satisfy all of the following:

1. Keep MCTS fast and local.
2. Make deployment-safe selection the default.
3. Preserve raw microbenchmark data for kernel authors.
4. Make end-to-end reporting explicit and opt-in.
5. Avoid rewriting search state with reporting data.
6. Keep existing projects readable through a migration layer.
7. Make the UI honest by default and explicit about what each metric means.

## 3. Benchmark Model

Kernel Forge should expose four benchmark modes.

### 3.1 `micro`

Purpose:
- drive search
- compare isolated kernels quickly

Characteristics:
- per-op only
- fast
- replay-based
- tolerant of partial coverage

Primary outputs:
- `micro.kernel_ms`
- `micro.pytorch_ms`
- `micro.speedup`

### 3.2 `deployment`

Purpose:
- decide whether a kernel is safe to recommend, export, or auto-select

Characteristics:
- per-op integrated replay
- strict correctness
- adapter and layout stats
- dtype and shape coverage summary

Primary outputs:
- `deployment.replay_ms`
- `deployment.replay_speedup`
- `deployment.correctness`
- `deployment.coverage`
- `deployment.safe`
- `deployment.recommended_backend`

### 3.3 `stress`

Purpose:
- surface robustness problems before deployment

Characteristics:
- targeted hard cases
- dynamic shapes
- non-contiguous inputs
- BF16 / FP16 / FP32
- optional fuzzing / perturbation buckets

Primary outputs:
- `stress.pass_rate`
- `stress.failures`
- `stress.coverage_gaps`

### 3.4 `e2e`

Purpose:
- measure actual model-level behavior and public-facing performance

Characteristics:
- harness-backed
- prompt or workload suite driven
- exact output comparison
- prefill / decode / total breakdown

Primary outputs:
- `e2e.prefill_ms`
- `e2e.decode_ms`
- `e2e.total_ms`
- `e2e.decode_tokens_per_s`
- `e2e.total_tokens_per_s`
- `e2e.exact_match`
- `e2e.runtime_fallbacks`

## 4. Core Policy

Kernel Forge should stop using one benchmark answer for all product decisions.

The policy should be:

- `micro` decides search ranking
- `deployment` decides default recommendation and export eligibility
- `stress` decides whether a deployment-safe kernel is robust enough for a
  higher-confidence badge
- `e2e` decides public reporting and model-level claims

This separation is the most important change in the roadmap.

## 5. Schema Changes

`op_benchmarks.json` should move from a flat mixed schema to an explicit
multi-mode schema. The old keys should remain readable for one transition
window, but new writes should use the new structure.

### 5.1 Current Problems In The Schema

- `winner` is ambiguous
- `kernel_ms` is used for both microbenchmarking and project-level reporting
- deployment fields exist but are second-class
- there is no first-class place for stress or e2e evidence
- there is no stable field for "recommended default"

### 5.2 Proposed `op_benchmarks.json` Shape

```json
{
  "schema_version": 2,
  "project": "example",
  "device": {
    "name": "NVIDIA GB10",
    "cuda": "12.4",
    "torch": "2.8.0"
  },
  "benchmarks": [
    {
      "op_name": "torch_nn_functional_linear",
      "selection": {
        "recommended_mode": "deployment",
        "recommended_backend": "optimized",
        "recommended_reason": "strict-pass + speedup + sufficient coverage",
        "export_allowed": true,
        "unsafe_override_required": false
      },
      "micro": {
        "status": "ready",
        "sample_count": 50,
        "pytorch_ms": 0.051,
        "kernel_ms": 0.032,
        "speedup": 1.59,
        "winner": "optimized"
      },
      "deployment": {
        "status": "ready",
        "sample_count": 50,
        "replay_pytorch_ms": 0.053,
        "replay_kernel_ms": 0.035,
        "replay_speedup": 1.51,
        "winner": "optimized",
        "correctness": {
          "strict_pass": true,
          "strict_mismatches": 0,
          "max_abs_diff": 0.0,
          "max_rel_diff": 0.0
        },
        "coverage": {
          "dtype_counts": {
            "float16": 10,
            "bfloat16": 40
          },
          "layout_counts": {
            "contiguous": 48,
            "noncontiguous": 2
          },
          "shape_buckets": 7,
          "adapter_stats": {}
        },
        "fallback": {
          "supported": true,
          "observed": 0,
          "reasons": []
        },
        "safe": true
      },
      "stress": {
        "status": "partial",
        "pass_rate": 0.98,
        "failures": [
          {
            "category": "noncontiguous_input",
            "count": 2
          }
        ]
      },
      "e2e": {
        "status": "not_run"
      },
      "legacy": {
        "winner": "optimized",
        "kernel_ms": 0.032,
        "deployment_safe_winner": "optimized"
      }
    }
  ]
}
```

### 5.3 New Required Fields

Every op benchmark row should expose:

- `selection.recommended_backend`
- `selection.export_allowed`
- `micro.*`
- `deployment.safe`
- `deployment.coverage`
- `deployment.fallback`
- `stress.status`
- `e2e.status`

### 5.4 Backward Compatibility

For one release window:

- continue reading flat legacy keys if `schema_version` is absent
- write both the new nested keys and a minimal `legacy` object
- migrate frontend readers to prefer the new nested fields first

### 5.5 Tree Metadata

Tree data should stop overloading a single `value` field with mixed meanings.

Add the following fields to root metadata or a sidecar benchmark file:

- `search_value_ms`
- `search_value_source`
- `latest_micro_kernel_ms`
- `latest_deployment_replay_ms`
- `latest_deployment_safe`
- `latest_stress_pass_rate`

Do not let deployment or e2e benchmarking overwrite `search_value_ms`.

## 6. CLI Design

The CLI should make benchmark intent explicit.

### 6.1 New Top-Level Flags

Extend:

```bash
python -m src.optimizer.workflow benchmark
```

with:

```bash
--mode micro|deployment|stress|e2e
--selection-policy safe|fastest|custom-only|mixed
--sample-strategy first|random|stratified
--sample-count N
--stress-profile default|layout|dtype|dynamic
--harness <name>
--prompt-suite <path>
--allow-unsafe-export
```

### 6.2 Recommended CLI Behavior

Examples:

```bash
python -m src.optimizer.workflow benchmark \
  --project my_project \
  --mode micro
```

```bash
python -m src.optimizer.workflow benchmark \
  --project my_project \
  --mode deployment \
  --sample-strategy stratified \
  --sample-count 200
```

```bash
python -m src.optimizer.workflow benchmark \
  --project my_project \
  --mode e2e \
  --harness qwen35a3b \
  --prompt-suite demo/qwen35a3b/prompt_suites/medium.jsonl
```

### 6.3 CLI Defaults

Defaults should be:

- benchmark command default mode: `deployment`
- optimize command internal scoring: `micro`
- export command selection policy: `safe`

This makes the interactive user safe by default without slowing down search.

## 7. API Design

The API should expose mode-aware responses rather than one flattened benchmark
blob.

### 7.1 New API Fields

For project benchmark endpoints:

- `available_modes`
- `default_mode`
- `selection_policy`
- `recommended_ops`
- `unsafe_ops`
- `reporting_scope`

For each op:

- `micro`
- `deployment`
- `stress`
- `e2e`
- `selection`

### 7.2 Proposed Read Endpoints

Keep current endpoints working, but add clearer ones:

- `GetProjectBenchmarksV2`
- `GetProjectBenchmarkSummary`
- `GetProjectSelectionPreview`
- `GetProjectStressReport`
- `GetProjectE2EReport`

### 7.3 Proposed Write Actions

- `RunProjectBenchmark`
- `SetProjectSelectionPolicy`
- `SetProjectDefaultBenchmarkMode`
- `ExportProjectSelection`

### 7.4 API Contracts

`SetProjectSelectionPolicy` should accept:

- `safe`
- `fastest`
- `mixed`
- `custom_only`

and return:

- selected ops
- rejected ops
- reasons per rejection

## 8. Default-Selection Policy

This policy should become the product default.

### 8.1 Project Default

When a user opens a project, the default selected kernels should be:

- `deployment.safe == true`
- `selection.recommended_backend == optimized`

Raw microbenchmark winners should not be auto-selected unless the user switches
the project into an advanced unsafe mode.

### 8.2 Export Default

Export should use:

- `selection_policy = safe`

If a user tries to export `fastest` or `mixed`, the UI must show:

- which ops are unsafe
- why they are unsafe
- whether correctness failed, fallback was observed, or coverage is incomplete

### 8.3 Search Default

MCTS and optimization should continue to use:

- `micro`

This is a backend/internal choice and should not be confused with the user's
project selection policy.

### 8.4 Advanced Policies

Expose the following policies:

- `safe`
  Deployment-safe defaults only.
- `mixed`
  Allow unsafe faster kernels, but keep correctness and coverage warnings.
- `fastest`
  Select raw microbenchmark winners only.
- `custom_only`
  Exclude ATen and cuBLAS wrapper-backed kernels where the metadata identifies
  them as wrappers.

## 9. UI And GUI Flows

The UI needs two major changes:

1. benchmark mode must be visible
2. recommendation status must be honest

### 9.1 Project Header

Replace the current single implied benchmark interpretation with:

- `Benchmark mode` pill
- `Selection policy` pill
- `Estimated op-level speedup` label for micro-only summaries
- `End-to-end speedup` label only when e2e data exists

The current "Model Speedup" phrasing should not be shown unless there is actual
e2e evidence.

### 9.2 Project Benchmarks Table

Add a mode switcher:

- `Deployment` default tab
- `Micro`
- `Stress`
- `End-to-end`

On the deployment tab, each op row should show:

- recommended backend
- strict correctness status
- fallback status
- dtype/layout coverage badge
- export eligibility

On the micro tab, show:

- raw speedup
- sample count
- search relevance

but include a warning banner:

- "Microbenchmark results do not imply deployment safety."

### 9.3 Selection Drawer

Add a project-level "Selection policy" control with the following flow:

1. User opens selection drawer.
2. UI defaults to `safe`.
3. UI previews selected ops and rejected ops.
4. Rejected ops include a reason chip:
   - `strict mismatch`
   - `fallback observed`
   - `coverage incomplete`
   - `wrapper excluded by policy`
5. User can switch to `mixed`, `fastest`, or `custom_only`.
6. Unsafe modes require an explicit confirmation checkbox.

### 9.4 Export Flow

Export should be a wizard:

1. Select policy.
2. Review included ops.
3. Review excluded ops and reasons.
4. Review benchmark evidence used for export.
5. Confirm export.

If any selected op is unsafe:

- show a red warning state
- require `allow_unsafe_export`
- persist export policy in the manifest

### 9.5 Dashboard Flow

Dashboard cards should separate:

- `Micro speedup estimate`
- `Deployment-safe coverage`
- `Stress pass rate`
- `E2E result`

Do not collapse them into one headline number.

### 9.6 Op Detail Page

Each operator page should show a benchmark evidence stack:

1. micro result
2. deployment replay result
3. stress result
4. e2e contribution or linked harness result

This is the right place for advanced users to inspect why an op is excluded
from safe selection.

## 10. MCTS And Kernel Benchmarking Impact

Changing the benchmark system should not accidentally slow optimization to a
crawl.

### 10.1 What Should Change

- MCTS should continue to optimize against a fast `micro` score.
- Search values should be stored separately from deployment and reporting data.
- Benchmark jobs should publish richer evidence without mutating the search
  value used by the tree.

### 10.2 What Should Not Change

- Do not make deployment replay the default scoring function inside MCTS.
- Do not make e2e the inner-loop search objective.
- Do not block search on stress or harness-backed jobs.

### 10.3 New Promotion Model

Use a two-stage model:

1. `search`
   MCTS generates candidate kernels using `micro`.
2. `promotion`
   benchmark job evaluates the best candidates under `deployment` and `stress`.

This lets Kernel Forge remain fast without misleading users.

### 10.4 Interaction With Existing Trees

The benchmark refresh path should stop calling a tree mutation that overwrites
root values with reporting measurements. Instead:

- write benchmark evidence into benchmark artifacts
- optionally write a root sidecar summary
- preserve the original search objective

## 11. Sampling And Methodology Fixes

These fixes are required regardless of UI work.

### 11.1 Sample Selection

Replace "first N entries" with configurable strategies:

- `first`
- `random`
- `stratified`

The default should be `stratified` for deployment and stress.

Stratification buckets should include:

- dtype
- shape family
- contiguous vs non-contiguous layout
- phase if available

### 11.2 Statistics

For all user-facing benchmark reports:

- median
- mean
- std
- p95
- sample count

For search:

- keep the fast score, but store more detailed stats when available

### 11.3 Cache Keys

Strengthen baseline cache signatures so different entry sets do not collide when
only later samples differ.

### 11.4 Capture Limits

Fix entry cap semantics and record the capture policy in metadata:

- source order
- max batches
- max per op
- sampling strategy

## 12. Rollout Plan By Commit

This section is intentionally commit-sized. Each commit should land
independently with tests.

### Commit 1: Schema v2 groundwork

Scope:
- add `schema_version = 2`
- add nested `micro`, `deployment`, `stress`, `e2e`, `selection`
- keep legacy read compatibility

Files:
- `src/optimizer/benchmarking/benchmark_ops.py`
- `src/optimizer/workflow.py`
- any benchmark readers
- tests for old and new schema parsing

Acceptance:
- existing projects still render
- new benchmark writes include v2 fields

### Commit 2: Stop corrupting tree search values

Scope:
- split search value from benchmark evidence
- stop benchmark refresh from overwriting root `value`
- add root-sidecar benchmark metadata if needed

Files:
- `src/optimizer/tree_store.py`
- `src/optimizer/benchmarking/benchmark_ops.py`
- `src/optimizer/pipeline.py`
- `src/optimizer/core/mcts.py`

Acceptance:
- MCTS history remains stable
- benchmark refresh no longer mutates search objective fields

### Commit 3: CLI mode plumbing

Scope:
- add `--mode`
- add `--selection-policy`
- add `--sample-strategy`
- add `--stress-profile`

Files:
- `src/optimizer/workflow.py`
- CLI docs
- benchmark orchestration helpers

Acceptance:
- benchmark command supports `micro`, `deployment`, `stress`, `e2e`

### Commit 4: Sampling and cache correctness

Scope:
- implement stratified sampling
- fix entry-cap off-by-one
- strengthen baseline cache signatures
- record sampling metadata

Files:
- `src/optimizer/benchmarking/profile_project.py`
- `src/optimizer/benchmarking/benchmark_ops.py`
- tests

Acceptance:
- sample selection is reproducible
- cache invalidates when entry sets meaningfully change

### Commit 5: Backend selection policy

Scope:
- implement `safe`, `mixed`, `fastest`, `custom_only`
- return reasons for inclusion and rejection
- change export default to `safe`

Files:
- `frontend/walkers/project_admin.jac`
- `frontend/walkers/job_supervisor.jac`
- backend selection helpers
- export code paths

Acceptance:
- no raw winner auto-selection in default flows
- unsafe export requires explicit override

### Commit 6: API v2 benchmark endpoints

Scope:
- add V2 project benchmark response shape
- expose mode-aware fields
- expose selection preview endpoint

Files:
- relevant walkers and API contracts
- docs in `docs/profiling/api.md`

Acceptance:
- frontend can consume mode-aware benchmark data without legacy flattening

### Commit 7: UI mode switch and honest labels

Scope:
- add benchmark mode tabs
- add selection policy controls
- relabel "Model Speedup" to the correct metric name
- add unsafe warnings and reason chips

Files:
- `frontend/app/project/Project.cl.jac`
- `frontend/app/project/shared/MetricsAdapter.cl.jac`
- `frontend/app/project/dashboard/DashboardHeader.cl.jac`
- related components

Acceptance:
- deployment tab is default
- users can inspect why an op is excluded
- no e2e claim is shown without e2e evidence

### Commit 8: Export wizard and unsafe guardrails

Scope:
- add export review wizard
- persist export policy
- require explicit unsafe acknowledgment

Files:
- project export UI
- export manifest writer
- tests

Acceptance:
- users cannot accidentally export raw fastest kernels without acknowledging
  risk

### Commit 9: Stress mode and coverage surfacing

Scope:
- implement stress benchmark mode
- surface layout/dtype/dynamic-shape failures
- add badges in UI

Files:
- benchmark backend
- UI badges and reports
- tests

Acceptance:
- stress results are visible and influence confidence badges

### Commit 10: E2E harness integration

Scope:
- register harness-backed e2e benchmark mode
- store e2e results separately from op-level micro and deployment data
- wire dashboard cards to e2e evidence when present

Files:
- benchmark orchestration
- harness registry
- dashboard readers

Acceptance:
- model-level claims can only come from e2e artifacts

### Commit 11: Docs and migration cleanup

Scope:
- update docs
- add migration notes
- deprecate direct raw `winner` consumption in frontend code

Files:
- `docs/cli.md`
- `docs/profiling/architecture.md`
- `docs/profiling/api.md`
- migration notes

Acceptance:
- docs match product behavior

## 13. Recommended Implementation Order

The highest-value order is:

1. schema v2
2. tree/search separation
3. backend selection policy
4. CLI mode plumbing
5. API v2
6. UI relabeling and selection controls
7. export guardrails
8. sampling and cache fixes
9. stress mode
10. e2e integration

This order fixes user trust first, then deepens the benchmark system.

## 14. Non-Goals

This roadmap does not recommend:

- using e2e latency as the inner-loop MCTS objective
- deleting raw microbenchmark data
- removing advanced unsafe benchmarking modes
- forcing all users onto the slowest benchmark path

## 15. Success Criteria

The roadmap is successful when:

1. default project selection uses deployment-safe kernels only
2. UI labels match the actual evidence level
3. MCTS search values are no longer rewritten by benchmark refresh
4. users can explicitly choose benchmark modes and selection policies
5. export and claims become honest by default
6. open-source users can understand the difference between "fast", "safe", and
   "proven end-to-end"
