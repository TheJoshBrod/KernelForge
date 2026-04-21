# Paper Benchmark Harness Implementation Plan

Current branch/context:
- Repo root: `/home/gb10/Projects/Kernal-Forge/CGinS`
- Branch: `current-origin-main`
- Commit: `473cc250c9ad45fa868c4895a3a48d3e33f04ea4`
- Initial target machine observed during inspection: `NVIDIA GB10`, CUDA toolkit `13.0`, PyTorch `2.10.0+cu130`, GPU capability `(12, 1)`

Inspected inputs:
- `README.md`
- `docs/cli.md`
- `docs/cast-runtime.md`
- `src/optimizer/benchmarking/*`
- `src/optimizer/workflow.py`
- `src/optimizer/tree_store.py`
- `kernelforge/run_cast.py`
- `tests/test_benchmark_harness.py`
- frontend dashboard/export surfaces under `frontend/app/project/...` and related walkers

Missing on this branch:
- `docs/benchmarking-roadmap.md` does not exist. The harness plan below treats that as absent rather than assuming a hidden roadmap.

## Why This Must Be Separate From The Existing Internal Benchmark Path

The existing benchmark stack is useful for internal operator ranking and UI state, but it is not paper-defensible as-is:
- `src/optimizer/benchmarking/benchmark_ops.py` explicitly labels its protocol as internal-only and without confidence intervals.
- The current operator benchmark path is keyed around captured operator entries, not frozen end-to-end LLM workloads.
- `profile_project.py` can fall back to default synthetic samples if no real validation data is present.
- `benchmark_ops.py` can estimate generated-kernel performance from the PyTorch baseline when direct profiling is unavailable (`ninja` case), which is unacceptable for paper claims.
- `kernelforge/run_cast.py` currently falls back to native PyTorch inside runtime patches on exceptions and only exposes that behavior through warnings/stdout, not structured evidence.
- The current `.cast` benchmark mode uses dummy synthetic inputs and mixes deployment/runtime preparation with ad hoc benchmarking behavior.

The new harness should therefore live under `paper_benchmarks/` and treat the existing internal benchmark path as:
- a source of optional appendix-style operator data,
- not the source of end-to-end claims,
- not the source of deployment correctness claims,
- not the source of workload freezing or provenance.

## Scope

Milestone 1:
- Build a paper-grade LLM benchmark harness for one target first: Qwen 3.5 / Qwen 35B-style benchmarking on this DGX Spark-class GB10 machine.
- Compare exactly three execution modes:
  - PyTorch eager
  - `torch.compile` / Inductor
  - Kernel Forge deployed kernels
- Separate prefill and decode.
- Use only frozen, real, non-synthetic prompt/data packs.
- Record full provenance and invalidate stale results on any hash mismatch.
- Gate every speedup claim on correctness.

Milestone 2:
- Scale the same harness to roughly 10 models by adding model configs, not by forking logic.

Out of scope for the first implementation:
- Reworking MCTS or optimizer internals
- Replacing the internal UI benchmark system
- Claiming general performance leadership beyond the tested models/hardware/workloads

## Implementation Plan

### Phase 0: Freeze The Benchmark Spec Before Running Anything

Add versioned suite manifests under `paper_benchmarks/` that fully define:
- model identifier and local model path
- tokenizer identifier/path
- dtype / quantization / attention backend settings
- batch size
- prompt pack version
- decode settings (`max_new_tokens`, greedy vs sampling, stop conditions)
- number of warmups and timed repetitions
- whether the mode is eager, compile, or deployed cast

No benchmark run is valid unless it references an immutable suite manifest version.

### Phase 1: Freeze Real Prompt Packs

Create a workload-freezing flow that:
- pulls prompts from real public datasets or real prompt corpora,
- records dataset/source identifiers and sample IDs,
- renders the exact prompts once,
- stores a frozen JSONL prompt pack plus a content hash,
- groups prompts into explicit length buckets for prefill and decode analysis,
- refuses to run if the prompt pack content hash has changed.

Planned benchmark policy:
- No synthetic prompts for paper runs.
- No on-the-fly prompt sampling during benchmark execution.
- Prefill and decode prompt sets are frozen independently.

### Phase 2: Provenance And Hash Discipline

Every run should emit a provenance manifest with:
- harness version
- git commit and branch
- dirty-worktree summary
- exact CLI invocation
- hostname
- OS / kernel / Python
- PyTorch version
- CUDA toolkit version
- GPU name / UUID / capability / driver
- model path and model file hash
- tokenizer path/hash
- prompt pack hash
- `.cast` package hash
- deployed kernel source hashes
- package dependency fingerprint

Cache/reuse rule:
- If any required hash changes, previous results are invalid and must not be reused.
- Missing provenance is a benchmark failure, not a neutral data point.

### Phase 3: Runner Architecture

Implement three runners behind a common interface:

1. `eager`
- Native PyTorch model load, eval mode, deterministic settings

2. `compile`
- Same model and weights
- `torch.compile` configuration recorded explicitly
- compile/setup time captured separately from steady-state timing

3. `cast`
- Load exported `.cast` package through `kernelforge.run_cast.load_cast`
- `.cast` extraction, JIT/precompiled load, and runtime setup captured separately from steady-state timing
- deployed-kernel coverage and fallback behavior recorded explicitly

The timed benchmark path must exclude:
- offline search/generation
- export/package creation
- compile time
- JIT load time
- first-run cache construction

Those costs should still be recorded in separate fields.

### Phase 4: LLM Measurement Protocol

For each prompt:

Prefill:
- Run the full prompt once the model is loaded/warmed.
- Measure prompt processing latency and prefill throughput separately.

Decode:
- Start from the same frozen prompt and run fixed-length greedy decode.
- Measure per-step decode latency and decode tok/s separately.
- Report decode independently from prefill and from combined end-to-end.

Statistical protocol:
- multiple warmups
- repeated timed runs
- process-isolated runs where appropriate
- randomized or rotated mode order to reduce thermal/order bias
- summary metrics based on medians and dispersion, with confidence intervals or bootstrap intervals at the report layer

### Phase 5: Correctness Gating

No mode is allowed to claim a win unless correctness passes.

Correctness checks should include:
- prefill logits comparison against eager with explicit tolerances
- decode token sequence comparison under deterministic greedy decode
- optional per-step logits/top-k sanity checks for debugging failures

Failure policy:
- output mismatch => result invalid
- hidden fallback => result invalid for deployed-kernel claims
- missing evidence => result invalid
- invalid runs remain visible in raw outputs but are excluded from claim tables

### Phase 6: Reporting Separation

Keep three result families separate:
- micro/operator results
- end-to-end LLM eager vs compile results
- deployment/runtime results for `.cast`

The harness should write:
- raw per-run JSONL
- normalized per-suite JSON summaries
- provenance manifests
- correctness summaries
- deployment coverage/fallback summaries
- paper-ready CSV/Markdown tables and plotting inputs

### Phase 7: Scale-Out To ~10 Models

After the Qwen 35B-style milestone works, scale by adding model registry entries:
- model config file
- tokenizer config
- prompt pack binding
- expected tolerance settings
- deployment/export recipe

No new measurement logic should be needed per model.

## Proposed File Tree

```text
paper_benchmarks/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── __init__.py
├── version.py
├── cli.py
├── suites/
│   ├── llm_dgx_spark_v1.yaml
│   └── paper_claims_policy_v1.yaml
├── models/
│   ├── qwen35_demo_gb10.yaml
│   └── registry.yaml
├── workloads/
│   ├── freeze_prompts.py
│   ├── sources.py
│   ├── rendering.py
│   └── frozen/
│       ├── qwen35_prefill_v1.jsonl
│       ├── qwen35_decode_v1.jsonl
│       └── manifests/
│           └── qwen35_workload_v1.json
├── provenance/
│   ├── env.py
│   ├── hashes.py
│   ├── package_fingerprint.py
│   └── manifest.py
├── runners/
│   ├── common.py
│   ├── eager.py
│   ├── compile.py
│   ├── cast.py
│   └── llm.py
├── correctness/
│   ├── llm.py
│   └── tolerances.py
├── results/
│   ├── schema.py
│   ├── writer.py
│   ├── validate.py
│   └── merge.py
├── reports/
│   ├── aggregate.py
│   ├── tables.py
│   └── plots.py
├── scripts/
│   └── run_qwen35_demo_gb10.sh
├── tests/
│   ├── test_provenance_hashes.py
│   ├── test_result_schema.py
│   ├── test_prompt_pack_freeze.py
│   ├── test_correctness_gate.py
│   └── test_cast_runner_metadata.py
└── outputs/
    ├── raw/
    ├── summaries/
    └── reports/
```

Notes:
- `outputs/` should be gitignored and treated as generated artifacts.
- If we later want appendix-style operator data, add `paper_benchmarks/micro/` rather than reusing the LLM result schema.

## Smallest Kernel Forge Core Changes Required

Only one core/runtime change looks strictly required for paper-defensible deployed-kernel benchmarking:

1. `kernelforge/run_cast.py`
- Add an opt-in structured runtime report for `load_cast(...)`.
- The report should expose, per op:
  - op name
  - source relpath
  - source hash
  - load mode (`precompiled`, `jit`, `skipped`, `native_fallback`)
  - fallback/skip reason
  - whether the patch was actually registered
- This must be machine-readable, not stdout-only.
- This is the smallest clean way to satisfy “do not hide fallback to PyTorch.”
- Cover it with focused tests.

No optimizer/MCTS rewrite is planned.
No change is planned to `src/optimizer/benchmarking/*` for milestone 1.
No change is planned to `workflow.py` unless a later headless export path becomes unavoidable.

## Benchmark Claims Allowed Only After The Harness Is Complete

Allowed claims, once the harness exists and the relevant runs pass:
- Exact eager vs `torch.compile` vs Kernel Forge deployed-kernel comparisons for a named model, named machine, named prompt pack, and named suite version
- Separate prefill latency / throughput claims
- Separate decode latency / throughput claims
- Separate steady-state deployment/runtime claims, with compile/JIT/setup costs reported independently
- Correctness-gated speedup claims
- Deployment coverage claims such as how many exported kernels actually loaded vs fell back
- Reproducibility claims tied to full provenance and matching hashes
- Appendix/operator-level claims, explicitly labeled as microbenchmarks and not mixed into end-to-end claims

## Benchmark Claims That Must Remain Forbidden

Forbidden claims:
- Any “Kernel Forge wins” claim when correctness failed or was not checked
- Any claim that hides PyTorch fallback inside the deployed runtime
- Any steady-state latency claim that mixes in offline generation, search, verification, export, compile, or JIT setup cost
- Any LLM latency claim that combines prefill and decode into a single opaque number
- Any comparison that omits either eager or `torch.compile`
- Any paper claim based on synthetic prompts or synthetic inputs
- Any claim based on stale results when hashes do not match
- Any use of missing evidence as a neutral speedup
- Any end-to-end claim inferred from operator microbenchmarks
- Any general “faster on all models” or “production-ready universal speedup” claim beyond the exact tested matrix
- Any claim based on estimated kernel timings, inferred fallback behavior, or stdout parsing alone

## Immediate Next Step After Plan Approval

Implement the harness skeleton under `paper_benchmarks/` first:
- suite schema
- provenance capture
- frozen prompt-pack format
- eager / compile / cast runners
- correctness gates
- raw result schema

Only after that should we wire in the Qwen 35B demo benchmark and produce the first real run.
