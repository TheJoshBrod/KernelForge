# Profiling and Benchmarking Architecture

## Canonical data layout

- Source logic: `src/optimizer/benchmarking/`
- CLI entrypoints: `src/cli/`
- Frontend orchestration: `frontend/walkers/project.jac`
- Runtime artifacts: `kernels/projects/<project>/...`

Generated runtime outputs:

- `kernels/projects/<project>/io/individual_ops/`
- `kernels/projects/<project>/io/summary.json`
- `kernels/projects/<project>/benchmarks/op_benchmarks.json`
- `kernels/projects/<project>/benchmarks/torch_baseline_cache.json`
- `kernels/projects/<project>/state.json`

## Execution model

The profile flow is one tracked job (`profile`) with staged progress:

1. `Preparing assets`
2. `Profiling operators`
3. `Benchmarking operators`

`src.cli.run_job` tracks lifecycle (`running`, `completed`, `error`) and pipeline stage messages.

## Robustness rules

- State and benchmark JSON writes use file locks and atomic replace.
- Torch baseline is cached by op signature + runtime fingerprint.
- Missing benchmark outputs do not silently appear as success; chart APIs expose status (`pending`, `error`, `empty`, `ready`).
- Legacy stuck state (`profile=queued` + `prepare=completed`) is reconciled on read.
