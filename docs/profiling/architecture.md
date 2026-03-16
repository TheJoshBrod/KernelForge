# Profiling and Benchmarking Architecture

## Canonical data layout

Runtime artifacts are written under:

- `kernels/projects/<project>/io/individual_ops/`
- `kernels/projects/<project>/io/summary.json`
- `kernels/projects/<project>/benchmarks/op_benchmarks.json`
- `kernels/projects/<project>/benchmarks/torch_baseline_cache.json`
- `kernels/projects/<project>/state.json`

## Execution model

### Profile stage

The `profile` job prepares project inputs and baseline benchmark data.

### Generate stage (standard path)

`generate` runs per operator (sequentially):

1. Generate kernel for operator
2. Validate compile/correctness via backend success marker
3. Optional optimize for the same operator
4. Optional benchmark refresh

This enables incremental chart updates while the run is in progress.

### Optimize stage

Optimization is MCTS-driven per operator and writes tree artifacts under `trees/<op>/`.

### Benchmark stage

Benchmark reads baseline data and optimized outputs to build `op_benchmarks.json`.

## Orchestration entrypoint

Canonical CLI orchestration is:

```bash
python -m src.optimizer.workflow <profile|generate|optimize|benchmark> ...
```

Frontend orchestration is implemented in:

- `frontend/walkers/kernel_job_runners.jac`

## Robustness rules

- State transitions are persisted in `state.json`.
- Stale process recovery avoids silent success.
- Chart status is explicit (`pending|error|empty|partial|ready`).
- Baseline benchmark cache is fingerprinted by runtime/device context.
