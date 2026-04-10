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

The `profile` job runs `profile_project.py` which does two things in sequence:

**1. Op counting and entry capture** (`torch.nn.functional` hooks)

All samples from the model's dataloader are run through the model. The **first pass** sets the canonical call counts written to `summary.json → op_counts` (single forward pass). All subsequent passes write additional captured tensor input/output pairs to `individual_ops/<op>/` — these extra entries are used by `benchmark_ops` to average latency over the full validation set without inflating call counts.

**2. DAG export** (`torch.jit.trace`)

A single forward pass is traced using `torch.jit.trace(model, samples[0])`. The inlined JIT graph is walked to extract meaningful NN ops and their tensor data-flow connections. Primitive ops (`reshape`, `add`, `flatten`, etc.) are filtered out but their connections are propagated so the graph remains fully connected across them. The result is written to `io/dag.json` and displayed in the **Data Flow view**.

### Stat sources by UI section

| Stat | Source file | Scope |
|------|-------------|-------|
| Calls (dashboard & workbench) | `summary.json → op_counts` | Single forward pass |
| PyTorch ms | `op_benchmarks.json → pytorch_ms` | Average over full validation set (up to 50 entries replayed) |
| Kernel ms (generated, not optimized) | `op_benchmarks.json → kernel_ms` | Average over same validation entries |
| Kernel ms (MCTS-optimized) | `nodes.db` or `improvement_log.json` | Best result from optimization search |
| Data Flow graph nodes/edges | `io/dag.json` | Single forward pass |

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
