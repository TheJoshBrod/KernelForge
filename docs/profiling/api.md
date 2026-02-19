# Profiling API Contract

## `GetProjectStatus`

Reads `kernels/projects/<project>/state.json` and reports job state.

`profile` state keys:

- `status`: `queued|running|paused|completed|error|cancelled`
- `message`: stage or error text
- `started_at`
- `finished_at`
- `log`
- `pid`

Legacy states are reconciled:

- if `profile.status=queued` and `prepare.status=completed`, status is normalized to `completed` or `error` based on benchmark/log presence.

## `GetDashboardCharts`

Reads:

- `benchmarks/op_benchmarks.json`
- `io/summary.json`

Returns:

- `operator_usage`: list
- `speed_comparison`: list
- `status`: `pending|error|empty|partial|ready`
- `profile_status`
- `profile_message`

`status` behavior:

- `pending`: profiling/benchmarking still running
- `error`: profiling failed
- `empty`: no benchmark outputs yet and no active run
- `partial`: benchmark output exists with recoverable per-op failures
- `ready`: benchmark output present
