# Profiling API Contract

## `GetProjectStatus`

Reads `kernels/projects/<project>/state.json` and reports tracked job states.

Primary keys used by current flow:

- `profile`
- `generate`
- `optimize_adv` (optimize job path still uses this key in current walkers)
- `benchmark`

Common state fields:

- `status`: `queued|running|paused|completed|error|cancelled`
- `phase`: backend stage text
- `message`: progress/error text
- `started_at`
- `updated_at`
- `finished_at`
- `log`
- `pid`
- `progress` (optional): `{current,total,percent,updated_at}`

Legacy compatibility:

- legacy `generate_adv` state is migrated to `generate` on read.

## `GetDashboardCharts`

Reads project benchmark + profile artifacts and returns dashboard chart payload.

Inputs:

- `kernels/projects/<project>/benchmarks/op_benchmarks.json`
- `kernels/projects/<project>/io/summary.json`

Returns:

- `operator_usage`: list of `{name,count,total_time_ms}`
- `speed_comparison`: list of `{name,pytorch,optimized?}`
- `status`: `pending|error|empty|partial|ready`
- `profile_status`
- `profile_message`
- `errors`

Behavior notes:

- `speed_comparison` may contain baseline-only rows while generation is in progress.
- `operator_usage` prefers optimized timing when available, else baseline timing.
- During per-operator generation/optimization, chart payload may update incrementally.
