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
- `default_mode`: benchmark mode that should drive the default UI interpretation
- `selection_policy`: selection policy used for default recommendations
- `speedup_label`: user-facing label for the headline speedup KPI
- `status`: `pending|error|empty|partial|ready`
- `profile_status`
- `profile_message`
- `errors`

Behavior notes:

- `speed_comparison` may contain baseline-only rows while generation is in progress.
- `operator_usage` and headline speedup estimates prefer deployment-safe recommended timings.
- During per-operator generation/optimization, chart payload may update incrementally.

## `GetProjectBenchmarks`

Reads `kernels/projects/<project>/benchmarks/op_benchmarks.json` and returns the
project benchmark artifact in a frontend-friendly shape.

Returns:

- `results`: benchmark rows
- `recommended_ops`: ops whose `selection.recommended_backend == optimized`
- `raw_winners`: ops that only win the raw microbenchmark
- `unsafe_ops`: rejected ops with reason strings
- `available_modes`: currently `micro|deployment|stress|e2e`
- `default_mode`
- `selection_policy`

Behavior notes:

- `winners` is retained as a legacy alias for `recommended_ops`
- callers should prefer `recommended_ops` over raw `winner`
- `results` may contain both schema v2 nested fields and legacy flat fields during
  the migration window

## `GetProjectSelectionPreview`

Reads the same benchmark artifact as `GetProjectBenchmarks`, applies a selection
policy, and returns a UI-ready preview for review/export surfaces.

Inputs:

- `projectName`
- `selectionPolicy`: `safe|mixed|fastest|custom_only`

Returns:

- `selection_policy`
- `selected_ops`
- `selected_details`: list of selected op entries with `reason`, `warning`,
  `export_allowed`, and `tags`
- `excluded_ops`: list of excluded op entries with `reason` and `tags`
- `recommended_ops`
- `raw_winners`
- `unsafe_selected_ops`
- `requires_unsafe_ack`
- `policy_note`
- `available_modes`
- `default_mode`

Behavior notes:

- `safe` selects only deployment-safe recommendations
- `mixed` allows raw microbenchmark winners alongside safe recommendations
- `fastest` selects raw microbenchmark winners only
- `custom_only` excludes wrapper-backed kernels when classification metadata is
  present, otherwise it falls back to deployment-safe recommendations and sets
  `policy_note`
