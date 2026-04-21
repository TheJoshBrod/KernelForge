# Design

## Separation

The paper harness is intentionally separate from `src/optimizer/benchmarking/*`.

It may import:
- public runtime APIs such as `kernelforge.run_cast`
- generic Python / PyTorch / Transformers functionality

It should not depend on:
- internal UI state files
- internal operator benchmark caches
- implicit speedup defaults

## Artifact Model

Artifacts are strict JSON documents validated with `pydantic` in `extra="forbid"` mode.

Artifact types:
- `run_manifest`
- `environment_snapshot`
- `benchmark_result`
- `summary_report`

Each artifact carries a common provenance block at the top level so validation does not depend on nested optional structures.

## Run Layout

Each run gets its own immutable directory:

```text
runs/<timestamp>_<model_id>_<suite_id>/
├── manifest.json
├── env.json
├── commands.txt
├── raw/
├── metrics/
├── correctness/
├── reports/
└── logs/
```

## Synthetic Guardrail

Synthetic workloads are rejected by default.

The only escape hatch is `--allow-synthetic-demo`, which:
- permits the run to start
- forces `paper_eligible=false`
- leaves an explicit audit trail in the artifacts

## Correctness

The scaffold records correctness status on every benchmark artifact.

It does not convert missing correctness into a win:
- summaries do not emit speedup claims unless correctness is `reference` or `passed`
- missing baselines never become `1.0x`

