# Data Collection and Benchmark Organization

This directory documents the paper data-collection contract for Kernel Forge
model projects. It is documentation only; collected model data should not be
written here.

## Scope

Each model gets one canonical collection file that contains all arms:

```text
paper_benchmarks/data_collection/models/<model_slug>.jsonl
```

There must not be one data file per arm. Each row in the model file is an
append-only JSON object tagged with an `arm`:

```text
zero_shot
optimize_5
optimize_10
optimize_20
optimize_50
```

The normal benchmark harness may still write run directories under
`paper_benchmarks/runs/<run_id>/`. The model JSONL file is the canonical
consolidated record that embeds raw JSON/text artifacts where practical and
links large binary artifacts by path, hash, and size.

## Required Arms

The arms are ordered and gated:

1. `zero_shot`
2. `optimize_5`
3. `optimize_10`
4. `optimize_20`
5. `optimize_50`

Do not start optimized arms until the zero-shot generation has produced usable
kernels and a usable cast export. If zero-shot fails, record the failure and
stop before collecting optimized arms.

## Record Layout

Every JSONL record must include these common fields:

```json
{
  "schema_version": 1,
  "record_type": "string",
  "model_slug": "gemma4-e2b-gb10",
  "project_name": "gemma4-e2b-gb10",
  "arm": "zero_shot",
  "run_id": "stable-run-id",
  "created_at": "2026-04-24T00:00:00Z",
  "source_paths": [],
  "source_hashes": {}
}
```

Use stable `run_id` values so records from generation, export, internal
benchmarking, and later external benchmarking can be joined without relying on
file order.

## Record Types

### `model_project_snapshot`

Record once per model project setup, and again if project configuration changes.

Required payload:
- model id and local model path
- project path
- Forge config JSON
- validation dataset path/hash
- hardware profile, CUDA availability, selected device, target backend
- Python, PyTorch, Transformers, CUDA, compiler, and git provenance
- dirty/untracked git summary

### `profile_snapshot`

Record after profiling completes.

Required payload:
- raw `io/summary.json`
- profiled device, especially `device`
- `op_counts`
- `skipped_counts`
- skip filters
- captured op directories
- per-op captured entry count
- for captured `.pt` files: path, sha256, byte size, shape/dtype metadata where available

Large `.pt` files should not be embedded as bytes in JSONL unless explicitly
requested. Record enough path/hash/metadata to prove exactly which inputs were
used.

### `forge_operator_benchmark`

Record after Kernel Forge operator benchmarking for the arm.

Required payload:
- raw `benchmarks/op_benchmarks.json`
- raw `benchmarks/torch_baseline_cache.json` if present
- PyTorch baseline latency per op
- generated/optimized kernel latency per op
- speedup per op
- correctness status per op
- failure or skip reason per op
- benchmark command line
- benchmark environment controls

### `generation_attempt`

Record for every generated or repaired kernel attempt.

Required payload:
- op name
- op directory
- generation phase: zero-shot, repair, or optimization
- iteration number and attempt number
- kernel node id when available
- source path and sha256
- validation status
- profiler status
- failure stage/reason
- attempts-to-correct summary
- raw attempt summary JSON and validation/profiling logs

### `llm_usage`

Record every LLM call that contributes to an arm or cast export.

Required payload:
- provider
- model
- reasoning effort
- step type, for example `generation`, `repair`, `optimization`, `verifier_summary`
- op name
- iteration number
- attempt number
- input tokens
- output tokens
- reasoning tokens
- total tokens
- computed cost
- raw usage row from `llm_usage.db`

Token records must be preserved per call and also summarized per:
- op
- arm
- cast export
- whole model file

### `cast_export`

Record once for every `.cast` export.

Required payload:
- cast export id
- cast file path
- cast file sha256
- cast file byte size
- raw cast manifest/export metadata
- export command or UI action metadata
- export selection policy
- paper eligibility status
- rejected export candidate summary
- selected kernels per op
- token usage totals tied to this export

The selected-kernel mapping is mandatory. It must be explicit enough to audit
which kernel implementation was packaged:

```json
{
  "selected_kernels": {
    "torch.nn.functional.linear": {
      "kernel_id": "kernel_4",
      "node_id": 4,
      "source_path": "kernels/projects/.../kernel_4.cu",
      "source_sha256": "..."
    },
    "torch.nn.functional.gelu": {
      "kernel_id": "kernel_2",
      "node_id": 2,
      "source_path": "kernels/projects/.../kernel_2.cu",
      "source_sha256": "..."
    }
  }
}
```

### `external_benchmark_run`

Record when the later external benchmark is run against a cast export.

Required payload:
- cast export id and cast hash used
- benchmark run directory
- raw metric JSON files
- raw summary JSON/CSV/Markdown
- model config path/hash
- suite config path/hash
- workload path/hash
- prompt token counts
- generated token counts
- output token hashes
- prefill latency and throughput
- decode latency and throughput
- total latency and throughput
- correctness/equality status
- eager and `torch_compile` baselines when present
- command line and environment snapshot

## File Organization

Use this layout for collection artifacts:

```text
paper_benchmarks/data_collection/
├── collect_zero_shot.py
├── docs/
│   └── README.md
├── artifacts/
│   └── <model_slug>/<run_id>/
│       ├── <model_slug>__zero_shot__full_forge.cast
│       ├── <model_slug>__zero_shot__mixed_forge.cast
│       └── collection_manifest.json
├── models/
│   ├── gemma4-e2b-gb10.jsonl
│   └── qwen35a3b-gb10.jsonl
└── indexes/
    └── collection_manifest.json
```

`models/*.jsonl` are the authoritative per-model data files. `indexes/` may
contain convenience manifests, but those files must be derivable from the model
JSONL files.

## Benchmark Run Organization

Internal Kernel Forge benchmarking stays with the Forge project:

```text
kernels/projects/<project>/
├── io/
│   ├── summary.json
│   └── individual_ops/
├── benchmarks/
│   ├── op_benchmarks.json
│   └── torch_baseline_cache.json
└── exports/
    └── <project-or-arm>.cast
```

External/paper benchmarking stays in immutable run directories:

```text
paper_benchmarks/runs/<run_id>/
├── manifest.json
├── env.json
├── commands.txt
├── metrics/
├── correctness/
├── reports/
└── logs/
```

The collection process reads from both locations and appends normalized records
to the single model JSONL file.

## Zero-Shot Collection Helper

After zero-shot generation is finished, first verify the Forge project is idle:

```bash
pgrep -af 'src.optimizer.workflow|src.generator.main|src.optimizer.pipeline'
jq '.generate, .profile' kernels/projects/<project_name>/state.json
jq '.active_tasks, .job_queue, .pending_operators' kernels/projects/<project_name>/queue.json
```

The project is ready to collect only when generation is completed, no optimize
job is active or queued, and the generated op directories contain the expected
`success.cuda` markers. Then collect with the repo virtualenv Python, not the
system Python:

```bash
/home/gb10/Projects/Kernal-Forge/.venv/bin/python \
  paper_benchmarks/data_collection/collect_zero_shot.py \
  --project <project_name> \
  --model-slug <model_slug>
```

The helper refuses to run while `state.json` marks generation as active unless
`--allow-running` is passed. It does not start generation or optimization.

For the zero-shot arm it exports and records:
- `full_forge`: every available successful zero-shot Forge kernel is forced
  into the cast; profiled ops without a generated kernel are recorded as
  missing full coverage
- `mixed_forge`: the normal `auto_best_fastest_valid` policy is used, so ops
  that should remain native/Torch fallback are explicitly recorded as fallback

The helper appends these record types to the model JSONL file:
- `model_project_snapshot`
- `profile_snapshot`
- `forge_operator_benchmark`
- `generation_attempt`
- `llm_usage`
- `cast_export` for full Forge
- `cast_export` for mixed Forge
- `arm_summary`

Every cast record includes `dispatch_by_profiled_op` so it is explicit which
profiled ops used a Forge kernel and which used Torch fallback.

### State and Artifact Capture

For each zero-shot collection, preserve the current Forge state in the model
JSONL record and the run manifest. The collector must capture or reference:

- `kernels/projects/<project>/config.json`
- `kernels/projects/<project>/state.json`
- `kernels/projects/<project>/queue.json`
- `kernels/projects/<project>/io/summary.json`
- `kernels/projects/<project>/benchmarks/op_benchmarks.json`
- `kernels/projects/<project>/benchmarks/torch_baseline_cache.json`
- `kernels/projects/<project>/logs/generate.log`
- project-level `llm_usage.db`
- per-op `kernels/generated/individual_op_kernels/<op>/llm_usage.db`
- per-op generated kernel sources, attempts, `success.cuda`, and tree metadata

LLM usage must be recorded both as raw DB rows and as explicit summaries:

- per call
- per op
- per op and step type
- per provider/model
- whole arm totals

The per-op summary must include call count, input tokens, output tokens,
reasoning tokens, total tokens, input cost, output cost, and total cost.

### CAST Storage Rules

Store `.cast` exports under the data-collection artifact directory, not only
inside the Forge project:

```text
paper_benchmarks/data_collection/artifacts/<model_slug>/<run_id>/
├── <model_slug>__zero_shot__full_forge.cast
├── <model_slug>__zero_shot__mixed_forge.cast
└── collection_manifest.json
```

The `full_forge` CAST force-selects every successful zero-shot generated kernel.
The `mixed_forge` CAST uses the normal fastest-valid policy and may dispatch
some or all ops to Torch fallback. Record both, even when they are identical.

Because `*.cast` is globally ignored by the repo, intentional paper CAST files
must be force-added when committing:

```bash
git add -f paper_benchmarks/data_collection/artifacts/<model_slug>/<run_id>/*.cast
```

Each CAST must have path, repo-relative path, SHA256, byte size, export
metadata, and selected-kernel dispatch table recorded in both the model JSONL
and the `collection_manifest.json`.

Current Gemma zero-shot example:

```text
paper_benchmarks/data_collection/artifacts/gemma4-e2b-gb10/gemma4-e2b-gb10__zero_shot__20260424T043053Z/
├── gemma4-e2b-gb10__zero_shot__full_forge.cast
├── gemma4-e2b-gb10__zero_shot__mixed_forge.cast
└── collection_manifest.json
```

## Collection Checks

Before an arm is considered complete, verify:
- profile summary exists and `device` is the intended accelerator
- op benchmark JSON exists
- every selected export kernel has a source path and hash
- the cast file has path, hash, and size recorded
- each op in the cast records the selected kernel id
- every LLM usage row has input, output, and reasoning tokens
- token totals reconcile per op, per arm, and per cast export
- explicit `usage_by_op` exists in the model JSONL and manifest
- external benchmark records point to the exact cast hash used
- correctness status is present for every reported speedup
- `full_forge` and `mixed_forge` dispatch tables are both recorded, including
  Forge-vs-Torch fallback decisions for every profiled op
- no optimization job was started unless the next arm was explicitly approved

If any check fails, append an explicit failure record for that arm instead of
silently omitting it.

Do not move to `optimize_5`, `optimize_10`, `optimize_20`, or `optimize_50`
until the zero-shot JSONL records, CAST files, manifest, per-op usage totals,
and dispatch tables have been verified.
