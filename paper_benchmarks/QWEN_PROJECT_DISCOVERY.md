# Qwen Project Discovery

Date: 2026-04-21 UTC

## Scope

Target project reference:

- `project/test_qwen%20-%20NVIDIA%20GB10/`

Search terms used:

- `test_qwen%20-%20NVIDIA%20GB10`
- `test_qwen - NVIDIA GB10`

This note identifies the canonical local Kernel Forge project that should be treated as the source of truth for Qwen CAST export, and assesses whether its current artifacts are sufficient for safe default CAST auto-selection.

## Search Roots

### Environment

- `KFORGE_DATA_DIR` was not set in the current shell environment.
- No other `KFORGE_*` project/data root override was set.

### Code-defined project roots

Frontend/backend walker storage roots:

- `frontend/walkers/catalog_db.jac` resolves `_projects_base_dir()` to:
  - `${KFORGE_DATA_DIR}/projects` if `KFORGE_DATA_DIR` is set
  - otherwise `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects`
- `frontend/walkers/job_supervisor.jac` resolves `_project_dir(name)` under that same base root and migrates legacy roots from:
  - `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/private/projects`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/projects`

Runtime path helper roots:

- `src/projects/paths.py` resolves `projects_root()` to:
  - `${KFORGE_DATA_DIR}/projects` if `KFORGE_DATA_DIR` is set
  - otherwise `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/private/projects`
- It also defines a legacy root:
  - `/home/gb10/Projects/Kernal-Forge/CGinS/projects`

### Filesystem roots searched

Searched exact roots:

- `/home/gb10/Projects/Kernal-Forge/CGinS/project`
- `/home/gb10/Projects/Kernal-Forge/CGinS/projects`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/private/projects`
- `/home/gb10/Projects/Kernal-Forge/CGinS/data`
- `/home/gb10/Projects/Kernal-Forge/CGinS/.kernelforge`
- `/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs`
- `/home/gb10/.kernelforge`
- `/home/gb10/.local/share`
- `/home/gb10/Projects/Kernal-Forge`
- `/home/gb10`

Existence results:

- Missing:
  - `/home/gb10/Projects/Kernal-Forge/CGinS/project`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/projects`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/private/projects`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/data`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/.kernelforge`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs`
  - `/home/gb10/.kernelforge`
- Present:
  - `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects`
  - `/home/gb10/.local/share`
  - `/home/gb10/Projects/Kernal-Forge`

Search results:

- No match was found under:
  - `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/private/projects`
  - `/home/gb10/Projects/Kernal-Forge/CGinS/projects`
  - `/home/gb10/.kernelforge`
  - `/home/gb10/.local/share`
- One decoded match was found under:
  - `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10`

## Canonical Project

### Encoded/decoded mapping

The requested encoded project reference maps to the decoded filesystem project name:

- Encoded/UI form: `project/test_qwen%20-%20NVIDIA%20GB10/`
- Decoded project name: `test_qwen - NVIDIA GB10`
- Filesystem path: `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10`

There was no literal on-disk directory named `test_qwen%20-%20NVIDIA%20GB10`.

### Project identity

Canonical project root:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10`

Project id/name/slug:

- Project name: `test_qwen - NVIDIA GB10`
- Base name from `config.json`: `test_qwen`
- URL-safe reference implied by UI/API encoding: `test_qwen%20-%20NVIDIA%20GB10`

Catalog registration:

- `kernels/projects/catalog.db` contains:
  - `name = test_qwen - NVIDIA GB10`
  - `project_path = /home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10`
  - `created_at = 2026-03-21T20:22:45.576787`
  - `updated_at = 2026-04-21T19:03:00.381300`

## Model Path Recorded By The Project

There is no explicit model snapshot path recorded in `config.json`, `state.json`, or `queue.json`.

What the project does record:

- `config.json` records:
  - `base_name = test_qwen`
  - `validation_dir = data/validation`
  - `backend = cuda`
  - empty `artifacts.weights`
- `model.py` encodes the model source policy:
  - HF model id: `Qwen/Qwen3.5-35B-A3B`
  - local snapshot env override: `KFORGE_QWEN35_DIR`
  - default local snapshot path: `/home/gb10/model-cache/Qwen3.5-35B-A3B`

Conclusion:

- The project code points at `/home/gb10/model-cache/Qwen3.5-35B-A3B` by default.
- That path is implied by `model.py`, not recorded as immutable project metadata.

## Artifact Inventory

### Project files found

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/config.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/model.py`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/state.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/queue.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/summary.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/op_benchmarks.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/qwen_tps_compare.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/exports/test_qwen - NVIDIA GB10.cast`

### Captured operator entries

Validation prompt pack:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/data/validation/prompts.jsonl`

Captured operator entry roots:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_conv1d`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_embedding`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_grouped_mm`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_linear`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_pad`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_sigmoid`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_silu`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_softmax`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/io/individual_ops/torch_nn_functional_softplus`

Captured entry counts:

| op | captured entry files |
| --- | ---: |
| `torch_nn_functional_conv1d` | 60 |
| `torch_nn_functional_embedding` | 2 |
| `torch_nn_functional_grouped_mm` | 160 |
| `torch_nn_functional_linear` | 201 |
| `torch_nn_functional_pad` | 201 |
| `torch_nn_functional_sigmoid` | 80 |
| `torch_nn_functional_silu` | 201 |
| `torch_nn_functional_softmax` | 100 |
| `torch_nn_functional_softplus` | 60 |

### Optimization results

Tree roots:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_conv1d`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_embedding`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_grouped_mm`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_linear`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_pad`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_sigmoid`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_silu`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_softmax`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_softplus`

Current best node per tree:

| op | best node id | best node path | best node value ms |
| --- | ---: | --- | ---: |
| `torch_nn_functional_conv1d` | 4 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_conv1d/kernels/kernel_4.cu` | 0.006138560175895691 |
| `torch_nn_functional_embedding` | 14 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_embedding/kernels/kernel_14.cu` | 0.006234560012817383 |
| `torch_nn_functional_grouped_mm` | 20 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_grouped_mm/kernels/kernel_20.cu` | 0.43983070373535155 |
| `torch_nn_functional_linear` | 4 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_linear/kernels/kernel_4.cu` | 0.005028480291366577 |
| `torch_nn_functional_pad` | 5 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_pad/kernels/kernel_5.cu` | 0.004108799993991852 |
| `torch_nn_functional_sigmoid` | 1 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_sigmoid/kernels/kernel_1.cu` | 0.002056320011615753 |
| `torch_nn_functional_silu` | 4 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_silu/kernels/kernel_4.cu` | 0.002199999988079071 |
| `torch_nn_functional_softmax` | 4 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_softmax/kernels/kernel_4.cu` | 0.004084799885749817 |
| `torch_nn_functional_softplus` | 1 | `kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_softplus/kernels/kernel_1.cu` | 0.0022067199647426605 |

Catalog metric rows for this project mark all nine ops as having a `forged_ms` value and `winner = forged`, but that catalog view is weaker than the richer benchmark JSON described below.

### Generated kernel source files

Generated kernel roots:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/*`

Per-op generated source summary:

| op | `kernel.cu` present | `.cu` files | `.so` files | attempt files |
| --- | --- | ---: | ---: | ---: |
| `torch_nn_functional_conv1d` | yes | 4 | 1 | 2 |
| `torch_nn_functional_embedding` | yes | 5 | 1 | 4 |
| `torch_nn_functional_grouped_mm` | yes | 2 | 0 | 1 |
| `torch_nn_functional_linear` | yes | 4 | 0 | 4 |
| `torch_nn_functional_pad` | yes | 3 | 0 | 201 |
| `torch_nn_functional_sigmoid` | yes | 3 | 0 | 2 |
| `torch_nn_functional_silu` | yes | 2 | 0 | 1 |
| `torch_nn_functional_softmax` | yes | 6 | 2 | 6 |
| `torch_nn_functional_softplus` | yes | 3 | 0 | 1 |

Representative generated-source paths:

- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_conv1d/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_embedding/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_grouped_mm/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_linear/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_pad/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_sigmoid/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_silu/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_softmax/kernel.cu`
- `kernels/projects/test_qwen - NVIDIA GB10/kernels/generated/individual_op_kernels/torch_nn_functional_softplus/kernel.cu`

### Benchmark rows and artifacts

Benchmark artifacts present:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/op_benchmarks.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/torch_baseline_cache.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/qwen_tps_compare.json`
- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/runtime_kernels/`

Observed benchmark artifact status:

- `op_benchmarks.json` timestamp: `2026-04-01T19:55:34.4052249410`
- `torch_baseline_cache.json` timestamp: `2026-04-01T19:55:34.4032498820`
- `qwen_tps_compare.json` timestamp: `2026-04-10T18:09:06.7057751910`
- `benchmarks/runtime_kernels/` exists but contains no files

Operator benchmark summary from `op_benchmarks.json`:

| op | benchmarked entries | micro winner | deployment winner | deployment safe winner | deployment strict pass |
| --- | ---: | --- | --- | --- | --- |
| `torch_nn_functional_conv1d` | 50 | optimized | optimized | optimized | true |
| `torch_nn_functional_embedding` | 2 | optimized | pytorch | pytorch | true |
| `torch_nn_functional_grouped_mm` | 50 | pytorch | pytorch | pytorch | false |
| `torch_nn_functional_linear` | 50 | pytorch | pytorch | pytorch | true |
| `torch_nn_functional_pad` | 50 | optimized | pytorch | pytorch | true |
| `torch_nn_functional_sigmoid` | 50 | pytorch | pytorch | pytorch | true |
| `torch_nn_functional_silu` | 50 | pytorch | pytorch | pytorch | true |
| `torch_nn_functional_softmax` | 50 | optimized | optimized | optimized | true |
| `torch_nn_functional_softplus` | 50 | pytorch | pytorch | pytorch | true |

Interpretation:

- At operator tier, only `conv1d` and `softmax` have an optimized deployment-safe winner.
- `grouped_mm` fails strict deployment correctness.
- `embedding`, `pad`, `linear`, `sigmoid`, `silu`, and `softplus` are not deployment-safe optimized winners.

End-to-end Qwen benchmark artifact present:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/qwen_tps_compare.json`

Important end-to-end observations:

- It compares `baseline`, `compiled`, and `forged`, not `eager`, `torch_compile`, and `kf_cast` under the paper harness.
- `compiled_vs_baseline.exact_generated_token_match = false`
- `forged_vs_baseline.exact_generated_token_match = false`
- `forged_vs_compiled.exact_generated_token_match = false`
- `forged.patch_stats` shows runtime fallback:
  - `torch_nn_functional_pad`: `fallback = 90`
  - `torch_nn_functional_silu`: `fallback = 2520`

### Existing `.cast` exports

Existing export:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/exports/test_qwen - NVIDIA GB10.cast`

Embedded export metadata found inside the `.cast`:

- `HEADER.json`
- `manifest.json`
- `checksums.sha256`

Header summary:

- `project_name = test_qwen - NVIDIA GB10`
- `optimized_op_count = 9`
- `has_precompiled = false`
- `weight_size_bytes = 0`

Manifest summary:

- `project_name = test_qwen - NVIDIA GB10`
- `model_class = ""`
- `model_init_args = {}`
- `weight_file = ""`
- `ops = 9`

Embedded `.cast` kernel provenance check:

For every op, the kernel bundled in the existing `.cast` matches a concrete tree kernel file, but not the current best tree node:

| op | exported tree kernel | exported node value ms | current best tree kernel | current best value ms |
| --- | --- | ---: | --- | ---: |
| `torch_nn_functional_conv1d` | `kernel_5.cu` | 0.024528961181640625 | `kernel_4.cu` | 0.006138560175895691 |
| `torch_nn_functional_embedding` | `kernel_9.cu` | 0.006320639848709107 | `kernel_14.cu` | 0.006234560012817383 |
| `torch_nn_functional_grouped_mm` | `kernel_9.cu` | 2.196890869140625 | `kernel_20.cu` | 0.43983070373535155 |
| `torch_nn_functional_linear` | `kernel_9.cu` | 0.007544320225715637 | `kernel_4.cu` | 0.005028480291366577 |
| `torch_nn_functional_pad` | `kernel_9.cu` | 0.005863040089607239 | `kernel_5.cu` | 0.004108799993991852 |
| `torch_nn_functional_sigmoid` | `kernel_5.cu` | 0.002985920011997223 | `kernel_1.cu` | 0.002056320011615753 |
| `torch_nn_functional_silu` | `kernel_9.cu` | 0.0043452799320220946 | `kernel_4.cu` | 0.002199999988079071 |
| `torch_nn_functional_softmax` | `kernel_5.cu` | 0.004101440012454986 | `kernel_4.cu` | 0.004084799885749817 |
| `torch_nn_functional_softplus` | `kernel_5.cu` | 0.003903680145740509 | `kernel_1.cu` | 0.0022067199647426605 |

Conclusion:

- The existing `.cast` is stale relative to the current best tree results.
- It does not embed the current fastest known tree kernels.

### Export/download metadata

Separate export/download metadata files were not found in the project root.

What exists:

- The `.cast` file itself
- Embedded `HEADER.json`
- Embedded `manifest.json`
- Embedded `checksums.sha256`

What was not found:

- no standalone export manifest beside the `.cast`
- no standalone download-selection manifest
- no export-time record of:
  - benchmark row ids
  - benchmark timestamps per exported op
  - benchmarked source hashes
  - exported source hashes linked to benchmark rows
  - deployment-safe winner selection rationale

## Missing Or Stale Artifacts

### Missing

- No immutable model snapshot path recorded in project JSON metadata
- No model weights artifact recorded in `config.json`
- Existing `.cast` has:
  - empty `weight_file`
  - empty `model_class`
  - no bundled weights
  - no precompiled binaries
- No `paper_benchmarks/runs/` Qwen artifacts exist yet
- `benchmarks/runtime_kernels/` is empty
- No export provenance manifest links exported kernels to benchmark rows and source hashes

### Stale or inconsistent

- Existing `.cast` embeds stale non-best tree kernels
- `qwen_tps_compare.json` is not paper-harness output
- `qwen_tps_compare.json` shows runtime fallback and non-exact output token matches
- `state.json` and `queue.json` still contain error/stale job state:
  - generate state marked `completed` with `result = error`
  - optimize state marked `error`
  - queue contains lingering completed and queued work entries

## Export Eligibility Assessment

### Are there benchmarked kernels eligible for export?

At operator tier, there is partial evidence:

- Yes, `torch_nn_functional_conv1d` has a deployment-safe optimized winner.
- Yes, `torch_nn_functional_softmax` has a deployment-safe optimized winner.

For the remaining ops:

- No deployment-safe optimized winner is present, or the safe winner is PyTorch:
  - `embedding`
  - `grouped_mm`
  - `linear`
  - `pad`
  - `sigmoid`
  - `silu`
  - `softplus`

Important constraint:

- Even for `conv1d` and `softmax`, the current artifacts do not fully satisfy strict export provenance requirements because the benchmark rows do not record source hashes for the exact exported kernel file, and the current `.cast` does not embed the current best kernels anyway.

### Is there enough evidence to auto-select fastest valid kernels?

No.

Reasons:

- The current `.cast` export is stale and does not match the current best tree kernels.
- Existing benchmark rows do not provide an explicit exported-source-hash-to-benchmarked-source-hash linkage.
- There is no standalone export selection manifest showing why each exported kernel was chosen.
- End-to-end Qwen runtime evidence shows fallback and non-exact output token matches.
- Several ops have `deployment_safe_winner = pytorch`, so a correct default export would need to skip those kernels rather than include them.

Under a strict policy of "best/fastest valid" meaning:

- correctness passed
- captured inputs exist
- benchmark timing is real and source-matched
- exported source hash matches benchmarked source hash
- no deployment-blocking error
- fastest median deployment-safe latency among eligible kernels

the current project artifacts are not sufficient to auto-select kernels safely.

## Blockers For Generating A Defensible `.cast`

Primary blockers:

- The current export is stale relative to the current tree best nodes.
- The project does not record an immutable model snapshot path in project metadata.
- The existing `.cast` is structurally incomplete for runtime loading:
  - `weight_file` is empty
  - `model_class` is empty
  - weights are not bundled
- Export provenance is missing:
  - no benchmark-to-export source hash linkage
  - no export selection manifest
  - no proof that embedded kernels are the exact benchmarked deployment-safe winners
- End-to-end Qwen runtime evidence is not paper-safe:
  - fallback occurred
  - exact generated token match is false

Secondary blockers:

- `grouped_mm` fails strict deployment correctness in operator benchmark results
- `pad` and `silu` show deployment/runtime issues in `qwen_tps_compare.json`
- `runtime_kernels` artifact directory is empty, so deployment-tier intermediate evidence is incomplete

## Conclusion

Canonical source project for Qwen CAST export:

- `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10`

Encoded reference mapping:

- `project/test_qwen%20-%20NVIDIA%20GB10/` maps to the decoded filesystem project root above

Current export readiness:

- This project does contain real captured entries, tree search results, generated kernels, benchmark artifacts, and an existing `.cast`.
- It does not yet contain enough trustworthy provenance to auto-select fastest valid kernels by default.
- The existing `.cast` should not be treated as the canonical paper/export artifact without rebuilding the export selection logic and re-establishing benchmark-to-export source matching.
