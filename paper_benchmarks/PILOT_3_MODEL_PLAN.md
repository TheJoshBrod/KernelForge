# Pilot 3-Model Plan

## Goal

Prepare the first defensible expansion beyond single-model Qwen benchmarking without jumping straight to a 10-model suite.

The pilot deliberately mixes:
- one MoE decoder LLM
- one dense decoder LLM
- one non-LLM conv-heavy model

This is not a win-seeking shortlist. The dense and conv-heavy entries are included specifically to make reviewer objections harder:
- "you only chose an MoE decoder where routing and grouped GEMMs favor your system"
- "you only evaluated autoregressive text generation"

## Selected Models

1. `qwen35a3b_pilot`
   Why included: primary MoE decoder target, already mirrored locally on this DGX Spark machine.

2. `llama31_8b_dense_pilot`
   Why included: dense decoder counterweight to Qwen; tests whether any gains survive outside MoE routing.

3. `resnet50_conv_vision_pilot`
   Why included: non-LLM conv-heavy coverage so the pilot is not all text-generation and not all transformer attention.

Config set:
- [registry.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/registry.yaml)
- [qwen35a3b_pilot.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/models/qwen35a3b_pilot.yaml)
- [llama31_8b_dense_pilot.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/models/llama31_8b_dense_pilot.yaml)
- [resnet50_conv_vision_pilot.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/models/resnet50_conv_vision_pilot.yaml)
- [qwen35a3b_pilot.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/suites/qwen35a3b_pilot.yaml)
- [llama31_8b_dense_pilot.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/suites/llama31_8b_dense_pilot.yaml)
- [resnet50_conv_vision_pilot.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/pilot_3_model/suites/resnet50_conv_vision_pilot.yaml)

## Current Execution State

What is runnable today with the current external harness:
- Qwen MoE decoder: yes, once frozen prompts and `.cast` path are provided.
- Dense decoder LLM: yes, once local snapshot, frozen prompts, and `.cast` path are provided.
- ResNet50 end-to-end: not yet. The current external harness is still causal-LM-centric for end-to-end execution. ResNet50 is included now as a preflight-complete pilot entry, and should remain in the set while the generic non-LLM runner is added.

What is runnable today for pilot validation:
- registry validation for all three entries
- LLM `preflight` once the frozen workload paths are filled
- operator-tier `run-ops` for conv-heavy captured entries if ResNet operator captures already exist

## Shared Pilot Policy

Applies to all 3 models:
- correctness comparator must be defined before any run counts as paper evidence
- eager baseline is always required
- `torch_compile` baseline is always required
- `kf_cast` is required for any Kernel Forge deployment claim
- frozen workload hash is required
- stale cache reuse is forbidden unless all hashes and runtime fields match
- warmup, compile, and load must stay out of steady-state latency
- missing workload, missing correctness comparator, or incomplete baseline requirements keep `paper_eligible=false`

## Model 1: Qwen MoE Decoder

Included because:
- it is the original paper target
- it exercises MoE routing, grouped GEMM, embedding, and decode-heavy LLM behavior
- the local model snapshot already exists at `/home/gb10/model-cache/Qwen3.5-35B-A3B`

What claim it can support:
- end-to-end LLM throughput claim against eager
- end-to-end LLM throughput claim against `torch_compile`
- Kernel Forge deployment claim for `.cast`
- prefill/decode split claim

Expected reviewer objection:
- "Qwen MoE is a favorable architecture for your kernels and does not show generality."

Required artifacts:
- frozen prompt suite file
- prompt suite hash
- Qwen model config hash
- eager results
- `torch_compile` results
- `.cast` package and hash
- `kf_cast` results with zero or explicitly reported fallback

What would invalidate the result:
- prompt suite changed after baseline collection
- exact token mismatch against eager for either `torch_compile` or `kf_cast`
- hidden fallback
- nonzero fallback under fail-on-fallback policy
- compile/load folded into steady-state timing
- missing raw samples

Batch and length matrix:
- batch sizes: `1`, `2`
- prompt buckets: `short` `1-128`, `medium` `129-512`, `long` `513-1024`
- `max_new_tokens: 128`
- `warmup_runs: 5`
- `timed_runs: 20`

Paper eligibility gates:
- `correctness_comparator: exact_token_match_against_eager`
- baselines: `eager`, `torch_compile`, `kf_cast`
- `expected_model_config_hash` pinned
- workload hash filled
- deployment artifact path filled and hashed

Exact run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/pilot_3_model/models/qwen35a3b_pilot.yaml \
  --suite-config paper_benchmarks/configs/pilot_3_model/suites/qwen35a3b_pilot.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/exported/qwen35a3b.cast \
  --out paper_benchmarks/runs/qwen35a3b_pilot \
  --fail-if-not-paper-eligible
```

Preflight command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli preflight \
  --model-config paper_benchmarks/configs/pilot_3_model/models/qwen35a3b_pilot.yaml \
  --suite-config paper_benchmarks/configs/pilot_3_model/suites/qwen35a3b_pilot.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/exported/qwen35a3b.cast
```

## Model 2: Dense Decoder LLM

Included because:
- it is the cleanest rebuttal to "your result is just MoE-specific"
- it keeps the pilot in the same causal-LM claim class while changing architecture
- it stays compatible with the current `run-llm` harness path once the local model snapshot exists

What claim it can support:
- end-to-end dense-decoder throughput claim against eager
- end-to-end dense-decoder throughput claim against `torch_compile`
- deployment claim if a `.cast` package exists

Expected reviewer objection:
- "You chose a dense model that still shares enough decode behavior with Qwen to favor the same hot-path kernels."

Required artifacts:
- local dense-decoder snapshot
- frozen prompt suite file
- workload hash
- eager baseline
- `torch_compile` baseline
- `.cast` package and hash for deployment claim

What would invalidate the result:
- prompt buckets changed after first collection
- missing `torch_compile` baseline
- exact token mismatch against eager
- hidden fallback in `kf_cast`
- using only easy prompt buckets after seeing results

Batch and length matrix:
- batch sizes: `1`, `4`
- prompt buckets: `short` `1-128`, `medium` `129-512`, `long` `513-1024`
- `max_new_tokens: 128`
- `warmup_runs: 5`
- `timed_runs: 20`

Paper eligibility gates:
- `correctness_comparator: exact_token_match_against_eager`
- baselines: `eager`, `torch_compile`, `kf_cast`
- workload hash filled
- deployment artifact path filled for any deployment claim

Exact run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/pilot_3_model/models/llama31_8b_dense_pilot.yaml \
  --suite-config paper_benchmarks/configs/pilot_3_model/suites/llama31_8b_dense_pilot.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/exported/llama31_8b.cast \
  --out paper_benchmarks/runs/llama31_8b_dense_pilot \
  --fail-if-not-paper-eligible
```

Preflight command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli preflight \
  --model-config paper_benchmarks/configs/pilot_3_model/models/llama31_8b_dense_pilot.yaml \
  --suite-config paper_benchmarks/configs/pilot_3_model/suites/llama31_8b_dense_pilot.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/exported/llama31_8b.cast
```

## Model 3: ResNet50 Conv/Vision

Included because:
- the pilot should not stay inside autoregressive text only
- conv-heavy vision is a plausible negative or neutral case and should remain in the set
- it gives reviewers a direct non-LLM counterexample if Kernel Forge wins are limited to LLM kernels

What claim it can support:
- today: preparedness and operator-tier conv-heavy evidence only
- later, after generic non-LLM end-to-end runner support: end-to-end vision throughput and deployment claims

Expected reviewer objection:
- "You added a vision model on paper, but your current end-to-end harness cannot run it yet."

Required artifacts:
- frozen image subset or frozen image-folder manifest
- dataset hash
- model weights/config hash
- eager baseline
- `torch_compile` baseline
- `.cast` package and hash if deployment is claimed

What would invalidate the result:
- changing the image subset after seeing throughput
- missing eager or `torch_compile`
- hidden fallback in any deployment path
- changing image resolution or batch matrix post-hoc

Batch and shape matrix:
- batch sizes: `1`, `32`
- shape buckets: `image_224`, `image_384`
- `warmup_runs: 20`
- `timed_runs: 20`

Paper eligibility gates:
- `correctness_comparator: logits_allclose_and_top1_exact_against_eager`
- baselines: `eager`, `torch_compile`, `kf_cast`
- workload hash filled
- generic non-LLM runner available before any end-to-end paper claim

Current pilot command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-model-registry \
  --registry paper_benchmarks/configs/pilot_3_model/registry.yaml
```

Current operator-tier command if ResNet50 captures already exist:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-ops \
  --entries-dir /abs/path/to/resnet50_captured_entries \
  --op aten.convolution \
  --kernel-source-or-cast /abs/path/to/resnet50.cast \
  --variants eager torch_compile kf_cast \
  --out paper_benchmarks/runs/resnet50_conv_operator_pilot \
  --fail-if-not-paper-eligible
```

Why this is still valid to include now:
- the pilot is about freezing the first 3-model set, not hiding models that need one more harness step
- dropping the non-LLM entry until after seeing Qwen/Llama results would be a subtler form of cherry-picking

## Pilot Outcome Policy

Allowed pilot outcomes:
- Qwen win, dense neutral, ResNet unresolved
- Qwen neutral, dense neutral, ResNet unresolved
- Qwen regression, dense neutral, ResNet unresolved

Not allowed:
- removing dense or conv-heavy entries because Qwen looks strong
- replacing ResNet50 with an easier vision model after seeing early results
- claiming pilot generality from Qwen alone

## Current Validation Command

Use this now:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-model-registry \
  --registry paper_benchmarks/configs/pilot_3_model/registry.yaml
```

This should report the pilot entries as configuration-complete but still non-paper until the workload files, hashes, and deployment artifacts are filled.
