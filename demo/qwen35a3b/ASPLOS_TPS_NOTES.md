# Qwen 3.5 35B TPS Validation Notes

This is an internal end-to-end validation note for the Qwen demo harness in
`demo/qwen35a3b/bench_qwen_tps.py`. It is not part of the core Kernel Forge
pipeline.

## Setup

- Project: `test_qwen - NVIDIA GB10`
- Model snapshot: `/home/gb10/model-cache/Qwen3.5-35B-A3B`
- Device: `cuda` on `NVIDIA GB10`
- Prompt set: project validation prompts, `prompt_selection=longest`
- Batch size: `1`
- Prompt truncation: `512` tokens
- Decode length: `32` greedy tokens
- Warmup: `1` run with `4` decode tokens
- Timed runs: `3`, report median

Important caveat:

- The current text-only wrapper loads the checkpoint through
  `Qwen3_5MoeForConditionalGeneration` and emits a warning that many `mtp.*`
  weights are unused. Relative comparisons below were run under the same load
  path, but this warning should be kept in mind for any public claim.

## Primary Results

### Embedding Only

This is the cleanest correctness-preserving run.

- Forged ops: `torch_nn_functional_embedding`
- Patch stats: `100/100` kernel hits, `0` fallbacks
- Exact generated token match: `true`
- Baseline generated tokens/s: `14.0129`
- Forged generated tokens/s: `13.9994`
- Generated tokens/s speedup: `0.9990x`
- Baseline total tokens/s: `36.3460`
- Forged total tokens/s: `36.3109`
- Decode speedup: `1.0028x`

Interpretation:

- This is effectively neutral. It is correctness-preserving, but there is no
  meaningful end-to-end TPS gain.

### Softmax Only

This is the only subset that produced a measurable TPS improvement on the
longer run.

- Forged ops: `torch_nn_functional_softmax`
- Patch stats: `4000/4000` kernel hits, `0` fallbacks
- Exact generated token match: `false`
- Baseline generated tokens/s: `13.8389`
- Forged generated tokens/s: `14.1452`
- Generated tokens/s speedup: `1.0221x`
- Baseline total tokens/s: `35.8946`
- Forged total tokens/s: `36.6890`
- Decode speedup: `1.0306x`
- Prefill speedup: `0.9774x`

Interpretation:

- This is a real TPS gain on the tested path, but it is not safe to present as
  a correctness-preserving end-to-end result because the greedy decode output
  diverges over `32` tokens.

## Screening Results

These smaller runs used `max_input_length=128`, `max_new_tokens=2`, and
`timed_runs=1` to screen candidate subsets before the longer comparisons.

### GUI Winner Set

- Forged ops: `conv1d + embedding + softmax`
- Exact generated token match: `true`
- Baseline generated tokens/s: `5.5736`
- Forged generated tokens/s: `5.0611`
- Speedup: `0.8886x`
- `conv1d` only hit `60/120` calls and fell back for the rest

### Best Tree `grouped_mm` Added

- Forged ops: `conv1d + embedding + softmax + grouped_mm`
- `grouped_mm` source: `trees/torch_nn_functional_grouped_mm/kernels/kernel_1.cu`
- Exact generated token match: `true`
- Baseline generated tokens/s: `5.5736`
- Forged generated tokens/s: `0.8596`
- Speedup: `0.1542x`
- `grouped_mm` hit `320/320` calls with no fallback

Interpretation:

- The optimized `grouped_mm` tree kernel is faster in isolated microbenchmarks,
  but it is a severe end-to-end regression on this Qwen runtime path.

## Bottom Line

For this setup, there is currently no correctness-preserving end-to-end Qwen TPS
win that is strong enough to present cleanly at ASPLOS:

- `embedding` is correctness-preserving but neutral
- `softmax` improves TPS slightly but changes the decoded output
- the current combined winner set regresses
- adding the best available `grouped_mm` tree kernel regresses sharply

The honest ASPLOS-safe claim today is:

- Kernel Forge is finding real per-operator wins in the GUI
- but those wins do not yet translate into a correctness-preserving end-to-end
  tokens/s improvement on this Qwen 3.5 35B inference path
