# ICLR 10-Model Benchmark Plan

## Scope

This document defines the first full 10-model paper suite for Kernel Forge benchmarking.

Rules:
- do not remove neutral or regression-capable models after seeing early results
- do not collapse the suite to only autoregressive LLMs
- do not turn operator-only or partial deployment evidence into end-to-end model claims
- incomplete entries remain `paper_eligible: false` until model paths, frozen workloads, hashes, and deployment artifacts are filled

Registry and template set:
- [registry.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/registry.yaml)
- models under [configs/model_suite_10/models](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/models)
- suites under [configs/model_suite_10/suites](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites)

## Suite-Level Policy

Required baselines for every model:
- `eager`
- `torch_compile`
- `kf_cast`

Required paper evidence for every end-to-end claim:
- frozen workload path
- workload hash
- exact environment provenance
- raw latency samples
- correctness evidence against eager
- separated load, compile, warmup, and steady-state timing
- explicit deployment artifact path and hash for any deployment claim

Negative-capable coverage intentionally included:
- transformer encoder
- vision transformer
- conv/vision
- embedding-heavy
- dynamic-shape workload

## 1. MoE Decoder LLM

Model category:
- MoE decoder LLM

Exact model id/path placeholder:
- model id: `qwen35a3b_iclr`
- model path: `/abs/path/to/Qwen3.5-35B-A3B`

Workload source:
- frozen prompt pack from real instruction/chat prompts

Prompt/data suite:
- [01_moe_decoder_llm_qwen_style.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/01_moe_decoder_llm_qwen_style.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/qwen35a3b.cast`

Correctness comparator:
- `exact_token_match_against_eager`

Metrics:
- prefill latency
- decode latency
- total generate latency
- total TPS
- prefill TPS
- decode TPS
- speedup vs eager
- speedup vs torch.compile
- fallback count
- kernel hit count

Batch/shape/length matrix:
- batch sizes: `1`
- prompt buckets: `short 1-128`, `medium 129-512`
- `max_new_tokens: 128`

Expected memory risk:
- very high; MoE routing plus batch growth can exceed device memory quickly

Paper eligibility gates:
- exact token equality on every timed run
- frozen prompt hash recorded
- model config hash recorded
- `.cast` hash recorded
- zero fallback unless explicitly downgraded to non-paper

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/01_moe_decoder_llm_qwen_style.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/01_moe_decoder_llm_qwen_style.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/qwen35a3b.cast \
  --out paper_benchmarks/runs/01_moe_decoder_llm_qwen_style \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "This is a favorable MoE target; the suite needs dense and non-LLM counterexamples."

## 2. Dense Decoder LLM

Model category:
- Dense decoder LLM

Exact model id/path placeholder:
- model id: `llama31_8b_instruct_iclr`
- model path: `/abs/path/to/Meta-Llama-3.1-8B-Instruct`

Workload source:
- frozen instruction/chat prompt pack, selected before benchmarking

Prompt/data suite:
- [02_dense_decoder_llm.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/02_dense_decoder_llm.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/llama31_8b.cast`

Correctness comparator:
- `exact_token_match_against_eager`

Metrics:
- same LLM metrics as model 1

Batch/shape/length matrix:
- batch sizes: `1`
- prompt buckets: `short 1-128`, `long 129-1024`
- `max_new_tokens: 128`

Expected memory risk:
- moderate; lower than Qwen 35B, but long-prompt decode still stresses memory and compile behavior

Paper eligibility gates:
- same as model 1

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/02_dense_decoder_llm.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/02_dense_decoder_llm.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/llama31_8b.cast \
  --out paper_benchmarks/runs/02_dense_decoder_llm \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "This is still autoregressive generation; show non-generation and non-LLM behavior too."

## 3. Transformer Encoder Model

Model category:
- Transformer encoder model

Exact model id/path placeholder:
- model id: `bert_large_uncased_iclr`
- model path: `/abs/path/to/bert-large-uncased`

Workload source:
- frozen tokenized encoder input corpus from real classification or retrieval text

Prompt/data suite:
- [03_transformer_encoder.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/03_transformer_encoder.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/bert_large_uncased.cast`

Correctness comparator:
- `hidden_state_allclose_against_eager`

Metrics:
- load
- compile
- warmup
- encoder steady-state latency
- tokens or examples per second
- speedup vs eager
- speedup vs torch.compile

Batch/shape/length matrix:
- batch sizes: `1`, `8`
- sequence buckets: `seq_128`, `seq_512`

Expected memory risk:
- moderate; encoder attention can still spike at larger batch and longer sequence length

Paper eligibility gates:
- hidden states within comparator threshold against eager
- frozen corpus hash recorded
- deployment artifact hash recorded if `.cast` is used

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/03_transformer_encoder.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/03_transformer_encoder.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/bert_large_uncased.cast \
  --out paper_benchmarks/runs/03_transformer_encoder \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Your current external runner is still causal-LM-centric; verify encoder execution before treating this as paper-ready."

## 4. Vision Transformer

Model category:
- Vision transformer

Exact model id/path placeholder:
- model id: `vit_base_patch16_224_iclr`
- model path: `/abs/path/to/vit_base_patch16_224`

Workload source:
- frozen image manifest from a real validation subset

Prompt/data suite:
- [04_vision_transformer.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/04_vision_transformer.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/vit_base_patch16_224.cast`

Correctness comparator:
- `logits_allclose_against_eager`

Metrics:
- load
- compile
- warmup
- steady-state image latency
- images/sec
- speedup vs eager
- speedup vs torch.compile

Batch/shape/length matrix:
- batch sizes: `1`, `16`
- image buckets: `224`, `384`

Expected memory risk:
- moderate to high at larger resolution and batch combinations

Paper eligibility gates:
- frozen image manifest hash
- identical preprocessing pipeline across variants
- logits within comparator threshold

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/04_vision_transformer.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/04_vision_transformer.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/vit_base_patch16_224.cast \
  --out paper_benchmarks/runs/04_vision_transformer \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "You included ViT, but did you keep the image set and resize policy frozen across all variants?"

## 5. Conv/Vision Model

Model category:
- Conv/vision model

Exact model id/path placeholder:
- model id: `resnet50_imagenet_iclr`
- model path: `/abs/path/to/resnet50_imagenet`

Workload source:
- frozen real-image subset with fixed crop/resize preprocessing

Prompt/data suite:
- [05_conv_vision.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/05_conv_vision.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/resnet50.cast`

Correctness comparator:
- `logits_allclose_against_eager`

Metrics:
- same image-model metrics as model 4

Batch/shape/length matrix:
- batch sizes: `1`, `32`
- image bucket: `224`

Expected memory risk:
- lower than ViT on activations, but workspace and batch-32 pressure can still matter

Paper eligibility gates:
- frozen image-set hash
- exact preprocessing parity
- deployment artifact hash if `.cast` used

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/05_conv_vision.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/05_conv_vision.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/resnet50.cast \
  --out paper_benchmarks/runs/05_conv_vision \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Conv-heavy models are a plausible neutral or regression case; did you keep it after seeing the outcome?"

## 6. Seq2seq / Encoder-Decoder Model

Model category:
- Seq2seq or encoder-decoder model

Exact model id/path placeholder:
- model id: `t5_large_iclr`
- model path: `/abs/path/to/t5-large`

Workload source:
- frozen source-target corpus from a real summarization or translation validation set

Prompt/data suite:
- [06_seq2seq_encoder_decoder.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/06_seq2seq_encoder_decoder.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/t5_large.cast`

Correctness comparator:
- `exact_token_match_against_eager`

Metrics:
- encoder prefill
- decode
- total generate
- tokens/sec
- speedups vs eager and torch.compile

Batch/shape/length matrix:
- batch size: `1`
- source/target buckets: `src_short_tgt_short`, `src_long_tgt_medium`
- `max_new_tokens: 128`

Expected memory risk:
- high decode variance and dual-graph memory pressure from encoder plus decoder

Paper eligibility gates:
- exact output token equality against eager
- frozen source-target suite hash
- `.cast` artifact hash

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/06_seq2seq_encoder_decoder.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/06_seq2seq_encoder_decoder.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/t5_large.cast \
  --out paper_benchmarks/runs/06_seq2seq_encoder_decoder \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Did you benchmark real encoder-decoder traffic or simplify the task into decoder-only text generation?"

## 7. Embedding-Heavy Model

Model category:
- Embedding-heavy model

Exact model id/path placeholder:
- model id: `dlrmv2_embedding_iclr`
- model path: `/abs/path/to/dlrmv2_or_equivalent`

Workload source:
- frozen sparse-feature batches from a real recommendation or ranking validation slice

Prompt/data suite:
- [07_embedding_heavy.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/07_embedding_heavy.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/dlrmv2.cast`

Correctness comparator:
- `embedding_cosine_similarity_against_eager`

Metrics:
- load
- compile
- warmup
- steady-state batch latency
- examples/sec
- speedups vs eager and torch.compile

Batch/shape/length matrix:
- batch sizes: `1`, `64`
- sequence buckets: `seq_64`, `seq_256`

Expected memory risk:
- very high table residency risk and lookup working-set pressure

Paper eligibility gates:
- frozen sparse input distribution
- embedding output comparator against eager
- deployment artifact path/hash for any deployment claim

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/07_embedding_heavy.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/07_embedding_heavy.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/dlrmv2.cast \
  --out paper_benchmarks/runs/07_embedding_heavy \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Sparse embedding traffic is usually a hard case; did you keep it even if it was neutral or worse?"

## 8. Dynamic-Shape Workload

Model category:
- Dynamic-shape workload

Exact model id/path placeholder:
- model id: `dynamic_shape_lm_iclr`
- model path: `/abs/path/to/dynamic_shape_lm`

Workload source:
- frozen variable-length prompt pack explicitly designed to force shape churn

Prompt/data suite:
- [08_dynamic_shape_workload.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/08_dynamic_shape_workload.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/dynamic_shape_lm.cast`

Correctness comparator:
- `exact_token_match_against_eager`

Metrics:
- same LLM metrics as models 1 and 2
- plus compile churn and fallback sensitivity

Batch/shape/length matrix:
- batch size: `1`
- buckets: `short_dynamic 1-64`, `long_dynamic 65-1024`
- `max_new_tokens: 128`

Expected memory risk:
- medium; the main risk is not peak memory but shape-polymorphic compile and fallback instability

Paper eligibility gates:
- exact token equality
- frozen variable-length suite hash
- hidden fallback forbidden

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/08_dynamic_shape_workload.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/08_dynamic_shape_workload.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/dynamic_shape_lm.cast \
  --out paper_benchmarks/runs/08_dynamic_shape_workload \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Did you include a truly hard dynamic-shape case, or only shape-stable prompts where compilation wins are easy?"

## 9. Batch-1 Latency Workload

Model category:
- Batch-1 latency workload

Exact model id/path placeholder:
- model id: `batch1_latency_lm_iclr`
- model path: `/abs/path/to/batch1_latency_lm`

Workload source:
- frozen real prompt pack tuned for single-request latency, not throughput

Prompt/data suite:
- [09_batch1_latency_workload.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/09_batch1_latency_workload.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/batch1_latency_lm.cast`

Correctness comparator:
- `exact_token_match_against_eager`

Metrics:
- p50/p95 latency
- per-token latency
- prefill/decode split
- speedup vs eager
- speedup vs torch.compile

Batch/shape/length matrix:
- batch size: `1`
- buckets: `latency_short 1-128`, `latency_medium 129-512`
- `max_new_tokens: 64`

Expected memory risk:
- lower memory risk, higher kernel launch and dispatch sensitivity

Paper eligibility gates:
- exact token equality
- frozen latency workload hash
- zero hidden fallback

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/09_batch1_latency_workload.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/09_batch1_latency_workload.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/batch1_latency_lm.cast \
  --out paper_benchmarks/runs/09_batch1_latency_workload \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Batch-1 latency is often less favorable than throughput; did you keep it even if the result was neutral?"

## 10. Batched Throughput Workload

Model category:
- Batched throughput workload

Exact model id/path placeholder:
- model id: `batched_throughput_lm_iclr`
- model path: `/abs/path/to/batched_throughput_lm`

Workload source:
- frozen batched prompt pack selected before any benchmark result is seen

Prompt/data suite:
- [10_batched_throughput_workload.yaml](/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/configs/model_suite_10/suites/10_batched_throughput_workload.yaml)

Baseline variants:
- `eager`
- `torch_compile`
- `kf_cast`

Kernel Forge artifact requirement:
- `/abs/path/to/batched_throughput_lm.cast`

Correctness comparator:
- `exact_token_match_against_eager`

Metrics:
- same LLM metrics as models 1 and 2
- throughput saturation behavior

Batch/shape/length matrix:
- batch sizes: `8`, `16`
- buckets: `throughput_medium 64-256`, `throughput_long 257-1024`
- `max_new_tokens: 128`

Expected memory risk:
- high; throughput batches can hit HBM limits and make deployment unstable or OOM-prone

Paper eligibility gates:
- exact token equality
- frozen batch schedule
- frozen prompt hash
- `.cast` path/hash
- fallback zero or explicitly non-paper

Run command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/model_suite_10/models/10_batched_throughput_workload.yaml \
  --suite-config paper_benchmarks/configs/model_suite_10/suites/10_batched_throughput_workload.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /abs/path/to/batched_throughput_lm.cast \
  --out paper_benchmarks/runs/10_batched_throughput_workload \
  --fail-if-not-paper-eligible
```

Likely reviewer objection:
- "Did you choose only favorable high-throughput batch sizes, or did you freeze the full batch schedule up front?"

## Execution Status

What this plan means today:
- registry coverage for all 10 models exists
- all 10 entries remain non-paper until paths, workloads, hashes, and artifacts are filled
- the suite intentionally includes neutral and regression-capable categories
- no full model benchmarks are run by this plan

Current validation command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-model-registry \
  --registry paper_benchmarks/configs/model_suite_10/registry.yaml
```
