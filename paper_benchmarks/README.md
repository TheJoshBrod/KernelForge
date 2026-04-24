# Paper Benchmarks

This package contains the standalone paper-grade benchmark harness for Kernel Forge.

Entry point:

```bash
python -m paper_benchmarks.paper_bench.cli --help
```

Primary goals:
- keep paper benchmarking separate from the internal operator-ranking pipeline
- enforce frozen workloads and provenance capture
- compare `eager`, `torch_compile`, and `kf_cast` without silently hiding fallback
- record raw latency samples, correctness state, hashes, and environment metadata

Key commands:
- `preflight`
- `run-llm`
- `run-ops`
- `validate-artifact`
- `summarize`

Config files:
- models: `paper_benchmarks/configs/models/*.yaml`
- suites: `paper_benchmarks/configs/suites/*.yaml`

LLM model configs support standalone files such as
`paper_benchmarks/configs/models/qwen35a3b.yaml`, with fields for:
- `model_id`
- `model_path`
- `tokenizer_path`
- `dtype`
- `device_map` or `device`
- `trust_remote_code`
- `max_memory`
- `local_files_only`
- `attn_implementation`
- `expected_model_config_hash`

LLM suite configs support standalone files such as
`paper_benchmarks/configs/suites/qwen_llm_paper.yaml`, with fields for:
- `prompt_file`
- `prompt_length_buckets`
- `batch_sizes`
- `max_new_tokens`
- `warmup_runs`
- `timed_runs`
- `generation_mode`
- `include_tokenization_in_timing`
- `measure_prefill_decode_separately`

Run outputs are written under:

```text
paper_benchmarks/runs/<timestamp>_<model_id>_<suite_id>/
```

Canonical model-level data collection is documented in:

```text
paper_benchmarks/data_collection/docs/README.md
```

That contract keeps one append-only JSONL file per model, with all arms
(`zero_shot`, `optimize_5`, `optimize_10`, `optimize_20`, `optimize_50`) in the
same file and linked to the exact cast export, selected kernels, token usage,
internal Forge benchmarks, and later external benchmark results.

The scaffold supports real workload execution where possible and explicit synthetic demo runs only when `--allow-synthetic-demo` is passed.

Examples:

Eager-only smoke:

```bash
python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/models/qwen35a3b.yaml \
  --suite-config paper_benchmarks/configs/suites/qwen_llm_paper.yaml \
  --variants eager \
  --out paper_benchmarks/runs/qwen35a3b_smoke
```

Eager + `torch.compile`:

```bash
python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/models/qwen35a3b.yaml \
  --suite-config paper_benchmarks/configs/suites/qwen_llm_paper.yaml \
  --variants eager torch_compile \
  --out paper_benchmarks/runs/qwen35a3b_compile
```

Eager + `torch.compile` + Kernel Forge deployment:

```bash
python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/models/qwen35a3b.yaml \
  --suite-config paper_benchmarks/configs/suites/qwen_llm_paper.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /path/to/qwen35a3b.cast \
  --out paper_benchmarks/runs/qwen35a3b_deployment
```

Full Qwen run on DGX Spark:

```bash
python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config paper_benchmarks/configs/models/qwen35a3b.yaml \
  --suite-config paper_benchmarks/configs/suites/qwen_llm_paper.yaml \
  --variants eager torch_compile kf_cast \
  --cast-package /path/to/qwen35a3b.cast \
  --out paper_benchmarks/runs/qwen35a3b_paper \
  --fail-if-not-paper-eligible
```

Captured operator benchmark:

```bash
python -m paper_benchmarks.paper_bench.cli run-ops \
  --entries-dir /path/to/captured/op_entries \
  --op aten.softmax \
  --kernel-source-or-cast /path/to/kernel_or_cast \
  --variants eager torch_compile kf_cast \
  --out paper_benchmarks/runs/op_softmax_paper \
  --fail-if-not-paper-eligible
```
