# Model Suite Expansion Plan

This directory defines the expansion framework for moving from a single Qwen benchmark target to a 10-model paper suite.

The goal is coverage, not cherry-picking. Do not remove a model because Kernel Forge is neutral, slower, harder to compile, or harder to deploy. Neutral and regression outcomes must remain in the reporting set once the suite is frozen.

## Categories

The 10 planned categories are:

1. MoE decoder LLM
2. Dense decoder LLM
3. Transformer encoder model
4. Vision transformer
5. Conv/vision model
6. Seq2seq or encoder-decoder model
7. Embedding-heavy model
8. Dynamic-shape workload
9. Batch-1 latency workload
10. Batched throughput workload

## Layout

- `registry.yaml`
  The top-level registry for the 10-model plan.
- `models/*.yaml`
  Per-model config templates.
- `suites/*.yaml`
  Per-workload suite templates.

## Policy

- Every entry defaults to `paper_eligible: false`.
- A model is not paper-ready until all required fields are filled and validated.
- Missing workload files, missing correctness comparators, or incomplete baseline requirements must leave the entry non-paper.
- `torch_compile` must be present in the baseline requirements for model-level claims.
- Deployment artifacts are optional at template time, but operator-only or baseline-only results must not be mislabeled as end-to-end deployment wins.

## Validation

Run the validator with:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-model-registry \
  --registry paper_benchmarks/configs/model_suite_10/registry.yaml
```

The validator reports:
- missing required fields
- workload file presence
- declared vs actual workload hash
- baseline requirement completeness
- declared vs effective paper eligibility
- neutral/regression/candidate expectation labels

## Freezing Rules

- Freeze model selection before benchmarking.
- Freeze workload selection before benchmarking.
- Do not drop negative models after seeing results.
- Do not turn operator-only wins into model-level wins.
- Report neutral and regression models alongside wins.
