# Pilot 3-Model Expansion

This directory defines the first pilot expansion beyond single-model Qwen benchmarking.

The pilot is intentionally not win-seeking. It covers:
- one MoE decoder LLM
- one dense decoder LLM
- one non-LLM conv/vision model

Policy:
- all entries default to `paper_eligible: false`
- no model becomes paper-ready until the frozen workload, workload hash, and baseline requirements are filled
- negative, neutral, and regression outcomes remain in scope
- the non-LLM model is included now even though the current end-to-end runner is still causal-LM-centric

Validation:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-model-registry \
  --registry paper_benchmarks/configs/pilot_3_model/registry.yaml
```
