# Paper Benchmark Report for qwen35a3b_smoke_compile

## Operator Benchmark

No rows.

## End-to-End Model Benchmark

No rows.

## Deployment/Runtime Benchmark

No rows.

## Offline Costs

| Variant | Stage | Group | P05 ms | Median ms | Mean ms | P95 ms | Total TPS | Prefill TPS | Decode TPS | Speedup vs eager | Speedup vs torch.compile | Correctness | Prompt count | Prompt suite hash | Token equality | Fallbacks | Kernel hits | Artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|
| eager | load | - | 74408.77146099228 | 74408.77146099228 | 74408.77146099228 | 74408.77146099228 | n/a | n/a | n/a | 1.0000 | n/a | reference | - | bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93 | - | - | - | /home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_compile/metrics/eager_load.json |

## Correctness

No items.

## Coverage

- `/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_compile/metrics/eager_load.json`: {"artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_compile/metrics/eager_load.json", "comparison_group": null, "coverage": {}, "prompt_count": null, "prompt_suite_hash": "bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93", "stage": "load", "variant": "eager"}

## Export/CAST Selection

No items.

## Failures/Regressions

No items.

## Paper-Eligible Claims

No items.

## Forbidden/Unsupported Claims

No items.
