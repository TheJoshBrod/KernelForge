# Paper Benchmark Report for qwen35a3b_smoke_eager

## Operator Benchmark

No rows.

## End-to-End Model Benchmark

No rows.

## Deployment/Runtime Benchmark

No rows.

## Offline Costs

| Variant | Stage | Group | P05 ms | Median ms | Mean ms | P95 ms | Total TPS | Prefill TPS | Decode TPS | Speedup vs eager | Speedup vs torch.compile | Correctness | Claim status | Claim category | Prompt count | Prompt suite hash | Token equality | Fallbacks | Kernel hits | Artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---:|---:|---|
| eager | load | - | 19495.19654805772 | 19495.19654805772 | 19495.19654805772 | 19495.19654805772 | n/a | n/a | n/a | 1.0000 | n/a | reference | paper_eligible | baseline | - | bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93 | - | - | - | /home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_eager/metrics/eager_load.json |
| eager | warmup | short_bs1 | 387151.8709177617 | 387151.8709177617 | 387151.8709177617 | 387151.8709177617 | n/a | n/a | n/a | 1.0000 | n/a | reference | paper_eligible | baseline | 2 | bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93 | - | - | - | /home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_eager/metrics/eager_warmup__short_bs1.json |

## Correctness

No items.

## Coverage

- `/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_eager/metrics/eager_load.json`: {"artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_eager/metrics/eager_load.json", "comparison_group": null, "coverage": {}, "prompt_count": null, "prompt_suite_hash": "bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93", "stage": "load", "variant": "eager"}
- `/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_eager/metrics/eager_warmup__short_bs1.json`: {"artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/runs/qwen35a3b_smoke_eager/metrics/eager_warmup__short_bs1.json", "comparison_group": "short_bs1", "coverage": {}, "prompt_count": 2, "prompt_suite_hash": "bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93", "stage": "warmup", "variant": "eager"}

## Export/CAST Selection

No items.

## Failures/Regressions

No items.

## Paper-Eligible Claims

No items.

## Forbidden/Unsupported Claims

No items.
