# Qwen CAST Export Report

- Project ref: `project/test_qwen%20-%20NVIDIA%20GB10/`
- Resolved project root: `/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10`
- Selection policy: `auto_best_fastest_valid`
- Exported CAST: `/home/gb10/Projects/Kernal-Forge/CGinS/paper_benchmarks/artifacts/qwen35a3b/qwen35a3b_auto_best_fastest_valid_d43a716_2026-04-21.cast`
- CAST SHA256: `31b5f4dc79c160f8ac290b84e09487067c1f3b62ee0e83c02e208e9a42f3ea95`
- Deployment-paper eligible: `False`
- All selected kernels deployment-paper eligible: `True`
- Used operator/micro-only evidence: `False`
- Target SM: `sm_121`
- GPU: `NVIDIA GB10`

## Selected Ops
- `torch_nn_functional_conv1d`: `torch_nn_functional_conv1d:deployment` tier=`deployment` hash=`5a7fffcce6ed19e280cdb51215beed8d43c867303fc3a03d2761cc38878be4f9`
- `torch_nn_functional_softmax`: `torch_nn_functional_softmax:deployment` tier=`deployment` hash=`62b2ab62d02126c4992be7f455d2eff18f4c98e09b4450b8cb91b6b532db7028`

## Rejected Candidates
- `torch_nn_functional_conv1d`: total=0 reasons={}
- `torch_nn_functional_embedding`: total=2 reasons={"baseline faster than kernel": 1, "stale benchmark row": 2}
- `torch_nn_functional_grouped_mm`: total=2 reasons={"baseline faster than kernel": 2, "correctness failed": 2, "micro-only evidence not allowed": 1, "stale benchmark row": 1}
- `torch_nn_functional_linear`: total=2 reasons={"baseline faster than kernel": 2, "correctness failed": 1, "micro-only evidence not allowed": 1}
- `torch_nn_functional_pad`: total=2 reasons={"baseline faster than kernel": 1, "deployment/runtime audit failed": 2, "stale benchmark row": 2, "unsafe kernel blocked": 2}
- `torch_nn_functional_sigmoid`: total=2 reasons={"baseline faster than kernel": 2, "correctness failed": 1, "micro-only evidence not allowed": 1}
- `torch_nn_functional_silu`: total=2 reasons={"baseline faster than kernel": 2, "correctness failed": 1, "deployment/runtime audit failed": 1, "micro-only evidence not allowed": 1, "unsafe kernel blocked": 1}
- `torch_nn_functional_softmax`: total=0 reasons={}
- `torch_nn_functional_softplus`: total=2 reasons={"baseline faster than kernel": 2, "correctness failed": 1, "micro-only evidence not allowed": 1}

## Preflight
- Checksum verified: `True`
- Loadability blockers: `weight_file missing from manifest; model_class missing from manifest; model_config.json missing and model_init_args empty`

## Manifest Summary
```json
{
  "export_paper_eligible": false,
  "model_class": "",
  "model_entrypoints": {
    "build_model": true,
    "load_weights": true,
    "sample_inputs": true
  },
  "project_ref": "project/test_qwen%20-%20NVIDIA%20GB10/",
  "project_root": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10",
  "selected_kernel_metadata": {
    "torch_nn_functional_conv1d": {
      "benchmark_artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/op_benchmarks.json",
      "benchmark_reference": {
        "artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/op_benchmarks.json",
        "audit_artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/qwen_tps_compare.json",
        "row_ref": "results[0]"
      },
      "benchmark_row_ref": "results[0]",
      "candidate_id": "torch_nn_functional_conv1d:deployment",
      "evidence_tier": "deployment",
      "kernel_source_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_conv1d/kernels/kernel_4.cu",
      "kernel_source_repo_relpath": "trees/torch_nn_functional_conv1d/kernels/kernel_4.cu",
      "selected_source_hash": "5a7fffcce6ed19e280cdb51215beed8d43c867303fc3a03d2761cc38878be4f9",
      "selection_reason": "auto_best_fastest_valid: selected deployment candidate with median 0.009846 ms"
    },
    "torch_nn_functional_softmax": {
      "benchmark_artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/op_benchmarks.json",
      "benchmark_reference": {
        "artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/op_benchmarks.json",
        "audit_artifact_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/benchmarks/qwen_tps_compare.json",
        "row_ref": "results[7]"
      },
      "benchmark_row_ref": "results[7]",
      "candidate_id": "torch_nn_functional_softmax:deployment",
      "evidence_tier": "deployment",
      "kernel_source_path": "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/trees/torch_nn_functional_softmax/kernels/kernel_4.cu",
      "kernel_source_repo_relpath": "trees/torch_nn_functional_softmax/kernels/kernel_4.cu",
      "selected_source_hash": "62b2ab62d02126c4992be7f455d2eff18f4c98e09b4450b8cb91b6b532db7028",
      "selection_reason": "auto_best_fastest_valid: selected deployment candidate with median 0.007075 ms"
    }
  },
  "selected_ops": [
    "torch_nn_functional_conv1d",
    "torch_nn_functional_softmax"
  ],
  "weight_file": ""
}
```
