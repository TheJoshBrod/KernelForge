# Qwen Paper Benchmark Runbook on DGX Spark / NVIDIA GB10

This runbook is specific to the Kernel Forge Qwen paper workflow in this branch.

Non-negotiable workflow rules:
- Canonical project and export source: `project/test_qwen%20-%20NVIDIA%20GB10/`
- Default CAST export policy: `auto_best_fastest_valid`
- Do not export Qwen `.cast` from another project unless this project cannot be resolved
- Do not use synthetic prompts for paper runs
- Do not claim a Qwen win unless the generated report explicitly allows that claim

All commands below assume the repo root is `/home/gb10/Projects/Kernal-Forge/CGinS` and the working Python is `./.venv/bin/python`.

## Common Variables

```bash
export PY=./.venv/bin/python
export PROJECT_REF='project/test_qwen%20-%20NVIDIA%20GB10/'
export MODEL_CONFIG=paper_benchmarks/configs/models/qwen35a3b.yaml
export PAPER_SUITE=paper_benchmarks/configs/suites/qwen_llm_paper.yaml
export SMOKE_SUITE=paper_benchmarks/configs/suites/qwen_llm_smoke.yaml
export ARTIFACT_DIR=paper_benchmarks/artifacts/qwen35a3b
export PAPER_RUN_DIR=paper_benchmarks/runs/qwen35a3b_dgx_spark_paper
export SMOKE_RUN_DIR=paper_benchmarks/runs/qwen35a3b_dgx_spark_smoke
```

Current frozen paper suite facts:
- Prompt suite: `paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.jsonl`
- Prompt suite hash: `bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93`
- Prompt buckets: `short=1..128`, `medium=129..512`, `long=513..1024`
- Paper batch sizes: `[1]`
- Paper `warmup_runs`: `5`
- Paper `timed_runs`: `20`
- Paper `max_new_tokens`: `128`
- Smoke `warmup_runs`: `1`
- Smoke `timed_runs`: `2`
- Smoke `max_new_tokens`: `16`

## 1. Preflight

### Git Status

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git worktree list
```

Record the branch, commit, and whether the tree is dirty before any benchmark run.

### GPU / PyTorch / CUDA / GB10 Capability

```bash
nvidia-smi -L
nvidia-smi
```

```bash
$PY - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        f"gpu[{i}] name={p.name} total_memory_gb={p.total_memory / (1024**3):.2f} "
        f"capability={p.major}.{p.minor}"
    )
PY
```

If PyTorch reports a GB10 capability warning, record it in the run notes. Do not hide it.

### Model Path Check

```bash
$PY - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("paper_benchmarks/configs/models/qwen35a3b.yaml").read_text())
for key in ("model_path", "tokenizer_path", "model_config_path", "cast_package_path"):
    value = cfg.get(key)
    print(key, value, Path(value).exists() if value else None)
print("expected_model_config_hash", cfg.get("expected_model_config_hash"))
PY
```

Expected local model path on this host:
- `/home/gb10/model-cache/Qwen3.5-35B-A3B`

Do not hardcode a different model path unless it actually exists and you are intentionally changing the frozen environment.

### Prompt Suite Hash

```bash
$PY - <<'PY'
from pathlib import Path
import json
from paper_benchmarks.paper_bench.provenance import sha256_path

prompt_path = Path("paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.jsonl")
manifest_path = Path("paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.manifest.json")
print("prompt_exists", prompt_path.exists(), prompt_path)
print("manifest_exists", manifest_path.exists(), manifest_path)
print("prompt_sha256", sha256_path(prompt_path) if prompt_path.exists() else None)
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    print("manifest_prompt_sha256", manifest.get("prompt_file_hash"))
    print("synthetic_workload", manifest.get("synthetic_workload"))
PY
```

The prompt file hash must stay `bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93` for the current paper workflow.

### Project Resolver Check

```bash
$PY -m paper_benchmarks.paper_bench.cli inspect-project \
  --project-ref "$PROJECT_REF"
```

Expected outcome:
- the project resolves successfully
- the resolver maps the encoded ref to the decoded on-disk project
- the output includes the `auto_best_fastest_valid` selection report

### CAST Export Candidate Check

Use the `inspect-project` output above to review:
- `auto_best_fastest_valid.selected_ops`
- `auto_best_fastest_valid.rejected_candidates`
- `auto_best_fastest_valid.export_paper_eligible`

If no valid kernels are selected, stop and fix export evidence before benchmarking.

### Existing CAST Package Check

First inspect the CAST currently referenced by the model config:

```bash
$PY - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("paper_benchmarks/configs/models/qwen35a3b.yaml").read_text())
print(cfg["cast_package_path"])
PY
```

Then inspect it:

```bash
$PY -m paper_benchmarks.paper_bench.cli inspect-cast \
  --cast-package "/home/gb10/Projects/Kernal-Forge/CGinS/kernels/projects/test_qwen - NVIDIA GB10/exports/test_qwen - NVIDIA GB10.cast"
```

Check:
- manifest exists
- selected ops are present
- selected source hashes are present
- `selection_policy` is `auto_best_fastest_valid`
- `project_ref` is `project/test_qwen%20-%20NVIDIA%20GB10/`
- `loadability_blockers` is empty or understood
- whether precompiled binaries exist

### Harness Preflight

Use the actual exported package path once you have one. Until then, the model config path is the default.

```bash
$PY -m paper_benchmarks.paper_bench.cli preflight \
  --model-config "$MODEL_CONFIG" \
  --suite-config "$PAPER_SUITE" \
  --variants eager torch_compile kf_cast \
  --project-ref "$PROJECT_REF" \
  --cast-package "/abs/path/to/qwen35a3b_auto_best_fastest_valid.cast"
```

## 2. Export CAST from Kernel Forge Project

The requested shape in the product brief was:

```bash
python -m paper_benchmarks.paper_bench.cli export-cast \
  --project-ref 'project/test_qwen%20-%20NVIDIA%20GB10/' \
  --selection-policy auto_best_fastest_valid \
  --out paper_benchmarks/artifacts/qwen35a3b/
```

The actual CLI on this branch differs:
- there is no `--selection-policy` flag because `export-cast` already defaults to `auto_best_fastest_valid`
- the output directory flag is `--artifact-dir`, not `--out`

Use the actual commands below.

### Inspect the Canonical Project

```bash
$PY -m paper_benchmarks.paper_bench.cli inspect-project \
  --project-ref "$PROJECT_REF"
```

### Export the CAST Package

```bash
$PY -m paper_benchmarks.paper_bench.cli export-cast \
  --project-ref "$PROJECT_REF" \
  --artifact-dir "$ARTIFACT_DIR"
```

Optional explicit filename:

```bash
$PY -m paper_benchmarks.paper_bench.cli export-cast \
  --project-ref "$PROJECT_REF" \
  --artifact-dir "$ARTIFACT_DIR" \
  --artifact-name "qwen35a3b_auto_best_fastest_valid_<commit>_<date>.cast"
```

After export:
- note the `cast_package_path` printed by the command
- note the `cast_package_sha256`
- review `EXPORT_REPORT.md` and `export_report.json` in `$ARTIFACT_DIR`

Validate the exported package:

```bash
$PY -m paper_benchmarks.paper_bench.cli inspect-cast \
  --cast-package "$ARTIFACT_DIR/<exported-cast-file>.cast"
```

## 3. Smoke Benchmark

Smoke runs are for wiring and runtime sanity only. They are not paper evidence and must not be summarized as a Qwen model win.

### Eager Only Smoke

```bash
$PY -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config "$MODEL_CONFIG" \
  --suite-config "$SMOKE_SUITE" \
  --variants eager \
  --out "${SMOKE_RUN_DIR}_eager"
```

### Eager + torch.compile Smoke

```bash
$PY -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config "$MODEL_CONFIG" \
  --suite-config "$SMOKE_SUITE" \
  --variants eager torch_compile \
  --compile-backend inductor \
  --out "${SMOKE_RUN_DIR}_compile"
```

### `kf_cast` Smoke

```bash
$PY -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config "$MODEL_CONFIG" \
  --suite-config "$SMOKE_SUITE" \
  --variants kf_cast \
  --project-ref "$PROJECT_REF" \
  --cast-package "$ARTIFACT_DIR/<exported-cast-file>.cast" \
  --no-kf-require-precompiled \
  --kf-allow-jit \
  --kf-fail-on-fallback \
  --kf-record-runtime-stats \
  --out "${SMOKE_RUN_DIR}_kf_cast"
```

Smoke rules:
- do not claim a paper result from smoke
- do not reuse smoke artifacts as paper artifacts
- if smoke fails, fix runtime or environment first

## 4. Full Qwen Paper Run

Use:
- project ref: `project/test_qwen%20-%20NVIDIA%20GB10/`
- exported CAST package path: the artifact from section 2
- prompt suite: `paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.jsonl`
- prompt suite hash: `bf02a32ec3e11fe8a5eec24da64b75d1794a621112e36f51a16dae24b3092b93`
- warmup/timed counts: `5` / `20`
- batch sizes: `[1]`
- prompt buckets: `short`, `medium`, `long`
- `max_new_tokens: 128`
- `generation_mode: greedy`
- `include_tokenization_in_timing: false`
- `measure_prefill_decode_separately: true`

### Preferred Full Run When Precompiled Binaries Exist

```bash
$PY -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config "$MODEL_CONFIG" \
  --suite-config "$PAPER_SUITE" \
  --variants eager torch_compile kf_cast \
  --project-ref "$PROJECT_REF" \
  --cast-package "$ARTIFACT_DIR/<exported-cast-file>.cast" \
  --compile-backend inductor \
  --kf-require-precompiled \
  --no-kf-allow-jit \
  --kf-fail-on-fallback \
  --kf-record-runtime-stats \
  --out "$PAPER_RUN_DIR" \
  --fail-if-not-paper-eligible
```

### Full Run When JIT Is Required

Use this only if `inspect-cast` shows no usable precompiled binary for GB10.

```bash
$PY -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config "$MODEL_CONFIG" \
  --suite-config "$PAPER_SUITE" \
  --variants eager torch_compile kf_cast \
  --project-ref "$PROJECT_REF" \
  --cast-package "$ARTIFACT_DIR/<exported-cast-file>.cast" \
  --compile-backend inductor \
  --no-kf-require-precompiled \
  --kf-allow-jit \
  --kf-fail-on-fallback \
  --kf-record-runtime-stats \
  --out "$PAPER_RUN_DIR" \
  --fail-if-not-paper-eligible
```

Notes:
- `torch.compile` compile time is reported separately from steady-state
- Kernel Forge load, setup, and JIT time are reported separately from steady-state
- warmup is excluded from steady-state
- no hidden fallback is allowed in a paper-eligible `kf_cast` result
- do not use `--reuse-cache` on the first evidence run

## 5. Validation and Reporting

### Validate Benchmark Artifacts

```bash
for f in "$PAPER_RUN_DIR"/metrics/*.json "$PAPER_RUN_DIR"/reports/summary.json; do
  $PY -m paper_benchmarks.paper_bench.cli validate-artifact "$f" || exit 1
done
```

### Summarize the Run

```bash
$PY -m paper_benchmarks.paper_bench.cli summarize \
  --run-dir "$PAPER_RUN_DIR"
```

Artifacts to inspect:
- `$PAPER_RUN_DIR/reports/summary.md`
- `$PAPER_RUN_DIR/reports/summary.json`
- `$PAPER_RUN_DIR/reports/summary.csv`

### Inspect Report Claims

```bash
$PY - <<'PY'
from pathlib import Path
import json

summary = json.loads(Path("paper_benchmarks/runs/qwen35a3b_dgx_spark_paper/reports/summary.json").read_text())
print("paper_eligible", summary["paper_eligible"])
print("paper_eligibility_issues", summary["paper_eligibility_issues"])
print("paper_eligible_claims")
for item in summary["paper_eligible_claims"]:
    print("-", item)
print("forbidden_claims")
for item in summary["forbidden_claims"]:
    print("-", item)
print("failure_regressions")
for item in summary["failure_regressions"]:
    print("-", item)
PY
```

The `Export/CAST Selection` section in the report must include:
- project ref
- CAST package path/hash
- selection policy
- selected ops and source hashes
- evidence tier per op
- rejected export candidates

## 6. Failure Interpretation

### `torch_compile` Failed

- Record it as a `torch_compile` failure
- Do not replace it with eager and do not claim a compile comparison
- The model-speedup claim against `torch.compile` is unsupported

### Fallback Occurred

- If `fallback_count > 0` and `--kf-fail-on-fallback` was enabled, the `kf_cast` result is not paper eligible
- Do not hide fallback behind aggregate latency
- Inspect runtime stats and per-op hits/fallbacks before rerunning

### Token Mismatch

- Any timed `kf_cast` or `torch_compile` output mismatch against eager makes the model-speedup result unsafe
- Safe interpretation: `Unsafe speedup; invalid model-speedup result.`

### No Speedup vs `torch.compile`

- Safe interpretation: `Kernel Forge beats eager but does not beat torch.compile on this workload.`
- Do not rewrite that as a win over `torch.compile`

### No Valid CAST Kernels Selected

- `export-cast` should fail loudly unless `--allow-native-package` was explicitly requested
- Fix correctness, input capture, timing provenance, or source-hash mismatches before retrying

### Micro-Only or Operator-Only Export

- Export may still be possible if product policy permits it
- The manifest/report must show non-deployment evidence
- Do not treat it as deployment-paper evidence for model-level claims

### GB10 Architecture Warning

- If PyTorch warns that GB10 capability is outside the supported range, record the exact warning and versions
- Treat the environment as potentially unsupported until validated
- Do not hide the warning in the paper appendix

### OOM

- Reduce concurrent workload only by changing the frozen suite in a new explicit revision
- Do not silently shrink prompt lengths, batch sizes, or `max_new_tokens` inside the benchmark loop

### JIT Too Slow

- JIT/setup time belongs in offline or deployment setup costs, not steady-state inference
- If JIT is unusably slow, prefer producing precompiled binaries instead of folding JIT into throughput claims

## 7. Safe and Unsafe Claims

Safe claims:
- `Kernel Forge improves model throughput on this workload.` Only when the final report includes that exact paper-eligible claim.
- `Kernel Forge beats eager but does not beat torch.compile on this workload.`
- `Operator wins did not translate into an end-to-end model win.`
- `Unsafe speedup; invalid model-speedup result.`

Unsafe claims:
- Any Qwen win claim from smoke
- Any model win claim when `torch_compile` failed or is missing
- Any model win claim when timed outputs do not exactly match eager
- Any model win claim with hidden or nonzero fallback
- Any model win claim from operator-only or micro-only evidence
- Any claim using a `.cast` not exported from `project/test_qwen%20-%20NVIDIA%20GB10/` unless the report clearly states otherwise

Do not assume Qwen wins. The report decides whether any claim is allowed.
