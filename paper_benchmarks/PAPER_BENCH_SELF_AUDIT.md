# Paper Benchmark Self-Audit

Date: 2026-04-21 UTC

Scope:
- audit the standalone paper harness under `paper_benchmarks/`
- verify claim gating, provenance, correctness, cache reuse, and report separation
- run the fast release-blocking test target
- run a real CLI smoke benchmark with a tiny local causal LM
- validate produced artifacts and regenerate the report

## Executive Result

The harness is suitable to proceed to a real frozen-model benchmark such as Qwen, with two audit-time bugs fixed before signoff:

1. Real Hugging Face tokenizers returned `BatchEncoding`, but the LLM runner only accepted plain `dict` objects. This blocked real CLI runs and is now fixed in `paper_benchmarks/paper_bench/llm_runner.py`.
2. The report deployment/runtime section previously included all `kf_cast` rows, which could sweep operator-only rows into a deployment section. This is now fixed in `paper_benchmarks/paper_bench/report.py`.

After these fixes:
- `paper_benchmarks/run_paper_bench_tests.sh` passed
- the CLI smoke benchmark passed
- artifact validation passed
- report generation passed

## Answers

### 1. Can this harness support a claim against eager?

Yes, with limits.

Safe eager-relative claim supported:
- "Kernel Forge beats eager on this workload."

Not supported from eager alone:
- a paper-grade model-throughput claim that implies superiority over the strongest PyTorch baseline

Reason:
- report gating refuses a full model-speedup claim if `torch_compile` is missing
- a weaker eager-relative statement is preserved when `kf_cast` beats eager but not `torch_compile`

### 2. Can this harness support a claim against `torch.compile`?

Yes.

Required conditions:
- `eager`, `torch_compile`, and `kf_cast` all ran
- exact token equality against eager for every timed run
- no hidden fallback
- no synthetic workload
- frozen hashed workload
- deployment path used
- timed sample count meets the suite minimum
- artifacts remain paper-eligible

### 3. Can this harness support a deployment claim for Kernel Forge `.cast` runtime?

Yes, for the deployed `kf_cast` path only.

Required evidence:
- `.cast` path present
- `.cast` SHA256 present
- runtime patch enabled
- runtime stats recorded
- fallback reported
- correctness passed

Direct source operator benchmarks are explicitly operator-only and must not be labeled deployment.

### 4. Does it separate compile/load/warmup/offline cost from steady-state inference?

Yes.

Evidence:
- separate stages: `load`, `compile`, `warmup`, `prefill`, `decode`, `total_generate`
- separate fields: `steady_state_time_ms`, `compile_time_ms`, runtime load and JIT fields
- tests explicitly cover warmup exclusion and compile separation

### 5. Does it separate prefill and decode for LLMs?

Yes.

Evidence:
- distinct LLM artifacts for `prefill`, `decode`, and `total_generate`
- distinct TPS metrics for prefill and decode
- distinct per-sample token counts and latencies

### 6. Does it reject synthetic workloads for paper runs?

Yes.

Behavior:
- synthetic workloads require `--allow-synthetic-demo`
- allowing synthetic runs still marks them `paper_eligible=false`
- synthetic runs validate as demos, not as paper evidence

### 7. Does it reject hidden fallback?

Yes.

Behavior:
- missing fallback reporting makes `kf_cast` non-paper
- nonzero fallback with `fail_on_fallback=true` makes the run non-paper
- report claim gating refuses throughput claims if fallback is hidden or nonzero

### 8. Does it compare output tokens for every timed run?

Yes.

Evidence:
- per-run `output_token_hashes` recorded in sample records and raw outputs
- `torch_compile` and `kf_cast` are compared against eager on every timed run
- missing per-run output-hash evidence now makes model-level artifacts non-paper

### 9. Does it prevent stale cache reuse?

Yes.

Behavior:
- default is no reuse unless `--reuse-cache` is passed
- reuse requires exact signature match across harness version, git, hashes, environment, compile settings, workload selection, sample matrix, and artifact hashes
- artifacts with no raw samples are not reusable

### 10. Does it prevent operator micro wins from being summarized as model wins?

Yes.

Evidence:
- operator and model sections are separate
- claim gating does not average operator wins into model wins
- safe fallback claim exists: "Operator wins did not translate into an end-to-end model win."
- audit-time fix now keeps operator-only rows out of the deployment/runtime section

### 11. Does it record enough provenance for reproduction?

Yes, materially enough for paper reproduction.

Recorded fields include:
- git availability, commit, branch, dirty status, dirty summary, untracked summary
- timestamp, hostname, OS, Python, PyTorch, CUDA, cuDNN
- GPU name/count/device properties and driver info when available
- relevant CUDA/PyTorch environment variables
- package versions including `torch`, `transformers`, `numpy`, `triton` when installed
- determinism controls
- exact command line
- model id, model path, model path hash, model config path, model config hash
- suite path/hash, workload path/hash
- cast package path/hash and visible kernel hashes when used

If git metadata or required hashes are missing, the run becomes non-paper.

### 12. What claims are still unsafe?

Still unsafe:
- any claim from a synthetic or demo run
- any model-speedup claim without both `eager` and `torch_compile`
- any claim when outputs do not exactly match eager for every timed run
- any claim with hidden or nonzero fallback unless explicitly downgraded and reported
- any deployment claim from direct source operator benchmarks
- any claim that averages operator wins into model wins
- any claim that reuses stale cache after hash, config, environment, or selection changes
- any Qwen claim before a real frozen Qwen prompt set, model snapshot, and `.cast` package are run through this harness

## Audit-Time Findings Fixed

### A. Hugging Face tokenizer compatibility bug

Problem:
- real HF tokenizers return `BatchEncoding`
- the runner rejected those during prompt length computation and batch tokenization

Fix:
- accept generic mapping-like tokenizer outputs, not only plain `dict`

Files:
- `paper_benchmarks/paper_bench/llm_runner.py`
- `paper_benchmarks/tests/test_llm_baselines.py`

### B. Deployment report section over-included operator rows

Problem:
- deployment/runtime section included all `kf_cast` rows
- operator-only rows could appear in a deployment section

Fix:
- deployment section now excludes `Stage.operator`

Files:
- `paper_benchmarks/paper_bench/report.py`
- `paper_benchmarks/tests/test_report.py`

## Verification Evidence

### 1. Release-blocking fast test target

Command:

```bash
paper_benchmarks/run_paper_bench_tests.sh
```

Result:
- `87 passed` in `paper_benchmarks/tests`
- `2 passed` in `tests/test_benchmark_harness.py`

Observed warnings:
- existing pytest `_jac_finder` assert-rewrite warning
- existing GB10 CUDA capability warning from the local PyTorch build

### 2. Real CLI smoke benchmark

Tiny local model:
- GPT-2 style random tiny causal LM saved locally under `/tmp/paper_bench_smoke_audit_YVUNEs/tiny_model`
- synthetic prompt suite used intentionally as a smoke/demo workload

Command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli run-llm \
  --model-config /tmp/paper_bench_smoke_audit_YVUNEs/tiny_model.yaml \
  --suite-config /tmp/paper_bench_smoke_audit_YVUNEs/tiny_suite.yaml \
  --variants eager torch_compile \
  --allow-synthetic-demo \
  --out /tmp/paper_bench_smoke_audit_YVUNEs/run
```

Result:

```json
{
  "ok": true,
  "run_dir": "/tmp/paper_bench_smoke_audit_YVUNEs/run",
  "summary_rows": 11,
  "paper_eligible": false,
  "paper_eligibility_issues": [
    "paper run not requested or explicitly downgraded",
    "synthetic workload used",
    "no claim-eligible steady-state rows found",
    "missing kf_cast variant for model-level comparison",
    "no comparable model-level stage across eager, torch_compile, and kf_cast"
  ]
}
```

Interpretation:
- this is the expected outcome for a synthetic eager+compile smoke run
- it proves the real CLI path works
- it does not support any paper claim

### 3. Artifact validation

Benchmark artifact validation:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-artifact \
  /tmp/paper_bench_smoke_audit_YVUNEs/run/metrics/torch_compile_total_generate.json
```

Result:

```json
{
  "ok": true,
  "artifact_type": "benchmark_result",
  "schema_version": "1.0",
  "paper_eligible": false,
  "paper_eligibility_issues": [
    "paper run not requested or explicitly downgraded",
    "synthetic workload used"
  ]
}
```

Summary artifact validation:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli validate-artifact \
  /tmp/paper_bench_smoke_audit_YVUNEs/run/reports/summary.json
```

Result:

```json
{
  "ok": true,
  "artifact_type": "summary_report",
  "schema_version": "1.0",
  "paper_eligible": false,
  "paper_eligibility_issues": [
    "paper run not requested or explicitly downgraded",
    "synthetic workload used",
    "no claim-eligible steady-state rows found",
    "missing kf_cast variant for model-level comparison",
    "no comparable model-level stage across eager, torch_compile, and kf_cast"
  ]
}
```

### 4. Report generation

Command:

```bash
./.venv/bin/python -m paper_benchmarks.paper_bench.cli summarize \
  --run-dir /tmp/paper_bench_smoke_audit_YVUNEs/run
```

Result:

```json
{
  "ok": true,
  "run_dir": "/tmp/paper_bench_smoke_audit_YVUNEs/run",
  "rows": 11
}
```

Generated outputs:
- `/tmp/paper_bench_smoke_audit_YVUNEs/run/reports/summary.md`
- `/tmp/paper_bench_smoke_audit_YVUNEs/run/reports/summary.json`
- `/tmp/paper_bench_smoke_audit_YVUNEs/run/reports/summary.csv`

## Conclusion

The harness is ready for a real Qwen benchmark run, subject to the normal paper-run prerequisites:
- frozen prompt suite
- real Qwen snapshot path and hashes
- real `.cast` package and hashes
- `eager`, `torch_compile`, and `kf_cast` all present
- no hidden fallback
- exact token equality

The smoke run performed here is intentionally non-paper and should not be used for claims.
