from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import benchmark
from .compat import assert_supported_device, chip_family, ensure_llamacpp_commit, get_llamacpp_commit
from .constants import (
    LLM_TUNING_ATTEMPTS_QUICK,
    PASS_MAX_REGRESSION_PCT,
    PASS_PRIMARY_UPLIFT_PCT,
    current_cache_root,
)
from .device_probe import probe_device
from .feasibility import derive_allowed_params, evaluate_candidate_feasibility
from .kernel_patch import (
    KernelPatchError,
    build_kernel_patch_candidate,
    classify_compile_record,
    classify_correctness_record,
    kernel_candidate_dict,
)
from .model_probe import assert_supported_model, probe_model
from .model_store import ensure_default_model
from .op_profile import (
    compare_stage1_op_profiles,
    run_op_correctness_checks,
    run_stage1_op_profile,
    suggest_ggml_ops_from_hotspots,
)
from .pack import create_pack, disable_pack, export_pack, set_active_pack
from .runtime_args import sanitize_runtime_args
from .tuner import LlmTuningError, propose_llm_candidate
from .types import BenchmarkResult, KernelCompileRecord, KernelCorrectnessRecord, OptimizeReport


class OptimizeError(RuntimeError):
    pass


def _append_attempt_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _compute_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    by_name_before = {row["profile"]["name"]: row for row in before}
    by_name_after = {row["profile"]["name"]: row for row in after}

    per_profile: dict[str, dict[str, float | None]] = {}
    for name, b in by_name_before.items():
        a = by_name_after.get(name)
        if not a:
            continue
        b_metrics = b["metrics"]
        a_metrics = a["metrics"]

        def uplift(before_v: float | None, after_v: float | None) -> float | None:
            if before_v is None or after_v is None or before_v == 0:
                return None
            return ((after_v - before_v) / before_v) * 100.0

        per_profile[name] = {
            "prefill_uplift_pct": uplift(
                b_metrics.get("prefill_tokens_per_sec"),
                a_metrics.get("prefill_tokens_per_sec"),
            ),
            "decode_uplift_pct": uplift(
                b_metrics.get("decode_tokens_per_sec"),
                a_metrics.get("decode_tokens_per_sec"),
            ),
            "ttft_delta_pct": uplift(
                b_metrics.get("ttft_ms"),
                a_metrics.get("ttft_ms"),
            ),
        }

    return {"profiles": per_profile}


def _primary_uplift(delta: dict[str, Any]) -> float:
    best = float("-inf")
    profiles = delta.get("profiles") or {}
    for row in profiles.values():
        for metric in ("prefill_uplift_pct", "decode_uplift_pct"):
            value = row.get(metric)
            if isinstance(value, (int, float)) and value > best:
                best = float(value)
    return best if best != float("-inf") else float("-1e9")


def _best_decode_uplift(delta: dict[str, Any]) -> float:
    vals: list[float] = []
    profiles = delta.get("profiles") or {}
    for row in profiles.values():
        value = row.get("decode_uplift_pct")
        if isinstance(value, (int, float)):
            vals.append(float(value))
    if not vals:
        return float("-1e9")
    # Robust objective: optimize the worst profile uplift, not just the best.
    return min(vals)


def _worst_regression_magnitude(delta: dict[str, Any]) -> float:
    worst = 0.0
    profiles = delta.get("profiles") or {}
    for row in profiles.values():
        for metric in ("prefill_uplift_pct", "decode_uplift_pct"):
            value = row.get(metric)
            if isinstance(value, (int, float)) and value < 0:
                worst = max(worst, abs(float(value)))
    return worst


def _passes_gate(delta: dict[str, Any]) -> bool:
    best_primary = _primary_uplift(delta)
    min_decode = _best_decode_uplift(delta)
    worst_regression_mag = _worst_regression_magnitude(delta)
    return (
        best_primary >= PASS_PRIMARY_UPLIFT_PCT
        and worst_regression_mag <= PASS_MAX_REGRESSION_PCT
        and min_decode >= (-PASS_MAX_REGRESSION_PCT)
    )


def _score_delta(delta: dict[str, Any]) -> float:
    decode_primary = _best_decode_uplift(delta)
    worst_regression_mag = _worst_regression_magnitude(delta)
    penalty = 2.0 * max(0.0, worst_regression_mag - PASS_MAX_REGRESSION_PCT)
    return decode_primary - penalty


def _aggregate_compile(records: list[KernelCompileRecord]) -> KernelCompileRecord:
    if not records:
        return KernelCompileRecord(
            attempted=False,
            success=False,
            classification="not_attempted",
        )
    if all(r.success for r in records):
        merged_hash = "|".join([r.stderr_hash for r in records if r.stderr_hash])
        return KernelCompileRecord(
            attempted=True,
            success=True,
            classification="compiled_or_loaded",
            stderr_hash=merged_hash,
        )
    first_failure = next((r for r in records if not r.success), records[0])
    return KernelCompileRecord(
        attempted=True,
        success=False,
        classification=first_failure.classification,
        stderr_hash=first_failure.stderr_hash,
        error=first_failure.error,
    )


def _aggregate_correctness(
    *,
    baseline_results: list[BenchmarkResult],
    tuned_results: list[BenchmarkResult],
    strict_parity: bool = False,
) -> KernelCorrectnessRecord:
    by_profile_base = {r.profile.name: r for r in baseline_results}
    checks: list[KernelCorrectnessRecord] = []
    for tuned in tuned_results:
        base = by_profile_base.get(tuned.profile.name)
        if base is None:
            checks.append(
                KernelCorrectnessRecord(
                    attempted=True,
                    success=False,
                    classification="missing_baseline_profile",
                    details={"profile": tuned.profile.name},
                )
            )
            continue
        checks.append(
            classify_correctness_record(
                baseline=base,
                candidate=tuned,
                strict_parity=strict_parity,
            )
        )

    if checks and all(c.success for c in checks):
        details: dict[str, Any] = {c.classification: c.details for c in checks}
        return KernelCorrectnessRecord(
            attempted=True,
            success=True,
            classification="metric_sanity_ok",
            details=details,
        )
    if not checks:
        return KernelCorrectnessRecord(
            attempted=False,
            success=False,
            classification="not_attempted",
            details={},
        )
    first_failure = next((c for c in checks if not c.success), checks[0])
    return KernelCorrectnessRecord(
        attempted=True,
        success=False,
        classification=first_failure.classification,
        details=first_failure.details,
    )


def _dispatch_rule_id_for_patch(*, patch_hash: str, stage: str, attempt_idx: int) -> str:
    base = str(patch_hash or "").strip()
    if not base:
        base = f"{stage}_attempt_{attempt_idx}"
    token = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base).strip("._-")
    return f"rule_{token[:64]}" if token else ""


def _first_dispatch_audit(results: list[BenchmarkResult]) -> dict[str, Any]:
    for res in results:
        for run in res.runs:
            if isinstance(run.get("dispatch_audit"), dict):
                return dict(run.get("dispatch_audit"))
    return {}


def _dispatch_proof_failure(
    *,
    results: list[BenchmarkResult],
    candidate_resources_expected: bool,
) -> str:
    if not candidate_resources_expected:
        return ""
    for result in results:
        if result.metrics.prefill_tokens_per_sec is None or result.metrics.decode_tokens_per_sec is None:
            continue
        for run in result.runs:
            dispatch = dict(run.get("dispatch_audit")) if isinstance(run.get("dispatch_audit"), dict) else {}
            status = str(
                run.get("dispatch_audit_status")
                or dispatch.get("dispatch_audit_status")
                or ""
            ).strip().lower()
            if status != "ok":
                if status == "parse_fail":
                    return "dispatch_audit_parse_fail"
                if status in {"missing", "backend_noaudit"}:
                    return "dispatch_audit_missing"
                return "dispatch_audit_missing"
            used = run.get("candidate_resources_used")
            if not isinstance(used, bool):
                used = dispatch.get("candidate_resources_used")
            if used is not True:
                return "candidate_resources_not_used"
    return ""


def optimize_for_apple_silicon(
    *,
    model_path: str | None,
    profile_mode: str,
    gate_mode: str,
    prompts_path: str | None,
    llamacpp_root: Path,
    cgins_version: str,
    strict_commit: bool = True,
    attempt_budget: int | None = None,
    study_tag: str = "",
    emit_attempt_log: str = "",
    kernel_mode: str = "none",
    candidate_cache_dir: str = "",
    strict_parity: bool = False,
    reuse_policy: str = "chip_family",
    profile_set: str = "mixed",
    profiling_mode: str = "op_perf_required",
    stage0_feasibility: bool = True,
    stage1_op_sql: bool = True,
    stage2_logits_gate: bool = True,
    op_perf_timeout_sec: float = 90.0,
    op_perf_cache: str = "on",
    op_perf_min_rows: int = 1,
    op_perf_op_filter: str = "MUL_MAT",
    op_perf_case_limit: int = 64,
    op_perf_case_seed: int = 0,
    op_perf_warmup_iters: int = 1,
    op_perf_bench_iters: int = 3,
    op_perf_reject_regression_pct: float = 10.0,
    op_perf_promote_topk: int = 3,
    op_test_timeout_sec: float = 45.0,
    op_test_cache: str = "on",
    op_test_min_rows: int = 1,
    op_test_case_limit: int = 32,
    op_test_case_seed: int = 0,
) -> OptimizeReport:
    kernel_mode = (kernel_mode or "none").strip().lower()
    if kernel_mode not in {"none", "oneshot", "iterative"}:
        raise OptimizeError("kernel_mode must be one of: none, oneshot, iterative")
    profiling_mode_norm = (profiling_mode or "op_perf_required").strip().lower()
    if profiling_mode_norm not in {"op_perf_required", "heuristic"}:
        raise OptimizeError("profiling_mode must be one of: op_perf_required, heuristic")

    profile_set_norm = (profile_set or "mixed").strip().lower()
    if profile_set_norm not in {"smoke", "claim", "mixed"}:
        raise OptimizeError("profile_set must be one of: smoke, claim, mixed")
    profile_mode_norm = (profile_mode or "both").strip().lower()
    if profile_set_norm == "smoke":
        if profile_mode_norm == "both":
            profile_mode = "chat,long_smoke"
        elif profile_mode_norm == "long":
            profile_mode = "long_smoke"
    elif profile_set_norm == "claim":
        if profile_mode_norm == "both":
            profile_mode = "chat,long_claim"
        elif profile_mode_norm == "long":
            profile_mode = "long_claim"
    op_perf_reject_regression_pct = float(max(0.0, op_perf_reject_regression_pct))
    op_perf_promote_topk = int(max(0, op_perf_promote_topk))
    stage1_force_strict_regression = bool(profile_set_norm == "claim" and strict_parity)
    if stage1_force_strict_regression:
        op_perf_promote_topk = 0

    device = probe_device()
    assert_supported_device(device)
    allowed_params = asdict(derive_allowed_params(device))

    if model_path:
        resolved_model_path = Path(model_path).expanduser().resolve()
    else:
        resolved_model_path = ensure_default_model()

    model = probe_model(resolved_model_path)
    assert_supported_model(model)

    commit_ok, commit_msg = ensure_llamacpp_commit(llamacpp_root, strict=strict_commit)
    if not commit_ok:
        raise OptimizeError(commit_msg)
    local_commit = get_llamacpp_commit(llamacpp_root)

    llama_cli = benchmark.resolve_llama_cli(llamacpp_root)
    if kernel_mode != "none":
        metal_tools = benchmark.resolve_metal_toolchain_paths()
        if not bool(metal_tools.get("success")):
            raise OptimizeError(
                "Metal toolchain preflight failed: unable to resolve xcrun metal/metallib. "
                "Select full Xcode or set DEVELOPER_DIR and retry."
            )
    prompts = benchmark.load_prompt_suite(prompts_path)
    cache_root = (
        Path(candidate_cache_dir).expanduser().resolve()
        if candidate_cache_dir
        else (current_cache_root() / "candidates")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    dispatch_audit_dir = cache_root / "dispatch_audit"
    dispatch_audit_dir.mkdir(parents=True, exist_ok=True)

    stage_baselines: dict[str, list[BenchmarkResult]] = {}

    def get_stage_baseline(stage: str) -> list[BenchmarkResult]:
        key = stage.lower()
        if key in stage_baselines:
            return stage_baselines[key]
        runs = benchmark.run_benchmarks(
            llama_cli=llama_cli,
            model_path=model.path,
            profile_mode=profile_mode,
            gate_mode=key,
            prompts=prompts,
            resources_path=None,
            extra_args=None,
            capture_raw_output=True,
            dispatch_attempt_id=f"baseline_{key}",
            dispatch_rule_id="",
            dispatch_audit_dir=dispatch_audit_dir,
            candidate_resources_expected=False,
        )
        stage_baselines[key] = runs
        return runs

    baseline_requested = get_stage_baseline(gate_mode.lower())
    baseline_requested_dict = benchmark.benchmark_results_to_dict(baseline_requested)

    strict_guardrails = {
        "min_primary_uplift_pct": PASS_PRIMARY_UPLIFT_PCT,
        "max_allowed_regression_pct": PASS_MAX_REGRESSION_PCT,
        "allowed_params": allowed_params,
        "correctness": {
            "seed": 42,
            "temperature": 0,
            "greedy": True,
            "strict_parity": bool(strict_parity),
        },
    }

    dynamic_ladder = False
    if isinstance(attempt_budget, int) and attempt_budget > 0:
        max_attempts = attempt_budget
    elif gate_mode.lower() == "full":
        max_attempts = 1
        dynamic_ladder = True
    else:
        max_attempts = LLM_TUNING_ATTEMPTS_QUICK

    followup_stage = "full"

    tuning_attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    last_llm_meta: dict[str, Any] = {}
    attempt_log_path = Path(emit_attempt_log).expanduser().resolve() if emit_attempt_log else None
    stage1_promotions_used = 0

    attempt_idx = 0
    while attempt_idx < max_attempts:
        attempt_idx += 1
        started = time.time()

        stage = gate_mode.lower()
        if dynamic_ladder and stage == "full":
            stage = "quick" if attempt_idx == 1 else followup_stage

        baseline_for_stage = get_stage_baseline(stage)
        baseline_for_stage_dict = benchmark.benchmark_results_to_dict(baseline_for_stage)
        stage_profiles = benchmark.workload_profiles(profile_mode, stage)
        op_profiles: list[dict[str, Any]] = []
        if stage1_op_sql:
            op_profiles = [
                run_stage1_op_profile(
                    llamacpp_root=llamacpp_root,
                    model_path=model.path,
                    profile=sp,
                    profiling_mode=profiling_mode_norm,
                    rank_metric="time",
                    timeout_sec=float(op_perf_timeout_sec),
                    cache_mode=op_perf_cache,
                    min_rows=int(max(1, op_perf_min_rows)),
                    op_filter=op_perf_op_filter,
                    perf_case_limit=int(max(0, op_perf_case_limit)),
                    perf_case_seed=int(max(0, op_perf_case_seed)),
                    perf_warmup_iters=int(max(0, op_perf_warmup_iters)),
                    perf_bench_iters=int(max(0, op_perf_bench_iters)),
                )
                for sp in stage_profiles
            ]
            if profiling_mode_norm == "op_perf_required":
                failed = [p for p in op_profiles if not bool(p.get("success"))]
                if failed:
                    raise OptimizeError(
                        "op_perf_required profiling failed: "
                        + "; ".join(str(p.get("reason", "unknown")) for p in failed)
                    )

        try:
            candidate, llm_meta = propose_llm_candidate(
                device=device,
                model=model,
                profile_mode=profile_mode,
                gate_mode=stage,
                baseline=baseline_for_stage_dict,
                previous_attempts=tuning_attempts,
                allowed_params=allowed_params,
            )
        except LlmTuningError as exc:
            if attempt_log_path is not None:
                _append_attempt_log(
                    attempt_log_path,
                    {
                        "ts": time.time(),
                        "study_tag": study_tag,
                        "attempt": attempt_idx,
                        "stage": stage,
                        "model_path": str(model.path),
                        "profile_mode": profile_mode,
                        "gate_mode": gate_mode,
                        "provider": last_llm_meta.get("provider", ""),
                        "model": last_llm_meta.get("model", ""),
                        "success": False,
                        "error": str(exc),
                    },
                )
            raise OptimizeError(f"LLM tuning failed on attempt {attempt_idx}: {exc}") from exc

        last_llm_meta = llm_meta
        runtime_args = sanitize_runtime_args(candidate.runtime_args)
        template_mutations = getattr(candidate, "template_mutations", {}) or {}
        source_patches = getattr(candidate, "source_patches", []) or []

        kernel_candidate = None
        resources_path: Path | None = None
        force_source_compile = False
        compile_error = ""
        compile_meta: dict[str, Any] = {}
        op_gate: dict[str, Any] = {"attempted": False, "success": True, "classification": "not_attempted"}
        candidate_op_profiles: list[dict[str, Any]] = []
        op_perf_delta_pct: float | None = None
        op_perf_common_rows = 0
        op_perf_baseline_total_ms: float | None = None
        op_perf_candidate_total_ms: float | None = None
        op_perf_compare_key = ""
        op_perf_decision = "not_evaluated"
        op_perf_promoted = False
        op_perf_compare_rows: list[dict[str, Any]] = []
        candidate_dispatch_rule_id = ""
        feasibility_record = evaluate_candidate_feasibility(
            device=device,
            template_mutations=template_mutations,
            kernel_overrides=candidate.kernel_overrides,
            allowed_params=allowed_params,
        )
        if stage0_feasibility and not feasibility_record.success:
            compile_error = (
                f"{feasibility_record.classification}:"
                + ";".join(feasibility_record.reasons)
            )

        if kernel_mode != "none" and not compile_error:
            if kernel_mode == "oneshot" and attempt_idx > 1:
                # one-shot kernel mode only executes one kernel attempt
                break
            try:
                kernel_candidate = build_kernel_patch_candidate(
                    llamacpp_root=llamacpp_root,
                    candidate_cache_dir=cache_root,
                    candidate_id=f"{model.sha256[:10]}-{profile_mode}-{stage}-a{attempt_idx}",
                    template_mutations=template_mutations,
                    source_patches=source_patches,
                )
                resources_path = Path(kernel_candidate.resources_dir)
                compile_meta = benchmark.prepare_candidate_resources_for_benchmark(
                    resources_path=resources_path,
                    llamacpp_commit=local_commit,
                    chip_family=chip_family(device.chip),
                    macos_version=device.macos_version,
                    source_hash=str(kernel_candidate.source_hash),
                    candidate_hash=str(kernel_candidate.patch_hash),
                )
                candidate_dispatch_rule_id = _dispatch_rule_id_for_patch(
                    patch_hash=str(kernel_candidate.patch_hash),
                    stage=stage,
                    attempt_idx=attempt_idx,
                )
                if bool(compile_meta.get("compile_warmup_done")):
                    force_source_compile = False
                else:
                    force_source_compile = True
                if not bool(compile_meta.get("success")):
                    compile_error = str(
                        compile_meta.get("error")
                        or compile_meta.get("classification")
                        or "compile_warmup_failed"
                    )
                elif stage1_op_sql:
                    candidate_hash = str(kernel_candidate.patch_hash)
                    candidate_op_profiles = [
                        run_stage1_op_profile(
                            llamacpp_root=llamacpp_root,
                            model_path=model.path,
                            profile=sp,
                            profiling_mode=profiling_mode_norm,
                            rank_metric="time",
                            timeout_sec=float(op_perf_timeout_sec),
                            cache_mode=op_perf_cache,
                            min_rows=int(max(1, op_perf_min_rows)),
                            op_filter=op_perf_op_filter,
                            resources_path=resources_path,
                            candidate_hash=candidate_hash,
                            perf_case_limit=int(max(0, op_perf_case_limit)),
                            perf_case_seed=int(max(0, op_perf_case_seed)),
                            perf_warmup_iters=int(max(0, op_perf_warmup_iters)),
                            perf_bench_iters=int(max(0, op_perf_bench_iters)),
                        )
                        for sp in stage_profiles
                    ]
                    if profiling_mode_norm == "op_perf_required":
                        failed = [p for p in candidate_op_profiles if not bool(p.get("success"))]
                        if failed:
                            compile_error = (
                                "op_perf_required candidate profiling failed: "
                                + "; ".join(str(p.get("reason", "unknown")) for p in failed)
                            )
                    if not compile_error:
                        deltas: list[float] = []
                        common_rows_total = 0
                        baseline_total_ms_sum = 0.0
                        candidate_total_ms_sum = 0.0
                        compare_key = ""
                        for base_p, cand_p in zip(op_profiles, candidate_op_profiles):
                            base_rows = list(base_p.get("ops") or [])
                            cand_rows = list(cand_p.get("ops") or [])
                            cmp_row = compare_stage1_op_profiles(
                                baseline_ops=base_rows,
                                candidate_ops=cand_rows,
                            )
                            op_perf_compare_rows.append(cmp_row)
                            if not compare_key:
                                compare_key = str(cmp_row.get("compare_key") or "")
                            common_rows = int(cmp_row.get("common_rows") or 0)
                            common_rows_total += common_rows
                            baseline_total = cmp_row.get("baseline_total_ms")
                            candidate_total = cmp_row.get("candidate_total_ms")
                            if isinstance(baseline_total, (int, float)):
                                baseline_total_ms_sum += float(baseline_total)
                            if isinstance(candidate_total, (int, float)):
                                candidate_total_ms_sum += float(candidate_total)
                            delta_val = cmp_row.get("delta_pct")
                            if not isinstance(delta_val, (int, float)):
                                continue
                            deltas.append(float(delta_val))
                        op_perf_common_rows = int(common_rows_total)
                        op_perf_compare_key = compare_key
                        if baseline_total_ms_sum > 0:
                            op_perf_baseline_total_ms = float(baseline_total_ms_sum)
                        if candidate_total_ms_sum > 0:
                            op_perf_candidate_total_ms = float(candidate_total_ms_sum)
                        if deltas:
                            op_perf_delta_pct = sum(deltas) / len(deltas)
                            reject_cutoff = -abs(float(op_perf_reject_regression_pct))
                            if op_perf_delta_pct < reject_cutoff:
                                if stage1_promotions_used < int(max(0, op_perf_promote_topk)):
                                    stage1_promotions_used += 1
                                    op_perf_promoted = True
                                    op_perf_decision = "promoted_hard_regression"
                                else:
                                    op_perf_decision = "op_perf_regression_reject"
                                    compile_error = f"op_perf_regression:{op_perf_delta_pct:.3f}"
                            elif op_perf_delta_pct < 0.0:
                                op_perf_decision = "stage1_neutral"
                            else:
                                op_perf_decision = "stage1_pass"
                        else:
                            op_perf_decision = "op_perf_uncomparable"
                elif stage2_logits_gate:
                    op_gate = run_op_correctness_checks(
                        llamacpp_root=llamacpp_root,
                        ops=suggest_ggml_ops_from_hotspots(
                            [
                                str(op_row.get("op", ""))
                                for profile_row in op_profiles
                                for op_row in (profile_row.get("ops") or [])
                            ]
                        ),
                        resources_path=resources_path,
                        backend="Metal",
                        max_ops=3,
                        timeout_s=float(op_test_timeout_sec),
                        profile_name=",".join([sp.name for sp in stage_profiles]),
                        ctx=int(max([sp.ctx for sp in stage_profiles] or [0])),
                        candidate_hash=str(kernel_candidate.patch_hash) if kernel_candidate else "",
                        cache_mode=str(op_test_cache or "on"),
                        min_rows=int(max(1, op_test_min_rows)),
                        case_limit=int(max(0, op_test_case_limit)),
                        case_seed=int(max(0, op_test_case_seed)),
                        required=(profiling_mode_norm == "op_perf_required"),
                    )
                    if not bool(op_gate.get("success")):
                        compile_error = str(op_gate.get("classification") or "op_numeric_mismatch")
            except KernelPatchError as exc:
                compile_error = str(exc)

        if compile_error:
            compile_record = KernelCompileRecord(
                attempted=True,
                success=False,
                classification=(
                    "static_feasibility_reject"
                    if str(compile_error).startswith("static_feasibility_reject")
                    else "op_perf_regression"
                    if str(compile_error).startswith("op_perf_regression")
                    else str(op_gate.get("classification"))
                    if bool(op_gate.get("attempted")) and not bool(op_gate.get("success"))
                    else "patch_application_error"
                ),
                error=compile_error,
                compile_warmup_done=bool(compile_meta.get("compile_warmup_done", False)),
                pipeline_cache_key=str(compile_meta.get("pipeline_cache_key", "")),
                compile_time_ms=(
                    float(compile_meta["compile_time_ms"])
                    if isinstance(compile_meta.get("compile_time_ms"), (int, float))
                    else None
                ),
                toolchain_fingerprint=str(compile_meta.get("toolchain_fingerprint", "")),
            )
            correctness_record = KernelCorrectnessRecord(
                attempted=False,
                success=False,
                classification=(
                    "static_feasibility_reject"
                    if str(compile_error).startswith("static_feasibility_reject")
                    else "op_perf_regression"
                    if str(compile_error).startswith("op_perf_regression")
                    else str(op_gate.get("classification"))
                    if bool(op_gate.get("attempted")) and not bool(op_gate.get("success"))
                    else "not_attempted"
                ),
                details={
                    "feasibility": asdict(feasibility_record),
                    "op_gate": op_gate,
                    "candidate_op_profiles": candidate_op_profiles,
                    "baseline_op_profiles": op_profiles,
                    "op_perf_compare_rows": op_perf_compare_rows,
                    "op_perf_delta_pct": op_perf_delta_pct,
                    "op_perf_common_rows": int(op_perf_common_rows),
                    "op_perf_baseline_total_ms": op_perf_baseline_total_ms,
                    "op_perf_candidate_total_ms": op_perf_candidate_total_ms,
                    "op_perf_compare_key": op_perf_compare_key,
                    "op_perf_decision": op_perf_decision,
                    "op_perf_promoted": bool(op_perf_promoted),
                },
            )
            after = []
            after_dict = []
            delta = {"profiles": {}}
            pass_gate = False
            score = float("-1e9")
            valid_attempt = False
        else:
            after = benchmark.run_benchmarks(
                llama_cli=llama_cli,
                model_path=model.path,
                profile_mode=profile_mode,
                gate_mode=stage,
                prompts=prompts,
                resources_path=resources_path,
                extra_args=runtime_args,
                capture_raw_output=True,
                force_source_compile=force_source_compile,
                dispatch_attempt_id=f"{(study_tag or 'opt')}_attempt_{attempt_idx}_{stage}",
                dispatch_rule_id=candidate_dispatch_rule_id,
                dispatch_audit_dir=dispatch_audit_dir,
                candidate_resources_expected=bool(resources_path),
            )
            after_dict = benchmark.benchmark_results_to_dict(after)
            delta = _compute_delta(baseline_for_stage_dict, after_dict)
            compile_record = _aggregate_compile([classify_compile_record(r) for r in after])
            if compile_meta:
                compile_record.compile_warmup_done = bool(compile_meta.get("compile_warmup_done", False))
                compile_record.pipeline_cache_key = str(compile_meta.get("pipeline_cache_key", ""))
                compile_record.compile_time_ms = (
                    float(compile_meta["compile_time_ms"])
                    if isinstance(compile_meta.get("compile_time_ms"), (int, float))
                    else None
                )
                compile_record.toolchain_fingerprint = str(compile_meta.get("toolchain_fingerprint", ""))
            correctness_record = _aggregate_correctness(
                baseline_results=baseline_for_stage,
                tuned_results=after,
                strict_parity=strict_parity,
            )
            dispatch_failure = _dispatch_proof_failure(
                results=after,
                candidate_resources_expected=bool(resources_path),
            )
            if dispatch_failure:
                correctness_record = KernelCorrectnessRecord(
                    attempted=True,
                    success=False,
                    classification=dispatch_failure,
                    details={
                        "dispatch_audit": _first_dispatch_audit(after),
                    },
                )
            pass_gate = (
                compile_record.success and correctness_record.success and _passes_gate(delta)
            )
            score = _score_delta(delta) if (compile_record.success and correctness_record.success) else float("-1e9")
            valid_attempt = bool(compile_record.success and correctness_record.success)

        attempt_dispatch_audit = _first_dispatch_audit(after)
        attempt_dispatch_status = str(
            attempt_dispatch_audit.get("dispatch_audit_status") or ""
        )
        attempt_candidate_expected = bool(resources_path)
        attempt_candidate_used = attempt_dispatch_audit.get("candidate_resources_used")
        if not isinstance(attempt_candidate_used, bool):
            attempt_candidate_used = None
        attempt_entry = {
            "attempt": attempt_idx,
            "stage": stage,
            "candidate_name": candidate.candidate_name,
            "rationale": candidate.rationale,
            "kernel_overrides": candidate.kernel_overrides,
            "runtime_args": runtime_args,
            "template_mutations": template_mutations,
            "source_patches": source_patches,
            "kernel_candidate": kernel_candidate_dict(kernel_candidate) if kernel_candidate else {},
            "dispatch_rule_id": str(
                attempt_dispatch_audit.get("selected_dispatch_rule_id")
                or candidate_dispatch_rule_id
                or ""
            ),
            "dispatch_audit_status": attempt_dispatch_status,
            "candidate_resources_expected": attempt_candidate_expected,
            "candidate_resources_used": attempt_candidate_used,
            "dispatch_audit_path": str(attempt_dispatch_audit.get("dispatch_audit_path") or ""),
            "dispatch_audit_source": str(attempt_dispatch_audit.get("dispatch_audit_source") or ""),
            "metallib_source": str(attempt_dispatch_audit.get("metallib_source") or ""),
            "dispatch_audit": attempt_dispatch_audit,
            "optimized": {"results": after_dict},
            "delta": delta,
            "pass_gate": pass_gate,
            "score": score,
            "compile_record": asdict(compile_record),
            "correctness_record": asdict(correctness_record),
            "feasibility_record": asdict(feasibility_record),
            "op_gate": op_gate,
            "op_profiles": op_profiles,
            "candidate_op_profiles": candidate_op_profiles,
            "op_perf_delta_pct": op_perf_delta_pct,
            "op_perf_common_rows": int(op_perf_common_rows),
            "op_perf_baseline_total_ms": op_perf_baseline_total_ms,
            "op_perf_candidate_total_ms": op_perf_candidate_total_ms,
            "op_perf_compare_key": op_perf_compare_key,
            "op_perf_decision": op_perf_decision,
            "op_perf_promoted": bool(op_perf_promoted),
            "op_perf_compare_rows": op_perf_compare_rows,
            "stage0_feasibility": bool(stage0_feasibility),
            "stage1_op_sql": bool(stage1_op_sql),
            "stage2_logits_gate": bool(stage2_logits_gate),
            "op_perf_timeout_sec": float(op_perf_timeout_sec),
            "op_perf_cache": str(op_perf_cache),
            "op_perf_min_rows": int(max(1, op_perf_min_rows)),
            "op_perf_op_filter": str(op_perf_op_filter),
            "op_perf_case_limit": int(max(0, op_perf_case_limit)),
            "op_perf_case_seed": int(max(0, op_perf_case_seed)),
            "op_perf_warmup_iters": int(max(0, op_perf_warmup_iters)),
            "op_perf_bench_iters": int(max(0, op_perf_bench_iters)),
            "op_perf_reject_regression_pct": float(op_perf_reject_regression_pct),
            "op_perf_promote_topk": int(max(0, op_perf_promote_topk)),
            "op_test_timeout_sec": float(op_test_timeout_sec),
            "op_test_cache": str(op_test_cache),
            "op_test_min_rows": int(max(1, op_test_min_rows)),
            "op_test_case_limit": int(max(0, op_test_case_limit)),
            "op_test_case_seed": int(max(0, op_test_case_seed)),
            "provider": llm_meta.get("provider", ""),
            "model": llm_meta.get("model", ""),
            "constraint_repairs": list(llm_meta.get("constraint_repairs") or []),
            "allowed_params": dict(llm_meta.get("allowed_params") or allowed_params),
            "study_tag": study_tag,
            "started_at": started,
            "finished_at": time.time(),
            "valid": valid_attempt,
        }
        tuning_attempts.append(attempt_entry)
        if attempt_log_path is not None:
            _append_attempt_log(attempt_log_path, attempt_entry)

        if valid_attempt and (best is None or score > float(best.get("score", float("-inf")))):
            best = attempt_entry

        if dynamic_ladder and attempt_idx == 1:
            if valid_attempt:
                max_attempts = 1 + 5
                followup_stage = "full"
            else:
                max_attempts = 1 + 3
                followup_stage = "quick"

    if best is None:
        raise OptimizeError("LLM tuning produced no valid kernel/runtime candidates.")

    selected_stage = str(best.get("stage", gate_mode.lower())).lower()
    baseline_selected = get_stage_baseline(selected_stage)
    baseline_selected_dict = benchmark.benchmark_results_to_dict(baseline_selected)

    selected_kernel = best.get("kernel_candidate") or {}
    resources_source_dir = None
    kernel_patch_metadata: dict[str, Any] | None = None
    if selected_kernel:
        resources_source_dir = Path(str(selected_kernel.get("resources_dir", ""))).expanduser().resolve()
        kernel_patch_metadata = {
            "template_version": selected_kernel.get("template_version", ""),
            "patch_hash": selected_kernel.get("patch_hash", ""),
            "source_hash": selected_kernel.get("source_hash", ""),
            "template_mutations": selected_kernel.get("template_mutations", {}),
            "source_patches": selected_kernel.get("source_patches", []),
        }

    pack_id, pack_dir, _manifest = create_pack(
        llamacpp_root=llamacpp_root,
        device=device,
        model=model,
        profile_mode=profile_mode,
        gate_mode=gate_mode,
        cgins_version=cgins_version,
        llamacpp_commit=local_commit,
        bench_before={"results": baseline_selected_dict},
        bench_after=best["optimized"],
        kernel_overrides=best.get("kernel_overrides", {}),
        runtime_args=best.get("runtime_args", []),
        strict_guardrails=strict_guardrails,
        resources_source_dir=resources_source_dir,
        kernel_patch_metadata=kernel_patch_metadata,
        tuning_session={
            "attempt": best["attempt"],
            "stage": selected_stage,
            "candidate_name": best["candidate_name"],
            "rationale": best["rationale"],
            "provider": best.get("provider", ""),
            "model": best.get("model", ""),
            "score": best.get("score"),
            "compile_record": best.get("compile_record", {}),
            "correctness_record": best.get("correctness_record", {}),
        },
        reuse_policy=reuse_policy,
    )

    set_active_pack(model_sha=model.sha256, device_fingerprint=device.fingerprint, pack_id=pack_id)

    return OptimizeReport(
        success=True,
        reason=f"{commit_msg}; selected candidate={best['candidate_name']}",
        model_path=str(model.path),
        profile_mode=profile_mode,
        gate_mode=gate_mode,
        pack_id=pack_id,
        pack_dir=str(pack_dir),
        baseline={"results": baseline_selected_dict},
        optimized=best["optimized"],
        delta=best["delta"],
        pass_gate=bool(best["pass_gate"]),
        tuning={
            "mode": "llm_kernel" if kernel_mode != "none" else "llm_runtime",
            "kernel_mode": kernel_mode,
            "profile_set": profile_set_norm,
            "profiling_mode": profiling_mode_norm,
            "stage0_feasibility": bool(stage0_feasibility),
            "stage1_op_sql": bool(stage1_op_sql),
            "stage2_logits_gate": bool(stage2_logits_gate),
            "op_perf_timeout_sec": float(op_perf_timeout_sec),
            "op_perf_cache": str(op_perf_cache),
            "op_perf_min_rows": int(max(1, op_perf_min_rows)),
            "op_perf_op_filter": str(op_perf_op_filter),
            "op_perf_case_limit": int(max(0, op_perf_case_limit)),
            "op_perf_case_seed": int(max(0, op_perf_case_seed)),
            "op_perf_warmup_iters": int(max(0, op_perf_warmup_iters)),
            "op_perf_bench_iters": int(max(0, op_perf_bench_iters)),
            "op_perf_reject_regression_pct": float(op_perf_reject_regression_pct),
            "op_perf_promote_topk": int(max(0, op_perf_promote_topk)),
            "op_perf_promotions_used": int(stage1_promotions_used),
            "stage1_force_strict_regression": bool(stage1_force_strict_regression),
            "op_test_timeout_sec": float(op_test_timeout_sec),
            "op_test_cache": str(op_test_cache),
            "op_test_min_rows": int(max(1, op_test_min_rows)),
            "op_test_case_limit": int(max(0, op_test_case_limit)),
            "op_test_case_seed": int(max(0, op_test_case_seed)),
            "attempt_budget": max_attempts,
            "provider": last_llm_meta.get("provider", ""),
            "model": last_llm_meta.get("model", ""),
            "constraint_repairs": list(last_llm_meta.get("constraint_repairs") or []),
            "allowed_params": allowed_params,
            "attempts": tuning_attempts,
            "selected_attempt": int(best["attempt"]),
            "selected_stage": selected_stage,
            "study_tag": study_tag,
            "reuse_policy": reuse_policy,
        },
    )


def disable_active_pack_for_model(*, model_path: str, device_fingerprint: str) -> dict[str, Any]:
    model = probe_model(model_path)
    disable_pack(model_sha=model.sha256, device_fingerprint=device_fingerprint)
    return {
        "success": True,
        "model_sha256": model.sha256,
        "device_fingerprint": device_fingerprint,
    }


def export_pack_file(*, pack_dir: Path, out_path: Path) -> dict[str, Any]:
    out = export_pack(pack_dir, out_path)
    return {"success": True, "path": str(out)}
