from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
import zlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.auth.credentials import apply_auth_env, resolve_auth
from src.config import ensure_llm_config, load_config_data

from . import benchmark
from .compat import assert_supported_device, chip_family, ensure_llamacpp_commit, get_llamacpp_commit
from .constants import PASS_MAX_REGRESSION_PCT, configure_cache_root, current_cache_root
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
from .op_profile import (
    compare_stage1_op_profiles,
    resolve_test_backend_ops,
    run_op_correctness_checks,
    run_stage1_op_profile,
    suggest_ggml_ops_from_hotspots,
)
from .roofline import build_roofline_analysis
from .runtime_args import sanitize_runtime_args
from .tuner import LlmTuningError, propose_llm_candidate
from .types import (
    BenchmarkResult,
    ClaimDecision,
    HotspotAttributionRecord,
    KernelCorrectnessRecord,
    ScheduleRecord,
    StudyAttemptRecord,
    StudyRunRecord,
    StudySummary,
    WorkloadProfile,
)

SUPPORTED_ARMS = (
    "baseline",
    "flash",
    "oneshot",
    "iterative",
    "oneshot_kernel",
    "iterative_kernel",
)
DEFAULT_ARMS = ["baseline", "flash", "oneshot_kernel", "iterative_kernel"]
DEFAULT_PROFILES = ["chat", "long"]
FALLBACK_OPENAI_MODELS = ["gpt-5.3-codex", "gpt-5.2-codex"]
DEFAULT_ABBA_CYCLES = 4
DEFAULT_WARMUP_BLOCKS = 1
DEFAULT_DECODE_CLAIM_THRESHOLD_PCT = 30.0
PROFILING_MODE_HEURISTIC = "heuristic"
PROFILING_MODE_OP_PERF_REQUIRED = "op_perf_required"
PARITY_STAGE_NONE = "none"
PARITY_STAGE_NUMERIC = "numeric"
PARITY_STAGE_SEMANTIC = "semantic"
PARITY_STAGE_CLAIM = "claim"


class StudyError(RuntimeError):
    pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _power_state() -> dict[str, Any]:
    out = {
        "raw": "",
        "on_ac_power": None,
        "charging": None,
        "percent": None,
    }
    try:
        proc = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False)
    except Exception:
        return out

    text = (proc.stdout or "").strip()
    out["raw"] = text
    first = text.splitlines()[0].lower() if text else ""
    out["on_ac_power"] = "ac power" in first

    for line in text.splitlines()[1:]:
        lower = line.lower()
        if "%" in line:
            try:
                pct = line.split("%")[0].split()[-1]
                out["percent"] = int(pct)
            except Exception:
                pass
        if "charging" in lower:
            out["charging"] = True
        if "discharging" in lower:
            out["charging"] = False
        if "charged" in lower and out["charging"] is None:
            out["charging"] = None

    return out


def _resolve_profiles(raw: str) -> list[str]:
    if not raw:
        return list(DEFAULT_PROFILES)
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    allowed = {"chat", "long", "long_smoke", "long_claim"}
    bad = [x for x in items if x not in allowed]
    if bad:
        raise StudyError(f"Unsupported profiles: {', '.join(bad)}")
    return items


def _resolve_arms(raw: str) -> list[str]:
    if not raw:
        return list(DEFAULT_ARMS)
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    alias_map = {
        "oneshot": "oneshot_kernel",
        "iterative": "iterative_kernel",
    }
    items = [alias_map.get(x, x) for x in items]
    bad = [x for x in items if x not in SUPPORTED_ARMS]
    if bad:
        raise StudyError(f"Unsupported arms: {', '.join(bad)}")
    if "baseline" not in items:
        items = ["baseline"] + items
    dedup: list[str] = []
    for item in items:
        if item not in dedup:
            dedup.append(item)
    return dedup


def _kernel_budget_allocation(*, arms: list[str], kernel_total_budget: int) -> dict[str, int]:
    total = int(max(0, kernel_total_budget))
    allocation = {
        "oneshot_kernel": 0,
        "iterative_kernel": 0,
    }
    if total <= 0:
        return allocation

    has_oneshot = "oneshot_kernel" in arms
    has_iterative = "iterative_kernel" in arms

    if has_oneshot and has_iterative:
        allocation["oneshot_kernel"] = 1
        allocation["iterative_kernel"] = max(0, total - 1)
        return allocation
    if has_iterative:
        allocation["iterative_kernel"] = total
        return allocation
    if has_oneshot:
        allocation["oneshot_kernel"] = 1
    return allocation


def _dispatch_attempt_id(*parts: str) -> str:
    raw = ":".join(str(p).strip() for p in parts if str(p).strip())
    if not raw:
        return "run"
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_.:-")
    return token[:180] or "run"


def _dispatch_rule_id_for_patch(*, patch_hash: str, arm_id: str, attempt_id: str) -> str:
    base = str(patch_hash or "").strip()
    if not base:
        base = _dispatch_attempt_id(arm_id, attempt_id)
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_.-")
    return f"rule_{token[:64]}" if token else ""


def _load_matrix(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise StudyError(f"Matrix file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StudyError(f"Failed to parse matrix JSON: {exc}") from exc

    if isinstance(raw, list):
        models = raw
    elif isinstance(raw, dict) and isinstance(raw.get("models"), list):
        models = raw["models"]
    else:
        raise StudyError("Matrix must be a list or object with 'models' list")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(models):
        if isinstance(item, str):
            p = Path(item)
            entry = {
                "id": p.stem,
                "path": str((path.parent / p).resolve() if not p.is_absolute() else p),
            }
            out.append(entry)
            continue
        if not isinstance(item, dict):
            raise StudyError(f"Invalid model entry at index {idx}")

        model_path = str(item.get("path", "")).strip()
        if not model_path:
            raise StudyError(f"Model entry {idx} missing 'path'")
        p = Path(model_path)
        if not p.is_absolute():
            p = (path.parent / p).resolve()

        out.append(
            {
                "id": str(item.get("id", "")).strip() or p.stem,
                "path": str(p),
                "sha256": str(item.get("sha256", "")).strip().lower(),
                "url": str(item.get("url", "")).strip(),
            }
        )

    if not out:
        raise StudyError("Matrix did not include any models")
    return out


def _profile_for(name: str, gate_mode: str) -> WorkloadProfile:
    profiles = benchmark.workload_profiles(name, gate_mode)
    if not profiles:
        raise StudyError(f"Unable to resolve profile '{name}'")
    return profiles[0]


def _all_return_codes_zero(result: BenchmarkResult) -> bool:
    return all((run.get("return_code") == 0) for run in result.runs)


def _metrics_present(result: BenchmarkResult) -> bool:
    return (
        result.metrics.prefill_tokens_per_sec is not None
        and result.metrics.decode_tokens_per_sec is not None
    )


def _dispatch_audit_payload(run: dict[str, Any]) -> dict[str, Any]:
    return dict(run.get("dispatch_audit")) if isinstance(run.get("dispatch_audit"), dict) else {}


def _dispatch_run_field(run: dict[str, Any], key: str, default: Any = None) -> Any:
    value = run.get(key)
    if value is not None:
        return value
    dispatch = _dispatch_audit_payload(run)
    return dispatch.get(key, default)


def _dispatch_audit_failure_reason(run: dict[str, Any]) -> str:
    status = str(_dispatch_run_field(run, "dispatch_audit_status", "") or "").strip().lower()
    if status != "ok":
        if status == "parse_fail":
            return "dispatch_audit_parse_fail"
        if status in {"missing", "backend_noaudit"}:
            return "dispatch_audit_missing"
        return "dispatch_audit_missing"
    used = _dispatch_run_field(run, "candidate_resources_used", None)
    if used is not True:
        return "candidate_resources_not_used"
    return ""


def _long_prompt_target_ok(
    run: dict[str, Any],
    *,
    tolerance: int,
    require_proof: bool,
) -> tuple[bool, str]:
    target = run.get("prompt_tokens_target")
    if not isinstance(target, (int, float)) or float(target) <= 0:
        return (False, "long_prompt_target_unavailable") if require_proof else (True, "ok")
    actual = run.get("prompt_tokens_actual")
    if actual is None:
        actual = run.get("prompt_tokens")
    if not isinstance(actual, (int, float)):
        return (False, "long_prompt_target_unavailable") if require_proof else (True, "ok")
    if float(actual) + float(max(0, tolerance)) < float(target):
        return False, "long_prompt_target_miss"
    return True, "ok"


def _result_valid(
    result: BenchmarkResult,
    *,
    long_prompt_tolerance: int = 0,
    require_long_prompt_target: bool = False,
    candidate_resources_expected: bool = False,
) -> tuple[bool, str]:
    if not _all_return_codes_zero(result):
        return False, "non_zero_return_code"
    if not _metrics_present(result):
        return False, "missing_prefill_or_decode"
    if result.profile.name.startswith("long"):
        for run in result.runs:
            ok, reason = _long_prompt_target_ok(
                run,
                tolerance=long_prompt_tolerance,
                require_proof=require_long_prompt_target,
            )
            if not ok:
                return False, reason
    if candidate_resources_expected:
        for run in result.runs:
            dispatch_reason = _dispatch_audit_failure_reason(run)
            if dispatch_reason:
                return False, dispatch_reason
    return True, "ok"


def _delta_pct(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return ((after - before) / before) * 100.0


def _delta_payload(profile: str, before: BenchmarkResult, after: BenchmarkResult) -> dict[str, Any]:
    return {
        "profiles": {
            profile: {
                "prefill_uplift_pct": _delta_pct(
                    before.metrics.prefill_tokens_per_sec,
                    after.metrics.prefill_tokens_per_sec,
                ),
                "decode_uplift_pct": _delta_pct(
                    before.metrics.decode_tokens_per_sec,
                    after.metrics.decode_tokens_per_sec,
                ),
                "ttft_delta_pct": _delta_pct(before.metrics.ttft_ms, after.metrics.ttft_ms),
            }
        }
    }


def _score_delta(delta: dict[str, Any], profile: str) -> float:
    row = (delta.get("profiles") or {}).get(profile, {})
    decode = row.get("decode_uplift_pct")
    decode_primary = float(decode) if isinstance(decode, (int, float)) else float("-1e9")
    regressions = []
    for key in ("prefill_uplift_pct", "decode_uplift_pct"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value < 0:
            regressions.append(abs(float(value)))
    worst_regression_mag = max(regressions) if regressions else 0.0
    penalty = 2.0 * max(0.0, worst_regression_mag - PASS_MAX_REGRESSION_PCT)
    return decode_primary - penalty


def _pass_guardrail(delta: dict[str, Any], profile: str) -> bool:
    row = (delta.get("profiles") or {}).get(profile, {})
    values = [row.get("prefill_uplift_pct"), row.get("decode_uplift_pct")]
    regressions = [float(v) for v in values if isinstance(v, (int, float))]
    if not regressions:
        return False
    worst = min(regressions)
    return worst >= (-PASS_MAX_REGRESSION_PCT)


def _crossover_orders(a: str, b: str, *, cycles: int = DEFAULT_ABBA_CYCLES) -> list[tuple[str, str]]:
    pattern = [(a, b), (b, a), (b, a), (a, b)]
    n = max(1, int(cycles))
    out: list[tuple[str, str]] = []
    for _ in range(n):
        out.extend(pattern)
    return out


def _materialize_schedule(
    *,
    model_ids: list[str],
    profiles: list[str],
    arms: list[str],
    abba_cycles: int,
    warmup_blocks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order_idx = 0
    warmups = max(1, int(warmup_blocks))
    cycles = max(1, int(abba_cycles))

    for model_id in model_ids:
        for profile in profiles:
            for warm_i in range(1, warmups + 1):
                order_idx += 1
                row = ScheduleRecord(
                    model_id=model_id,
                    profile=profile,
                    arm_id="baseline",
                    stage="warmup",
                    block_id=f"warmup_{model_id}_{profile}_baseline_w{warm_i}",
                    cycle_index=0,
                    order_index=order_idx,
                    first_arm="baseline",
                    second_arm="baseline",
                    order_label="warmup",
                    is_warmup=True,
                )
                rows.append(asdict(row))

            for arm_id in [a for a in arms if a != "baseline"]:
                for warm_i in range(1, warmups + 1):
                    order_idx += 1
                    row = ScheduleRecord(
                        model_id=model_id,
                        profile=profile,
                        arm_id=arm_id,
                        stage="warmup",
                        block_id=f"warmup_{model_id}_{profile}_{arm_id}_w{warm_i}",
                        cycle_index=0,
                        order_index=order_idx,
                        first_arm=arm_id,
                        second_arm=arm_id,
                        order_label="warmup",
                        is_warmup=True,
                    )
                    rows.append(asdict(row))

                for block_i, (first_arm, second_arm) in enumerate(
                    _crossover_orders("baseline", arm_id, cycles=cycles), start=1
                ):
                    order_idx += 1
                    cycle_idx = ((block_i - 1) // 4) + 1
                    row = ScheduleRecord(
                        model_id=model_id,
                        profile=profile,
                        arm_id=arm_id,
                        stage="measure",
                        block_id=f"{model_id}_{profile}_{arm_id}_block_{block_i}",
                        cycle_index=cycle_idx,
                        order_index=order_idx,
                        first_arm=first_arm,
                        second_arm=second_arm,
                        order_label=f"{first_arm}->{second_arm}",
                        is_warmup=False,
                    )
                    rows.append(asdict(row))
    return rows


def generate_schedule_preview(
    *,
    matrix_path: Path,
    profiles: list[str],
    arms: list[str],
    abba_cycles: int,
    warmup_blocks: int,
) -> dict[str, Any]:
    matrix = _load_matrix(matrix_path)
    model_ids = [str(row.get("id", "")).strip() or Path(str(row.get("path", ""))).stem for row in matrix]
    resolved_profiles = _resolve_profiles(",".join([p for p in profiles if p]))
    resolved_arms = _resolve_arms(",".join([a for a in arms if a]))
    schedule_rows = _materialize_schedule(
        model_ids=model_ids,
        profiles=resolved_profiles,
        arms=resolved_arms,
        abba_cycles=int(max(1, abba_cycles)),
        warmup_blocks=int(max(1, warmup_blocks)),
    )
    return {
        "generated_at_utc": _utcnow(),
        "matrix_path": str(matrix_path),
        "models": model_ids,
        "profiles": resolved_profiles,
        "arms": resolved_arms,
        "abba_cycles": int(max(1, abba_cycles)),
        "warmup_blocks": int(max(1, warmup_blocks)),
        "rows": schedule_rows,
        "counts": {
            "rows_total": len(schedule_rows),
            "warmup_rows": sum(1 for r in schedule_rows if bool(r.get("is_warmup"))),
            "measurement_rows": sum(1 for r in schedule_rows if not bool(r.get("is_warmup"))),
        },
    }


def _extract_kernel_mentions(run: dict[str, Any]) -> list[str]:
    text = f"{run.get('stdout', '')}\n{run.get('stderr', '')}"
    if not text.strip():
        return []
    names = re.findall(r"\bkernel[_\s:'\"]+([A-Za-z0-9_]+)\b", text, flags=re.IGNORECASE)
    out: list[str] = []
    for name in names:
        value = str(name).strip()
        if value and value not in out:
            out.append(value)
    return out


def _infer_hotspot_ops(
    *,
    runtime_args: list[str],
    kernel_template_version: str,
    patch_hash: str,
    run: dict[str, Any],
) -> list[str]:
    ops: list[str] = []
    args = [str(a).strip() for a in runtime_args]
    if "--flash-attn" in args or "-fa" in args:
        ops.extend(["softmax", "attention"])
    if "-ub" in args or "--ubatch-size" in args:
        ops.append("mul_mv_ext")
    if "-b" in args or "--batch-size" in args:
        ops.append("mul_mv")
    if kernel_template_version or patch_hash:
        ops.extend(["mul_mv_q4_k", "mul_mv_q5_k", "rms_norm"])

    for mention in _extract_kernel_mentions(run):
        m = mention.lower()
        if "soft" in m:
            ops.append("softmax")
        if "norm" in m:
            ops.append("rms_norm")
        if "q4" in m:
            ops.append("mul_mv_q4_k")
        if "q5" in m:
            ops.append("mul_mv_q5_k")
        if "mul" in m and "q4" not in m and "q5" not in m:
            ops.append("mul_mv")
    uniq: list[str] = []
    for item in ops:
        v = str(item).strip()
        if v and v not in uniq:
            uniq.append(v)
    return uniq


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "stdev": None,
            "ci_low": None,
            "ci_high": None,
        }

    vals = [float(v) for v in values]
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(max(1, samples)):
        sample = [vals[rng.randrange(0, len(vals))] for _ in range(len(vals))]
        boots.append(statistics.mean(sample))
    boots.sort()

    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "ci_low": _quantile(boots, 0.025),
        "ci_high": _quantile(boots, 0.975),
    }


def _normal_survival(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _wilcoxon_signed_rank(values: list[float]) -> dict[str, Any]:
    diffs = [float(v) for v in values if isinstance(v, (int, float)) and float(v) != 0.0]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "w": None, "z": None, "p_value": 1.0}

    abs_vals = [abs(v) for v in diffs]
    order = sorted(range(n), key=lambda i: abs_vals[i])

    ranks = [0.0] * n
    tie_counts: list[int] = []
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and abs_vals[order[j + 1]] == abs_vals[order[i]]:
            j += 1
        count = j - i + 1
        avg_rank = (rank + (rank + count - 1)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        if count > 1:
            tie_counts.append(count)
        rank += count
        i = j + 1

    w_pos = sum(ranks[idx] for idx, d in enumerate(diffs) if d > 0)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if tie_counts:
        correction = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts) / 48.0
        var_w -= correction
    if var_w <= 0:
        return {"n": n, "w": w_pos, "z": 0.0, "p_value": 1.0}

    continuity = 0.5 if w_pos > mean_w else (-0.5 if w_pos < mean_w else 0.0)
    z = (w_pos - mean_w - continuity) / math.sqrt(var_w)
    p = min(1.0, max(0.0, 2.0 * _normal_survival(abs(z))))
    return {"n": n, "w": w_pos, "z": z, "p_value": p}


def _holm_correct(rows: list[dict[str, Any]], *, p_key: str, out_key: str) -> None:
    indexed = [
        (idx, float(row.get(p_key)))
        for idx, row in enumerate(rows)
        if isinstance(row.get(p_key), (int, float))
    ]
    if not indexed:
        return

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted_sorted: list[float] = [1.0] * m
    running_max = 0.0
    for i, (_, p) in enumerate(indexed):
        scaled = min(1.0, (m - i) * p)
        running_max = max(running_max, scaled)
        adjusted_sorted[i] = running_max

    for rank, (row_idx, _) in enumerate(indexed):
        rows[row_idx][out_key] = adjusted_sorted[rank]


def _resolve_llm_identity(required: bool) -> dict[str, Any]:
    cfg, _ = load_config_data()
    status = resolve_auth(config=cfg, env=dict(os.environ), runtime_context={"in_container": False})
    apply_auth_env(status, os.environ)
    provider = ensure_llm_config().strip().lower()

    payload = {
        "provider": provider,
        "model": "",
        "auth": status.to_dict(),
    }

    if not required:
        return payload

    if status.mode_effective == "unconfigured" or not provider:
        raise StudyError(f"No usable LLM auth configured: {status.reason}")

    if provider == "openai":
        current = (os.environ.get("OPENAI_MODEL") or "").strip()
        if not current:
            current = FALLBACK_OPENAI_MODELS[0]
            os.environ["OPENAI_MODEL"] = current
        payload["model"] = current
        payload["fallback_models"] = [m for m in FALLBACK_OPENAI_MODELS if m != current]
    elif provider == "anthropic":
        payload["model"] = (os.environ.get("ANTHROPIC_MODEL") or "claude-opus-4-5-20251101").strip()
    elif provider == "gemini":
        payload["model"] = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()

    return payload


def _looks_like_model_unavailable(error: str) -> bool:
    text = (error or "").lower()
    return any(
        token in text
        for token in [
            "404",
            "model",
            "not found",
            "does not exist",
            "unsupported",
        ]
    )


def _propose_candidate_with_fallback(
    *,
    device,
    model,
    profile_mode: str,
    gate_mode: str,
    baseline: list[dict[str, Any]],
    previous_attempts: list[dict[str, Any]],
    llm_identity: dict[str, Any],
    allowed_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    provider = str(llm_identity.get("provider", "")).strip().lower()
    if provider != "openai":
        return propose_llm_candidate(
            device=device,
            model=model,
            profile_mode=profile_mode,
            gate_mode=gate_mode,
            baseline=baseline,
            previous_attempts=previous_attempts,
            allowed_params=allowed_params,
        )

    last_exc: Exception | None = None
    model_sequence: list[str] = []
    current = (os.environ.get("OPENAI_MODEL") or "").strip()
    if current:
        model_sequence.append(current)
    for fallback in llm_identity.get("fallback_models", []):
        fb = str(fallback).strip()
        if fb and fb not in model_sequence:
            model_sequence.append(fb)

    if not model_sequence:
        model_sequence = list(FALLBACK_OPENAI_MODELS)

    for model_name in model_sequence:
        os.environ["OPENAI_MODEL"] = model_name
        try:
            candidate, meta = propose_llm_candidate(
                device=device,
                model=model,
                profile_mode=profile_mode,
                gate_mode=gate_mode,
                baseline=baseline,
                previous_attempts=previous_attempts,
                allowed_params=allowed_params,
            )
            llm_identity["model"] = model_name
            return candidate, meta
        except LlmTuningError as exc:
            last_exc = exc
            if not _looks_like_model_unavailable(str(exc)):
                break

    if last_exc is None:
        raise LlmTuningError("Failed to propose candidate")
    raise last_exc


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    import struct

    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _write_placeholder_png(path: Path, width: int = 64, height: int = 32) -> None:
    import struct

    raw = bytearray()
    for _ in range(height):
        raw.append(0)
        raw.extend(b"\xff\xff\xff" * width)
    compressed = zlib.compress(bytes(raw), level=9)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _write_basic_svg(path: Path, title: str, rows: list[dict[str, Any]], value_key: str, group_key: str) -> None:
    width = 1200
    height = 700
    margin = 80
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin

    labels = [str(r.get(group_key, "")) for r in rows]
    vals = [float(r.get(value_key, 0.0) or 0.0) for r in rows]
    if not vals:
        vals = [0.0]
    if not labels:
        labels = ["n/a"]
    y_min = min(vals + [0.0])
    y_max = max(vals + [0.0])
    span = max(1e-9, y_max - y_min)

    bar_count = max(1, len(vals))
    bar_w = chart_w / bar_count

    def y_pos(v: float) -> float:
        return margin + chart_h - ((v - y_min) / span) * chart_h

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{margin}" y="40" font-size="24" font-family="Arial">{title}</text>')
    zero_y = y_pos(0.0)
    lines.append(
        f'<line x1="{margin}" y1="{zero_y:.2f}" x2="{margin + chart_w}" y2="{zero_y:.2f}" stroke="#999" stroke-width="1"/>'
    )
    lines.append(
        f'<rect x="{margin}" y="{margin}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#222" stroke-width="1"/>'
    )

    for idx, value in enumerate(vals):
        x = margin + idx * bar_w + 4
        bar_inner_w = max(2.0, bar_w - 8)
        y0 = y_pos(0.0)
        yv = y_pos(value)
        top = min(y0, yv)
        h = max(1.0, abs(yv - y0))
        color = "#2b8cbe" if value >= 0 else "#de2d26"
        lines.append(
            f'<rect x="{x:.2f}" y="{top:.2f}" width="{bar_inner_w:.2f}" height="{h:.2f}" fill="{color}" opacity="0.85"/>'
        )
        label = labels[idx][:20]
        lines.append(
            f'<text x="{x:.2f}" y="{margin + chart_h + 18}" font-size="10" font-family="Arial" transform="rotate(35 {x:.2f},{margin + chart_h + 18})">{label}</text>'
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _write_ci_svg(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    mean_key: str,
    low_key: str,
    high_key: str,
) -> None:
    points: list[dict[str, Any]] = []
    for row in rows:
        mean_v = _to_float(row.get(mean_key))
        low_v = _to_float(row.get(low_key))
        high_v = _to_float(row.get(high_key))
        if mean_v is None or low_v is None or high_v is None:
            continue
        points.append(
            {
                "label": f"{row.get('model_id')}|{row.get('profile')}|{row.get('arm_id')}",
                "mean": mean_v,
                "low": low_v,
                "high": high_v,
            }
        )
    if not points:
        points = [{"label": "n/a", "mean": 0.0, "low": 0.0, "high": 0.0}]

    width = 1500
    height = 800
    margin = 90
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin

    lo = min([p["low"] for p in points] + [0.0])
    hi = max([p["high"] for p in points] + [0.0])
    span = max(1e-9, hi - lo)
    step = chart_w / max(1, len(points))

    def y_pos(v: float) -> float:
        return margin + chart_h - ((v - lo) / span) * chart_h

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{margin}" y="42" font-size="24" font-family="Arial">{title}</text>')
    zero_y = y_pos(0.0)
    lines.append(f'<line x1="{margin}" y1="{zero_y:.2f}" x2="{margin + chart_w}" y2="{zero_y:.2f}" stroke="#9a9a9a"/>')
    lines.append(f'<rect x="{margin}" y="{margin}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#222"/>')

    for idx, point in enumerate(points):
        xc = margin + idx * step + step / 2.0
        y_low = y_pos(point["low"])
        y_high = y_pos(point["high"])
        y_mean = y_pos(point["mean"])
        lines.append(f'<line x1="{xc:.2f}" y1="{y_low:.2f}" x2="{xc:.2f}" y2="{y_high:.2f}" stroke="#444" stroke-width="2"/>')
        lines.append(f'<line x1="{xc - 5:.2f}" y1="{y_low:.2f}" x2="{xc + 5:.2f}" y2="{y_low:.2f}" stroke="#444" stroke-width="2"/>')
        lines.append(f'<line x1="{xc - 5:.2f}" y1="{y_high:.2f}" x2="{xc + 5:.2f}" y2="{y_high:.2f}" stroke="#444" stroke-width="2"/>')
        lines.append(f'<circle cx="{xc:.2f}" cy="{y_mean:.2f}" r="3.5" fill="#0b6e4f"/>')
        label = point["label"][:28]
        lines.append(
            f'<text x="{xc:.2f}" y="{margin + chart_h + 18}" font-size="10" text-anchor="end" font-family="Arial" transform="rotate(45 {xc:.2f},{margin + chart_h + 18})">{label}</text>'
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_line_svg(path: Path, *, title: str, points: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    vals = [(int(p[x_key]), _to_float(p.get(y_key))) for p in points if isinstance(p.get(x_key), int) and _to_float(p.get(y_key)) is not None]
    vals = [(x, y) for x, y in vals if y is not None]
    if not vals:
        vals = [(1, 0.0)]

    width = 900
    height = 600
    margin = 80
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin

    x_vals = [v[0] for v in vals]
    y_vals = [float(v[1]) for v in vals]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals + [0.0]), max(y_vals + [0.0])
    x_span = max(1, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    def x_pos(v: int) -> float:
        return margin + ((v - x_min) / x_span) * chart_w

    def y_pos(v: float) -> float:
        return margin + chart_h - ((v - y_min) / y_span) * chart_h

    poly = " ".join([f"{x_pos(x):.2f},{y_pos(float(y)):.2f}" for x, y in vals])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="40" font-size="22" font-family="Arial">{title}</text>',
        f'<rect x="{margin}" y="{margin}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#222"/>',
        f'<polyline points="{poly}" fill="none" stroke="#136f63" stroke-width="2"/>',
    ]
    for x, y in vals:
        lines.append(f'<circle cx="{x_pos(x):.2f}" cy="{y_pos(float(y)):.2f}" r="3.2" fill="#136f63"/>')
        lines.append(f'<text x="{x_pos(x):.2f}" y="{margin + chart_h + 18}" font-size="10" text-anchor="middle" font-family="Arial">{x}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_heatmap_svg(path: Path, *, title: str, rows: list[dict[str, Any]]) -> None:
    y_labels = sorted({f"{r.get('model_id')}|{r.get('profile')}" for r in rows})
    x_labels = sorted({str(r.get("arm_id")) for r in rows})
    if not y_labels or not x_labels:
        y_labels = ["n/a"]
        x_labels = ["n/a"]

    grid: dict[tuple[str, str], float] = {}
    vals: list[float] = []
    for row in rows:
        key = (f"{row.get('model_id')}|{row.get('profile')}", str(row.get("arm_id")))
        v = _to_float(row.get("mean_delta_decode_pct"))
        if v is not None:
            grid[key] = v
            vals.append(v)

    lo = min(vals + [-1.0])
    hi = max(vals + [1.0])
    span = max(1e-9, hi - lo)

    def color_for(v: float | None) -> str:
        if v is None:
            return "#eeeeee"
        t = (v - lo) / span
        r = int(220 * (1.0 - t))
        g = int(90 + 130 * t)
        b = int(90 + 90 * (1.0 - abs(t - 0.5) * 2.0))
        return f"rgb({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))})"

    width = 1400
    height = 900
    margin_x = 220
    margin_y = 110
    cell_w = max(40.0, (width - margin_x - 40) / max(1, len(x_labels)))
    cell_h = max(24.0, (height - margin_y - 80) / max(1, len(y_labels)))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_x}" y="42" font-size="24" font-family="Arial">{title}</text>',
    ]
    for xi, xlab in enumerate(x_labels):
        x = margin_x + xi * cell_w
        lines.append(f'<text x="{x + cell_w / 2:.2f}" y="{margin_y - 12}" font-size="11" text-anchor="middle" font-family="Arial">{xlab}</text>')
    for yi, ylab in enumerate(y_labels):
        y = margin_y + yi * cell_h
        lines.append(f'<text x="{margin_x - 8}" y="{y + cell_h * 0.7:.2f}" font-size="10" text-anchor="end" font-family="Arial">{ylab[:40]}</text>')
        for xi, xlab in enumerate(x_labels):
            x = margin_x + xi * cell_w
            v = grid.get((ylab, xlab))
            fill = color_for(v)
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}" stroke="#ffffff"/>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_basic_plots(
    plot_dir: Path,
    ci_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]] | None = None,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    decode_rows = [
        {
            "group": f"{r.get('model_id')}|{r.get('profile')}|{r.get('arm_id')}",
            "mean": r.get("mean_delta_decode_pct", 0.0),
        }
        for r in ci_rows
    ]
    prefill_rows = [
        {
            "group": f"{r.get('model_id')}|{r.get('profile')}|{r.get('arm_id')}",
            "mean": r.get("mean_delta_prefill_pct", 0.0),
        }
        for r in ci_rows
    ]
    order_rows = [
        {
            "group": f"{r.get('model_id')}|{r.get('profile')}|{r.get('arm_id')}|{r.get('order_label')}",
            "mean": r.get("delta_decode_pct", 0.0),
        }
        for r in paired_rows
        if isinstance(r.get("delta_decode_pct"), (int, float))
    ]

    _write_basic_svg(plot_dir / "decode_effects.svg", "Decode Delta (%)", decode_rows, "mean", "group")
    _write_basic_svg(plot_dir / "prefill_effects.svg", "Prefill Delta (%)", prefill_rows, "mean", "group")
    _write_basic_svg(plot_dir / "crossover_decode.svg", "Crossover Decode Delta (%)", order_rows, "mean", "group")
    _write_ci_svg(
        plot_dir / "arm_ci_decode.svg",
        title="Per-Model/Arm Decode Delta with 95% CI",
        rows=ci_rows,
        mean_key="mean_delta_decode_pct",
        low_key="ci95_decode_low",
        high_key="ci95_decode_high",
    )
    _write_ci_svg(
        plot_dir / "arm_ci_prefill.svg",
        title="Per-Model/Arm Prefill Delta with 95% CI",
        rows=ci_rows,
        mean_key="mean_delta_prefill_pct",
        low_key="ci95_prefill_low",
        high_key="ci95_prefill_high",
    )

    _write_heatmap_svg(
        plot_dir / "regression_heatmap.svg",
        title="Decode Delta Heatmap (model/profile x arm)",
        rows=ci_rows,
    )

    attempt_points: list[dict[str, Any]] = []
    if attempt_rows:
        by_attempt: dict[int, list[float]] = {}
        for row in attempt_rows:
            if str(row.get("arm_id")) not in {"iterative", "iterative_kernel"}:
                continue
            score = _to_float(row.get("score"))
            if score is None:
                continue
            attempt_id = str(row.get("attempt_id", ""))
            suffix = attempt_id.rsplit("_", 1)[-1]
            try:
                idx = int(suffix)
            except Exception:
                continue
            by_attempt.setdefault(idx, []).append(score)
        for idx in sorted(by_attempt):
            vals = by_attempt[idx]
            attempt_points.append({"attempt": idx, "mean_score": statistics.mean(vals)})
    _write_line_svg(
        plot_dir / "iterative_convergence.svg",
        title="Iterative Arm Convergence (mean score by attempt)",
        points=attempt_points,
        x_key="attempt",
        y_key="mean_score",
    )

    for stem in [
        "decode_effects",
        "prefill_effects",
        "crossover_decode",
        "arm_ci_decode",
        "arm_ci_prefill",
        "regression_heatmap",
        "iterative_convergence",
    ]:
        _write_placeholder_png(plot_dir / f"{stem}.png")


def render_study_plots(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    ci_csv_path = output_dir / "ci_results.csv"
    paired_csv_path = output_dir / "paired_deltas.csv"
    if not ci_csv_path.exists() or not paired_csv_path.exists():
        raise StudyError(
            f"Expected ci_results.csv and paired_deltas.csv under {output_dir}"
        )

    def _read_csv(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    ci_rows = _read_csv(ci_csv_path)
    paired_rows = _read_csv(paired_csv_path)

    def _to_float_inplace(rows: list[dict[str, Any]], keys: list[str]) -> None:
        for row in rows:
            for key in keys:
                value = row.get(key)
                if value in ("", None):
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except Exception:
                    pass

    _to_float_inplace(
        ci_rows,
        [
            "mean_delta_prefill_pct",
            "mean_delta_decode_pct",
            "ci95_prefill_low",
            "ci95_prefill_high",
            "ci95_decode_low",
            "ci95_decode_high",
        ],
    )
    _to_float_inplace(
        paired_rows,
        ["delta_prefill_pct", "delta_decode_pct", "delta_ttft_pct"],
    )

    attempts_rows: list[dict[str, Any]] = []
    attempts_path = output_dir / "attempts.jsonl"
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                attempts_rows.append(json.loads(text))
            except Exception:
                continue

    plots_dir = output_dir / "plots"
    _write_basic_plots(plots_dir, ci_rows, paired_rows, attempts_rows)

    manifest_path = output_dir / "study_manifest.json"
    summary_path = output_dir / "summary.json"
    if manifest_path.exists() and summary_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8")).get("summary", {})
            summary_obj = StudySummary(
                success=bool(summary_payload.get("success", True)),
                output_dir=str(summary_payload.get("output_dir", output_dir)),
                generated_at_utc=str(summary_payload.get("generated_at_utc", _utcnow())),
                models_tested=int(summary_payload.get("models_tested", 0)),
                profiles_tested=list(summary_payload.get("profiles_tested", [])),
                arms_tested=list(summary_payload.get("arms_tested", [])),
                total_blocks=int(summary_payload.get("total_blocks", 0)),
                total_runs=int(summary_payload.get("total_runs", 0)),
                invalid_blocks=int(summary_payload.get("invalid_blocks", 0)),
                invalid_runs=int(summary_payload.get("invalid_runs", 0)),
                block_failure_rate=_to_float(summary_payload.get("block_failure_rate")),
                run_failure_rate=_to_float(summary_payload.get("run_failure_rate")),
                claim_matrix=dict(summary_payload.get("claim_matrix", {})),
            )
            pval_rows: list[dict[str, Any]] = []
            pval_path = output_dir / "pvalues_corrected.csv"
            if pval_path.exists():
                pval_rows = _read_csv(pval_path)
            _write_methods_note(
                output_dir=output_dir,
                manifest=manifest,
                summary=summary_obj,
                ci_rows=ci_rows,
                p_rows=pval_rows,
            )
        except Exception:
            pass

    return {
        "success": True,
        "plots_dir": str(plots_dir),
        "files": sorted([p.name for p in plots_dir.glob("*") if p.is_file()]),
    }


def _write_methods_note(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    summary: StudySummary,
    ci_rows: list[dict[str, Any]],
    p_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Apple Silicon Validation Methods Note")
    lines.append("")
    lines.append("## Plain-Language Summary")
    lines.append(
        "This study measures whether optimization arms improve llama.cpp prefill/decode throughput on Apple Silicon."
    )
    lines.append(
        "Each comparison is run with blocked ABBA crossover ordering to reduce ordering bias, and warmup runs are excluded from statistical claims."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- Generated at (UTC): {summary.generated_at_utc}")
    lines.append(f"- Git commit: {manifest.get('git_commit', '')}")
    lines.append(f"- llama.cpp commit: {manifest.get('llamacpp_commit', '')}")
    lines.append(f"- Profiles: {', '.join(manifest.get('profiles', []))}")
    lines.append(f"- Arms: {', '.join(manifest.get('arms', []))}")
    lines.append(f"- Gate mode: {manifest.get('gate_mode', '')}")
    lines.append(f"- Bootstrap samples: {manifest.get('bootstrap_samples', '')}")
    lines.append(f"- Cooldown seconds: {manifest.get('cooldown_seconds', '')}")
    lines.append(f"- Power precheck: {json.dumps(manifest.get('power_precheck', {}), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Exclusion Rules")
    lines.append("- Any non-zero subprocess return code invalidates that block.")
    lines.append("- Missing prefill or decode metrics invalidates that block.")
    lines.append("- Invalid blocks are logged and excluded from inferential statistics.")
    lines.append("")
    lines.append("## Statistical Appendix")
    lines.append("- Primary metrics: prefill and decode delta (%) from matched paired blocks.")
    lines.append("- Confidence intervals: bootstrap 95% CI with replacement.")
    lines.append("- Significance: Wilcoxon signed-rank test, Holm correction across comparisons.")
    lines.append(
        f"- Claim rule: decode CI95 lower bound must exceed {manifest.get('decode_claim_threshold_pct', DEFAULT_DECODE_CLAIM_THRESHOLD_PCT)}% and regression guardrails must pass."
    )
    lines.append("")
    lines.append("## Study Totals")
    lines.append(f"- Models tested: {summary.models_tested}")
    lines.append(f"- Total paired blocks: {summary.total_blocks}")
    lines.append(f"- Total subprocess runs: {summary.total_runs}")
    lines.append(f"- Invalid blocks: {summary.invalid_blocks} (rate={summary.block_failure_rate})")
    lines.append(f"- Invalid runs: {summary.invalid_runs} (rate={summary.run_failure_rate})")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- study_manifest.json")
    lines.append("- schedule.json")
    lines.append("- runs_raw.jsonl")
    lines.append("- attempts.jsonl")
    lines.append("- summary.json")
    lines.append("- throughput_report.json")
    lines.append("- throughput_report.csv")
    lines.append("- claim_decisions.json")
    lines.append("- hotspots.json")
    lines.append("- hotspots_op_perf.json")
    lines.append("- op_profiles.json")
    lines.append("- roofline_analysis.json")
    lines.append("- exclusions.csv")
    lines.append("- metrics_by_block.csv")
    lines.append("- paired_deltas.csv")
    lines.append("- ci_results.csv")
    lines.append("- pvalues_corrected.csv")
    lines.append("- plots/")

    note_path = output_dir / "methods_note.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")


def _build_throughput_report(attempt_records: list[dict[str, Any]]) -> dict[str, Any]:
    def _gate_stats(
        *,
        pass_field: str,
        evaluated_field: str,
    ) -> tuple[int, int, float | None]:
        evaluated = 0
        passed = 0
        for row in attempt_records:
            evaluated_raw = row.get(evaluated_field)
            pass_raw = row.get(pass_field)
            if isinstance(evaluated_raw, bool):
                if not evaluated_raw:
                    continue
                evaluated += 1
                if pass_raw is True:
                    passed += 1
                continue
            # Backward compatibility for attempt records emitted before *_evaluated fields existed.
            if isinstance(pass_raw, bool):
                evaluated += 1
                if pass_raw:
                    passed += 1
        rate = (float(passed / evaluated) if evaluated > 0 else None)
        return int(evaluated), int(passed), rate

    rejection_counter: Counter[str] = Counter()
    stderr_counter: Counter[str] = Counter()
    kernel_counter: Counter[str] = Counter()
    dispatch_rule_counter: Counter[str] = Counter()
    dispatch_status_counter: Counter[str] = Counter()

    dispatch_total = 0
    dispatch_metallib_present = 0
    candidate_resources_expected_count = 0
    candidate_resources_used_count = 0

    def _dispatch_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
        value = row.get(key)
        if value is not None:
            return value
        dispatch = row.get("dispatch_audit")
        if isinstance(dispatch, dict):
            return dispatch.get(key, default)
        return default

    for row in attempt_records:
        if not bool(row.get("valid")):
            reason = str(row.get("error") or "").strip()
            if not reason:
                reason = str((row.get("compile_record") or {}).get("classification") or "").strip()
            if not reason:
                reason = str((row.get("correctness_record") or {}).get("classification") or "").strip()
            reason_key = (reason.split(":", 1)[0] or "unknown").strip() or "unknown"
            rejection_counter[reason_key] += 1

        stderr_hash = str((row.get("compile_record") or {}).get("stderr_hash") or row.get("compile_stderr_hash") or "").strip()
        if stderr_hash:
            stderr_counter[stderr_hash] += 1

        dispatch = row.get("dispatch_audit")
        dispatch_status = str(_dispatch_value(row, "dispatch_audit_status", "") or "").strip()
        if dispatch_status:
            dispatch_status_counter[dispatch_status] += 1
        expected_raw = _dispatch_value(row, "candidate_resources_expected", None)
        expected = bool(expected_raw) if isinstance(expected_raw, bool) else False
        if expected:
            candidate_resources_expected_count += 1
            used_raw = _dispatch_value(row, "candidate_resources_used", None)
            if used_raw is True:
                candidate_resources_used_count += 1

        has_dispatch_data = (
            (isinstance(dispatch, dict) and bool(dispatch))
            or bool(dispatch_status)
        )
        if has_dispatch_data:
            dispatch_total += 1
            if _dispatch_value(row, "metallib_present", None) is True:
                dispatch_metallib_present += 1
            rid = str(
                _dispatch_value(row, "selected_dispatch_rule_id", "")
                or row.get("dispatch_rule_id")
                or ""
            ).strip()
            if rid:
                dispatch_rule_counter[rid] += 1
            for item in list(_dispatch_value(row, "top_kernels", []) or []):
                if not isinstance(item, dict):
                    continue
                kernel_name = str(item.get("kernel", "")).strip()
                mentions = item.get("mentions")
                if not kernel_name:
                    continue
                if isinstance(mentions, (int, float)):
                    kernel_counter[kernel_name] += int(max(1, int(mentions)))
                else:
                    kernel_counter[kernel_name] += 1

    gate_a_evaluated, gate_a_pass_count, compile_success_rate = _gate_stats(
        pass_field="gate_a_pass",
        evaluated_field="gate_a_evaluated",
    )
    gate_b_evaluated, gate_b_pass_count, gate_b_pass_rate = _gate_stats(
        pass_field="gate_b_pass",
        evaluated_field="gate_b_evaluated",
    )
    gate_c_evaluated, gate_c_pass_count, gate_c_pass_rate = _gate_stats(
        pass_field="gate_c_pass",
        evaluated_field="gate_c_evaluated",
    )
    gate_d_evaluated, gate_d_pass_count, gate_d_pass_rate = _gate_stats(
        pass_field="gate_d_pass",
        evaluated_field="gate_d_evaluated",
    )

    unique_rule_ids = sorted(dispatch_rule_counter.keys())
    dispatch_rule_coverage = {
        "dispatch_runs_total": int(dispatch_total),
        "unique_rule_ids": unique_rule_ids,
        "unique_rule_count": int(len(unique_rule_ids)),
        "coverage_rate": (
            float(len(unique_rule_ids) / dispatch_total)
            if dispatch_total > 0
            else None
        ),
    }

    return {
        "attempts_total": int(len(attempt_records)),
        "gate_a_evaluated": gate_a_evaluated,
        "gate_b_evaluated": gate_b_evaluated,
        "gate_c_evaluated": gate_c_evaluated,
        "gate_d_evaluated": gate_d_evaluated,
        "gate_a_pass_count": gate_a_pass_count,
        "gate_b_pass_count": gate_b_pass_count,
        "gate_c_pass_count": gate_c_pass_count,
        "gate_d_pass_count": gate_d_pass_count,
        "compile_success_rate": compile_success_rate,
        "gate_b_pass_rate": gate_b_pass_rate,
        "gate_c_pass_rate": gate_c_pass_rate,
        "gate_d_pass_rate": gate_d_pass_rate,
        "top_rejection_reasons": [
            {"reason": key, "count": int(count)}
            for key, count in rejection_counter.most_common(5)
        ],
        "top_compile_stderr_hashes": [
            {"stderr_hash": key, "count": int(count)}
            for key, count in stderr_counter.most_common(5)
        ],
        "top_dispatched_kernels": [
            {"kernel": key, "mentions": int(count)}
            for key, count in kernel_counter.most_common(5)
        ],
        "dispatch_audit_status_counts": {
            key: int(dispatch_status_counter[key]) for key in sorted(dispatch_status_counter.keys())
        },
        "candidate_resources_expected_count": int(candidate_resources_expected_count),
        "candidate_resources_used_count": int(candidate_resources_used_count),
        "candidate_resources_used_rate": (
            float(candidate_resources_used_count / candidate_resources_expected_count)
            if candidate_resources_expected_count > 0
            else None
        ),
        "audit_missing_count": int(dispatch_status_counter.get("missing", 0)),
        "audit_parse_fail_count": int(dispatch_status_counter.get("parse_fail", 0)),
        "backend_noaudit_count": int(dispatch_status_counter.get("backend_noaudit", 0)),
        "dispatch_metallib_load_rate": (
            float(dispatch_metallib_present / dispatch_total)
            if dispatch_total > 0
            else None
        ),
        "dispatch_rule_coverage": dispatch_rule_coverage,
    }


def _metrics_row(
    *,
    model_id: str,
    model_sha: str,
    profile: str,
    arm_id: str,
    block_id: str,
    order_index: int,
    order_label: str,
    is_warmup: bool,
    runtime_args: list[str],
    result: BenchmarkResult,
    kernel_template_version: str = "",
    patch_hash: str = "",
    candidate_source_hash: str = "",
    compile_result: str = "",
    compile_stderr_hash: str = "",
    compile_warmup_done: bool = False,
    pipeline_cache_key: str = "",
    compile_time_ms: float | None = None,
    profiling_mode: str = "heuristic",
    long_prompt_tolerance: int = 0,
    require_long_prompt_target: bool = False,
    candidate_resources_expected: bool = False,
) -> dict[str, Any]:
    valid, reason = _result_valid(
        result,
        long_prompt_tolerance=long_prompt_tolerance,
        require_long_prompt_target=require_long_prompt_target,
        candidate_resources_expected=candidate_resources_expected,
    )
    first_run = result.runs[0] if result.runs else {}
    dispatch_audit = (
        dict(first_run.get("dispatch_audit"))
        if isinstance(first_run.get("dispatch_audit"), dict)
        else {}
    )
    dispatch_audit_status = str(
        first_run.get("dispatch_audit_status")
        or dispatch_audit.get("dispatch_audit_status")
        or ""
    )
    candidate_expected = first_run.get("candidate_resources_expected")
    if not isinstance(candidate_expected, bool):
        if isinstance(dispatch_audit.get("candidate_resources_expected"), bool):
            candidate_expected = bool(dispatch_audit.get("candidate_resources_expected"))
        else:
            candidate_expected = bool(candidate_resources_expected)
    candidate_used = first_run.get("candidate_resources_used")
    if not isinstance(candidate_used, bool):
        if isinstance(dispatch_audit.get("candidate_resources_used"), bool):
            candidate_used = bool(dispatch_audit.get("candidate_resources_used"))
        else:
            candidate_used = None
    return {
        "model_id": model_id,
        "model_sha256": model_sha,
        "profile": profile,
        "arm_id": arm_id,
        "block_id": block_id,
        "order_index": order_index,
        "order_label": order_label,
        "is_warmup": is_warmup,
        "runtime_args": json.dumps(runtime_args),
        "kernel_template_version": kernel_template_version,
        "patch_hash": patch_hash,
        "candidate_source_hash": candidate_source_hash,
        "compile_result": compile_result,
        "compile_stderr_hash": compile_stderr_hash,
        "compile_warmup_done": compile_warmup_done,
        "pipeline_cache_key": pipeline_cache_key,
        "compile_time_ms": compile_time_ms,
        "profiling_mode": profiling_mode,
        "op_perf_status": first_run.get("op_perf_status", ""),
        "op_perf_cache_hit": first_run.get("op_perf_cache_hit"),
        "op_perf_rows_emitted": first_run.get("op_perf_rows_emitted"),
        "op_perf_cache_key": first_run.get("op_perf_cache_key", ""),
        "prompt_cache_mode": first_run.get("prompt_cache_mode", ""),
        "prompt_cache_file": first_run.get("prompt_cache_file", ""),
        "prompt_cache_build_elapsed_ms": first_run.get("prompt_cache_build_elapsed_ms"),
        "prompt_cache_isolated": first_run.get("prompt_cache_isolated"),
        "dispatch_rule_id": first_run.get(
            "dispatch_rule_id",
            dispatch_audit.get("selected_dispatch_rule_id", ""),
        ),
        "metallib_path": dispatch_audit.get("metallib_path", ""),
        "metallib_present": dispatch_audit.get("metallib_present"),
        "metallib_source": first_run.get("metallib_source", dispatch_audit.get("metallib_source", "")),
        "dispatch_audit_status": dispatch_audit_status,
        "candidate_resources_expected": candidate_expected,
        "candidate_resources_used": candidate_used,
        "dispatch_audit_path": first_run.get("dispatch_audit_path", dispatch_audit.get("dispatch_audit_path", "")),
        "dispatch_audit_source": first_run.get("dispatch_audit_source", dispatch_audit.get("dispatch_audit_source", "")),
        "top_dispatched_kernels": json.dumps(dispatch_audit.get("top_kernels", [])),
        "prompt_tokens_target": first_run.get("prompt_tokens_target"),
        "prompt_tokens_actual": first_run.get("prompt_tokens_actual", first_run.get("prompt_tokens")),
        "prompt_tokens_target_met": first_run.get("prompt_tokens_target_met"),
        "prefill_tokens_per_sec": result.metrics.prefill_tokens_per_sec,
        "decode_tokens_per_sec": result.metrics.decode_tokens_per_sec,
        "ttft_ms": result.metrics.ttft_ms,
        "p50_token_latency_ms": result.metrics.p50_token_latency_ms,
        "p95_token_latency_ms": result.metrics.p95_token_latency_ms,
        "peak_memory_mib": result.metrics.peak_memory_mib,
        "elapsed_seconds": result.elapsed_seconds,
        "all_return_codes_zero": _all_return_codes_zero(result),
        "all_metrics_present": _metrics_present(result),
        "valid": valid,
        "invalid_reason": reason,
    }


def _emit_runs_raw(
    *,
    out_path: Path,
    model_id: str,
    model_path: str,
    model_sha: str,
    profile: str,
    arm_id: str,
    attempt_id: str,
    block_id: str,
    is_warmup: bool,
    runtime_args: list[str],
    result: BenchmarkResult,
    git_commit: str,
    llamacpp_commit: str,
    power_state_snapshot: dict[str, Any],
    order_index_start: int,
    kernel_template_version: str = "",
    patch_hash: str = "",
    candidate_source_hash: str = "",
    compile_result: str = "",
    compile_stderr_hash: str = "",
    compile_warmup_done: bool = False,
    pipeline_cache_key: str = "",
    compile_time_ms: float | None = None,
    profiling_mode: str = "heuristic",
    long_prompt_tolerance: int = 0,
    require_long_prompt_target: bool = False,
    candidate_resources_expected: bool = False,
    allow_heuristic_hotspots: bool = True,
    hotspot_override_ops: list[str] | None = None,
    hotspot_rows: list[dict[str, Any]] | None = None,
    exclusion_rows: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    idx = order_index_start
    invalid_local = 0
    for run_i, run in enumerate(result.runs):
        run_id = f"{model_id}:{profile}:{arm_id}:{block_id}:{run_i}:{idx}"
        live_power = _power_state()
        if live_power.get("on_ac_power") is None:
            live_power = dict(power_state_snapshot)
        run_valid = bool(run.get("return_code") == 0 and run.get("prefill_tokens_per_sec") is not None and run.get("decode_tokens_per_sec") is not None)
        run_reason = "non_zero_return_code_or_missing_metrics"
        if run_valid and str(profile).startswith("long"):
            target_ok, target_reason = _long_prompt_target_ok(
                run,
                tolerance=long_prompt_tolerance,
                require_proof=require_long_prompt_target,
            )
            if not target_ok:
                run_valid = False
                run_reason = target_reason
        if run_valid and candidate_resources_expected:
            dispatch_reason = _dispatch_audit_failure_reason(run)
            if dispatch_reason:
                run_valid = False
                run_reason = dispatch_reason
        if not run_valid:
            invalid_local += 1
            if exclusion_rows is not None:
                exclusion_rows.append(
                    {
                        "kind": "run_invalid",
                        "run_id": run_id,
                        "block_id": block_id,
                        "model_id": model_id,
                        "profile": profile,
                        "arm_id": arm_id,
                        "order_index": idx,
                        "reason": run_reason,
                        "return_code": run.get("return_code"),
                    }
                )
        dispatch_audit = (
            dict(run.get("dispatch_audit"))
            if isinstance(run.get("dispatch_audit"), dict)
            else {}
        )
        dispatch_audit_status = str(
            run.get("dispatch_audit_status")
            or dispatch_audit.get("dispatch_audit_status")
            or ""
        )
        candidate_expected = run.get("candidate_resources_expected")
        if not isinstance(candidate_expected, bool):
            if isinstance(dispatch_audit.get("candidate_resources_expected"), bool):
                candidate_expected = bool(dispatch_audit.get("candidate_resources_expected"))
            else:
                candidate_expected = bool(candidate_resources_expected)
        candidate_used = run.get("candidate_resources_used")
        if not isinstance(candidate_used, bool):
            if isinstance(dispatch_audit.get("candidate_resources_used"), bool):
                candidate_used = bool(dispatch_audit.get("candidate_resources_used"))
            else:
                candidate_used = None
        rec = StudyRunRecord(
            run_id=run_id,
            block_id=block_id,
            is_warmup=is_warmup,
            model_id=model_id,
            model_path=model_path,
            model_sha256=model_sha,
            profile=profile,
            arm_id=arm_id,
            attempt_id=attempt_id,
            order_index=idx,
            wall_clock_utc=_utcnow(),
            git_commit=git_commit,
            llamacpp_commit=llamacpp_commit,
            kernel_template_version=kernel_template_version,
            patch_hash=patch_hash,
            candidate_source_hash=candidate_source_hash,
            compile_result=compile_result,
            compile_stderr_hash=compile_stderr_hash,
            compile_warmup_done=compile_warmup_done,
            pipeline_cache_key=pipeline_cache_key,
            compile_time_ms=compile_time_ms,
            prompt_tokens_target=(
                int(run["prompt_tokens_target"])
                if isinstance(run.get("prompt_tokens_target"), (int, float))
                else None
            ),
            prompt_tokens_actual=(
                int(run["prompt_tokens_actual"])
                if isinstance(run.get("prompt_tokens_actual"), (int, float))
                else (
                    int(run["prompt_tokens"])
                    if isinstance(run.get("prompt_tokens"), (int, float))
                    else None
                )
            ),
            prompt_tokens_target_met=(
                bool(run.get("prompt_tokens_target_met"))
                if isinstance(run.get("prompt_tokens_target_met"), bool)
                else None
            ),
            runtime_args=list(runtime_args),
            power_state=dict(live_power),
            metrics={
                "prefill_tokens_per_sec": run.get("prefill_tokens_per_sec"),
                "decode_tokens_per_sec": run.get("decode_tokens_per_sec"),
                "ttft_ms": run.get("ttft_ms"),
                "token_latency_ms": run.get("token_latency_ms"),
                "peak_memory_mib": run.get("peak_memory_mib"),
                "prompt_tokens_target": run.get("prompt_tokens_target"),
                "prompt_tokens_actual": run.get("prompt_tokens_actual", run.get("prompt_tokens")),
                "prompt_tokens_target_met": run.get("prompt_tokens_target_met"),
            },
            raw_run=run,
            profiling_mode=profiling_mode,
            op_perf_status=str(run.get("op_perf_status", "")),
            op_perf_cache_hit=(
                bool(run.get("op_perf_cache_hit"))
                if isinstance(run.get("op_perf_cache_hit"), bool)
                else None
            ),
            op_perf_rows_emitted=(
                int(run.get("op_perf_rows_emitted"))
                if isinstance(run.get("op_perf_rows_emitted"), (int, float))
                else None
            ),
            op_perf_cache_key=str(run.get("op_perf_cache_key", "")),
            prompt_cache_mode=str(run.get("prompt_cache_mode", "")),
            prompt_cache_file=str(run.get("prompt_cache_file", "")),
            prompt_cache_build_elapsed_ms=(
                float(run.get("prompt_cache_build_elapsed_ms"))
                if isinstance(run.get("prompt_cache_build_elapsed_ms"), (int, float))
                else None
            ),
            prompt_cache_isolated=(
                bool(run.get("prompt_cache_isolated"))
                if isinstance(run.get("prompt_cache_isolated"), bool)
                else None
            ),
            dispatch_audit_status=dispatch_audit_status,
            candidate_resources_expected=candidate_expected,
            candidate_resources_used=candidate_used,
            dispatch_audit_path=str(
                run.get("dispatch_audit_path")
                or dispatch_audit.get("dispatch_audit_path")
                or ""
            ),
            dispatch_audit_source=str(
                run.get("dispatch_audit_source")
                or dispatch_audit.get("dispatch_audit_source")
                or ""
            ),
            metallib_source=str(
                run.get("metallib_source")
                or dispatch_audit.get("metallib_source")
                or ""
            ),
            dispatch_audit=dispatch_audit,
        )
        _append_jsonl(out_path, asdict(rec))
        if hotspot_rows is not None:
            hotspot_ops = list(hotspot_override_ops) if hotspot_override_ops else []
            if not hotspot_ops and allow_heuristic_hotspots:
                hotspot_ops = _infer_hotspot_ops(
                    runtime_args=runtime_args,
                    kernel_template_version=kernel_template_version,
                    patch_hash=patch_hash,
                    run=run,
                )
            hot = HotspotAttributionRecord(
                run_id=run_id,
                model_id=model_id,
                profile=profile,
                arm_id=arm_id,
                order_index=idx,
                decode_tokens_per_sec=run.get("decode_tokens_per_sec"),
                prefill_tokens_per_sec=run.get("prefill_tokens_per_sec"),
                runtime_args=list(runtime_args),
                hotspot_ops=hotspot_ops,
                source=(
                    "op_profile"
                    if hotspot_override_ops
                    else ("heuristic+output" if allow_heuristic_hotspots else "none")
                ),
                profiling_mode=profiling_mode,
                wall_clock_utc=_utcnow(),
                details={
                    "kernel_template_version": kernel_template_version,
                    "patch_hash": patch_hash,
                    "compile_result": compile_result,
                },
            )
            hotspot_rows.append(asdict(hot))
        idx += 1
    return idx, invalid_local


def _tune_arm_args(
    *,
    arm_id: str,
    model_id: str,
    model,
    profile: WorkloadProfile,
    baseline_result: BenchmarkResult,
    device,
    llama_cli: Path,
    prompts: list[str],
    attempts_out: Path,
    runs_raw_out: Path,
    git_commit: str,
    llamacpp_commit: str,
    power_state: dict[str, Any],
    order_index_start: int,
    llamacpp_root: Path,
    llm_identity: dict[str, Any],
    allowed_params: dict[str, Any] | None = None,
    gate_mode: str,
    candidate_cache_dir: Path,
    dispatch_audit_dir: Path | None = None,
    kernel_mode: str,
    strict_parity: bool,
    kernel_attempt_budget: int = 0,
    long_prompt_tolerance: int = 0,
    require_long_prompt_target: bool = False,
    allow_heuristic_hotspots: bool = True,
    op_hotspot_ops: list[str] | None = None,
    profiling_mode: str = "heuristic",
    baseline_op_profile: dict[str, Any] | None = None,
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
    prompt_cache_enabled: bool = False,
    prompt_cache_root: Path | None = None,
    official_claim_mode: bool = False,
    stage1_force_strict_regression: bool = False,
    attempt_log_path: Path | None = None,
    attempt_records: list[dict[str, Any]] | None = None,
    hotspot_rows: list[dict[str, Any]] | None = None,
    exclusion_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, int, int]:
    baseline_by_stage: dict[str, BenchmarkResult] = {gate_mode.lower(): baseline_result}

    prev_attempts: list[dict[str, Any]] = []
    best_score = float("-1e9")
    best_config: dict[str, Any] | None = None
    invalid_runs = 0

    budget_override = int(max(0, kernel_attempt_budget))
    max_attempts = budget_override if budget_override > 0 else 1
    followup_stage = gate_mode.lower()

    idx = order_index_start
    attempt_idx = 0
    stage1_promotions_used = 0
    stage1_promote_limit = int(max(0, op_perf_promote_topk))
    if stage1_force_strict_regression or official_claim_mode:
        stage1_promote_limit = 0

    prompt_cache_dir = (
        Path(prompt_cache_root).expanduser().resolve()
        if prompt_cache_root is not None
        else (candidate_cache_dir / "prompt_cache")
    )
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)

    def _candidate_prompt_cache_path(tag: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag.strip())[:120] or "cache"
        return prompt_cache_dir / f"{model.sha256[:12]}_{profile.name}_{arm_id}_{safe}.promptcache"

    while attempt_idx < max_attempts:
        attempt_idx += 1
        attempt_id = f"{arm_id}_attempt_{attempt_idx}"
        stage = gate_mode.lower()
        if arm_id in {"iterative", "iterative_kernel"}:
            stage = "quick" if attempt_idx == 1 else followup_stage

        stage_profile = profile
        if stage != gate_mode.lower():
            stage_profile = _profile_for(profile.name, stage)
        stage_baseline = baseline_by_stage.get(stage)
        if stage_baseline is None:
            use_prompt_cache = bool(
                prompt_cache_enabled and (stage_profile.name == "long_smoke") and (not official_claim_mode)
            )
            baseline_prompt_cache = _candidate_prompt_cache_path(f"{stage}_baseline")
            stage_baseline = benchmark.run_profile_benchmark(
                llama_cli=llama_cli,
                model_path=model.path,
                profile=stage_profile,
                prompts=prompts,
                resources_path=None,
                extra_args=[],
                capture_raw_output=True,
                enforce_long_prompt_target=require_long_prompt_target,
                long_prompt_token_tolerance=long_prompt_tolerance,
                prompt_cache_path=baseline_prompt_cache if use_prompt_cache else None,
                prompt_cache_ro=bool(use_prompt_cache),
                prompt_cache_all=bool(use_prompt_cache),
                build_prompt_cache_first=bool(use_prompt_cache),
                dispatch_attempt_id=_dispatch_attempt_id(
                    model_id,
                    profile.name,
                    arm_id,
                    attempt_id,
                    stage,
                    "baseline",
                ),
                dispatch_rule_id="",
                dispatch_audit_dir=dispatch_audit_dir,
                candidate_resources_expected=False,
            )
            idx, invalid_local = _emit_runs_raw(
                out_path=runs_raw_out,
                model_id=model_id,
                model_path=str(model.path),
                model_sha=model.sha256,
                profile=profile.name,
                arm_id="baseline",
                attempt_id=attempt_id,
                block_id=f"tuning_baseline_{arm_id}_{stage}",
                is_warmup=False,
                runtime_args=[],
                result=stage_baseline,
                git_commit=git_commit,
                llamacpp_commit=llamacpp_commit,
                power_state_snapshot=power_state,
                order_index_start=idx,
                profiling_mode=profiling_mode,
                long_prompt_tolerance=long_prompt_tolerance,
                require_long_prompt_target=require_long_prompt_target,
                candidate_resources_expected=False,
                allow_heuristic_hotspots=allow_heuristic_hotspots,
                hotspot_override_ops=op_hotspot_ops,
                hotspot_rows=hotspot_rows,
                exclusion_rows=exclusion_rows,
            )
            invalid_runs += invalid_local
            baseline_by_stage[stage] = stage_baseline
        baseline_dict = benchmark.benchmark_results_to_dict([stage_baseline])

        proposal_kwargs = {
            "device": device,
            "model": model,
            "profile_mode": profile.name,
            "gate_mode": stage,
            "baseline": baseline_dict,
            "previous_attempts": prev_attempts,
            "llm_identity": llm_identity,
            "allowed_params": allowed_params,
        }
        try:
            try:
                candidate, llm_meta = _propose_candidate_with_fallback(**proposal_kwargs)
            except TypeError as exc:
                # Backward-compatible with tests/mocks still using the legacy proposer signature.
                if "unexpected keyword argument 'allowed_params'" in str(exc):
                    proposal_kwargs.pop("allowed_params", None)
                    candidate, llm_meta = _propose_candidate_with_fallback(**proposal_kwargs)
                else:
                    raise
        except Exception as exc:
            record = StudyAttemptRecord(
                model_id=model_id,
                model_sha256=model.sha256,
                profile=profile.name,
                arm_id=arm_id,
                attempt_id=attempt_id,
                wall_clock_utc=_utcnow(),
                git_commit=git_commit,
                llamacpp_commit=llamacpp_commit,
                order_index=idx,
                provider=str(llm_identity.get("provider", "")),
                model=str(llm_identity.get("model", "")),
                candidate_name="",
                rationale="",
                stage=stage,
                constraint_repairs=[],
                allowed_params=dict(allowed_params or {}),
                runtime_args=[],
                kernel_overrides={},
                power_state=_power_state(),
                benchmark={},
                valid=False,
                error=str(exc),
                score=None,
                delta={},
            )
            _append_jsonl(attempts_out, asdict(record))
            if attempt_records is not None:
                attempt_records.append(asdict(record))
            if exclusion_rows is not None:
                exclusion_rows.append(
                    {
                        "kind": "attempt_invalid",
                        "attempt_id": attempt_id,
                        "model_id": model_id,
                        "profile": profile.name,
                        "arm_id": arm_id,
                        "order_index": idx,
                        "reason": f"candidate_proposal_error:{exc}",
                    }
                )
            prev_attempts.append(
                {
                    "candidate_name": "",
                    "score": float("-1e9"),
                    "pass_gate": False,
                    "delta": {},
                }
            )
            if arm_id in {"oneshot", "oneshot_kernel"}:
                break
            if (
                budget_override <= 0
                and arm_id in {"iterative", "iterative_kernel"}
                and attempt_idx == 1
                and kernel_mode != "oneshot"
            ):
                max_attempts = 1 + 3
                followup_stage = "quick"
            continue

        runtime_args = sanitize_runtime_args(candidate.runtime_args)
        template_mutations = getattr(candidate, "template_mutations", {}) or {}
        source_patches = getattr(candidate, "source_patches", []) or []
        resources_path: Path | None = None
        kernel_candidate: dict[str, Any] = {}
        compile_meta: dict[str, Any] = {}
        patch_error = ""
        op_check: dict[str, Any] = {
            "attempted": False,
            "success": True,
            "classification": "not_attempted",
            "reason": "",
            "rows": [],
        }
        candidate_op_profile: dict[str, Any] = {}
        op_perf_compare: dict[str, Any] = {}
        op_perf_delta_pct: float | None = None
        op_perf_common_rows: int = 0
        op_perf_baseline_total_ms: float | None = None
        op_perf_candidate_total_ms: float | None = None
        op_perf_compare_key = ""
        op_perf_decision = "not_evaluated"
        op_perf_promoted = False
        candidate_dispatch_rule_id = ""
        feasibility_record = evaluate_candidate_feasibility(
            device=device,
            template_mutations=template_mutations,
            kernel_overrides=candidate.kernel_overrides,
            allowed_params=allowed_params,
        )
        if not feasibility_record.success:
            patch_error = f"{feasibility_record.classification}:{';'.join(feasibility_record.reasons)}"

        arm_is_kernel = arm_id.endswith("_kernel") and kernel_mode != "none"
        if arm_is_kernel and not patch_error:
            try:
                patched = build_kernel_patch_candidate(
                    llamacpp_root=llamacpp_root,
                    candidate_cache_dir=candidate_cache_dir,
                    candidate_id=f"{model.sha256[:10]}-{profile.name}-{arm_id}-{attempt_id}",
                    template_mutations=template_mutations,
                    source_patches=source_patches,
                )
                kernel_candidate = kernel_candidate_dict(patched)
                resources_path = Path(patched.resources_dir)
                compile_meta = benchmark.prepare_candidate_resources_for_benchmark(
                    resources_path=resources_path,
                    llamacpp_commit=llamacpp_commit,
                    chip_family=chip_family(device.chip),
                    macos_version=device.macos_version,
                    source_hash=str(kernel_candidate.get("source_hash", "")),
                    candidate_hash=str(kernel_candidate.get("patch_hash", "")),
                )
                if not bool(compile_meta.get("success")):
                    patch_error = str(compile_meta.get("error") or compile_meta.get("classification") or "compile_warmup_failed")
                else:
                    candidate_dispatch_rule_id = _dispatch_rule_id_for_patch(
                        patch_hash=str(kernel_candidate.get("patch_hash", "")),
                        arm_id=arm_id,
                        attempt_id=attempt_id,
                    )
                    candidate_hash = str(kernel_candidate.get("patch_hash", ""))
                    candidate_op_profile = run_stage1_op_profile(
                        llamacpp_root=llamacpp_root,
                        model_path=model.path,
                        profile=stage_profile,
                        profiling_mode=profiling_mode,
                        rank_metric="time",
                        op_filter=op_perf_op_filter,
                        timeout_sec=float(op_perf_timeout_sec),
                        cache_mode=op_perf_cache,
                        min_rows=int(max(1, op_perf_min_rows)),
                        resources_path=resources_path,
                        candidate_hash=candidate_hash,
                        perf_case_limit=int(max(0, op_perf_case_limit)),
                        perf_case_seed=int(max(0, op_perf_case_seed)),
                        perf_warmup_iters=int(max(0, op_perf_warmup_iters)),
                        perf_bench_iters=int(max(0, op_perf_bench_iters)),
                    )
                    if profiling_mode == PROFILING_MODE_OP_PERF_REQUIRED and not bool(candidate_op_profile.get("success")):
                        patch_error = str(candidate_op_profile.get("reason") or "op_perf_required_candidate_failed")
                    if not patch_error and isinstance(baseline_op_profile, dict):
                        b_rows = list(baseline_op_profile.get("ops") or [])
                        c_rows = list(candidate_op_profile.get("ops") or [])
                        op_perf_compare = compare_stage1_op_profiles(
                            baseline_ops=b_rows,
                            candidate_ops=c_rows,
                        )
                        op_perf_compare_key = str(op_perf_compare.get("compare_key") or "")
                        op_perf_delta_pct = (
                            float(op_perf_compare.get("delta_pct"))
                            if isinstance(op_perf_compare.get("delta_pct"), (int, float))
                            else None
                        )
                        op_perf_common_rows = int(op_perf_compare.get("common_rows") or 0)
                        op_perf_baseline_total_ms = (
                            float(op_perf_compare.get("baseline_total_ms"))
                            if isinstance(op_perf_compare.get("baseline_total_ms"), (int, float))
                            else None
                        )
                        op_perf_candidate_total_ms = (
                            float(op_perf_compare.get("candidate_total_ms"))
                            if isinstance(op_perf_compare.get("candidate_total_ms"), (int, float))
                            else None
                        )
                        if op_perf_common_rows <= 0:
                            op_perf_decision = "op_perf_uncomparable"
                        elif not isinstance(op_perf_delta_pct, float):
                            op_perf_decision = "op_perf_uncomparable"
                        else:
                            reject_cutoff = -abs(float(op_perf_reject_regression_pct))
                            if op_perf_delta_pct < reject_cutoff:
                                if stage1_promotions_used < stage1_promote_limit:
                                    stage1_promotions_used += 1
                                    op_perf_promoted = True
                                    op_perf_decision = "promoted_hard_regression"
                                else:
                                    op_perf_decision = "op_perf_regression_reject"
                                    patch_error = f"op_perf_regression:{op_perf_delta_pct:.3f}"
                            elif op_perf_delta_pct < 0.0:
                                op_perf_decision = "stage1_neutral"
                            else:
                                op_perf_decision = "stage1_pass"
                    elif not patch_error:
                        op_perf_decision = "op_perf_uncomparable"
                    if patch_error:
                        # Skip expensive model-level checks if op-perf already failed.
                        pass
                    else:
                        op_check = run_op_correctness_checks(
                            llamacpp_root=llamacpp_root,
                            ops=suggest_ggml_ops_from_hotspots(op_hotspot_ops or []),
                            resources_path=resources_path,
                            backend="Metal",
                            max_ops=3,
                            timeout_s=float(op_test_timeout_sec),
                            profile_name=stage_profile.name,
                            ctx=int(stage_profile.ctx),
                            candidate_hash=str(kernel_candidate.get("patch_hash", "")),
                            cache_mode=op_test_cache,
                            min_rows=int(max(1, op_test_min_rows)),
                            case_limit=int(max(0, op_test_case_limit)),
                            case_seed=int(max(0, op_test_case_seed)),
                            required=(profiling_mode == PROFILING_MODE_OP_PERF_REQUIRED),
                        )
                        if not bool(op_check.get("success")):
                            # In heuristic mode we still execute a timed benchmark run to
                            # collect dispatch-proof telemetry, while keeping the attempt
                            # invalid through correctness classification below.
                            if profiling_mode == PROFILING_MODE_OP_PERF_REQUIRED:
                                patch_error = str(op_check.get("classification") or "op_numeric_mismatch")
            except KernelPatchError as exc:
                patch_error = str(exc)

        if patch_error:
            tuned_result = BenchmarkResult(
                profile=stage_profile,
                metrics=stage_baseline.metrics,
                elapsed_seconds=0.0,
                runs=[
                    {
                        "return_code": 1,
                        "prefill_tokens_per_sec": None,
                        "decode_tokens_per_sec": None,
                        "stderr": patch_error,
                        "stdout": "",
                        "command": "",
                    }
                ],
            )
        else:
            use_prompt_cache = bool(
                prompt_cache_enabled and (stage_profile.name == "long_smoke") and (not official_claim_mode)
            )
            candidate_tag = str(kernel_candidate.get("patch_hash") or candidate.candidate_name or attempt_id)
            candidate_cache_path = _candidate_prompt_cache_path(f"{stage}_{candidate_tag}")
            tuned_result = benchmark.run_profile_benchmark(
                llama_cli=llama_cli,
                model_path=model.path,
                profile=stage_profile,
                prompts=prompts,
                resources_path=resources_path,
                extra_args=runtime_args,
                capture_raw_output=True,
                force_source_compile=bool(resources_path and not bool(compile_meta.get("compile_warmup_done"))),
                enforce_long_prompt_target=require_long_prompt_target,
                long_prompt_token_tolerance=long_prompt_tolerance,
                prompt_cache_path=candidate_cache_path if use_prompt_cache else None,
                prompt_cache_ro=bool(use_prompt_cache),
                prompt_cache_all=bool(use_prompt_cache),
                build_prompt_cache_first=bool(use_prompt_cache),
                dispatch_attempt_id=_dispatch_attempt_id(
                    model_id,
                    profile.name,
                    arm_id,
                    attempt_id,
                    stage,
                    "candidate",
                ),
                dispatch_rule_id=candidate_dispatch_rule_id,
                dispatch_audit_dir=dispatch_audit_dir,
                candidate_resources_expected=bool(resources_path),
            )

        compile_record = classify_compile_record(tuned_result, compile_meta=compile_meta)
        if candidate_op_profile:
            for run in tuned_result.runs:
                run["op_perf_status"] = str(candidate_op_profile.get("status", ""))
                run["op_perf_cache_hit"] = bool(candidate_op_profile.get("cache_hit")) if "cache_hit" in candidate_op_profile else None
                run["op_perf_rows_emitted"] = (
                    int(candidate_op_profile.get("rows_emitted"))
                    if isinstance(candidate_op_profile.get("rows_emitted"), (int, float))
                    else None
                )
                run["op_perf_cache_key"] = str(candidate_op_profile.get("cache_key", ""))
        if patch_error.startswith("static_feasibility_reject"):
            compile_record.classification = "static_feasibility_reject"
            compile_record.error = patch_error
        elif patch_error.startswith("op_perf_regression"):
            compile_record.classification = "op_perf_regression"
            compile_record.error = patch_error
        elif bool(op_check.get("attempted")) and not bool(op_check.get("success")):
            compile_record.classification = str(op_check.get("classification") or "op_numeric_mismatch")
            compile_record.error = str(op_check.get("reason") or compile_record.classification)
        if patch_error.startswith("static_feasibility_reject"):
            correctness_record = KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification="static_feasibility_reject",
                details={"feasibility": asdict(feasibility_record)},
            )
        elif patch_error.startswith("op_perf_regression"):
            correctness_record = KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification="op_perf_regression",
                details={
                    "baseline_op_profile": dict(baseline_op_profile or {}),
                    "candidate_op_profile": candidate_op_profile,
                    "op_perf_compare": op_perf_compare,
                    "op_perf_delta_pct": op_perf_delta_pct,
                },
            )
        elif bool(op_check.get("attempted")) and not bool(op_check.get("success")):
            correctness_record = KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification=str(op_check.get("classification") or "op_numeric_mismatch"),
                details={"op_check": op_check},
            )
        else:
            correctness_record = classify_correctness_record(
                baseline=stage_baseline,
                candidate=tuned_result,
                strict_parity=strict_parity,
            )
        idx, invalid_local = _emit_runs_raw(
            out_path=runs_raw_out,
            model_id=model_id,
            model_path=str(model.path),
            model_sha=model.sha256,
            profile=profile.name,
            arm_id=arm_id,
            attempt_id=attempt_id,
            block_id=f"tuning_{attempt_id}",
            is_warmup=False,
            runtime_args=runtime_args,
            result=tuned_result,
            git_commit=git_commit,
            llamacpp_commit=llamacpp_commit,
            power_state_snapshot=power_state,
            order_index_start=idx,
            kernel_template_version=str(kernel_candidate.get("template_version", "")),
            patch_hash=str(kernel_candidate.get("patch_hash", "")),
            candidate_source_hash=str(kernel_candidate.get("source_hash", "")),
            compile_result=compile_record.classification,
            compile_stderr_hash=compile_record.stderr_hash,
            compile_warmup_done=compile_record.compile_warmup_done,
            pipeline_cache_key=compile_record.pipeline_cache_key,
            compile_time_ms=compile_record.compile_time_ms,
            profiling_mode=profiling_mode,
            long_prompt_tolerance=long_prompt_tolerance,
            require_long_prompt_target=require_long_prompt_target,
            candidate_resources_expected=bool(resources_path),
            allow_heuristic_hotspots=allow_heuristic_hotspots,
            hotspot_override_ops=op_hotspot_ops,
            hotspot_rows=hotspot_rows,
            exclusion_rows=exclusion_rows,
        )
        invalid_runs += invalid_local

        delta = _delta_payload(profile.name, stage_baseline, tuned_result)
        score = _score_delta(delta, profile.name)
        valid, reason = _result_valid(
            tuned_result,
            long_prompt_tolerance=long_prompt_tolerance,
            require_long_prompt_target=require_long_prompt_target,
            candidate_resources_expected=bool(resources_path),
        )
        if not compile_record.success:
            valid = False
            reason = compile_record.classification
        if not correctness_record.success:
            valid = False
            reason = correctness_record.classification
        if not valid and exclusion_rows is not None:
            exclusion_rows.append(
                {
                    "kind": "attempt_invalid",
                    "attempt_id": attempt_id,
                    "model_id": model_id,
                    "profile": profile.name,
                    "arm_id": arm_id,
                    "order_index": idx,
                    "reason": reason,
                    "compile_classification": compile_record.classification,
                    "correctness_classification": correctness_record.classification,
                    "feasibility_classification": feasibility_record.classification,
                }
            )
        pass_gate = valid and _pass_guardrail(delta, profile.name)

        if valid and score > best_score:
            best_score = score
            best_config = {
                "runtime_args": runtime_args,
                "resources_path": str(resources_path) if resources_path else "",
                "kernel_candidate": kernel_candidate,
                "candidate_name": candidate.candidate_name,
                "rationale": candidate.rationale,
                "score": score,
                "stage": stage,
                "compile_record": asdict(compile_record),
                "correctness_record": asdict(correctness_record),
                "feasibility_record": asdict(feasibility_record),
                "candidate_op_profile": candidate_op_profile,
                "baseline_op_profile": dict(baseline_op_profile or {}),
                "op_perf_compare": op_perf_compare,
                "op_perf_delta_pct": op_perf_delta_pct,
                "op_perf_common_rows": int(op_perf_common_rows),
                "op_perf_baseline_total_ms": op_perf_baseline_total_ms,
                "op_perf_candidate_total_ms": op_perf_candidate_total_ms,
                "op_perf_compare_key": op_perf_compare_key,
                "op_perf_decision": op_perf_decision,
                "op_perf_promoted": bool(op_perf_promoted),
                "compile_warmup_done": compile_record.compile_warmup_done,
                "pipeline_cache_key": compile_record.pipeline_cache_key,
                "compile_time_ms": compile_record.compile_time_ms,
            }

        gate_a_evaluated = bool(isinstance(compile_meta, dict) and ("success" in compile_meta))
        gate_a_pass = bool(compile_meta.get("compile_warmup_done")) if gate_a_evaluated else None
        gate_b_evaluated = bool(op_check.get("attempted"))
        gate_b_pass = bool(op_check.get("success")) if gate_b_evaluated else None
        parity_evaluated = not bool(patch_error)
        gate_c_evaluated = bool(parity_evaluated)
        gate_c_pass = bool(correctness_record.success) if gate_c_evaluated else None
        gate_d_evaluated = bool(parity_evaluated)
        gate_d_pass = bool(pass_gate) if gate_d_evaluated else None
        first_tuned_run = tuned_result.runs[0] if tuned_result.runs else {}
        attempt_dispatch_audit = (
            dict(first_tuned_run.get("dispatch_audit"))
            if isinstance(first_tuned_run.get("dispatch_audit"), dict)
            else {}
        )
        attempt_dispatch_status = str(
            first_tuned_run.get("dispatch_audit_status")
            or attempt_dispatch_audit.get("dispatch_audit_status")
            or ""
        )
        attempt_candidate_expected = first_tuned_run.get("candidate_resources_expected")
        if not isinstance(attempt_candidate_expected, bool):
            if isinstance(attempt_dispatch_audit.get("candidate_resources_expected"), bool):
                attempt_candidate_expected = bool(attempt_dispatch_audit.get("candidate_resources_expected"))
            else:
                attempt_candidate_expected = bool(resources_path)
        attempt_candidate_used = first_tuned_run.get("candidate_resources_used")
        if not isinstance(attempt_candidate_used, bool):
            if isinstance(attempt_dispatch_audit.get("candidate_resources_used"), bool):
                attempt_candidate_used = bool(attempt_dispatch_audit.get("candidate_resources_used"))
            else:
                attempt_candidate_used = None

        record = StudyAttemptRecord(
            model_id=model_id,
            model_sha256=model.sha256,
            profile=profile.name,
            arm_id=arm_id,
            attempt_id=attempt_id,
            wall_clock_utc=_utcnow(),
            git_commit=git_commit,
            llamacpp_commit=llamacpp_commit,
            order_index=idx,
            provider=str(llm_meta.get("provider", "")),
            model=str(llm_meta.get("model", "")),
            candidate_name=candidate.candidate_name,
            rationale=candidate.rationale,
            stage=stage,
            constraint_repairs=list(llm_meta.get("constraint_repairs") or []),
            allowed_params=dict(llm_meta.get("allowed_params") or allowed_params or {}),
            kernel_template_version=str(kernel_candidate.get("template_version", "")),
            patch_hash=str(kernel_candidate.get("patch_hash", "")),
            candidate_source_hash=str(kernel_candidate.get("source_hash", "")),
            compile_record=asdict(compile_record),
            correctness_record=asdict(correctness_record),
            feasibility_record=asdict(feasibility_record),
            compile_warmup_done=compile_record.compile_warmup_done,
            pipeline_cache_key=compile_record.pipeline_cache_key,
            compile_time_ms=compile_record.compile_time_ms,
            runtime_args=runtime_args,
            kernel_overrides=candidate.kernel_overrides,
            power_state=_power_state(),
            benchmark={
                "prefill_tokens_per_sec": tuned_result.metrics.prefill_tokens_per_sec,
                "decode_tokens_per_sec": tuned_result.metrics.decode_tokens_per_sec,
                "ttft_ms": tuned_result.metrics.ttft_ms,
                "p50_token_latency_ms": tuned_result.metrics.p50_token_latency_ms,
                "p95_token_latency_ms": tuned_result.metrics.p95_token_latency_ms,
                "peak_memory_mib": tuned_result.metrics.peak_memory_mib,
                "elapsed_seconds": tuned_result.elapsed_seconds,
            },
            valid=valid,
            error="" if valid else reason,
            score=score,
            delta=delta,
            op_perf_status=str(candidate_op_profile.get("status", "")),
            op_perf_cache_hit=bool(candidate_op_profile.get("cache_hit")) if "cache_hit" in candidate_op_profile else None,
            op_perf_rows_emitted=(
                int(candidate_op_profile.get("rows_emitted"))
                if isinstance(candidate_op_profile.get("rows_emitted"), (int, float))
                else None
            ),
            op_perf_cache_key=str(candidate_op_profile.get("cache_key", "")),
            op_perf_delta_pct=op_perf_delta_pct,
            op_perf_common_rows=int(op_perf_common_rows),
            op_perf_baseline_total_ms=op_perf_baseline_total_ms,
            op_perf_candidate_total_ms=op_perf_candidate_total_ms,
            op_perf_compare_key=op_perf_compare_key,
            op_perf_decision=op_perf_decision,
            op_perf_promoted=bool(op_perf_promoted),
            gate_a_evaluated=gate_a_evaluated,
            gate_b_evaluated=gate_b_evaluated,
            gate_c_evaluated=gate_c_evaluated,
            gate_d_evaluated=gate_d_evaluated,
            gate_a_pass=gate_a_pass,
            gate_b_pass=gate_b_pass,
            gate_c_pass=gate_c_pass,
            gate_d_pass=gate_d_pass,
            dispatch_audit_status=attempt_dispatch_status,
            candidate_resources_expected=attempt_candidate_expected,
            candidate_resources_used=attempt_candidate_used,
            dispatch_audit_path=str(
                first_tuned_run.get("dispatch_audit_path")
                or attempt_dispatch_audit.get("dispatch_audit_path")
                or ""
            ),
            dispatch_audit_source=str(
                first_tuned_run.get("dispatch_audit_source")
                or attempt_dispatch_audit.get("dispatch_audit_source")
                or ""
            ),
            metallib_source=str(
                first_tuned_run.get("metallib_source")
                or attempt_dispatch_audit.get("metallib_source")
                or ""
            ),
            dispatch_audit=attempt_dispatch_audit,
        )
        record_payload = asdict(record)
        _append_jsonl(attempts_out, record_payload)
        if attempt_log_path is not None:
            _append_jsonl(attempt_log_path, record_payload)
        if attempt_records is not None:
            attempt_records.append(record_payload)

        prev_attempts.append(
            {
                "candidate_name": candidate.candidate_name,
                "score": score,
                "pass_gate": pass_gate,
                "delta": delta,
            }
        )

        if (
            budget_override <= 0
            and arm_id in {"iterative", "iterative_kernel"}
            and attempt_idx == 1
            and kernel_mode != "oneshot"
        ):
            max_attempts = 1 + (5 if valid else 3)
            followup_stage = "full" if valid else "quick"

    return best_config, idx, invalid_runs


def run_validation_study(
    *,
    matrix_path: Path,
    output_dir: Path,
    profiles: list[str],
    arms: list[str],
    llamacpp_root: Path,
    gate_mode: str,
    cooldown_seconds: float,
    bootstrap_samples: int,
    seed: int,
    require_ac_power: bool,
    strict_commit: bool,
    resume: bool,
    cache_root: Path | None = None,
    kernel_mode: str = "none",
    kernel_total_budget: int = 0,
    candidate_cache_dir: Path | None = None,
    attempt_log_path: Path | None = None,
    abba_cycles: int = DEFAULT_ABBA_CYCLES,
    abba_blocks_legacy: int = 0,
    warmup_blocks: int = DEFAULT_WARMUP_BLOCKS,
    strict_parity: bool = False,
    parity_stage: str = PARITY_STAGE_NUMERIC,
    profiling_mode: str = PROFILING_MODE_HEURISTIC,
    long_token_tolerance: int = 128,
    decode_claim_threshold_pct: float = DEFAULT_DECODE_CLAIM_THRESHOLD_PCT,
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
) -> StudySummary:
    resolved_cache_root = configure_cache_root(cache_root)
    profiles = _resolve_profiles(",".join(profiles))
    arms = _resolve_arms(",".join(arms))
    kernel_budget_allocation = _kernel_budget_allocation(
        arms=arms,
        kernel_total_budget=int(max(0, kernel_total_budget)),
    )
    matrix = _load_matrix(matrix_path)
    abba_cycles_effective = int(max(1, abba_cycles))
    abba_alias_used = False
    if int(abba_blocks_legacy) > 0:
        abba_cycles_effective = int(max(1, abba_blocks_legacy))
        abba_alias_used = True

    parity_stage_norm = str(parity_stage or PARITY_STAGE_NUMERIC).strip().lower()
    if parity_stage_norm not in {
        PARITY_STAGE_NONE,
        PARITY_STAGE_NUMERIC,
        PARITY_STAGE_SEMANTIC,
        PARITY_STAGE_CLAIM,
    }:
        raise StudyError(
            f"Unsupported parity_stage '{parity_stage}'. Use none, numeric, semantic, or claim."
        )
    tune_strict_parity = parity_stage_norm in {PARITY_STAGE_SEMANTIC, PARITY_STAGE_CLAIM}
    claim_strict_parity = parity_stage_norm == PARITY_STAGE_CLAIM
    if strict_parity:
        # Backward-compatible alias: strict_parity implies claim-level strictness.
        tune_strict_parity = True
        claim_strict_parity = True
        parity_stage_norm = PARITY_STAGE_CLAIM

    profiling_mode_norm = str(profiling_mode or PROFILING_MODE_HEURISTIC).strip().lower()
    if profiling_mode_norm not in {PROFILING_MODE_HEURISTIC, PROFILING_MODE_OP_PERF_REQUIRED}:
        raise StudyError(
            f"Unsupported profiling_mode '{profiling_mode}'. Use heuristic or op_perf_required."
        )
    allow_heuristic_hotspots = profiling_mode_norm == PROFILING_MODE_HEURISTIC
    op_perf_cache_mode = str(op_perf_cache or "on").strip().lower()
    if op_perf_cache_mode not in {"on", "off", "refresh"}:
        op_perf_cache_mode = "on"
    op_test_cache_mode = str(op_test_cache or "on").strip().lower()
    if op_test_cache_mode not in {"on", "off", "refresh"}:
        op_test_cache_mode = "on"
    op_perf_reject_regression_pct = float(max(0.0, op_perf_reject_regression_pct))
    op_perf_promote_topk = int(max(0, op_perf_promote_topk))

    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise StudyError(f"Output directory already exists and is non-empty: {output_dir}")
        output_dir = output_dir / f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if candidate_cache_dir is None:
        candidate_cache_dir = output_dir / "candidate_cache"
    candidate_cache_dir = Path(candidate_cache_dir).expanduser().resolve()
    candidate_cache_dir.mkdir(parents=True, exist_ok=True)

    runs_raw_path = output_dir / "runs_raw.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    summary_path = output_dir / "summary.json"
    claim_decisions_path = output_dir / "claim_decisions.json"
    schedule_path = output_dir / "schedule.json"
    hotspots_path = output_dir / "hotspots.json"
    hotspots_op_perf_path = output_dir / "hotspots_op_perf.json"
    op_profiles_path = output_dir / "op_profiles.json"
    roofline_path = output_dir / "roofline_analysis.json"
    throughput_report_json_path = output_dir / "throughput_report.json"
    throughput_report_csv_path = output_dir / "throughput_report.csv"
    exclusions_csv_path = output_dir / "exclusions.csv"
    metrics_csv_path = output_dir / "metrics_by_block.csv"
    paired_csv_path = output_dir / "paired_deltas.csv"
    ci_csv_path = output_dir / "ci_results.csv"
    pval_csv_path = output_dir / "pvalues_corrected.csv"
    dispatch_audit_dir = output_dir / "dispatch_audit"
    runs_raw_path.touch(exist_ok=True)
    attempts_path.touch(exist_ok=True)
    dispatch_audit_dir.mkdir(parents=True, exist_ok=True)
    if attempt_log_path is not None:
        attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_log_path.touch(exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    git_commit = _git_commit(repo_root)

    device = probe_device()
    assert_supported_device(device)
    allowed_params = asdict(derive_allowed_params(device))

    power_state = _power_state()
    if require_ac_power and not bool(power_state.get("on_ac_power")):
        raise StudyError("Strict mode requires AC power. Plug in power and rerun.")
    if require_ac_power and power_state.get("charging") is False:
        raise StudyError("Strict mode requires battery to not be discharging.")

    commit_ok, commit_msg = ensure_llamacpp_commit(llamacpp_root, strict=strict_commit)
    if not commit_ok:
        raise StudyError(commit_msg)
    llama_commit = get_llamacpp_commit(llamacpp_root)
    llama_cli = benchmark.resolve_llama_cli(llamacpp_root)
    test_backend_ops_bin = resolve_test_backend_ops(llamacpp_root)
    if profiling_mode_norm == PROFILING_MODE_OP_PERF_REQUIRED and test_backend_ops_bin is None:
        raise StudyError(
            "profiling_mode=op_perf_required requires test-backend-ops binary. "
            "Run scripts/apple_silicon/bootstrap.py with tests enabled first."
        )

    llm_required = any(arm in {"oneshot_kernel", "iterative_kernel"} for arm in arms)
    require_metal_toolchain = bool(llm_required or profiling_mode_norm == PROFILING_MODE_OP_PERF_REQUIRED)
    metal_tools = benchmark.resolve_metal_toolchain_paths()
    if require_metal_toolchain and not bool(metal_tools.get("success")):
        raise StudyError(
            "Metal toolchain preflight failed: unable to resolve xcrun metal/metallib. "
            "Select full Xcode or set DEVELOPER_DIR and rerun."
        )
    llm_identity = _resolve_llm_identity(required=llm_required)

    validated_models: list[dict[str, Any]] = []
    run_models: list[dict[str, Any]] = []
    for model_entry in matrix:
        model_id = str(model_entry.get("id", "")).strip()
        model_path = Path(str(model_entry.get("path", "")).strip()).expanduser().resolve()
        expected_sha = str(model_entry.get("sha256", "")).strip().lower()
        source_url = str(model_entry.get("url", "")).strip()
        if not model_path.exists():
            raise StudyError(f"Model missing: {model_path}")

        model = probe_model(model_path)
        assert_supported_model(model)
        if expected_sha and expected_sha != model.sha256:
            raise StudyError(
                f"Checksum mismatch for {model_id}: expected {expected_sha}, got {model.sha256}"
            )

        validated_models.append(
            {
                "id": model_id,
                "path": str(model.path),
                "source_url": source_url,
                "sha256": model.sha256,
                "expected_sha256": expected_sha,
                "name": model.name,
                "architecture": model.architecture,
                "quant": model.quant,
                "file_type_id": model.file_type_id,
                "size_bytes": model.size_bytes,
            }
        )
        run_models.append({"id": model_id, "model": model})

    schedule_rows = _materialize_schedule(
        model_ids=[str(row.get("id", "")).strip() for row in run_models],
        profiles=profiles,
        arms=arms,
        abba_cycles=abba_cycles_effective,
        warmup_blocks=int(max(1, warmup_blocks)),
    )
    _json_dump(
        schedule_path,
        {
            "generated_at_utc": _utcnow(),
            "abba_cycles": int(abba_cycles_effective),
            "warmup_blocks": int(max(1, warmup_blocks)),
            "rows": schedule_rows,
            "counts": {
                "rows_total": len(schedule_rows),
                "warmup_rows": sum(1 for row in schedule_rows if bool(row.get("is_warmup"))),
                "measurement_rows": sum(1 for row in schedule_rows if not bool(row.get("is_warmup"))),
            },
        },
    )

    official_claim_mode = bool(
        require_ac_power
        and claim_strict_parity
        and profiling_mode_norm == PROFILING_MODE_OP_PERF_REQUIRED
    )

    manifest = {
        "generated_at_utc": _utcnow(),
        "repo_root": str(repo_root),
        "git_commit": git_commit,
        "llamacpp_root": str(llamacpp_root),
        "llamacpp_commit": llama_commit,
        "llamacpp_commit_message": commit_msg,
        "llama_cli": str(llama_cli),
        "matrix_path": str(matrix_path),
        "matrix": matrix,
        "validated_models": validated_models,
        "profiles": profiles,
        "arms": arms,
        "gate_mode": gate_mode,
        "cooldown_seconds": cooldown_seconds,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "abba_cycles": int(abba_cycles_effective),
        "abba_blocks_deprecated_alias_used": bool(abba_alias_used),
        "abba_blocks_total": len(_crossover_orders("A", "B", cycles=abba_cycles_effective)),
        "abba_blocks": len(_crossover_orders("A", "B", cycles=abba_cycles_effective)),
        "warmup_blocks": int(max(1, warmup_blocks)),
        "strict_parity": bool(strict_parity),
        "parity_stage": parity_stage_norm,
        "strict_parity_tuning": bool(tune_strict_parity),
        "strict_parity_claim": bool(claim_strict_parity),
        "profiling_mode": profiling_mode_norm,
        "op_perf": {
            "timeout_sec": float(op_perf_timeout_sec),
            "cache_mode": op_perf_cache_mode,
            "min_rows": int(max(1, op_perf_min_rows)),
            "op_filter": str(op_perf_op_filter),
            "case_limit": int(max(0, op_perf_case_limit)),
            "case_seed": int(max(0, op_perf_case_seed)),
            "warmup_iters": int(max(0, op_perf_warmup_iters)),
            "bench_iters": int(max(0, op_perf_bench_iters)),
            "reject_regression_pct": float(op_perf_reject_regression_pct),
            "promote_topk": int(
                0
                if (claim_strict_parity or official_claim_mode)
                else int(max(0, op_perf_promote_topk))
            ),
        },
        "op_test": {
            "timeout_sec": float(op_test_timeout_sec),
            "cache_mode": op_test_cache_mode,
            "min_rows": int(max(1, op_test_min_rows)),
            "case_limit": int(max(0, op_test_case_limit)),
            "case_seed": int(max(0, op_test_case_seed)),
        },
        "long_token_tolerance": int(max(0, long_token_tolerance)),
        "prompt_cache_mode": "tuning_long_smoke",
        "prompt_cache_used_for_claim": False,
        "official_claim_mode": official_claim_mode,
        "decode_claim_threshold_pct": float(decode_claim_threshold_pct),
        "kernel_mode": kernel_mode,
        "kernel_total_budget": int(max(0, kernel_total_budget)),
        "kernel_budget_allocation": dict(kernel_budget_allocation),
        "cache_root": str(resolved_cache_root),
        "candidate_cache_dir": str(candidate_cache_dir),
        "dispatch_audit_dir": str(dispatch_audit_dir),
        "compile_cache_root": str((current_cache_root() / "compile_cache").expanduser().resolve()),
        "toolchain_fingerprint": benchmark.metal_toolchain_fingerprint(),
        "metal_toolchain": metal_tools,
        "toolchain_options": {
            "GGML_METAL_SHADER_DEBUG": os.environ.get("GGML_METAL_SHADER_DEBUG", ""),
            "GGML_METAL_NDEBUG": os.environ.get("GGML_METAL_NDEBUG", ""),
            "GGML_METAL_STD": os.environ.get("GGML_METAL_STD", ""),
            "GGML_METAL_MACOSX_VERSION_MIN": os.environ.get("GGML_METAL_MACOSX_VERSION_MIN", ""),
        },
        "strict_require_ac_power": require_ac_power,
        "power_precheck": power_state,
        "device": asdict(device),
        "allowed_params": allowed_params,
        "llm": llm_identity,
        "test_backend_ops": str(test_backend_ops_bin) if test_backend_ops_bin is not None else "",
        "schedule_artifact": str(schedule_path),
        "crossover_order": [
            "AB" if first == "A" else "BA"
            for first, _second in _crossover_orders("A", "B", cycles=abba_cycles_effective)
        ],
        "prompt_order_policy": "deterministic_round_robin_by_repeat_index",
    }
    _json_dump(output_dir / "study_manifest.json", manifest)

    metrics_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    attempt_records: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    op_profile_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    order_index = 0
    invalid_blocks_main = 0
    invalid_blocks_warmup = 0
    invalid_runs = 0

    prompts = benchmark.load_prompt_suite(None)

    claim_long_profiles = [p for p in profiles if p in {"long", "long_claim"}]
    if official_claim_mode and claim_long_profiles:
        long_precheck_rows: list[dict[str, Any]] = []
        for profile_name in claim_long_profiles:
            long_probe = _profile_for(profile_name, "quick")
            for model_entry in run_models:
                model_id = str(model_entry.get("id", "")).strip()
                model = model_entry["model"]
                probe_result = benchmark.run_profile_benchmark(
                    llama_cli=llama_cli,
                    model_path=model.path,
                    profile=long_probe,
                    prompts=prompts,
                    resources_path=None,
                    extra_args=[],
                    capture_raw_output=True,
                    enforce_long_prompt_target=True,
                    long_prompt_token_tolerance=int(max(0, long_token_tolerance)),
                    dispatch_attempt_id=_dispatch_attempt_id(
                        model_id,
                        profile_name,
                        "long_precheck",
                    ),
                    dispatch_rule_id="",
                    dispatch_audit_dir=dispatch_audit_dir,
                    candidate_resources_expected=False,
                )
                valid, reason = _result_valid(
                    probe_result,
                    long_prompt_tolerance=int(max(0, long_token_tolerance)),
                    require_long_prompt_target=True,
                )
                long_precheck_rows.append(
                    {
                        "model_id": model_id,
                        "profile": profile_name,
                        "valid": bool(valid),
                        "reason": reason,
                        "prompt_tokens_actual": [
                            r.get("prompt_tokens_actual", r.get("prompt_tokens")) for r in probe_result.runs
                        ],
                        "prompt_tokens_target": [r.get("prompt_tokens_target") for r in probe_result.runs],
                    }
                )
                if not valid:
                    raise StudyError(
                        "Official claim preflight failed long prompt target check "
                        f"for {model_id}/{profile_name}: {reason}"
                    )
        manifest["long_prompt_target_preflight"] = {
            "enabled": True,
            "rows": long_precheck_rows,
        }
        _json_dump(output_dir / "study_manifest.json", manifest)

    for model_entry in run_models:
        model_id = str(model_entry.get("id", "")).strip()
        model = model_entry["model"]

        for profile_name in profiles:
            profile = _profile_for(profile_name, gate_mode)
            op_profile = run_stage1_op_profile(
                llamacpp_root=llamacpp_root,
                model_path=model.path,
                profile=profile,
                profiling_mode=profiling_mode_norm,
                rank_metric="time",
                op_filter=str(op_perf_op_filter),
                timeout_sec=float(op_perf_timeout_sec),
                cache_mode=op_perf_cache_mode,
                min_rows=int(max(1, op_perf_min_rows)),
                perf_case_limit=int(max(0, op_perf_case_limit)),
                perf_case_seed=int(max(0, op_perf_case_seed)),
                perf_warmup_iters=int(max(0, op_perf_warmup_iters)),
                perf_bench_iters=int(max(0, op_perf_bench_iters)),
                artifacts_dir=output_dir / "op_profile_artifacts",
                artifact_prefix=f"{model_id}_{profile.name}_",
            )
            op_hotspots = list(op_profile.get("hotspot_ops") or [])
            profiling_mode = str(op_profile.get("profiling_mode", "heuristic"))
            if profiling_mode_norm == PROFILING_MODE_OP_PERF_REQUIRED and not bool(op_profile.get("success")):
                raise StudyError(
                    "profiling_mode=op_perf_required failed for "
                    f"{model_id}/{profile.name}: {op_profile.get('reason', 'unknown')}"
                )
            op_profile_rows.append(
                {
                    "model_id": model_id,
                    "model_sha256": model.sha256,
                    "profile": profile.name,
                    "profiling_mode": profiling_mode,
                    "profiling_mode_effective": str(op_profile.get("profiling_mode_effective", profiling_mode)),
                    "status": str(op_profile.get("status", "")),
                    "cache_hit": bool(op_profile.get("cache_hit")) if "cache_hit" in op_profile else None,
                    "cache_key": str(op_profile.get("cache_key", "")),
                    "rows_emitted": (
                        int(op_profile.get("rows_emitted"))
                        if isinstance(op_profile.get("rows_emitted"), (int, float))
                        else 0
                    ),
                    "elapsed_ms": _to_float(op_profile.get("elapsed_ms")),
                    "success": bool(op_profile.get("success")),
                    "reason": str(op_profile.get("reason", "")),
                    "command": str(op_profile.get("command", "")),
                    "ops": list(op_profile.get("ops") or []),
                    "hotspot_ops": op_hotspots,
                }
            )

            def _annotate_result_with_stage1_status(res: BenchmarkResult) -> None:
                for run in res.runs:
                    run["op_perf_status"] = str(op_profile.get("status", ""))
                    run["op_perf_cache_hit"] = (
                        bool(op_profile.get("cache_hit"))
                        if "cache_hit" in op_profile
                        else None
                    )
                    run["op_perf_rows_emitted"] = (
                        int(op_profile.get("rows_emitted"))
                        if isinstance(op_profile.get("rows_emitted"), (int, float))
                        else None
                    )
                    run["op_perf_cache_key"] = str(op_profile.get("cache_key", ""))

            # Warmup baseline first and use final warmup as tuning reference.
            baseline_warm: BenchmarkResult | None = None
            for warm_i in range(max(1, int(warmup_blocks))):
                warm_block_id = f"warmup_{model_id}_{profile.name}_baseline_w{warm_i + 1}"
                baseline_warm = benchmark.run_profile_benchmark(
                    llama_cli=llama_cli,
                    model_path=model.path,
                    profile=profile,
                    prompts=prompts,
                    resources_path=None,
                    extra_args=[],
                    capture_raw_output=True,
                    enforce_long_prompt_target=official_claim_mode,
                    long_prompt_token_tolerance=int(max(0, long_token_tolerance)),
                    dispatch_attempt_id=_dispatch_attempt_id(
                        model_id,
                        profile.name,
                        warm_block_id,
                        "baseline",
                    ),
                    dispatch_rule_id="",
                    dispatch_audit_dir=dispatch_audit_dir,
                    candidate_resources_expected=False,
                )
                _annotate_result_with_stage1_status(baseline_warm)
                order_index, invalid_local = _emit_runs_raw(
                    out_path=runs_raw_path,
                    model_id=model_id,
                    model_path=str(model.path),
                    model_sha=model.sha256,
                    profile=profile.name,
                    arm_id="baseline",
                    attempt_id="",
                    block_id=warm_block_id,
                    is_warmup=True,
                    runtime_args=[],
                    result=baseline_warm,
                    git_commit=git_commit,
                    llamacpp_commit=llama_commit,
                    power_state_snapshot=power_state,
                    order_index_start=order_index,
                    profiling_mode=profiling_mode,
                    long_prompt_tolerance=int(max(0, long_token_tolerance)),
                    require_long_prompt_target=official_claim_mode,
                    candidate_resources_expected=False,
                    allow_heuristic_hotspots=allow_heuristic_hotspots,
                    hotspot_override_ops=op_hotspots,
                    hotspot_rows=hotspot_rows,
                    exclusion_rows=exclusion_rows,
                )
                invalid_runs += invalid_local
                metrics_rows.append(
                    _metrics_row(
                        model_id=model_id,
                        model_sha=model.sha256,
                        profile=profile.name,
                        arm_id="baseline",
                        block_id=warm_block_id,
                        order_index=order_index,
                        order_label="warmup",
                        is_warmup=True,
                            runtime_args=[],
                            result=baseline_warm,
                            profiling_mode=profiling_mode,
                            long_prompt_tolerance=int(max(0, long_token_tolerance)),
                            require_long_prompt_target=official_claim_mode,
                            candidate_resources_expected=False,
                        )
                    )
                baseline_valid, _ = _result_valid(
                    baseline_warm,
                    long_prompt_tolerance=int(max(0, long_token_tolerance)),
                    require_long_prompt_target=official_claim_mode,
                    candidate_resources_expected=False,
                )
                if not baseline_valid:
                    invalid_blocks_warmup += 1
            if baseline_warm is None:
                raise StudyError(f"Failed to produce baseline warmup for {model_id}/{profile.name}")

            arm_configs: dict[str, dict[str, Any] | None] = {
                "baseline": {
                    "runtime_args": [],
                    "resources_path": "",
                    "kernel_candidate": {},
                    "compile_record": {},
                }
            }
            if "flash" in arms:
                arm_configs["flash"] = {
                    "runtime_args": ["--flash-attn", "on"],
                    "resources_path": "",
                    "kernel_candidate": {},
                    "compile_record": {},
                }

            if "oneshot_kernel" in arms:
                oneshot_budget = int(kernel_budget_allocation.get("oneshot_kernel") or 0)
                if int(max(0, kernel_total_budget)) > 0 and oneshot_budget <= 0:
                    arm_configs["oneshot_kernel"] = None
                    if exclusion_rows is not None:
                        exclusion_rows.append(
                            {
                                "kind": "arm_skipped",
                                "model_id": model_id,
                                "profile": profile.name,
                                "arm_id": "oneshot_kernel",
                                "reason": "kernel_budget_exhausted",
                            }
                        )
                else:
                    chosen, order_index, invalid_local = _tune_arm_args(
                        arm_id="oneshot_kernel",
                        model_id=model_id,
                        model=model,
                        profile=profile,
                        baseline_result=baseline_warm,
                        device=device,
                        llama_cli=llama_cli,
                        prompts=prompts,
                        attempts_out=attempts_path,
                        runs_raw_out=runs_raw_path,
                        git_commit=git_commit,
                        llamacpp_commit=llama_commit,
                        power_state=power_state,
                        order_index_start=order_index,
                        llamacpp_root=llamacpp_root,
                        llm_identity=llm_identity,
                        allowed_params=allowed_params,
                        gate_mode=gate_mode,
                        candidate_cache_dir=candidate_cache_dir,
                        dispatch_audit_dir=dispatch_audit_dir,
                        kernel_mode=kernel_mode,
                        strict_parity=tune_strict_parity,
                        kernel_attempt_budget=oneshot_budget,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        allow_heuristic_hotspots=allow_heuristic_hotspots,
                        op_hotspot_ops=op_hotspots,
                        profiling_mode=profiling_mode,
                        baseline_op_profile=op_profile,
                        op_perf_timeout_sec=float(op_perf_timeout_sec),
                        op_perf_cache=op_perf_cache_mode,
                        op_perf_min_rows=int(max(1, op_perf_min_rows)),
                        op_perf_op_filter=str(op_perf_op_filter),
                        op_perf_case_limit=int(max(0, op_perf_case_limit)),
                        op_perf_case_seed=int(max(0, op_perf_case_seed)),
                        op_perf_warmup_iters=int(max(0, op_perf_warmup_iters)),
                        op_perf_bench_iters=int(max(0, op_perf_bench_iters)),
                        op_perf_reject_regression_pct=float(op_perf_reject_regression_pct),
                        op_perf_promote_topk=int(max(0, op_perf_promote_topk)),
                        op_test_timeout_sec=float(op_test_timeout_sec),
                        op_test_cache=op_test_cache_mode,
                        op_test_min_rows=int(max(1, op_test_min_rows)),
                        op_test_case_limit=int(max(0, op_test_case_limit)),
                        op_test_case_seed=int(max(0, op_test_case_seed)),
                        prompt_cache_enabled=True,
                        prompt_cache_root=output_dir / "prompt_cache",
                        official_claim_mode=official_claim_mode,
                        stage1_force_strict_regression=bool(claim_strict_parity),
                        attempt_log_path=attempt_log_path,
                        attempt_records=attempt_records,
                        hotspot_rows=hotspot_rows,
                        exclusion_rows=exclusion_rows,
                    )
                    arm_configs["oneshot_kernel"] = chosen
                    invalid_runs += invalid_local

            if "iterative_kernel" in arms:
                iterative_budget = int(kernel_budget_allocation.get("iterative_kernel") or 0)
                if int(max(0, kernel_total_budget)) > 0 and iterative_budget <= 0:
                    arm_configs["iterative_kernel"] = None
                    if exclusion_rows is not None:
                        exclusion_rows.append(
                            {
                                "kind": "arm_skipped",
                                "model_id": model_id,
                                "profile": profile.name,
                                "arm_id": "iterative_kernel",
                                "reason": "kernel_budget_exhausted",
                            }
                        )
                else:
                    chosen, order_index, invalid_local = _tune_arm_args(
                        arm_id="iterative_kernel",
                        model_id=model_id,
                        model=model,
                        profile=profile,
                        baseline_result=baseline_warm,
                        device=device,
                        llama_cli=llama_cli,
                        prompts=prompts,
                        attempts_out=attempts_path,
                        runs_raw_out=runs_raw_path,
                        git_commit=git_commit,
                        llamacpp_commit=llama_commit,
                        power_state=power_state,
                        order_index_start=order_index,
                        llamacpp_root=llamacpp_root,
                        llm_identity=llm_identity,
                        allowed_params=allowed_params,
                        gate_mode=gate_mode,
                        candidate_cache_dir=candidate_cache_dir,
                        dispatch_audit_dir=dispatch_audit_dir,
                        kernel_mode=kernel_mode,
                        strict_parity=tune_strict_parity,
                        kernel_attempt_budget=iterative_budget,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        allow_heuristic_hotspots=allow_heuristic_hotspots,
                        op_hotspot_ops=op_hotspots,
                        profiling_mode=profiling_mode,
                        baseline_op_profile=op_profile,
                        op_perf_timeout_sec=float(op_perf_timeout_sec),
                        op_perf_cache=op_perf_cache_mode,
                        op_perf_min_rows=int(max(1, op_perf_min_rows)),
                        op_perf_op_filter=str(op_perf_op_filter),
                        op_perf_case_limit=int(max(0, op_perf_case_limit)),
                        op_perf_case_seed=int(max(0, op_perf_case_seed)),
                        op_perf_warmup_iters=int(max(0, op_perf_warmup_iters)),
                        op_perf_bench_iters=int(max(0, op_perf_bench_iters)),
                        op_perf_reject_regression_pct=float(op_perf_reject_regression_pct),
                        op_perf_promote_topk=int(max(0, op_perf_promote_topk)),
                        op_test_timeout_sec=float(op_test_timeout_sec),
                        op_test_cache=op_test_cache_mode,
                        op_test_min_rows=int(max(1, op_test_min_rows)),
                        op_test_case_limit=int(max(0, op_test_case_limit)),
                        op_test_case_seed=int(max(0, op_test_case_seed)),
                        prompt_cache_enabled=True,
                        prompt_cache_root=output_dir / "prompt_cache",
                        official_claim_mode=official_claim_mode,
                        stage1_force_strict_regression=bool(claim_strict_parity),
                        attempt_log_path=attempt_log_path,
                        attempt_records=attempt_records,
                        hotspot_rows=hotspot_rows,
                        exclusion_rows=exclusion_rows,
                    )
                    arm_configs["iterative_kernel"] = chosen
                    invalid_runs += invalid_local

            for arm_id in [a for a in arms if a != "baseline"]:
                cfg = arm_configs.get(arm_id)
                if cfg is None:
                    continue
                args = list(cfg.get("runtime_args") or [])
                resources_raw = str(cfg.get("resources_path", "")).strip()
                resources_path = Path(resources_raw) if resources_raw else None
                kernel_meta = cfg.get("kernel_candidate") or {}
                compile_meta = cfg.get("compile_record") or {}
                arm_dispatch_rule_id = (
                    _dispatch_rule_id_for_patch(
                        patch_hash=str(kernel_meta.get("patch_hash", "")),
                        arm_id=arm_id,
                        attempt_id=f"{model_id}:{profile.name}:warmup",
                    )
                    if resources_path is not None
                    else ""
                )
                for warm_i in range(max(1, int(warmup_blocks))):
                    warm_block_id = f"warmup_{model_id}_{profile.name}_{arm_id}_w{warm_i + 1}"
                    arm_warm = benchmark.run_profile_benchmark(
                        llama_cli=llama_cli,
                        model_path=model.path,
                        profile=profile,
                        prompts=prompts,
                        resources_path=resources_path,
                        extra_args=args,
                        capture_raw_output=True,
                        force_source_compile=bool(
                            resources_path and not bool(compile_meta.get("compile_warmup_done", False))
                        ),
                        enforce_long_prompt_target=official_claim_mode,
                        long_prompt_token_tolerance=int(max(0, long_token_tolerance)),
                        dispatch_attempt_id=_dispatch_attempt_id(
                            model_id,
                            profile.name,
                            warm_block_id,
                            arm_id,
                        ),
                        dispatch_rule_id=arm_dispatch_rule_id,
                        dispatch_audit_dir=dispatch_audit_dir,
                        candidate_resources_expected=bool(resources_path),
                    )
                    _annotate_result_with_stage1_status(arm_warm)
                    order_index, invalid_local = _emit_runs_raw(
                        out_path=runs_raw_path,
                        model_id=model_id,
                        model_path=str(model.path),
                        model_sha=model.sha256,
                        profile=profile.name,
                        arm_id=arm_id,
                        attempt_id="",
                        block_id=warm_block_id,
                        is_warmup=True,
                        runtime_args=args,
                        result=arm_warm,
                        git_commit=git_commit,
                        llamacpp_commit=llama_commit,
                        power_state_snapshot=power_state,
                        order_index_start=order_index,
                        profiling_mode=profiling_mode,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        candidate_resources_expected=bool(resources_path),
                        allow_heuristic_hotspots=allow_heuristic_hotspots,
                        hotspot_override_ops=op_hotspots,
                        kernel_template_version=str(kernel_meta.get("template_version", "")),
                        patch_hash=str(kernel_meta.get("patch_hash", "")),
                        candidate_source_hash=str(kernel_meta.get("source_hash", "")),
                        compile_result=str(compile_meta.get("classification", "")),
                        compile_stderr_hash=str(compile_meta.get("stderr_hash", "")),
                        compile_warmup_done=bool(compile_meta.get("compile_warmup_done", False)),
                        pipeline_cache_key=str(compile_meta.get("pipeline_cache_key", "")),
                        compile_time_ms=compile_meta.get("compile_time_ms"),
                        hotspot_rows=hotspot_rows,
                        exclusion_rows=exclusion_rows,
                    )
                    invalid_runs += invalid_local
                    metrics_rows.append(
                        _metrics_row(
                            model_id=model_id,
                            model_sha=model.sha256,
                            profile=profile.name,
                            arm_id=arm_id,
                            block_id=warm_block_id,
                            order_index=order_index,
                            order_label="warmup",
                            is_warmup=True,
                            runtime_args=args,
                            result=arm_warm,
                            kernel_template_version=str(kernel_meta.get("template_version", "")),
                            patch_hash=str(kernel_meta.get("patch_hash", "")),
                            candidate_source_hash=str(kernel_meta.get("source_hash", "")),
                            compile_result=str(compile_meta.get("classification", "")),
                            compile_stderr_hash=str(compile_meta.get("stderr_hash", "")),
                            compile_warmup_done=bool(compile_meta.get("compile_warmup_done", False)),
                            pipeline_cache_key=str(compile_meta.get("pipeline_cache_key", "")),
                            compile_time_ms=compile_meta.get("compile_time_ms"),
                            profiling_mode=profiling_mode,
                            long_prompt_tolerance=int(max(0, long_token_tolerance)),
                            require_long_prompt_target=official_claim_mode,
                            candidate_resources_expected=bool(resources_path),
                        )
                    )
                    arm_warm_valid, _ = _result_valid(
                        arm_warm,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        candidate_resources_expected=bool(resources_path),
                    )
                    if not arm_warm_valid:
                        invalid_blocks_warmup += 1

            # Main ABBA/crossover measurement.
            for arm_id in [a for a in arms if a != "baseline"]:
                cfg = arm_configs.get(arm_id)
                if cfg is None:
                    continue

                for block_idx, (first_arm, second_arm) in enumerate(
                    _crossover_orders("baseline", arm_id, cycles=abba_cycles_effective),
                    start=1,
                ):
                    first_cfg = arm_configs.get(first_arm) or {}
                    second_cfg = arm_configs.get(second_arm) or {}
                    first_args = list(first_cfg.get("runtime_args") or [])
                    second_args = list(second_cfg.get("runtime_args") or [])
                    first_res_path = Path(str(first_cfg.get("resources_path", ""))) if str(first_cfg.get("resources_path", "")).strip() else None
                    second_res_path = Path(str(second_cfg.get("resources_path", ""))) if str(second_cfg.get("resources_path", "")).strip() else None
                    first_kernel = first_cfg.get("kernel_candidate") or {}
                    second_kernel = second_cfg.get("kernel_candidate") or {}
                    first_compile = first_cfg.get("compile_record") or {}
                    second_compile = second_cfg.get("compile_record") or {}
                    order_label = f"{first_arm}->{second_arm}"
                    block_id = f"{model_id}_{profile.name}_{arm_id}_block_{block_idx}"
                    first_dispatch_rule_id = (
                        _dispatch_rule_id_for_patch(
                            patch_hash=str(first_kernel.get("patch_hash", "")),
                            arm_id=first_arm,
                            attempt_id=block_id,
                        )
                        if first_res_path is not None
                        else ""
                    )
                    second_dispatch_rule_id = (
                        _dispatch_rule_id_for_patch(
                            patch_hash=str(second_kernel.get("patch_hash", "")),
                            arm_id=second_arm,
                            attempt_id=block_id,
                        )
                        if second_res_path is not None
                        else ""
                    )

                    first_res = benchmark.run_profile_benchmark(
                        llama_cli=llama_cli,
                        model_path=model.path,
                        profile=profile,
                        prompts=prompts,
                        resources_path=first_res_path,
                        extra_args=first_args,
                        capture_raw_output=True,
                        force_source_compile=bool(
                            first_res_path and not bool(first_compile.get("compile_warmup_done", False))
                        ),
                        enforce_long_prompt_target=official_claim_mode,
                        long_prompt_token_tolerance=int(max(0, long_token_tolerance)),
                        dispatch_attempt_id=_dispatch_attempt_id(
                            model_id,
                            profile.name,
                            block_id,
                            first_arm,
                            "first",
                        ),
                        dispatch_rule_id=first_dispatch_rule_id,
                        dispatch_audit_dir=dispatch_audit_dir,
                        candidate_resources_expected=bool(first_res_path),
                    )
                    _annotate_result_with_stage1_status(first_res)
                    order_index, invalid_local = _emit_runs_raw(
                        out_path=runs_raw_path,
                        model_id=model_id,
                        model_path=str(model.path),
                        model_sha=model.sha256,
                        profile=profile.name,
                        arm_id=first_arm,
                        attempt_id="",
                        block_id=block_id,
                        is_warmup=False,
                        runtime_args=first_args,
                        result=first_res,
                        git_commit=git_commit,
                        llamacpp_commit=llama_commit,
                        power_state_snapshot=power_state,
                        order_index_start=order_index,
                        profiling_mode=profiling_mode,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        candidate_resources_expected=bool(first_res_path),
                        allow_heuristic_hotspots=allow_heuristic_hotspots,
                        hotspot_override_ops=op_hotspots,
                        kernel_template_version=str(first_kernel.get("template_version", "")),
                        patch_hash=str(first_kernel.get("patch_hash", "")),
                        candidate_source_hash=str(first_kernel.get("source_hash", "")),
                        compile_result=str(first_compile.get("classification", "")),
                        compile_stderr_hash=str(first_compile.get("stderr_hash", "")),
                        compile_warmup_done=bool(first_compile.get("compile_warmup_done", False)),
                        pipeline_cache_key=str(first_compile.get("pipeline_cache_key", "")),
                        compile_time_ms=first_compile.get("compile_time_ms"),
                        hotspot_rows=hotspot_rows,
                        exclusion_rows=exclusion_rows,
                    )
                    invalid_runs += invalid_local
                    metrics_rows.append(
                        _metrics_row(
                            model_id=model_id,
                            model_sha=model.sha256,
                            profile=profile.name,
                            arm_id=first_arm,
                            block_id=block_id,
                            order_index=order_index,
                            order_label=order_label,
                            is_warmup=False,
                            runtime_args=first_args,
                            result=first_res,
                            kernel_template_version=str(first_kernel.get("template_version", "")),
                            patch_hash=str(first_kernel.get("patch_hash", "")),
                            candidate_source_hash=str(first_kernel.get("source_hash", "")),
                            compile_result=str(first_compile.get("classification", "")),
                            compile_stderr_hash=str(first_compile.get("stderr_hash", "")),
                            compile_warmup_done=bool(first_compile.get("compile_warmup_done", False)),
                            pipeline_cache_key=str(first_compile.get("pipeline_cache_key", "")),
                            compile_time_ms=first_compile.get("compile_time_ms"),
                            profiling_mode=profiling_mode,
                            long_prompt_tolerance=int(max(0, long_token_tolerance)),
                            require_long_prompt_target=official_claim_mode,
                            candidate_resources_expected=bool(first_res_path),
                        )
                    )

                    second_res = benchmark.run_profile_benchmark(
                        llama_cli=llama_cli,
                        model_path=model.path,
                        profile=profile,
                        prompts=prompts,
                        resources_path=second_res_path,
                        extra_args=second_args,
                        capture_raw_output=True,
                        force_source_compile=bool(
                            second_res_path and not bool(second_compile.get("compile_warmup_done", False))
                        ),
                        enforce_long_prompt_target=official_claim_mode,
                        long_prompt_token_tolerance=int(max(0, long_token_tolerance)),
                        dispatch_attempt_id=_dispatch_attempt_id(
                            model_id,
                            profile.name,
                            block_id,
                            second_arm,
                            "second",
                        ),
                        dispatch_rule_id=second_dispatch_rule_id,
                        dispatch_audit_dir=dispatch_audit_dir,
                        candidate_resources_expected=bool(second_res_path),
                    )
                    _annotate_result_with_stage1_status(second_res)
                    order_index, invalid_local = _emit_runs_raw(
                        out_path=runs_raw_path,
                        model_id=model_id,
                        model_path=str(model.path),
                        model_sha=model.sha256,
                        profile=profile.name,
                        arm_id=second_arm,
                        attempt_id="",
                        block_id=block_id,
                        is_warmup=False,
                        runtime_args=second_args,
                        result=second_res,
                        git_commit=git_commit,
                        llamacpp_commit=llama_commit,
                        power_state_snapshot=power_state,
                        order_index_start=order_index,
                        profiling_mode=profiling_mode,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        candidate_resources_expected=bool(second_res_path),
                        allow_heuristic_hotspots=allow_heuristic_hotspots,
                        hotspot_override_ops=op_hotspots,
                        kernel_template_version=str(second_kernel.get("template_version", "")),
                        patch_hash=str(second_kernel.get("patch_hash", "")),
                        candidate_source_hash=str(second_kernel.get("source_hash", "")),
                        compile_result=str(second_compile.get("classification", "")),
                        compile_stderr_hash=str(second_compile.get("stderr_hash", "")),
                        compile_warmup_done=bool(second_compile.get("compile_warmup_done", False)),
                        pipeline_cache_key=str(second_compile.get("pipeline_cache_key", "")),
                        compile_time_ms=second_compile.get("compile_time_ms"),
                        hotspot_rows=hotspot_rows,
                        exclusion_rows=exclusion_rows,
                    )
                    invalid_runs += invalid_local
                    metrics_rows.append(
                        _metrics_row(
                            model_id=model_id,
                            model_sha=model.sha256,
                            profile=profile.name,
                            arm_id=second_arm,
                            block_id=block_id,
                            order_index=order_index,
                            order_label=order_label,
                            is_warmup=False,
                            runtime_args=second_args,
                            result=second_res,
                            kernel_template_version=str(second_kernel.get("template_version", "")),
                            patch_hash=str(second_kernel.get("patch_hash", "")),
                            candidate_source_hash=str(second_kernel.get("source_hash", "")),
                            compile_result=str(second_compile.get("classification", "")),
                            compile_stderr_hash=str(second_compile.get("stderr_hash", "")),
                            compile_warmup_done=bool(second_compile.get("compile_warmup_done", False)),
                            pipeline_cache_key=str(second_compile.get("pipeline_cache_key", "")),
                            compile_time_ms=second_compile.get("compile_time_ms"),
                            profiling_mode=profiling_mode,
                            long_prompt_tolerance=int(max(0, long_token_tolerance)),
                            require_long_prompt_target=official_claim_mode,
                            candidate_resources_expected=bool(second_res_path),
                        )
                    )

                    baseline_res = first_res if first_arm == "baseline" else second_res
                    arm_res = first_res if first_arm == arm_id else second_res
                    baseline_res_path = first_res_path if first_arm == "baseline" else second_res_path
                    arm_res_path = first_res_path if first_arm == arm_id else second_res_path

                    baseline_ok, baseline_reason = _result_valid(
                        baseline_res,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        candidate_resources_expected=bool(baseline_res_path),
                    )
                    arm_ok, arm_reason = _result_valid(
                        arm_res,
                        long_prompt_tolerance=int(max(0, long_token_tolerance)),
                        require_long_prompt_target=official_claim_mode,
                        candidate_resources_expected=bool(arm_res_path),
                    )
                    block_valid = bool(baseline_ok and arm_ok)
                    invalid_reason = ""
                    if not baseline_ok:
                        invalid_reason = baseline_reason
                    elif not arm_ok:
                        invalid_reason = arm_reason
                    if claim_strict_parity and block_valid:
                        parity = classify_correctness_record(
                            baseline=baseline_res,
                            candidate=arm_res,
                            strict_parity=True,
                        )
                        if not parity.success:
                            block_valid = False
                            invalid_reason = parity.classification
                    if not invalid_reason and not block_valid:
                        invalid_reason = "non_zero_return_code_or_missing_metrics"
                    if not block_valid:
                        invalid_blocks_main += 1
                        exclusion_rows.append(
                            {
                                "kind": "block_invalid",
                                "block_id": block_id,
                                "model_id": model_id,
                                "profile": profile.name,
                                "arm_id": arm_id,
                                "order_index": block_idx,
                                "reason": invalid_reason,
                            }
                        )

                    delta_prefill = _delta_pct(
                        baseline_res.metrics.prefill_tokens_per_sec,
                        arm_res.metrics.prefill_tokens_per_sec,
                    )
                    delta_decode = _delta_pct(
                        baseline_res.metrics.decode_tokens_per_sec,
                        arm_res.metrics.decode_tokens_per_sec,
                    )
                    delta_ttft = _delta_pct(
                        baseline_res.metrics.ttft_ms,
                        arm_res.metrics.ttft_ms,
                    )

                    paired_rows.append(
                        {
                            "model_id": model_id,
                            "model_sha256": model.sha256,
                            "profile": profile.name,
                            "arm_id": arm_id,
                            "block_id": block_id,
                            "order_index": block_idx,
                            "order_label": order_label,
                            "delta_prefill_pct": delta_prefill,
                            "delta_decode_pct": delta_decode,
                            "delta_ttft_pct": delta_ttft,
                            "valid": block_valid,
                            "invalid_reason": invalid_reason,
                        }
                    )

                    if cooldown_seconds > 0:
                        time.sleep(cooldown_seconds)

    # Build statistics from valid paired deltas.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in paired_rows:
        key = (str(row["model_id"]), str(row["profile"]), str(row["arm_id"]))
        grouped.setdefault(key, []).append(row)

    ci_rows: list[dict[str, Any]] = []
    p_rows: list[dict[str, Any]] = []
    claim_matrix: dict[str, Any] = {}
    claim_decisions: list[dict[str, Any]] = []

    for (model_id, profile, arm_id), rows in sorted(grouped.items()):
        valid_rows = [r for r in rows if r.get("valid")]
        prefill_vals = [float(r["delta_prefill_pct"]) for r in valid_rows if isinstance(r.get("delta_prefill_pct"), (int, float))]
        decode_vals = [float(r["delta_decode_pct"]) for r in valid_rows if isinstance(r.get("delta_decode_pct"), (int, float))]
        ttft_vals = [float(r["delta_ttft_pct"]) for r in valid_rows if isinstance(r.get("delta_ttft_pct"), (int, float))]

        pre = _bootstrap_ci(prefill_vals, samples=bootstrap_samples, seed=seed + 101)
        dec = _bootstrap_ci(decode_vals, samples=bootstrap_samples, seed=seed + 202)
        tt = _bootstrap_ci(ttft_vals, samples=bootstrap_samples, seed=seed + 303)

        ci_row = {
            "model_id": model_id,
            "profile": profile,
            "arm_id": arm_id,
            "n_blocks_total": len(rows),
            "n_blocks_valid": len(valid_rows),
            "failure_rate_blocks": (1.0 - (len(valid_rows) / len(rows))) if rows else 1.0,
            "mean_delta_prefill_pct": pre["mean"],
            "stdev_delta_prefill_pct": pre["stdev"],
            "ci95_prefill_low": pre["ci_low"],
            "ci95_prefill_high": pre["ci_high"],
            "mean_delta_decode_pct": dec["mean"],
            "stdev_delta_decode_pct": dec["stdev"],
            "ci95_decode_low": dec["ci_low"],
            "ci95_decode_high": dec["ci_high"],
            "mean_delta_ttft_pct": tt["mean"],
            "stdev_delta_ttft_pct": tt["stdev"],
            "ci95_ttft_low": tt["ci_low"],
            "ci95_ttft_high": tt["ci_high"],
        }
        ci_rows.append(ci_row)

        w_prefill = _wilcoxon_signed_rank(prefill_vals)
        w_decode = _wilcoxon_signed_rank(decode_vals)
        p_rows.append(
            {
                "model_id": model_id,
                "profile": profile,
                "arm_id": arm_id,
                "metric": "prefill",
                "n": w_prefill["n"],
                "w": w_prefill["w"],
                "z": w_prefill["z"],
                "p_value": w_prefill["p_value"],
            }
        )
        p_rows.append(
            {
                "model_id": model_id,
                "profile": profile,
                "arm_id": arm_id,
                "metric": "decode",
                "n": w_decode["n"],
                "w": w_decode["w"],
                "z": w_decode["z"],
                "p_value": w_decode["p_value"],
            }
        )

    _holm_correct(p_rows, p_key="p_value", out_key="p_value_holm")

    decode_p_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in p_rows:
        if str(row.get("metric")) != "decode":
            continue
        key = (str(row.get("model_id")), str(row.get("profile")), str(row.get("arm_id")))
        decode_p_map[key] = row

    for ci_row in ci_rows:
        model_id = str(ci_row.get("model_id", ""))
        profile = str(ci_row.get("profile", ""))
        arm_id = str(ci_row.get("arm_id", ""))
        dec_ci_low = ci_row.get("ci95_decode_low")
        dec_ci_high = ci_row.get("ci95_decode_high")
        primary_positive = isinstance(dec_ci_low, (int, float)) and float(dec_ci_low) > float(decode_claim_threshold_pct)
        guardrail = True
        for key in ("ci95_prefill_low", "ci95_decode_low"):
            value = ci_row.get(key)
            if isinstance(value, (int, float)) and float(value) < (-PASS_MAX_REGRESSION_PCT):
                guardrail = False
        claim = bool(primary_positive and guardrail)
        claim_matrix.setdefault(model_id, {}).setdefault(profile, {})[arm_id] = {
            "claim_faster": claim,
            "primary_positive": primary_positive,
            "guardrail_pass": guardrail,
        }

        p_info = decode_p_map.get((model_id, profile, arm_id), {})
        reason = "decode_ci_threshold_and_guardrail_passed" if claim else "threshold_or_guardrail_failed"
        decision = ClaimDecision(
            scope="profile",
            model_id=model_id,
            profile=profile,
            arm_id=arm_id,
            metric="decode",
            threshold_pct=float(decode_claim_threshold_pct),
            n_blocks_total=int(ci_row.get("n_blocks_total") or 0),
            n_blocks_valid=int(ci_row.get("n_blocks_valid") or 0),
            mean_delta_pct=_to_float(ci_row.get("mean_delta_decode_pct")),
            ci95_low=_to_float(dec_ci_low),
            ci95_high=_to_float(dec_ci_high),
            p_value=_to_float(p_info.get("p_value")),
            p_value_holm=_to_float(p_info.get("p_value_holm")),
            guardrail_pass=guardrail,
            claim_faster=claim,
            profiling_mode=profiling_mode_norm,
            toolchain_fingerprint=manifest.get("toolchain_fingerprint", ""),
            schedule_proof_id=schedule_path.name,
            reason=reason,
        )
        claim_decisions.append(asdict(decision))

    # Aggregate model-level equal-weight (chat,long) decode claims for each arm.
    per_model_arm_profile: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in paired_rows:
        if not row.get("valid"):
            continue
        val = row.get("delta_decode_pct")
        if not isinstance(val, (int, float)):
            continue
        key = (str(row.get("model_id")), str(row.get("arm_id")), str(row.get("profile")))
        per_model_arm_profile.setdefault(key, {})[int(row.get("order_index") or 0)] = float(val)

    model_arm_keys = sorted({(k[0], k[1]) for k in per_model_arm_profile.keys()})
    long_profile_for_claim = "long"
    for candidate in ("long_claim", "long", "long_smoke"):
        if candidate in profiles:
            long_profile_for_claim = candidate
            break
    for model_id, arm_id in model_arm_keys:
        chat = per_model_arm_profile.get((model_id, arm_id, "chat"), {})
        longp = per_model_arm_profile.get((model_id, arm_id, long_profile_for_claim), {})
        common = sorted(set(chat.keys()) & set(longp.keys()))
        if not common:
            continue
        merged = [0.5 * (chat[i] + longp[i]) for i in common]
        ci = _bootstrap_ci(merged, samples=bootstrap_samples, seed=seed + 707)
        w = _wilcoxon_signed_rank(merged)
        guardrail = True
        chat_ci = next((r for r in ci_rows if r.get("model_id") == model_id and r.get("arm_id") == arm_id and r.get("profile") == "chat"), None)
        long_ci = next(
            (
                r
                for r in ci_rows
                if r.get("model_id") == model_id
                and r.get("arm_id") == arm_id
                and r.get("profile") == long_profile_for_claim
            ),
            None,
        )
        for row in [chat_ci, long_ci]:
            if not isinstance(row, dict):
                continue
            for key in ("ci95_prefill_low", "ci95_decode_low"):
                value = row.get(key)
                if isinstance(value, (int, float)) and float(value) < (-PASS_MAX_REGRESSION_PCT):
                    guardrail = False
        ci_low = _to_float(ci.get("ci_low"))
        primary_positive = isinstance(ci_low, (int, float)) and float(ci_low) > float(decode_claim_threshold_pct)
        claim = bool(primary_positive and guardrail)
        reason = "decode_mean_ci_threshold_and_guardrail_passed" if claim else "threshold_or_guardrail_failed"
        decision = ClaimDecision(
            scope="model_equal_weight_chat_long",
            model_id=model_id,
            arm_id=arm_id,
            profile=long_profile_for_claim,
            metric="decode",
            threshold_pct=float(decode_claim_threshold_pct),
            n_blocks_total=len(common),
            n_blocks_valid=len(common),
            mean_delta_pct=_to_float(ci.get("mean")),
            ci95_low=ci_low,
            ci95_high=_to_float(ci.get("ci_high")),
            p_value=_to_float(w.get("p_value")),
            p_value_holm=None,
            guardrail_pass=guardrail,
            claim_faster=claim,
            profiling_mode=profiling_mode_norm,
            toolchain_fingerprint=manifest.get("toolchain_fingerprint", ""),
            schedule_proof_id=schedule_path.name,
            reason=reason,
        )
        claim_decisions.append(asdict(decision))

    throughput_report = _build_throughput_report(attempt_records)
    compile_success_rate = _to_float(throughput_report.get("compile_success_rate"))
    gate_b_pass_rate = _to_float(throughput_report.get("gate_b_pass_rate"))
    gate_c_pass_rate = _to_float(throughput_report.get("gate_c_pass_rate"))
    gate_d_pass_rate = _to_float(throughput_report.get("gate_d_pass_rate"))
    candidate_pass_through = {
        "attempts_total": int(throughput_report.get("attempts_total") or 0),
        "gate_a_evaluated": int(throughput_report.get("gate_a_evaluated") or 0),
        "gate_b_evaluated": int(throughput_report.get("gate_b_evaluated") or 0),
        "gate_c_evaluated": int(throughput_report.get("gate_c_evaluated") or 0),
        "gate_d_evaluated": int(throughput_report.get("gate_d_evaluated") or 0),
        "gate_a_pass_count": int(throughput_report.get("gate_a_pass_count") or 0),
        "gate_b_pass_count": int(throughput_report.get("gate_b_pass_count") or 0),
        "gate_c_pass_count": int(throughput_report.get("gate_c_pass_count") or 0),
        "gate_d_pass_count": int(throughput_report.get("gate_d_pass_count") or 0),
        "compile_success_rate": compile_success_rate,
        "gate_b_pass_rate": gate_b_pass_rate,
        "gate_c_pass_rate": gate_c_pass_rate,
        "gate_d_pass_rate": gate_d_pass_rate,
        "readiness_compile_ge_0_70": bool(
            compile_success_rate is not None and compile_success_rate >= 0.70
        ),
        "readiness_gate_b_ge_0_35": bool(
            gate_b_pass_rate is not None and gate_b_pass_rate >= 0.35
        ),
    }

    summary_obj = StudySummary(
        success=True,
        output_dir=str(output_dir),
        generated_at_utc=_utcnow(),
        models_tested=len(run_models),
        profiles_tested=list(profiles),
        arms_tested=list(arms),
        total_blocks=len(paired_rows),
        total_runs=order_index,
        invalid_blocks=invalid_blocks_main,
        invalid_runs=invalid_runs,
        block_failure_rate=((invalid_blocks_main / len(paired_rows)) if paired_rows else 0.0),
        run_failure_rate=((invalid_runs / order_index) if order_index else 0.0),
        claim_matrix=claim_matrix,
    )

    _json_dump(
        summary_path,
        {
            "summary": asdict(summary_obj),
            "ci_results": ci_rows,
            "pvalues": p_rows,
            "claim_decisions": claim_decisions,
            "candidate_pass_through": candidate_pass_through,
            "throughput_report_path": str(throughput_report_json_path),
            "provenance": {
                "git_commit": git_commit,
                "llamacpp_commit": llama_commit,
                "invalid_blocks_warmup": invalid_blocks_warmup,
                "invalid_blocks_measurement": invalid_blocks_main,
            },
        },
    )
    _json_dump(throughput_report_json_path, throughput_report)
    _write_csv(
        throughput_report_csv_path,
        [
            {
                "attempts_total": int(throughput_report.get("attempts_total") or 0),
                "gate_a_evaluated": int(throughput_report.get("gate_a_evaluated") or 0),
                "gate_b_evaluated": int(throughput_report.get("gate_b_evaluated") or 0),
                "gate_c_evaluated": int(throughput_report.get("gate_c_evaluated") or 0),
                "gate_d_evaluated": int(throughput_report.get("gate_d_evaluated") or 0),
                "gate_a_pass_count": int(throughput_report.get("gate_a_pass_count") or 0),
                "gate_b_pass_count": int(throughput_report.get("gate_b_pass_count") or 0),
                "gate_c_pass_count": int(throughput_report.get("gate_c_pass_count") or 0),
                "gate_d_pass_count": int(throughput_report.get("gate_d_pass_count") or 0),
                "compile_success_rate": throughput_report.get("compile_success_rate"),
                "gate_b_pass_rate": throughput_report.get("gate_b_pass_rate"),
                "gate_c_pass_rate": throughput_report.get("gate_c_pass_rate"),
                "gate_d_pass_rate": throughput_report.get("gate_d_pass_rate"),
                "dispatch_metallib_load_rate": throughput_report.get("dispatch_metallib_load_rate"),
                "dispatch_audit_status_counts": json.dumps(
                    throughput_report.get("dispatch_audit_status_counts") or {}
                ),
                "candidate_resources_expected_count": int(
                    throughput_report.get("candidate_resources_expected_count") or 0
                ),
                "candidate_resources_used_count": int(
                    throughput_report.get("candidate_resources_used_count") or 0
                ),
                "candidate_resources_used_rate": throughput_report.get("candidate_resources_used_rate"),
                "audit_missing_count": int(throughput_report.get("audit_missing_count") or 0),
                "audit_parse_fail_count": int(
                    throughput_report.get("audit_parse_fail_count") or 0
                ),
                "backend_noaudit_count": int(throughput_report.get("backend_noaudit_count") or 0),
                "dispatch_runs_total": ((throughput_report.get("dispatch_rule_coverage") or {}).get("dispatch_runs_total")),
                "dispatch_rule_unique_count": ((throughput_report.get("dispatch_rule_coverage") or {}).get("unique_rule_count")),
                "dispatch_rule_coverage_rate": ((throughput_report.get("dispatch_rule_coverage") or {}).get("coverage_rate")),
                "dispatch_rule_ids": json.dumps(((throughput_report.get("dispatch_rule_coverage") or {}).get("unique_rule_ids") or [])),
                "top_rejection_reasons": json.dumps(throughput_report.get("top_rejection_reasons") or []),
                "top_compile_stderr_hashes": json.dumps(throughput_report.get("top_compile_stderr_hashes") or []),
                "top_dispatched_kernels": json.dumps(throughput_report.get("top_dispatched_kernels") or []),
            }
        ],
        fieldnames=[
            "attempts_total",
            "gate_a_evaluated",
            "gate_b_evaluated",
            "gate_c_evaluated",
            "gate_d_evaluated",
            "gate_a_pass_count",
            "gate_b_pass_count",
            "gate_c_pass_count",
            "gate_d_pass_count",
            "compile_success_rate",
            "gate_b_pass_rate",
            "gate_c_pass_rate",
            "gate_d_pass_rate",
            "dispatch_metallib_load_rate",
            "dispatch_audit_status_counts",
            "candidate_resources_expected_count",
            "candidate_resources_used_count",
            "candidate_resources_used_rate",
            "audit_missing_count",
            "audit_parse_fail_count",
            "backend_noaudit_count",
            "dispatch_runs_total",
            "dispatch_rule_unique_count",
            "dispatch_rule_coverage_rate",
            "dispatch_rule_ids",
            "top_rejection_reasons",
            "top_compile_stderr_hashes",
            "top_dispatched_kernels",
        ],
    )
    _json_dump(claim_decisions_path, {"rows": claim_decisions})
    _json_dump(
        hotspots_path,
        {
            "rows": hotspot_rows,
            "summary": {
                "count": len(hotspot_rows),
                "profiles": sorted({str(r.get("profile", "")) for r in hotspot_rows}),
                "arms": sorted({str(r.get("arm_id", "")) for r in hotspot_rows}),
            },
        },
    )
    _json_dump(
        hotspots_op_perf_path,
        {
            "rows": [
                row
                for row in op_profile_rows
                if str(row.get("profiling_mode", "")).strip().lower() == "test-backend-ops-sql"
            ],
            "summary": {
                "count": sum(
                    1
                    for row in op_profile_rows
                    if str(row.get("profiling_mode", "")).strip().lower() == "test-backend-ops-sql"
                ),
                "models": sorted({str(r.get("model_id", "")) for r in op_profile_rows}),
                "profiles": sorted({str(r.get("profile", "")) for r in op_profile_rows}),
            },
        },
    )
    _json_dump(
        op_profiles_path,
        {
            "rows": op_profile_rows,
            "summary": {
                "count": len(op_profile_rows),
                "models": sorted({str(r.get("model_id", "")) for r in op_profile_rows}),
                "profiles": sorted({str(r.get("profile", "")) for r in op_profile_rows}),
            },
        },
    )
    _json_dump(
        roofline_path,
        build_roofline_analysis(
            device=device,
            metrics_rows=metrics_rows,
            target_uplift_pct=float(decode_claim_threshold_pct),
        ),
    )

    _write_csv(
        metrics_csv_path,
        metrics_rows,
        [
            "model_id",
            "model_sha256",
            "profile",
            "arm_id",
            "block_id",
            "order_index",
            "order_label",
            "is_warmup",
            "runtime_args",
            "kernel_template_version",
            "patch_hash",
            "candidate_source_hash",
            "compile_result",
            "compile_stderr_hash",
            "compile_warmup_done",
            "pipeline_cache_key",
            "compile_time_ms",
            "profiling_mode",
            "op_perf_status",
            "op_perf_cache_hit",
            "op_perf_rows_emitted",
            "op_perf_cache_key",
            "prompt_cache_mode",
            "prompt_cache_file",
            "prompt_cache_build_elapsed_ms",
            "prompt_cache_isolated",
            "dispatch_rule_id",
            "metallib_path",
            "metallib_present",
            "metallib_source",
            "dispatch_audit_status",
            "candidate_resources_expected",
            "candidate_resources_used",
            "dispatch_audit_path",
            "dispatch_audit_source",
            "top_dispatched_kernels",
            "prompt_tokens_target",
            "prompt_tokens_actual",
            "prompt_tokens_target_met",
            "prefill_tokens_per_sec",
            "decode_tokens_per_sec",
            "ttft_ms",
            "p50_token_latency_ms",
            "p95_token_latency_ms",
            "peak_memory_mib",
            "elapsed_seconds",
            "all_return_codes_zero",
            "all_metrics_present",
            "valid",
            "invalid_reason",
        ],
    )

    _write_csv(
        paired_csv_path,
        paired_rows,
        [
            "model_id",
            "model_sha256",
            "profile",
            "arm_id",
            "block_id",
            "order_index",
            "order_label",
            "delta_prefill_pct",
            "delta_decode_pct",
            "delta_ttft_pct",
            "valid",
            "invalid_reason",
        ],
    )

    _write_csv(
        ci_csv_path,
        ci_rows,
        [
            "model_id",
            "profile",
            "arm_id",
            "n_blocks_total",
            "n_blocks_valid",
            "failure_rate_blocks",
            "mean_delta_prefill_pct",
            "stdev_delta_prefill_pct",
            "ci95_prefill_low",
            "ci95_prefill_high",
            "mean_delta_decode_pct",
            "stdev_delta_decode_pct",
            "ci95_decode_low",
            "ci95_decode_high",
            "mean_delta_ttft_pct",
            "stdev_delta_ttft_pct",
            "ci95_ttft_low",
            "ci95_ttft_high",
        ],
    )

    _write_csv(
        pval_csv_path,
        p_rows,
        [
            "model_id",
            "profile",
            "arm_id",
            "metric",
            "n",
            "w",
            "z",
            "p_value",
            "p_value_holm",
        ],
    )
    _write_csv(
        exclusions_csv_path,
        exclusion_rows,
        [
            "kind",
            "run_id",
            "block_id",
            "attempt_id",
            "model_id",
            "profile",
            "arm_id",
            "order_index",
            "reason",
            "return_code",
            "compile_classification",
            "correctness_classification",
            "feasibility_classification",
        ],
    )

    _write_basic_plots(plots_dir, ci_rows, paired_rows, attempt_records)
    _write_methods_note(
        output_dir=output_dir,
        manifest=manifest,
        summary=summary_obj,
        ci_rows=ci_rows,
        p_rows=p_rows,
    )

    return summary_obj
