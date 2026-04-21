from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .provenance import safe_sha256_path

POLICY_AUTO_BEST_FASTEST_VALID = "auto_best_fastest_valid"
_EVIDENCE_RANK = {
    "deployment": 0,
    "operator": 1,
    "micro_only": 2,
    "missing": 3,
}


class CastSelectionError(RuntimeError):
    pass


class NoEligibleCastKernelsError(CastSelectionError):
    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "ok", "passed", "pass"}:
        return True
    if text in {"false", "0", "no", "n", "failed", "fail"}:
        return False
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def _coerce_latency_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    samples: list[float] = []
    for item in value:
        parsed = _coerce_float(item)
        if parsed is None:
            continue
        samples.append(parsed)
    return samples


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_timestamp(path: Path | None) -> datetime | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _resolve_path(raw_path: Any, *, base_dir: Path | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def _relative_or_none(path: Path | None, root: Path | None) -> str | None:
    if path is None:
        return None
    if root is None:
        return str(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _artifact_timestamp(path: Path, payload: Mapping[str, Any] | None) -> datetime | None:
    if isinstance(payload, Mapping):
        parsed = _parse_timestamp(payload.get("timestamp"))
        if parsed is not None:
            return parsed
    return _file_timestamp(path)


def _latency_stats(candidate: Mapping[str, Any]) -> tuple[float | None, float | None, float | None, list[float]]:
    samples = _coerce_latency_list(candidate.get("latency_samples_ms"))
    median_ms = _coerce_float(candidate.get("median_latency_ms"))
    mean_ms = _coerce_float(candidate.get("mean_latency_ms"))
    p95_ms = _coerce_float(candidate.get("p95_latency_ms"))

    if samples:
        median_ms = statistics.median(samples)
        mean_ms = statistics.fmean(samples)
        if len(samples) == 1:
            p95_ms = samples[0]
        else:
            p95_ms = statistics.quantiles(samples, n=20, method="inclusive")[-1]

    if median_ms is None and mean_ms is not None:
        median_ms = mean_ms
    if mean_ms is None and median_ms is not None:
        mean_ms = median_ms
    if p95_ms is None and median_ms is not None:
        p95_ms = median_ms
    return median_ms, p95_ms, mean_ms, samples


def _candidate_reference(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reference = candidate.get("benchmark_reference")
    if isinstance(reference, dict):
        return dict(reference)
    return {
        "artifact_path": candidate.get("benchmark_artifact_path"),
        "row_ref": candidate.get("benchmark_row_ref"),
    }


def _normalize_error_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _evaluate_candidate(
    candidate: Mapping[str, Any],
    *,
    project_root: Path | None,
    allow_operator_only: bool,
    allow_micro_only: bool,
    unsafe_override: bool,
) -> dict[str, Any]:
    evidence_tier = str(candidate.get("evidence_tier") or "missing").strip() or "missing"
    kernel_source_path = _resolve_path(candidate.get("kernel_source_path"), base_dir=project_root)
    benchmark_source_path = _resolve_path(candidate.get("benchmark_source_path"), base_dir=project_root)
    benchmark_artifact_path = _resolve_path(candidate.get("benchmark_artifact_path"), base_dir=project_root)

    median_ms, p95_ms, mean_ms, samples = _latency_stats(candidate)
    candidate_source_hash = safe_sha256_path(kernel_source_path) if kernel_source_path else None
    benchmark_source_hash = str(candidate.get("benchmark_source_hash")) if candidate.get("benchmark_source_hash") else None

    timing_verified_at = _parse_timestamp(candidate.get("timing_verified_timestamp"))
    audit_verified_at = _parse_timestamp(candidate.get("audit_verified_timestamp"))
    verified_at = _parse_timestamp(candidate.get("verified_timestamp"))
    if verified_at is None:
        verified_candidates = [dt for dt in [timing_verified_at, audit_verified_at] if dt is not None]
        if verified_candidates:
            verified_at = min(verified_candidates)

    rejection_reasons: list[str] = []
    if not candidate.get("has_captured_inputs", False):
        rejection_reasons.append("missing captured inputs")

    correctness_passed = _coerce_bool(candidate.get("correctness_passed"))
    if correctness_passed is not True:
        rejection_reasons.append("correctness failed")

    if not candidate.get("timing_real", False):
        rejection_reasons.append("timing not real")
    if median_ms is None:
        rejection_reasons.append("missing measured latency")
    if candidate.get("is_tier_winner") is False:
        rejection_reasons.append("baseline faster than kernel")

    if kernel_source_path is None or not kernel_source_path.exists():
        rejection_reasons.append("missing kernel source")
    if benchmark_source_path is None:
        rejection_reasons.append("missing benchmark source")
    elif not benchmark_source_path.exists():
        rejection_reasons.append("missing benchmark source")
    elif kernel_source_path is not None and kernel_source_path != benchmark_source_path:
        rejection_reasons.append("benchmark source path mismatch")

    if benchmark_source_hash and candidate_source_hash and benchmark_source_hash != candidate_source_hash:
        rejection_reasons.append("stale source hash mismatch")

    source_file_mtime = _file_timestamp(kernel_source_path)
    if benchmark_source_hash is None:
        if verified_at is None:
            rejection_reasons.append("stale reuse without matching hashes")
        elif source_file_mtime is not None and source_file_mtime > verified_at:
            rejection_reasons.append("stale benchmark row")

    if candidate.get("stale") is True:
        rejection_reasons.append("stale benchmark row")

    benchmark_errors = _normalize_error_list(candidate.get("benchmark_errors"))
    if benchmark_errors:
        rejection_reasons.append("benchmark errors")

    if candidate.get("fallback_only") is True:
        rejection_reasons.append("fallback-only candidate")

    runtime_audit_passed = _coerce_bool(candidate.get("runtime_audit_passed"))
    if runtime_audit_passed is False:
        rejection_reasons.append("deployment/runtime audit failed")

    if evidence_tier == "operator" and not allow_operator_only:
        rejection_reasons.append("operator-only evidence not allowed")
    if evidence_tier == "micro_only" and not allow_micro_only:
        rejection_reasons.append("micro-only evidence not allowed")
    if evidence_tier == "missing":
        rejection_reasons.append("missing evidence tier")

    unsafe = bool(candidate.get("unsafe")) or runtime_audit_passed is False
    if unsafe and not unsafe_override:
        rejection_reasons.append("unsafe kernel blocked")

    stable_id = (
        str(candidate.get("stable_id") or "")
        or _relative_or_none(kernel_source_path, project_root)
        or str(candidate.get("candidate_id") or "")
    )

    eligible = len(rejection_reasons) == 0
    paper_eligible = eligible and evidence_tier == "deployment" and not unsafe

    return {
        **dict(candidate),
        "candidate_id": str(candidate.get("candidate_id") or stable_id or candidate.get("op") or ""),
        "op": str(candidate.get("op") or ""),
        "stable_id": stable_id,
        "evidence_tier": evidence_tier,
        "kernel_source_path": str(kernel_source_path) if kernel_source_path else None,
        "kernel_source_repo_relpath": _relative_or_none(kernel_source_path, project_root),
        "benchmark_source_path": str(benchmark_source_path) if benchmark_source_path else None,
        "benchmark_source_repo_relpath": _relative_or_none(benchmark_source_path, project_root),
        "selected_source_hash": candidate_source_hash,
        "benchmark_source_hash": benchmark_source_hash,
        "benchmark_artifact_path": str(benchmark_artifact_path) if benchmark_artifact_path else candidate.get("benchmark_artifact_path"),
        "benchmark_reference": _candidate_reference(candidate),
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
        "mean_latency_ms": mean_ms,
        "latency_samples_ms": samples if samples else _coerce_latency_list(candidate.get("latency_samples_ms")),
        "timing_verified_timestamp": _timestamp_to_iso(timing_verified_at),
        "audit_verified_timestamp": _timestamp_to_iso(audit_verified_at),
        "verified_timestamp": _timestamp_to_iso(verified_at),
        "benchmark_errors": benchmark_errors,
        "runtime_audit_passed": runtime_audit_passed,
        "unsafe": unsafe,
        "eligible": eligible,
        "paper_eligible": paper_eligible,
        "rejection_reasons": sorted(dict.fromkeys(rejection_reasons)),
    }


def _sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence_rank = _EVIDENCE_RANK.get(str(candidate.get("evidence_tier") or "missing"), 99)
    median_ms = _coerce_float(candidate.get("median_latency_ms"))
    p95_ms = _coerce_float(candidate.get("p95_latency_ms"))
    mean_ms = _coerce_float(candidate.get("mean_latency_ms"))
    verified_at = _parse_timestamp(candidate.get("verified_timestamp"))
    verified_rank = 1 if verified_at is None else 0
    verified_sort = 0.0 if verified_at is None else -verified_at.timestamp()
    stable_id = str(candidate.get("stable_id") or candidate.get("candidate_id") or "")
    return (
        evidence_rank,
        median_ms if median_ms is not None else 1e300,
        p95_ms if p95_ms is not None else 1e300,
        mean_ms if mean_ms is not None else 1e300,
        verified_rank,
        verified_sort,
        stable_id,
    )


def select_fastest_valid_kernels(
    candidates_by_op: Mapping[str, list[Mapping[str, Any]]],
    *,
    project_root: str | Path | None = None,
    policy_name: str = POLICY_AUTO_BEST_FASTEST_VALID,
    allow_operator_only: bool = True,
    allow_micro_only: bool = False,
    unsafe_override: bool = False,
    fail_if_empty: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve(strict=False) if project_root else None
    selected_ops: dict[str, dict[str, Any]] = {}
    rejected_candidates: dict[str, list[dict[str, Any]]] = {}
    unselected_ops: list[str] = []

    for op_name in sorted(candidates_by_op.keys()):
        raw_candidates = list(candidates_by_op.get(op_name) or [])
        evaluated = [
            _evaluate_candidate(
                {**dict(candidate), "op": op_name},
                project_root=root,
                allow_operator_only=allow_operator_only,
                allow_micro_only=allow_micro_only,
                unsafe_override=unsafe_override,
            )
            for candidate in raw_candidates
        ]
        evaluated.sort(key=_sort_key)
        eligible = [candidate for candidate in evaluated if candidate["eligible"]]
        rejected_candidates[op_name] = [candidate for candidate in evaluated if not candidate["eligible"]]

        if not eligible:
            unselected_ops.append(op_name)
            continue

        selected = dict(eligible[0])
        selected["selection_reason"] = (
            f"{policy_name}: selected {selected['evidence_tier']} candidate with median "
            f"{selected['median_latency_ms']:.6f} ms"
        )
        selected_ops[op_name] = selected

    exportable = bool(selected_ops)
    export_paper_eligible = (
        exportable
        and not unselected_ops
        and all(bool(candidate.get("paper_eligible")) for candidate in selected_ops.values())
    )
    report = {
        "policy_name": policy_name,
        "project_root": str(root) if root else None,
        "allow_operator_only": bool(allow_operator_only),
        "allow_micro_only": bool(allow_micro_only),
        "unsafe_override": bool(unsafe_override),
        "selected_ops": selected_ops,
        "selected_kernel_map": {
            op_name: candidate["kernel_source_path"]
            for op_name, candidate in selected_ops.items()
            if candidate.get("kernel_source_path")
        },
        "selected_op_count": len(selected_ops),
        "rejected_candidates": rejected_candidates,
        "unselected_ops": unselected_ops,
        "exportable": exportable,
        "export_paper_eligible": export_paper_eligible,
    }
    if fail_if_empty and not exportable:
        raise NoEligibleCastKernelsError(
            "No eligible kernels matched auto_best_fastest_valid. Review rejected_candidates for details.",
            report=report,
        )
    return report


def _project_op_rows(project_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int], datetime | None, Path]:
    op_benchmarks_path = project_root / "benchmarks" / "op_benchmarks.json"
    payload = _load_json(op_benchmarks_path) or {}
    rows_by_op: dict[str, dict[str, Any]] = {}
    row_index_by_op: dict[str, int] = {}
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                continue
            op_name = str(row.get("op") or "").strip()
            if not op_name:
                continue
            rows_by_op[op_name] = row
            row_index_by_op[op_name] = index
    return rows_by_op, row_index_by_op, _artifact_timestamp(op_benchmarks_path, payload), op_benchmarks_path


def _project_qwen_compare(project_root: Path) -> tuple[dict[str, Any], datetime | None, Path]:
    compare_path = project_root / "benchmarks" / "qwen_tps_compare.json"
    payload = _load_json(compare_path) or {}
    return payload, _artifact_timestamp(compare_path, payload), compare_path


def _captured_entry_state(project_root: Path, op_name: str, benchmarked_files: list[str]) -> tuple[bool, int]:
    entries_dir = project_root / "io" / "individual_ops" / op_name
    if not entries_dir.exists():
        return False, 0
    if benchmarked_files:
        existing = sum(1 for item in benchmarked_files if (entries_dir / item).exists())
        return existing > 0, existing
    files = list(entries_dir.glob("entry_*.pt"))
    return bool(files), len(files)


def _operator_source_path(project_root: Path, op_name: str, row: Mapping[str, Any], deployment_source_path: Path | None) -> Path | None:
    explicit = (
        row.get("kernel_source_path")
        or row.get("kernel_source_relpath")
        or row.get("kernel_source")
    )
    if explicit:
        return _resolve_path(explicit, base_dir=project_root)
    if deployment_source_path is not None and str(row.get("winner") or "") == "optimized":
        return deployment_source_path
    generated_path = project_root / "kernels" / "generated" / "individual_op_kernels" / op_name / "kernel.cu"
    if generated_path.exists() and str(row.get("kernel_status") or "") == "ok":
        return generated_path
    return None


def _build_project_candidates(project_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(project_root).expanduser().resolve()
    rows_by_op, row_index_by_op, op_bench_ts, op_benchmarks_path = _project_op_rows(root)
    qwen_compare, qwen_ts, qwen_compare_path = _project_qwen_compare(root)
    forged = qwen_compare.get("forged") if isinstance(qwen_compare, dict) else {}
    patch_sources = forged.get("patch_sources") if isinstance(forged, dict) else {}
    patch_stats = forged.get("patch_stats") if isinstance(forged, dict) else {}

    all_ops = set(rows_by_op.keys())
    io_root = root / "io" / "individual_ops"
    if io_root.exists():
        all_ops.update(child.name for child in io_root.iterdir() if child.is_dir())

    candidates_by_op: dict[str, list[dict[str, Any]]] = {}
    for op_name in sorted(all_ops):
        row = rows_by_op.get(op_name, {})
        deployment_source_path = _resolve_path(
            (
                row.get("deployment_source_path")
                or row.get("deployment_kernel_source_path")
                or (patch_sources.get(op_name) if isinstance(patch_sources, Mapping) else None)
            ),
            base_dir=root,
        )
        deployment_stats = patch_stats.get(op_name, {}) if isinstance(patch_stats, Mapping) else {}
        deployment_correctness = row.get("deployment_correctness") if isinstance(row, Mapping) else {}
        benchmarked_files = [
            str(item)
            for item in (
                row.get("integrated_kernel_benchmarked_entry_files")
                or row.get("benchmarked_entry_files")
                or []
            )
        ]
        has_entries, entry_count = _captured_entry_state(root, op_name, benchmarked_files)

        candidates: list[dict[str, Any]] = []
        if row:
            deployment_errors = []
            if isinstance(deployment_correctness, Mapping):
                deployment_errors.extend(_normalize_error_list(deployment_correctness.get("errors")))
            if str(row.get("integrated_kernel_status") or "") and str(row.get("integrated_kernel_status") or "") != "ok":
                deployment_errors.append(str(row.get("integrated_kernel_status")))

            deployment_candidate = {
                "op": op_name,
                "candidate_id": f"{op_name}:deployment",
                "kernel_source_path": str(deployment_source_path) if deployment_source_path else None,
                "benchmark_source_path": str(deployment_source_path) if deployment_source_path else None,
                "benchmark_source_hash": row.get("deployment_source_hash") or row.get("deployment_kernel_source_hash"),
                "benchmark_artifact_path": str(op_benchmarks_path),
                "benchmark_row_ref": f"results[{row_index_by_op.get(op_name, -1)}]",
                "benchmark_reference": {
                    "artifact_path": str(op_benchmarks_path),
                    "row_ref": f"results[{row_index_by_op.get(op_name, -1)}]",
                    "audit_artifact_path": str(qwen_compare_path) if qwen_compare_path.exists() else None,
                },
                "evidence_tier": "deployment",
                "latency_samples_ms": row.get("integrated_kernel_entry_latencies_ms") or [],
                "median_latency_ms": row.get("integrated_kernel_ms"),
                "correctness_passed": (
                    _coerce_bool(deployment_correctness.get("strict_pass"))
                    if isinstance(deployment_correctness, Mapping)
                    else None
                ),
                "has_captured_inputs": has_entries and entry_count > 0,
                "timing_real": (
                    str(row.get("integrated_kernel_status") or "") == "ok"
                    and not bool(row.get("integrated_kernel_estimated"))
                ),
                "is_tier_winner": str(row.get("deployment_safe_winner") or row.get("deployment_winner") or "") == "optimized",
                "benchmark_errors": deployment_errors,
                "runtime_audit_passed": (
                    bool(deployment_stats.get("calls", 0)) and int(deployment_stats.get("fallback", 0) or 0) == 0
                    and not str(deployment_stats.get("last_error") or "").strip()
                ) if deployment_stats else None,
                "fallback_only": (
                    bool(deployment_stats.get("calls", 0))
                    and int(deployment_stats.get("kernel_success", 0) or 0) <= 0
                ) if deployment_stats else False,
                "timing_verified_timestamp": _timestamp_to_iso(op_bench_ts),
                "audit_verified_timestamp": _timestamp_to_iso(qwen_ts),
                "verified_timestamp": _timestamp_to_iso(min(dt for dt in [op_bench_ts, qwen_ts] if dt is not None))
                if any(dt is not None for dt in [op_bench_ts, qwen_ts])
                else None,
            }
            candidates.append(deployment_candidate)

            operator_source_path = _operator_source_path(root, op_name, row, deployment_source_path)
            operator_correctness = row.get("kernel_correctness")
            operator_strict_pass = None
            if isinstance(operator_correctness, Mapping):
                operator_strict_pass = _coerce_bool(operator_correctness.get("strict_pass"))
            elif operator_source_path is not None and deployment_source_path is not None and operator_source_path == deployment_source_path:
                operator_strict_pass = (
                    _coerce_bool(deployment_correctness.get("strict_pass"))
                    if isinstance(deployment_correctness, Mapping)
                    else None
                )
            operator_tier = "operator" if operator_strict_pass is True else "micro_only"
            operator_errors = []
            if isinstance(operator_correctness, Mapping):
                operator_errors.extend(_normalize_error_list(operator_correctness.get("errors")))
            if str(row.get("kernel_status") or "") and str(row.get("kernel_status") or "") != "ok":
                operator_errors.append(str(row.get("kernel_status")))
            operator_candidate = {
                "op": op_name,
                "candidate_id": f"{op_name}:operator",
                "kernel_source_path": str(operator_source_path) if operator_source_path else None,
                "benchmark_source_path": str(operator_source_path) if operator_source_path else None,
                "benchmark_source_hash": row.get("kernel_source_hash"),
                "benchmark_artifact_path": str(op_benchmarks_path),
                "benchmark_row_ref": f"results[{row_index_by_op.get(op_name, -1)}]",
                "evidence_tier": operator_tier,
                "latency_samples_ms": row.get("kernel_entry_latencies_ms") or [],
                "median_latency_ms": row.get("kernel_ms"),
                "correctness_passed": operator_strict_pass,
                "has_captured_inputs": has_entries and entry_count > 0,
                "timing_real": (
                    str(row.get("kernel_status") or "") == "ok"
                    and not bool(row.get("kernel_estimated"))
                ),
                "is_tier_winner": str(row.get("winner") or "") == "optimized",
                "benchmark_errors": operator_errors,
                "runtime_audit_passed": (
                    bool(deployment_stats.get("calls", 0)) and int(deployment_stats.get("fallback", 0) or 0) == 0
                    and not str(deployment_stats.get("last_error") or "").strip()
                ) if deployment_stats and operator_source_path == deployment_source_path else None,
                "fallback_only": False,
                "timing_verified_timestamp": _timestamp_to_iso(op_bench_ts),
                "verified_timestamp": _timestamp_to_iso(op_bench_ts),
            }
            candidates.append(operator_candidate)
        candidates_by_op[op_name] = candidates
    return candidates_by_op


def select_project_export_kernels(
    project_root: str | Path,
    *,
    allow_operator_only: bool = True,
    allow_micro_only: bool = False,
    unsafe_override: bool = False,
    fail_if_empty: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    report = select_fastest_valid_kernels(
        _build_project_candidates(root),
        project_root=root,
        allow_operator_only=allow_operator_only,
        allow_micro_only=allow_micro_only,
        unsafe_override=unsafe_override,
        fail_if_empty=fail_if_empty,
    )
    report["project_root"] = str(root)
    return report
