from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_op(name: str) -> str:
    return str(name).replace(".", "_").replace("/", "_")


def _required_ops(project_dir: Path) -> list[str]:
    summary = _load_json(project_dir / "io" / "summary.json")
    counts = summary.get("op_counts")
    if isinstance(counts, dict):
        return sorted(_normalize_op(op) for op in counts.keys())
    individual_ops = project_dir / "io" / "individual_ops"
    if individual_ops.exists():
        return sorted(path.name for path in individual_ops.iterdir() if path.is_dir())
    return []


def _fallback_reports(project_dir: Path) -> dict[str, dict[str, Any]]:
    generated_root = project_dir / "kernels" / "generated" / "individual_op_kernels"
    reports: dict[str, dict[str, Any]] = {}
    if not generated_root.exists():
        return reports
    for op_dir in generated_root.iterdir():
        if not op_dir.is_dir():
            continue
        for rel in ("fallback_required.json", "attempts/fallback.json"):
            report = _load_json(op_dir / rel)
            if report:
                reports[op_dir.name] = report
                break
    return reports


def _failure_reports(project_dir: Path) -> dict[str, dict[str, Any]]:
    generated_root = project_dir / "kernels" / "generated" / "individual_op_kernels"
    reports: dict[str, dict[str, Any]] = {}
    if not generated_root.exists():
        return reports
    for op_dir in generated_root.iterdir():
        if not op_dir.is_dir():
            continue
        report = _load_json(op_dir / "attempts" / "failure.json")
        if report:
            reports[op_dir.name] = report
    return reports


def collect_cast_policy_metadata(
    project_dir: str | Path,
    *,
    selected_kernel_map: dict[str, Any] | None = None,
    skipped_ops: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_path = Path(project_dir)
    required = _required_ops(project_path)
    selected = {_normalize_op(op) for op in (selected_kernel_map or {}).keys()}
    manual_fallback = {_normalize_op(op) for op in (skipped_ops or {}).keys()}
    fallback_reports = _fallback_reports(project_path)
    failure_reports = _failure_reports(project_path)

    fallback_ops = set(manual_fallback)
    fallback_reasons: dict[str, str] = {}

    for op, report in fallback_reports.items():
        fallback_ops.add(_normalize_op(op))
        fallback_reasons[_normalize_op(op)] = str(
            report.get("reason")
            or report.get("message")
            or "requires_torch_fallback"
        )

    for op in manual_fallback:
        fallback_reasons.setdefault(op, "manual_torch_fallback")

    missing_required = [
        op for op in required if op not in selected and op not in fallback_ops
    ]
    for op in missing_required:
        fallback_ops.add(op)
        fallback_reasons.setdefault(op, "missing_forged_kernel")

    failed_ops: dict[str, str] = {}
    for op, report in failure_reports.items():
        norm = _normalize_op(op)
        failed_ops[norm] = str(
            report.get("reason")
            or report.get("message")
            or report.get("stage")
            or "generation_failed"
        )

    torch_fallback_ops = sorted(fallback_ops)
    full_forged_publishable = not torch_fallback_ops and not missing_required
    mixed_forged_publishable = all(
        op in selected or op in fallback_ops for op in required
    )

    return {
        "required_ops": required,
        "selected_ops": sorted(selected),
        "torch_fallback_ops": torch_fallback_ops,
        "fallback_reasons": {
            op: fallback_reasons.get(op, "torch_fallback")
            for op in torch_fallback_ops
        },
        "failed_ops": failed_ops,
        "missing_required_ops": missing_required,
        "full_forged_publishable": full_forged_publishable,
        "mixed_forged_publishable": mixed_forged_publishable,
        "mixed_equals_full": not torch_fallback_ops,
        "cast_policy": "mixed_forged" if torch_fallback_ops else "full_forged",
    }
