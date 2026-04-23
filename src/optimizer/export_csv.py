"""CLI exporter: dump a project's analytics to four CSVs.

Usage:
    python -m src.optimizer.export_csv <project_name> --out <dir>
                                       [--n-kernels N] [--ops op1,op2]

Reads only; writes best_performance.csv, iterations_to_correct.csv,
failure_modes.csv, token_usage.csv (+ token_usage_calls.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional


CATEGORIES = (
    "CompilationFailed",
    "NumericalMismatch",
    "RuntimeError",
    "ExtractFailed",
    "LLMApiError",
    "SetupError",
    "Control",
    "Unknown",
)


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project(project_name: str) -> Optional[Path]:
    root = _repo_root() / "kernels" / "projects" / project_name
    return root if root.exists() else None


def _trees_root(project_dir: Path) -> Path:
    return project_dir / "trees"


def _discover_optimizer_ops(project_dir: Path, filter_ops: Optional[set[str]]) -> list[Path]:
    """Each direct subdir of trees/ that owns a nodes.db is one operator."""
    out: list[Path] = []
    trees = _trees_root(project_dir)
    if not trees.exists():
        return out
    for child in sorted(trees.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "nodes.db").exists():
            continue
        if filter_ops and child.name not in filter_ops:
            continue
        out.append(child)
    return out


def _discover_generator_ops(project_dir: Path, filter_ops: Optional[set[str]]) -> list[Path]:
    """Generator op dirs under kernels/generated/individual_op_kernels."""
    out: list[Path] = []
    gen = project_dir / "kernels" / "generated" / "individual_op_kernels"
    if not gen.exists():
        return out
    for child in sorted(gen.iterdir()):
        if not child.is_dir():
            continue
        if filter_ops and child.name not in filter_ops:
            continue
        out.append(child)
    return out


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _classify_error_text(text: Optional[str]) -> str:
    if not text:
        return "Unknown"
    if "[Compilation Failed]" in text or "[Import/Compilation Failed]" in text:
        return "CompilationFailed"
    if "[Output Mismatch" in text:
        return "NumericalMismatch"
    if "[Runtime Error" in text:
        return "RuntimeError"
    if "Failed to extract code" in text:
        return "ExtractFailed"
    return "Unknown"


def _classify_generator_stage(stage: str, log_text: Optional[str]) -> str:
    mapping = {
        "llm_api": "LLMApiError",
        "setup": "SetupError",
        "control": "Control",
        "codex_repair": "Unknown",
    }
    if stage in mapping:
        return mapping[stage]
    if stage == "llm_validate":
        return _classify_error_text(log_text)
    return "Unknown"


# ---------------------------------------------------------------------------
# nodes.db helpers
# ---------------------------------------------------------------------------


def _load_nodes(op_dir: Path) -> list[dict]:
    """Return list of node dicts ordered by id ascending, with parent resolved."""
    db = op_dir / "nodes.db"
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, visits, value, median_time_ms, code, "
            "improvement_description, attempts_to_correct, phase "
            "FROM nodes ORDER BY id ASC"
        ).fetchall()
        parents: dict[int, int] = {}
        try:
            for pid, cid in conn.execute("SELECT parent_id, child_id FROM edges"):
                parents[cid] = pid
        except sqlite3.OperationalError:
            pass
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "parent": parents.get(r["id"], -1),
            "value": r["value"],
            "median_time_ms": r["median_time_ms"],
            "code": r["code"],
            "improvement_description": r["improvement_description"],
            "attempts_to_correct": r["attempts_to_correct"],
            "phase": r["phase"],
        })
    return out


def _find_root(nodes: list[dict]) -> Optional[dict]:
    for n in nodes:
        if n["parent"] == -1:
            return n
    return nodes[0] if nodes else None


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def _export_best_performance(
    project_dir: Path, op_dirs: list[Path], n_kernels: Optional[int], out_path: Path
) -> None:
    fields = [
        "operator", "kernels_sampled", "n_kernels_cap",
        "baseline_kernel_id", "baseline_time_ms",
        "best_kernel_id", "best_time_ms", "best_speedup_vs_baseline",
        "best_improvement_description", "source",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for op_dir in op_dirs:
            row = _best_row_for_op(op_dir, n_kernels)
            if row is not None:
                w.writerow(row)


def _best_row_for_op(op_dir: Path, n_kernels: Optional[int]) -> Optional[dict]:
    op = op_dir.name
    best_json = op_dir / "best.json"

    # Prefer best.json only when it matches the requested window.
    if best_json.exists():
        try:
            data = json.loads(best_json.read_text())
        except Exception:
            data = None
        if isinstance(data, dict):
            cap_matches = (
                n_kernels is None
                or int(data.get("n_kernels_cap") or 0) == n_kernels
            )
            if cap_matches:
                return {
                    "operator": op,
                    "kernels_sampled": data.get("kernels_sampled"),
                    "n_kernels_cap": data.get("n_kernels_cap"),
                    "baseline_kernel_id": data.get("baseline_kernel_id"),
                    "baseline_time_ms": data.get("baseline_time_ms"),
                    "best_kernel_id": data.get("best_kernel_id"),
                    "best_time_ms": data.get("best_time_ms"),
                    "best_speedup_vs_baseline": data.get("best_speedup_vs_baseline"),
                    "best_improvement_description": data.get("best_improvement_description"),
                    "source": "best.json",
                }

    nodes = _load_nodes(op_dir)
    if not nodes:
        return None
    profiled = [n for n in nodes if n["value"] is not None]
    if n_kernels is not None:
        profiled = profiled[:n_kernels]
    if not profiled:
        return None
    root = _find_root(nodes)
    baseline_time = root["median_time_ms"] if root else None
    best = min(profiled, key=lambda n: n["value"])
    speedup = (
        baseline_time / best["value"]
        if baseline_time and best["value"] and best["value"] > 0
        else None
    )
    return {
        "operator": op,
        "kernels_sampled": len(profiled),
        "n_kernels_cap": n_kernels,
        "baseline_kernel_id": root["id"] if root else None,
        "baseline_time_ms": baseline_time,
        "best_kernel_id": best["id"],
        "best_time_ms": best["value"],
        "best_speedup_vs_baseline": speedup,
        "best_improvement_description": best["improvement_description"],
        "source": "nodes.db",
    }


def _export_iterations(
    project_dir: Path,
    optimizer_ops: list[Path],
    generator_ops: list[Path],
    out_path: Path,
) -> None:
    fields = ["operator", "phase", "node_id", "attempts_to_correct", "success"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for op_dir in generator_ops:
            summary = op_dir / "attempts" / "summary.json"
            if not summary.exists():
                continue
            try:
                data = json.loads(summary.read_text())
            except Exception:
                continue
            w.writerow({
                "operator": op_dir.name,
                "phase": "generator",
                "node_id": "",
                "attempts_to_correct": data.get("attempts_to_correct"),
                "success": bool(data.get("success")),
            })
        for op_dir in optimizer_ops:
            for n in _load_nodes(op_dir):
                if n["attempts_to_correct"] is None:
                    continue
                w.writerow({
                    "operator": op_dir.name,
                    "phase": "optimizer",
                    "node_id": n["id"],
                    "attempts_to_correct": n["attempts_to_correct"],
                    "success": n["value"] is not None,
                })


def _export_failures(
    project_dir: Path,
    optimizer_ops: list[Path],
    generator_ops: list[Path],
    out_path: Path,
) -> None:
    fields = ["operator", "phase", "category", "count"]
    rows: list[dict] = []

    for op_dir in generator_ops:
        fpath = op_dir / "attempts" / "failure.json"
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        stage = str(data.get("stage") or "")
        log_text = None
        if stage == "llm_validate":
            logs = sorted((op_dir / "attempts").glob("log-*.txt"))
            if logs:
                try:
                    log_text = logs[-1].read_text(errors="replace")
                except Exception:
                    log_text = None
            if log_text is None:
                log_text = str(data.get("message") or "")
        category = _classify_generator_stage(stage, log_text)
        rows.append({
            "operator": op_dir.name,
            "phase": "generator",
            "category": category,
            "count": 1,
        })

    for op_dir in optimizer_ops:
        dump = op_dir / "garbage_dump"
        if not dump.exists():
            continue
        counts: dict[str, int] = {c: 0 for c in CATEGORIES}
        # Pair each kernel_iter*.cu/py with a same-iter/attempt llm_response_*.txt
        # if present; otherwise classify as Unknown.
        responses = {}
        for rtxt in dump.glob("llm_response_iter*_attempt*_*.txt"):
            key = _dump_key(rtxt.name)
            if key:
                responses[key] = rtxt
        seen_keys: set[str] = set()
        for kfile in list(dump.glob("kernel_iter*_attempt*.cu")) + list(dump.glob("kernel_iter*_attempt*.py")):
            key = _dump_key(kfile.name)
            if not key:
                continue
            seen_keys.add(key)
            text = ""
            err_sidecar = kfile.with_suffix(".err.txt")
            if not err_sidecar.exists():
                err_sidecar = dump / f"kernel_{key}.err.txt"
            if err_sidecar.exists():
                try:
                    text = err_sidecar.read_text(errors="replace")
                except Exception:
                    text = ""
            if not text:
                rtxt = responses.get(key)
                if rtxt is not None:
                    try:
                        text = rtxt.read_text(errors="replace")
                    except Exception:
                        text = ""
            counts[_classify_error_text(text)] += 1
        # Response-only entries (no kernel file)
        for key, rtxt in responses.items():
            if key in seen_keys:
                continue
            try:
                text = rtxt.read_text(errors="replace")
            except Exception:
                text = ""
            counts[_classify_error_text(text)] += 1
        for cat, cnt in counts.items():
            if cnt == 0:
                continue
            rows.append({
                "operator": op_dir.name,
                "phase": "optimizer",
                "category": cat,
                "count": cnt,
            })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _dump_key(name: str) -> Optional[str]:
    """Extract 'iter{I}_attempt{A}' shard from a garbage_dump filename."""
    import re
    m = re.search(r"(iter[^_]+_attempt[^_.]+)", name)
    return m.group(1) if m else None


def _export_token_usage(
    project_dir: Path,
    optimizer_ops: list[Path],
    generator_ops: list[Path],
    out_summary: Path,
    out_calls: Path,
) -> None:
    summary_fields = [
        "operator", "phase", "calls", "input_tokens", "output_tokens",
        "reasoning_tokens", "total_tokens", "input_cost_usd",
        "output_cost_usd", "total_cost_usd",
    ]
    call_fields = [
        "operator", "phase", "ts", "step_type", "iteration", "attempt",
        "provider", "model", "input_tokens", "output_tokens",
        "reasoning_tokens", "total_cost_usd",
    ]

    phase_groups = [("generator", generator_ops), ("optimizer", optimizer_ops)]

    with out_summary.open("w", newline="") as sf, out_calls.open("w", newline="") as cf:
        sw = csv.DictWriter(sf, fieldnames=summary_fields)
        cw = csv.DictWriter(cf, fieldnames=call_fields)
        sw.writeheader()
        cw.writeheader()

        grand = {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "input_cost_usd": 0.0,
            "output_cost_usd": 0.0, "total_cost_usd": 0.0,
        }

        for phase, op_dirs in phase_groups:
            phase_total = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "reasoning_tokens": 0, "input_cost_usd": 0.0,
                "output_cost_usd": 0.0, "total_cost_usd": 0.0,
            }
            for op_dir in op_dirs:
                agg = _aggregate_usage(op_dir)
                total_tokens = (
                    agg["input_tokens"] + agg["output_tokens"] + agg["reasoning_tokens"]
                )
                sw.writerow({
                    "operator": op_dir.name,
                    "phase": phase,
                    "calls": agg["calls"],
                    "input_tokens": agg["input_tokens"],
                    "output_tokens": agg["output_tokens"],
                    "reasoning_tokens": agg["reasoning_tokens"],
                    "total_tokens": total_tokens,
                    "input_cost_usd": agg["input_cost_usd"],
                    "output_cost_usd": agg["output_cost_usd"],
                    "total_cost_usd": agg["total_cost_usd"],
                })
                for key in phase_total:
                    phase_total[key] += agg[key]
                for call in _iter_usage_calls(op_dir):
                    cw.writerow({"operator": op_dir.name, "phase": phase, **call})

            sw.writerow({
                "operator": f"TOTAL_{phase.upper()}",
                "phase": phase,
                "calls": phase_total["calls"],
                "input_tokens": phase_total["input_tokens"],
                "output_tokens": phase_total["output_tokens"],
                "reasoning_tokens": phase_total["reasoning_tokens"],
                "total_tokens": (
                    phase_total["input_tokens"] + phase_total["output_tokens"]
                    + phase_total["reasoning_tokens"]
                ),
                "input_cost_usd": phase_total["input_cost_usd"],
                "output_cost_usd": phase_total["output_cost_usd"],
                "total_cost_usd": phase_total["total_cost_usd"],
            })
            for key in grand:
                grand[key] += phase_total[key]

        sw.writerow({
            "operator": "TOTAL",
            "phase": "all",
            "calls": grand["calls"],
            "input_tokens": grand["input_tokens"],
            "output_tokens": grand["output_tokens"],
            "reasoning_tokens": grand["reasoning_tokens"],
            "total_tokens": (
                grand["input_tokens"] + grand["output_tokens"] + grand["reasoning_tokens"]
            ),
            "input_cost_usd": grand["input_cost_usd"],
            "output_cost_usd": grand["output_cost_usd"],
            "total_cost_usd": grand["total_cost_usd"],
        })


def _aggregate_usage(op_dir: Path) -> dict:
    db = op_dir / "llm_usage.db"
    empty = {
        "calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
        "input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0,
    }
    if not db.exists():
        return empty
    try:
        with sqlite3.connect(str(db)) as conn:
            row = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
                "COALESCE(SUM(reasoning_tokens),0), "
                "COALESCE(SUM(input_cost_usd),0.0), COALESCE(SUM(output_cost_usd),0.0), "
                "COALESCE(SUM(total_cost_usd),0.0) "
                "FROM llm_calls"
            ).fetchone()
    except Exception:
        return empty
    return {
        "calls": int(row[0] or 0),
        "input_tokens": int(row[1] or 0),
        "output_tokens": int(row[2] or 0),
        "reasoning_tokens": int(row[3] or 0),
        "input_cost_usd": float(row[4] or 0.0),
        "output_cost_usd": float(row[5] or 0.0),
        "total_cost_usd": float(row[6] or 0.0),
    }


def _iter_usage_calls(op_dir: Path):
    db = op_dir / "llm_usage.db"
    if not db.exists():
        return
    try:
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute(
                "SELECT ts, step_type, iteration, attempt, provider, model, "
                "input_tokens, output_tokens, reasoning_tokens, total_cost_usd "
                "FROM llm_calls ORDER BY ts ASC"
            ).fetchall()
    except Exception:
        return
    for r in rows:
        yield {
            "ts": r[0], "step_type": r[1], "iteration": r[2], "attempt": r[3],
            "provider": r[4], "model": r[5],
            "input_tokens": r[6], "output_tokens": r[7],
            "reasoning_tokens": r[8], "total_cost_usd": r[9],
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="export_csv",
        description="Export project analytics CSVs (best perf, iterations, failures, tokens).",
    )
    p.add_argument("project_name", help="Project directory name under kernels/projects/.")
    p.add_argument("--out", required=True, help="Output directory for CSVs.")
    p.add_argument("--n-kernels", type=int, default=None,
                   help="Truncate per-op to the first N profiled nodes for best-performance.")
    p.add_argument("--ops", default=None,
                   help="Comma-separated operator filter (default: all).")
    args = p.parse_args(argv)

    project_dir = _resolve_project(args.project_name)
    if project_dir is None:
        print(f"error: project not found: kernels/projects/{args.project_name}",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    filter_ops = set(s.strip() for s in args.ops.split(",")) if args.ops else None
    optimizer_ops = _discover_optimizer_ops(project_dir, filter_ops)
    generator_ops = _discover_generator_ops(project_dir, filter_ops)

    _export_best_performance(
        project_dir, optimizer_ops, args.n_kernels, out_dir / "best_performance.csv"
    )
    _export_iterations(
        project_dir, optimizer_ops, generator_ops, out_dir / "iterations_to_correct.csv"
    )
    _export_failures(
        project_dir, optimizer_ops, generator_ops, out_dir / "failure_modes.csv"
    )
    _export_token_usage(
        project_dir, optimizer_ops, generator_ops,
        out_dir / "token_usage.csv",
        out_dir / "token_usage_calls.csv",
    )

    print(f"Exported CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
