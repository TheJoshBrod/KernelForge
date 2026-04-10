from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path


def _ensure_tree_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                visits INTEGER,
                value REAL,
                best_subtree_value REAL,
                code TEXT,
                improvement_description TEXT,
                timestamp REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                parent_id INTEGER,
                child_id INTEGER,
                PRIMARY KEY (parent_id, child_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id)")
        conn.commit()


def _find_generated_kernel_source(op_generated_dir: Path) -> Path | None:
    preferred = ("kernel.cu", "kernel.py", "kernel.metal")
    for name in preferred:
        candidate = op_generated_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(op_generated_dir.glob("kernel.*")):
        if candidate.is_file():
            return candidate
    return None


def _normalize_kernel_ms(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        ms = float(value)
    except Exception:
        return None
    if ms <= 0.0:
        return None
    return ms


def _default_root_code_rel(project_dir: Path, op_name: str) -> str:
    kernels_dir = project_dir / "trees" / op_name / "kernels"
    if kernels_dir.exists():
        candidates = sorted(kernels_dir.glob("kernel_0.*"))
        if not candidates:
            candidates = sorted(kernels_dir.glob("kernel_*.*"))
        if candidates:
            return f"{op_name}/kernels/{candidates[0].name}"
    return f"{op_name}/kernels/kernel_0.cu"


def _load_generated_root_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _root_snapshot(project_dir: Path, op_name: str) -> dict[str, float | str | None]:
    tree_op_dir = project_dir / "trees" / op_name
    db_path = tree_op_dir / "nodes.db"
    if not db_path.exists():
        return {
            "search_value_ms": None,
            "search_best_subtree_value_ms": None,
            "root_code_relpath": None,
        }

    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value, best_subtree_value, code FROM nodes WHERE id = 0"
            ).fetchone()
    except Exception:
        row = None

    if row is None:
        return {
            "search_value_ms": None,
            "search_best_subtree_value_ms": None,
            "root_code_relpath": None,
        }

    value, best_subtree_value, code = row
    return {
        "search_value_ms": float(value) if value is not None else None,
        "search_best_subtree_value_ms": (
            float(best_subtree_value) if best_subtree_value is not None else None
        ),
        "root_code_relpath": str(code) if code else None,
    }


def _write_generated_root_meta(
    project_dir: Path,
    op_name: str,
    *,
    kernel_relpath: str | None = None,
    backend: str = "",
    micro_kernel_ms: float | None = None,
    deployment_kernel_ms: float | None = None,
    benchmark_source: str = "",
    updated_at: float | None = None,
) -> dict:
    tree_op_dir = project_dir / "trees" / op_name
    tree_op_dir.mkdir(parents=True, exist_ok=True)
    meta_path = tree_op_dir / "generated_root.json"
    payload = _load_generated_root_meta(meta_path)
    now_ts = float(updated_at if updated_at is not None else time.time())

    if kernel_relpath:
        payload["kernel_relpath"] = str(kernel_relpath)
    if backend:
        payload["backend"] = str(backend)

    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, dict):
        benchmarks = {}

    if micro_kernel_ms is not None or benchmark_source:
        micro_payload = benchmarks.get("micro")
        if not isinstance(micro_payload, dict):
            micro_payload = {}
        if micro_kernel_ms is not None:
            micro_payload["kernel_ms"] = float(micro_kernel_ms)
        if benchmark_source:
            micro_payload["source"] = str(benchmark_source)
        if backend:
            micro_payload["backend"] = str(backend)
        micro_payload["updated_at"] = now_ts
        benchmarks["micro"] = micro_payload

    if deployment_kernel_ms is not None or benchmark_source:
        deployment_payload = benchmarks.get("deployment")
        if not isinstance(deployment_payload, dict):
            deployment_payload = {}
        if deployment_kernel_ms is not None:
            deployment_payload["kernel_ms"] = float(deployment_kernel_ms)
        if benchmark_source:
            deployment_payload["source"] = str(benchmark_source)
        if backend:
            deployment_payload["backend"] = str(backend)
        deployment_payload["updated_at"] = now_ts
        benchmarks["deployment"] = deployment_payload

    if benchmarks:
        payload["benchmarks"] = benchmarks

    payload["updated_at"] = now_ts
    payload.update(_root_snapshot(project_dir, op_name))

    try:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    return payload


def publish_generated_root(
    project_dir: Path,
    op_name: str,
    kernel_ms: float | None = None,
    backend: str = "",
    description: str = "Generated baseline kernel",
) -> dict:
    generated_op_dir = (
        project_dir / "kernels" / "generated" / "individual_op_kernels" / op_name
    )
    if not generated_op_dir.exists():
        return {"ok": False, "reason": "missing_generated_dir"}

    kernel_src = _find_generated_kernel_source(generated_op_dir)
    if kernel_src is None:
        return {"ok": False, "reason": "missing_kernel_source"}

    tree_op_dir = project_dir / "trees" / op_name
    kernels_dir = tree_op_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    target_kernel = kernels_dir / f"kernel_0{kernel_src.suffix}"
    shutil.copy2(kernel_src, target_kernel)

    normalized_ms = _normalize_kernel_ms(kernel_ms)
    code_rel = f"{op_name}/kernels/{target_kernel.name}"
    now_ts = time.time()

    db_path = tree_op_dir / "nodes.db"
    _ensure_tree_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT visits, value, best_subtree_value, code FROM nodes WHERE id = 0"
        ).fetchone()

        if row is None:
            if normalized_ms is not None:
                conn.execute(
                    """
                    INSERT INTO nodes
                    (id, visits, value, best_subtree_value, code, improvement_description, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        0,
                        1,
                        normalized_ms,
                        normalized_ms,
                        code_rel,
                        description,
                        now_ts,
                    ),
                )
        else:
            existing_visits, existing_value, existing_best, existing_code = row
            visits = max(int(existing_visits or 0), 1)
            value = existing_value if existing_value is not None else normalized_ms
            if existing_best is not None:
                best = existing_best
            else:
                best = value
            code = str(existing_code) if existing_code else code_rel
            conn.execute(
                """
                UPDATE nodes
                SET visits = ?, value = ?, best_subtree_value = ?, code = ?,
                    improvement_description = ?, timestamp = ?
                WHERE id = 0
                """,
                (visits, value, best, code, description, now_ts),
            )
        conn.commit()

    _write_generated_root_meta(
        project_dir,
        op_name,
        kernel_relpath=str(target_kernel.relative_to(project_dir)),
        backend=str(backend or ""),
        micro_kernel_ms=normalized_ms,
        benchmark_source=(
            "publish_generated_root_benchmarked" if normalized_ms is not None else "publish_generated_root"
        ),
        updated_at=now_ts,
    )

    return {
        "ok": True,
        "tree_op_dir": str(tree_op_dir),
        "kernel_relpath": str(target_kernel.relative_to(project_dir)),
        "kernel_ms": normalized_ms,
        "benchmarked": normalized_ms is not None,
    }


def write_root_benchmark_metadata(
    project_dir: Path,
    op_name: str,
    *,
    micro_kernel_ms: float | None = None,
    deployment_kernel_ms: float | None = None,
    backend: str = "",
    benchmark_source: str = "benchmark_refresh",
) -> dict:
    normalized_micro = _normalize_kernel_ms(micro_kernel_ms)
    normalized_deployment = _normalize_kernel_ms(deployment_kernel_ms)
    meta = _write_generated_root_meta(
        project_dir,
        op_name,
        backend=backend,
        micro_kernel_ms=normalized_micro,
        deployment_kernel_ms=normalized_deployment,
        benchmark_source=benchmark_source,
    )
    return {
        "ok": True,
        "tree_op_dir": str(project_dir / "trees" / op_name),
        "micro_kernel_ms": normalized_micro,
        "deployment_kernel_ms": normalized_deployment,
        "meta": meta,
    }


def update_root_value(
    project_dir: Path,
    op_name: str,
    kernel_ms: float | None,
    description: str = "Benchmarked kernel",
) -> dict:
    normalized_ms = _normalize_kernel_ms(kernel_ms)
    if normalized_ms is None:
        return {"ok": False, "reason": "invalid_kernel_ms"}

    tree_op_dir = project_dir / "trees" / op_name
    db_path = tree_op_dir / "nodes.db"
    _ensure_tree_schema(db_path)
    now_ts = time.time()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT visits, value, best_subtree_value, code FROM nodes WHERE id = 0"
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO nodes
                (id, visits, value, best_subtree_value, code, improvement_description, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    0,
                    1,
                    normalized_ms,
                    normalized_ms,
                    _default_root_code_rel(project_dir, op_name),
                    description,
                    now_ts,
                ),
            )
        else:
            existing_visits, _, existing_best, existing_code = row
            visits = max(int(existing_visits or 0), 1)
            if existing_best is None:
                best = normalized_ms
            else:
                best = min(float(existing_best), normalized_ms)
            code_rel = str(existing_code) if existing_code else _default_root_code_rel(project_dir, op_name)
            conn.execute(
                """
                UPDATE nodes
                SET visits = ?, value = ?, best_subtree_value = ?, code = ?,
                    improvement_description = ?, timestamp = ?
                WHERE id = 0
                """,
                (visits, normalized_ms, best, code_rel, description, now_ts),
            )
        conn.commit()

    return {"ok": True, "tree_op_dir": str(tree_op_dir), "kernel_ms": normalized_ms}
