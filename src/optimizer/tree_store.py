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
                mean_time_ms REAL,
                best_subtree_value REAL,
                code TEXT,
                improvement_description TEXT,
                timestamp REAL
            )
            """
        )
        node_columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)")}
        if "mean_time_ms" not in node_columns:
            conn.execute("ALTER TABLE nodes ADD COLUMN mean_time_ms REAL")
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


def _best_canonical_subtree_ms(
    conn: sqlite3.Connection, fallback_ms: float | None
) -> float | None:
    row = conn.execute(
        """
        SELECT MIN(mean_time_ms)
        FROM nodes
        WHERE mean_time_ms IS NOT NULL AND mean_time_ms > 0
        """
    ).fetchone()
    if row and row[0] is not None:
        try:
            return float(row[0])
        except Exception:
            return fallback_ms
    return fallback_ms


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

    if normalized_ms is not None:
        db_path = tree_op_dir / "nodes.db"
        _ensure_tree_schema(db_path)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT visits, value, mean_time_ms, best_subtree_value, code FROM nodes WHERE id = 0"
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO nodes
                    (id, visits, value, mean_time_ms, best_subtree_value, code, improvement_description, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        0,
                        1,
                        normalized_ms,
                        normalized_ms,
                        normalized_ms,
                        code_rel,
                        description,
                        now_ts,
                    ),
                )
            else:
                existing_visits, existing_value, existing_mean, _, existing_code = row
                visits = max(int(existing_visits or 0), 1)
                value = normalized_ms if normalized_ms is not None else existing_value
                mean_time_ms = (
                    normalized_ms if normalized_ms is not None else existing_mean
                )
                best = _best_canonical_subtree_ms(conn, mean_time_ms)
                code = code_rel if not existing_code else str(existing_code)
                conn.execute(
                    """
                    UPDATE nodes
                    SET visits = ?, value = ?, mean_time_ms = ?, best_subtree_value = ?, code = ?,
                        improvement_description = ?, timestamp = ?
                    WHERE id = 0
                    """,
                    (visits, value, mean_time_ms, best, code, description, now_ts),
                )
            conn.commit()

    meta_path = tree_op_dir / "generated_root.json"
    meta_payload = {
        "op": op_name,
        "kernel_relpath": str(target_kernel.relative_to(project_dir)),
        "kernel_ms": normalized_ms,
        "backend": str(backend or ""),
        "updated_at": now_ts,
    }
    try:
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {
        "ok": True,
        "tree_op_dir": str(tree_op_dir),
        "kernel_relpath": str(target_kernel.relative_to(project_dir)),
        "kernel_ms": normalized_ms,
        "benchmarked": normalized_ms is not None,
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
            "SELECT visits, value, mean_time_ms, best_subtree_value, code FROM nodes WHERE id = 0"
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO nodes
                (id, visits, value, mean_time_ms, best_subtree_value, code, improvement_description, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    0,
                    1,
                    normalized_ms,
                    normalized_ms,
                    normalized_ms,
                    _default_root_code_rel(project_dir, op_name),
                    description,
                    now_ts,
                ),
            )
        else:
            existing_visits, _, _, _, existing_code = row
            visits = max(int(existing_visits or 0), 1)
            best = _best_canonical_subtree_ms(conn, normalized_ms)
            code_rel = str(existing_code) if existing_code else _default_root_code_rel(project_dir, op_name)
            conn.execute(
                """
                UPDATE nodes
                SET visits = ?, value = ?, mean_time_ms = ?, best_subtree_value = ?, code = ?,
                    improvement_description = ?, timestamp = ?
                WHERE id = 0
                """,
                (visits, normalized_ms, normalized_ms, best, code_rel, description, now_ts),
            )
        conn.commit()

    return {"ok": True, "tree_op_dir": str(tree_op_dir), "kernel_ms": normalized_ms}
