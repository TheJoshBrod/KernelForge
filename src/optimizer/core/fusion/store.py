"""
SQLite persistence for fusion groups and attempts.
Follows patterns from tree_store.py and catalog.db.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.optimizer.core.fusion.types import (
    AttemptStatus,
    FusionAttempt,
    FusionGenStatus,
    FusionGroup,
    FusionUIStatus,
)


def _db_path(project_dir: Path) -> Path:
    """Return path to fusion.db for a project."""
    return project_dir / "fusion.db"


def ensure_fusion_schema(project_dir: Path) -> Path:
    """Create fusion.db with schema if not exists. Returns db path."""
    db_path = _db_path(project_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fusion_groups (
                id TEXT PRIMARY KEY,
                pattern_name TEXT NOT NULL,
                name TEXT,
                members_json TEXT NOT NULL,
                ui_status TEXT NOT NULL,
                gen_status TEXT,
                score REAL,
                estimated_speedup REAL,
                rationale TEXT,
                color_index INTEGER,
                baseline_ms REAL,
                fused_ms REAL,
                actual_speedup REAL,
                best_kernel_path TEXT,
                llm_model TEXT,
                created_at REAL,
                updated_at REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fusion_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                attempt_num INTEGER NOT NULL,
                status TEXT NOT NULL,
                kernel_path TEXT,
                error_message TEXT,
                fused_ms REAL,
                llm_model TEXT,
                timestamp REAL,
                FOREIGN KEY (group_id) REFERENCES fusion_groups(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_group ON fusion_attempts(group_id)"
        )
        conn.commit()

    return db_path


def sync_from_json(project_dir: Path) -> int:
    """
    Sync fusion_groups.json into fusion.db.
    Only imports groups that don't already exist in DB.
    Returns number of groups imported.
    """
    json_path = project_dir / "fusion_groups.json"
    if not json_path.exists():
        return 0

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    groups = data.get("groups", [])
    if not groups:
        return 0

    db_path = ensure_fusion_schema(project_dir)
    now = time.time()
    imported = 0

    with sqlite3.connect(db_path) as conn:
        for g in groups:
            group_id = g.get("id", "")
            if not group_id:
                continue

            # Check if already exists
            exists = conn.execute(
                "SELECT 1 FROM fusion_groups WHERE id = ?", (group_id,)
            ).fetchone()
            if exists:
                continue

            members = g.get("members", [])
            conn.execute(
                """
                INSERT INTO fusion_groups
                (id, pattern_name, name, members_json, ui_status, gen_status,
                 score, estimated_speedup, rationale, color_index, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    g.get("pattern_name", ""),
                    g.get("name", ""),
                    json.dumps(members),
                    g.get("status", "proposed"),
                    None,  # gen_status starts as None
                    g.get("score", 0.0),
                    g.get("estimated_speedup", 1.0),
                    g.get("rationale", ""),
                    g.get("color_index", 0),
                    now,
                    now,
                ),
            )
            imported += 1

        conn.commit()

    return imported


def update_group_status(
    project_dir: Path, group_id: str, gen_status: str | None = None, **kwargs: Any
) -> None:
    """
    Update generation status and optional fields.

    Supported kwargs:
    - baseline_ms, fused_ms, actual_speedup
    - best_kernel_path, llm_model
    - ui_status
    """
    db_path = ensure_fusion_schema(project_dir)
    now = time.time()

    # Build SET clause dynamically
    updates = ["updated_at = ?"]
    values: list[Any] = [now]

    if gen_status is not None:
        updates.append("gen_status = ?")
        values.append(gen_status)

    for field in [
        "baseline_ms",
        "fused_ms",
        "actual_speedup",
        "best_kernel_path",
        "llm_model",
        "ui_status",
    ]:
        if field in kwargs:
            updates.append(f"{field} = ?")
            val = kwargs[field]
            if isinstance(val, Path):
                val = str(val)
            values.append(val)

    values.append(group_id)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"UPDATE fusion_groups SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        conn.commit()


def record_attempt(project_dir: Path, group_id: str, attempt: dict) -> int:
    """
    Record a fusion attempt. Returns attempt ID.

    Expected dict keys:
    - attempt_num, status, kernel_path, error_message, fused_ms, llm_model
    """
    db_path = ensure_fusion_schema(project_dir)
    now = time.time()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO fusion_attempts
            (group_id, attempt_num, status, kernel_path, error_message, fused_ms, llm_model, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                attempt.get("attempt_num", 0),
                attempt.get("status", ""),
                str(attempt["kernel_path"]) if attempt.get("kernel_path") else None,
                attempt.get("error_message"),
                attempt.get("fused_ms"),
                attempt.get("llm_model"),
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_accepted_groups(project_dir: Path) -> list[FusionGroup]:
    """
    Get all groups with ui_status='accepted' that need generation.
    Returns groups where gen_status is None, 'pending', or 'failed'.
    """
    db_path = _db_path(project_dir)
    if not db_path.exists():
        # Try to sync from JSON first
        sync_from_json(project_dir)
        if not db_path.exists():
            return []

    groups: list[FusionGroup] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM fusion_groups
            WHERE ui_status = 'accepted'
              AND (gen_status IS NULL OR gen_status IN ('pending', 'failed'))
            ORDER BY score DESC
            """
        ).fetchall()

        for row in rows:
            members = json.loads(row["members_json"]) if row["members_json"] else []
            groups.append(
                FusionGroup(
                    id=row["id"],
                    pattern_name=row["pattern_name"],
                    name=row["name"] or "",
                    members=members,
                    ui_status=FusionUIStatus(row["ui_status"]),
                    gen_status=FusionGenStatus(row["gen_status"])
                    if row["gen_status"]
                    else None,
                    score=row["score"] or 0.0,
                    estimated_speedup=row["estimated_speedup"] or 1.0,
                    rationale=row["rationale"] or "",
                    color_index=row["color_index"] or 0,
                    baseline_ms=row["baseline_ms"],
                    fused_ms=row["fused_ms"],
                    actual_speedup=row["actual_speedup"],
                    best_kernel_path=Path(row["best_kernel_path"])
                    if row["best_kernel_path"]
                    else None,
                    llm_model=row["llm_model"],
                )
            )

    return groups


def get_group_by_id(project_dir: Path, group_id: str) -> FusionGroup | None:
    """Get a specific fusion group by ID."""
    db_path = _db_path(project_dir)
    if not db_path.exists():
        sync_from_json(project_dir)
        if not db_path.exists():
            return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM fusion_groups WHERE id = ?", (group_id,)
        ).fetchone()

        if not row:
            return None

        members = json.loads(row["members_json"]) if row["members_json"] else []
        return FusionGroup(
            id=row["id"],
            pattern_name=row["pattern_name"],
            name=row["name"] or "",
            members=members,
            ui_status=FusionUIStatus(row["ui_status"]),
            gen_status=FusionGenStatus(row["gen_status"]) if row["gen_status"] else None,
            score=row["score"] or 0.0,
            estimated_speedup=row["estimated_speedup"] or 1.0,
            rationale=row["rationale"] or "",
            color_index=row["color_index"] or 0,
            baseline_ms=row["baseline_ms"],
            fused_ms=row["fused_ms"],
            actual_speedup=row["actual_speedup"],
            best_kernel_path=Path(row["best_kernel_path"])
            if row["best_kernel_path"]
            else None,
            llm_model=row["llm_model"],
        )


def get_group_attempts(project_dir: Path, group_id: str) -> list[FusionAttempt]:
    """Get all attempts for a group, ordered by attempt_num."""
    db_path = _db_path(project_dir)
    if not db_path.exists():
        return []

    attempts: list[FusionAttempt] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM fusion_attempts
            WHERE group_id = ?
            ORDER BY attempt_num ASC
            """,
            (group_id,),
        ).fetchall()

        for row in rows:
            attempts.append(
                FusionAttempt(
                    id=row["id"],
                    group_id=row["group_id"],
                    attempt_num=row["attempt_num"],
                    status=AttemptStatus(row["status"]),
                    kernel_path=Path(row["kernel_path"]) if row["kernel_path"] else None,
                    error_message=row["error_message"],
                    fused_ms=row["fused_ms"],
                    llm_model=row["llm_model"],
                    timestamp=row["timestamp"],
                )
            )

    return attempts
