"""
src/llm/usage_db.py
Project-scoped SQLite store for LLM token usage + cost per generation /
optimization / verifier-summary call.

DB path: <proj_dir>/llm_usage.db

WAL mode is enabled so the main process (Path A: GenModel writes) and spawned
verifier subprocesses (Path B: LiteLLM success callback writes) can write
concurrently without `database is locked` errors.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from src.llm.pricing import compute_cost


_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL    NOT NULL,
    job_key          TEXT,
    operator         TEXT,
    step_type        TEXT    NOT NULL,
    iteration        INTEGER,
    attempt          INTEGER,
    provider         TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    input_tokens     INTEGER NOT NULL,
    output_tokens    INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    input_cost_usd   REAL    NOT NULL,
    output_cost_usd  REAL    NOT NULL,
    total_cost_usd   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_operator ON llm_calls(operator);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts       ON llm_calls(ts);
"""


def _db_path(proj_dir: Path | str) -> Path:
    return Path(proj_dir) / "llm_usage.db"


def project_usage_dir_from_op_dir(op_proj_dir: Path | str | None) -> Path | None:
    """Return the project root that owns usage telemetry for an operator dir.

    Optimization operator directories live under `<project>/trees/<operator>`.
    The UI reads usage from `<project>/llm_usage.db`, not `<project>/trees`.
    Older layouts may pass `<project>/<operator>` directly, so fall back to
    the immediate parent when the `trees` directory is not present.
    """
    if op_proj_dir is None:
        return None
    op_path = Path(op_proj_dir)
    parent = op_path.parent
    if parent.name == "trees":
        return parent.parent
    return parent


def _connect(proj_dir: Path | str) -> sqlite3.Connection:
    path = _db_path(proj_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(_SCHEMA)
    return conn


def log_llm_call(
    proj_dir: Path | str,
    usage: dict[str, Any],
    *,
    step_type: str,
    job_key: str | None = None,
    operator: str | None = None,
    iteration: int | None = None,
    attempt: int | None = None,
) -> None:
    """Insert one row describing a single LLM call.

    `usage` must have at minimum: provider, model, input_tokens, output_tokens.
    Optional: reasoning_tokens (default 0).
    Failures are swallowed — telemetry must never break the main pipeline.
    """
    if not usage:
        return
    try:
        provider = str(usage.get("provider", "")).strip()
        model = str(usage.get("model", "")).strip()
        if not provider or not model:
            return
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
        in_cost, out_cost, total_cost = compute_cost(model, input_tokens, output_tokens)

        conn = _connect(proj_dir)
        try:
            conn.execute(
                """
                INSERT INTO llm_calls (
                    ts, job_key, operator, step_type, iteration, attempt,
                    provider, model,
                    input_tokens, output_tokens, reasoning_tokens,
                    input_cost_usd, output_cost_usd, total_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), job_key, operator, step_type, iteration, attempt,
                    provider, model,
                    input_tokens, output_tokens, reasoning_tokens,
                    in_cost, out_cost, total_cost,
                ),
            )
        finally:
            conn.close()
    except Exception:
        return


def _aggregate(conn: sqlite3.Connection, where: str, params: tuple) -> dict[str, Any]:
    row = conn.execute(
        f"""
        SELECT
            COALESCE(SUM(input_tokens), 0),
            COALESCE(SUM(output_tokens), 0),
            COALESCE(SUM(reasoning_tokens), 0),
            COALESCE(SUM(total_cost_usd), 0.0),
            COUNT(*)
        FROM llm_calls
        {where}
        """,
        params,
    ).fetchone()
    return {
        "input_tokens": int(row[0] or 0),
        "output_tokens": int(row[1] or 0),
        "reasoning_tokens": int(row[2] or 0),
        "total_cost_usd": float(row[3] or 0.0),
        "calls": int(row[4] or 0),
    }


def get_project_totals(proj_dir: Path | str) -> dict[str, Any]:
    if not _db_path(proj_dir).exists():
        return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_cost_usd": 0.0, "calls": 0}
    conn = _connect(proj_dir)
    try:
        return _aggregate(conn, "", ())
    finally:
        conn.close()


def get_operator_totals(proj_dir: Path | str) -> list[dict[str, Any]]:
    """Per-operator aggregates. Returns list sorted by total_cost descending."""
    if not _db_path(proj_dir).exists():
        return []
    conn = _connect(proj_dir)
    try:
        rows = conn.execute(
            """
            SELECT
                COALESCE(operator, ''),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(reasoning_tokens), 0),
                COALESCE(SUM(total_cost_usd), 0.0),
                COUNT(*)
            FROM llm_calls
            GROUP BY operator
            ORDER BY SUM(total_cost_usd) DESC
            """
        ).fetchall()
        return [
            {
                "operator": r[0],
                "input_tokens": int(r[1] or 0),
                "output_tokens": int(r[2] or 0),
                "reasoning_tokens": int(r[3] or 0),
                "total_cost_usd": float(r[4] or 0.0),
                "calls": int(r[5] or 0),
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_recent_calls(proj_dir: Path | str, limit: int = 50) -> list[dict[str, Any]]:
    if not _db_path(proj_dir).exists():
        return []
    conn = _connect(proj_dir)
    try:
        rows = conn.execute(
            """
            SELECT ts, operator, step_type, iteration, attempt,
                   provider, model,
                   input_tokens, output_tokens, reasoning_tokens, total_cost_usd
            FROM llm_calls
            ORDER BY ts DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [
            {
                "ts": float(r[0] or 0.0),
                "operator": r[1] or "",
                "step_type": r[2] or "",
                "iteration": r[3],
                "attempt": r[4],
                "provider": r[5] or "",
                "model": r[6] or "",
                "input_tokens": int(r[7] or 0),
                "output_tokens": int(r[8] or 0),
                "reasoning_tokens": int(r[9] or 0),
                "total_cost_usd": float(r[10] or 0.0),
            }
            for r in rows
        ]
    finally:
        conn.close()
