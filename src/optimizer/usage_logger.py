"""
src/optimizer/usage_logger.py
Per-op LLM usage logger. Writes one row per LLM call into
<op_dir>/llm_usage.db (table llm_calls) with the schema consumed by
src/optimizer/export_csv.py.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    ts REAL NOT NULL,
    step_type TEXT,
    iteration INTEGER,
    attempt INTEGER,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    input_cost_usd REAL DEFAULT 0.0,
    output_cost_usd REAL DEFAULT 0.0,
    total_cost_usd REAL DEFAULT 0.0
)
"""


class LLMUsageLogger:
    """Append-only SQLite logger for LLM token usage, scoped to one op tree.

    Each instance owns a path; writes are short-lived connections so
    multiple worker processes can log into the same DB safely.
    """

    def __init__(self, op_dir: Path):
        self.db_path = Path(op_dir) / "llm_usage.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute(_SCHEMA)
                conn.commit()
        except Exception:
            pass

    def log(
        self,
        *,
        step_type: Optional[str],
        iteration: Optional[int],
        attempt: Optional[int],
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        input_cost_usd: float = 0.0,
        output_cost_usd: float = 0.0,
        total_cost_usd: float = 0.0,
    ) -> None:
        try:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute(
                    "INSERT INTO llm_calls (ts, step_type, iteration, attempt, "
                    "provider, model, input_tokens, output_tokens, reasoning_tokens, "
                    "input_cost_usd, output_cost_usd, total_cost_usd) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        step_type,
                        int(iteration) if iteration is not None else None,
                        int(attempt) if attempt is not None else None,
                        provider,
                        model,
                        int(input_tokens or 0),
                        int(output_tokens or 0),
                        int(reasoning_tokens or 0),
                        float(input_cost_usd or 0.0),
                        float(output_cost_usd or 0.0),
                        float(total_cost_usd or 0.0),
                    ),
                )
                conn.commit()
        except Exception:
            pass
