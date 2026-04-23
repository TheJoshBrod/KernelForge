from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers to seed a fake project tree matching the real on-disk layout.
# ---------------------------------------------------------------------------


def _seed_nodes_db(op_dir: Path, rows: list[dict], edges: list[tuple[int, int]]) -> None:
    """rows: {id,value,median_time_ms,code,improvement_description,attempts_to_correct,phase}"""
    db = op_dir / "nodes.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY,
                visits INTEGER,
                value REAL,
                median_time_ms REAL,
                best_subtree_value REAL,
                code TEXT,
                improvement_description TEXT,
                timestamp REAL,
                attempts_to_correct INTEGER,
                phase TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE edges (
                parent_id INTEGER, child_id INTEGER,
                PRIMARY KEY (parent_id, child_id)
            )
        """)
        for r in rows:
            conn.execute(
                "INSERT INTO nodes (id, visits, value, median_time_ms, best_subtree_value, "
                "code, improvement_description, timestamp, attempts_to_correct, phase) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["id"], 1, r.get("value"), r.get("median_time_ms"),
                    None, r.get("code", ""),
                    r.get("improvement_description", ""), 0.0,
                    r.get("attempts_to_correct"), r.get("phase", "OPTIMIZATION"),
                ),
            )
        for pid, cid in edges:
            conn.execute("INSERT INTO edges VALUES (?, ?)", (pid, cid))


def _seed_llm_usage_db(op_dir: Path, calls: list[dict]) -> None:
    """Each call dict carries provider, model, input_tokens, output_tokens,
    reasoning_tokens, input_cost_usd, output_cost_usd, total_cost_usd, step_type."""
    db = op_dir / "llm_usage.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                job_key TEXT,
                operator TEXT,
                step_type TEXT NOT NULL,
                iteration INTEGER,
                attempt INTEGER,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                input_cost_usd REAL NOT NULL,
                output_cost_usd REAL NOT NULL,
                total_cost_usd REAL NOT NULL
            )
        """)
        for i, c in enumerate(calls):
            conn.execute(
                "INSERT INTO llm_calls (ts, operator, step_type, provider, model, "
                "input_tokens, output_tokens, reasoning_tokens, "
                "input_cost_usd, output_cost_usd, total_cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    float(i), op_dir.name, c["step_type"], c["provider"], c["model"],
                    c["input_tokens"], c["output_tokens"],
                    c.get("reasoning_tokens", 0),
                    c["input_cost_usd"], c["output_cost_usd"], c["total_cost_usd"],
                ),
            )


def _build_project(tmp_path: Path, name: str) -> Path:
    proj = tmp_path / "kernels" / "projects" / name
    (proj / "trees").mkdir(parents=True)
    return proj


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_export(tmp_path, monkeypatch):
    """Build a fake project with two ops and assert all four CSVs."""
    # Redirect _repo_root to tmp_path so the exporter picks up our fake tree.
    from src.optimizer import export_csv as mod

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    proj = _build_project(tmp_path, "p1")

    # Operator A: 5 optimizer nodes (root + 4 children, one failed).
    op_a = proj / "trees" / "matmul"
    op_a.mkdir()
    _seed_nodes_db(
        op_a,
        rows=[
            {"id": 0, "value": 10.0, "median_time_ms": 10.0,
             "improvement_description": "root", "attempts_to_correct": 1},
            {"id": 1, "value": 8.0,  "median_time_ms": 8.0,
             "improvement_description": "tile", "attempts_to_correct": 2},
            {"id": 2, "value": None, "median_time_ms": None,
             "improvement_description": "failed tile", "attempts_to_correct": 3},
            {"id": 3, "value": 5.0,  "median_time_ms": 5.0,
             "improvement_description": "vectorize", "attempts_to_correct": 1},
            {"id": 4, "value": 6.0,  "median_time_ms": 6.0,
             "improvement_description": "coalesce", "attempts_to_correct": 1},
        ],
        edges=[(0, 1), (1, 2), (1, 3), (3, 4)],
    )
    # best.json intentionally omitted — force fallback to nodes.db.

    # Garbage dump: 2 compile fails + 1 runtime fail.
    dump = op_a / "garbage_dump"
    dump.mkdir()
    (dump / "kernel_iter0_attempt1.cu").write_text("// bad kernel")
    (dump / "llm_response_iter0_attempt1_compile.txt").write_text(
        "[Compilation Failed] expected ';'"
    )
    (dump / "kernel_iter0_attempt2.cu").write_text("// bad kernel 2")
    (dump / "llm_response_iter0_attempt2_compile.txt").write_text(
        "[Compilation Failed] missing brace"
    )
    (dump / "kernel_iter1_attempt1.cu").write_text("// bad runtime")
    (dump / "llm_response_iter1_attempt1_rt.txt").write_text(
        "[Runtime Error entry_0] CUDA illegal memory"
    )

    # Token usage for matmul.
    _seed_llm_usage_db(op_a, [
        {"step_type": "generate", "provider": "openai", "model": "gpt-5",
         "input_tokens": 100, "output_tokens": 200, "reasoning_tokens": 50,
         "input_cost_usd": 0.001, "output_cost_usd": 0.004, "total_cost_usd": 0.005},
        {"step_type": "optimize", "provider": "openai", "model": "gpt-5",
         "input_tokens": 50, "output_tokens": 75, "reasoning_tokens": 25,
         "input_cost_usd": 0.0005, "output_cost_usd": 0.0015, "total_cost_usd": 0.002},
    ])

    # Operator B: has best.json with n_kernels_cap=3.
    op_b = proj / "trees" / "relu"
    op_b.mkdir()
    _seed_nodes_db(
        op_b,
        rows=[
            {"id": 0, "value": 2.0, "median_time_ms": 2.0,
             "improvement_description": "root", "attempts_to_correct": 1},
            {"id": 1, "value": 1.5, "median_time_ms": 1.5,
             "improvement_description": "v1", "attempts_to_correct": 1},
            {"id": 2, "value": 1.0, "median_time_ms": 1.0,
             "improvement_description": "v2", "attempts_to_correct": 2},
        ],
        edges=[(0, 1), (1, 2)],
    )
    (op_b / "best.json").write_text(json.dumps({
        "op_name": "relu",
        "n_kernels_cap": 3,
        "kernels_sampled": 2,
        "baseline_kernel_id": 0,
        "baseline_time_ms": 2.0,
        "best_kernel_id": 2,
        "best_time_ms": 1.0,
        "best_speedup_vs_baseline": 2.0,
        "best_improvement_description": "v2",
    }))
    _seed_llm_usage_db(op_b, [
        {"step_type": "generate", "provider": "anthropic", "model": "claude-sonnet-4-6",
         "input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 0,
         "input_cost_usd": 0.0001, "output_cost_usd": 0.0004, "total_cost_usd": 0.0005},
    ])

    # Generator side: matmul has a summary.json + failure.json.
    gen = proj / "kernels" / "generated" / "individual_op_kernels" / "matmul"
    (gen / "attempts").mkdir(parents=True)
    (gen / "attempts" / "summary.json").write_text(json.dumps({
        "attempts_to_correct": 3, "success": True,
    }))
    (gen / "attempts" / "failure.json").write_text(json.dumps({
        "stage": "llm_validate",
        "message": "failed",
        "function_name": "matmul",
    }))
    (gen / "attempts" / "log-0.txt").write_text("[Output Mismatch entry_0] max_abs_diff=0.5")

    # Run the exporter.
    out = tmp_path / "out"
    rc = mod.main(["p1", "--out", str(out), "--n-kernels", "3"])
    assert rc == 0

    # best_performance.csv
    best_rows = _read_csv(out / "best_performance.csv")
    assert len(best_rows) == 2
    by_op = {r["operator"]: r for r in best_rows}

    # matmul: fallback to nodes.db, --n-kernels=3 truncates profiled list (5 profiled: ids 0,1,3,4 → only 4 profiled).
    # With n_kernels=3 we take the first 3 profiled: ids 0,1,3 → best is id=3 time=5.0.
    m = by_op["matmul"]
    assert m["source"] == "nodes.db"
    assert m["best_kernel_id"] == "3"
    assert float(m["best_time_ms"]) == 5.0
    assert float(m["baseline_time_ms"]) == 10.0
    assert abs(float(m["best_speedup_vs_baseline"]) - 2.0) < 1e-9
    assert m["n_kernels_cap"] == "3"

    # relu: best.json used (n_kernels_cap matches).
    r = by_op["relu"]
    assert r["source"] == "best.json"
    assert r["best_kernel_id"] == "2"
    assert float(r["best_time_ms"]) == 1.0

    # iterations_to_correct.csv — generator row + optimizer rows for matmul, optimizer rows for relu.
    it_rows = _read_csv(out / "iterations_to_correct.csv")
    gen_rows = [r for r in it_rows if r["phase"] == "generator"]
    opt_rows = [r for r in it_rows if r["phase"] == "optimizer"]
    assert len(gen_rows) == 1 and gen_rows[0]["operator"] == "matmul"
    assert gen_rows[0]["attempts_to_correct"] == "3"
    assert gen_rows[0]["success"] == "True"

    matmul_opt = [r for r in opt_rows if r["operator"] == "matmul"]
    relu_opt = [r for r in opt_rows if r["operator"] == "relu"]
    assert len(matmul_opt) == 5  # all 5 nodes have attempts_to_correct
    assert len(relu_opt) == 3

    # Failed node (id=2) in matmul has success=False; others True.
    failed = [r for r in matmul_opt if r["node_id"] == "2"][0]
    assert failed["success"] == "False"

    # failure_modes.csv — matmul optimizer: 2 compile + 1 runtime; matmul generator: NumericalMismatch from log-0.
    fm_rows = _read_csv(out / "failure_modes.csv")
    opt_matmul = {r["category"]: int(r["count"])
                  for r in fm_rows if r["operator"] == "matmul" and r["phase"] == "optimizer"}
    assert opt_matmul.get("CompilationFailed") == 2
    assert opt_matmul.get("RuntimeError") == 1
    gen_matmul = [r for r in fm_rows if r["operator"] == "matmul" and r["phase"] == "generator"]
    assert len(gen_matmul) == 1
    assert gen_matmul[0]["category"] == "NumericalMismatch"

    # token_usage.csv — totals should equal sum of per-op rows.
    tok_rows = _read_csv(out / "token_usage.csv")
    by_op_tok = {r["operator"]: r for r in tok_rows}
    m_tok = by_op_tok["matmul"]
    assert int(m_tok["input_tokens"]) == 150
    assert int(m_tok["output_tokens"]) == 275
    assert int(m_tok["reasoning_tokens"]) == 75
    assert int(m_tok["total_tokens"]) == 150 + 275 + 75
    assert abs(float(m_tok["total_cost_usd"]) - 0.007) < 1e-9

    total = by_op_tok["TOTAL"]
    assert int(total["input_tokens"]) == 150 + 10
    assert int(total["output_tokens"]) == 275 + 20
    assert int(total["reasoning_tokens"]) == 75 + 0
    assert int(total["total_tokens"]) == int(total["input_tokens"]) + int(total["output_tokens"]) + int(total["reasoning_tokens"])
    assert abs(float(total["total_cost_usd"]) - (0.007 + 0.0005)) < 1e-9

    # token_usage_calls.csv — one row per seeded call (3 total).
    call_rows = _read_csv(out / "token_usage_calls.csv")
    assert len(call_rows) == 3


def test_fallback_no_best_json_no_usage_db(tmp_path, monkeypatch):
    """A pre-rebase project: no best.json, no llm_usage.db. Should not crash."""
    from src.optimizer import export_csv as mod
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    proj = _build_project(tmp_path, "pold")
    op = proj / "trees" / "sigmoid"
    op.mkdir()
    _seed_nodes_db(
        op,
        rows=[
            {"id": 0, "value": 3.0, "median_time_ms": 3.0,
             "improvement_description": "root", "attempts_to_correct": 1},
            {"id": 1, "value": 2.0, "median_time_ms": 2.0,
             "improvement_description": "a", "attempts_to_correct": 1},
        ],
        edges=[(0, 1)],
    )

    out = tmp_path / "out"
    rc = mod.main(["pold", "--out", str(out)])
    assert rc == 0

    tok_rows = _read_csv(out / "token_usage.csv")
    # One per-op zero row + TOTAL row.
    assert len(tok_rows) == 2
    assert int(tok_rows[0]["input_tokens"]) == 0
    assert int(tok_rows[0]["total_tokens"]) == 0

    best_rows = _read_csv(out / "best_performance.csv")
    assert len(best_rows) == 1
    assert best_rows[0]["source"] == "nodes.db"
    assert best_rows[0]["best_kernel_id"] == "1"


def test_project_not_found(tmp_path, monkeypatch, capsys):
    from src.optimizer import export_csv as mod
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    rc = mod.main(["nonexistent", "--out", str(tmp_path / "o")])
    assert rc == 2


def test_ops_filter(tmp_path, monkeypatch):
    from src.optimizer import export_csv as mod
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)

    proj = _build_project(tmp_path, "p")
    for name in ("a", "b"):
        op = proj / "trees" / name
        op.mkdir()
        _seed_nodes_db(
            op,
            rows=[{"id": 0, "value": 1.0, "median_time_ms": 1.0,
                   "improvement_description": "r", "attempts_to_correct": 1}],
            edges=[],
        )

    out = tmp_path / "out"
    rc = mod.main(["p", "--out", str(out), "--ops", "a"])
    assert rc == 0

    best = _read_csv(out / "best_performance.csv")
    assert len(best) == 1 and best[0]["operator"] == "a"
