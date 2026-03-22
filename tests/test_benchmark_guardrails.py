from __future__ import annotations

import os
import sys
import json
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.benchmarking import benchmark_ops
from src.optimizer import workflow


def test_discover_captured_op_dirs_ignores_empty_directories(tmp_path: Path):
    io_root = tmp_path / "io"
    valid_dir = io_root / "torch_nn_functional_softmax"
    valid_dir.mkdir(parents=True)
    (valid_dir / "entry_000001.pt").write_text("x", encoding="utf-8")

    empty_dir = io_root / "aten_add"
    empty_dir.mkdir()

    discovered = benchmark_ops._discover_captured_op_dirs(io_root)
    assert discovered == {"torch_nn_functional_softmax": valid_dir}


def test_select_candidate_ops_prefers_captured_ops_over_summary_counts():
    op_dirs = {
        "torch_nn_functional_softmax": Path("/tmp/softmax"),
        "torch_nn_functional_silu": Path("/tmp/silu"),
    }
    op_counts = {
        "torch_nn_functional_softmax": 50,
        "torch_nn_functional_silu": 140,
        "aten__grouped_mm": 80,
    }

    candidate_ops, allowed_existing_ops = benchmark_ops._select_candidate_ops(
        op_dirs,
        op_counts,
        [],
    )

    assert candidate_ops == [
        "torch_nn_functional_silu",
        "torch_nn_functional_softmax",
    ]
    assert allowed_existing_ops == {
        "torch_nn_functional_softmax",
        "torch_nn_functional_silu",
    }


def test_select_candidate_ops_filters_selected_ops_against_captured_ops():
    op_dirs = {
        "torch_nn_functional_softmax": Path("/tmp/softmax"),
        "torch_nn_functional_silu": Path("/tmp/silu"),
    }
    op_counts = {
        "torch_nn_functional_softmax": 50,
        "torch_nn_functional_silu": 140,
        "aten__grouped_mm": 80,
    }

    candidate_ops, allowed_existing_ops = benchmark_ops._select_candidate_ops(
        op_dirs,
        op_counts,
        ["aten__grouped_mm", "torch_nn_functional_silu"],
    )

    assert candidate_ops == ["torch_nn_functional_silu"]
    assert allowed_existing_ops == {
        "torch_nn_functional_softmax",
        "torch_nn_functional_silu",
    }


def test_generated_profile_error_ninja_does_not_fake_kernel_speed():
    errors: list[str] = []

    (
        kernel_status,
        kernel_ms,
        backend,
        kernel_estimated,
        kernel_entry_latencies,
        kernel_benchmarked_entry_files,
    ) = benchmark_ops._apply_generated_profile_result(
        op_name="torch_nn_functional_linear",
        pytorch_ms=0.123,
        kernel_status="missing",
        kernel_ms=None,
        backend="",
        kernel_estimated=False,
        kernel_entry_latencies=[],
        kernel_benchmarked_entry_files=[],
        generated_stats=None,
        generated_status="generated_profile_error_ninja",
        generated_backend="cuda",
        errors=errors,
    )

    assert kernel_status == "generated_profile_error_ninja"
    assert kernel_ms is None
    assert backend == "cuda"
    assert kernel_estimated is False
    assert kernel_entry_latencies == []
    assert kernel_benchmarked_entry_files == []
    assert errors == [
        "torch_nn_functional_linear: generated kernel profiling unavailable (install ninja for direct kernel benchmarking)"
    ]


def test_ensure_local_toolchain_on_path_prepends_python_bin(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")

    benchmark_ops._ensure_local_toolchain_on_path()

    path_parts = os.environ["PATH"].split(":")
    assert path_parts[0] == str(Path(sys.executable).parent)


def test_workflow_with_python_bin_on_path_prepends_python_bin():
    env = workflow._with_python_bin_on_path({"PATH": "/usr/bin"})

    assert env["PATH"].split(":")[0] == str(Path(sys.executable).parent)


def test_workflow_run_does_not_wait_for_grandchild_stdio_to_close(tmp_path: Path):
    cmd = [
        sys.executable,
        "-c",
        (
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)']); "
            "print('parent complete', flush=True)"
        ),
    ]

    started = time.monotonic()
    rc, detail = workflow._run(cmd, tmp_path, workflow._with_python_bin_on_path())
    elapsed = time.monotonic() - started

    assert rc == 0
    assert "parent complete" in detail
    assert elapsed < 1.5


def test_resolve_tree_kernel_source_prefers_real_kernel_file_from_nodes_db(
    tmp_path: Path,
):
    optimized_root = tmp_path / "trees"
    op_dir = optimized_root / "torch_nn_functional_linear"
    kernels_dir = op_dir / "kernels"
    kernels_dir.mkdir(parents=True)
    kernel_path = kernels_dir / "kernel_7.cu"
    kernel_path.write_text("// kernel", encoding="utf-8")

    conn = sqlite3.connect(op_dir / "nodes.db")
    conn.execute(
        """
        CREATE TABLE nodes (
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
        INSERT INTO nodes
        (id, visits, value, best_subtree_value, code, improvement_description, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            7,
            1,
            0.42,
            0.42,
            "torch_nn_functional_linear/kernels/kernel_7.cu",
            "candidate",
            0.0,
        ),
    )
    conn.commit()
    conn.close()

    source_path, backend = benchmark_ops._resolve_tree_kernel_source(
        optimized_root, "torch_nn_functional_linear"
    )

    assert source_path == kernel_path
    assert backend == "cuda"


def test_load_kernel_benchmark_ignores_unverified_ok_rows(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "project"
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "op": "torch_nn_functional_linear",
                        "kernel_status": "ok",
                        "kernel_ms": 0.5,
                        "backend": "cuda",
                        "kernel_entry_latencies_ms": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert workflow._load_kernel_benchmark(project_dir, "torch_nn_functional_linear") == (
        None,
        "",
    )


def test_load_kernel_benchmark_accepts_measured_rows(tmp_path: Path):
    project_dir = tmp_path / "project"
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "op": "torch_nn_functional_linear",
                        "kernel_status": "ok",
                        "kernel_ms": 0.5,
                        "backend": "cuda",
                        "kernel_entry_latencies_ms": [0.49, 0.51],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert workflow._load_kernel_benchmark(project_dir, "torch_nn_functional_linear") == (
        0.5,
        "cuda",
    )
