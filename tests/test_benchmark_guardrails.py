from __future__ import annotations

import os
import sys
import json
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.benchmarking import benchmark_ops
from src.optimizer.backends.cuda import loader as cuda_loader
from src.optimizer.backends.cuda import verifier as cuda_verifier
from src.optimizer import workflow
from src.optimizer import tree_store


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


def test_workflow_run_streams_child_output(tmp_path: Path, capsys):
    cmd = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('first line', flush=True); "
            "time.sleep(0.1); "
            "print('second line', flush=True)"
        ),
    ]

    rc, detail = workflow._run(cmd, tmp_path, workflow._with_python_bin_on_path())

    stdout = capsys.readouterr().out
    assert rc == 0
    assert "first line" in stdout
    assert "second line" in stdout
    assert "first line" in detail
    assert "second line" in detail


def test_run_benchmark_forwards_mode_and_selection_policy(tmp_path: Path, monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(workflow, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(workflow, "_with_python_bin_on_path", lambda env=None: {})

    def fake_run(cmd, root, env):
        captured["cmd"] = cmd
        captured["root"] = root
        captured["env"] = env
        return 0, ""

    monkeypatch.setattr(workflow, "_run", fake_run)

    args = type(
        "Args",
        (),
        {
            "project": "demo",
            "mode": "deployment",
            "selection_policy": "safe",
        },
    )()

    assert workflow.run_benchmark(args) == 0
    assert captured["cmd"] == [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.benchmark_ops",
        "--project",
        "demo",
        "--mode",
        "deployment",
        "--selection-policy",
        "safe",
    ]


def test_cuda_loader_uses_current_device_capability_for_arch_list(monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cuda_loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_loader.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda_loader.torch.cuda, "get_device_capability", lambda *args, **kwargs: (12, 1))

    cuda_loader._ensure_torch_cuda_arch_list()

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.1"


def test_cuda_loader_compile_timeout_returns_instead_of_hanging(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KFORGE_TEST_CUDA_COMPILE_SLEEP_SECONDS", "2")

    started = time.monotonic()
    success, detail = cuda_loader.compile_code_string_with_timeout(
        code="torch::Tensor launch(torch::Tensor x) { return x; }",
        name="kforge_timeout_test",
        build_dir=str(tmp_path / "build"),
        timeout_seconds=0.5,
    )
    elapsed = time.monotonic() - started

    assert success is False
    assert "[Compilation Timeout]" in detail
    assert elapsed < 1.5


def test_cuda_verifier_surfaces_compile_timeout_without_entering_worker(
    monkeypatch,
    tmp_path: Path,
):
    tmp_dir = tmp_path / "tmp"
    io_dir = tmp_path / "io"
    tmp_dir.mkdir()
    io_dir.mkdir()

    monkeypatch.setattr(
        cuda_verifier.loader,
        "compile_code_string_with_timeout",
        lambda **kwargs: (False, "[Compilation Timeout]\nCUDA extension compilation timed out after 0.5 seconds."),
    )

    success, detail = cuda_verifier.validate_kernel(
        "torch::Tensor launch(torch::Tensor x) { return x; }",
        {"tmp_dir": tmp_dir, "io_dir": io_dir, "entry_files": []},
    )

    assert success is False
    assert "[Compilation Timeout]" in detail


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


def test_load_kernel_benchmark_accepts_schema_v2_micro_rows(tmp_path: Path):
    project_dir = tmp_path / "project"
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "benchmarks": [
                    {
                        "op": "torch_nn_functional_linear",
                        "micro": {
                            "status": "ready",
                            "kernel_ms": 0.45,
                            "backend": "cuda",
                            "entry_latencies_ms": [0.44, 0.46],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert workflow._load_kernel_benchmark(project_dir, "torch_nn_functional_linear") == (
        0.45,
        "cuda",
    )


def test_publish_generated_root_preserves_existing_search_value_and_writes_benchmark_meta(
    tmp_path: Path,
):
    project_dir = tmp_path / "project"
    op_name = "torch_nn_functional_linear"
    generated_op_dir = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / op_name
    )
    generated_op_dir.mkdir(parents=True)
    (generated_op_dir / "kernel.cu").write_text("// kernel", encoding="utf-8")

    tree_op_dir = project_dir / "trees" / op_name
    db_path = tree_op_dir / "nodes.db"
    tree_store._ensure_tree_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO nodes
            (id, visits, value, best_subtree_value, code, improvement_description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                0,
                3,
                1.23,
                0.91,
                f"{op_name}/kernels/kernel_0.cu",
                "Existing root",
                0.0,
            ),
        )
        conn.commit()

    result = tree_store.publish_generated_root(
        project_dir,
        op_name,
        kernel_ms=0.42,
        backend="cuda",
        description="Generated baseline kernel (benchmarked)",
    )

    assert result["ok"] is True

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT value, best_subtree_value FROM nodes WHERE id = 0"
        ).fetchone()
    assert row == (1.23, 0.91)

    meta_path = tree_op_dir / "generated_root.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["search_value_ms"] == 1.23
    assert meta["search_best_subtree_value_ms"] == 0.91
    assert meta["benchmarks"]["micro"]["kernel_ms"] == 0.42
    assert meta["benchmarks"]["micro"]["backend"] == "cuda"
