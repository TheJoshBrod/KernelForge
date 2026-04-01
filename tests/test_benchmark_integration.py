from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.benchmarking import benchmark_ops
from src.optimizer.benchmarking import integration


def _load_qwen_bench_module():
    module_path = Path(__file__).resolve().parents[1] / "demo" / "qwen35a3b" / "bench_qwen_tps.py"
    spec = importlib.util.spec_from_file_location("test_qwen_bench_qwen_tps", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_launch_args_preserves_noncontiguous_layout_by_default():
    tensor = torch.randn(4, 8).transpose(0, 1)
    stats = integration.empty_adapter_stats()

    call_args = integration.prepare_launch_args(
        [tensor],
        [("input", "torch::Tensor")],
        adapter_stats=stats,
    )

    assert call_args[0] is tensor
    assert call_args[0].is_contiguous() is False
    assert stats["tensor_arg_count"] == 1
    assert stats["noncontiguous_tensor_arg_count"] == 1
    assert stats["preserved_layout_tensor_arg_count"] == 1
    assert stats["contiguous_copy_count"] == 0


def test_prepare_launch_args_counts_forced_contiguous_copies():
    tensor = torch.randn(4, 8).transpose(0, 1)
    stats = integration.empty_adapter_stats()

    call_args = integration.prepare_launch_args(
        [tensor],
        [("input", "torch::Tensor")],
        force_contiguous=True,
        adapter_stats=stats,
    )

    assert call_args[0].is_contiguous()
    assert call_args[0].data_ptr() != tensor.data_ptr()
    assert stats["tensor_arg_count"] == 1
    assert stats["noncontiguous_tensor_arg_count"] == 1
    assert stats["preserved_layout_tensor_arg_count"] == 0
    assert stats["contiguous_copy_count"] == 1


def test_winner_from_measurements_requires_real_candidate_win():
    assert (
        benchmark_ops._winner_from_measurements(
            pytorch_ms=1.0,
            candidate_status="ok",
            candidate_ms=0.5,
        )
        == "optimized"
    )
    assert (
        benchmark_ops._winner_from_measurements(
            pytorch_ms=1.0,
            candidate_status="ok",
            candidate_ms=1.5,
        )
        == "pytorch"
    )
    assert (
        benchmark_ops._winner_from_measurements(
            pytorch_ms=1.0,
            candidate_status="integrated_profile_error",
            candidate_ms=0.5,
        )
        == "pytorch"
    )
    assert (
        benchmark_ops._winner_from_measurements(
            pytorch_ms=1.0,
            candidate_status="ok",
            candidate_ms=0.5,
            correctness_ok=False,
        )
        == "pytorch"
    )


def test_summarize_output_correctness_tracks_strict_vs_loose_tolerances():
    generated = torch.tensor([1.0, 2.0], dtype=torch.float32)
    ground_truth = torch.tensor([1.0075, 2.0], dtype=torch.float32)

    summary = benchmark_ops._summarize_output_correctness(generated, ground_truth)

    assert summary["strict_match"] is False
    assert summary["loose_match"] is True
    assert summary["max_abs_diff"] == pytest.approx(0.0075, rel=1e-4, abs=1e-6)
    assert summary["mean_abs_diff"] > 0.0


def test_safe_deployment_winner_requires_strict_correctness():
    assert (
        benchmark_ops._safe_deployment_winner(
            pytorch_ms=1.0,
            candidate_status="ok",
            candidate_ms=0.8,
            correctness_summary={"strict_pass": True},
        )
        == "optimized"
    )
    assert (
        benchmark_ops._safe_deployment_winner(
            pytorch_ms=1.0,
            candidate_status="ok",
            candidate_ms=0.8,
            correctness_summary={"strict_pass": False},
        )
        == "pytorch"
    )


def test_default_forged_ops_prefers_deployment_winner(tmp_path: Path):
    module = _load_qwen_bench_module()
    project_dir = tmp_path / "project"
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "op": "torch_nn_functional_softmax",
                        "winner": "optimized",
                        "kernel_status": "ok",
                        "deployment_winner": "optimized",
                        "deployment_safe_winner": "optimized",
                        "integrated_kernel_status": "ok",
                        "deployment_correctness": {"strict_pass": True},
                    },
                    {
                        "op": "torch_nn_functional_linear",
                        "winner": "optimized",
                        "kernel_status": "ok",
                        "deployment_winner": "pytorch",
                        "deployment_safe_winner": "pytorch",
                        "integrated_kernel_status": "ok",
                        "deployment_correctness": {"strict_pass": False},
                    },
                    {
                        "op": "torch_nn_functional_embedding",
                        "winner": "optimized",
                        "kernel_status": "ok",
                    },
                    {
                        "op": "torch_nn_functional_pad",
                        "winner": "optimized",
                        "kernel_status": "ok",
                        "deployment_winner": "optimized",
                        "deployment_safe_winner": "pytorch",
                        "integrated_kernel_status": "integrated_profile_error",
                        "deployment_correctness": {"strict_pass": False},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module._default_forged_ops(project_dir) == [
        "torch_nn_functional_softmax",
        "torch_nn_functional_embedding",
    ]


def test_resolve_kernel_source_honors_benchmark_source_preference(tmp_path: Path):
    module = _load_qwen_bench_module()
    project_dir = tmp_path / "project"
    generated_kernel = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / "torch_nn_functional_linear"
        / "kernel.cu"
    )
    generated_kernel.parent.mkdir(parents=True)
    generated_kernel.write_text("// generated", encoding="utf-8")

    tree_kernel = (
        project_dir
        / "trees"
        / "torch_nn_functional_linear"
        / "kernels"
        / "kernel_1.cu"
    )
    tree_kernel.parent.mkdir(parents=True)
    tree_kernel.write_text("// tree", encoding="utf-8")

    nodes_db = project_dir / "trees" / "torch_nn_functional_linear" / "nodes.db"
    import sqlite3

    with sqlite3.connect(nodes_db) as conn:
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
                1,
                1,
                0.1,
                0.1,
                "torch_nn_functional_linear/kernels/kernel_1.cu",
                "candidate",
                0.0,
            ),
        )
        conn.commit()

    assert (
        module._resolve_kernel_source(
            project_dir,
            "torch_nn_functional_linear",
            prefer_tree_best=False,
            preferred_source="optimized_tree",
        )
        == tree_kernel
    )
    assert (
        module._resolve_kernel_source(
            project_dir,
            "torch_nn_functional_linear",
            prefer_tree_best=True,
            preferred_source="generated",
        )
        == generated_kernel
    )
