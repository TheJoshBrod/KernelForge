from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.benchmarking import benchmark_ops


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
