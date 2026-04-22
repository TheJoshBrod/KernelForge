"""
Context gathering for fusion group member operations.
Gathers IO, existing kernels, and timing data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.optimizer.core.fusion.types import MemberOpContext


def _find_kernel_source(op_dir: Path) -> Path | None:
    """Find kernel source file in an operator directory."""
    preferred = ("kernel.cu", "kernel.py", "kernel.metal")
    for name in preferred:
        candidate = op_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(op_dir.glob("kernel.*")):
        if candidate.is_file():
            return candidate
    return None


def _load_tensor_shapes_from_entry(entry_path: Path) -> dict[str, Any]:
    """Load tensor shapes and dtypes from a .pt entry file."""
    shapes: dict[str, Any] = {}
    try:
        data = torch.load(entry_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            inputs = data.get("inputs", {})
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    shapes[f"input_{key}"] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
            output = data.get("output")
            if isinstance(output, torch.Tensor):
                shapes["output"] = {
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                }
    except Exception:
        pass
    return shapes


def _get_op_timing(benchmarks: dict, bench_op: str) -> tuple[float | None, float | None]:
    """Get pytorch_ms and kernel_ms from benchmarks for an operator."""
    results = benchmarks.get("results", [])
    for row in results:
        if not isinstance(row, dict):
            continue
        if row.get("op") == bench_op:
            pytorch_ms = row.get("pytorch_ms")
            kernel_ms = row.get("kernel_ms")
            try:
                pytorch_ms = float(pytorch_ms) if pytorch_ms else None
            except (TypeError, ValueError):
                pytorch_ms = None
            try:
                kernel_ms = float(kernel_ms) if kernel_ms else None
            except (TypeError, ValueError):
                kernel_ms = None
            return pytorch_ms, kernel_ms
    return None, None


def gather_op_context(
    project_dir: Path,
    node_id: str,
    op_type: str,
    benchmarks: dict,
    dag_nodes: list[dict] | None = None,
) -> MemberOpContext:
    """
    Gather context for a single operation.

    Sources:
    - io/individual_ops/{op_name}/entry_*.pt - tensor shapes, dtypes
    - kernels/generated/individual_op_kernels/{op_name}/kernel.cu - existing kernel
    - benchmarks/op_benchmarks.json - timing data

    Args:
        project_dir: Path to project directory
        node_id: Node ID from DAG (e.g., "conv2d_4")
        op_type: Operation type (e.g., "conv2d")
        benchmarks: Loaded op_benchmarks.json dict
        dag_nodes: Optional list of DAG nodes for bench_op lookup
    """
    # Derive bench_op name (e.g., "torch_nn_functional_conv2d")
    bench_op = f"torch_nn_functional_{op_type}"

    # Try to get bench_op from DAG nodes if available
    if dag_nodes:
        for node in dag_nodes:
            if node.get("id") == node_id:
                bench_op = node.get("bench_op", bench_op)
                break

    # Find IO directory
    io_dir = project_dir / "io" / "individual_ops" / bench_op
    if not io_dir.exists():
        io_dir = None

    # Get tensor shapes from first entry
    tensor_shapes: dict[str, list[int]] = {}
    dtype: str | None = None
    if io_dir:
        entries = sorted(io_dir.glob("entry_*.pt"))
        if entries:
            shape_info = _load_tensor_shapes_from_entry(entries[0])
            for key, info in shape_info.items():
                tensor_shapes[key] = info.get("shape", [])
                if dtype is None and "dtype" in info:
                    dtype = info["dtype"]

    # Find existing kernel
    generated_dir = project_dir / "kernels" / "generated" / "individual_op_kernels" / bench_op
    kernel_path = _find_kernel_source(generated_dir) if generated_dir.exists() else None
    kernel_code = None
    if kernel_path and kernel_path.exists():
        try:
            kernel_code = kernel_path.read_text(encoding="utf-8")
        except Exception:
            pass

    # Get timing from benchmarks
    pytorch_ms, kernel_ms = _get_op_timing(benchmarks, bench_op)

    return MemberOpContext(
        node_id=node_id,
        op_type=op_type,
        bench_op=bench_op,
        io_dir=io_dir,
        existing_kernel_path=kernel_path,
        existing_kernel_code=kernel_code,
        tensor_shapes=tensor_shapes,
        dtype=dtype,
        pytorch_ms=pytorch_ms,
        kernel_ms=kernel_ms,
    )


def compute_baseline_timing(contexts: list[MemberOpContext]) -> float | None:
    """
    Compute baseline timing as sum of individual op timings.
    Prefers kernel_ms over pytorch_ms for each op.
    Returns None if no valid timings.
    """
    total = 0.0
    has_any = False

    for ctx in contexts:
        # Prefer kernel_ms if available, else pytorch_ms
        ms = ctx.kernel_ms if ctx.kernel_ms is not None else ctx.pytorch_ms
        if ms is not None and ms > 0:
            total += ms
            has_any = True

    return total if has_any else None


def load_dag_and_benchmarks(
    project_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    """
    Load DAG nodes, edges, and benchmarks from project.

    Returns:
        (dag_nodes, dag_edges, benchmarks)
    """
    dag_nodes: list[dict] = []
    dag_edges: list[dict] = []
    benchmarks: dict = {}

    # Load DAG
    dag_path = project_dir / "io" / "dag.json"
    if dag_path.exists():
        try:
            dag_data = json.loads(dag_path.read_text(encoding="utf-8"))
            dag_nodes = dag_data.get("nodes", [])
            dag_edges = dag_data.get("edges", [])
        except Exception:
            pass

    # Load benchmarks
    bench_path = project_dir / "benchmarks" / "op_benchmarks.json"
    if bench_path.exists():
        try:
            benchmarks = json.loads(bench_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return dag_nodes, dag_edges, benchmarks
