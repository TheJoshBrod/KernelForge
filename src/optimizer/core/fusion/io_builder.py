"""
Synthetic IO creation for fused kernel validation.
Creates input/output pairs by running ops sequentially through PyTorch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.optimizer.core.fusion.types import FusionGroup, MemberOpContext


# Map op types to PyTorch functions
OP_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "conv2d": F.conv2d,
    "batch_norm": F.batch_norm,
    "relu": F.relu,
    "max_pool2d": F.max_pool2d,
    "avg_pool2d": F.avg_pool2d,
    "adaptive_avg_pool2d": F.adaptive_avg_pool2d,
    "linear": F.linear,
    "gelu": F.gelu,
    "silu": F.silu,
    "leaky_relu": F.leaky_relu,
    "softmax": F.softmax,
    "layer_norm": F.layer_norm,
    "dropout": F.dropout,
}


def _load_entry(entry_path: Path) -> dict[str, Any]:
    """Load a .pt entry file."""
    return torch.load(entry_path, map_location="cpu", weights_only=False)


def _run_op_sequence(
    first_op_inputs: dict[str, torch.Tensor],
    member_contexts: list[MemberOpContext],
    project_dir: Path,
) -> torch.Tensor | None:
    """
    Run a sequence of operations through PyTorch to get expected output.

    This is a simplified implementation - for complex ops, the actual
    entry data should be used to get proper parameters.
    """
    if not member_contexts:
        return None

    # For simple element-wise ops that can chain, we track the tensor flow
    # For complex ops, we need proper parameters from IO entries

    current_tensor = None

    for i, ctx in enumerate(member_contexts):
        op_type = ctx.op_type

        if i == 0:
            # First op: use inputs from entry
            if ctx.io_dir and ctx.io_dir.exists():
                entries = sorted(ctx.io_dir.glob("entry_*.pt"))
                if entries:
                    data = _load_entry(entries[0])
                    inputs = data.get("inputs", {})
                    # For first op, we start with its output to avoid parameter complexity
                    output = data.get("output")
                    if isinstance(output, torch.Tensor):
                        current_tensor = output.clone()
                        continue

            # Fallback if no IO data
            return None

        else:
            # Subsequent ops: apply to current tensor
            if current_tensor is None:
                return None

            if op_type == "relu":
                current_tensor = F.relu(current_tensor)
            elif op_type == "gelu":
                current_tensor = F.gelu(current_tensor)
            elif op_type == "silu":
                current_tensor = F.silu(current_tensor)
            elif op_type == "leaky_relu":
                current_tensor = F.leaky_relu(current_tensor)
            elif op_type == "batch_norm":
                # For batch_norm in fusion, we need the actual parameters
                if ctx.io_dir and ctx.io_dir.exists():
                    entries = sorted(ctx.io_dir.glob("entry_*.pt"))
                    if entries:
                        data = _load_entry(entries[0])
                        # Get batch_norm output directly
                        output = data.get("output")
                        if isinstance(output, torch.Tensor):
                            current_tensor = output.clone()
                            continue
                # Skip if no data
                pass
            else:
                # For other ops, try to get from IO
                if ctx.io_dir and ctx.io_dir.exists():
                    entries = sorted(ctx.io_dir.glob("entry_*.pt"))
                    if entries:
                        data = _load_entry(entries[0])
                        output = data.get("output")
                        if isinstance(output, torch.Tensor):
                            current_tensor = output.clone()
                            continue

    return current_tensor


def build_fusion_io(
    project_dir: Path,
    group: FusionGroup,
    member_contexts: list[MemberOpContext],
) -> Path:
    """
    Create synthetic IO directory for fused kernel validation.

    For each entry in the first member's IO:
    - Input: First member's input tensors
    - Output: Last member's output tensor (computed via PyTorch)

    Stored in: io/fusion_groups/{group_id}/
    """
    fusion_io_dir = project_dir / "io" / "fusion_groups" / group.id
    fusion_io_dir.mkdir(parents=True, exist_ok=True)

    if not member_contexts:
        return fusion_io_dir

    first_ctx = member_contexts[0]
    last_ctx = member_contexts[-1]

    if not first_ctx.io_dir or not first_ctx.io_dir.exists():
        return fusion_io_dir

    # Get entries from first op
    first_entries = sorted(first_ctx.io_dir.glob("entry_*.pt"))
    if not first_entries:
        return fusion_io_dir

    # Build synthetic IO for each entry
    for idx, first_entry in enumerate(first_entries[:10]):  # Limit to 10 entries
        try:
            first_data = _load_entry(first_entry)
            first_inputs = first_data.get("inputs", {})

            # Get last op's output
            last_output = None
            if last_ctx.io_dir and last_ctx.io_dir.exists():
                last_entries = sorted(last_ctx.io_dir.glob("entry_*.pt"))
                if len(last_entries) > idx:
                    last_data = _load_entry(last_entries[idx])
                    last_output = last_data.get("output")

            if last_output is None:
                # Try to compute via op sequence
                last_output = _run_op_sequence(first_inputs, member_contexts, project_dir)

            if last_output is None:
                continue

            # Collect all inputs from all member ops (for fused kernel signature)
            all_inputs: dict[str, torch.Tensor] = {}
            for op_idx, ctx in enumerate(member_contexts):
                if ctx.io_dir and ctx.io_dir.exists():
                    ctx_entries = sorted(ctx.io_dir.glob("entry_*.pt"))
                    if len(ctx_entries) > idx:
                        ctx_data = _load_entry(ctx_entries[idx])
                        ctx_inputs = ctx_data.get("inputs", {})
                        for key, val in ctx_inputs.items():
                            if isinstance(val, torch.Tensor):
                                # Prefix with op index to avoid collisions
                                all_inputs[f"op{op_idx}_{key}"] = val

            # Save synthetic entry
            synthetic_entry = {
                "inputs": all_inputs,
                "output": last_output,
                "fusion_group_id": group.id,
                "members": group.members,
            }

            entry_path = fusion_io_dir / f"entry_{idx:06d}.pt"
            torch.save(synthetic_entry, entry_path)

        except Exception as e:
            print(f"[fusion io_builder] Error building entry {idx}: {e}")
            continue

    return fusion_io_dir


def collect_all_member_inputs(
    member_contexts: list[MemberOpContext],
    entry_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Collect all input tensors from all member operations for a given entry.
    Used to build the fused kernel's input signature.
    """
    all_inputs: dict[str, torch.Tensor] = {}

    for op_idx, ctx in enumerate(member_contexts):
        if not ctx.io_dir or not ctx.io_dir.exists():
            continue

        entries = sorted(ctx.io_dir.glob("entry_*.pt"))
        if len(entries) <= entry_idx:
            continue

        try:
            data = _load_entry(entries[entry_idx])
            inputs = data.get("inputs", {})
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor):
                    all_inputs[f"op{op_idx}_{key}"] = val
        except Exception:
            continue

    return all_inputs


def get_expected_output(
    member_contexts: list[MemberOpContext],
    entry_idx: int = 0,
) -> torch.Tensor | None:
    """Get expected output tensor from last member's IO entry."""
    if not member_contexts:
        return None

    last_ctx = member_contexts[-1]
    if not last_ctx.io_dir or not last_ctx.io_dir.exists():
        return None

    entries = sorted(last_ctx.io_dir.glob("entry_*.pt"))
    if len(entries) <= entry_idx:
        return None

    try:
        data = _load_entry(entries[entry_idx])
        output = data.get("output")
        if isinstance(output, torch.Tensor):
            return output
    except Exception:
        pass

    return None
