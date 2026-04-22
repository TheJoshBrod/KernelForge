"""
Triton-specific fusion prompt generation.
Generates prompts for LLM to create fused Triton kernels.
"""
from __future__ import annotations

from typing import Any

from src.optimizer.core.fusion.types import FusionGroup, MemberOpContext


def get_fusion_sys_prompt() -> str:
    """System prompt for Triton kernel fusion generation."""
    return """
SYSTEM PROMPT — Triton Kernel Fusion Generator
-------------------------------------------------

Your job is to **generate a FUSED Triton kernel** that combines multiple sequential operations
into a single kernel launch, eliminating intermediate memory traffic.

-----------------------------------------------
 FUSION OBJECTIVE
-----------------------------------------------
- **Primary Goal**: Eliminate intermediate tensor allocations and global memory round-trips
- **Secondary Goals**: Reduce kernel launch overhead, improve memory locality
- **Constraint**: Output must numerically match sequential execution of original operations

-----------------------------------------------
 TRITON KERNEL STRUCTURE
-----------------------------------------------

Triton kernels use a block-based programming model:

```python
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    input_ptr, output_ptr,
    # ... other inputs ...
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load, compute, store
    x = tl.load(input_ptr + offsets, mask=mask)
    # ... fused operations ...
    tl.store(output_ptr + offsets, result, mask=mask)
```

-----------------------------------------------
 CRITICAL: INTERMEDIATE DATA HANDLING
-----------------------------------------------

For fused operations, keep intermediate data in registers:

```python
# Op 1 output stays in registers
op1_result = compute_op1(x)

# Op 2 uses register data directly
op2_result = compute_op2(op1_result)

# Final write to global memory
tl.store(output_ptr + offsets, op2_result, mask=mask)
```

DO NOT write intermediate results to global memory!

-----------------------------------------------
 OUTPUT RULES (CRITICAL)
-----------------------------------------------
1. Output a `# [START FEEDBACK] ... # [END FEEDBACK]` block first
2. The kernel block must start with `# [START kernel.py]` and end with `# [END kernel.py]`
3. Include the complete Python file with imports, kernel, and launch wrapper
4. Use triton.autotune for optimal block sizes
5. The launch() function must match the required signature

-----------------------------------------------
 LAUNCH WRAPPER TEMPLATE
-----------------------------------------------

```python
def launch(input: torch.Tensor, ...) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_kernel[grid](
        input, output,
        # ... other args ...
        n_elements,
        BLOCK_SIZE=1024
    )
    return output
```

-----------------------------------------------
 COMMON TRITON FUSION PATTERNS
-----------------------------------------------

**Element-wise Chains (relu, gelu, silu)**:
- Load once, apply all ops, store once
- Keep everything in registers

**Linear + Activation**:
- Fuse matmul output with activation
- Use tl.dot for the matrix multiply

**Normalization + Activation**:
- Compute mean/var in shared memory
- Apply normalization and activation in registers

-----------------------------------------------
 PITFALLS TO AVOID
-----------------------------------------------
- DO NOT use global memory for intermediate results
- DO NOT forget the mask for bounds checking
- DO NOT hardcode tensor sizes - use runtime parameters
- DO NOT forget to handle edge cases (last block)
"""


def generate_fusion_prompt(
    gpu_info: dict[str, Any],
    group: FusionGroup,
    member_contexts: list[MemberOpContext],
    previous_error: str | None = None,
) -> str:
    """
    Generate prompt for fused Triton kernel generation.

    Args:
        gpu_info: GPU specs dict from backend.get_device_specs()
        group: Fusion group being generated
        member_contexts: Context for each member operation
        previous_error: Error message from previous attempt (for retry)
    """
    lines = [
        "# Task: Generate Fused Triton Kernel",
        "",
        f"Generate a SINGLE fused Triton kernel for pattern: **{group.pattern_name}**",
        "",
        "Operations to fuse (in execution order):",
    ]

    for i, ctx in enumerate(member_contexts, 1):
        lines.append(f"{i}. {ctx.op_type} (node: {ctx.node_id})")

    lines.extend([
        "",
        "## Goal",
        "",
        "Eliminate intermediate memory writes between operations. Instead of multiple kernel",
        "launches and intermediate tensor allocations, produce ONE Triton kernel.",
        "",
    ])

    # GPU Specs section
    lines.extend([
        "## GPU Specs",
        "",
        "```",
    ])

    gpu_fields = [
        ("GPU Name", gpu_info.get("gpu_name", "Unknown")),
        ("Compute Capability", gpu_info.get("compute_capability", "Unknown")),
        ("SM Count", gpu_info.get("sm_count", "Unknown")),
    ]

    for name, value in gpu_fields:
        lines.append(f"{name}: {value}")

    lines.extend([
        "```",
        "",
    ])

    # Member Operations section
    lines.extend([
        "## Member Operations",
        "",
    ])

    for i, ctx in enumerate(member_contexts, 1):
        lines.extend([
            f"### Operation {i}: {ctx.op_type}",
            "",
        ])

        if ctx.tensor_shapes:
            lines.append("**Tensor Specs:**")
            for name, shape in ctx.tensor_shapes.items():
                dtype = ctx.dtype or "float32"
                lines.append(f"- {name}: shape={shape}, dtype={dtype}")
            lines.append("")

        if ctx.existing_kernel_code:
            lines.extend([
                "**Existing Kernel (reference):**",
                "",
                "```python",
                ctx.existing_kernel_code[:2000] + ("..." if len(ctx.existing_kernel_code) > 2000 else ""),
                "```",
                "",
            ])

    # Output Requirements
    lines.extend([
        "## Output Requirements",
        "",
        "The fused kernel MUST have this launch signature:",
        "",
        "```python",
        "def launch(",
    ])

    # Build signature from all member inputs
    seen_params: set[str] = set()
    param_lines: list[str] = []

    for op_idx, ctx in enumerate(member_contexts):
        for param_name in ctx.tensor_shapes.keys():
            if param_name.startswith("input_"):
                clean_name = param_name[6:]
            else:
                clean_name = param_name

            if len(member_contexts) > 1:
                full_name = f"{ctx.op_type}_{clean_name}"
            else:
                full_name = clean_name

            if full_name not in seen_params and param_name != "output":
                seen_params.add(full_name)
                param_lines.append(f"    {full_name}: torch.Tensor")

    lines.append(",\n".join(param_lines))
    lines.extend([
        ") -> torch.Tensor:",
        "```",
        "",
    ])

    # Error feedback (if retrying)
    if previous_error:
        lines.extend([
            "## Previous Attempt Failed",
            "",
            "Your previous kernel had the following error:",
            "",
            "```",
            previous_error[:3000],
            "```",
            "",
            "Please fix this specific issue.",
            "",
        ])

    return "\n".join(lines)
