"""
CUDA-specific fusion prompt generation.
Generates prompts for LLM to create fused CUDA kernels.
"""
from __future__ import annotations

from typing import Any

from src.optimizer.core.fusion.types import FusionGroup, MemberOpContext


def get_fusion_sys_prompt() -> str:
    """System prompt for CUDA kernel fusion generation."""
    return """
SYSTEM PROMPT — CUDA Kernel Fusion Generator
-------------------------------------------------

Your job is to **generate a FUSED CUDA kernel** that combines multiple sequential operations
into a single kernel launch, eliminating intermediate memory traffic.

-----------------------------------------------
 FUSION OBJECTIVE
-----------------------------------------------
- **Primary Goal**: Eliminate intermediate tensor allocations and global memory round-trips
- **Secondary Goals**: Reduce kernel launch overhead, improve memory locality
- **Constraint**: Output must numerically match sequential execution of original operations

-----------------------------------------------
 CRITICAL: THREAD SYNCHRONIZATION STRATEGY
-----------------------------------------------

When fusing operations, you MUST use a UNIFIED thread block and grid strategy.
DO NOT stitch together kernels with different block sizes.

**Requirements:**

1. **Single Block Size**: Choose ONE block size for ALL fused operations
   - Consider: warp size (32), SM occupancy, shared memory limits
   - Good choices: 128, 256, or 512 threads per block
   - Base on output tensor dimensions and available shared memory

2. **Shared Memory for Intermediate Results**:
   - Store intermediate tensors (between ops) in SHARED MEMORY
   - This eliminates global memory writes between fused operations
   - Pattern: `__shared__ float intermediate[TILE_SIZE];`

3. **Synchronization Points**:
   - Use `__syncthreads()` AFTER writing to shared memory
   - Use `__syncthreads()` BEFORE reading from shared memory
   - Example pattern:
     ```cpp
     // Op 1 output
     float op1_result = compute_op1(...);
     smem[threadIdx.x] = op1_result;
     __syncthreads();

     // Op 2 reads from shared, writes to shared
     float op2_result = compute_op2(smem[threadIdx.x], ...);
     smem[threadIdx.x] = op2_result;
     __syncthreads();

     // Final op reads from shared, writes to global
     output[global_idx] = compute_op3(smem[threadIdx.x]);
     ```

4. **Bounds Checking WITHOUT Early Returns**:
   - All threads must participate in __syncthreads()
   - Use masking for computation, not early returns:
     ```cpp
     bool valid = global_idx < total_elements;
     float val = valid ? input[global_idx] : 0.0f;
     // ... operations ...
     if (valid) output[global_idx] = result;
     ```

-----------------------------------------------
 OUTPUT RULES (CRITICAL)
-----------------------------------------------
1. Output a `// [START FEEDBACK] ... // [END FEEDBACK]` block first, explaining your approach
2. The kernel block must start with `// [START kernel.cu]` and end with `// [END kernel.cu]`
3. Use the EXACT function signature provided in the prompt
4. All intermediate data MUST stay in shared memory or registers (no global memory between ops)
5. Do NOT put optimization comments inside the kernel code — all reasoning goes in FEEDBACK

-----------------------------------------------
 COMMON FUSION PATTERNS
-----------------------------------------------

**Conv + BatchNorm + ReLU (CBR)**:
- Tile conv output in shared memory
- Apply batch_norm normalization in-place in shared mem
- Apply ReLU and write to global output
- Single __syncthreads() between each stage

**Linear + Activation**:
- Tile matrix multiply output in shared memory
- Apply activation in-place
- Write final result to global memory

**Reduction Sequences**:
- Use cooperative reduction in shared memory
- Apply post-reduction ops (softmax, layer_norm) in shared mem

-----------------------------------------------
 SHARED MEMORY SIZING
-----------------------------------------------

Calculate shared memory needs BEFORE writing the kernel:
- Intermediate tensor: elements_per_block * sizeof(float)
- Consider: shared_mem_per_block from GPU specs (usually 48KB-164KB)
- If shared memory insufficient, use tiled approach with multiple passes

-----------------------------------------------
 PITFALLS TO AVOID
-----------------------------------------------
- DO NOT use different block sizes for different operations
- DO NOT early-return threads before __syncthreads() calls
- DO NOT read shared memory before writing thread has synced
- DO NOT exceed shared memory limits (check GPU specs)
- DO NOT assume specific tensor layouts — derive from input shapes
- DO NOT ignore bounds checking for non-uniform tensor sizes
"""


def generate_fusion_prompt(
    gpu_info: dict[str, Any],
    group: FusionGroup,
    member_contexts: list[MemberOpContext],
    previous_error: str | None = None,
) -> str:
    """
    Generate prompt for fused kernel generation.

    Args:
        gpu_info: GPU specs dict from backend.get_device_specs()
        group: Fusion group being generated
        member_contexts: Context for each member operation
        previous_error: Error message from previous attempt (for retry)
    """
    lines = [
        "# Task: Generate Fused CUDA Kernel",
        "",
        f"Generate a SINGLE fused kernel for pattern: **{group.pattern_name}**",
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
        "launches and intermediate tensor allocations, produce ONE kernel.",
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
        ("Warp Size", gpu_info.get("warp_size", 32)),
        ("Max Threads per Block", gpu_info.get("max_threads_per_block", 1024)),
        ("Max Shared Memory per Block", f"{gpu_info.get('max_shared_memory_per_block', 49152)} bytes"),
        ("Total Global Memory", f"{gpu_info.get('total_memory_mb', 0)} MB"),
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

        # Tensor shapes
        if ctx.tensor_shapes:
            lines.append("**Tensor Specs:**")
            for name, shape in ctx.tensor_shapes.items():
                dtype = ctx.dtype or "float32"
                lines.append(f"- {name}: shape={shape}, dtype={dtype}")
            lines.append("")

        # Timing info
        if ctx.pytorch_ms or ctx.kernel_ms:
            lines.append("**Timing:**")
            if ctx.pytorch_ms:
                lines.append(f"- PyTorch: {ctx.pytorch_ms:.4f} ms")
            if ctx.kernel_ms:
                lines.append(f"- Forged Kernel: {ctx.kernel_ms:.4f} ms")
            lines.append("")

        # Existing kernel (if available)
        if ctx.existing_kernel_code:
            lines.extend([
                "**Existing Kernel (reference only - may use different block size):**",
                "",
                "```cpp",
                ctx.existing_kernel_code[:2000] + ("..." if len(ctx.existing_kernel_code) > 2000 else ""),
                "```",
                "",
            ])

    # Output Requirements section
    lines.extend([
        "## Output Requirements",
        "",
        "The fused kernel MUST have this signature:",
        "",
        "```cpp",
        "torch::Tensor launch(",
    ])

    # Build signature from all member inputs
    seen_params: set[str] = set()
    param_lines: list[str] = []

    for op_idx, ctx in enumerate(member_contexts):
        for param_name in ctx.tensor_shapes.keys():
            if param_name.startswith("input_"):
                clean_name = param_name[6:]  # Remove "input_" prefix
            else:
                clean_name = param_name

            # Add op prefix to avoid collisions
            if len(member_contexts) > 1:
                full_name = f"{ctx.op_type}_{clean_name}"
            else:
                full_name = clean_name

            if full_name not in seen_params and param_name != "output":
                seen_params.add(full_name)
                param_lines.append(f"    torch::Tensor {full_name}")

    lines.append(",\n".join(param_lines))
    lines.extend([
        ");",
        "```",
        "",
    ])

    # Synchronization requirements
    lines.extend([
        "## CRITICAL: Thread Synchronization",
        "",
        "1. Use a UNIFIED block size (recommend 256 threads)",
        "2. Store intermediate results in shared memory",
        "3. Use __syncthreads() between operations",
        "4. All threads must participate in sync (no early returns)",
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
            previous_error[:3000],  # Limit error length
            "```",
            "",
            "Please fix this specific issue. Common causes:",
            "- Race condition: missing __syncthreads() between shared memory write/read",
            "- Out-of-bounds: thread indexing exceeds tensor dimensions",
            "- Block size mismatch: different ops assuming different thread counts",
            "- Shared memory overflow: intermediate tensors exceed available shared memory",
            "- Compilation error: syntax issue or missing header",
            "",
        ])

    return "\n".join(lines)
