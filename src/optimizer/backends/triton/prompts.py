"""
Triton Backend Prompts.
Generates system prompt and optimization prompts for LLM-driven Triton kernel optimization.
"""


def get_sys_prompt() -> str:
    return """
SYSTEM PROMPT — Triton Kernel Optimizer
-------------------------------------------------

Your job is to **optimize an existing, validated Triton kernel** for maximum performance on the target GPU architecture.

The kernel you receive has **already been validated for correctness**. Your focus is purely on **performance optimization**.

-----------------------------------------------
 OPTIMIZATION OBJECTIVE
-----------------------------------------------
- **Primary Goal**: Reduce kernel execution time
- **Secondary Goals**: Improve memory throughput, increase occupancy, reduce latency
- **Constraint**: Maintain numerical correctness (output must match reference)

-----------------------------------------------
 CURRENT KERNEL CONTEXT
-----------------------------------------------
You will be provided with:
1. **Current kernel implementation** (kernel.py) - This is CORRECT but potentially slow
2. **Performance metrics** from the last run:
   - Execution time (ms)
3. **Target architecture**: GPU model and compute capability
4. **Benchmark configuration**: Input shapes, dtypes, problem size

-----------------------------------------------
 OUTPUT RULES (CRITICAL)
-----------------------------------------------
1. Output **exactly ONE code block** with the optimized kernel
2. The code block must start with:
```python
# [START kernel.py]
```
   and end with:
```python
# [END kernel.py]
```

3. **Preserve the exact function signature** of `launch()` - parameter order and types must NOT change
4. The optimized code must be a drop-in replacement for the current kernel
5. Include a brief comment at the top explaining your optimization strategy (2-3 lines max)
6. Do NOT delegate the target operation to ANY wrapper; implement the computation directly in the kernel.

-----------------------------------------------
 TRITON PROGRAMMING MODEL
-----------------------------------------------

Triton uses a **block-level programming model** where each program instance processes
a block of elements. Key concepts:

**Core Primitives:**
- `tl.program_id(axis)` — Get the current program's ID along an axis
- `tl.arange(0, BLOCK_SIZE)` — Generate a range of offsets within a block
- `tl.load(ptr + offsets, mask=mask)` — Load data from global memory
- `tl.store(ptr + offsets, value, mask=mask)` — Store data to global memory
- `tl.dot(a, b)` — Matrix multiplication (uses tensor cores)
- `tl.where(cond, a, b)` — Element-wise conditional selection
- `tl.sum(x, axis=0)` — Reduction along an axis
- `tl.max(x, axis=0)` — Max reduction

**Kernel Structure:**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x  # your operation here
    tl.store(output_ptr + offsets, output, mask=mask)

def launch(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

-----------------------------------------------
 OPTIMIZATION STRATEGIES TO CONSIDER
-----------------------------------------------

**Block Size Tuning:**
- Use `tl.constexpr` parameters for block sizes
- Test powers of 2: 64, 128, 256, 512, 1024
- Larger blocks = fewer programs, but more register pressure
- Use `@triton.autotune` with `triton.Config` entries for auto-selection

**Memory Optimizations:**
- Coalesced access: ensure consecutive threads access consecutive memory
- Block pointers (`tl.make_block_ptr`) for 2D tiling patterns
- Use `tl.load` with `eviction_policy='evict_last'` for data to be reused
- Increase `num_stages` for software pipelining of memory loads
- Vectorized loads happen automatically when access is contiguous

**Compute Optimizations:**
- `tl.dot` for matmul-like patterns (leverages tensor cores)
- Fuse elementwise operations before/after reductions
- Use `tl.where` instead of Python branching for data-dependent control flow
- Minimize use of `tl.atomic_add` — prefer local reduction then single atomic

**Parallelization:**
- Multi-axis grids for 2D/3D problems: `tl.program_id(0)`, `tl.program_id(1)`
- `num_warps` tuning: 1-8 warps per program (default 4)
- Persistent kernels for small reductions

**Auto-tuning:**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def _kernel(...):
    ...
```

**L2 Cache Optimization:**
- Swizzle program ordering to improve L2 locality for tiled matmuls
- Group programs that access nearby memory regions

-----------------------------------------------
 COMMON PITFALLS TO AVOID
-----------------------------------------------
- **Using Python loops inside JIT kernels** — Use `tl.arange` and vectorized ops
- **Forgetting masks** — Always mask loads/stores for boundary elements
- **Non-power-of-2 BLOCK_SIZE** — Triton requires power-of-2 block sizes
- **Register pressure** — Large blocks with many live values = register spills
- **Incorrect grid calculation** — Use `triton.cdiv(N, BLOCK_SIZE)` not `N // BLOCK_SIZE`
- **Mixing CPU and GPU code in JIT** — All code inside `@triton.jit` runs on GPU
- **`tl.constexpr` misuse** — Only block dims and compile-time constants should be constexpr

-----------------------------------------------
 EXPECTED OUTPUT FORMAT
-----------------------------------------------

```python
# [START FEEDBACK]
# OPTIMIZATION: [Brief 1-2 line description of what changed]
#     - ...
#     - ...
# RATIONALE: [Why this should improve performance]
#     - ...
# [END FEEDBACK]

# [START kernel.py]
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(...):
    # Your optimized kernel here
    ...

def launch(...):  # EXACT same signature as before
    # Your optimized host code here
    ...

# [END kernel.py]
```

-----------------------------------------------
 CRITICAL REMINDERS
-----------------------------------------------
- The current kernel is **CORRECT** - maintain correctness
- Focus on **performance**, not code style or readability (within reason)
- **Preserve the launch() signature exactly** - parameter order matters
- All imports (`torch`, `triton`, `triton.language as tl`) must be at the top of the file
- The kernel file must be self-contained (no external dependencies beyond torch/triton)
- If a change doesn't help or hurts performance, it will be reverted in the next iteration

-----------------------------------------------
Your optimized output will be saved to a kernel.py file and must follow ALL rules above.
-----------------------------------------------
"""


def generate_gpu_optimization_prompt(gpu_info: dict,
                                     kernel_code: str,
                                     improvement_log: list[str],
                                     ancestor_codes: list[tuple[int, str]] = None,
                                     failed_siblings: list[str] = None) -> str:
    """
    Generates a structured prompt for an LLM to optimize a Triton kernel
    based on specific GPU hardware architecture and constraints.

    Args:
        gpu_info: GPU specifications dictionary
        kernel_code: Current kernel code to optimize
        improvement_log: List of optimization attempt records
        ancestor_codes: List of (iteration_id, code_string) tuples from ancestor nodes
    """

    # 1. Determine Architecture Family & Specific Advice
    cc_raw = str(gpu_info.get('compute_capability', "0.0"))
    cc = 0.0
    try:
        cc = float(cc_raw)
    except (TypeError, ValueError):
        cc_digits = "".join(ch for ch in cc_raw if ch.isdigit() or ch == ".")
        if cc_digits:
            try:
                cc = float(cc_digits)
            except (TypeError, ValueError):
                cc = 0.0
    gpu_name = gpu_info.get('gpu_name', 'GPU')
    is_amd_arch = "gfx" in cc_raw.lower()
    arch_name = "Unknown"
    specific_tips = ""

    if is_amd_arch:
        # AMD ROCm GPU
        arch_name = f"AMD ROCm ({gpu_info.get('compute_capability', 'unknown')})"
        specific_tips = (
            "- **Wavefront Size:** AMD uses 64-wide wavefronts (vs NVIDIA's 32-wide warps).\n"
            "- **`num_warps`:** Adjust for wavefront size — `num_warps=2` = 128 threads on AMD.\n"
            "- **LDS (Shared Memory):** AMD GPUs have different LDS bank layout."
        )
    elif cc >= 8.0:
        arch_name = "Ampere / Ada Lovelace / Hopper (SM 8.0+)"
        specific_tips = (
            "- **Software Pipelining:** Increase `num_stages` (3-5) to overlap loads with compute.\n"
            "- **L2 Cache:** Use program swizzling for better L2 locality in tiled matmuls.\n"
            "- **Tensor Cores:** Use `tl.dot` for 16-bit precision math for 4x-8x throughput.\n"
            "- **Large Blocks:** Modern GPUs can handle larger BLOCK_SIZE (256-1024)."
        )
    elif cc >= 7.0:
        arch_name = "Volta / Turing (SM 7.0 - 7.5)"
        specific_tips = (
            "- **Tensor Cores:** `tl.dot` available for fp16/bf16.\n"
            "- **Warp Primitives:** Triton handles warp-level ops internally, focus on block size tuning."
        )
    elif cc >= 5.0:
        arch_name = "Maxwell / Pascal (SM 5.0 - 6.0)"
        specific_tips = (
            "- **No Tensor Cores:** Focus on memory coalescing and occupancy.\n"
            "- **Shared Memory:** Triton handles shared mem automatically; tune BLOCK_SIZE carefully."
        )
    else:
        arch_name = "Unknown Architecture"
        specific_tips = "- Focus on basic optimizations: block size tuning, memory coalescing."

    constraints = (
        f"- **Max Threads per Block:** {gpu_info.get('max_threads_per_block', 'N/A')}\n"
        f"- **Shared Memory per Block:** {gpu_info.get('shared_mem_per_block_kb', 'N/A')} KB\n"
        f"- **Warp Size:** {gpu_info.get('warp_size', 32)}\n"
        f"- **Number of SMs:** {gpu_info.get('num_sms', 'N/A')}\n"
        f"- **Memory Bandwidth:** {gpu_info.get('peak_memory_bandwidth_gbps', 'N/A')} GB/s"
    )

    # 2. Process Lineage History — last 5 ancestors from parental chain
    history_blocks = []
    best_iter = 0
    best_runtime = float('inf')
    best_speedup = 1.0

    # Pass 1: Find best across full lineage
    if improvement_log:
        for entry in improvement_log:
            rt = entry.get('results', {}).get('mean_time_ms', float('inf'))
            if rt < best_runtime:
                best_runtime = rt
                best_iter = entry.get('iteration', 0)
                best_speedup = entry.get('speedup_vs_baseline', 1.0)
                if not (0 < best_speedup < float('inf')):
                    best_speedup = 1.0

    # Baseline reference from root (first entry — collect_ancestry returns root→leaf)
    baseline_ref = ""
    if improvement_log:
        root_rt = improvement_log[0].get('results', {}).get('mean_time_ms', float('inf'))
        root_id = improvement_log[0].get('iteration', 0)
        if 0 < root_rt < float('inf'):
            baseline_ref = f"> Baseline: {root_rt:.4f} ms (root iteration {root_id})"

    # Pass 2: Slice last 5 (collect_ancestry guarantees root→leaf lineage only)
    window = improvement_log[-5:] if len(improvement_log) > 5 else list(improvement_log)
    pruned = len(improvement_log) > len(window)

    # Best-in-lineage anchor — prepend separately if outside the window
    best_in_window = any(e.get('iteration') == best_iter for e in window)
    best_anchor = ""
    if not best_in_window and best_iter != 0:
        best_entry = next((e for e in improvement_log if e.get('iteration') == best_iter), None)
        if best_entry:
            bt = best_entry.get('results', {}).get('mean_time_ms', 0.0)
            best_anchor = (
                f"### >>> TARGET TO BEAT — ITERATION {best_iter}:"
                f" BEST IN LINEAGE | {best_speedup:.2f}x vs baseline <<<\n"
                f"- **Runtime: {bt:.4f} ms**\n"
                f"- Strategy:\n"
                f"> {best_entry.get('attempted', 'No description.')}\n"
                f"---"
            )

    if not window:
        history_section = "> *No previous attempts recorded.*"
    else:
        for entry in window:
            iter_num = entry.get('iteration', '?')
            strategy_text = entry.get('attempted', 'No description.')
            rt = entry.get('results', {}).get('mean_time_ms', 0.0)
            speedup_parent = entry.get('speedup_vs_parent', 1.0)
            speedup_base = entry.get('speedup_vs_baseline', 1.0)

            # Zero-floor: reject inf/NaN/non-positive values
            if not (0 < speedup_parent < float('inf')):
                speedup_parent = 1.0
            if not (0 < speedup_base < float('inf')):
                speedup_base = 1.0

            if iter_num == best_iter:
                block = (
                    f"\n### >>> ITERATION {iter_num}:"
                    f" BEST IN LINEAGE ({speedup_parent:.2f}x vs parent)"
                    f" | {speedup_base:.2f}x vs baseline <<<\n"
                    f"- **Runtime: {rt:.4f} ms**\n"
                    f"- Strategy:\n"
                    f"> {strategy_text}\n"
                    f"---"
                )
            elif speedup_parent >= 1.0:
                block = (
                    f"\n**ITERATION {iter_num}:"
                    f" STEP FORWARD ({speedup_parent:.2f}x vs parent)"
                    f" | {speedup_base:.2f}x vs baseline**\n"
                    f"- Runtime: {rt:.4f} ms\n"
                    f"- Strategy:\n"
                    f"> {strategy_text}\n"
                    f"---"
                )
            else:
                block = (
                    f"\n**ITERATION {iter_num}:"
                    f" STEP BACK ({speedup_parent:.2f}x vs parent)"
                    f" | {speedup_base:.2f}x vs baseline**\n"
                    f"- Runtime: {rt:.4f} ms\n"
                    f"- Strategy:\n"
                    f"> {strategy_text}\n"
                    f"---"
                )
            history_blocks.append(block)

        history_section = "\n".join(history_blocks)

        prefix_parts = []
        if baseline_ref:
            prefix_parts.append(baseline_ref)
        if pruned:
            prefix_parts.append(f"> *(Showing last {len(window)} of {len(improvement_log)} ancestors in this lineage)*")
        if best_anchor:
            prefix_parts.append(best_anchor)
        if prefix_parts:
            history_section = "\n".join(prefix_parts) + "\n\n" + history_section

    # 3. Failed Approaches
    failed_section = ""
    if failed_siblings:
        failed_lines = "\n".join(f"- {desc}" for desc in failed_siblings)
        failed_section = f"**Previously Failed Approaches (DO NOT repeat these)**\n{failed_lines}\n"

    # 4. Ancestor Codes — show evolution trail up to a 12KB budget
    ancestor_section = ""
    if ancestor_codes and len(ancestor_codes) > 0:
        ancestor_section = "**Code Evolution**\n"
        char_budget = 12000
        chars_used = 0
        blocks = []
        for iter_id, code in ancestor_codes:
            label = "Parent" if (iter_id, code) == ancestor_codes[-1] else f"Ancestor (iteration {iter_id})"
            remaining = char_budget - chars_used
            if remaining <= 0:
                break
            display_code = code if len(code) <= remaining else code[:remaining] + "\n# ... truncated"
            chars_used += len(display_code)
            blocks.append(f"<details><summary>{label} — iteration {iter_id}</summary>\n```python\n{display_code}\n```\n</details>")
        ancestor_section += "\n".join(blocks)

    prompt = f"""
### Task: Optimize Triton Kernel for {gpu_name} ({arch_name})

**Hardware Constraints**
{constraints}

{failed_section}
**Optimization History (Learn from this)**
{history_section}

**Current Best:** Iteration {best_iter} ({best_speedup:.2f}x speedup)

{ancestor_section}

**Source Code to Optimize**
```python
{kernel_code}
```

**Instructions**
1. **Review History:** Avoid repeating strategies marked STEP BACK unless they enabled a subsequent STEP FORWARD.
2. **Architecture Strategy:** {specific_tips}
3. **Generate Code:** Preserve `launch(...)` signature.
"""
    return prompt.strip()


def generate_new_root_prompt(
    operator_spec: dict,
    existing_roots: list[dict],
    profiler_context: dict = None
) -> str:
    """Generate prompt for creating a new independent root Triton kernel.

    Unlike optimization prompts, this creates a kernel from scratch using the
    generator approach, but includes existing roots' code to encourage diversity.

    Args:
        operator_spec: Function specification from generator (params, shapes, etc.)
        existing_roots: List of {id, runtime_ms, code_preview} from get_existing_roots()
        profiler_context: Optional ATen ops and CUDA kernels from profiler

    Returns:
        Prompt string for LLM
    """

    # Format operator specification
    func_name = operator_spec.get('function_name', 'unknown_operator')
    params = operator_spec.get('parameters', [])

    params_section = ""
    for i, param in enumerate(params, 1):
        params_section += f"\n{i}. `{param.get('name', f'arg{i}')}` ({param.get('type', 'auto')})"
        if 'shape' in param:
            params_section += f"\n   - Shape: {param['shape']}"
        if 'description' in param:
            params_section += f"\n   - {param['description']}"

    # Format existing roots section (the key for diversity)
    if existing_roots:
        roots_section = """
---

## ⚠️ EXISTING IMPLEMENTATIONS - Use a DIFFERENT Approach!

The following kernels already exist. Your task is to create a **fundamentally different**
implementation. Consider different:
- Block sizes and grid configurations
- Algorithmic approaches (tiling, persistent kernels, multi-axis grids, etc.)
- Memory access patterns (vectorized, strided, block pointers, etc.)
- Fusion strategies (fuse more/fewer ops into one kernel)

"""
        for root in existing_roots:
            roots_section += f"""
### Root {root['id']}

```python
{root['code_preview']}
```

---
"""
    else:
        roots_section = ""

    # Format profiler context if available
    profiler_section = ""
    if profiler_context:
        if profiler_context.get('aten_ops'):
            profiler_section += "\n**ATen Operations:**\n"
            for op in profiler_context['aten_ops']:
                profiler_section += f"- {op}\n"
        if profiler_context.get('cuda_kernels'):
            profiler_section += "\n**CUDA Kernels Launched:**\n"
            for kernel in profiler_context['cuda_kernels']:
                profiler_section += f"- {kernel}\n"

    prompt = f"""
# New Root Triton Kernel Generation

You are creating a **new independent Triton kernel** for the operator below.
This will be a fresh starting point for optimization, separate from existing implementations.

---

## Operator to Implement: `{func_name}`

Based on {operator_spec.get('num_calls', 'multiple')} tracked call(s):

**Parameters:**
{params_section}

{profiler_section}

{roots_section}

---

## Output Requirements

1. Wrap your kernel in:
```python
# [START kernel.py]
... your code ...
# [END kernel.py]
```

2. Include a `# [START FEEDBACK]` section explaining your approach:
```python
# [START FEEDBACK]
# Brief description of the strategy used in this kernel
# [END FEEDBACK]
```

3. The file must be self-contained with all imports:
```python
import torch
import triton
import triton.language as tl
```

4. Implement the `launch()` function with correct parameter handling
5. Focus on a **unique optimization strategy** different from existing roots

Now generate a complete, working kernel.py file.
"""
    return prompt.strip()
