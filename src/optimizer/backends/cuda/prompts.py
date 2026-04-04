"""
src/optimizer/components/llm/prompts.py
Generates prompts for LLM systems including "sys_prompt" AND iterative refinement "gpu_optimization" prompt.
"""


def get_sys_prompt() -> str:
    return """
SYSTEM PROMPT — CUDA Kernel Optimizer
-------------------------------------------------

Your job is to **optimize an existing, validated CUDA kernel** for maximum performance on the target GPU architecture.

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
1. **Current kernel implementation** (kernel.cu) - This is CORRECT but potentially slow
2. **Performance metrics** from the last run:
   - Execution time (ms)
   - Memory bandwidth utilization (%)
   - GPU occupancy (%)
   - Register usage per thread
   - Shared memory usage per block
3. **Target architecture**: GPU model and compute capability
4. **Benchmark configuration**: Input shapes, dtypes, problem size

-----------------------------------------------
 OUTPUT RULES (CRITICAL)
-----------------------------------------------
1. Output a `// [START FEEDBACK] ... // [END FEEDBACK]` block first, then the kernel block
2. The kernel block must start with `// [START kernel.cu]` and end with `// [END kernel.cu]`
3. **Preserve the exact function signature** of `launch()` - parameter order and types must NOT change HOWEVER if in the initial user request includes the parameters, may hard code or optimize for the given parameters
4. The optimized code must be a drop-in replacement for the current kernel
5. Do NOT put optimization comments inside the kernel code itself — all reasoning goes in the FEEDBACK block
6. Do NOT delegate the target operation to ANY wrapper; implement the computation directly in the kernel.

-----------------------------------------------------------------------------------------------
STARTING IDEAS FOR OPTIMIZATION STRATEGIES TO CONSIDER (but also be creative AND MAKE YOUR OWN
----------------------------------------------------------------------------------------------

**Memory Optimizations:**
- Coalesced memory access patterns
- Shared memory usage to reduce global memory traffic
- Memory access alignment (128-byte transactions)
- Minimize bank conflicts in shared memory
- Use of texture memory or constant memory where appropriate
- Prefetching and hiding memory latency

**Compute Optimizations:**
- Increase arithmetic intensity (compute-to-memory ratio)
- Loop unrolling
- Instruction-level parallelism
- Reduce divergent branches (warp divergence)
- Use of fast math intrinsics (__fmaf_rn, rsqrtf, etc.)
- Vectorized loads/stores (float4, int4)

**Parallelization Optimizations:**
- Grid-stride loops for better load balancing
- Optimal block dimensions (multiples of warp size)
- Maximize occupancy (balance registers, shared memory, threads)
- Work per thread tuning (too little = overhead, too much = low occupancy)
- Persistent kernels for small problem sizes

**Algorithm-Level Optimizations:**
- Reduction pattern improvements (warp shuffles instead of shared memory)
- Tiling strategies for cache reuse
- Kernel fusion to reduce kernel launch overhead
- Atomic operation minimization or replacement with warp-level primitives

-----------------------------------------------
 ARCHITECTURE-SPECIFIC CONSIDERATIONS
-----------------------------------------------

**For Ampere/Ada (SM 8.0+):**
- Leverage async copy (cp.async) for pipelined memory access
- Use warp-level primitives (__shfl_*, __ballot_sync, __reduce_*)
- Utilize tensor cores if applicable (wmma API)
- Consider L2 cache residency hints

**For Volta/Turing (SM 7.0-7.5):**
- Warp-level primitives available
- Independent thread scheduling (mind sync points)
- Cooperative groups for flexible synchronization

**For Maxwell/Pascal (SM 5.0-6.0):**
- Focus on coalescing and shared memory
- Limited warp-level primitives
- Standard reduction patterns

-----------------------------------------------
 OPTIMIZATION PROCESS GUIDELINES
-----------------------------------------------

1. **Analyze bottleneck** from metrics:
   - Low bandwidth utilization → memory-bound, optimize access patterns
   - Low occupancy → register pressure or shared memory limits
   - High execution time but good metrics → algorithm inefficiency

2. **Make targeted changes**:
   - Apply 1-3 optimization techniques per iteration
   - Avoid premature optimization or over-complicating code
   - Document what you changed and why

3. **Preserve correctness**:
   - Do NOT change the mathematical algorithm unless you're confident
   - Maintain synchronization where needed
   - Be careful with aggressive optimizations (fast math, relaxed atomics)

4. **Common pitfalls to avoid**:
   - Over-optimization that hurts readability without gains
   - Breaking memory coalescing by changing access patterns incorrectly
   - Increasing register pressure that kills occupancy
   - Removing necessary __syncthreads() calls
   - **Using reinterpret_cast without checking alignment** (PyTorch tensors are NOT guaranteed 16-byte aligned!)
   - **Using __launch_bounds__ incorrectly** (causing compilation failures)
   - **Assuming grid size constraints** (always handle arbitrary N)
   - **Complex template metaprogramming** (keep C++ simple and readable)

-----------------------------------------------
 EXPECTED OUTPUT FORMAT
-----------------------------------------------

```cpp
// [START FEEDBACK]
    OPTIMIZATION: [Brief 1-2 line description of what changed]
        - ...
        - ...
    RATIONALE: [Why this should improve performance]
        - ...
// [END FEEDBACK]

// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE ============

// Your optimized kernel(s) here

// ============ HOST CODE ============

torch::Tensor launch(...) {  // EXACT same signature as before
    // Your optimized host code here
}

// [END kernel.cu]
```

-----------------------------------------------
 CRITICAL REMINDERS
-----------------------------------------------
- The current kernel is **CORRECT** - maintain correctness
- Focus on **performance**, not code style or readability (within reason)
- **Measure, don't guess** - apply optimizations based on bottleneck analysis
- **Preserve the launch() signature exactly** - parameter order matters
- Test one optimization at a time when possible
- If a change doesn't help or hurts performance, it will be reverted in the next iteration

-----------------------------------------------
Your optimized output will be saved to a kernel.cu follow and must follow ALL rules above.
-----------------------------------------------
"""


def generate_gpu_optimization_prompt(gpu_info: dict,
                                     kernel_code: str,
                                     improvement_log: list[str],
                                     ancestor_codes: list[tuple[int, str]] = None,
                                     failed_siblings: list[str] = None) -> str:
    """
    Generates a structured prompt for an LLM to optimize a CUDA kernel 
    based on specific GPU hardware architecture and constraints.
    
    Args:
        gpu_info: GPU specifications dictionary
        kernel_code: Current kernel code to optimize
        improvement_log: List of optimization attempt records
        ancestor_codes: List of (iteration_id, code_string) tuples from ancestor nodes
    """

    # 1. Determine Architecture Family & Specific Advice
    cc_raw = str(gpu_info.get('compute_capability', "0.0"))
    try:
        cc = float(cc_raw)
    except (TypeError, ValueError):
        cc_digits = "".join(ch for ch in cc_raw if ch.isdigit() or ch == ".")
        try:
            cc = float(cc_digits) if cc_digits else 0.0
        except (TypeError, ValueError):
            cc = 0.0
    arch_name = "Unknown"
    specific_tips = ""

    if cc >= 8.0:
        arch_name = "Ampere / Ada Lovelace / Hopper (SM 8.0+)"
        # Advice for modern cards (RTX 30xx, 40xx, A100, H100)
        specific_tips = (
            "- **Pipeline Memory:** Use `cp.async` to hide global memory latency.\n"
            "- **L2 Cache:** Optimize for L2 residency; avoid thrashing the cache.\n"
            "- **Tensor Cores:** Leverage `mma.sync` or WMMA instructions for 16-bit precision math to get 4x-8x throughput over CUDA cores.\n"
            "- **Shared Memory Swizzling:** Use XOR-based indexing (swizzling) for shared memory tiles to eliminate bank conflicts in SDPA kernels.\n"
            "- **Sparse Arithmetic:** Utilize Ampere's 2:4 structured sparsity support if weights can be pruned for a 2x math speedup."
        )
    elif cc >= 7.0:
        arch_name = "Volta / Turing (SM 7.0 - 7.5)"
        # Advice for V100, T4, RTX 20xx, GTX 16xx
        specific_tips = (
            "- **Independent Thread Scheduling:** Threads can diverge freely. You MUST use `__syncwarp()` or `__shfl_sync` explicitly.\n"
            "- **Warp Primitives:** Prefer `__shfl_sync` over shared memory for reduction."
        )
    elif cc >= 5.0:
        arch_name = "Maxwell / Pascal (SM 5.0 - 6.0)"
        # Advice for GTX 9xx, GTX 10xx, P100
        specific_tips = (
            "- **Strict Coalescing:** Memory alignment is critical here.\n"
            "- **Shared Memory:** Heavy reuse is required to overcome lower bandwidth."
        )
    else:
        arch_name = "Legacy (Pre-Maxwell)"
        specific_tips = "- Focus on basic memory coalescing."
    bw = gpu_info.get('peak_memory_bandwidth_gbps') or 0.0
    bw_line = f"\n- **Memory Bandwidth:** {bw} GB/s" if bw > 0 else ""
    constraints = (
        f"- **Max Threads per Block:** {gpu_info.get('max_threads_per_block', 'N/A')}\n"
        f"- **Max Registers per Block:** {gpu_info.get('registers_per_block', 'N/A')} "
        f"(High register usage will limit occupancy)\n"
        f"- **Shared Memory per Block:** {gpu_info.get('shared_mem_per_block_kb', 'N/A')} KB\n"
        f"- **Warp Size:** {gpu_info.get('warp_size', 32)}"
        f"{bw_line}"
    )

# 2. Process Lineage History — last 5 ancestors from parental chain
    history_blocks = []
    best_iter = 0
    best_runtime = float('inf')
    best_speedup = 1.0

    # Pass 1: Find best across full lineage
    if improvement_log:
        for entry in improvement_log:
            rt = entry.get('results', {}).get('min_time_ms', float('inf'))
            if rt < best_runtime:
                best_runtime = rt
                best_iter = entry.get('iteration', 0)
                best_speedup = entry.get('speedup_vs_baseline', 1.0)
                if not (0 < best_speedup < float('inf')):
                    best_speedup = 1.0

    # Baseline reference from root (first entry — collect_ancestry returns root→leaf)
    baseline_ref = ""
    if improvement_log:
        root_rt = improvement_log[0].get('results', {}).get('min_time_ms', float('inf'))
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
            bt = best_entry.get('results', {}).get('min_time_ms', 0.0)
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
            rt = entry.get('results', {}).get('min_time_ms', 0.0)
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
            display_code = code if len(code) <= remaining else code[:remaining] + "\n// ... truncated"
            chars_used += len(display_code)
            blocks.append(f"<details><summary>{label} — iteration {iter_id}</summary>\n```cpp\n{display_code}\n```\n</details>")
        ancestor_section += "\n".join(blocks)

    prompt = f"""
### Task: Optimize CUDA Kernel for {gpu_info.get('gpu_name', 'GPU')} ({arch_name})

**Hardware Constraints**
{constraints}

{failed_section}
**Optimization History (Learn from this)**
{history_section}

**Current Best:** Iteration {best_iter} ({best_speedup:.2f}x speedup)

{ancestor_section}

**Source Code to Optimize**
```cpp
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
    """Generate prompt for creating a new independent root kernel.
    
    Unlike optimization prompts, this creates a kernel from scratch using the
    generator approach, but includes existing roots' code to encourage diversity.
    Root kernels are GPU-agnostic - optimization prompts handle GPU-specific tuning.
    
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
- Memory access patterns (coalesced vs. strided, vectorized loads, etc.)
- Thread/block configurations
- Algorithmic approaches (tiling, persistent kernels, warp-level primitives, etc.)
- Data reuse strategies

"""
        for root in existing_roots:
            roots_section += f"""
### Root {root['id']}

```cpp
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
# New Root Kernel Generation

You are creating a **new independent kernel** for the operator below.
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
```cpp
// [START kernel.cu]
... your code ...
// [END kernel.cu]
```

2. Include a `// [START FEEDBACK]` section explaining your approach:
```cpp
// [START FEEDBACK]
// Brief description of the strategy used in this kernel
// [END FEEDBACK]
```

3. Implement the `launch()` function with correct parameter handling
4. Focus on a **unique optimization strategy** different from existing roots

Now generate a complete, working kernel.cu file.
"""
    return prompt.strip()
