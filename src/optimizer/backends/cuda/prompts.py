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
1. Output **exactly ONE code block** with the optimized kernel
2. The code block must start with:
```cpp
// [START kernel.cu]
```
   and end with:
```cpp
// [END kernel.cu]
```

3. **Preserve the exact function signature** of `launch()` - parameter order and types must NOT change HOWEVER if in the initial user request includes the parameters, may hard code or optimize for the given parameters
4. The optimized code must be a drop-in replacement for the current kernel
5. Include a brief comment at the top explaining your optimization strategy (2-3 lines max)

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
                                     ancestor_codes: list[tuple[int, str]] = None) -> str:
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
    constraints = (
        f"- **Max Threads per Block:** {gpu_info.get('max_threads_per_block', 'N/A')}\n"
        f"- **Max Registers per Block:** {gpu_info.get('registers_per_block', 'N/A')} "
        f"(High register usage will limit occupancy)\n"
        f"- **Shared Memory per Block:** {gpu_info.get('shared_mem_per_block_kb', 'N/A')} KB\n"
        f"- **Warp Size:** {gpu_info.get('warp_size', 32)}\n"
        f"- **Memory Bandwidth:** {gpu_info.get('peak_memory_bandwidth_gbps', 'N/A')} GB/s"
    )

# 2. Process the Improvement Log with SLIDING WINDOW PRUNING
    history_blocks = []
    best_speedup = 0.0
    best_iter = 0
    best_runtime = float('inf')

    # Pass 1: Find best
    if improvement_log:
        for entry in improvement_log:
            results = entry.get('results', {})
            mean_time = results.get('mean_time_ms', float('inf'))
            if mean_time < best_runtime:
                best_runtime = mean_time
                best_iter = entry.get('iteration', 0)
                best_speedup = entry.get('speedup_vs_baseline', 1.0)

    # Pass 2: Filter relevant entries (Pruning)
    relevant_indices = set()
    if best_iter != 0: relevant_indices.add(best_iter) # Always keep best
    
    # Keep last 5 attempts
    total_entries = len(improvement_log)
    for i in range(max(0, total_entries - 5), total_entries):
        relevant_indices.add(improvement_log[i].get('iteration'))

    sorted_log = [e for e in improvement_log if e.get('iteration') in relevant_indices]
    
    if not sorted_log:
        history_section = "> *No previous attempts recorded.*"
    else:
        # Build blocks
        for entry in sorted_log:
            iter_num = entry.get('iteration', '?')
            strategy_text = entry.get('attempted', 'No description.')
            results = entry.get('results', {})
            mean_time = results.get('mean_time_ms', 0.0)
            speedup_base = entry.get('speedup_vs_baseline', 1.0)
            
            if iter_num == best_iter:
                label = f"CURRENT BEST ({speedup_base:.2f}x)"
            elif speedup_base < 1.0:
                label = f"REGRESSION ({speedup_base:.2f}x)"
            else:
                label = f"IMPROVEMENT ({speedup_base:.2f}x)"

            block = f"""
**ITERATION {iter_num}: {label}**
- Runtime: {mean_time:.4f} ms
- Strategy:
> {strategy_text}
---"""
            history_blocks.append(block)

        history_section = "\n".join(history_blocks)
        if len(sorted_log) < len(improvement_log):
             history_section = f"> *(History pruned: {len(sorted_log)}/{len(improvement_log)} items)*\n\n" + history_section

    # 3. Ancestor Codes (Limit to Parent Only to save tokens)
    ancestor_section = ""
    if ancestor_codes and len(ancestor_codes) > 0:
        ancestor_section = "**Code Evolution (Immediate Parent)**\n"
        # Only take the last one (Parent)
        last_iter, last_code = ancestor_codes[-1] 
        display_code = last_code if len(last_code) < 6000 else last_code[:6000] + "\n// ... truncated"
        ancestor_section += f"<details><summary>Iteration {last_iter}</summary>\n```cpp\n{display_code}\n```\n</details>"

    prompt = f"""
### Task: Optimize CUDA Kernel for {gpu_info.get('gpu_name', 'GPU')} ({arch_name})

**Hardware Constraints**
{constraints}

**Optimization History (Learn from this)**
{history_section}

**Current Best:** Iteration {best_iter} ({best_speedup:.2f}x speedup)

{ancestor_section}

**Source Code to Optimize**
```cpp
{kernel_code}
```

**Instructions**
1. **Review History:** Avoid strategies marked as Regression.
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
