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

def generate_gpu_optimization_prompt(gpu_info: dict, kernel_code: str, improvement_log: list[str]) -> str:
   """
   Generates a structured prompt for an LLM to optimize a CUDA kernel 
   based on specific GPU hardware architecture and constraints.
   """
   
   # 1. Determine Architecture Family & Specific Advice
   cc = float(gpu_info.get('compute_capability', 0.0))
   arch_name = "Unknown"

   # 1. Determine Architecture Family & Specific Advice
   cc = float(gpu_info.get('compute_capability', 0.0))
   arch_name = "Unknown"
   specific_tips = ""  # <--- NEW VARIABLE

   if cc >= 8.0:
      arch_name = "Ampere / Ada Lovelace / Hopper (SM 8.0+)"
      # Advice for modern cards (RTX 30xx, 40xx, A100, H100)
      specific_tips = (
         "- **Pipeline Memory:** Use `cp.async` to hide global memory latency.\n"
         "- **L2 Cache:** Optimize for L2 residency; avoid thrashing the cache."
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

   # 2. Process the Improvement Log
   history_blocks = []
   best_speedup = 0.0
   best_iter = 0

   if not improvement_log:
      history_section = "> *No previous attempts recorded. Starting from baseline.*"
   else:
      for entry in improvement_log:
            iter_num = entry.get('iteration', '?')
            strategy_text = entry.get('attempted', 'No description provided.')
            
            # Extract metrics
            results = entry.get('results', {})
            mean_time = results.get('mean_time_ms', 0.0)
            speedup_base = entry.get('speedup_vs_baseline', 1.0)
            
            # Determine Outcome Icon
            if speedup_base > best_speedup:
               best_speedup = speedup_base
               best_iter = iter_num
               outcome_header = f"**ITERATION {iter_num}: NEW BEST ({speedup_base:.2f}x Speedup)**"
            elif speedup_base < 1.0:
               outcome_header = f"**ITERATION {iter_num}: REGRESSION ({speedup_base:.2f}x Speedup)**"
            else:
               outcome_header = f"**ITERATION {iter_num}: IMPROVEMENT ({speedup_base:.2f}x Speedup)**"

            # Clean up the multi-line strategy text for indentation
            # We wrap it in a blockquote for visual distinction
            formatted_strategy = "\n> ".join(strategy_text.splitlines())

            block = f"""
{outcome_header}
- **Runtime:** {mean_time:.4f} ms
- **Strategy & Rationale:**
> {formatted_strategy}
---"""
            history_blocks.append(block)   
      history_section = "\n".join(history_blocks)
   
   
   
   # 4. Construct the Final Prompt
   prompt = f"""
### Task: Optimize CUDA Kernel for {gpu_info.get('gpu_name', 'Specific GPU')}

**Target Hardware Context**
I am running this kernel on a **{gpu_info.get('gpu_name', 'GPU')}** with Compute Capability **{gpu_info.get('compute_capability')}** ({arch_name}).

**Hardware Constraints & Limits**
{constraints}

**Optimization History (LEARN FROM THIS)**
Below is the log of previous optimization attempts. Use this to determine what works and what fails.

{history_section}

**Current Best Result:** Iteration {best_iter} with {best_speedup:.2f}x speedup.

**Source Code of best optimization**
```cpp
{kernel_code}
```

**Instructions for the Output**
1. **Review the History:** Look at the table above.
   - Identify which tools/techniques (e.g., shared memory, warp shuffles, unrolling) yielded the "BEST" results.
   - **DO NOT** repeat strategies marked as "Regression".
   
2. **Architecture Strategy:** - You are optimizing for **{arch_name}**.
   - If previous attempts to use specific features (like aggressive unrolling) failed, try a different orthogonal approach (like memory vectorization).
   Below are some specific suggestions for the how to optimize on the current GPU architecture
   {specific_tips}
   
   
3. **Generate Code:** - Provide the fully optimized kernel code inside ```cpp <content> ``` blocks.
   - Ensure the `launch(...)` signature remains unchanged.
 """

   return prompt.strip()
