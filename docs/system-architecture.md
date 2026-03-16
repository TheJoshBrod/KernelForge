# System Architecture

The Kernel Forge system operates in two distinct, sequential phases to produce high-performance custom CUDA kernels for PyTorch models.

## 1. Generator (Correctness Pipeline)
**Goal:** Produce a functionally correct CUDA kernel that exactly matches the behavior of a target PyTorch operator.

This phase acts as an automated "developer loop" that iteratively writes and fixes code until it works.

### Workflow
1.  **Profiling & Interception**: The system hooks into PyTorch's dispatch mechanism to capture real execution data (input tensors, attributes, and expected output tensors) for a specific operator (e.g., `torch.nn.functional.linear`).
2.  **LLM Generation**:
    -   The system prompts an LLM (Claude, Gemini, OpenAI, or Ollama) with the operator signature and input metadata.
    -   The LLM generates a candidate CUDA kernel (`kernel.cu`) and a C++ launch signature.
    -   *Source: `src/generator/generator.py`*
3.  **Verification Loop**:
    -   **JIT Compilation**: The generated code is dynamically compiled using `torch.utils.cpp_extension.load_inline`. Compile-time errors are captured immediately.
    -   **Runtime Validation**: The compiled kernel is executed with the captured input tensors. Its output is compared against the ground-truth PyTorch output using `torch.allclose`.
    -   **Feedback**: usage errors (tracebacks) or numerical mismatches are fed back to the LLM to generate a fix.
    -   *Source: `src/optimizer/backends/cuda/verifier.py` and `src/optimizer/backends/triton/verifier.py`*

**Output:** A base "correct" kernel that is functionally equivalent to the PyTorch implementation but not yet optimized.

---

## 2. Optimizer (Performance Pipeline)
**Goal:** Refine the valid kernel to maximize performance on specific hardware (e.g., NVIDIA GeForce RTX 3090).

This phase treats code optimization as a search problem, utilizing **Monte Carlo Tree Search (MCTS)** to explore the space of possible optimizations.

### Workflow
1.  **Initialization**: A project is created for the specific GPU and operator. The "Correct" kernel from Phase 1 serves as the root node of the optimization tree.
2.  **MCTS Selection**:
    -   The system analyzes the tree of existing kernel versions (`KernelNode`s).
    -   It selects a promising parent node to "expand" based on its performance value and visit count.
    -   *Source: [`src/optimizer/core/mcts.py`](../src/optimizer/core/mcts.py)*
3.  **Iterative Refinement**:
    -   **Generation**: The LLM is given the parent kernel's code and its entire optimization history (lineage of changes). It is prompted to apply a specific speedup strategy (e.g., "tiling," "loop unrolling," "vectorized memory access").
    -   **Profiling**: The new kernel is compiled and benchmarked on the actual hardware to measure `mean_time_ms`.
    -   *Source: `src/optimizer/backends/cuda/profiler.py` and `src/optimizer/backends/triton/profiler.py`*
4.  **Tree Update**:
    -   The new kernel is saved as a child node with its performance metrics (speedup vs. parent).
    -   The results propagate up the tree to influence future selection decisions.

### Data Structure
The optimizer maintains a persistent tree structure on disk:
-   **nodes/**: JSON metadata for each attempt (id, performance, parent ID, improvement description).
-   **attempts/**: The raw `.cu` source code for each variation.

**Output:** A library of optimized kernels tuned for the specific device, significantly faster than the generic baseline.
