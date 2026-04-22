# Breaking the Machine Learning Performance Tax: Introducing Kernel Forge

*How Agentic Search and MCTS are replacing manual CUDA tuning to optimize PyTorch models.*

---

## The Kernel You'll Never Write

You've been here before. A model in production, latency just a hair above where it needs to be, profiler open, staring at the same few operators eating 40% of your inference time. You know the performance is there and the hardware is capable you've read the benchmarks. What sits between you and that throughput is a custom GPU kernel, and you know it.

So you start down the path. You open the CUDA documentation. You think about tile sizes: 16, 32, 64? Shared memory usage? Loop unrolling factor? Thread block dimensions? Pipelining stages to overlap compute and memory? Then you realize you're stacking these decisions multiplicatively. For a single matrix multiplication kernel, the combination of tile sizes, pipelining stages, vectorization width, thread block shape, memory access patterns, and instruction scheduling produces a search space somewhere north of 10^12 distinct configurations. Not all of them are valid. Not all of the valid ones are worth trying. But you can't know which are good without benchmarking on actual hardware, and you can't benchmark all of them — not in a week, not in a year, not in a career.

The standard answer is to hire someone who has spent years building intuition about this space. A kernel engineer who knows from experience that for your specific GPU architecture, a tile size of 32 with four pipeline stages and vectorized loads outperforms a tile size of 64 with two stages. A person who has read the architecture docs, written the experiments, and internalized the patterns. That expertise is real and genuinely hard-won. It also takes one to four weeks per kernel, costs a full-time salary, and produces code that is tightly coupled to a specific hardware generation.

Even the well-resourced teams end up doing neither. They accept the performance their framework gives them and move on. The gap between "what PyTorch gives you" and "what your hardware can actually do" becomes a tax, paid in latency, in compute costs, in servers that could be serving twice the traffic if only someone had the time.

Kernel Forge exists to eliminate that tax.

---

## The Landscape of Half-Solutions

To understand why Kernel Forge is structured the way it is, it helps to understand why the obvious approaches don't work.

Static compilers like XLA and TVM represent the most principled attempt at automated optimization. They analyze computation graphs, apply fusion rules, and emit optimized code using pre-defined cost models. For the workloads they were designed around, they work well. The problem is the word "pre-defined." When your model uses an unusual sequence of operations, when your tensor shapes fall outside the expected range, when you're targeting a GPU that postdates the last update to the cost model the fixed heuristics silently fail. They produce code that is *plausible* rather than *optimal*, and there is no mechanism to tell you the difference. The compiler isn't wrong; it's just operating on assumptions that no longer hold.

Triton addressed the authoring problem. Instead of hand-writing CUDA, you express kernels in a Python-like DSL that compiles to efficient GPU code. It genuinely lowered the barrier: a competent Python programmer can write a Triton kernel where previously you needed a CUDA specialist. But Triton gave you a better pen. It didn't write the essay. You still have to decide what tile size to use, which loops to unroll, how many pipeline stages to request. The search space is identical; the writing is just less painful.

The most recent wave of approaches uses LLMs directly. Ask the model to write a kernel, benchmark it, ask the model to improve it, repeat. On simple operations, this sometimes works. On anything complex, three failure modes appear. The first is local minima: after a handful of iterations, the LLM settles into a small neighborhood of similar-looking kernels, suggesting incremental variations of what it already tried. It has no mechanism to backtrack and explore a completely different approach. The second is premature pruning: some approaches keep a beam of promising candidates, but the beam is narrow to contain costs, and genuinely different architectural ideas get eliminated before they have a chance to develop. A design that starts 10% slower but would become 40% faster with three more iterations of refinement never gets those iterations. The third is zero-shot brittleness: asking an LLM to generate a fully-optimized kernel from scratch has sub-20% success rates on real operations, not because LLMs are bad at code, but because they hallucinate hardware constraints. They suggest tile sizes that overflow shared memory, thread counts that exceed hardware limits, or memory access patterns that cause bank conflicts. These subtle violations compile fine but corrupt results or can crash at runtime.

Each of these approaches solves part of the problem. None of them solve the search.

---

## What If the Search Was the Compiler?

The central idea behind Kernel Forge is a reframing. Rather than treating kernel optimization as a code-generation problem provide an LLM the specification, get back an optimized kernel, then treat it as a search problem by giving the LLM the tools to explore that search space intelligently.

Kernel Forge breaks its pipeline into two strict phases. Phase one has one job: produce a correct kernel. Not necesarily a "fast" kernel, just a correct one. Kernel Forge profiles your model at the operator level, capturing real input tensors with real shapes and real dtypes from actual forward passes. It sends those shapes, the operation specification, and the hardware constraints to an LLM, which generates a candidate kernel. If the kernel doesn't compile, the error goes back to the LLM for repair. If it compiles but produces wrong outputs, that goes back too. This loop runs up to eight times before moving on. By the end of phase one, you have a kernel that is provably correct on *your* actual data, but not yet optimized.

Phase two is where the real work begins, driven by Monte Carlo Tree Search (MCTS).

If you've followed modern AI, you've already seen MCTS in action, even if you didn't know it by name. MCTS has powered systems like AlphaGo, enabling superhuman play not by memorizing patterns, but by systematically exploring the consequences of possible moves.

But MCTS isn't limited to games.

It also sits at the core of AlphaDev, DeepMind's system for discovering faster low-level algorithms. Instead of searching over moves on a board, AlphaDev searches over sequences of assembly instructions to construct programs one instruction at a time and evaluating them on real hardware.

The result: novel sorting routines that outperform decades of human-optimized implementations in the LLVM standard library.

While the domain of AlphaGo and AlphaDev are different from each other, the philosophy is the same:

1. define a search space
2. explore it with tree search
3. evaluate outcomes
4. backpropagate improvements

The core insight in both cases is the same: you do not need to evaluate every possible move to make excellent decisions. You build a tree, you evaluate promising branches more deeply, and you update your assessment of entire lines of play based on what you discover.

Kernel Forge applies this exploration model to kernel optimization. Each node in the tree is a kernel variant where proposed improvement are the edges. These improvements can range from changing the tile size, unrolling a loop, switching from row-major to column-major memory access, to adding software pipelining. Promising branches get explored further, while dead ends are abandoned. The tree accumulates the knowledge of every previous experiment, and that knowledge shapes every future one.

To guide node selection, Kernel Forge uses Upper Confidence bounds applied to Trees (UCT).

Because the objective is to minimize runtime, the selection policy is adapted accordingly: nodes with lower observed latency are preferred (exploitation), while under-explored nodes receive an exploration bonus that encourages the search to probe new regions.

In practice, this is equivalent to applying standard UCT to a negated reward (−runtime), but expressed directly in terms of latency. The result is the same balance: aggressively refine fast kernels while still allocating effort to unexplored variants.

---

## Teaching the LLM to Learn from Itself

The MCTS structure is necessary but not sufficient alone.

Prompting an LLM "improve this kernel" will just repeatedly make the same mistakes, propose things that have already been tried, and ignore the context that determines what's worth trying next. Kernel Forge solves this with a prompt that is carefully engineered to make the LLM progressively smarter as the search runs longer and tree structure grows.

Every optimization prompt contains three distinct signals. The first is a pruned log of previous iterations along the current lineage, each labeled as CURRENT BEST, IMPROVEMENT, or REGRESSION with its measured runtime. Rather than shgit cheowing every attempt, the system always includes the best-performing iteration and the five most recent ones — enough to convey the trajectory without burying the signal in noise. The second signal is source code evolution: the actual kernel source of up to three ancestors, rendered under a strict 12-kilobyte character budget. The LLM isn't just told that a tile size change helped — it can read the code that implemented it and see exactly what changed between generations. The third signal is what distinguishes this from simple history injection: failed sibling context. Before generating a proposal for any node, Kernel Forge queries the MCTS tree for failed children of the current parent — attempts already made from this exact point that didn't compile, produced incorrect outputs, or caused a regression — and for failed siblings of the parent, children of the grandparent that tried similar things one level up the tree. Both land in the prompt under the explicit header "Previously Failed Approaches (DO NOT repeat these)." The LLM receives not just what worked along its lineage, but what failed sideways, before writing a single line.

Controlling how broadly that search expands is progressive widening with annealing. Early in the search, each node can spawn many children — the branching factor starts wide, with an exponent of 0.5, meaning a node with 100 visits can grow up to 10 children. This encourages broad exploration: many different optimization strategies get tried, even ones that seem unlikely. As the search matures, the exponent anneals linearly down to 0.3 over 1000 node visits, narrowing the branching factor and shifting toward exploitation of the most promising directions. Stagnation pruning complements this: if a subtree has been visited 10 times without finding any improvement over its own node's performance, it gets skipped during selection. The search doesn't abandon it permanently — it just deprioritizes it in favor of branches that are still producing gains.

There is one detail in the implementation that is worth pausing on because it is genuinely elegant. When a kernel fails to compile or produces incorrect outputs, it still gets saved to the MCTS tree. Its value is recorded as `None`. When the UCT selection algorithm encounters a node with a `None` value, it returns infinity as the score. Since UCT selects the node with the lowest score in this minimization framework, failed nodes are never selected again. No special-case logic, no explicit blacklist, no conditional checks scattered through the codebase. The math handles it. This is the kind of design decision that reveals careful thinking about the system as a whole rather than a collection of features bolted together.

---

## What $2.10 Actually Buys You

Kernel Forge ran on ResNet-50 targeting an NVIDIA GTX 1660 Ti — a consumer GPU from 2019, not a data center accelerator. The benchmark protocol is production-grade: 25 warmup iterations discarded, 100 timed iterations recorded using CUDA events for GPU-side timing, real ResNet-50 tensors at production batch sizes, and numerical correctness verified with `torch.allclose` before any timing is reported.

Batch normalization came in at 1.89x faster than PyTorch eager. This is an operation where PyTorch's generic implementation has to handle arbitrary tensor layouts, arbitrary numbers of features, arbitrary epsilon values — it is built for flexibility, not for the specific shapes in your model. Kernel Forge's generated kernel knows exactly what it's dealing with: your shapes, your layout, your hardware. The gap between generic and specific is 1.89x.

Relu landed at 1.50x. For an elementwise operation that should be memory-bandwidth-bound, that number reflects better memory access patterns, better vectorization, and better use of the hardware's available bandwidth for the specific tensor dimensions in play.

Conv2d at 1.22x is the result that takes a moment to appreciate. Conv2d is the operation that GPU vendors have spent the most effort optimizing. cuDNN's convolution implementations include hand-tuned assembly for specific kernel sizes, hardware-specific intrinsics, and years of engineering work from people who helped design the hardware. Beating it at all — let alone by 22% — on a specific combination of hardware and tensor shapes is a meaningful result. It means MCTS found an optimization path that the general-purpose implementation didn't take.

Linear layers at 1.11x follow the same logic. End-to-end across the full ResNet-50 forward pass, these per-operator gains compound to a 1.24x overall speedup.

The cost of this optimization run: $2.10. Claude Sonnet 4.6, approximately 450,000 input tokens and 150,000 output tokens across the full search, running for roughly an hour of wall-clock time on consumer hardware.

To be precise about what you are comparing: the alternative to spending $2.10 is not zero. It is weeks of a specialized engineer's time, at specialized engineer salaries, producing kernels that are tightly coupled to current hardware and need to be rewritten when you upgrade your fleet. Or it is accepting whatever performance your framework's defaults provide, which is often 20-40% below what the hardware can actually deliver.

MCTS search gets smarter with budget. More iterations mean deeper exploration of promising branches, more data on which optimization strategies transfer across operators, and a richer tree for the LLM to draw context from. The results above represent a constrained run. Larger budgets produce larger gains.

---

## The Last Mile

Getting a faster kernel is only half the story. The other half is getting it into production without dragging along everything needed to produce it.

Kernel Forge exports optimized models as self-contained `.cast` files. Under the hood they are ZIP archives containing your model weights, the optimized CUDA kernels, and a vendored runtime that requires nothing beyond standard PyTorch and CUDA. Running an optimized model looks like `python3 kernelforge/run_cast.py model.cast`. No Kernel Forge installation, no LLM dependencies, no MCTS tree or optimization state. The inference artifact is fully independent.

Getting there involves a web dashboard that shows you exactly what's happening at every step. After you upload your model code and weights and click Start Forge, the profiling phase captures real tensor I/O from your forward passes using PyTorch dispatch hooks. The generation phase shows live progress as kernels are generated and validated for each operator. The optimization phase renders the MCTS tree in real time — an interactive graph where you can click any node, read its kernel source, and see its measured performance. When an iteration produces a gain, the tree updates. When a branch stagnates, you can watch the search shift its attention elsewhere. For teams that want to understand what the optimizer is doing rather than treating it as a black box, the workbench makes every decision inspectable.

For headless or scripted runs — CI pipelines, overnight optimization jobs, benchmark sweeps — the CLI provides the same pipeline without the browser. If you don't have a local GPU, remote execution over SSH means Kernel Forge can profile and benchmark on a remote server while you manage projects locally.

---

## Beyond Vendor Libraries

For thirty years, the problem of GPU optimization has been framed as a knowledge-encoding problem. You take what kernel engineers already know, the heuristics, the tile size intuitions, the fusion rules, and you encode it into a compiler. You make the compiler smarter by encoding more knowledge. XLA, TVM, MLIR: tremendous engineering achievements, all following this playbook.

The playbook has limits. Static knowledge can't anticipate the full distribution of models and hardware. Heuristics that work for the common case fail for the edge case. The gap between "what the compiler knows" and "what your hardware can do for your model" is filled by that performance tax.

Kernel Forge flips that playbook. Instead of encoding knowledge into rules, you give a language model a framework for learning from its own experiments. Instead of precomputing optimization decisions, you search for them at deployment time, on real hardware, with real data. The result is optimization that adapts to *your* specific combination of hardware, model, and input distribution.

As language models improve at reasoning about hardware constraints, as search budgets grow, as the approach extends from individual operators to fused subgraphs and communication primitives, the case for intelligent search over static compilation will compound. The operators that fall to a $2.10 optimization run today will be joined by the operations that required human expertise yesterday.

You don't need to hire a kernel engineer. You don't need to wait for the next compiler release. You upload a model, spend the cost of a coffee, and get faster inference on your hardware, for your data, starting now.

Kernel Forge is open source: [github.com/KernelForge](https://github.com/TheJoshBrod/KernelForge)
