# Features

## Automated kernel generation

- Hooks into PyTorch's dispatch mechanism to capture real input/output tensor pairs for each operator during a forward pass.
- Prompts an LLM with the operator signature and captured data to generate a candidate kernel.
- Compiles and validates the kernel in a tight loop - compile errors and numerical mismatches are automatically fed back to the LLM until output matches `torch.allclose`.

## MCTS-driven optimization

- Treats optimization as a tree search: each kernel variant is a node, each LLM-suggested rewrite (tiling, loop unrolling, vectorized memory access, etc.) is a child branch.
- Monte Carlo Tree Search balances exploring new strategies against exploiting branches that have already yielded speedups.
- The full tree is persisted to disk so runs can be paused, resumed, or inspected at any time.

## CUDA and Triton backends

- **CUDA**: generates `.cu` kernels compiled via `torch.utils.cpp_extension`. Supports NVIDIA GPUs.
- **Triton**: generates `.py` Triton kernels profiled via `triton.testing.do_bench`. Supports both NVIDIA and **AMD ROCm** GPUs (via `rocm-smi` for device specs).

## Remote execution over SSH

- Both backends support offloading compilation and benchmarking to a remote GPU host over SSH.
- KernelForge bootstraps a Python environment on the remote host automatically (`~/kforge_workspace/venv`), uploading dependencies and worker scripts as needed.
- Auth supports password or key-based (RSA, ED25519, ECDSA, DSA).

## Multi-LLM support

Kernel generation and optimization are driven by any text generation model from the following providers:

- **Anthropic**
- **OpenAI**
- **Google**

Provider is inferred automatically from the selected model name.

## Web dashboard

- Live pipeline status and per-operator progress while runs are active.
- Speed comparison charts (baseline PyTorch vs. optimized kernel) that update as operators complete.
- MCTS tree inspector to browse every optimization attempt and its measured speedup.
- **Automatic mode:** forge all discovered operators in one click.
- **Manual mode:** select specific operators to target.

### Dashboard — Operator Results table

| Column | What it shows | Scope |
|--------|--------------|-------|
| Op | Operator name (e.g. `conv2d`) | — |
| PyTorch ms | Average PyTorch latency per call | Average over full validation set |
| % Time | Share of total inference time (`avg_ms × calls / total`) | Derived |
| Calls | Number of times the op runs per inference | Single forward pass |
| Kernel ms | Optimized kernel latency per call | Best result from MCTS search (if optimized); otherwise average over full validation set |
| Speedup | PyTorch ms / Kernel ms | Derived |

### Operator Workbench

Lists every profiled operator with call frequency and timing. Selecting an operator opens the MCTS tree inspector for that op.

| Badge mode | Source |
|-----------|--------|
| Frequency (# calls) | Single forward pass count from `summary.json` |
| % Time | Derived from avg latency × calls / total |
| Avg ms | Average PyTorch latency over full validation set |

### Data Flow view

Renders the operator-level compute graph for a single forward pass so you can identify where kernel fusions could be made.

- **Nodes** — each meaningful NN op (`conv2d`, `batch_norm`, `relu`, `max_pool2d`, etc.). Primitive tensor ops (`reshape`, `add`, `flatten`, etc.) are hidden but their data-flow connections are preserved so the graph stays fully connected.
- **Edges** — directed tensor data flow. Multiple incoming edges on a node indicate a residual/skip connection.
- **Node IDs** — formatted as `<op>_<N>` where `N` is the op's position in the global execution sequence (not the Nth call of that op type). There can be 53 `conv2d` nodes even though the highest-numbered one might be `conv2d_82`, because batch_norm and relu nodes occupy the indices in between.
- **Re-Profile** — triggers a fresh profiling run and polls every 10 seconds, auto-updating the graph when the new `dag.json` is written.
- **Export to Mermaid** — copies the graph as Mermaid `graph TD` syntax for use in documentation or external tools.

## Operators profiled

The profiling system targets `torch.nn.functional` operators by default:

| Category | Operators |
|----------|-----------|
| Convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d` |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `adaptive_max_pool2d` |
| Linear / Attention | `linear`, `scaled_dot_product_attention` |
| Normalization | `layer_norm`, `batch_norm` |
| Activation | `relu`, `gelu`, `softmax` |
| Other | `embedding`, `dropout`, `pad` |

Shape ops, tensor creation ops, and random ops are skipped by default but can be enabled via config.

## Portable project formats

- **`.anvil`**: complete project snapshot (profiling data, kernels, MCTS trees) that can be moved to another machine and resumed inside KernelForge.
- **`.cast`**: self-contained inference package bundling model weights and optimized kernels. Loadable with only `torch` installed; no KernelForge required at inference time.
