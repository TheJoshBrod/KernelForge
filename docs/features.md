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
