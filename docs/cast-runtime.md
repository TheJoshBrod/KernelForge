# `run_cast.py` Reference

`run_cast.py` is the standalone runtime loader for KernelForge `.cast` inference packages. It requires only `torch` and no KernelForge installation.

## CLI

```
python3 run_cast.py <file>.cast [options]

Options:
  --device cuda|cpu     Target device (default: cuda if available)
  --runs N              Number of timed inference passes (default: 5)
  --no-kernels          Skip kernel loading, run with native PyTorch ops
  --opt-level -O0..-O3  NVCC optimisation level for JIT fallback (default: -O0)
  --model-args JSON     JSON config string for model instantiation
                        e.g. '{"model_type":"resnet","num_labels":1000}'
                        Used when .cast has no model_config.json
```

## Kernel loading

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | `compiled/sm_XX/<op>.so` in archive for current GPU | `importlib` dlopen, no NVCC |
| 2 | `kernel.cu` present | JIT compile via `load_inline` (NVCC, cached in `build/`) |
| 3 | Neither | Warning, op skipped, native PyTorch used |

JIT compilation uses `load_inline` (not `load()`): splits the kernel into a tiny C++ host declaration and the full CUDA device source. This is significantly lower on peak memory than compiling the whole `.cu` as one translation unit, which matters in memory-constrained environments.

## Known limitations

- Op patching is hardcoded per op name. A generic dispatch mechanism via `torch.library` under the `cast::` namespace is the intended next step.
- Precompiled binaries are SM-specific. A `.cast` exported on sm_75 falls back to JIT on sm_80. Bundle multiple SM targets by exporting from different GPUs.
- `loader.py` inside the archive is a stub (reserved for `zipimport`-based loading without installing `run_cast.py`).
- `wrapper.py` inside the archive is a stub (reserved for a future `torch.library`-based dispatch wrapper).

## Format spec

See `docs/FileFormat.md` for the full `.cast` archive layout and schema.
