# `.cast` Runtime — Changes & How It Works

## What Was Built

`run_cast.py` is the runtime loader for KernelForge `.cast` inference packages. It did not exist before this session — the file contained terminal output from a previous Claude Code session and threw an `IndentationError` immediately.

---

## Changes Made

### 1. `run_cast.py` — created from scratch

The file was rewritten as a working Python script. The full load sequence is documented in the "How It Works" section below.

Key problems solved during development:

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| `IndentationError` on line 2 | File contained terminal session output, not Python | Rewrote the file |
| `PyInit_` not found in `.so` | `torch.utils.cpp_extension.load()` requires `PYBIND11_MODULE`; LLM-generated kernels don't have it | Switched to `load_inline` (same approach as `loader.py`) which auto-generates bindings from the `launch` signature |
| NVCC killed (exit 137) | OOM — WSL2 had ~3 GB free, swap exhausted; `load()` compiles the whole `.cu` as one unit | `load_inline` splits into a tiny C++ declaration + CUDA source, dramatically lower peak memory |
| `inspect.getfile` → "built-in class" | `@auto_docstring` in transformers calls `inspect.getsourcefile(cls)`; the class's `__module__` (`cast_model`) was not in `sys.modules`, so the lookup returned `None` | Register the module with `sys.modules["cast_model"] = mod` **before** calling `exec_module`, which is how Python's normal import system works |
| `num_labels` mismatch | `ResNetConfig().to_dict()` doesn't include `num_labels` as a direct key (it's a computed property from `id2label`); `id2label` had 2 entries | Rebuild `id2label`/`label2id` from `num_labels` before calling `AutoConfig.for_model` |
| Recompiling on every run | The stale-`.so` cleanup code ran unconditionally, deleting the valid cache before every invocation | Removed the cleanup; `load_inline` handles mtime-based caching internally |

### 2. `frontend/walkers/project.jac` — export walker

Four additions to the `DownloadCast` walker:

**a) `model_class` now populated**
The export previously hardcoded `"model_class": ""`. Now it AST-parses `model.py` to extract the first class name and stores it in `manifest.json`:
```python
import ast
tree = ast.parse(model_py_bytes.decode("utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        model_class_name = node.name
        break
```

**b) Precompiled `.so` bundled**
The export now detects the current GPU's SM version via `torch.cuda.get_device_capability()` and bundles the compiled `.so` (which lives alongside `kernel.cu` in the project) under `compiled/sm_XX/<op_name>.so`. The manifest's `precompiled` dict and the HEADER's `has_precompiled` / `precompiled_sm_versions` fields are updated accordingly.

**c) `model_config.json` auto-generated**
For HuggingFace models, if no `model_config.json` exists in the project directory, the export now:
1. Imports the model class from `model.py` via `importlib`
2. Calls `ModelClass.config_class()` to get default config values
3. Loads the weight state dict and patches `num_labels` by looking for the classifier head weight (`classifier.1.weight`, `classifier.weight`, `fc.weight`, etc.)
4. Serialises the result as `model_config.json` inside the archive

**d) `model_config.json` explicit file takes priority**
If `model_config.json` already exists in the project directory, it is used as-is and the auto-generation is skipped.

---

## How It Works

### `.cast` File Format

A `.cast` file is a ZIP archive with this layout:

```
my_model.cast  (ZIP)
├── HEADER.json                         ← must be first entry
├── manifest.json
├── checksums.sha256
├── model.py                            ← model class definition
├── model_config.json                   ← HuggingFace config (architecture + num_labels)
├── weights/
│   └── <sha256>.pt                     ← model state dict
├── kernels/
│   └── <op_name>/
│       ├── kernel.cu                   ← CUDA source (JIT fallback)
│       └── wrapper.py                  ← stub (reserved)
├── compiled/
│   └── sm_75/
│       └── <op_name>.so               ← precompiled Python C extension
└── loader.py                           ← stub (reserved for pip cast package)
```

### Load Sequence (`run_cast.py`)

```
run_cast.py <file>.cast
    │
    ├── 1. Read HEADER.json
    │       Validate file_type == "kernelforge_inference"
    │
    ├── 2. Verify checksums.sha256
    │       SHA-256 of every file checked against checksums.sha256
    │       SHA-256 of checksums.sha256 checked against HEADER.archive_checksum
    │
    ├── 3. Extract archive to ~/.cache/cast/<sha256-of-archive>/
    │       Cache keyed on archive hash — re-run skips extraction
    │
    ├── 4. For each op in manifest["ops"]:
    │       a. Detect current GPU SM version (e.g. sm_75)
    │       b. If compiled/sm_75/<op>.so exists → importlib.load from .so (no NVCC)
    │       c. Else → JIT compile kernel.cu via load_inline (NVCC, cached in build/)
    │       d. Monkey-patch the corresponding torch.nn.functional.* function
    │
    ├── 5. Load model class from model.py
    │       importlib.spec_from_file_location("cast_model", model.py)
    │       Register in sys.modules before exec (required for inspect.getfile)
    │       Use manifest["model_class"] to find the class; fallback to nn.Module scan
    │
    ├── 6. Instantiate model
    │       Priority:
    │         --model-args JSON  →  AutoConfig.for_model + ModelClass(config)
    │         model_init_args    →  ModelClass(**model_init_args)
    │         model_config.json  →  AutoConfig.for_model + ModelClass(config)
    │                                 (id2label/label2id rebuilt from num_labels)
    │
    ├── 7. Load weights
    │       torch.load(weights/<sha256>.pt, weights_only=True)
    │       model.load_state_dict(state_dict)
    │       model.eval()
    │
    └── 8. Run inference
            Warmup pass + N timed passes
            Reports output shape, average latency, top-5 predictions
```

### Kernel Loading: Precompiled vs JIT

The precompiled path is preferred — it avoids NVCC entirely:

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | `compiled/sm_XX/<op>.so` in archive for current GPU | `importlib` dlopen — instant |
| 2 | `kernel.cu` present | `load_inline` JIT compile via NVCC (cached in `build/`) |
| 3 | Neither | Warning, op skipped, native PyTorch used |

### Kernel Compilation (`load_inline`)

Mirrors `src/optimizer/backends/cuda/loader.py` exactly:

```python
# 1. Extract the launch() C++ declaration from kernel.cu
cpp_src = "torch::Tensor launch(...);"

# 2. Compile — separates C++ host and CUDA device compilation
load_inline(
    cpp_sources=cpp_src,      # tiny host-side declaration only
    cuda_sources=cuda_src,    # full CUDA device code
    functions=["launch"],     # auto-generates Python bindings
    ...
)
```

This is significantly lighter on memory than `load()` which compiles the whole `.cu` as a single translation unit. Important in memory-constrained environments (WSL2).

### CLI Reference

```
python3 run_cast.py <file>.cast [options]

Options:
  --device cuda|cpu     Target device (default: cuda if available)
  --runs N              Number of timed inference passes (default: 5)
  --no-kernels          Skip kernel loading, run with native PyTorch ops
  --opt-level -O0..-O3  NVCC optimisation for JIT fallback (default: -O0)
  --model-args JSON     JSON config string for model instantiation
                        e.g. '{"model_type":"resnet","num_labels":1000}'
                        Used when .cast has no model_config.json
```

---

## Known Limitations

- **Op patching is currently hardcoded** to `torch_nn_functional_linear`. A generic dispatch mechanism (e.g. `torch.library` registration under the `cast::` namespace) is the next step.
- **Precompiled binaries are SM-specific** — a `.cast` exported on sm_75 will fall back to JIT on sm_80. Multiple SM targets can be bundled by exporting from machines with different GPUs.
- **`loader.py` inside the archive is a stub** — the spec defines it as a vendored runtime for `zipimport`-based loading without installing `run_cast.py`. Not yet implemented.
- **`wrapper.py` inside the archive is a stub** — reserved for a future `torch.library`-based dispatch wrapper.
