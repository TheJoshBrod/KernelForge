# Plan: Extract `run_cast.py` into a Reusable `castlib` Python Package

## Context

`run_cast.py` is a 347-line standalone CLI script that loads and runs `.cast` inference packages. It handles archive validation, kernel compilation, op patching, model loading, and benchmarking all in one flat file. The goal is to decompose it into a clean, pip-installable library that abstracts the `.cast` format, making it reusable as both an importable API (`cast.load("model.cast")`) and a vendored `loader.py` inside `.cast` files themselves.

---

## Package Structure

New top-level directory `castlib/` (independent from `src/` backend):

```
castlib/
  pyproject.toml
  src/
    cast/
      __init__.py       # Public API: load(), CastArchive, exceptions, __version__
      _archive.py       # CastArchive class: ZIP handling, validation, caching
      _compiler.py      # Kernel JIT compilation and precompiled .so loading
      _patcher.py       # PatchContext: op monkey-patching with revert support
      _model.py         # Model class discovery, instantiation, weight loading
      _errors.py        # CastCorruptedError, CastVersionError, etc.
      _types.py         # TypedDicts for HEADER.json, manifest.json schemas
      _cli.py           # CLI entrypoint (port of current main())
      py.typed          # PEP 561 marker
  tools/
    bundle_loader.py    # Script to flatten library into single loader.py for vendoring
```

---

## Module Breakdown

### 1. `_errors.py` - Exception hierarchy

```
CastError (base)
  CastCorruptedError   - checksum mismatch, missing files
  CastVersionError     - incompatible format_version
  CastRuntimeError     - CUDA unavailable, compilation failure
  CastModelError       - no model class found, instantiation failure
```

Replaces all bare `RuntimeError` in current code.

### 2. `_types.py` - Data structures

TypedDicts matching FileFormat.md schemas: `CastHeader`, `CastManifest`, `CastOp`, `CastRuntimeRequirements`. Lightweight, no runtime validation beyond `json.loads`.

### 3. `_archive.py` - `CastArchive` class

The central abstraction. Ports lines 14-93 of `run_cast.py`.

- `__init__(path)` - resolve path, compute cache key (SHA-256)
- `header` / `manifest` - lazy-loaded properties
- `validate()` - check file_type, format_version, checksums; check min_torch/cuda versions
- `extract(force=False)` - extract to `~/.cache/cast/<sha256>/`, skip if cached
- `_verify_checksums(zf)` - direct port of current `verify_checksums()`

### 4. `_compiler.py` - Kernel loading

Ports lines 37-60 and 95-133 of `run_cast.py`.

- `compile_kernel(kernel_cu_path, op_name, build_dir, opt_level)` - JIT via `load_inline`
- `load_precompiled(so_path, op_name)` - dlopen via `importlib`
- `load_op_kernel(op, cache_dir, build_dir, opt_level)` - high-level: try precompiled, fall back to JIT
- `detect_gpu_sm()` - return `"sm_XX"` or None

### 5. `_patcher.py` - Op patching with `PatchContext`

Ports lines 135-196 of `run_cast.py`. Key improvement: tracks all patches so they can be reverted.

- `PatchContext.patch_op(op, ext_module, kernel_cu_path)` - apply one patch
- `PatchContext.revert_all()` - restore original `torch.nn.functional.*` functions
- `_make_monkeypatch(ext, orig_fn, n_launch, orig_params)` - the closure that handles arg resolution and CUDA tensor movement (unchanged logic)

### 6. `_model.py` - Model loading

Ports lines 198-276 of `run_cast.py`.

- `import_model_module(model_py)` - importlib into isolated namespace
- `discover_model_class(module, manifest)` - find nn.Module subclass
- `instantiate_model(model_class, manifest, cache_dir, model_args)` - handle manifest args, model_config.json, or explicit override
- `load_weights(model, weight_path)` - torch.load + load_state_dict + eval

`transformers` remains a lazy import (optional dep).

### 7. `__init__.py` - Public API

```python
def load(
    path: str,
    *,
    model_args: dict | None = None,
    no_kernels: bool = False,
    opt_level: str = "-O3",
    device: str | None = None,
    verbose: bool = True,
) -> torch.nn.Module:
```

Orchestrates: validate -> extract -> compile kernels -> patch ops -> load model -> load weights -> move to device -> return.

### 8. `_cli.py` - CLI

Port of current `main()`. Registered as `cast` console script via pyproject.toml.

### 9. `pyproject.toml`

- Build: hatchling
- Dependencies: `torch>=2.1.0` only
- Optional: `transformers` via `pip install castlib[hf]`
- Console script: `cast = "cast._cli:main"`

---

## Line-by-line Migration Map

| `run_cast.py` lines | Target |
|---|---|
| 14-34 (`verify_checksums`) | `_archive.py` |
| 37-60 (`compile_kernel`) | `_compiler.py` |
| 63-93 (path, cache, header, extract) | `_archive.py` |
| 95-196 (compile + patch loop) | `_compiler.py` + `_patcher.py` |
| 198-276 (model loading) | `_model.py` |
| 279-346 (`main`) | `_cli.py` |

---

## Other Changes

- **Logging**: Replace all `print()` with `logging.getLogger("cast")`. Controlled by `verbose` param.
- **`run_cast.py`**: Replace with thin shim: `from cast._cli import main; main()`
- **`loader.py` vendoring**: `tools/bundle_loader.py` flattens the library modules into a single file for embedding in `.cast` archives.
- **DownloadCast walker** (`frontend/walkers/optimization_results.jac` ~line 1058): Update to use the bundled `loader.py` instead of the current stub. (Can be deferred to a follow-up.)

---

## Critical Files

- `/home/jodab/KernelForge/kernelforge/run_cast.py` - source of all logic
- `/home/jodab/KernelForge/docs/FileFormat.md` - authoritative .cast spec
- `/home/jodab/KernelForge/docs/cast-runtime.md` - current runtime docs
- `/home/jodab/KernelForge/frontend/walkers/optimization_results.jac` - DownloadCast walker (generates .cast files)
- `/home/jodab/KernelForge/ResNet-50-4.cast` - test fixture

---

## Implementation Order

1. Create skeleton: `pyproject.toml`, `__init__.py`, `py.typed`
2. `_errors.py` + `_types.py` (no deps)
3. `_archive.py` (port + test against `ResNet-50-4.cast`)
4. `_compiler.py` (port compile_kernel + precompiled loading)
5. `_patcher.py` (port monkey-patch loop with PatchContext)
6. `_model.py` (port model discovery/instantiation/weights)
7. `__init__.py` `load()` - wire together
8. `_cli.py` - port main(), verify identical behavior
9. Update `run_cast.py` to thin shim
10. `tools/bundle_loader.py` for loader.py generation (optional follow-up)

## Verification

- `python -c "from cast import load; model = load('ResNet-50-4.cast'); print(model)"` works
- `cast ResNet-50-4.cast --runs 1` CLI produces same output as current `python kernelforge/run_cast.py ResNet-50-4.cast --runs 1`
- `pip install -e castlib/` succeeds
- Reverting patches: `ctx.revert_all()` restores original `F.*` functions
