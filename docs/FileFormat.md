# KernelForge File Format Specification

**Version:** 1.0
**Formats:** `.anvil` (project archive) · `.cast` (inference package)

---

## Overview

KernelForge defines two file formats for packaging and distributing optimized models:

| Format | Purpose | Audience |
|--------|---------|----------|
| `.anvil` | Full project snapshot for backup, transfer, and continued optimization inside KernelForge | KernelForge users |
| `.cast` | Self-contained inference package with baked-in optimized kernels | End users / deployment |

Both formats are **ZIP archives** with a specific internal layout and a mandatory header file as the first entry in the archive. This mirrors the convention used by PyTorch's `.pt` files — the extension signals semantics, the container is standard ZIP.

---

## Shared Conventions

### Archive Structure Rules

- The **first entry** in every archive MUST be `HEADER.json`. ZIP readers can extract this single entry efficiently using the ZIP central directory without decompressing the full archive.
- All paths inside the archive are **relative and forward-slash separated** regardless of host OS.
- Binary data (weights, compiled objects) is stored **as-is** — no additional encoding.
- All JSON files use **UTF-8** encoding with no BOM.
- Timestamps are **ISO 8601** strings in UTC (e.g., `2026-02-20T14:30:00Z`).

### Integrity

Every archive contains a `checksums.sha256` file at the root. It lists the SHA-256 hash of every other file in the archive, one per line, in the format:

```
<hex_hash>  <relative/path/to/file>
```

The `HEADER.json` field `archive_checksum` holds the SHA-256 of `checksums.sha256` itself, allowing a fast top-level integrity check without reading every file.

### Versioning

Both formats carry a `format_version` field in `HEADER.json`. Version numbers follow `MAJOR.MINOR`:

- **MAJOR** bump: breaking change, old readers cannot load the file.
- **MINOR** bump: backwards-compatible addition, old readers should warn but can load.

---

## `.anvil` — Project Archive

### Purpose

An `.anvil` file is a **complete, portable snapshot of a KernelForge project**. It captures everything needed to restore the project into KernelForge on any machine — or to continue kernel optimization exactly where it left off.

`.anvil` files are produced and consumed exclusively by KernelForge. They are not intended for direct inference use.

### Internal Layout

```
project.anvil  (ZIP)
│
├── HEADER.json                        ← MUST be first entry
├── manifest.json
├── checksums.sha256
│
├── model.py                           ← original PyTorch model source
├── config.json                        ← sanitized project config (no absolute paths)
│
├── artifacts/
│   └── weights/
│       ├── <sha256_hex>.pt            ← weight files, SHA-256 named
│       └── meta.json                  ← name/hash/size registry
│
├── profiling/
│   └── <op_name>/
│       ├── entry_<n>.pt               ← captured I/O tensor pairs
│       └── summary.json               ← op profiling summary
│
├── kernels/
│   └── <op_name>/
│       ├── kernel.py                  ← best known Python dispatch wrapper
│       └── kernel.cu                  ← best known CUDA source
│
├── benchmarks/
│   └── op_benchmarks.json
│
└── trees/                             ← present only in "full" export mode
    └── <op_name>/
        └── nodes.db                   ← MCTS tree (SQLite)
```

### `HEADER.json` Schema

```json
{
  "format_version": "1.0",
  "file_type": "kernelforge_project",
  "export_mode": "snapshot | full",
  "project_name": "<string>",
  "created_at": "<ISO 8601>",
  "exported_at": "<ISO 8601>",
  "kernelforge_version": "<semver>",
  "target_device": "cuda | mps | cpu",
  "contents": {
    "has_weights": true,
    "weight_count": 1,
    "has_profiling_data": true,
    "profiled_op_count": 12,
    "has_kernels": true,
    "kernel_count": 6,
    "has_mcts_trees": false,
    "uncompressed_size_bytes": 52428800
  },
  "archive_checksum": "<sha256 of checksums.sha256>"
}
```

#### `export_mode`

| Value | Description |
|-------|-------------|
| `snapshot` | Best kernels and results only. Cannot resume MCTS optimization. |
| `full` | Includes `trees/` with all MCTS nodes. Full optimization state is preserved. |

### `manifest.json` Schema

```json
{
  "project_name": "<string>",
  "created_at": "<ISO 8601>",
  "exported_at": "<ISO 8601>",
  "llm_info": {
    "provider": "anthropic | openai | google | ollama",
    "model": "<model name>"
  },
  "target_device": "cuda | mps | cpu",
  "ops": [
    {
      "name": "<torch op name>",
      "has_profiling_data": true,
      "profiling_entries": 3,
      "has_kernel": true,
      "benchmark_speedup": 2.41
    }
  ],
  "weights": [
    {
      "id": "<sha256[:16]>",
      "name": "<original filename>",
      "sha256": "<full hex>",
      "size_bytes": 524288000
    }
  ]
}
```

### `config.json` (Sanitized)

The project `config.json` is included verbatim **except**:

- `validation_dir` is set to `null` (absolute host path, not portable).
- Any path-valued fields are stripped to `null`.
- All other fields (generator settings, profile filters, LLM info) are preserved as-is.

### Import Flow (`.anvil` → KernelForge)

1. Open ZIP, read `HEADER.json` only. Validate `format_version` and `file_type == "kernelforge_project"`.
2. Display project summary to user (name, op count, export mode, size). Wait for confirmation.
3. Verify `checksums.sha256` against `archive_checksum`. Reject if mismatch.
4. Extract to a temporary directory.
5. Create a new project directory at `$KERNELFORGE_DATA_DIR/projects/<project_name>/`.
   - If a project with that name already exists, prompt the user to rename or overwrite.
6. Copy `model.py`.
7. Re-register weights: copy `artifacts/weights/*.pt` into the new project, rebuild `config.json` artifact entries with correct local `relpath` values using `meta.json` as the source of truth.
8. Copy `profiling/` → `io/individual_ops/`.
9. Copy `kernels/` → `kernels/generated/individual_op_kernels/`.
10. If `has_mcts_trees`: copy `trees/` → `trees/`.
11. Write `config.json` from the sanitized version, setting `validation_dir` back to `null`.
12. Write a fresh `state.json` with all jobs initialized to `idle`. **Never restore** old job state — PIDs, log paths, and progress are machine-specific.
13. Delete temp directory.

### What Is Excluded

| Excluded | Reason |
|----------|--------|
| `state.json` | Job state (PIDs, progress) is machine-specific and meaningless on import |
| `.uploads/` | Temporary base64 staging files, never relevant post-creation |
| `exports/` | Old old export zips, redundant |
| `logs/` | Not functional data; large; excluded by default |
| Absolute paths | All path fields scrubbed to `null` before packaging |

---

## `.cast` — Inference Package

### Purpose

A `.cast` file is a **self-contained, deployable inference artifact**. It bundles a trained model's weights with the KernelForge-optimized CUDA kernels that replace its hot-path operators. Anyone with the `cast` Python runtime can load a `.cast` file and run inference immediately — no KernelForge installation required.

`.cast` files are produced by KernelForge and consumed by the `cast` runtime package (pip-installable, lightweight, depends only on `torch`).

### Internal Layout

```
my_model.cast  (ZIP)
│
├── HEADER.json                        ← MUST be first entry
├── manifest.json
├── checksums.sha256
│
├── model.py                           ← model class definition (for module reconstruction)
│
├── weights/
│   └── <sha256_hex>.pt                ← model weights (standard state_dict format)
│
├── kernels/
│   └── <op_name>/
│       ├── kernel.cu                  ← CUDA source (portable, JIT-compiled on target)
│       └── wrapper.py                 ← Python dispatch glue for torch.library registration
│
├── compiled/                          ← optional; pre-compiled binaries per SM version
│   ├── sm_80/
│   │   └── <op_name>.so
│   ├── sm_86/
│   │   └── <op_name>.so
│   └── sm_89/
│       └── <op_name>.so
│
└── loader.py                          ← vendored cast runtime (fallback, no pip required)
```

### `HEADER.json` Schema

```json
{
  "format_version": "1.0",
  "file_type": "kernelforge_inference",
  "project_name": "<string>",
  "exported_at": "<ISO 8601>",
  "kernelforge_version": "<semver>",
  "runtime": {
    "min_cast_version": "0.1",
    "min_torch_version": "2.1.0",
    "min_cuda_version": "12.0",
    "target_sm_versions": ["sm_80", "sm_86", "sm_89"]
  },
  "contents": {
    "optimized_op_count": 6,
    "total_op_count": 12,
    "has_precompiled": true,
    "precompiled_sm_versions": ["sm_86", "sm_89"],
    "weight_size_bytes": 524288000
  },
  "archive_checksum": "<sha256 of checksums.sha256>"
}
```

### `manifest.json` Schema

```json
{
  "project_name": "<string>",
  "exported_at": "<ISO 8601>",
  "model_class": "<ClassName>",
  "model_init_args": {},
  "weight_file": "weights/<sha256_hex>.pt",
  "ops": [
    {
      "name": "<torch op name>",
      "kernel_dir": "kernels/<op_name>/",
      "wrapper": "kernels/<op_name>/wrapper.py",
      "cuda_source": "kernels/<op_name>/kernel.cu",
      "precompiled": {
        "sm_86": "compiled/sm_86/<op_name>.so",
        "sm_89": "compiled/sm_89/<op_name>.so"
      },
      "benchmark_speedup": 2.41,
      "torch_op": "torch.nn.functional.linear"
    }
  ]
}
```

#### `model_init_args`

If the model class `__init__` requires arguments beyond the default, they are stored here as a JSON-serializable dict. On load, the model is instantiated as `ModelClass(**model_init_args)` before `load_state_dict` is called. If the model can be reconstructed from the state dict alone (e.g., all architecture info is implied by weight shapes), this field is an empty object.

### `loader.py` (Vendored Runtime)

Every `.cast` file embeds a copy of the `cast` runtime as `loader.py`. This allows loading without a pip installation using Python's `zipimport`:

```python
import zipimport
importer = zipimport.zipimporter("my_model.cast")
cast = importer.load_module("loader")
model = cast.load("my_model.cast")
output = model(input_tensor)
```

The vendored `loader.py` is functionally identical to the pip-installed `cast` package at the version embedded. The pip package is preferred when available; `loader.py` is a zero-dependency fallback.

### Load Flow (`.cast` → inference)

```
cast.load("my_model.cast")
    │
    ├── 1. Read HEADER.json
    │       Validate format_version and file_type == "kernelforge_inference"
    │       Check min_torch_version against torch.__version__
    │       Check min_cuda_version against torch.version.cuda
    │       Warn if current GPU SM version not in target_sm_versions
    │
    ├── 2. Verify checksums.sha256 against archive_checksum
    │       Raise CastCorruptedError on mismatch
    │
    ├── 3. Compute cache key: SHA-256 of the archive itself
    │       Cache dir: ~/.cache/cast/<cache_key>/
    │       If cache dir exists and is valid → skip extraction (step 4)
    │
    ├── 4. Extract archive to cache dir
    │
    ├── 5. Detect current GPU SM version (e.g., sm_86)
    │
    ├── 6. For each op in manifest["ops"]:
    │       a. If precompiled .so exists for current SM → dlopen it
    │       b. Else → JIT compile kernel.cu via torch.utils.cpp_extension.load()
    │              Compilation artifacts cached in cache dir
    │       c. Register op with torch.library under namespace "cast"
    │
    ├── 7. Import model class from model.py (exec into isolated namespace)
    │
    ├── 8. Instantiate model: ModelClass(**model_init_args)
    │
    ├── 9. torch.load(weight_file) → model.load_state_dict(state_dict)
    │
    ├── 10. Wrap model.forward() to route registered ops through cast:: dispatch
    │
    └── 11. Return model — ready for inference
```

#### Kernel Registration

Optimized kernels are registered with PyTorch's custom operator system under the `cast` namespace:

```
cast::<op_name>(Tensor self, ...) -> Tensor
```

The dispatch wrapper in `wrapper.py` replaces the original `torch.nn.functional.*` call at the Python level using a monkey-patch applied during model wrapping (step 10). This requires no changes to `model.py` or the calling code.

#### Compilation Fallback Order

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | Pre-compiled `.so` for exact SM version exists | Load directly, no compilation |
| 2 | Pre-compiled `.so` for compatible SM version exists (lower SM, same major) | Load with warning |
| 3 | `kernel.cu` source present | JIT compile via `torch.utils.cpp_extension.load()` |
| 4 | None of the above | Skip kernel for this op, fall back to native PyTorch |

### What Is Excluded

| Excluded | Reason |
|----------|--------|
| Profiling I/O tensors (`entry_*.pt`) | Not needed for inference; required only for kernel validation in KernelForge |
| MCTS trees (`nodes.db`) | Optimization history, irrelevant at inference time |
| `state.json` / job logs | KernelForge internal state, not meaningful outside KernelForge |
| Benchmark data | Informational only; not needed to run the model |
| Generation attempt history | Intermediate artifacts, not needed for final kernel |

---

## Re-importing a `.cast` into KernelForge

A `.cast` file can be ingested by KernelForge using the standard project import flow. KernelForge inspects `HEADER.json` and branches on `file_type`:

| `file_type` | Import Path |
|-------------|-------------|
| `kernelforge_project` | Full `.anvil` import flow — all data restored |
| `kernelforge_inference` | Partial import — weights and best kernels recovered; profiling data absent |

When importing a `.cast`, KernelForge will:

1. Create a new project from the embedded `model.py` and weights.
2. Install the optimized kernels as if they were generated kernels.
3. Mark all profiling jobs as **requires re-run** — profiling I/O tensors are absent, so kernel correctness cannot be re-verified until profiling completes.
4. Mark all generate/optimize jobs as **completed (imported)** — kernels exist and can be immediately benchmarked or further optimized.

---

## Format Comparison

| Property | `.anvil` | `.cast` |
|----------|----------|---------|
| Container | ZIP | ZIP |
| First entry | `HEADER.json` | `HEADER.json` |
| `file_type` | `kernelforge_project` | `kernelforge_inference` |
| Requires KernelForge to use | Yes (import) | No |
| Runnable for inference | No | Yes |
| Includes profiling tensors | Yes | No |
| Includes MCTS trees | Optional (`full` mode) | No |
| Includes pre-compiled `.so` | No | Optional |
| Embeds runtime loader | No | Yes (`loader.py`) |
| Primary audience | KernelForge users | Deployment / end users |
| Re-importable into KernelForge | Yes (full restore) | Yes (partial restore) |
