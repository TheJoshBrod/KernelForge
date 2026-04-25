#!/usr/bin/env python3
"""Load and run a .cast inference package produced by KernelForge."""

import argparse
import contextlib
import copy
import glob
import hashlib
import inspect
import json
import os
import re
import shutil
import threading
import time
import zipfile
from typing import Any, Callable

import torch.nn as nn

from src.optimizer.quantized import prepare_tinygemm_linear_launch_args


def verify_checksums(zf: zipfile.ZipFile) -> None:
    header = json.loads(zf.read("HEADER.json"))
    checksum_bytes = zf.read("checksums.sha256")
    archive_checksum = hashlib.sha256(checksum_bytes).hexdigest()
    if archive_checksum != header["archive_checksum"]:
        raise RuntimeError(
            f"Archive checksum mismatch: expected {header['archive_checksum']}, got {archive_checksum}"
        )
    for line in checksum_bytes.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_hash, rel_path = parts[0], parts[1].strip()
        if rel_path == "checksums.sha256":
            continue
        actual_hash = hashlib.sha256(zf.read(rel_path)).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {rel_path}")


def ensure_cuda_toolkit_env() -> str | None:
    """Point CUDA_HOME/CUDACXX/PATH and torch cpp_extension at a working nvcc."""

    candidates: list[str] = []

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        candidates.append(os.path.abspath(os.path.expanduser(cuda_home)))

    cudacxx = os.getenv("CUDACXX")
    if cudacxx:
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.expanduser(cudacxx)))))

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(nvcc_on_path))))

    candidates.append("/usr/local/cuda")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))

    seen: set[str] = set()
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)

        nvcc = os.path.join(resolved, "bin", "nvcc")
        if not os.path.exists(nvcc):
            continue

        os.environ["CUDA_HOME"] = resolved
        os.environ["CUDACXX"] = nvcc

        path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
        nvcc_dir = os.path.dirname(nvcc)
        if nvcc_dir not in path_entries:
            os.environ["PATH"] = nvcc_dir if not path_entries else nvcc_dir + os.pathsep + os.environ["PATH"]

        try:
            import torch.utils.cpp_extension as cpp_extension

            cpp_extension.CUDA_HOME = resolved
        except Exception:
            pass

        return resolved

    return None


def compile_kernel(kernel_cu_path: str, op_name: str, build_dir: str, opt_level: str = "-O3"):
    import re

    cuda_home = ensure_cuda_toolkit_env()
    if cuda_home is None:
        raise RuntimeError("No CUDA toolkit with nvcc was found for JIT compilation.")

    import torch.utils.cpp_extension as cpp_extension

    cpp_extension.CUDA_HOME = cuda_home
    load_inline = cpp_extension.load_inline

    with open(kernel_cu_path) as f:
        cuda_src = f.read()

    # Extract the launch() declaration for the C++ header (same approach as loader.py)
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", cuda_src)
    if not match:
        raise RuntimeError(f"Could not find 'launch' signature in {kernel_cu_path}")
    cpp_src = match.group(1) + ";"

    print(f"  Loading/compiling {op_name} ({opt_level}) ...")
    return load_inline(
        name=op_name,
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["launch"],
        extra_cuda_cflags=[opt_level],
        build_directory=build_dir,
        verbose=False,
        with_cuda=True,
    )


_PATCH_STATE = threading.local()
_ORIGINAL_FUNCTIONALS: dict[str, Callable[..., Any]] = {}
_ATEN_FALLBACK_CALL_RE = re.compile(
    r"\bat::(?:adaptive_avg_pool\d*d|batch_norm|conv\d*d|linear|max_pool\d*d|relu)\s*\("
)
_KNOWN_FAST_OPS = {
    "torch_nn_functional_linear",
    "torch_nn_functional_max_pool2d",
    "torch_nn_functional_relu",
    "torch_nn_functional_batch_norm",
}
_FOCUS_OPS = {
    "torch_nn_functional_linear",
    "torch_nn_functional_max_pool2d",
    "torch_nn_functional_batch_norm",
}
_KERNEL_POLICIES = ("all", "skip_aten", "known_fast", "focus_ops")


def _fresh_runtime_stats() -> dict[str, Any]:
    return {
        "patched_calls": 0,
        "kernel_launches_attempted": 0,
        "kernel_launches_succeeded": 0,
        "kernel_launches_failed": 0,
        "fallbacks_to_original": 0,
        "exception_fallback_count": 0,
        "contiguous_copy_count": 0,
        "adaptation_count": 0,
        "per_op": {},
    }


def _ensure_per_op_stats(runtime_stats: dict[str, Any] | None, op_name: str) -> dict[str, Any] | None:
    if runtime_stats is None:
        return None
    per_op = runtime_stats.setdefault("per_op", {})
    return per_op.setdefault(
        op_name,
        {
            "patched_calls": 0,
            "kernel_launches_attempted": 0,
            "kernel_launches_succeeded": 0,
            "kernel_launches_failed": 0,
            "fallbacks_to_original": 0,
            "exception_fallback_count": 0,
            "contiguous_copy_count": 0,
            "adaptation_count": 0,
            "last_exception": None,
            "fallback_reasons": {},
        },
    )


def _increment_stat(bucket: dict[str, Any] | None, key: str, amount: int = 1) -> None:
    if bucket is None:
        return
    bucket[key] = int(bucket.get(key, 0)) + amount


def _record_fallback(
    runtime_stats: dict[str, Any] | None,
    op_name: str,
    reason: str,
    *,
    exception: Exception | None = None,
) -> None:
    if runtime_stats is None:
        return
    op_stats = _ensure_per_op_stats(runtime_stats, op_name)
    _increment_stat(runtime_stats, "fallbacks_to_original")
    _increment_stat(op_stats, "fallbacks_to_original")
    if exception is not None:
        _increment_stat(runtime_stats, "exception_fallback_count")
        _increment_stat(op_stats, "exception_fallback_count")
        op_stats["last_exception"] = f"{type(exception).__name__}: {exception}"
    reasons = op_stats.setdefault("fallback_reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def get_runtime_stats(model: Any | None = None) -> dict[str, Any]:
    stats = getattr(model, "_kf_runtime_stats", None)
    if not isinstance(stats, dict):
        return _fresh_runtime_stats()
    return copy.deepcopy(stats)


def reset_runtime_stats(model: Any | None = None) -> dict[str, Any]:
    stats = getattr(model, "_kf_runtime_stats", None)
    if not isinstance(stats, dict):
        return _fresh_runtime_stats()
    stats.clear()
    stats.update(_fresh_runtime_stats())
    return get_runtime_stats(model)


def _patch_stack() -> list[dict[str, Callable[..., Any]]]:
    stack = getattr(_PATCH_STATE, "stack", None)
    if stack is None:
        stack = []
        _PATCH_STATE.stack = stack
    return stack


@contextlib.contextmanager
def _activate_functional_patches(
    patch_map: dict[str, Callable[..., Any]],
):
    stack = _patch_stack()
    stack.append(patch_map)
    try:
        yield
    finally:
        stack.pop()


def _ensure_functional_dispatch(fn_attr: str) -> Callable[..., Any] | None:
    import torch.nn.functional as F

    current = getattr(F, fn_attr, None)
    if current is None:
        return None

    original = _ORIGINAL_FUNCTIONALS.get(fn_attr)
    if original is not None:
        return original

    original = current
    _ORIGINAL_FUNCTIONALS[fn_attr] = original

    def dispatched(*args, **kwargs):
        stack = getattr(_PATCH_STATE, "stack", None)
        if stack:
            patch = stack[-1].get(fn_attr)
            if patch is not None:
                return patch(*args, **kwargs)
        return original(*args, **kwargs)

    setattr(F, fn_attr, dispatched)
    return original


def _launch_arity(kernel_cu: str, ext: Any) -> int | None:
    n_launch = None
    if os.path.exists(kernel_cu):
        try:
            with open(kernel_cu) as handle:
                cu_src = handle.read()
            match = re.search(r"torch::Tensor\s+launch\s*\(([^)]*)\)", cu_src)
            if match:
                params = [part.strip() for part in match.group(1).split(",") if part.strip()]
                n_launch = len(params)
        except Exception:
            pass
    if n_launch is None:
        try:
            n_launch = len(inspect.signature(ext.launch).parameters)
        except Exception:
            pass
    return n_launch


def _kernel_calls_aten_fallback(kernel_cu: str) -> bool:
    if not os.path.exists(kernel_cu):
        return False
    try:
        with open(kernel_cu) as handle:
            cuda_src = handle.read()
    except Exception:
        return False
    return bool(_ATEN_FALLBACK_CALL_RE.search(cuda_src))


def _kernel_policy_skip_reason(op_name: str, kernel_cu: str, kernel_policy: str) -> str | None:
    if kernel_policy == "all":
        return None

    uses_aten_fallback = _kernel_calls_aten_fallback(kernel_cu)

    if kernel_policy == "skip_aten":
        if uses_aten_fallback:
            return "kernel source falls back to ATen"
        return None

    if kernel_policy == "known_fast":
        if uses_aten_fallback:
            return "kernel source falls back to ATen"
        if op_name not in _KNOWN_FAST_OPS:
            return "op is not in the known-fast allowlist"
        return None

    if kernel_policy == "focus_ops":
        if uses_aten_fallback:
            return "kernel source falls back to ATen"
        if op_name not in _FOCUS_OPS:
            return "op is not in the focus-ops allowlist"
        return None

    raise ValueError(f"Unknown kernel policy: {kernel_policy}")


def _build_functional_patch(
    *,
    op_name: str,
    ext: Any,
    orig_fn: Callable[..., Any],
    n_launch: int | None,
    orig_params: list[str] | None,
    runtime_stats: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    import torch

    def patched(*args, **kwargs):
        op_stats = _ensure_per_op_stats(runtime_stats, op_name)
        _increment_stat(runtime_stats, "patched_calls")
        _increment_stat(op_stats, "patched_calls")
        try:
            if orig_params is not None:
                resolved = {orig_params[i]: value for i, value in enumerate(args) if i < len(orig_params)}
                resolved.update(kwargs)
                ordered = [resolved.get(name) for name in orig_params]
            else:
                ordered = list(args)

            special_args = prepare_tinygemm_linear_launch_args(
                op_name,
                ordered,
                {},
                {"params": orig_params or []},
            )
            if special_args is not None:
                call_args = special_args
                tensor_args = [value for value in call_args if isinstance(value, torch.Tensor)]
                _increment_stat(runtime_stats, "adaptation_count")
                _increment_stat(op_stats, "adaptation_count")
            else:
                limit = n_launch if n_launch is not None else len(ordered)
                call_args: list[Any] = []
                tensor_args: list[torch.Tensor] = []
                for value in ordered[:limit]:
                    if isinstance(value, torch.Tensor):
                        tensor_args.append(value)
                        if not value.is_contiguous():
                            _increment_stat(runtime_stats, "contiguous_copy_count")
                            _increment_stat(runtime_stats, "adaptation_count")
                            _increment_stat(op_stats, "contiguous_copy_count")
                            _increment_stat(op_stats, "adaptation_count")
                        call_args.append(value.contiguous())
                    else:
                        call_args.append(value)

            if not tensor_args:
                _record_fallback(runtime_stats, op_name, "no_tensor_args")
                return orig_fn(*args, **kwargs)

            if any(not tensor.is_cuda for tensor in tensor_args):
                _record_fallback(runtime_stats, op_name, "non_cuda_tensor")
                return orig_fn(*args, **kwargs)

            first_device = tensor_args[0].device
            if any(tensor.device != first_device for tensor in tensor_args):
                _record_fallback(runtime_stats, op_name, "mixed_device_tensor")
                return orig_fn(*args, **kwargs)

            _increment_stat(runtime_stats, "kernel_launches_attempted")
            _increment_stat(op_stats, "kernel_launches_attempted")
            result = ext.launch(*call_args)
            _increment_stat(runtime_stats, "kernel_launches_succeeded")
            _increment_stat(op_stats, "kernel_launches_succeeded")
            return result
        except Exception as exc:
            _increment_stat(runtime_stats, "kernel_launches_failed")
            _increment_stat(op_stats, "kernel_launches_failed")
            _record_fallback(runtime_stats, op_name, "kernel_exception", exception=exc)
            return orig_fn(*args, **kwargs)

    patched.__name__ = f"cast_patch_{op_name}"
    return patched


class CastModelRuntime(nn.Module):
    """A thin nn.Module wrapper that activates cast kernel patches per forward."""

    def __init__(self, model, functional_patches: dict[str, Callable[..., Any]]):
        import torch.nn as nn

        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        super().__init__()
        self.model = model
        self._cast_functional_patches = functional_patches

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = self._modules.get("model")
            if model is None:
                raise
            return getattr(model, name)

    def forward(self, *args, **kwargs):
        if not self._cast_functional_patches:
            return self.model(*args, **kwargs)
        with _activate_functional_patches(self._cast_functional_patches):
            return self.model(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self(*args, **kwargs)


def _load_model_module(module_name: str, model_py: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create a module spec for {model_py}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        del sys.modules[module_name]
        raise
    return mod


def _resolve_model_class(mod, model_class_name: str):
    if model_class_name and hasattr(mod, model_class_name):
        return getattr(mod, model_class_name)
    candidates = [
        value for value in vars(mod).values()
        if isinstance(value, type) and issubclass(value, nn.Module) and value.__module__ not in ("torch.nn", "builtins")
    ]
    return candidates[-1] if candidates else None


def _build_model_from_class(ModelClass, manifest: dict[str, Any], model_args: dict | None, cache_dir: str):
    model_init_args = manifest.get("model_init_args") or {}
    model_config_file = os.path.join(cache_dir, "model_config.json")

    if model_args:
        try:
            from transformers import AutoConfig

            cfg_dict = dict(model_args)
            model_type = cfg_dict.pop("model_type", None)
            config = AutoConfig.for_model(model_type, **cfg_dict)
            return ModelClass(config)
        except Exception as exc:
            raise RuntimeError(f"Failed to instantiate model from --model-args: {exc}") from exc

    if model_init_args:
        return ModelClass(**model_init_args)

    if os.path.exists(model_config_file):
        try:
            from transformers import AutoConfig

            cfg_dict = json.load(open(model_config_file))
            model_type = cfg_dict.pop("model_type", None)
            n_labels = cfg_dict.get("num_labels")
            if n_labels:
                cfg_dict["id2label"] = {str(i): f"LABEL_{i}" for i in range(n_labels)}
                cfg_dict["label2id"] = {f"LABEL_{i}": i for i in range(n_labels)}
            config = AutoConfig.for_model(model_type, **cfg_dict)
            return ModelClass(config)
        except Exception as exc:
            raise RuntimeError(f"Failed to instantiate model from model_config.json: {exc}") from exc

    return None


def _invoke_entrypoint(fn, *, weight_file: str | None, device: str | None, model_args: dict | None):
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    accepts_var_args = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)
    kwargs: dict[str, Any] = {}

    if model_args:
        for key, value in model_args.items():
            if accepts_var_kw or key in signature.parameters:
                kwargs[key] = value

    if device is not None and (accepts_var_kw or "device" in signature.parameters):
        kwargs["device"] = device

    args: list[Any] = []
    if weight_file is not None:
        positional_params = [
            param for param in params
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if positional_params or accepts_var_args:
            args.append(weight_file)

    return fn(*args, **kwargs)


def load_cast(
    cast_path: str,
    model_args: dict | None = None,
    no_kernels: bool = False,
    opt_level: str = "-O3",
    device: str | None = None,
    kernel_policy: str = "all",
    allow_jit: bool = True,
    require_precompiled: bool = False,
    record_runtime_stats: bool = False,
):
    import torch

    def _file_sha256(path: str) -> str | None:
        if not os.path.exists(path) or not os.path.isfile(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    load_started = time.perf_counter()
    cast_path = os.path.abspath(cast_path)
    cache_key = hashlib.sha256(open(cast_path, "rb").read()).hexdigest()
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "cast", cache_key)

    print(f"Loading {cast_path}")

    with zipfile.ZipFile(cast_path) as zf:
        # 1. Validate header
        header = json.loads(zf.read("HEADER.json"))
        if header["file_type"] != "kernelforge_inference":
            raise RuntimeError(f"Expected kernelforge_inference, got {header['file_type']}")
        print(f"  Project : {header['project_name']}")
        print(f"  Version : {header['format_version']}")

        # 2. Verify checksums
        print("  Verifying checksums ...")
        verify_checksums(zf)
        print("  Checksums OK")

        # 3. Extract to cache dir
        if not os.path.isdir(cache_dir):
            print(f"  Extracting to {cache_dir}")
            zf.extractall(cache_dir)
        else:
            print(f"  Using cached extraction at {cache_dir}")

        manifest = json.loads(zf.read("manifest.json"))

    # 4. JIT compile kernels and patch ops
    build_dir = os.path.join(cache_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    _F_PREFIX = "torch_nn_functional_"
    functional_patches: dict[str, Callable[..., Any]] = {}
    runtime_stats = _fresh_runtime_stats()
    runtime_report: dict[str, Any] = {
        "cast_path": cast_path,
        "cache_key": cache_key,
        "cache_dir": cache_dir,
        "kernel_policy": kernel_policy,
        "allow_jit": bool(allow_jit),
        "require_precompiled": bool(require_precompiled),
        "record_runtime_stats": bool(record_runtime_stats),
        "runtime_patch_enabled": False,
        "selected_ops": [],
        "loaded_kernels": [],
        "op_reports": [],
        "kernel_source_hashes": {},
        "precompiled_binary_hashes": {},
        "jit_compile_time_ms": 0.0,
        "precompiled_load_time_ms": 0.0,
    }

    if kernel_policy not in _KERNEL_POLICIES:
        raise ValueError(
            f"Unknown kernel policy '{kernel_policy}'. Expected one of: {', '.join(_KERNEL_POLICIES)}"
        )

    for op in manifest["ops"]:
        op_name = op["name"]
        kernel_cu = os.path.join(cache_dir, op["cuda_source"])
        op_report: dict[str, Any] = {
            "op_name": op_name,
            "cuda_source": op.get("cuda_source"),
            "kernel_source_path": kernel_cu,
            "kernel_source_hash": _file_sha256(kernel_cu),
            "load_mode": "skipped",
            "skip_reason": None,
            "patch_registered": False,
            "precompiled_binary_path": None,
            "precompiled_binary_hash": None,
        }
        if op_report["kernel_source_hash"] is not None and op.get("cuda_source"):
            runtime_report["kernel_source_hashes"][op["cuda_source"]] = op_report["kernel_source_hash"]

        if no_kernels:
            print(f"  [--no-kernels] Skipping kernel for {op_name}")
            op_report["skip_reason"] = "no_kernels"
            runtime_report["op_reports"].append(op_report)
            continue

        if not torch.cuda.is_available():
            print(f"  [WARN] CUDA not available — skipping kernel for {op_name}")
            op_report["skip_reason"] = "cuda_unavailable"
            runtime_report["op_reports"].append(op_report)
            continue

        skip_reason = _kernel_policy_skip_reason(op_name, kernel_cu, kernel_policy)
        if skip_reason:
            print(f"  [kernel-policy={kernel_policy}] Skipping kernel for {op_name}: {skip_reason}")
            op_report["skip_reason"] = skip_reason
            runtime_report["op_reports"].append(op_report)
            continue

        # Try precompiled .so for the current GPU first
        gpu_sm = "sm_{0}{1}".format(*torch.cuda.get_device_capability())
        precompiled = op.get("precompiled", {})
        so_rel = precompiled.get(gpu_sm)
        so_path = os.path.join(cache_dir, so_rel) if so_rel else None

        ext = None
        if so_path and os.path.exists(so_path):
            try:
                import importlib.util as _ilu

                precompiled_started = time.perf_counter()
                _spec = _ilu.spec_from_file_location(op_name, so_path)
                ext = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(ext)  # type: ignore[union-attr]
                runtime_report["precompiled_load_time_ms"] += (time.perf_counter() - precompiled_started) * 1000.0
                op_report["load_mode"] = "precompiled"
                op_report["precompiled_binary_path"] = so_path
                op_report["precompiled_binary_hash"] = _file_sha256(so_path)
                if so_rel and op_report["precompiled_binary_hash"] is not None:
                    runtime_report["precompiled_binary_hashes"][so_rel] = op_report["precompiled_binary_hash"]
                print(f"  Loaded precompiled {op_name} ({gpu_sm})")
            except Exception as exc:
                print(f"  [WARN] Failed to load precompiled {op_name} ({gpu_sm}): {exc}")
                op_report["precompiled_load_error"] = f"{type(exc).__name__}: {exc}"

        if ext is None:
            if so_rel:
                print(f"  [WARN] Precompiled .so not found for {gpu_sm}, falling back to JIT")
            if require_precompiled:
                raise RuntimeError(
                    f"Precompiled kernel required for {op_name} on {gpu_sm}, but no usable shared object was available."
                )
            if not allow_jit:
                raise RuntimeError(
                    f"JIT loading is disabled and no usable precompiled kernel was available for {op_name} on {gpu_sm}."
                )
            if not os.path.exists(kernel_cu):
                print(f"  [WARN] No kernel.cu for {op_name}, skipping")
                op_report["skip_reason"] = "kernel_source_missing"
                runtime_report["op_reports"].append(op_report)
                continue
            try:
                jit_started = time.perf_counter()
                ext = compile_kernel(kernel_cu, op_name, build_dir, opt_level=opt_level)
                runtime_report["jit_compile_time_ms"] += (time.perf_counter() - jit_started) * 1000.0
                op_report["load_mode"] = "jit"
            except Exception as exc:
                print(f"  [WARN] Failed to prepare kernel for {op_name}, using native PyTorch: {exc}")
                op_report["skip_reason"] = f"jit_failed: {type(exc).__name__}: {exc}"
                runtime_report["op_reports"].append(op_report)
                continue

        # Generic patch: decode torch.nn.functional.<attr> from the op_name convention.
        if not op_name.startswith(_F_PREFIX):
            print(f"  [WARN] '{op_name}' does not follow torch_nn_functional_* convention, skipping patch")
            op_report["skip_reason"] = "unsupported_op_naming"
            runtime_report["op_reports"].append(op_report)
            continue

        fn_attr = op_name[len(_F_PREFIX):]
        original = _ensure_functional_dispatch(fn_attr)
        if original is None:
            print(f"  [WARN] torch.nn.functional.{fn_attr} not found, skipping patch")
            op_report["skip_reason"] = f"functional_target_missing:{fn_attr}"
            runtime_report["op_reports"].append(op_report)
            continue

        n_launch = _launch_arity(kernel_cu, ext)

        # Resolve the original function's parameter names so we can handle
        # both positional and keyword call sites correctly.
        try:
            orig_params = list(inspect.signature(original).parameters.keys())
        except Exception:
            orig_params = None

        functional_patches[fn_attr] = _build_functional_patch(
            op_name=op_name,
            ext=ext,
            orig_fn=original,
            n_launch=n_launch,
            orig_params=orig_params,
            runtime_stats=runtime_stats if record_runtime_stats else None,
        )
        op_report["patch_registered"] = True
        runtime_report["selected_ops"].append(op_name)
        runtime_report["loaded_kernels"].append(
            {
                "op_name": op_name,
                "fn_attr": fn_attr,
                "load_mode": op_report["load_mode"],
                "kernel_source_path": kernel_cu,
                "kernel_source_hash": op_report["kernel_source_hash"],
                "precompiled_binary_path": op_report["precompiled_binary_path"],
                "precompiled_binary_hash": op_report["precompiled_binary_hash"],
            }
        )
        runtime_report["op_reports"].append(op_report)
        print(f"  Registered runtime patch torch.nn.functional.{fn_attr} → {op_name}")

    # 5. Load model implementation from model.py
    model_py = os.path.join(cache_dir, "model.py")
    module_name = f"cast_model_{cache_key[:12]}"
    mod = _load_model_module(module_name, model_py)

    model_class_name = str(manifest.get("model_class") or "").strip()
    model_entrypoints = manifest.get("model_entrypoints", {}) if isinstance(manifest.get("model_entrypoints"), dict) else {}
    build_model_fn = getattr(mod, "build_model", None) if model_entrypoints.get("build_model") or hasattr(mod, "build_model") else None
    load_weights_fn = getattr(mod, "load_weights", None) if model_entrypoints.get("load_weights") or hasattr(mod, "load_weights") else None
    ModelClass = _resolve_model_class(mod, model_class_name)

    model = None
    model_load_strategy = None
    weight_relpath = str(manifest.get("weight_file") or "").strip()
    weight_file = os.path.join(cache_dir, weight_relpath) if weight_relpath else None

    if ModelClass is not None:
        print(f"  Model class: {ModelClass.__name__}")
        model = _build_model_from_class(ModelClass, manifest, model_args, cache_dir)
        if model is not None:
            model_load_strategy = "model_class"
            if weight_file:
                import torch

                print(f"  Loading weights from {weight_relpath} ...")
                state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)

    if model is None and callable(load_weights_fn):
        print("  Loading model via model.py::load_weights(...)")
        model = _invoke_entrypoint(
            load_weights_fn,
            weight_file=weight_file or "",
            device=device or "cpu",
            model_args=model_args,
        )
        model_load_strategy = "load_weights"

    if model is None and callable(build_model_fn):
        print("  Loading model via model.py::build_model(...)")
        model = _invoke_entrypoint(
            build_model_fn,
            weight_file=None,
            device=device,
            model_args=model_args,
        )
        model_load_strategy = "build_model"

    if model is None:
        raise RuntimeError(
            "Cannot instantiate model from the .cast package. "
            "Expected a loadable model_class or model.py build_model/load_weights entrypoint."
        )

    if not isinstance(model, nn.Module):
        raise RuntimeError(f"Loaded object from .cast is not an nn.Module: {type(model).__name__}")

    model.eval()

    runtime_model = CastModelRuntime(model, functional_patches)
    runtime_model.eval()
    if device:
        runtime_model = runtime_model.to(device)

    runtime_report["runtime_patch_enabled"] = bool(functional_patches)
    runtime_report["load_modes"] = {
        entry["op_name"]: entry.get("load_mode")
        for entry in runtime_report["op_reports"]
    }
    runtime_report["selected_ops"] = sorted(dict.fromkeys(runtime_report["selected_ops"]))
    runtime_report["runtime_load_time_ms"] = (time.perf_counter() - load_started) * 1000.0
    runtime_report["setup_time_ms"] = max(
        float(runtime_report["runtime_load_time_ms"]) - float(runtime_report["jit_compile_time_ms"]),
        0.0,
    )
    runtime_report["model_load_strategy"] = model_load_strategy
    runtime_report["model_entrypoints"] = {
        "model_class": model_class_name,
        "build_model": bool(callable(build_model_fn)),
        "load_weights": bool(callable(load_weights_fn)),
    }

    runtime_model._kf_runtime_report = runtime_report
    runtime_model._kf_runtime_stats = runtime_stats
    runtime_model._kf_runtime_stats_recording_enabled = bool(record_runtime_stats)

    return runtime_model


def _parse_shape(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("input shape must contain at least one dimension")
    dims = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in dims):
        raise ValueError("input shape dimensions must be positive integers")
    return dims


def _benchmark_loaded_model(model, device: str, runs: int, input_shape: tuple[int, ...]) -> None:
    import torch

    dummy = torch.randn(*input_shape, device=device)
    print(f"Running {runs} inference pass(es) with input shape {list(dummy.shape)} ...")

    with torch.inference_mode():
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            out = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / runs * 1000

    logits = out.logits if hasattr(out, "logits") else out
    if hasattr(logits, "shape"):
        print(f"Output shape    : {list(logits.shape)}")
    print(f"Average latency : {elapsed_ms:.2f} ms")
    if getattr(logits, "ndim", 0) >= 2 and logits.shape[0] >= 1:
        top5 = logits[0].topk(min(5, logits.shape[-1]))
        print(f"Top-5 indices   : {top5.indices.tolist()}")
        print(f"Top-5 scores    : {[f'{v:.4f}' for v in top5.values.tolist()]}")


def main(*, default_kernel_policy: str = "all") -> None:
    import torch

    parser = argparse.ArgumentParser(description="Run a KernelForge .cast inference package")
    parser.add_argument("cast_file", help="Path to .cast file")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--model-args",
        metavar="JSON",
        default=None,
        help='JSON config to instantiate the model, e.g. \'{"model_type":"resnet",...}\'. '
             "Used when the .cast has no model_config.json.",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="Skip JIT kernel compilation and run with native PyTorch ops.",
    )
    parser.add_argument(
        "--opt-level",
        default="-O3",
        choices=["-O0", "-O1", "-O2", "-O3"],
        help="NVCC optimisation level for JIT compilation (default: -O3).",
    )
    parser.add_argument(
        "--kernel-policy",
        default=default_kernel_policy,
        choices=list(_KERNEL_POLICIES),
        help=(
            "Kernel selection policy: "
            "'all' loads every exported kernel, "
            "'skip_aten' skips kernels that just call back into ATen, "
            "'known_fast' keeps only the current known-fast allowlist, "
            "'focus_ops' keeps only the smallest fast-op subset."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a dummy-input benchmark after loading the model.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Inference passes for timing with --benchmark")
    parser.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Comma-separated dummy input shape for --benchmark (default: 1,3,224,224).",
    )
    args = parser.parse_args()

    extra_model_args = json.loads(args.model_args) if args.model_args else None
    model = load_cast(
        args.cast_file,
        model_args=extra_model_args,
        no_kernels=args.no_kernels,
        opt_level=args.opt_level,
        device=args.device,
        kernel_policy=args.kernel_policy,
    )
    print(f"\nModel ready on {args.device}")
    print(f"Kernel policy  : {args.kernel_policy}")
    print("Import load_cast(...) in production code to load a .cast as a normal model object.")

    if args.benchmark:
        input_shape = _parse_shape(args.input_shape)
        _benchmark_loaded_model(model, args.device, args.runs, input_shape)


if __name__ == "__main__":
    main()
