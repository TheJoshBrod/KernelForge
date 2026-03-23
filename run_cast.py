#!/usr/bin/env python3
"""Load and run a .cast inference package produced by KernelForge."""

import argparse
import contextlib
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
) -> Callable[..., Any]:
    import torch

    def patched(*args, **kwargs):
        try:
            if orig_params is not None:
                resolved = {orig_params[i]: value for i, value in enumerate(args) if i < len(orig_params)}
                resolved.update(kwargs)
                ordered = [resolved.get(name) for name in orig_params]
            else:
                ordered = list(args)

            limit = n_launch if n_launch is not None else len(ordered)
            call_args: list[Any] = []
            tensor_args: list[torch.Tensor] = []
            for value in ordered[:limit]:
                if isinstance(value, torch.Tensor):
                    tensor_args.append(value)
                    call_args.append(value.contiguous())
                else:
                    call_args.append(value)

            if not tensor_args:
                return orig_fn(*args, **kwargs)

            if any(not tensor.is_cuda for tensor in tensor_args):
                return orig_fn(*args, **kwargs)

            first_device = tensor_args[0].device
            if any(tensor.device != first_device for tensor in tensor_args):
                return orig_fn(*args, **kwargs)

            return ext.launch(*call_args)
        except Exception:
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


def load_cast(
    cast_path: str,
    model_args: dict | None = None,
    no_kernels: bool = False,
    opt_level: str = "-O3",
    device: str | None = None,
    kernel_policy: str = "all",
):
    import torch

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

    if kernel_policy not in _KERNEL_POLICIES:
        raise ValueError(
            f"Unknown kernel policy '{kernel_policy}'. Expected one of: {', '.join(_KERNEL_POLICIES)}"
        )

    for op in manifest["ops"]:
        op_name = op["name"]
        kernel_cu = os.path.join(cache_dir, op["cuda_source"])

        if no_kernels:
            print(f"  [--no-kernels] Skipping kernel for {op_name}")
            continue

        if not torch.cuda.is_available():
            print(f"  [WARN] CUDA not available — skipping kernel for {op_name}")
            continue

        skip_reason = _kernel_policy_skip_reason(op_name, kernel_cu, kernel_policy)
        if skip_reason:
            print(f"  [kernel-policy={kernel_policy}] Skipping kernel for {op_name}: {skip_reason}")
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

                _spec = _ilu.spec_from_file_location(op_name, so_path)
                ext = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(ext)  # type: ignore[union-attr]
                print(f"  Loaded precompiled {op_name} ({gpu_sm})")
            except Exception as exc:
                print(f"  [WARN] Failed to load precompiled {op_name} ({gpu_sm}): {exc}")

        if ext is None:
            if so_rel:
                print(f"  [WARN] Precompiled .so not found for {gpu_sm}, falling back to JIT")
            if not os.path.exists(kernel_cu):
                print(f"  [WARN] No kernel.cu for {op_name}, skipping")
                continue
            try:
                ext = compile_kernel(kernel_cu, op_name, build_dir, opt_level=opt_level)
            except Exception as exc:
                print(f"  [WARN] Failed to prepare kernel for {op_name}, using native PyTorch: {exc}")
                continue

        # Generic patch: decode torch.nn.functional.<attr> from the op_name convention.
        if not op_name.startswith(_F_PREFIX):
            print(f"  [WARN] '{op_name}' does not follow torch_nn_functional_* convention, skipping patch")
            continue

        fn_attr = op_name[len(_F_PREFIX):]
        original = _ensure_functional_dispatch(fn_attr)
        if original is None:
            print(f"  [WARN] torch.nn.functional.{fn_attr} not found, skipping patch")
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
        )
        print(f"  Registered runtime patch torch.nn.functional.{fn_attr} → {op_name}")

    # 5. Load model class from model.py
    import importlib.util
    import sys

    model_py = os.path.join(cache_dir, "model.py")
    spec = importlib.util.spec_from_file_location("cast_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so inspect.getfile can resolve the module via
    # sys.modules — without this, inspect raises "is a built-in class".
    sys.modules["cast_model"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        del sys.modules["cast_model"]
        raise

    model_class_name = manifest.get("model_class", "")
    if model_class_name and hasattr(mod, model_class_name):
        ModelClass = getattr(mod, model_class_name)
    else:
        candidates = [
            v for v in vars(mod).values()
            if isinstance(v, type) and issubclass(v, nn.Module) and v.__module__ not in ("torch.nn", "builtins")
        ]
        if not candidates:
            raise RuntimeError("No model class found in model.py")
        ModelClass = candidates[-1]

    print(f"  Model class: {ModelClass.__name__}")

    # 6. Instantiate model
    model_init_args = manifest.get("model_init_args") or {}
    model_config_file = os.path.join(cache_dir, "model_config.json")
    if model_args:
        # CLI override: --model-args '{"model_type": "resnet", ...}'
        try:
            from transformers import AutoConfig
            cfg_dict = dict(model_args)
            model_type = cfg_dict.pop("model_type", None)
            config = AutoConfig.for_model(model_type, **cfg_dict)
            model = ModelClass(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model from --model-args: {e}") from e
    elif model_init_args:
        model = ModelClass(**model_init_args)
    elif os.path.exists(model_config_file):
        # HuggingFace model — load config from the bundled model_config.json
        try:
            from transformers import AutoConfig
            cfg_dict = json.load(open(model_config_file))
            model_type = cfg_dict.pop("model_type", None)
            # Rebuild id2label/label2id to be consistent with num_labels so
            # HuggingFace doesn't override num_labels with len(id2label).
            n = cfg_dict.get("num_labels")
            if n:
                cfg_dict["id2label"] = {str(i): f"LABEL_{i}" for i in range(n)}
                cfg_dict["label2id"] = {f"LABEL_{i}": i for i in range(n)}
            config = AutoConfig.for_model(model_type, **cfg_dict)
            model = ModelClass(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model from model_config.json: {e}") from e
    else:
        raise RuntimeError(
            "Cannot instantiate model: no model_init_args in manifest and "
            "no model_config.json bundled in the .cast file.\n"
            "Pass --model-args '{\"model_type\": ...}' or re-export with a "
            "model_config.json saved alongside model.py in the project."
        )

    # 7. Load weights
    import torch
    weight_file = os.path.join(cache_dir, manifest["weight_file"])
    print(f"  Loading weights from {manifest['weight_file']} ...")
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    runtime_model = CastModelRuntime(model, functional_patches)
    runtime_model.eval()
    if device:
        runtime_model = runtime_model.to(device)

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
