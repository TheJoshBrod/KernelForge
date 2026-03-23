from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import kernelforge
from run_cast import _launch_arity, compile_kernel, verify_checksums

_F_PREFIX = "torch_nn_functional_"
_DEFAULT_SKIP_OPS = {"torch_nn_functional_conv2d"}
_FUNCTIONAL_PARAM_NAMES: dict[str, list[str]] = {
    "adaptive_avg_pool2d": ["input", "output_size"],
    "batch_norm": ["input", "running_mean", "running_var", "weight", "bias", "training", "momentum", "eps"],
    "conv2d": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
    "linear": ["input", "weight", "bias"],
    "max_pool2d": ["input", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"],
    "relu": ["input", "inplace"],
}
_FUNCTIONAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "batch_norm": {"weight": None, "bias": None, "training": False, "momentum": 0.1, "eps": 1e-5},
    "conv2d": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    "linear": {"bias": None},
    "max_pool2d": {"stride": None, "padding": 0, "dilation": 1, "ceil_mode": False, "return_indices": False},
    "relu": {"inplace": False},
}


def _candidate_cast_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.getenv("RESNET50_CAST_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    default_export = (
        REPO_ROOT
        / "kernels"
        / "projects"
        / "oioioio - RTX 3050 Laptop GPU"
        / "exports"
        / "oioioio - RTX 3050 Laptop GPU.cast"
    )
    candidates.append(default_export)

    exports_root = REPO_ROOT / "kernels" / "projects"
    candidates.extend(sorted(exports_root.glob("*/exports/*.cast")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _find_resnet50_cast() -> Path:
    for path in _candidate_cast_paths():
        if path.exists():
            return path
    pytest.skip("No ResNet-50 .cast export found. Set RESNET50_CAST_PATH to a cast file.")


def _cache_dir_for(cast_path: Path) -> Path:
    cache_key = hashlib.sha256(cast_path.read_bytes()).hexdigest()
    return Path.home() / ".cache" / "cast" / cache_key


def _extract_cast(cast_path: Path) -> tuple[Path, dict[str, Any]]:
    cache_dir = _cache_dir_for(cast_path)
    with zipfile.ZipFile(cast_path) as zf:
        verify_checksums(zf)
        if not cache_dir.is_dir():
            zf.extractall(cache_dir)
        manifest = json.loads(zf.read("manifest.json"))
    return cache_dir, manifest


def _ensure_cuda_toolkit() -> None:
    candidates: list[Path] = []

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home))

    cudacxx = os.getenv("CUDACXX")
    if cudacxx:
        candidates.append(Path(cudacxx).resolve().parent.parent)

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(Path(nvcc_on_path).resolve().parent.parent)

    candidates.append(Path("/usr/local/cuda"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)

        nvcc = resolved / "bin" / "nvcc"
        if not nvcc.exists():
            continue

        os.environ["CUDA_HOME"] = str(resolved)
        os.environ.setdefault("CUDACXX", str(nvcc))

        path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
        nvcc_dir = str(nvcc.parent)
        if nvcc_dir not in path_entries:
            os.environ["PATH"] = nvcc_dir if not path_entries else nvcc_dir + os.pathsep + os.environ["PATH"]
        return

    pytest.skip("No CUDA toolkit with nvcc was found for JIT kernel benchmarking.")


def _capture_functional_calls(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    fn_attrs: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    captures: dict[str, dict[str, Any]] = {}
    originals: dict[str, Any] = {}

    for fn_attr in fn_attrs:
        original = getattr(F, fn_attr, None)
        if original is not None:
            originals[fn_attr] = original

    def make_wrapper(name: str, original):
        def wrapped(*args, **kwargs):
            captures.setdefault(name, {"args": args, "kwargs": kwargs})
            return original(*args, **kwargs)

        wrapped.__name__ = f"capture_{name}"
        return wrapped

    for fn_attr, original in originals.items():
        setattr(F, fn_attr, make_wrapper(fn_attr, original))

    try:
        with torch.no_grad():
            _ = model.run(pixel_values=pixel_values)
            if pixel_values.is_cuda:
                torch.cuda.synchronize()
    finally:
        for fn_attr, original in originals.items():
            setattr(F, fn_attr, original)

    return captures, originals


def _load_extension(
    cache_dir: Path,
    op: dict[str, Any],
    *,
    opt_level: str,
) -> tuple[Any, Path]:
    op_name = op["name"]
    kernel_cu = cache_dir / op["cuda_source"]
    gpu_sm = "sm_{0}{1}".format(*torch.cuda.get_device_capability())
    precompiled = op.get("precompiled", {})
    so_rel = precompiled.get(gpu_sm)
    so_path = cache_dir / so_rel if so_rel else None

    if so_path and so_path.exists():
        spec = importlib.util.spec_from_file_location(op_name, so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load precompiled module for {op_name}")
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        return ext, kernel_cu

    if not kernel_cu.exists():
        raise RuntimeError(f"No kernel source found for {op_name}")

    _ensure_cuda_toolkit()
    build_dir = cache_dir / "build_kernel_bench"
    build_dir.mkdir(parents=True, exist_ok=True)
    ext = compile_kernel(str(kernel_cu), op_name, str(build_dir), opt_level=opt_level)
    return ext, kernel_cu


def _prepare_launch_args(
    fn_attr: str,
    original,
    ext: Any,
    kernel_cu: Path,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[Any]:
    param_names = _FUNCTIONAL_PARAM_NAMES.get(fn_attr)
    if param_names is not None:
        resolved = {param_names[i]: value for i, value in enumerate(args) if i < len(param_names)}
        resolved.update(kwargs)
        defaults = _FUNCTIONAL_DEFAULTS.get(fn_attr, {})
        ordered = [resolved.get(name, defaults.get(name)) for name in param_names]
        if fn_attr == "max_pool2d" and len(ordered) >= 3 and ordered[2] is None:
            ordered[2] = ordered[1]
    else:
        try:
            signature = inspect.signature(original)
        except Exception:
            signature = None

        if signature is not None:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            ordered = [bound.arguments.get(name) for name in signature.parameters.keys()]
        else:
            ordered = list(args)

    if not ordered:
        ordered = list(args)

    launch_arity = _launch_arity(str(kernel_cu), ext)
    try:
        ext_arity = len(inspect.signature(ext.launch).parameters)
    except Exception:
        ext_arity = None

    if launch_arity is None:
        limit = ext_arity if ext_arity is not None else len(ordered)
    elif ext_arity is None:
        limit = launch_arity
    else:
        limit = max(launch_arity, ext_arity)

    launch_args: list[Any] = []
    for value in ordered[:limit]:
        if isinstance(value, torch.Tensor):
            launch_args.append(value.contiguous())
        else:
            launch_args.append(value)
    return launch_args


def _benchmark_callable(
    fn,
    *,
    device: str,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> float:
    with torch.no_grad():
        if device == "cuda":
            for _ in range(warmup_runs):
                fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(timed_runs):
                fn()
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / timed_runs

        for _ in range(warmup_runs):
            fn()
        start = time.perf_counter()
        for _ in range(timed_runs):
            fn()
        return (time.perf_counter() - start) / timed_runs * 1000.0


def _benchmark_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in args:
        if isinstance(value, torch.Tensor):
            parts.append(f"Tensor{tuple(value.shape)}:{value.dtype}")
        else:
            parts.append(repr(value))
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}=Tensor{tuple(value.shape)}:{value.dtype}")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def _assert_outputs_close(op_name: str, torch_output: Any, kernel_output: Any) -> None:
    assert isinstance(torch_output, torch.Tensor), f"{op_name}: expected tensor output from PyTorch"
    assert isinstance(kernel_output, torch.Tensor), f"{op_name}: expected tensor output from custom kernel"
    assert torch_output.shape == kernel_output.shape, f"{op_name}: output shapes differ"

    atol = 1e-3 if torch_output.dtype in (torch.float16, torch.bfloat16) else 1e-4
    rtol = 1e-3 if torch_output.dtype in (torch.float16, torch.bfloat16) else 1e-4
    torch.testing.assert_close(torch_output, kernel_output, atol=atol, rtol=rtol)


def test_cast_kernel_benchmarks_vs_pytorch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for kernel-vs-PyTorch benchmarking.")

    cast_path = _find_resnet50_cast()
    cache_dir, manifest = _extract_cast(cast_path)

    device = "cuda"
    pixel_values = torch.randn(1, 3, 224, 224, device=device)
    model = kernelforge.load(str(cast_path), device=device, no_kernels=True)

    benchable_ops = [
        op for op in manifest["ops"]
        if op["name"].startswith(_F_PREFIX) and op["name"] not in _DEFAULT_SKIP_OPS
    ]
    fn_attrs = [op["name"][len(_F_PREFIX):] for op in benchable_ops]

    captures, originals = _capture_functional_calls(model, pixel_values, fn_attrs)

    results: list[dict[str, Any]] = []
    for op in benchable_ops:
        op_name = op["name"]
        fn_attr = op_name[len(_F_PREFIX):]
        capture = captures.get(fn_attr)
        original = originals.get(fn_attr)
        if capture is None or original is None:
            continue

        ext, kernel_cu = _load_extension(cache_dir, op, opt_level="-O3")
        launch_args = _prepare_launch_args(
            fn_attr,
            original,
            ext,
            kernel_cu,
            capture["args"],
            capture["kwargs"],
        )

        with torch.no_grad():
            torch_output = original(*capture["args"], **capture["kwargs"])
            kernel_output = ext.launch(*launch_args)
            if device == "cuda":
                torch.cuda.synchronize()

        _assert_outputs_close(op_name, torch_output, kernel_output)

        torch_ms = _benchmark_callable(
            lambda: original(*capture["args"], **capture["kwargs"]),
            device=device,
        )
        kernel_ms = _benchmark_callable(
            lambda: ext.launch(*launch_args),
            device=device,
        )

        results.append(
            {
                "op_name": op_name,
                "signature": _benchmark_signature(capture["args"], capture["kwargs"]),
                "torch_ms": torch_ms,
                "kernel_ms": kernel_ms,
                "speedup": torch_ms / kernel_ms if kernel_ms else float("inf"),
            }
        )

    assert results, "No benchmarkable optimized kernels were found in the .cast file."

    print("\nKernel benchmark results:")
    for result in results:
        print(
            f"  {result['op_name']:<40}"
            f"kernel {result['kernel_ms']:>8.3f} ms | "
            f"torch {result['torch_ms']:>8.3f} ms | "
            f"speedup {result['speedup']:>6.2f}x"
        )
        print(f"    sample: {result['signature']}")
