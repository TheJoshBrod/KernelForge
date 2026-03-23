#!/usr/bin/env python3
"""Autotuning .cast runtime that keeps only end-to-end beneficial kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch.nn as nn

import run_cast

_DEFAULT_MARGIN = 0.01


def ensure_cuda_toolkit_env() -> str | None:
    """Point CUDA_HOME/CUDACXX/PATH at a working nvcc if one is available."""

    candidates: list[Path] = []

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home))

    cudacxx = os.getenv("CUDACXX")
    if cudacxx:
        candidates.append(Path(cudacxx).expanduser().resolve().parent.parent)

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
        return str(resolved)

    return None


def _normalize_input(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, dict):
        return {str(key): _normalize_input(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_input(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _input_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    payload = {
        "args": _normalize_input(args),
        "kwargs": _normalize_input(kwargs),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _first_tensor_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    import torch

    def visit(value: Any) -> str | None:
        if isinstance(value, torch.Tensor):
            return value.device.type
        if isinstance(value, dict):
            for item in value.values():
                found = visit(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                found = visit(item)
                if found is not None:
                    return found
            return None
        return None

    found = visit(args)
    if found is not None:
        return found
    found = visit(kwargs)
    return found or "cpu"


def _flatten_output(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return ("tensor", value)
    if hasattr(value, "to_tuple") and callable(value.to_tuple):
        return ("tuple_like", tuple(_flatten_output(item) for item in value.to_tuple()))
    if isinstance(value, dict):
        return ("dict", tuple((str(key), _flatten_output(val)) for key, val in sorted(value.items(), key=lambda item: str(item[0]))))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_flatten_output(item) for item in value))
    return ("value", value)


def _outputs_match(reference: Any, candidate: Any) -> bool:
    import torch

    ref = _flatten_output(reference)
    cand = _flatten_output(candidate)

    def compare(left: Any, right: Any) -> bool:
        if left[0] != right[0]:
            return False
        kind = left[0]
        if kind == "tensor":
            try:
                atol = 1e-3 if left[1].dtype in (torch.float16, torch.bfloat16) else 1e-4
                rtol = 1e-3 if left[1].dtype in (torch.float16, torch.bfloat16) else 1e-4
                torch.testing.assert_close(left[1], right[1], atol=atol, rtol=rtol)
                return True
            except Exception:
                return False
        if kind in {"tuple_like", "seq"}:
            if len(left[1]) != len(right[1]):
                return False
            return all(compare(l_item, r_item) for l_item, r_item in zip(left[1], right[1]))
        if kind == "dict":
            if len(left[1]) != len(right[1]):
                return False
            return all(
                l_key == r_key and compare(l_val, r_val)
                for (l_key, l_val), (r_key, r_val) in zip(left[1], right[1])
            )
        return left[1] == right[1]

    return compare(ref, cand)


def _cast_cache_key(cast_path: str) -> str:
    cast_bytes = Path(cast_path).read_bytes()
    return hashlib.sha256(cast_bytes).hexdigest()


class AutotunedCastModelRuntime(nn.Module):
    """A runtime wrapper that benchmarks patch subsets and keeps only the winners."""

    def __init__(
        self,
        runtime_model: run_cast.CastModelRuntime,
        *,
        cast_path: str,
        opt_level: str,
        base_policy: str,
        warmup_runs: int,
        timed_runs: int,
        improvement_margin: float,
        cache_enabled: bool,
    ) -> None:
        if not isinstance(runtime_model, run_cast.CastModelRuntime):
            raise TypeError("runtime_model must be a CastModelRuntime")
        super().__init__()
        self.model = runtime_model
        self._all_patches = dict(runtime_model._cast_functional_patches)
        self._active_patch_names: list[str] = list(self._all_patches.keys())
        self._signature_patch_names: dict[str, list[str]] = {}
        self._signature_patch_maps: dict[str, dict[str, Any]] = {}
        self._opt_level = opt_level
        self._base_policy = base_policy
        self._warmup_runs = warmup_runs
        self._timed_runs = timed_runs
        self._improvement_margin = improvement_margin
        self._cache_enabled = cache_enabled
        self._cast_key = _cast_cache_key(cast_path)
        self._cache_path = self._build_cache_path(self._cast_key)
        self._cache_data = self._load_cache()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = self._modules.get("model")
            if model is None:
                raise
            return getattr(model, name)

    @staticmethod
    def _build_cache_path(cast_key: str) -> Path:
        cache_root = Path.home() / ".cache" / "cast_autotune"
        gpu_name = "cpu"
        if os.getenv("CUDA_VISIBLE_DEVICES", "") == "":
            pass
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        except Exception:
            pass
        return cache_root / f"{cast_key}_{gpu_name}.json"

    def _load_cache(self) -> dict[str, Any]:
        if not self._cache_enabled or not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text())
        except Exception:
            return {}

    def _persist_cache(self) -> None:
        if not self._cache_enabled:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._cache_data, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile("w", dir=self._cache_path.parent, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        temp_path.replace(self._cache_path)

    def _select_patch_names(self, names: list[str]) -> dict[str, Any]:
        return {name: self._all_patches[name] for name in names if name in self._all_patches}

    def _set_active_patch_names(self, names: list[str]) -> None:
        self._active_patch_names = list(names)
        self.model._cast_functional_patches = self._select_patch_names(names)

    def _run_once(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return self.model.run(*args, **kwargs)

    def _benchmark_patch_names(self, patch_names: list[str], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, float]:
        import torch

        self._set_active_patch_names(patch_names)
        device_type = _first_tensor_device(args, kwargs)

        with torch.inference_mode():
            output = None
            for _ in range(self._warmup_runs):
                output = self._run_once(args, kwargs)
                if device_type == "cuda":
                    torch.cuda.synchronize()

            if device_type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(self._timed_runs):
                    output = self._run_once(args, kwargs)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end) / self._timed_runs
            else:
                t0 = time.perf_counter()
                for _ in range(self._timed_runs):
                    output = self._run_once(args, kwargs)
                elapsed_ms = (time.perf_counter() - t0) / self._timed_runs * 1000.0

        return output, elapsed_ms

    def autotune(self, *args: Any, **kwargs: Any) -> list[str]:
        signature = _input_signature(args, kwargs)
        cached = self._signature_patch_names.get(signature)
        if cached is not None:
            return list(cached)

        cache_entry = self._cache_data.get(signature)
        if isinstance(cache_entry, list):
            names = [name for name in cache_entry if name in self._all_patches]
            self._signature_patch_names[signature] = names
            self._signature_patch_maps[signature] = self._select_patch_names(names)
            self._set_active_patch_names(names)
            return list(names)

        baseline_output, baseline_ms = self._benchmark_patch_names([], args, kwargs)

        current_names: list[str] = []
        current_ms = baseline_ms
        current_output = baseline_output
        remaining = list(self._all_patches.keys())

        while remaining:
            best_candidate_name: str | None = None
            best_candidate_output = None
            best_candidate_ms = current_ms

            for candidate in remaining:
                candidate_names = current_names + [candidate]
                candidate_output, candidate_ms = self._benchmark_patch_names(candidate_names, args, kwargs)
                if not _outputs_match(baseline_output, candidate_output):
                    continue
                if candidate_ms < best_candidate_ms:
                    best_candidate_name = candidate
                    best_candidate_output = candidate_output
                    best_candidate_ms = candidate_ms

            if best_candidate_name is None:
                break
            if best_candidate_ms >= current_ms * (1.0 - self._improvement_margin):
                break

            current_names.append(best_candidate_name)
            remaining.remove(best_candidate_name)
            current_ms = best_candidate_ms
            current_output = best_candidate_output

        improved = True
        while improved and current_names:
            improved = False
            for candidate in list(current_names):
                trial_names = [name for name in current_names if name != candidate]
                trial_output, trial_ms = self._benchmark_patch_names(trial_names, args, kwargs)
                if not _outputs_match(baseline_output, trial_output):
                    continue
                if trial_ms < current_ms * (1.0 - self._improvement_margin):
                    current_names = trial_names
                    current_ms = trial_ms
                    current_output = trial_output
                    improved = True
                    break

        if not _outputs_match(baseline_output, current_output):
            current_names = []

        self._signature_patch_names[signature] = list(current_names)
        self._signature_patch_maps[signature] = self._select_patch_names(current_names)
        self._cache_data[signature] = list(current_names)
        self._set_active_patch_names(current_names)
        self._persist_cache()
        return list(current_names)

    def active_patch_names(self, *args: Any, **kwargs: Any) -> list[str]:
        if args or kwargs:
            return self.autotune(*args, **kwargs)
        return list(self._active_patch_names)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        signature = _input_signature(args, kwargs)
        patch_names = self._signature_patch_names.get(signature)
        if patch_names is None:
            patch_names = self.autotune(*args, **kwargs)
        self._set_active_patch_names(patch_names)
        return self.model.run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)


def load_cast(
    cast_path: str,
    *,
    device: str | None = None,
    model_args: dict | None = None,
    opt_level: str = "-O3",
    base_policy: str = "skip_aten",
    warmup_runs: int = 3,
    timed_runs: int = 5,
    improvement_margin: float = _DEFAULT_MARGIN,
    cache_enabled: bool = True,
) -> AutotunedCastModelRuntime:
    ensure_cuda_toolkit_env()

    runtime_model = run_cast.load_cast(
        cast_path,
        model_args=model_args,
        no_kernels=False,
        opt_level=opt_level,
        device=device,
        kernel_policy=base_policy,
    )

    return AutotunedCastModelRuntime(
        runtime_model,
        cast_path=cast_path,
        opt_level=opt_level,
        base_policy=base_policy,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        improvement_margin=improvement_margin,
        cache_enabled=cache_enabled,
    )


def _parse_shape(raw: str) -> tuple[int, ...]:
    return run_cast._parse_shape(raw)


def _benchmark_loaded_model(model: AutotunedCastModelRuntime, device: str, runs: int, input_shape: tuple[int, ...]) -> None:
    import torch

    dummy = torch.randn(*input_shape, device=device)
    tuned = model.autotune(dummy)
    print(f"Autotuned kernels: {tuned}")
    run_cast._benchmark_loaded_model(model, device, runs, input_shape)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Run a KernelForge .cast inference package with end-to-end autotuning")
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
        help='JSON config to instantiate the model, e.g. \'{"model_type":"resnet",...}\'.',
    )
    parser.add_argument(
        "--opt-level",
        default="-O3",
        choices=["-O0", "-O1", "-O2", "-O3"],
        help="NVCC optimisation level for JIT compilation (default: -O3).",
    )
    parser.add_argument(
        "--base-policy",
        default="skip_aten",
        choices=list(run_cast._KERNEL_POLICIES),
        help="Candidate kernel policy before autotuning (default: skip_aten).",
    )
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warmup runs for autotuning benchmarks.")
    parser.add_argument("--timed-runs", type=int, default=5, help="Timed runs for autotuning benchmarks.")
    parser.add_argument(
        "--improvement-margin",
        type=float,
        default=_DEFAULT_MARGIN,
        help="Minimum fractional improvement required to keep a kernel (default: 0.01).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable autotune cache persistence.")
    parser.add_argument("--benchmark", action="store_true", help="Run a dummy-input benchmark after loading the model.")
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
        device=args.device,
        model_args=extra_model_args,
        opt_level=args.opt_level,
        base_policy=args.base_policy,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        improvement_margin=args.improvement_margin,
        cache_enabled=not args.no_cache,
    )
    print(f"\nModel ready on {args.device}")
    print(f"Candidate policy: {args.base_policy}")

    if args.benchmark:
        input_shape = _parse_shape(args.input_shape)
        _benchmark_loaded_model(model, args.device, args.runs, input_shape)


if __name__ == "__main__":
    main()
