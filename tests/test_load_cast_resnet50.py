from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

import kernelforge


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


def _run_once(model: torch.nn.Module, pixel_values: torch.Tensor) -> object:
    device = pixel_values.device.type

    output = model.run(pixel_values=pixel_values)
    if device == "cuda":
        torch.cuda.synchronize()
    return output


def _benchmark_models_alternating(
    pytorch_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    warmup_runs: int = 3,
    timed_runs: int = 5,
) -> tuple[object, float, object, float]:
    pytorch_total_ms = 0.0
    optimized_total_ms = 0.0

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _run_once(pytorch_model, pixel_values)
            _run_once(optimized_model, pixel_values)

        pytorch_output = None
        optimized_output = None
        for _ in range(timed_runs):
            start = time.perf_counter()
            pytorch_output = _run_once(pytorch_model, pixel_values)
            pytorch_total_ms += (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            optimized_output = _run_once(optimized_model, pixel_values)
            optimized_total_ms += (time.perf_counter() - start) * 1000.0

    return (
        pytorch_output,
        pytorch_total_ms / timed_runs,
        optimized_output,
        optimized_total_ms / timed_runs,
    )


def test_load_cast_resnet50_benchmark() -> None:
    cast_path = _find_resnet50_cast()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    optimized_model = kernelforge.load(str(cast_path), device=device, opt_level="-O3")
    pytorch_model = kernelforge.load(str(cast_path), device=device, no_kernels=True)

    assert isinstance(optimized_model, torch.nn.Module)
    assert isinstance(pytorch_model, torch.nn.Module)
    assert optimized_model.training is False
    assert pytorch_model.training is False
    assert hasattr(optimized_model, "run")
    assert hasattr(pytorch_model, "run")

    pytorch_output, pytorch_ms, optimized_output, optimized_ms = _benchmark_models_alternating(
        pytorch_model,
        optimized_model,
        pixel_values,
        warmup_runs=3,
        timed_runs=5,
    )

    optimized_logits = optimized_output.logits if hasattr(optimized_output, "logits") else optimized_output
    pytorch_logits = pytorch_output.logits if hasattr(pytorch_output, "logits") else pytorch_output

    assert isinstance(optimized_logits, torch.Tensor)
    assert isinstance(pytorch_logits, torch.Tensor)
    assert optimized_logits.device.type == device
    assert pytorch_logits.device.type == device
    assert optimized_logits.shape == pytorch_logits.shape
    assert optimized_logits.shape[0] == 1
    assert optimized_logits.ndim == 2
    assert optimized_logits.shape[1] > 0

    print(f"\n.cast runtime average latency : {optimized_ms:.2f} ms")
    print(f"PyTorch average latency      : {pytorch_ms:.2f} ms")
