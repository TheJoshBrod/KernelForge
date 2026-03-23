from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

import run_cast
import run_cast_autotune


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


def _benchmark_model(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    warmup_runs: int = 3,
    timed_runs: int = 5,
) -> tuple[object, float]:
    device = pixel_values.device.type

    with torch.inference_mode():
        output = None
        for _ in range(warmup_runs):
            output = model.run(pixel_values=pixel_values)
            if device == "cuda":
                torch.cuda.synchronize()

        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(timed_runs):
                output = model.run(pixel_values=pixel_values)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end) / timed_runs
        else:
            import time

            t0 = time.perf_counter()
            for _ in range(timed_runs):
                output = model.run(pixel_values=pixel_values)
            elapsed_ms = (time.perf_counter() - t0) / timed_runs * 1000.0

    return output, elapsed_ms


def test_run_cast_autotune_resnet50_speed() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the autotuned cast benchmark.")

    cast_path = _find_resnet50_cast()
    device = "cuda"
    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    run_cast_autotune.ensure_cuda_toolkit_env()

    pytorch_model = run_cast.load_cast(str(cast_path), device=device, no_kernels=True)
    full_model = run_cast.load_cast(str(cast_path), device=device, opt_level="-O3")
    autotuned_model = run_cast_autotune.load_cast(
        str(cast_path),
        device=device,
        opt_level="-O3",
        base_policy="skip_aten",
        warmup_runs=3,
        timed_runs=5,
        cache_enabled=False,
    )

    tuned_patch_names = autotuned_model.autotune(pixel_values=pixel_values)

    pytorch_output, pytorch_ms = _benchmark_model(pytorch_model, pixel_values)
    full_output, full_ms = _benchmark_model(full_model, pixel_values)
    autotuned_output, autotuned_ms = _benchmark_model(autotuned_model, pixel_values)

    pytorch_logits = pytorch_output.logits if hasattr(pytorch_output, "logits") else pytorch_output
    full_logits = full_output.logits if hasattr(full_output, "logits") else full_output
    autotuned_logits = autotuned_output.logits if hasattr(autotuned_output, "logits") else autotuned_output

    assert isinstance(pytorch_logits, torch.Tensor)
    assert isinstance(full_logits, torch.Tensor)
    assert isinstance(autotuned_logits, torch.Tensor)
    assert pytorch_logits.shape == full_logits.shape == autotuned_logits.shape
    assert torch.allclose(pytorch_logits, full_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(pytorch_logits, autotuned_logits, atol=1e-3, rtol=1e-3)

    print("\nAutotuned kernels         :", tuned_patch_names)
    print(f"PyTorch fallback latency  : {pytorch_ms:.2f} ms")
    print(f"run_cast full latency     : {full_ms:.2f} ms")
    print(f"autotuned runtime latency : {autotuned_ms:.2f} ms")
