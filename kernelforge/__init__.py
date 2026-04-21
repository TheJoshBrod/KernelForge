"""KernelForge runtime loading helpers."""

from __future__ import annotations

from kernelforge.run_cast import CastModelRuntime, get_runtime_stats, load_cast, reset_runtime_stats


def load(
    cast_path: str,
    *,
    device: str | None = None,
    model_args: dict | None = None,
    no_kernels: bool = False,
    opt_level: str = "-O3",
) -> CastModelRuntime:
    return load_cast(
        cast_path,
        model_args=model_args,
        no_kernels=no_kernels,
        opt_level=opt_level,
        device=device,
    )


__all__ = ["CastModelRuntime", "get_runtime_stats", "load", "load_cast", "reset_runtime_stats"]
