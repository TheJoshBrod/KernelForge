from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

from src.optimizer.config.settings import MIN_CUDA_VERSION


def _extract_signature(code: str) -> str:
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", code)
    if not match:
        raise ValueError("Could not find 'launch' signature in kernel.cu")
    return match.group(1) + ";"


def _detect_cuda_home() -> str:
    from src.optimizer.config.settings import _detect_cuda_home as _settings_detect_cuda_home
    return _settings_detect_cuda_home()


def _check_nvcc_version(cuda_home: str) -> None:
    """Raise RuntimeError if the installed nvcc is below the minimum supported version."""
    import subprocess

    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.exists():
        nvcc_path = shutil.which("nvcc")
        if not nvcc_path:
            raise RuntimeError(
                f"nvcc not found in {cuda_home}/bin or on PATH. "
                "Ensure CUDA is installed and CUDA_HOME is set correctly."
            )
        nvcc = Path(nvcc_path)

    try:
        out = subprocess.check_output([str(nvcc), "--version"], stderr=subprocess.STDOUT).decode()
    except Exception as e:
        raise RuntimeError(f"Failed to run nvcc --version: {e}") from e

    match = re.search(r"release (\d+)\.(\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse nvcc version from output:\n{out}")

    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"CUDA {major}.{minor} is too old. "
            f"Kernel Forge requires CUDA {MIN_CUDA_VERSION[0]}.{MIN_CUDA_VERSION[1]} or newer. "
            f"Detected nvcc at: {nvcc}"
        )


_cuda_env_ready: bool = False


def ensure_cuda_env() -> None:
    global _cuda_env_ready
    if _cuda_env_ready:
        return
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = _detect_cuda_home()
    _check_nvcc_version(os.environ["CUDA_HOME"])
    python_bin = os.path.dirname(sys.executable)
    if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH','')}"
    _cuda_env_ready = True


def target_device() -> str:
    value = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if value in {"gpu", "cuda"}:
        return "cuda"
    if value == "mps":
        return "mps"
    if value == "cpu":
        return "cpu"
    return "cuda"


_CUDA_FLAGS = [
    "-O3",
    "-use_fast_math",
    "--expt-relaxed-constexpr",
]


def load_kernel(kernel_dir: Path, name: str | None = None, build_dir: Path | None = None) -> Any:
    kernel_dir = Path(kernel_dir)
    kernel_path = kernel_dir / "kernel.cu"
    if not kernel_path.exists():
        raise FileNotFoundError(f"kernel.cu not found in {kernel_dir}")

    code = kernel_path.read_text(encoding="utf-8")
    signature = _extract_signature(code)

    module_name = name or f"kforge_{kernel_dir.name}"
    build_dir = build_dir or (kernel_dir / ".build")
    build_dir.mkdir(parents=True, exist_ok=True)

    target_device = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if target_device in {"gpu", "cuda"} or target_device == "":
        ensure_cuda_env()
        module = load_inline(
            name=module_name,
            cpp_sources=signature,
            cuda_sources=code,
            functions=["launch"],
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cuda_cflags=_CUDA_FLAGS,
        )
    else:
        module = load_inline(
            name=module_name,
            cpp_sources=code,
            functions=["launch"],
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=False,
        )
    return module


def compile_code_string(code: str, name: str, build_dir: str, verbose: bool = False) -> Any:
    """
    Compiles CUDA code string directly.
    """
    signature = _extract_signature(code)
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    
    device = target_device()
    if device in {"gpu", "cuda"}:
        ensure_cuda_env()
        return load_inline(
            name=name,
            cpp_sources=signature,
            cuda_sources=code,
            functions=["launch"],
            build_directory=build_dir,
            verbose=verbose,
            with_cuda=True,
            extra_cuda_cflags=_CUDA_FLAGS,
        )
    else:
        return load_inline(
            name=name,
            cpp_sources=code,
            functions=["launch"],
            build_directory=build_dir,
            verbose=verbose,
            with_cuda=False,
        )


def load_export(export_path: str | Path, extract_dir: str | Path | None = None) -> dict[str, Any]:
    export_path = Path(export_path)
    if not export_path.exists():
        raise FileNotFoundError(export_path)

    temp_dir = None
    if export_path.suffix == ".zip":
        if extract_dir:
            extract_root = Path(extract_dir)
            extract_root.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="kforge_export_")
            extract_root = Path(temp_dir.name)
        with zipfile.ZipFile(export_path, "r") as zipf:
            zipf.extractall(extract_root)
    else:
        extract_root = export_path

    kernels_root = extract_root / "kernels"
    if not kernels_root.exists():
        raise FileNotFoundError(f"kernels/ not found in {extract_root}")

    modules: dict[str, Any] = {}
    for kernel_path in kernels_root.rglob("kernel.cu"):
        op_dir = kernel_path.parent
        op_name = op_dir.name
        modules[op_name] = load_kernel(op_dir, name=f"kforge_{op_name}")

    if temp_dir:
        temp_dir.cleanup()

    return modules
