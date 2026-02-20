from __future__ import annotations

import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


def _extract_signature(code: str) -> str:
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", code)
    if not match:
        raise ValueError("Could not find 'launch' signature in kernel.cu")
    return match.group(1) + ";"


def ensure_cuda_env() -> None:
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
    python_bin = os.path.dirname(sys.executable)
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH','')}"


def target_device() -> str:
    value = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if value in {"gpu", "cuda"}:
        return "cuda"
    if value == "mps":
        return "mps"
    if value == "cpu":
        return "cpu"
    return "cuda"


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
        _ensure_cuda_env()
        module = load_inline(
            name=module_name,
            cpp_sources=signature,
            cuda_sources=code,
            functions=["launch"],
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
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
