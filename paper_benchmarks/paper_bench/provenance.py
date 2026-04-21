from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from . import HARNESS_VERSION

RELEVANT_ENV_VARS = [
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "CUDACXX",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDNN_LOGDEST_DBG",
    "CUDNN_LOGINFO_DBG",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "KFORGE_TARGET_DEVICE",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TORCH_COMPILE_DEBUG",
    "TOKENIZERS_PARALLELISM",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
]

PACKAGE_NAMES = ("torch", "transformers", "numpy", "triton")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_text(cmd: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        return (
            subprocess.check_output(
                cmd,
                cwd=str(cwd) if cwd else None,
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _run_lines(cmd: list[str], cwd: str | Path | None = None) -> list[str]:
    text = _run_text(cmd, cwd=cwd)
    if not text:
        return []
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_directory_files_stable(root: str | Path) -> Iterable[tuple[str, Path]]:
    base = Path(root)
    files = [
        child
        for child in base.rglob("*")
        if child.is_file()
    ]
    for child in sorted(files, key=lambda p: p.relative_to(base).as_posix()):
        yield child.relative_to(base).as_posix(), child


def sha256_directory(path: str | Path) -> str:
    base = Path(path)
    digest = hashlib.sha256()
    for relpath, child in _iter_directory_files_stable(base):
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        return sha256_directory(target)
    raise FileNotFoundError(target)


def safe_sha256_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return sha256_path(target)


def compute_suite_hash(suite_path: str | Path, workload_path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(suite_path).encode("utf-8"))
    digest.update(sha256_path(workload_path).encode("utf-8"))
    return digest.hexdigest()


def safe_compute_suite_hash(suite_path: str | Path | None, workload_path: str | Path | None) -> str | None:
    if not suite_path or not workload_path:
        return None
    suite_file = Path(suite_path)
    workload = Path(workload_path)
    if not suite_file.exists() or not workload.exists():
        return None
    return compute_suite_hash(suite_file, workload)


def hash_visible_paths(paths: Iterable[str | Path] | None) -> dict[str, str]:
    if not paths:
        return {}
    result: dict[str, str] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        target = Path(raw_path)
        if not target.exists():
            continue
        result[str(target)] = sha256_path(target)
    return dict(sorted(result.items()))


def collect_git_info(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    git_marker = root / ".git"
    if not git_marker.exists():
        return {
            "git_available": False,
            "git_commit": "unknown",
            "git_branch": None,
            "git_dirty": False,
            "git_dirty_summary": [],
            "git_untracked_summary": [],
            "git_status_short": [],
        }

    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run_text(["git", "branch", "--show-current"], cwd=root)
    status_lines = _run_lines(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root)
    dirty_summary = _run_lines(["git", "diff", "--stat", "--compact-summary", "HEAD"], cwd=root)
    untracked_summary = [line[3:] for line in status_lines if line.startswith("?? ")]
    git_available = commit is not None
    return {
        "git_available": git_available,
        "git_commit": commit or "unknown",
        "git_branch": branch or None,
        "git_dirty": bool(status_lines),
        "git_dirty_summary": dirty_summary,
        "git_untracked_summary": untracked_summary,
        "git_status_short": status_lines,
    }


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


def collect_driver_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    nvidia_query = _run_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if nvidia_query:
        info["nvidia_smi_query"] = nvidia_query
        first = nvidia_query[0].split(",")
        if len(first) >= 4:
            info["driver_version"] = first[3].strip()
    nvcc_version = _run_text(["nvcc", "--version"])
    if nvcc_version:
        info["nvcc_version"] = nvcc_version
    runtime_version = _run_text(["nvidia-smi"])
    if runtime_version:
        info["nvidia_smi_output"] = runtime_version
    return info


def _torch_device_properties_to_dict(index: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": str(getattr(props, "name", "")),
        "major": int(getattr(props, "major", 0)),
        "minor": int(getattr(props, "minor", 0)),
        "total_memory_bytes": int(getattr(props, "total_memory", 0)),
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
        "max_threads_per_block": int(getattr(props, "max_threads_per_block", 0)),
        "max_threads_per_multi_processor": int(getattr(props, "max_threads_per_multi_processor", 0)),
        "shared_memory_per_block_bytes": int(getattr(props, "shared_memory_per_block", 0)),
        "warp_size": int(getattr(props, "warp_size", 0)),
        "is_integrated": bool(getattr(props, "is_integrated", False)),
    }


def collect_gpu_info() -> tuple[list[str], int, list[dict[str, Any]], dict[str, Any]]:
    names: list[str] = []
    properties: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    if torch.cuda.is_available():
        count = int(torch.cuda.device_count())
        for idx in range(count):
            try:
                properties.append(_torch_device_properties_to_dict(idx))
                names.append(properties[-1]["name"] or f"cuda:{idx}")
            except Exception:
                names.append(f"cuda:{idx}")
        try:
            details["device_capability"] = str(torch.cuda.get_device_capability(0))
        except Exception:
            details["device_capability"] = None
        try:
            details["current_device"] = int(torch.cuda.current_device())
        except Exception:
            details["current_device"] = None
        return names, count, properties, details
    return [], 0, properties, details


def collect_relevant_env() -> dict[str, str]:
    return {key: os.environ[key] for key in RELEVANT_ENV_VARS if key in os.environ}


def collect_determinism_controls() -> dict[str, Any]:
    controls: dict[str, Any] = {}
    try:
        controls["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        controls["cuda_matmul_allow_tf32"] = None
    try:
        controls["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        controls["cudnn_allow_tf32"] = None
    try:
        controls["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        controls["cudnn_benchmark"] = None
    try:
        controls["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    except Exception:
        controls["cudnn_deterministic"] = None
    try:
        controls["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:
        controls["deterministic_algorithms_enabled"] = None
    try:
        controls["deterministic_debug_mode"] = str(torch.get_deterministic_debug_mode())
    except Exception:
        controls["deterministic_debug_mode"] = None
    return controls


def _default_paper_eligibility_issues(
    *,
    requested_paper_eligible: bool,
    git_available: bool,
    git_commit: str,
    workload_hash: str | None,
    model_id: str,
    model_path: str,
    model_path_hash: str | None,
    synthetic_workload: bool,
) -> list[str]:
    issues: list[str] = []
    if not requested_paper_eligible:
        issues.append("paper run not requested or explicitly downgraded")
    if not git_available:
        issues.append("git metadata unavailable")
    if not git_commit or git_commit == "unknown":
        issues.append("git commit unknown")
    if not workload_hash:
        issues.append("workload hash missing")
    if not model_id:
        issues.append("model_id missing")
    if not model_path:
        issues.append("model_path missing")
    if not model_path_hash:
        issues.append("model_path_hash missing")
    if synthetic_workload:
        issues.append("synthetic workload used")
    return issues


def collect_common_fields(
    *,
    repo_root: str | Path,
    model_id: str,
    model_path: str,
    model_config_path: str | None,
    suite_id: str,
    suite_path: str,
    workload_path: str,
    command_line: list[str],
    paper_eligible: bool,
    synthetic_workload: bool,
    cast_package_path: str | None = None,
    registry_path: str | None = None,
    exported_kernel_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    git_info = collect_git_info(repo_root)
    gpu_names, gpu_count, gpu_properties, gpu_details = collect_gpu_info()
    driver_info = collect_driver_info()
    driver_info.update(gpu_details)

    model_path_hash = safe_sha256_path(model_path)
    model_config_hash = safe_sha256_path(model_config_path)
    cast_package_hash = safe_sha256_path(cast_package_path)
    workload_hash = safe_sha256_path(workload_path)
    suite_hash = safe_compute_suite_hash(suite_path, workload_path)

    config_hashes = {
        key: value
        for key, value in {
            "suite": safe_sha256_path(suite_path),
            "registry": safe_sha256_path(registry_path),
            "model_config": model_config_hash,
        }.items()
        if value
    }
    package_versions = collect_package_versions()
    determinism_controls = collect_determinism_controls()
    try:
        cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.version() else None
    except Exception:
        cudnn_version = None

    paper_eligibility_issues = _default_paper_eligibility_issues(
        requested_paper_eligible=paper_eligible,
        git_available=git_info["git_available"],
        git_commit=git_info["git_commit"],
        workload_hash=workload_hash,
        model_id=model_id,
        model_path=model_path,
        model_path_hash=model_path_hash,
        synthetic_workload=synthetic_workload,
    )

    return {
        "benchmark_harness_version": HARNESS_VERSION,
        "timestamp_utc": utc_now_iso(),
        "git_available": git_info["git_available"],
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "git_branch": git_info["git_branch"],
        "git_dirty_summary": git_info["git_dirty_summary"],
        "git_untracked_summary": git_info["git_untracked_summary"],
        "command_line": command_line,
        "command_line_text": shlex.join(command_line),
        "hostname": socket.gethostname(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": cudnn_version,
        "gpu_names": gpu_names,
        "gpu_count": gpu_count,
        "gpu_properties": gpu_properties,
        "driver_info": driver_info,
        "relevant_env": collect_relevant_env(),
        "package_versions": package_versions,
        "determinism_controls": determinism_controls,
        "model_id": model_id,
        "model_path": str(model_path),
        "model_path_hash": model_path_hash,
        "model_config_path": str(model_config_path) if model_config_path else None,
        "model_config_hash": model_config_hash,
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "suite_hash": suite_hash or "",
        "workload_path": str(workload_path),
        "workload_hash": workload_hash,
        "config_hashes": config_hashes,
        "cast_package_path": str(cast_package_path) if cast_package_path else None,
        "cast_package_hash": cast_package_hash,
        "exported_kernel_hashes": hash_visible_paths(exported_kernel_paths),
        "paper_eligible": not paper_eligibility_issues,
        "paper_eligibility_issues": paper_eligibility_issues,
        "synthetic_workload": synthetic_workload,
    }


def build_environment_artifact_fields() -> dict[str, Any]:
    driver_info = collect_driver_info()
    return {
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_mps_available": bool(
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
        "torch_device_capability": (
            str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "nvcc_version": driver_info.get("nvcc_version"),
        "nvidia_smi_output": driver_info.get("nvidia_smi_output"),
    }
