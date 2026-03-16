from pathlib import Path

from src.projects.paths import repo_root


BASELINE_ROOT = repo_root() / "kernels" / "generated" / "individual_op_kernels"


def _normalize_op_name(op_name: str) -> str:
    return op_name.replace(".", "_").replace("/", "_")


def baseline_kernel_path(op_name: str) -> Path:
    op_key = _normalize_op_name(op_name)
    cu_path = BASELINE_ROOT / op_key / "kernel.cu"
    if cu_path.exists():
        return cu_path
    py_path = BASELINE_ROOT / op_key / "kernel.py"
    if py_path.exists():
        return py_path
    return cu_path  # default to .cu even if it doesn't exist


def has_baseline_kernel(op_name: str) -> bool:
    return baseline_kernel_path(op_name).exists()


def load_baseline_kernel(op_name: str) -> str | None:
    path = baseline_kernel_path(op_name)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def template_for_prompt(op_name: str, max_chars: int = 8000) -> str | None:
    kernel = load_baseline_kernel(op_name)
    if not kernel:
        return None
    if len(kernel) <= max_chars:
        return kernel
    return kernel[:max_chars] + "\n// [TRUNCATED TEMPLATE]\n"
