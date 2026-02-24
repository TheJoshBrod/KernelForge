"""Utilities for profiling collectors."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(cmd: list[str], timeout: float = 1.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except Exception:
        return default


def is_rocm_runtime() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        if getattr(torch.version, "hip", None):
            return True
    except Exception:
        pass

    try:
        cfg = torch.__config__.show()
        # Check for USE_ROCM=ON specifically to avoid false positives from strings like LIBKINETO_NOROCTRACER or USE_ROCM=OFF that contain "rocm" but mean ROCm is disabled.
        if "USE_ROCM=ON" in cfg:
            return True
    except Exception:
        pass

    return False

