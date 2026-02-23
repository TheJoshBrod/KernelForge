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
        cfg_l = cfg.lower()
        # Check for USE_ROCM=ON (not just "rocm" which matches USE_ROCM=OFF)
        if "use_rocm=on" in cfg_l.replace(" ", ""):
            return True
        # Check for HIP compiler or runtime indicators
        if "hip" in cfg_l:
            return True
        return False
    except Exception:
        return False

