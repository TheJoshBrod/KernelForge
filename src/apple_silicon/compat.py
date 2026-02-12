from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .constants import LLAMA_CPP_PINNED_COMMIT
from .types import DeviceProfile, ModelProfile


def get_llamacpp_commit(llamacpp_root: Path) -> str:
    if not llamacpp_root.exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(llamacpp_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def ensure_llamacpp_commit(llamacpp_root: Path, *, strict: bool = True) -> tuple[bool, str]:
    actual = get_llamacpp_commit(llamacpp_root)
    if not actual:
        return False, "Unable to read local llama.cpp git commit"
    if actual == LLAMA_CPP_PINNED_COMMIT:
        return True, "llama.cpp commit matches pinned version"
    msg = (
        f"llama.cpp commit mismatch: expected {LLAMA_CPP_PINNED_COMMIT}, "
        f"found {actual}. Run bootstrap to realign toolchain."
    )
    if strict:
        return False, msg
    return True, msg


def assert_supported_device(device: DeviceProfile) -> None:
    if not device.is_apple_silicon:
        raise RuntimeError("Apple Silicon is required for this optimization path")
    if not device.metal_supported:
        raise RuntimeError("Metal support was not detected on this machine")


def chip_family(chip_name: str) -> str:
    text = (chip_name or "").strip().lower()
    m = re.search(r"\bm(\d+)\b", text)
    if m:
        return f"m{m.group(1)}"
    if not text:
        return ""
    cleaned = text.replace("apple", "").strip()
    return cleaned.split(" ")[0] if cleaned else text


def os_minor(version: str) -> str:
    text = (version or "").strip()
    if not text:
        return ""
    parts = text.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def quant_family(quant: str) -> str:
    text = (quant or "").strip().lower()
    if not text:
        return ""
    m = re.match(r"(q\d+_k)", text)
    if m:
        return m.group(1)
    return text


def manifest_compatible(
    manifest: dict[str, Any],
    *,
    device: DeviceProfile,
    model: ModelProfile,
    llamacpp_commit: str,
) -> tuple[bool, str]:
    expected_commit = str(manifest.get("llamacpp_commit", ""))
    if expected_commit and expected_commit != llamacpp_commit:
        return False, "Pack commit does not match local llama.cpp commit"

    compat = manifest.get("compatibility") or {}
    reuse_policy = str(compat.get("reuse_policy", "machine")).strip().lower() or "machine"
    expected_fingerprint = str(compat.get("device_fingerprint", "")).strip()
    expected_chip_family = str(compat.get("chip_family", "")).strip().lower()
    expected_os_compat = str(compat.get("os_compat", "")).strip().lower()
    expected_model_arch = str(compat.get("model_arch", "")).strip().lower()
    expected_quant_family = str(compat.get("quant_family", "")).strip().lower()

    current_chip_family = chip_family(device.chip)
    current_os_minor = os_minor(device.macos_version).lower()
    current_model_arch = (model.architecture or "").strip().lower()
    current_quant_family = quant_family(model.quant)

    if reuse_policy == "machine":
        if expected_fingerprint and expected_fingerprint != device.fingerprint:
            return False, "Pack device fingerprint does not match this machine"
    elif reuse_policy == "chip_family":
        if expected_chip_family and expected_chip_family != current_chip_family:
            return False, "Pack chip family does not match this machine"
    elif reuse_policy == "chip_family+os_minor":
        if expected_chip_family and expected_chip_family != current_chip_family:
            return False, "Pack chip family does not match this machine"
        if expected_os_compat and expected_os_compat != current_os_minor:
            return False, "Pack OS minor version is incompatible"
    else:
        return False, f"Unsupported reuse policy: {reuse_policy}"

    if expected_model_arch and expected_model_arch != current_model_arch:
        return False, "Pack model architecture is incompatible"
    if expected_quant_family and expected_quant_family != current_quant_family:
        return False, "Pack quant family is incompatible"

    expected_model = str(manifest.get("model_sha256", ""))
    if expected_model and expected_model != model.sha256 and reuse_policy == "machine":
        return False, "Pack model hash does not match selected GGUF"

    return True, "ok"


def load_manifest(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
