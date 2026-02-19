from __future__ import annotations

import base64
import shutil
import zipfile
from pathlib import Path


def _read_text(path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _decode_weights(project_dir: Path, weights_b64: str) -> None:
    if not weights_b64:
        return
    payload = base64.b64decode(weights_b64.encode("utf-8"))
    (project_dir / "weights.pt").write_bytes(payload)


def _decode_validation_zip(project_dir: Path, validation_b64: str) -> None:
    if not validation_b64:
        return

    payload = base64.b64decode(validation_b64.encode("utf-8"))
    uploads_dir = project_dir / ".uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = uploads_dir / "validation_upload.zip"
    zip_path.write_bytes(payload)

    validation_dir = project_dir / "data" / "validation"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(validation_dir)

    try:
        zip_path.unlink()
    except Exception:
        pass


def prepare_uploaded_assets(
    project_dir: Path,
    weights_b64_path: str = "",
    validation_b64_path: str = "",
    validation_name_path: str = "",
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    weights_b64 = _read_text(weights_b64_path)
    validation_b64 = _read_text(validation_b64_path)
    _ = _read_text(validation_name_path)

    if weights_b64:
        print("[benchmarking.assets] Decoding weights...")
        _decode_weights(project_dir, weights_b64)
    if validation_b64:
        print("[benchmarking.assets] Decoding/extracting validation zip...")
        _decode_validation_zip(project_dir, validation_b64)

