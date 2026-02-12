from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from .constants import (
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_URL,
    MODELS_CACHE_DIR,
    env_default_model_sha256,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        with out_path.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def ensure_default_model() -> Path:
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_CACHE_DIR / DEFAULT_MODEL_FILENAME
    checksum_path = MODELS_CACHE_DIR / f"{DEFAULT_MODEL_FILENAME}.sha256.json"

    expected = env_default_model_sha256()

    if model_path.exists():
        got = _sha256(model_path)
        if expected and got == expected:
            return model_path
        if expected and got != expected:
            model_path.unlink(missing_ok=True)

    _download(DEFAULT_MODEL_URL, model_path)
    got = _sha256(model_path)

    if expected and got != expected:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {DEFAULT_MODEL_NAME}: expected {expected}, got {got}"
        )

    checksum_path.write_text(
        json.dumps(
            {
                "name": DEFAULT_MODEL_NAME,
                "filename": DEFAULT_MODEL_FILENAME,
                "url": DEFAULT_MODEL_URL,
                "sha256": got,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path
