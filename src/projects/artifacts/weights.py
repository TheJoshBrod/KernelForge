"""Checkpoint artifact ingestion helpers (.pth/.pt)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _normalize_name(name: str | None) -> str:
    candidate = (name or "checkpoint.pth").strip()
    base = Path(candidate).name
    if not base.endswith((".pth", ".pt")):
        base = f"{base}.pth"
    return base


def ingest_pth_bytes(project_dir: str | Path, name: str | None, payload: bytes) -> dict:
    pdir = Path(project_dir)
    safe_name = _normalize_name(name)
    digest = hashlib.sha256(payload).hexdigest()
    ext = ".pth" if safe_name.endswith(".pth") else ".pt"

    weights_root = pdir / "artifacts" / "weights"
    weights_root.mkdir(parents=True, exist_ok=True)

    stored_name = f"{digest}{ext}"
    stored_path = weights_root / stored_name
    if not stored_path.exists():
        stored_path.write_bytes(payload)

    return {
        "id": digest[:16],
        "name": safe_name,
        "sha256": digest,
        "size_bytes": len(payload),
        "relpath": str(stored_path.relative_to(pdir)),
        "source": "upload",
    }

