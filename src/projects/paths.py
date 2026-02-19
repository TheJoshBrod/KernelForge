"""Canonical runtime project paths."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    env = os.environ.get("CGINS_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "kernels" / "private"


def projects_root(create: bool = True) -> Path:
    root = data_root() / "projects"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def legacy_projects_root() -> Path:
    return repo_root() / "projects"


def project_dir(name: str, *, create_root: bool = True, prefer_existing: bool = True) -> Path:
    primary = projects_root(create=create_root) / name
    legacy = legacy_projects_root() / name
    if prefer_existing:
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
    return primary

