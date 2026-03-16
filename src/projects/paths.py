"""Canonical runtime project paths."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("KFORGE_REPO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _normalize_dir(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _data_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("KFORGE_DATA_DIR", "").strip()
    if env_dir:
        candidates.append(_normalize_dir(env_dir))
    candidates.append(repo_root() / "kernels" / "private")
    candidates.append(repo_root() / "kernels")
    return candidates


def data_root() -> Path:
    for candidate in _data_root_candidates():
        if candidate.exists():
            return candidate
    if env_dir := os.environ.get("KFORGE_DATA_DIR", "").strip():
        return _normalize_dir(env_dir)
    return repo_root() / "kernels" / "private"


def config_path() -> Path:
    override = os.environ.get("KFORGE_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return data_root() / "config.json"


def ensure_data_root() -> None:
    path = data_root()
    path.mkdir(parents=True, exist_ok=True)


def cache_root(create: bool = True) -> Path:
    cache_dir = data_root() / "cache"
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def data_tmp_root(create: bool = True) -> Path:
    tmp_dir = data_root() / "tmp"
    if create:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def data_projects_root(create: bool = True) -> Path:
    root = data_root() / "projects"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def data_catalog_path() -> Path:
    return data_projects_root(create=False) / "catalog.db"


def data_logs_root(create: bool = True) -> Path:
    logs_dir = data_root() / "logs"
    if create:
        logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def projects_root(create: bool = True) -> Path:
    root = data_projects_root(create=create)
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
