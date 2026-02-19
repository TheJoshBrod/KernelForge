from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_dir_for_name(project_name: str, create: bool = False) -> Path:
    root = repo_root()
    private = root / "kernels" / "private" / "projects" / project_name
    primary = root / "kernels" / "projects" / project_name
    legacy = root / "projects" / project_name

    if private.exists():
        return private
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    if create:
        private.mkdir(parents=True, exist_ok=True)
    return private


def find_latest_optimized_dir(project_name: str) -> Path | None:
    root = repo_root()
    candidates = [
        root / "kernels" / "optimized",
        root / "kernels" / "private" / "optimized",
    ]

    suffix = f"_{project_name}"
    matches: list[Path] = []
    for base in candidates:
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.endswith(suffix):
                matches.append(child)

    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]
