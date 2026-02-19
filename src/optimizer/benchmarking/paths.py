from __future__ import annotations

import shutil
from pathlib import Path

_MIGRATED_LEGACY_PROJECTS = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _migrate_legacy_projects(root: Path) -> None:
    global _MIGRATED_LEGACY_PROJECTS
    if _MIGRATED_LEGACY_PROJECTS:
        return

    canonical_root = root / "kernels" / "projects"
    canonical_root.mkdir(parents=True, exist_ok=True)

    legacy_roots = [
        root / "kernels" / "private" / "projects",
        root / "projects",
    ]

    for legacy_root in legacy_roots:
        if not legacy_root.exists():
            continue
        for child in legacy_root.iterdir():
            dst = canonical_root / child.name
            if dst.exists():
                continue

            source = child
            symlink_path = None
            if child.is_symlink():
                symlink_path = child
                resolved = child.resolve(strict=False)
                if not resolved.exists() or not resolved.is_dir():
                    continue
                source = resolved

            try:
                shutil.move(str(source), str(dst))
            except Exception:
                if source.exists() and source.is_dir():
                    shutil.copytree(source, dst, dirs_exist_ok=True)
                else:
                    continue

            if symlink_path is not None and symlink_path.exists():
                try:
                    symlink_path.unlink()
                except Exception:
                    pass

        try:
            next(legacy_root.iterdir())
        except StopIteration:
            try:
                legacy_root.rmdir()
            except Exception:
                pass
        except Exception:
            pass

    _MIGRATED_LEGACY_PROJECTS = True


def project_dir_for_name(project_name: str, create: bool = False) -> Path:
    root = repo_root()
    _migrate_legacy_projects(root)

    primary = root / "kernels" / "projects" / project_name

    if primary.exists():
        return primary
    if create:
        primary.mkdir(parents=True, exist_ok=True)
    return primary


def find_latest_optimized_dir(project_name: str) -> Path | None:
    base = repo_root() / "kernels" / "optimized"
    if not base.exists():
        return None

    suffix = f"_{project_name}"
    matches: list[Path] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.endswith(suffix):
            matches.append(child)

    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]
