#!/usr/bin/env python3
"""Prepare uploaded assets for a project, then trigger profile/benchmark pipeline."""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
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


def _project_dir(repo_root: Path, project_name: str) -> Path:
    primary = repo_root / "kernels" / "private" / "projects" / project_name
    legacy = repo_root / "projects" / project_name
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    return primary


def _decode_weights(project_dir: Path, weights_b64: str) -> None:
    if not weights_b64:
        return
    payload = base64.b64decode(weights_b64.encode("utf-8"))
    target = project_dir / "weights.pt"
    target.write_bytes(payload)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--weights-b64-path", default="")
    parser.add_argument("--validation-b64-path", default="")
    parser.add_argument("--validation-name-path", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    project_dir = _project_dir(repo_root, args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    weights_b64 = _read_text(args.weights_b64_path)
    validation_b64 = _read_text(args.validation_b64_path)
    _ = _read_text(args.validation_name_path)  # Reserved for future use.

    if weights_b64:
        print("[prepare_project_assets] Decoding weights...")
        _decode_weights(project_dir, weights_b64)

    if validation_b64:
        print("[prepare_project_assets] Decoding/extracting validation zip...")
        _decode_validation_zip(project_dir, validation_b64)

    pipeline_script = repo_root / "scripts" / "profile_then_benchmark.py"
    if not pipeline_script.exists():
        print("[prepare_project_assets] profile_then_benchmark.py not found; done.")
        return 0

    cmd = [sys.executable, str(pipeline_script), "--project", args.project]
    print(f"[prepare_project_assets] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(repo_root)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

