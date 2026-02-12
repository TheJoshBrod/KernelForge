#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _kernel_path(project: str, op: str) -> Path:
    return (
        _repo_root()
        / "projects"
        / project
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / op
        / "kernel.cu"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare per-op Codex workspace")
    parser.add_argument("--project", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_kernel = _kernel_path(args.project, args.op)
    if not src_kernel.exists():
        print(f"kernel.cu not found: {src_kernel}")
        return 1

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dst_kernel = work_dir / "kernel.cu"
    if dst_kernel.exists() and not args.overwrite:
        print(f"workspace already exists: {dst_kernel}")
        return 1

    shutil.copy2(src_kernel, dst_kernel)
    (work_dir / "attempts").mkdir(parents=True, exist_ok=True)
    (work_dir / "README.txt").write_text(
        "Edit kernel.cu only. This workspace is prepared by CGinS.",
        encoding="utf-8",
    )
    print(f"prepared: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
