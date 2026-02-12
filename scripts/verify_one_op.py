#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import shutil
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _paths(project: str, op: str) -> tuple[Path, Path]:
    root = _repo_root() / "projects" / project
    kernel = root / "kernels" / "generated" / "individual_op_kernels" / op / "kernel.cu"
    io_dir = root / "io" / "individual_ops" / op
    return kernel, io_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one generated kernel op against captured IO")
    parser.add_argument("--project", required=True)
    parser.add_argument("--op", required=True)
    args = parser.parse_args()

    kernel_path, io_dir = _paths(args.project, args.op)
    if not kernel_path.exists():
        print(f"kernel not found: {kernel_path}")
        return 1
    if not io_dir.exists():
        print(f"io directory not found: {io_dir}")
        return 1

    try:
        from src.generator.verifier import validate_kernel
    except Exception as exc:
        print(f"failed to import verifier: {exc}")
        return 1

    entry_files = sorted(glob.glob(str(io_dir / "entry_*.pt")))
    if not entry_files:
        print(f"no entry_*.pt files in {io_dir}")
        return 1

    cu_code = kernel_path.read_text(encoding="utf-8")
    tmp_root = Path(tempfile.mkdtemp(prefix="cgins_verify_one_op_"))
    try:
        for i, entry_file in enumerate(entry_files):
            tmpdir = tmp_root / f"case_{i}"
            tmpdir.mkdir(parents=True, exist_ok=True)
            log_file = io_dir / f"verify_{i}.log"
            call_ok, exec_ok, message = validate_kernel(cu_code, entry_file, log_file, tmpdir)
            if not (call_ok and exec_ok):
                print(message)
                return 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
