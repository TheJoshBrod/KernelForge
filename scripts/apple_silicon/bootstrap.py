#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apple_silicon.constants import LLAMA_CPP_PINNED_COMMIT, LLAMA_CPP_REPO_URL


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def bootstrap(
    llamacpp_root: Path,
    force_clean: bool = False,
    build_tests: bool = True,
    metal_shader_debug: str = "OFF",
    metal_ndebug: str = "ON",
    metal_std: str = "",
    metal_macosx_version_min: str = "",
) -> dict:
    vendor_root = llamacpp_root.parent
    vendor_root.mkdir(parents=True, exist_ok=True)

    if force_clean and llamacpp_root.exists():
        shutil.rmtree(llamacpp_root)

    if not llamacpp_root.exists():
        run(["git", "clone", LLAMA_CPP_REPO_URL, str(llamacpp_root)])
    else:
        run(["git", "-C", str(llamacpp_root), "fetch", "--all", "--tags"])

    run(["git", "-C", str(llamacpp_root), "checkout", LLAMA_CPP_PINNED_COMMIT])

    build_dir = llamacpp_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        "-S",
        str(llamacpp_root),
        "-B",
        str(build_dir),
        "-DGGML_METAL=ON",
        f"-DLLAMA_BUILD_TESTS={'ON' if build_tests else 'OFF'}",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        f"-DGGML_METAL_SHADER_DEBUG={metal_shader_debug}",
        f"-DGGML_METAL_NDEBUG={metal_ndebug}",
    ]
    if metal_std:
        cmake_args.append(f"-DGGML_METAL_STD={metal_std}")
    if metal_macosx_version_min:
        cmake_args.append(f"-DGGML_METAL_MACOSX_VERSION_MIN={metal_macosx_version_min}")

    run(cmake_args)

    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])

    llama_cli = build_dir / "bin" / "llama-cli"
    llama_bench = build_dir / "bin" / "llama-bench"
    test_backend_ops = build_dir / "bin" / "test-backend-ops"
    if build_tests and not test_backend_ops.exists():
        raise RuntimeError(
            "Expected test-backend-ops to be built but it is missing at "
            f"{test_backend_ops}. Re-run bootstrap with a clean build directory."
        )

    return {
        "success": True,
        "llamacpp_root": str(llamacpp_root),
        "pinned_commit": LLAMA_CPP_PINNED_COMMIT,
        "llama_cli": str(llama_cli),
        "llama_bench": str(llama_bench),
        "test_backend_ops": str(test_backend_ops),
        "llama_cli_exists": llama_cli.exists(),
        "llama_bench_exists": llama_bench.exists(),
        "test_backend_ops_exists": test_backend_ops.exists(),
        "build_tests": bool(build_tests),
        "metal_options": {
            "GGML_METAL_SHADER_DEBUG": metal_shader_debug,
            "GGML_METAL_NDEBUG": metal_ndebug,
            "GGML_METAL_STD": metal_std,
            "GGML_METAL_MACOSX_VERSION_MIN": metal_macosx_version_min,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap pinned llama.cpp build for CGinS Apple Silicon flow")
    parser.add_argument(
        "--llamacpp-root",
        default=str(REPO_ROOT / ".vendor" / "llama.cpp"),
        help="Destination directory for llama.cpp clone/build",
    )
    parser.add_argument("--force-clean", action="store_true", help="Delete existing llama.cpp directory first")
    parser.add_argument("--with-tests", action="store_true", help="Build llama.cpp tests (enables test-backend-ops)")
    parser.add_argument("--no-tests", action="store_true", help="Skip llama.cpp tests build")
    parser.add_argument(
        "--metal-shader-debug",
        default="OFF",
        help="Set GGML_METAL_SHADER_DEBUG (e.g. ON/OFF)",
    )
    parser.add_argument(
        "--metal-ndebug",
        default="ON",
        help="Set GGML_METAL_NDEBUG (e.g. ON/OFF)",
    )
    parser.add_argument(
        "--metal-std",
        default="",
        help="Optional GGML_METAL_STD value (toolchain pinning)",
    )
    parser.add_argument(
        "--metal-macosx-version-min",
        default="",
        help="Optional GGML_METAL_MACOSX_VERSION_MIN value",
    )
    args = parser.parse_args()
    build_tests = True
    if args.no_tests:
        build_tests = False
    elif args.with_tests:
        build_tests = True

    try:
        result = bootstrap(
            Path(args.llamacpp_root).expanduser().resolve(),
            force_clean=args.force_clean,
            build_tests=build_tests,
            metal_shader_debug=str(args.metal_shader_debug).strip() or "OFF",
            metal_ndebug=str(args.metal_ndebug).strip() or "ON",
            metal_std=str(args.metal_std).strip(),
            metal_macosx_version_min=str(args.metal_macosx_version_min).strip(),
        )
    except Exception as exc:
        print({"success": False, "error": str(exc)})
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
