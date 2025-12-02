#!/usr/bin/env python3
"""
generate_all_optimized_kernels.py

Scans:
  generated_kernels/PyTorchFunctions/* for kernel.cu

Finds matching entry files in:
  benchmarks/generate_benchmarks/PyTorchFunctions/*/entry_*.pt

Then runs optimize_kernel() for each.
"""

from pathlib import Path
from optimize_kernel import optimize_kernel, gpu_presets
import glob
import sys

# ---------------------------------------------------------------------
# ROOTS
# ---------------------------------------------------------------------
BASE_KERNEL_ROOT = Path("generated_kernels/PyTorchFunctions")
BASE_ENTRY_ROOT  = Path("benchmarks/generate_benchmarks/PyTorchFunctions")
OUT_ROOT         = Path("benchmarks/run_benchmarks/optimized_compiled_kernels")


def main():
    print(f"🔍 Scanning {BASE_KERNEL_ROOT} for kernels...\n")

    kernel_dirs = sorted([p for p in BASE_KERNEL_ROOT.iterdir() if p.is_dir()])

    kernels_to_optimize = []
    for kdir in kernel_dirs:
        name = kdir.name

        kernel_cu = kdir / "kernel.cu"
        if not kernel_cu.exists():
            print(f"⚠ Missing kernel.cu for: {name}")
            continue

        # corresponding benchmark directory
        entry_dir = BASE_ENTRY_ROOT / name
        entry_files = sorted(entry_dir.glob("entry_*.pt"))

        if not entry_files:
            print(f"⚠ No entry_*.pt files found for: {name}")
            continue

        # pick first entry file
        entry_file = entry_files[0]

        kernels_to_optimize.append((name, kernel_cu, entry_file))

    if not kernels_to_optimize:
        print("❌ No kernels found to optimize.")
        return

    print(f"📌 Found {len(kernels_to_optimize)} kernels to optimize.\n")

    # ------------------------------------------------------------------
    # Run optimization
    # ------------------------------------------------------------------
    specs = gpu_presets["3090"]

    for name, kernel_cu, entry_file in kernels_to_optimize:
        print("="*80)
        print(f"🚀 Optimizing kernel: {name}")
        print(f"    kernel.cu: {kernel_cu}")
        print(f"    entry:     {entry_file}\n")

        outdir = OUT_ROOT / name
        outdir.mkdir(parents=True, exist_ok=True)

        optimize_kernel(
            input_cu = kernel_cu,
            entry_file = entry_file,
            output_dir = outdir,
            gpu_specs = specs
        )


if __name__ == "__main__":
    main()
