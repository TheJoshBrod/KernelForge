#!/usr/bin/env python3
"""
optimize_kernel.py (NVCC-free version)

This version:
 - Generates LLM candidates
 - Validates them using existing validate_kernel()
 - DOES NOT compile with NVCC
 - DOES NOT load as Python extensions
 - DOES NOT profile kernels

Runner.py will later benchmark the generated kernels.
"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from byllm.lib import Model, by
from src.generator.verifier import validate_kernel


# ----------------------------
# GPU target specs
# ----------------------------
gpu_presets = {
    "3090": {
        "name": "NVIDIA GeForce RTX 3090",
        "arch": "sm_86",

        "sm_count": 82,
        "l2_cache_kb": 6144,
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "max_threads_per_sm": 1536,
        "shared_mem_per_sm_kb": 100,
        "shared_mem_per_block_kb": 48,
        "shared_mem_optin_kb": 100,
        "registers_per_sm": 65536,
        "registers_per_block": 65536,
        "max_registers_per_thread": 255,
        "sm_clock_mhz": 2100,
        "mem_clock_mhz": 9751,
        "vram_gb": 24,
    }
}

llm = Model(model_name="claude-sonnet-4-5-20250929")

@by(llm)
def llm_generate_cuda(system_prompt: str, user_prompt: str) -> str:
    return f"{system_prompt}\n\n{user_prompt}"


# ----------------------------
# Helper: sanitize LLM output
# ----------------------------
def clean_llm_cuda(text: str) -> str:
    """
    Remove backticks, ```cuda fences, markdown artifacts.
    """
    text = text.replace("```cuda", "")
    text = text.replace("```c++", "")
    text = text.replace("```cpp", "")
    text = text.replace("```", "")
    # Trim junk whitespace
    return text.strip()


# ----------------------------
# LLM optimization prompt
# ----------------------------
def llm_optimize_kernel(original_code: str, gpu_specs: dict, attempt: int) -> str:
    # Compact hardware descriptor
    hw = (
        f"GPU={gpu_specs['name']}, "
        f"arch={gpu_specs['arch']}, "
        f"warp={gpu_specs['warp_size']}, "
        f"max_threads={gpu_specs['max_threads_per_block']}, "
        f"smem_block={gpu_specs['shared_mem_per_block_kb']}KB, "
        f"regs_thread={gpu_specs['max_registers_per_thread']}"
    )

    system_prompt = f"""
You are an expert CUDA kernel engineer.
Optimize CUDA kernels for the target hardware: {hw}.

Rules:
- Keep the launch(...) API exactly the same.
- Improve memory coalescing, occupancy, warp efficiency, and register usage.
- Avoid unnecessary __syncthreads().
- Prefer shared memory tiling or warp-level ops when useful.
- DO NOT output markdown, text, commentary, or backticks.
- Output ONLY valid CUDA C++ code.
"""

    user_prompt = f"""
Optimize this CUDA kernel (attempt #{attempt}).
Make small but meaningful hardware-aware improvements.
Keep the same external behavior.

{original_code}

Return ONLY CUDA code:
"""

    raw = llm_generate_cuda(system_prompt, user_prompt)
    return clean_llm_cuda(raw)


# ----------------------------
# Main optimization
# ----------------------------
def optimize_kernel(input_cu: Path, entry_file: Path, output_dir: Path, gpu_specs: dict):
    print(f"\n🔧 Optimizing kernel (NVCC-free): {input_cu}")

    base_code = input_cu.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_code = base_code
    found_valid = False

    ATTEMPTS = 4

    for attempt in range(ATTEMPTS):
        print(f"\n=== Generating candidate #{attempt} ===")

        optimized_code = llm_optimize_kernel(base_code, gpu_specs, attempt)

        candidate_cu = output_dir / f"candidate_{attempt}.cu"
        candidate_log = output_dir / f"log_{attempt}.txt"
        candidate_cu.write_text(optimized_code)

        print(f"→ Validating candidate #{attempt}")

        with TemporaryDirectory() as tmp:
            call_ok, exec_ok, _ = validate_kernel(
                generated_cu_code=optimized_code,
                entry_file=entry_file,
                log_file_path=candidate_log,
                tmpdir=Path(tmp),
            )

        if not (call_ok and exec_ok):
            print(f"✗ Candidate #{attempt} FAILED verification.")
            continue

        print(f"✓ Candidate #{attempt} PASSED verification.")
        best_code = optimized_code
        found_valid = True
        break

    # ----------------------------
    # Save final result
    # ----------------------------
    final_path = output_dir / "kernel.cu"
    final_path.write_text(best_code)

    if not found_valid:
        print("\n⚠ No optimized candidate passed. Using original kernel.\n")
    else:
        print("\n🏆 Saved first valid optimized kernel.\n")

    print(f"→ Output written to: {final_path}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gpu", default="3090")

    args = parser.parse_args()

    optimize_kernel(
        input_cu=Path(args.input),
        entry_file=Path(args.entry),
        output_dir=Path(args.outdir),
        gpu_specs=gpu_presets[args.gpu],
    )
