#!/usr/bin/env python3
"""Run a .cast inference package produced by KernelForge."""

import argparse
import hashlib
import json
import os
import time
import zipfile


def verify_checksums(zf: zipfile.ZipFile) -> None:
    header = json.loads(zf.read("HEADER.json"))
    checksum_bytes = zf.read("checksums.sha256")
    archive_checksum = hashlib.sha256(checksum_bytes).hexdigest()
    if archive_checksum != header["archive_checksum"]:
        raise RuntimeError(
            f"Archive checksum mismatch: expected {header['archive_checksum']}, got {archive_checksum}"
        )
    for line in checksum_bytes.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_hash, rel_path = parts[0], parts[1].strip()
        if rel_path == "checksums.sha256":
            continue
        actual_hash = hashlib.sha256(zf.read(rel_path)).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {rel_path}")


def compile_kernel(kernel_cu_path: str, op_name: str, build_dir: str, opt_level: str = "-O0"):
    import re
    from torch.utils.cpp_extension import load_inline

    with open(kernel_cu_path) as f:
        cuda_src = f.read()

    # Extract the launch() declaration for the C++ header (same approach as loader.py)
    match = re.search(r"(torch::Tensor\s+launch\s*\([^)]*\))", cuda_src)
    if not match:
        raise RuntimeError(f"Could not find 'launch' signature in {kernel_cu_path}")
    cpp_src = match.group(1) + ";"

    print(f"  Loading/compiling {op_name} ({opt_level}) ...")
    return load_inline(
        name=op_name,
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["launch"],
        extra_cuda_cflags=[opt_level],
        build_directory=build_dir,
        verbose=False,
        with_cuda=True,
    )


def load_cast(cast_path: str, model_args: dict | None = None, no_kernels: bool = False, opt_level: str = "-O0"):
    import torch
    import torch.nn.functional as F

    cast_path = os.path.abspath(cast_path)
    cache_key = hashlib.sha256(open(cast_path, "rb").read()).hexdigest()
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "cast", cache_key)

    print(f"Loading {cast_path}")

    with zipfile.ZipFile(cast_path) as zf:
        # 1. Validate header
        header = json.loads(zf.read("HEADER.json"))
        if header["file_type"] != "kernelforge_inference":
            raise RuntimeError(f"Expected kernelforge_inference, got {header['file_type']}")
        print(f"  Project : {header['project_name']}")
        print(f"  Version : {header['format_version']}")

        # 2. Verify checksums
        print("  Verifying checksums ...")
        verify_checksums(zf)
        print("  Checksums OK")

        # 3. Extract to cache dir
        if not os.path.isdir(cache_dir):
            print(f"  Extracting to {cache_dir}")
            zf.extractall(cache_dir)
        else:
            print(f"  Using cached extraction at {cache_dir}")

        manifest = json.loads(zf.read("manifest.json"))

    # 4. JIT compile kernels and patch ops
    build_dir = os.path.join(cache_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    original_linear = F.linear

    for op in manifest["ops"]:
        op_name = op["name"]
        kernel_cu = os.path.join(cache_dir, op["cuda_source"])

        if no_kernels:
            print(f"  [--no-kernels] Skipping kernel for {op_name}")
            continue

        if not torch.cuda.is_available():
            print(f"  [WARN] CUDA not available — skipping kernel for {op_name}")
            continue

        # Try precompiled .so for the current GPU first
        gpu_sm = "sm_{0}{1}".format(*torch.cuda.get_device_capability())
        precompiled = op.get("precompiled", {})
        so_rel = precompiled.get(gpu_sm)
        so_path = os.path.join(cache_dir, so_rel) if so_rel else None

        if so_path and os.path.exists(so_path):
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(op_name, so_path)
            ext = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(ext)  # type: ignore[union-attr]
            print(f"  Loaded precompiled {op_name} ({gpu_sm})")
        else:
            if so_rel:
                print(f"  [WARN] Precompiled .so not found for {gpu_sm}, falling back to JIT")
            if not os.path.exists(kernel_cu):
                print(f"  [WARN] No kernel.cu for {op_name}, skipping")
                continue
            if "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
            ext = compile_kernel(kernel_cu, op_name, build_dir, opt_level=opt_level)

        if op_name == "torch_nn_functional_linear":
            def _make_patched(ext=ext, orig=original_linear):
                def patched_linear(input, weight, bias=None):
                    try:
                        inp = input.cuda().contiguous()
                        w = weight.cuda().contiguous()
                        b = bias.cuda().contiguous() if bias is not None else None
                        return ext.launch(inp, w, b)
                    except Exception:
                        return orig(input, weight, bias)
                return patched_linear
            F.linear = _make_patched()
            print(f"  Patched torch.nn.functional.linear → {op_name}")

    # 5. Load model class from model.py
    import importlib.util
    import sys
    import torch.nn as nn

    model_py = os.path.join(cache_dir, "model.py")
    spec = importlib.util.spec_from_file_location("cast_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so inspect.getfile can resolve the module via
    # sys.modules — without this, inspect raises "is a built-in class".
    sys.modules["cast_model"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        del sys.modules["cast_model"]
        raise

    model_class_name = manifest.get("model_class", "")
    if model_class_name and hasattr(mod, model_class_name):
        ModelClass = getattr(mod, model_class_name)
    else:
        candidates = [
            v for v in vars(mod).values()
            if isinstance(v, type) and issubclass(v, nn.Module) and v.__module__ not in ("torch.nn", "builtins")
        ]
        if not candidates:
            raise RuntimeError("No model class found in model.py")
        ModelClass = candidates[-1]

    print(f"  Model class: {ModelClass.__name__}")

    # 6. Instantiate model
    model_init_args = manifest.get("model_init_args") or {}
    model_config_file = os.path.join(cache_dir, "model_config.json")
    if model_args:
        # CLI override: --model-args '{"model_type": "resnet", ...}'
        try:
            from transformers import AutoConfig
            cfg_dict = dict(model_args)
            model_type = cfg_dict.pop("model_type", None)
            config = AutoConfig.for_model(model_type, **cfg_dict)
            model = ModelClass(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model from --model-args: {e}") from e
    elif model_init_args:
        model = ModelClass(**model_init_args)
    elif os.path.exists(model_config_file):
        # HuggingFace model — load config from the bundled model_config.json
        try:
            from transformers import AutoConfig
            cfg_dict = json.load(open(model_config_file))
            model_type = cfg_dict.pop("model_type", None)
            # Rebuild id2label/label2id to be consistent with num_labels so
            # HuggingFace doesn't override num_labels with len(id2label).
            n = cfg_dict.get("num_labels")
            if n:
                cfg_dict["id2label"] = {str(i): f"LABEL_{i}" for i in range(n)}
                cfg_dict["label2id"] = {f"LABEL_{i}": i for i in range(n)}
            config = AutoConfig.for_model(model_type, **cfg_dict)
            model = ModelClass(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model from model_config.json: {e}") from e
    else:
        raise RuntimeError(
            "Cannot instantiate model: no model_init_args in manifest and "
            "no model_config.json bundled in the .cast file.\n"
            "Pass --model-args '{\"model_type\": ...}' or re-export with a "
            "model_config.json saved alongside model.py in the project."
        )

    # 7. Load weights
    import torch
    weight_file = os.path.join(cache_dir, manifest["weight_file"])
    print(f"  Loading weights from {manifest['weight_file']} ...")
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Run a KernelForge .cast inference package")
    parser.add_argument("cast_file", help="Path to .cast file")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument("--runs", type=int, default=5, help="Inference passes for timing")
    parser.add_argument(
        "--model-args",
        metavar="JSON",
        default=None,
        help='JSON config to instantiate the model, e.g. \'{"model_type":"resnet",...}\'. '
             "Used when the .cast has no model_config.json.",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="Skip JIT kernel compilation and run with native PyTorch ops.",
    )
    parser.add_argument(
        "--opt-level",
        default="-O0",
        choices=["-O0", "-O1", "-O2", "-O3"],
        help="NVCC optimisation level for JIT compilation (default: -O0).",
    )
    args = parser.parse_args()

    extra_model_args = json.loads(args.model_args) if args.model_args else None
    model = load_cast(
        args.cast_file,
        model_args=extra_model_args,
        no_kernels=args.no_kernels,
        opt_level=args.opt_level,
    )
    device = args.device
    model = model.to(device)
    print(f"\nModel ready on {device}")

    dummy = torch.randn(1, 3, 224, 224, device=device)
    print(f"Running {args.runs} inference pass(es) with input shape {list(dummy.shape)} ...")

    with torch.no_grad():
        # warmup
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.runs):
            out = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / args.runs * 1000

    logits = out.logits if hasattr(out, "logits") else out
    print(f"Output shape    : {list(logits.shape)}")
    print(f"Average latency : {elapsed_ms:.2f} ms")
    top5 = logits[0].topk(5)
    print(f"Top-5 indices   : {top5.indices.tolist()}")
    print(f"Top-5 scores    : {[f'{v:.4f}' for v in top5.values.tolist()]}")


if __name__ == "__main__":
    main()
