import argparse
import importlib.util
import inspect
import json
import os
from functools import wraps
from pathlib import Path

import torch
import torch.nn.functional as F

SKIP_FUNCTIONS = [
    "has_torch_function",
    "handle_torch_function",
    "is_storage",
    "result_type",
    "get_default_dtype",
]

calls = {}
_wrapped = set()
ENABLE_WRAPPING = True


def wrap_function(module, func_name):
    if not ENABLE_WRAPPING:
        return
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)

    module_path = module.__name__

    # Get function signature to extract parameter order and defaults
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
    except (ValueError, TypeError):
        params = []
        defaults = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{module_path}.{func_name}"

        ser_args = [
            a.detach().cpu() if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        ser_kwargs = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        output = func(*args, **kwargs)

        if isinstance(output, torch.Tensor):
            ser_output = output.detach().cpu()
        elif isinstance(output, (list, tuple)):
            ser_output = [
                o.detach().cpu() if isinstance(o, torch.Tensor) else o
                for o in output
            ]
        else:
            ser_output = output

        calls.setdefault(key, [])
        calls[key].append({
            "function_name": key,
            "args": ser_args,
            "kwargs": ser_kwargs,
            "output": ser_output,
            "signature": {
                "params": params,
                "defaults": defaults
            }
        })

        return output

    setattr(module, func_name, wrapper)


def wrap_torch_nn_functional():
    for name in dir(F):
        if name.startswith("_"):
            continue
        if any(skip in name for skip in ["torch_function", "storage", "result_type", "dtype"]):
            continue
        obj = getattr(F, name)
        if callable(obj):
            wrap_function(F, name)


def save_entries(func_name, entries, base_dir, max_per_op=200):
    func_dir = os.path.join(base_dir, func_name.replace(".", "_").replace("/", "_"))
    os.makedirs(func_dir, exist_ok=True)

    existing_count = len([n for n in os.listdir(func_dir) if n.startswith("entry_") and n.endswith(".pt")])
    if existing_count > max_per_op:
        return

    for idx, entry in enumerate(entries):
        if existing_count + idx > max_per_op:
            return
        file_path = os.path.join(func_dir, f"entry_{existing_count + idx:06d}.pt")
        torch.save(entry, file_path)


def flush_calls(base_dir):
    op_counts = {}
    for func_name, entries in calls.items():
        save_entries(func_name, entries, base_dir)
        op_counts[func_name] = op_counts.get(func_name, 0) + len(entries)
    calls.clear()
    return op_counts


def import_model_module(model_path: Path):
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_with_optional_device(fn, weights_path: Path, device: str):
    try:
        sig = inspect.signature(fn)
        if "device" in sig.parameters:
            return fn(str(weights_path), device=device)
    except (ValueError, TypeError):
        pass
    return fn(str(weights_path))


def load_model(module, weights_path: Path, device: str):
    if hasattr(module, "load_weights") and weights_path.exists():
        model = _call_with_optional_device(module.load_weights, weights_path, device)
        return model

    if hasattr(module, "build_model"):
        model = module.build_model()
    elif hasattr(module, "get_model"):
        model = module.get_model()
    else:
        raise RuntimeError("model.py must define build_model() or load_weights().")

    if weights_path.exists():
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state, torch.nn.Module):
            model = state
        else:
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")
            except Exception as e:
                print(f"Warning: failed to load state_dict: {e}")

    return model


def normalize_inputs(sample):
    if isinstance(sample, dict):
        return (), sample
    if isinstance(sample, (list, tuple)):
        if len(sample) == 2 and isinstance(sample[1], dict):
            args = sample[0] if isinstance(sample[0], (list, tuple)) else (sample[0],)
            return tuple(args), sample[1]
        return tuple(sample), {}
    return (sample,), {}


def move_to_device(obj, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def get_samples(module, project_dir: Path, max_batches: int):
    if hasattr(module, "sample_inputs"):
        data = module.sample_inputs()
    elif hasattr(module, "get_sample_inputs"):
        data = module.get_sample_inputs()
    elif hasattr(module, "make_example_input"):
        data = module.make_example_input()
    elif hasattr(module, "get_dataloader"):
        data = module.get_dataloader()
    elif hasattr(module, "get_validation_dataloader"):
        data = module.get_validation_dataloader()
    else:
        raise RuntimeError(
            "model.py must define sample_inputs()/make_example_input() or get_dataloader()."
        )

    if isinstance(data, torch.utils.data.DataLoader):
        samples = []
        for i, batch in enumerate(data):
            if i >= max_batches:
                break
            samples.append(batch)
        return samples

    if isinstance(data, (list, tuple)):
        return list(data)

    return [data]


def main():
    parser = argparse.ArgumentParser(description="Profile a project model to capture per-op inputs/outputs.")
    parser.add_argument("--project", type=str, default=None, help="Project name under projects/")
    parser.add_argument("--project-dir", type=str, default=None, help="Full path to project directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for entry_*.pt files")
    parser.add_argument("--max-batches", type=int, default=10, help="Max batches to profile")
    args = parser.parse_args()

    if args.project_dir:
        project_dir = Path(args.project_dir)
    elif args.project:
        project_dir = Path("projects") / args.project
    else:
        raise RuntimeError("Provide --project or --project-dir")

    model_path = project_dir / "model.py"
    weights_path = project_dir / "weights.pt"
    if not model_path.exists():
        raise RuntimeError(f"Missing model.py at {model_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = project_dir / "io" / "individual_ops"

    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrap_torch_nn_functional()

    module = import_model_module(model_path)
    model = load_model(module, weights_path, device)
    model.to(device)
    model.eval()

    samples = get_samples(module, project_dir, args.max_batches)

    op_totals = {}
    with torch.no_grad():
        for sample in samples:
            args_tuple, kwargs = normalize_inputs(sample)
            args_tuple = move_to_device(args_tuple, device)
            kwargs = move_to_device(kwargs, device)

            try:
                model(*args_tuple, **kwargs)
            except TypeError:
                model(*args_tuple)

            batch_counts = flush_calls(str(out_dir))
            for k, v in batch_counts.items():
                op_totals[k] = op_totals.get(k, 0) + v

    summary_path = out_dir.parent / "summary.json"
    summary = {
        "project": project_dir.name,
        "device": device,
        "op_counts": op_totals,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved profiling entries to {out_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
