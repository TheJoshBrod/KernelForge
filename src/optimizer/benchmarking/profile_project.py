from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import importlib
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .paths import project_dir_for_name
from .state import write_json_file

SKIP_FUNCTIONS = {
    "has_torch_function",
    "handle_torch_function",
    "is_storage",
    "result_type",
    "get_default_dtype",
}

# Only wrap functions that originate from these modules — anything else
# (dispatch guards from torch._C, JIT helpers, typing annotations, etc.) is skipped.
_COMPUTE_MODULES = frozenset({
    "torch",
    "torch.nn.functional",
    "torch._C._nn",
})

DEFAULT_SKIP_OPS = {
    "dropout",
    "dropout_",
    "alpha_dropout",
    "feature_alpha_dropout",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "randint",
    "bernoulli",
    "multinomial",
    "normal",
    "uniform",
    "poisson",
    "exponential",
    "view",
    "reshape",
    "permute",
    "transpose",
    "t",
    "squeeze",
    "unsqueeze",
    "expand",
    "expand_as",
    "as_strided",
    "flatten",
    "size",
    "stride",
    "numel",
    "dim",
    "shape",
    "to",
    "_to_copy",
    "contiguous",
    "clone",
    "copy_",
    "detach",
    "empty",
    "zeros",
    "ones",
    "full",
    "arange",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "new_empty",
    "new_zeros",
    "new_ones",
    "new_full",
}

DEFAULT_SKIP_PREFIXES = {"rand", "randn", "randint"}

PROFILE_ALLOW_OPS: set[str] = set()
PROFILE_SKIP_OPS: set[str] = set(DEFAULT_SKIP_OPS)
PROFILE_SKIP_PREFIXES: set[str] = set(DEFAULT_SKIP_PREFIXES)

calls: dict[str, list[dict[str, Any]]] = {}
_wrapped: set[Any] = set()
ENABLE_WRAPPING = True
skipped_counts: dict[str, int] = {}


def _serialize(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, (list, tuple)):
        return type(v)(_serialize(x) for x in v)
    if isinstance(v, dict):
        return {k: _serialize(x) for k, x in v.items()}
    return v  # torch.dtype, torch.device, int, float, bool, None, etc.


# Known signatures for C-extension ops that lack Python-inspectable signatures.
# Matches the public PyTorch API parameter order exactly.
_KNOWN_SIGS: dict[str, dict] = {  # noqa: E501  (line-length; values kept readable)
    "torch.nn.functional.linear": {
        "params": ["input", "weight", "bias"],
        "defaults": {"bias": None},
    },
    "torch.nn.functional.conv1d": {
        "params": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    },
    "torch.nn.functional.conv2d": {
        "params": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    },
    "torch.nn.functional.conv3d": {
        "params": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    },
    "torch.nn.functional.conv_transpose1d": {
        "params": ["input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1},
    },
    "torch.nn.functional.conv_transpose2d": {
        "params": ["input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1},
    },
    "torch.nn.functional.max_pool2d": {
        "params": ["input", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"],
        "defaults": {"stride": None, "padding": 0, "dilation": 1, "ceil_mode": False, "return_indices": False},
    },
    "torch.nn.functional.avg_pool2d": {
        "params": ["input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        "defaults": {"stride": None, "padding": 0, "ceil_mode": False, "count_include_pad": True, "divisor_override": None},
    },
    "torch.nn.functional.adaptive_avg_pool2d": {
        "params": ["input", "output_size"],
        "defaults": {},
    },
    "torch.nn.functional.adaptive_max_pool2d": {
        "params": ["input", "output_size", "return_indices"],
        "defaults": {"return_indices": False},
    },
}


def _build_signature(op_dict: dict) -> inspect.Signature:
    parameters = []
    for name in op_dict["params"]:
        if name in op_dict["defaults"]:
            param = inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=op_dict["defaults"][name],
            )
        else:
            param = inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        parameters.append(param)
    return inspect.Signature(parameters)


# Pre-built Signature objects — computed once at import time.
_COMPILED_SIGS: dict[str, inspect.Signature] = {
    k: _build_signature(v) for k, v in _KNOWN_SIGS.items()
}


@contextmanager
def _patched_auto_docstring():
    restored = []
    try:
        auto_docstring_module = importlib.import_module("transformers.utils.auto_docstring")
        utils_module = importlib.import_module("transformers.utils")

        def _identity(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def _decorator(obj):
                return obj

            return _decorator

        original_mod = getattr(auto_docstring_module, "auto_docstring", None)
        if original_mod:
            restored.append((auto_docstring_module, "auto_docstring", original_mod))
            auto_docstring_module.auto_docstring = _identity

        original_utils = getattr(utils_module, "auto_docstring", None)
        if original_utils:
            restored.append((utils_module, "auto_docstring", original_utils))
            utils_module.auto_docstring = _identity
    except Exception:
        restored = []

    try:
        yield
    finally:
        for module, attr, original in restored:
            try:
                setattr(module, attr, original)
            except Exception:
                pass


def _normalize_op_name(full_key: str) -> str:
    return full_key.split(".")[-1].lower().strip()


def _load_profile_filters(config: dict[str, Any]) -> None:
    global PROFILE_ALLOW_OPS, PROFILE_SKIP_OPS, PROFILE_SKIP_PREFIXES
    PROFILE_ALLOW_OPS = set()
    PROFILE_SKIP_OPS = set(DEFAULT_SKIP_OPS)
    PROFILE_SKIP_PREFIXES = set(DEFAULT_SKIP_PREFIXES)

    profile_cfg = config.get("profile") if isinstance(config, dict) else None
    if isinstance(profile_cfg, dict):
        allow_ops = profile_cfg.get("allow_ops") or profile_cfg.get("allowlist") or []
        skip_ops = profile_cfg.get("skip_ops") or profile_cfg.get("skiplist") or []
        skip_prefixes = profile_cfg.get("skip_prefixes") or []
        PROFILE_ALLOW_OPS = {str(op).lower() for op in allow_ops if op}
        PROFILE_SKIP_OPS.update({str(op).lower() for op in skip_ops if op})
        PROFILE_SKIP_PREFIXES.update({str(op).lower() for op in skip_prefixes if op})


def _should_skip(full_key: str) -> bool:
    op_name = _normalize_op_name(full_key)
    if PROFILE_ALLOW_OPS:
        return op_name not in PROFILE_ALLOW_OPS and full_key.lower() not in PROFILE_ALLOW_OPS
    if op_name in PROFILE_SKIP_OPS or full_key.lower() in PROFILE_SKIP_OPS:
        return True
    for prefix in PROFILE_SKIP_PREFIXES:
        if op_name.startswith(prefix):
            return True
    return False


def wrap_function(module, func_name: str) -> None:
    if not ENABLE_WRAPPING:
        return
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)
    module_path = module.__name__

    # Precompute full signature once per wrapped function.
    # Priority: _COMPILED_SIGS (hardcoded, reliable) → inspect (Python-native ops).
    # Using module_path.func_name avoids the func.__module__ trap where C-extension
    # ops report their internal module (e.g. torch._C._nn) rather than the public one.
    op_key = f"{module_path}.{func_name}"
    _func_sig = _COMPILED_SIGS.get(op_key)
    _sig_params: list[str] = []
    _sig_defaults: dict[str, Any] = {}

    if _func_sig:
        _sig_params = list(_func_sig.parameters.keys())
        _sig_defaults = {
            k: v.default
            for k, v in _func_sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
    else:
        try:
            _func_sig = inspect.signature(func)
            for name, param in _func_sig.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                _sig_params.append(name)
                if param.default is not inspect.Parameter.empty:
                    _sig_defaults[name] = param.default
        except (ValueError, TypeError):
            pass
        if not _sig_params:
            # Stub (*args, **kwargs) or uninspectable — discard the signature object
            # so the fallback recording path is used in wrapper.
            _func_sig = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{module_path}.{func_name}"
        output = func(*args, **kwargs)

        if _should_skip(key):
            skipped_counts[key] = skipped_counts.get(key, 0) + 1
            return output

        ser_output = _serialize(output)
        calls.setdefault(key, [])

        # Try to resolve full parameter set including defaults.
        # _func_sig is always a real inspect.Signature when set, so bind() +
        # apply_defaults() handles both Python-native and C-extension ops uniformly.
        if _func_sig and _sig_params:
            try:
                bound = _func_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                resolved_kwargs = {k: _serialize(v) for k, v in bound.arguments.items()}
                calls[key].append({
                    "function_name": key,
                    "args": [],
                    "kwargs": resolved_kwargs,
                    "output": ser_output,
                    "signature": {"params": _sig_params, "defaults": _sig_defaults},
                })
                return output
            except TypeError:
                pass  # bind failed — fall through to original

        # Original recording path (fallback)
        calls[key].append({
            "function_name": key,
            "args": [_serialize(a) for a in args],
            "kwargs": {k: _serialize(v) for k, v in kwargs.items()},
            "output": ser_output,
        })
        return output

    setattr(module, func_name, wrapper)


def wrap_torch_nn_functional() -> None:
    for name in dir(F):
        if name.startswith("_"):
            continue
        if name in SKIP_FUNCTIONS:
            continue
        obj = getattr(F, name)
        if not callable(obj):
            continue
        if (getattr(obj, "__module__", "") or "") not in _COMPUTE_MODULES:
            continue
        wrap_function(F, name)


def save_entries(func_name: str, entries: list[dict[str, Any]], base_dir: str, max_per_op: int = 200) -> None:
    func_dir = os.path.join(base_dir, func_name.replace(".", "_").replace("/", "_"))
    os.makedirs(func_dir, exist_ok=True)

    existing_count = len(
        [
            n
            for n in os.listdir(func_dir)
            if n.startswith("entry_") and n.endswith(".pt")
        ]
    )
    if existing_count > max_per_op:
        return

    for idx, entry in enumerate(entries):
        if existing_count + idx > max_per_op:
            return
        file_path = os.path.join(func_dir, f"entry_{existing_count + idx:06d}.pt")
        torch.save(entry, file_path)


def flush_calls(base_dir: str) -> dict[str, int]:
    op_counts: dict[str, int] = {}
    for func_name, entries in calls.items():
        save_entries(func_name, entries, base_dir)
        op_counts[func_name] = op_counts.get(func_name, 0) + len(entries)
    calls.clear()
    return op_counts


def import_model_module(model_path: Path):
    with _patched_auto_docstring():
        spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module at {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def load_project_config(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to read config.json: {e}")
        return {}


def _call_with_optional_path(fn, path_val: str | None):
    if not path_val:
        return fn()
    try:
        sig = inspect.signature(fn)
        for name in [
            "data_dir",
            "dataset_path",
            "validation_path",
            "validation_set",
            "path",
            "root",
        ]:
            if name in sig.parameters:
                return fn(**{name: path_val})
        if len(sig.parameters) == 1:
            return fn(path_val)
    except Exception as e:
        print(f"Warning: failed to call dataloader with path: {e}")
    return fn()


def _call_with_optional_device(fn, weights_path: Path, device: str):
    try:
        sig = inspect.signature(fn)
        if "device" in sig.parameters:
            return fn(str(weights_path), device=device)
    except (ValueError, TypeError):
        pass
    return fn(str(weights_path))


def _instantiate_discovered_model(module):
    candidates = []
    for _, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        try:
            if not issubclass(obj, torch.nn.Module):
                continue
        except Exception:
            continue
        if obj.__module__ != module.__name__:
            continue
        candidates.append(obj)

    if not candidates:
        raise RuntimeError("Could not discover a model class in model.py")

    def _priority(cls):
        name = cls.__name__.lower()
        score = 0
        if "for" in name:
            score += 2
        if "model" in name:
            score += 1
        return score

    candidates.sort(key=_priority, reverse=True)

    last_error = None
    for cls in candidates:
        try:
            config_cls = getattr(cls, "config_class", None)
            if config_cls:
                try:
                    cfg = config_cls()
                    return cls(cfg)
                except Exception:
                    pass

            sig = inspect.signature(cls)
            required = []
            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if p.default == inspect.Parameter.empty:
                    required.append(p.name)
            if not required:
                return cls()
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise RuntimeError(f"Failed to instantiate discovered model class: {last_error}")
    raise RuntimeError("Failed to instantiate discovered model class")


def load_model(module, weights_path: Path, device: str):
    if hasattr(module, "load_weights") and weights_path.exists():
        return _call_with_optional_device(module.load_weights, weights_path, device)

    if hasattr(module, "build_model"):
        model = module.build_model()
    elif hasattr(module, "get_model"):
        model = module.get_model()
    else:
        model = _instantiate_discovered_model(module)

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


def get_samples(module, max_batches: int, validation_path: str | None):
    if hasattr(module, "sample_inputs"):
        data = module.sample_inputs()
    elif hasattr(module, "get_sample_inputs"):
        data = module.get_sample_inputs()
    elif hasattr(module, "make_example_input"):
        data = module.make_example_input()
    elif hasattr(module, "get_dataloader"):
        data = _call_with_optional_path(module.get_dataloader, validation_path)
    elif hasattr(module, "get_validation_dataloader"):
        data = _call_with_optional_path(module.get_validation_dataloader, validation_path)
    else:
        data = None

    if isinstance(data, torch.utils.data.DataLoader):
        samples = []
        for i, batch in enumerate(data):
            if i >= max_batches:
                break
            samples.append(batch)
        return samples
    if isinstance(data, (list, tuple)):
        return list(data)
    if data is not None:
        return [data]
    return []


def _resolve_device() -> str:
    target = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if target == "mps" and hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    if target in {"gpu", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _collect_fallback_aten_stats(model, sample, device: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    try:
        from torch.profiler import ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        args_tuple, kwargs = normalize_inputs(sample)
        args_tuple = move_to_device(args_tuple, device)
        kwargs = move_to_device(kwargs, device)

        with torch.no_grad():
            with torch.profiler.profile(activities=activities) as prof:
                try:
                    model(*args_tuple, **kwargs)
                except TypeError:
                    model(*args_tuple)

        for event in prof.key_averages():
            name = str(event.key) if hasattr(event, "key") else ""
            if not name.startswith("aten::"):
                continue
            op_name = name.replace("::", "_").replace(".", "_").replace("/", "_")
            count = int(getattr(event, "count", 0) or 0)
            if count <= 0:
                continue
            cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            avg_ms = (cpu_us / 1000.0) / float(count) if count > 0 else 0.0
            prev = out[op_name] if op_name in out else {"count": 0.0, "avg_ms": 0.0}
            total_count = float(prev["count"]) + float(count)
            weighted_ms = (float(prev["avg_ms"]) * float(prev["count"])) + (avg_ms * float(count))
            out[op_name] = {
                "count": total_count,
                "avg_ms": (weighted_ms / total_count) if total_count > 0 else 0.0,
            }
    except Exception:
        return {}
    return out


def _default_sample_from_model(model) -> Any:
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
    except Exception:
        params = {}

    sample = {}
    if "pixel_values" in params:
        sample["pixel_values"] = torch.randn(1, 3, 224, 224)
    if "input_ids" in params:
        sample["input_ids"] = torch.randint(0, 1000, (1, 32), dtype=torch.long)
    if "attention_mask" in params:
        sample["attention_mask"] = torch.ones((1, 32), dtype=torch.long)

    if sample:
        return sample

    if "x" in params:
        return torch.randn(1, 3, 224, 224)
    return torch.randn(1, 3, 224, 224)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile a project model to capture per-op inputs/outputs."
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    if args.project_dir:
        project_dir = Path(args.project_dir)
    elif args.project:
        project_dir = project_dir_for_name(args.project)
    else:
        raise RuntimeError("Provide --project or --project-dir")

    model_path = project_dir / "model.py"
    weights_path = project_dir / "weights.pt"
    if not model_path.exists():
        raise RuntimeError(f"Missing model.py at {model_path}")

    out_dir = Path(args.out_dir) if args.out_dir else project_dir / "io" / "individual_ops"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    config = load_project_config(project_dir)
    _load_profile_filters(config)
    wrap_torch_nn_functional()

    module = import_model_module(model_path)
    model = load_model(module, weights_path, device)
    model.to(device)
    model.eval()

    validation_raw = config.get("validation_dir") or config.get("validation_set") or ""
    validation_path = None
    if validation_raw:
        candidate = Path(validation_raw)
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        if candidate.exists():
            validation_path = str(candidate)
        else:
            print(f"Warning: validation path not found: {candidate}")

    samples = get_samples(module, args.max_batches, validation_path)
    if not samples:
        samples = [_default_sample_from_model(model)]
    op_totals: dict[str, int] = {}
    op_profile_ms: dict[str, float] = {}

    def _run_sample(sample):
        args_tuple, kwargs = normalize_inputs(sample)
        args_tuple = move_to_device(args_tuple, device)
        kwargs = move_to_device(kwargs, device)
        try:
            model(*args_tuple, **kwargs)
        except TypeError:
            model(*args_tuple)

    with torch.no_grad():
        # First pass: capture op counts (single forward pass = canonical call count)
        _run_sample(samples[0])
        first_pass_counts = flush_calls(str(out_dir))
        for k, v in first_pass_counts.items():
            op_totals[k] = v

        # Remaining passes: capture more timing entries from the full validation set
        # (benchmark_ops replays these for avg latency) — counts are not accumulated
        for sample in samples[1:]:
            _run_sample(sample)
            flush_calls(str(out_dir))  # writes entries for timing, counts discarded

    if samples and len(op_totals) < 3:
        fallback_stats = _collect_fallback_aten_stats(model, samples[0], device)
        for k in fallback_stats:
            info = fallback_stats[k]
            v = int(info["count"]) if "count" in info else 0
            if k not in op_totals:
                op_totals[k] = v
            if "avg_ms" in info:
                current_count = float(op_totals.get(k, 0))
                prev_ms = float(op_profile_ms.get(k, 0.0))
                prev_count = float(v) if v > 0 else current_count
                new_count = float(v)
                if k in op_profile_ms and prev_count > 0 and new_count > 0:
                    combined = (prev_ms * prev_count) + (float(info["avg_ms"]) * new_count)
                    op_profile_ms[k] = combined / (prev_count + new_count)
                else:
                    op_profile_ms[k] = float(info["avg_ms"])

    summary_path = out_dir.parent / "summary.json"
    summary = {
        "project": project_dir.name,
        "device": device,
        "op_counts": op_totals,
        "op_profile_ms": op_profile_ms,
        "skipped_counts": skipped_counts,
        "skip_filters": {
            "allow_ops": sorted(PROFILE_ALLOW_OPS),
            "skip_ops": sorted(PROFILE_SKIP_OPS),
            "skip_prefixes": sorted(PROFILE_SKIP_PREFIXES),
        },
    }
    write_json_file(summary_path, summary)
    print(f"Saved profiling entries to {out_dir}")
    print(f"Summary written to {summary_path}")

    # Export op-level DAG via torch.jit.trace (records actual execution with real data)
    dag_path = out_dir.parent / "dag.json"
    dag_data = {"nodes": [], "edges": []}

    # Maps JIT aten op names -> display names for meaningful NN ops.
    # Primitives (add, reshape, etc.) are absent and will be propagated-through
    # so the graph stays fully connected across them.
    # NOTE: torch.jit inlined graphs lower conv2d -> _convolution, so that is
    # the key we actually see; we display it as "conv2d".
    _DAG_OPS: dict[str, str] = {
        # Convolutions (JIT lowers all conv variants to _convolution)
        "_convolution": "conv2d",
        "conv1d": "conv1d",
        "conv2d": "conv2d",
        "conv3d": "conv3d",
        "conv_transpose1d": "conv_transpose1d",
        "conv_transpose2d": "conv_transpose2d",
        # Normalization
        "batch_norm": "batch_norm",
        "native_batch_norm": "batch_norm",
        "layer_norm": "layer_norm",
        "group_norm": "group_norm",
        "instance_norm": "instance_norm",
        # Activations
        "relu": "relu",
        "gelu": "gelu",
        "silu": "silu",
        "hardswish": "hardswish",
        "leaky_relu": "leaky_relu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "softmax": "softmax",
        "log_softmax": "log_softmax",
        "hardtanh": "hardtanh",
        "mish": "mish",
        # Pooling
        "max_pool1d": "max_pool1d",
        "max_pool2d": "max_pool2d",
        "max_pool3d": "max_pool3d",
        "avg_pool1d": "avg_pool1d",
        "avg_pool2d": "avg_pool2d",
        "avg_pool3d": "avg_pool3d",
        "adaptive_avg_pool1d": "adaptive_avg_pool1d",
        "adaptive_avg_pool2d": "adaptive_avg_pool2d",
        "adaptive_max_pool2d": "adaptive_max_pool2d",
        # Linear
        "linear": "linear",
        "addmm": "linear",
        "mm": "mm",
        # Other NN ops
        "dropout": "dropout",
        "embedding": "embedding",
        "scaled_dot_product_attention": "attention",
        "matmul": "matmul",
        "bmm": "bmm",
        "upsample_bilinear2d": "upsample_bilinear2d",
        "upsample_nearest2d": "upsample_nearest2d",
    }

    try:
        sample = samples[0]
        args_tuple, kwargs = normalize_inputs(sample)
        args_tuple = move_to_device(args_tuple, device)
        kwargs = move_to_device(kwargs, device)

        with torch.no_grad():
            if kwargs and not args_tuple:
                traced = torch.jit.trace(
                    model, example_kwarg_inputs=kwargs, strict=False
                )
            else:
                traced = torch.jit.trace(model, args_tuple, strict=False)

        graph = traced.inlined_graph

        # Maps output-value unique() -> set of meaningful node_ids that produced it.
        # Filtered (non-DAG_OPS) nodes propagate their inputs' sources to their
        # outputs so downstream meaningful ops stay connected across primitives.
        value_to_sources: dict[int, set[str]] = {}
        seen_edges: set[tuple[str, str]] = set()
        idx = 0

        for jit_node in graph.nodes():
            kind = jit_node.kind()  # e.g. "aten::conv2d"
            if "::" not in kind:
                continue
            short_op = kind.split("::")[1].rstrip("_")

            # Collect all meaningful source nodes reachable via inputs
            input_sources: set[str] = set()
            for inp in jit_node.inputs():
                srcs = value_to_sources.get(inp.unique())
                if srcs:
                    input_sources.update(srcs)

            display_op = _DAG_OPS.get(short_op)
            if display_op is not None:
                node_id = f"{display_op}_{idx}"
                dag_data["nodes"].append({"id": node_id, "op": display_op})

                for src_id in input_sources:
                    edge_key = (src_id, node_id)
                    if edge_key not in seen_edges:
                        dag_data["edges"].append(
                            {"source": src_id, "target": node_id}
                        )
                        seen_edges.add(edge_key)

                # This node's outputs originate from this node
                for out in jit_node.outputs():
                    value_to_sources[out.unique()] = {node_id}

                idx += 1
            elif display_op is None:
                # Filtered op: pass sources through so the chain stays connected
                for out in jit_node.outputs():
                    value_to_sources[out.unique()] = input_sources.copy()

        print(
            f"JIT trace DAG: {len(dag_data['nodes'])} nodes, "
            f"{len(dag_data['edges'])} edges"
        )
    except Exception as e:
        print(f"JIT trace DAG failed: {e}")

    write_json_file(dag_path, dag_data)
    print(f"DAG written to {dag_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
