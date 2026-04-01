from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import torch


_EMPTY = inspect.Parameter.empty

_KNOWN_FUNCTION_SIGNATURES: dict[str, tuple[list[str], dict[str, Any]]] = {
    "conv1d": (
        ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        {
            "bias": None,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
        },
    ),
    "embedding": (
        [
            "input",
            "weight",
            "padding_idx",
            "max_norm",
            "norm_type",
            "scale_grad_by_freq",
            "sparse",
        ],
        {
            "padding_idx": None,
            "max_norm": None,
            "norm_type": 2.0,
            "scale_grad_by_freq": False,
            "sparse": False,
        },
    ),
    "grouped_mm": (
        ["mat_a", "mat_b", "offs", "bias", "out_dtype"],
        {
            "offs": None,
            "bias": None,
            "out_dtype": None,
        },
    ),
    "linear": (
        ["input", "weight", "bias"],
        {
            "bias": None,
        },
    ),
    "pad": (
        ["input", "pad", "mode", "value"],
        {
            "mode": "constant",
            "value": None,
        },
    ),
    "sigmoid": (
        ["input"],
        {},
    ),
    "silu": (
        ["input", "inplace"],
        {
            "inplace": False,
        },
    ),
    "softmax": (
        ["input", "dim", "_stacklevel", "dtype"],
        {
            "dim": None,
            "_stacklevel": 3,
            "dtype": None,
        },
    ),
    "softplus": (
        ["input", "beta", "threshold"],
        {
            "beta": 1.0,
            "threshold": 20.0,
        },
    ),
}


def empty_adapter_stats() -> dict[str, int]:
    return {
        "tensor_arg_count": 0,
        "noncontiguous_tensor_arg_count": 0,
        "preserved_layout_tensor_arg_count": 0,
        "contiguous_copy_count": 0,
        "device_move_count": 0,
    }


def merge_adapter_stats(total: dict[str, int], delta: dict[str, int] | None) -> dict[str, int]:
    if not delta:
        return total
    for key, value in delta.items():
        try:
            total[key] = int(total.get(key, 0)) + int(value)
        except Exception:
            continue
    return total


def normalize_launch_value(param_name: str, param_type: str, value: Any) -> Any:
    if "std::vector<int64_t>" in param_type:
        if isinstance(value, int):
            return [value]
        if isinstance(value, torch.Size):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
    if "int64_t" in param_type and isinstance(value, bool):
        return int(value)
    return value


def split_signature_params(signature_text: str) -> list[str]:
    params: list[str] = []
    current: list[str] = []
    depth = 0
    for char in signature_text:
        if char in "<({[":
            depth += 1
        elif char in ">)}]":
            depth = max(depth - 1, 0)
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                params.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        params.append(tail)
    return params


def parse_launch_params(kernel_path: Path) -> list[tuple[str, str]]:
    if not kernel_path.exists():
        return []
    text = kernel_path.read_text(encoding="utf-8")
    match = re.search(r"torch::Tensor\s+launch\s*\((.*?)\)\s*\{", text, re.S)
    if not match:
        return []
    params: list[tuple[str, str]] = []
    for raw_param in split_signature_params(match.group(1)):
        cleaned = " ".join(raw_param.split())
        if not cleaned:
            continue
        pieces = cleaned.rsplit(" ", 1)
        if len(pieces) != 2:
            continue
        param_type, param_name = pieces
        params.append((param_name.strip(), param_type.strip()))
    return params


def extract_launch_arity(kernel_path: Path | None, launch_obj: Any) -> int | None:
    if kernel_path is not None:
        params = parse_launch_params(kernel_path)
        if params:
            return len(params)
    try:
        return len(inspect.signature(launch_obj).parameters)
    except Exception:
        return None


def launch_params_for_runtime_kernel(
    kernel_path: Path | None,
    launch_obj: Any,
) -> list[tuple[str, str]]:
    params = parse_launch_params(kernel_path) if kernel_path is not None else []
    if params:
        return params
    arity = extract_launch_arity(kernel_path, launch_obj)
    return [(f"arg_{index}", "") for index in range(arity or 0)]


def _signature_params_from_metadata(signature_meta: Any) -> list[tuple[str, Any]]:
    if not isinstance(signature_meta, dict):
        return []
    params = signature_meta.get("params")
    defaults = signature_meta.get("defaults", {})
    if not isinstance(params, list):
        return []
    if not isinstance(defaults, dict):
        defaults = {}
    out: list[tuple[str, Any]] = []
    for item in params:
        name = str(item).strip()
        if not name:
            continue
        out.append((name, defaults.get(name, _EMPTY)))
    return out


def signature_params_for_call(
    *,
    op_name: str = "",
    func: Any = None,
    signature_meta: Any = None,
) -> list[tuple[str, Any]]:
    params = _signature_params_from_metadata(signature_meta)
    if params:
        return params

    if func is not None:
        try:
            sig = inspect.signature(func)
        except Exception:
            sig = None
        if sig is not None:
            return [
                (name, param.default)
                for name, param in sig.parameters.items()
                if param.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]

    fn_name = str(op_name or "")
    if fn_name.startswith("torch_nn_functional_"):
        fn_name = fn_name.replace("torch_nn_functional_", "", 1)
    if func is not None and not fn_name:
        fn_name = str(getattr(func, "__name__", "") or "")
    if fn_name in _KNOWN_FUNCTION_SIGNATURES:
        names, defaults = _KNOWN_FUNCTION_SIGNATURES[fn_name]
        return [(name, defaults.get(name, _EMPTY)) for name in names]
    return []


def resolve_ordered_args(
    args: Any,
    kwargs: dict[str, Any] | None,
    *,
    op_name: str = "",
    func: Any = None,
    signature_meta: Any = None,
) -> list[Any]:
    if isinstance(args, tuple):
        raw_args = list(args)
    elif isinstance(args, list):
        raw_args = list(args)
    else:
        raw_args = [args]

    raw_kwargs = kwargs if isinstance(kwargs, dict) else {}
    orig_params = signature_params_for_call(
        op_name=op_name,
        func=func,
        signature_meta=signature_meta,
    )
    if not orig_params:
        return raw_args

    param_names = [name for name, _ in orig_params]
    resolved = {
        param_names[index]: value
        for index, value in enumerate(raw_args)
        if index < len(param_names)
    }
    resolved.update(raw_kwargs)

    ordered: list[Any] = []
    for param_name, default in orig_params:
        if param_name in resolved:
            ordered.append(resolved[param_name])
        elif default is not _EMPTY:
            ordered.append(default)
        else:
            ordered.append(None)
    return ordered


def prepare_launch_args(
    ordered_args: list[Any],
    launch_params: list[tuple[str, str]],
    *,
    ensure_device: str | None = None,
    force_contiguous: bool = False,
    adapter_stats: dict[str, int] | None = None,
) -> list[Any]:
    call_args: list[Any] = []
    limit = len(launch_params) or len(ordered_args)
    for index, value in enumerate(ordered_args[:limit]):
        param_name = ""
        param_type = ""
        if index < len(launch_params):
            param_name, param_type = launch_params[index]
            value = normalize_launch_value(param_name, param_type, value)
        if isinstance(value, torch.Tensor):
            if adapter_stats is not None:
                adapter_stats["tensor_arg_count"] = int(adapter_stats.get("tensor_arg_count", 0)) + 1
            tensor = value
            if ensure_device and tensor.device.type != ensure_device:
                tensor = tensor.to(ensure_device)
                if adapter_stats is not None:
                    adapter_stats["device_move_count"] = int(adapter_stats.get("device_move_count", 0)) + 1
            if tensor.dim() > 0 and not tensor.is_contiguous():
                if adapter_stats is not None:
                    adapter_stats["noncontiguous_tensor_arg_count"] = (
                        int(adapter_stats.get("noncontiguous_tensor_arg_count", 0)) + 1
                    )
                if force_contiguous:
                    tensor = tensor.contiguous()
                    if adapter_stats is not None:
                        adapter_stats["contiguous_copy_count"] = (
                            int(adapter_stats.get("contiguous_copy_count", 0)) + 1
                        )
                elif adapter_stats is not None:
                    adapter_stats["preserved_layout_tensor_arg_count"] = (
                        int(adapter_stats.get("preserved_layout_tensor_arg_count", 0)) + 1
                    )
            call_args.append(tensor)
        else:
            call_args.append(value)
    return call_args


def invoke_kernel_launch(
    ext: Any,
    *,
    args: Any,
    kwargs: dict[str, Any] | None,
    launch_params: list[tuple[str, str]],
    op_name: str = "",
    func: Any = None,
    signature_meta: Any = None,
    ensure_device: str | None = None,
    force_contiguous: bool = False,
    adapter_stats: dict[str, int] | None = None,
) -> Any:
    ordered_args = resolve_ordered_args(
        args,
        kwargs,
        op_name=op_name,
        func=func,
        signature_meta=signature_meta,
    )
    call_args = prepare_launch_args(
        ordered_args,
        launch_params,
        ensure_device=ensure_device,
        force_contiguous=force_contiguous,
        adapter_stats=adapter_stats,
    )
    return ext.launch(*call_args)
