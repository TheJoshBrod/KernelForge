"""Helpers for quantized tensor subclasses captured in Kernel Forge traces."""

from __future__ import annotations

from typing import Any, Callable

import torch


TINY_GEMM_LINEAR_ABI = "quanto_tinygemm_int4_linear_v1"
TINY_GEMM_LINEAR_SIGNATURE = (
    "torch::Tensor launch(torch::Tensor input, torch::Tensor packed_weight, "
    "torch::Tensor scale_shift, c10::optional<torch::Tensor> bias, "
    "int64_t out_features, int64_t in_features, int64_t group_size, int64_t axis)"
)

_LINEAR_NAMES = {
    "linear",
    "aten.linear",
    "aten::linear",
    "torch.nn.functional.linear",
    "torch_nn_functional_linear",
}


def _type_name(value: Any) -> str:
    cls = type(value)
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__qualname__", getattr(cls, "__name__", str(cls)))
    return f"{module}.{name}" if module else str(name)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "tensor_subclass": _type_name(tensor),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
    }


def is_linear_function(function_name: str | None) -> bool:
    if not function_name:
        return False
    normalized = str(function_name).strip()
    return normalized in _LINEAR_NAMES or normalized.endswith(".linear")


def is_tinygemm_qbits_tensor(value: Any) -> bool:
    """Return True for Quanto TinyGemm packed weight tensor subclasses.

    The captured object can be fragile to stringify; detection intentionally uses
    attribute presence and tensor-ness, not repr().
    """

    if not torch.is_tensor(value):
        return False
    if not all(hasattr(value, attr) for attr in ("_data", "_scale_shift", "_group_size", "_axis")):
        return False
    packed = getattr(value, "_data", None)
    scale_shift = getattr(value, "_scale_shift", None)
    return torch.is_tensor(packed) and torch.is_tensor(scale_shift)


def describe_tinygemm_qbits_tensor(value: Any) -> dict[str, Any] | None:
    if not is_tinygemm_qbits_tensor(value):
        return None

    packed = getattr(value, "_data")
    scale_shift = getattr(value, "_scale_shift")
    logical_shape = list(value.shape)
    out_features = _safe_int(logical_shape[0]) if len(logical_shape) >= 1 else 0
    in_features = _safe_int(logical_shape[1]) if len(logical_shape) >= 2 else 0

    return {
        "kernel_abi": TINY_GEMM_LINEAR_ABI,
        "tensor_subclass": _type_name(value),
        "logical_dtype": str(value.dtype),
        "logical_shape": logical_shape,
        "logical_stride": list(value.stride()),
        "logical_numel": int(value.numel()),
        "packed": _tensor_meta(packed),
        "scale_shift": _tensor_meta(scale_shift),
        "group_size": _safe_int(getattr(value, "_group_size", 0)),
        "axis": _safe_int(getattr(value, "_axis", 0)),
        "out_features": out_features,
        "in_features": in_features,
    }


def _param_map(args: list[Any] | tuple[Any, ...], kwargs: dict[str, Any] | None, signature: dict[str, Any] | None) -> dict[str, Any]:
    params = list((signature or {}).get("params") or [])
    mapped: dict[str, Any] = {}
    for idx, value in enumerate(args):
        name = params[idx] if idx < len(params) else f"arg{idx}"
        mapped[name] = value
    if kwargs:
        mapped.update(kwargs)
    return mapped


def tinygemm_linear_weight(
    function_name: str | None,
    args: list[Any] | tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    signature: dict[str, Any] | None = None,
) -> Any | None:
    if not is_linear_function(function_name):
        return None

    mapped = _param_map(args, kwargs, signature)
    weight = mapped.get("weight")
    if weight is None and len(args) > 1:
        weight = args[1]
    return weight if is_tinygemm_qbits_tensor(weight) else None


def tinygemm_linear_abi(
    function_name: str | None,
    args: list[Any] | tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    signature: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    weight = tinygemm_linear_weight(function_name, args, kwargs, signature)
    desc = describe_tinygemm_qbits_tensor(weight) if weight is not None else None
    if desc is None:
        return None
    return {
        "name": TINY_GEMM_LINEAR_ABI,
        "launch_signature": TINY_GEMM_LINEAR_SIGNATURE,
        "weight": desc,
    }


def prepare_tinygemm_linear_launch_args(
    function_name: str | None,
    args: list[Any] | tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    signature: dict[str, Any] | None = None,
    *,
    move_to_device: Callable[[Any], Any] | None = None,
) -> list[Any] | None:
    """Expand a captured F.linear(input, TinyGemmWeightQBitsTensor, bias) call.

    Returns None unless the call is a TinyGemm INT4 linear.  The returned list is
    the internal Kernel Forge launch ABI:
    input, packed_weight, scale_shift, bias, out_features, in_features,
    group_size, axis.
    """

    if not is_linear_function(function_name):
        return None

    mapped = _param_map(args, kwargs, signature)
    input_tensor = mapped.get("input", args[0] if args else None)
    weight = mapped.get("weight", args[1] if len(args) > 1 else None)
    bias = mapped.get("bias", args[2] if len(args) > 2 else None)

    desc = describe_tinygemm_qbits_tensor(weight)
    if desc is None:
        return None

    move = move_to_device or (lambda item: item)
    packed = getattr(weight, "_data")
    scale_shift = getattr(weight, "_scale_shift")
    return [
        move(input_tensor),
        move(packed),
        move(scale_shift),
        move(bias) if torch.is_tensor(bias) else bias,
        int(desc["out_features"]),
        int(desc["in_features"]),
        int(desc["group_size"]),
        int(desc["axis"]),
    ]


def detect_kernel_source_abi(function_name: str | None, cuda_source: str) -> str | None:
    """Best-effort ABI marker for exported CAST metadata."""

    if not is_linear_function(function_name):
        return None
    if "packed_weight" in cuda_source and "scale_shift" in cuda_source and "group_size" in cuda_source:
        return TINY_GEMM_LINEAR_ABI
    return None
