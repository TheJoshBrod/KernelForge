from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 2
TENSOR_DESCRIPTOR_KEY = "__kforge_tensor_descriptor__"
RECOMPUTE_OUTPUT_KEY = "__kforge_recompute_output__"


def _normalize_device(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    value = str(device).strip().lower()
    if value in {"gpu", "cuda", "triton"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "mps":
        return "mps" if hasattr(torch, "backends") and torch.backends.mps.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    return value or "cpu"


def _dtype_from_string(value: str) -> torch.dtype:
    name = str(value or "torch.float32")
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else torch.float32


def _stable_seed(parts: list[Any]) -> int:
    raw = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)


def tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
        "numel": int(tensor.numel()),
        "storage_offset": int(tensor.storage_offset()),
    }


def is_tensor_descriptor(value: Any) -> bool:
    return isinstance(value, dict) and value.get(TENSOR_DESCRIPTOR_KEY) is True


def is_recompute_output(value: Any) -> bool:
    return isinstance(value, dict) and value.get(RECOMPUTE_OUTPUT_KEY) is True


def descriptor_meta(value: Any) -> dict[str, Any]:
    if is_recompute_output(value):
        meta = value.get("meta")
        return meta if isinstance(meta, dict) else {}
    if is_tensor_descriptor(value):
        meta = value.get("meta")
        return meta if isinstance(meta, dict) else {}
    return {}


def contains_tensor_descriptor(value: Any) -> bool:
    if is_tensor_descriptor(value) or is_recompute_output(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_tensor_descriptor(item) for item in value)
    if isinstance(value, dict):
        return any(contains_tensor_descriptor(item) for item in value.values())
    return False


def tensor_descriptor(
    kind: str,
    tensor: torch.Tensor,
    *,
    name: str = "",
    role: str = "",
    key: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    meta = tensor_meta(tensor)
    return {
        TENSOR_DESCRIPTOR_KEY: True,
        "kind": kind,
        "name": name,
        "role": role,
        "key": key,
        "meta": meta,
        "synthetic_seed": int(seed if seed is not None else _stable_seed([kind, name, role, key, meta])),
    }


def recompute_output_descriptor(function_name: str, output: Any) -> dict[str, Any]:
    meta = tensor_meta(output) if torch.is_tensor(output) else {"type": type(output).__name__}
    return {
        RECOMPUTE_OUTPUT_KEY: True,
        "function_name": function_name,
        "meta": meta,
    }


def _storage_size(shape: list[int], stride: list[int], storage_offset: int) -> int:
    if not shape:
        return max(1, storage_offset + 1)
    if any(int(dim) == 0 for dim in shape):
        return 0
    max_index = int(storage_offset)
    for dim, st in zip(shape, stride):
        max_index += (int(dim) - 1) * int(st)
    return max(0, max_index + 1)


def _make_base_tensor(size: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    if size <= 0:
        return torch.empty((0,), dtype=dtype, device=device)

    gen_device = device if device in {"cpu", "cuda"} else "cpu"
    try:
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(seed)
    except Exception:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    def _on_device(fn):
        try:
            return fn(device=device, generator=generator)
        except Exception:
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(seed)
            return fn(device="cpu", generator=cpu_generator).to(device)

    if dtype.is_floating_point:
        return _on_device(lambda **kw: torch.randn((size,), dtype=dtype, **kw))
    if getattr(dtype, "is_complex", False):
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = _on_device(lambda **kw: torch.randn((size,), dtype=real_dtype, **kw))
        imag = _on_device(lambda **kw: torch.randn((size,), dtype=real_dtype, **kw))
        return torch.complex(real, imag).to(dtype)
    if dtype == torch.bool:
        return _on_device(lambda **kw: torch.randint(0, 2, (size,), dtype=torch.int8, **kw)).to(torch.bool)
    if dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        return _on_device(lambda **kw: torch.randint(0, 97, (size,), dtype=dtype, **kw))

    return torch.zeros((size,), dtype=dtype, device=device)


def materialize_tensor_descriptor(value: dict[str, Any], device: str | torch.device | None = None) -> torch.Tensor:
    meta = descriptor_meta(value)
    shape = [int(dim) for dim in meta.get("shape", [])]
    stride_raw = meta.get("stride")
    stride = [int(st) for st in stride_raw] if isinstance(stride_raw, list) else []
    dtype = _dtype_from_string(str(meta.get("dtype", "torch.float32")))
    target_device = _normalize_device(device or meta.get("device") or "cpu")
    storage_offset = int(meta.get("storage_offset", 0) or 0)
    seed = int(value.get("synthetic_seed") or _stable_seed([value.get("kind"), value.get("name"), meta]))

    if not stride or len(stride) != len(shape):
        base = _make_base_tensor(max(1, int(meta.get("numel", 1) or 1)), dtype, target_device, seed)
        return base[: int(meta.get("numel", 1) or 1)].reshape(shape)

    storage_size = _storage_size(shape, stride, storage_offset)
    if storage_size <= 0:
        return torch.empty_strided(shape, stride, dtype=dtype, device=target_device)
    base = _make_base_tensor(storage_size, dtype, target_device, seed)
    return torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)


def materialize_value(value: Any, device: str | torch.device | None = None) -> Any:
    if is_tensor_descriptor(value):
        return materialize_tensor_descriptor(value, device=device)
    if torch.is_tensor(value):
        return value.to(_normalize_device(device)) if device is not None else value
    if isinstance(value, list):
        return [materialize_value(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_value(item, device=device) for item in value)
    if isinstance(value, dict):
        return {key: materialize_value(item, device=device) for key, item in value.items()}
    return value


def get_function(function_name: str):
    parts = str(function_name).split(".")
    if len(parts) < 2:
        raise ValueError(f"Cannot resolve function name: {function_name}")
    module_name = ".".join(parts[:-1])
    attr = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def materialize_profile_entry(
    entry: dict[str, Any],
    *,
    device: str | torch.device | None = None,
    recompute_output: bool = True,
) -> dict[str, Any]:
    out = dict(entry)
    args = materialize_value(entry.get("args", []), device=device)
    kwargs = materialize_value(entry.get("kwargs", {}) or {}, device=device)
    out["args"] = args
    out["kwargs"] = kwargs

    output = entry.get("output")
    if recompute_output and is_recompute_output(output):
        try:
            fn = get_function(str(output.get("function_name") or entry.get("function_name")))
            with torch.no_grad():
                out["output"] = fn(*args, **kwargs)
        except Exception as exc:
            out["output_recompute_error"] = str(exc)
            meta = descriptor_meta(output)
            if meta.get("shape") is not None:
                out["output"] = materialize_tensor_descriptor(
                    {
                        TENSOR_DESCRIPTOR_KEY: True,
                        "kind": "output_spec",
                        "name": "output",
                        "meta": meta,
                        "synthetic_seed": _stable_seed(["output", entry.get("function_name"), meta]),
                    },
                    device=device,
                )
            else:
                out["output"] = None
    else:
        out["output"] = materialize_value(output, device=device)

    return out


def load_profile_entry(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
    recompute_output: bool = True,
    materialize: bool = True,
) -> dict[str, Any]:
    try:
        entry = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        entry = torch.load(path, map_location=map_location)
    if not materialize or not isinstance(entry, dict):
        return entry
    return materialize_profile_entry(entry, device=device, recompute_output=recompute_output)
