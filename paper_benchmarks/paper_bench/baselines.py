from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class CompileSettings:
    backend: str = "inductor"
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
        }


@dataclass(frozen=True)
class TimedCallResult:
    elapsed_ms: float
    value: Any


def resolve_torch_dtype(dtype_name: str | None):
    if not dtype_name:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]


def compile_settings_from_dict(payload: dict[str, Any] | None) -> CompileSettings:
    if not payload:
        return CompileSettings()
    return CompileSettings(
        backend=str(payload.get("backend") or "inductor"),
        mode=str(payload["mode"]) if payload.get("mode") is not None else None,
        fullgraph=bool(payload.get("fullgraph", False)),
        dynamic=bool(payload.get("dynamic", False)),
    )


def sync_device(device: str | None) -> None:
    target = (device or "").lower()
    if target.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif target == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def timed_call(
    device: str | None,
    fn: Callable[[], Any],
    *,
    inference_mode: bool = True,
) -> TimedCallResult:
    sync_device(device)
    start = time.perf_counter()
    if inference_mode:
        with torch.inference_mode():
            value = fn()
    else:
        value = fn()
    sync_device(device)
    return TimedCallResult(
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        value=value,
    )


def load_transformers_causal_lm(model_spec, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_torch_dtype(model_spec.torch_dtype)
    local_files_only = bool(getattr(model_spec, "local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.tokenizer_path or model_spec.model_path,
        local_files_only=local_files_only,
        trust_remote_code=model_spec.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": model_spec.trust_remote_code,
        "torch_dtype": dtype,
    }
    if getattr(model_spec, "attn_implementation", None):
        model_kwargs["attn_implementation"] = model_spec.attn_implementation
    quantization_config = getattr(model_spec, "quantization_config", None)
    if isinstance(quantization_config, dict) and quantization_config:
        if str(quantization_config.get("type") or "").lower() == "quanto":
            from transformers import QuantoConfig

            config_payload = {key: value for key, value in quantization_config.items() if key != "type"}
            model_kwargs["quantization_config"] = QuantoConfig(**config_payload)
        else:
            model_kwargs["quantization_config"] = dict(quantization_config)
    used_device_map = False
    if getattr(model_spec, "device_map", None) is not None:
        model_kwargs["device_map"] = model_spec.device_map
        used_device_map = True
    elif getattr(model_spec, "placement_profile", None) == "single_cuda" and device and str(device).startswith("cuda"):
        model_kwargs["device_map"] = {"": "cuda:0"}
        used_device_map = True
    if getattr(model_spec, "max_memory", None):
        model_kwargs["max_memory"] = model_spec.max_memory
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_spec.model_path, **model_kwargs)
    model.eval()
    if device and not used_device_map:
        model.to(device)
    load_ms = (time.perf_counter() - start) * 1000.0
    return model, tokenizer, load_ms


def compile_model(
    model: Any,
    settings: CompileSettings | dict[str, Any] | None = None,
) -> tuple[Any, float]:
    compile_settings = compile_settings_from_dict(settings if isinstance(settings, dict) else (
        settings.as_dict() if isinstance(settings, CompileSettings) else None
    ))
    start = time.perf_counter()
    compiled = torch.compile(
        model,
        backend=compile_settings.backend,
        mode=compile_settings.mode,
        fullgraph=compile_settings.fullgraph,
        dynamic=compile_settings.dynamic,
    )
    compile_ms = (time.perf_counter() - start) * 1000.0
    return compiled, compile_ms
