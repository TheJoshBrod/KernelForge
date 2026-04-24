from __future__ import annotations

import torch

from src.generator.prompts import prompts


class TinyGemmWeightQBitsTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value: torch.Tensor):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)


def test_quantized_tensor_subclasses_are_marked_in_prompt(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    weight = TinyGemmWeightQBitsTensor(torch.empty((4, 8), dtype=torch.bfloat16))
    calls = [
        {
            "function_name": "torch.nn.functional.linear",
            "args": [],
            "kwargs": {
                "input": torch.empty((2, 8), dtype=torch.bfloat16),
                "weight": weight,
                "bias": None,
            },
            "output": torch.empty((2, 4), dtype=torch.bfloat16),
            "signature": {
                "params": ["input", "weight", "bias"],
                "defaults": {"bias": None},
            },
        }
    ]

    spec = prompts.generate_function_spec_from_calls(calls, "torch.nn.functional.linear")
    weight_spec = next(param for param in spec["parameters"] if param["name"] == "weight")

    assert weight_spec["quantized_tensor"] is True
    assert "TinyGemmWeightQBitsTensor" in weight_spec["quantized_tensor_types"][0]

    prompt = prompts.generate_full_llm_prompt(calls, "torch.nn.functional.linear")

    assert "Quantized Tensor Warning" in prompt
    assert "A dense BF16 matmul kernel is incorrect" in prompt
