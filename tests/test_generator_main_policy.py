from __future__ import annotations

import torch

from src.generator import main as generator_main


class TinyGemmWeightQBitsTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value: torch.Tensor):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)


def test_prompt_entry_sampling_keeps_shape_spread() -> None:
    files = [f"entry_{idx:06d}.pt" for idx in range(10)]

    sampled = generator_main._sample_entry_files_for_prompt(files, max_entries=4)

    assert sampled == [
        "entry_000000.pt",
        "entry_000003.pt",
        "entry_000006.pt",
        "entry_000009.pt",
    ]


def test_quantized_linear_requires_mixed_fallback() -> None:
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
            "signature": {
                "params": ["input", "weight", "bias"],
                "defaults": {"bias": None},
            },
        }
    ]

    fallback = generator_main._quantized_linear_fallback(
        calls,
        "torch.nn.functional.linear",
    )

    assert fallback is not None
    assert fallback["reason"] == "quantized_linear_packed_weight"
    assert fallback["cast_policy"]["full_forged_publishable"] is False
    assert fallback["cast_policy"]["mixed_forged_publishable"] is True
    assert fallback["cast_policy"]["torch_fallback_ops"] == [
        "torch_nn_functional_linear"
    ]
