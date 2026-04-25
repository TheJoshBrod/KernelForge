from __future__ import annotations

import torch

from src.generator.prompts import prompts
from src.optimizer import quantized


class TinyGemmWeightQBitsTensor(torch.Tensor):
    pass


class TinyGemmPackedTensor(torch.Tensor):
    pass


def _fake_tinygemm_packed(raw: torch.Tensor) -> torch.Tensor:
    logical = torch.empty_strided((4, 8), (8, 1), dtype=torch.uint8, device=raw.device)
    packed = torch.Tensor._make_subclass(TinyGemmPackedTensor, logical, False)
    packed._data = raw
    return packed


def _fake_tinygemm_weight() -> torch.Tensor:
    logical = torch.empty((4, 8), dtype=torch.bfloat16)
    weight = torch.Tensor._make_subclass(TinyGemmWeightQBitsTensor, logical, False)
    raw = torch.empty_strided((4, 4), (4, 1), dtype=torch.uint8)
    weight._data = _fake_tinygemm_packed(raw)
    weight._scale_shift = torch.empty_strided((1, 4, 2), (8, 2, 1), dtype=torch.bfloat16)
    weight._group_size = 128
    weight._axis = 0
    return weight


def test_describes_tinygemm_qbits_without_repr() -> None:
    weight = _fake_tinygemm_weight()

    desc = quantized.describe_tinygemm_qbits_tensor(weight)

    assert desc is not None
    assert desc["kernel_abi"] == quantized.TINY_GEMM_LINEAR_ABI
    assert desc["logical_dtype"] == "torch.bfloat16"
    assert desc["logical_shape"] == [4, 8]
    assert desc["packed"]["dtype"] == "torch.uint8"
    assert desc["packed"]["shape"] == [4, 4]
    assert desc["packed_raw"]["shape"] == [4, 4]
    assert desc["packed_wrapper"]["shape"] == [4, 8]
    assert desc["packed_layout"] == quantized.TINY_GEMM_LINEAR_PACKED_LAYOUT
    assert desc["logical_unpack_dequant"] == quantized.TINY_GEMM_LINEAR_DEQUANT_FORMULA
    assert desc["scale_shift"]["shape"] == [1, 4, 2]
    assert desc["group_size"] == 128
    assert desc["axis"] == 0


def test_expands_only_tinygemm_linear_launch_args() -> None:
    input_tensor = torch.empty((2, 8), dtype=torch.bfloat16)
    weight = _fake_tinygemm_weight()
    bias = torch.empty((4,), dtype=torch.bfloat16)
    moved: list[torch.Tensor] = []
    moved_raw = torch.empty_strided((4, 4), (4, 1), dtype=torch.uint8)
    moved_wrapper = _fake_tinygemm_packed(moved_raw)

    def move(value):
        if torch.is_tensor(value):
            moved.append(value)
        if value is weight._data:
            return moved_wrapper
        return value

    launch_args = quantized.prepare_tinygemm_linear_launch_args(
        "torch.nn.functional.linear",
        [input_tensor, weight, bias],
        {},
        {"params": ["input", "weight", "bias"], "defaults": {"bias": None}},
        move_to_device=move,
    )

    assert launch_args is not None
    assert launch_args[0] is input_tensor
    assert launch_args[1] is moved_raw
    assert launch_args[2] is weight._scale_shift
    assert launch_args[3] is bias
    assert launch_args[4:] == [4, 8, 128, 0]
    assert moved[0] is weight._data
    assert moved[1] is input_tensor
    assert moved[2] is weight._scale_shift
    assert moved[3] is bias
    assert quantized.prepare_tinygemm_linear_launch_args(
        "torch.nn.functional.softmax",
        [input_tensor, weight, bias],
        {},
        {"params": ["input", "weight", "bias"]},
    ) is None
    assert quantized.prepare_tinygemm_linear_launch_args(
        "torch.nn.functional.linear",
        [input_tensor, torch.empty((4, 8), dtype=torch.bfloat16), bias],
        {},
        {"params": ["input", "weight", "bias"]},
    ) is None


def test_prompt_exposes_special_abi_for_tinygemm_linear(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    input_tensor = torch.empty((2, 8), dtype=torch.bfloat16)
    weight = _fake_tinygemm_weight()
    output = torch.empty((2, 4), dtype=torch.bfloat16)
    calls = [
        {
            "function_name": "torch.nn.functional.linear",
            "args": [input_tensor, weight],
            "kwargs": {"bias": None},
            "output": output,
            "signature": {"params": ["input", "weight", "bias"], "defaults": {"bias": None}},
        }
    ]

    spec = prompts.generate_function_spec_from_calls(calls, "torch.nn.functional.linear")
    prompt = prompts.format_operator_prompt(spec)

    assert spec["kernel_abi"]["name"] == quantized.TINY_GEMM_LINEAR_ABI
    assert "Kernel Forge Special ABI" in prompt
    assert "packed_weight_raw" in prompt
    assert "scale_shift" in prompt
    assert "_weight_int4pack_mm" in prompt
    assert "row-major low/high nibbles" in prompt
    assert "Do not read the logical weight as dense BF16" in prompt


def test_detects_tinygemm_linear_abi_versions() -> None:
    source_v2 = """
torch::Tensor launch(torch::Tensor input, torch::Tensor packed_weight_raw,
                     torch::Tensor scale_shift, c10::optional<torch::Tensor> bias,
                     int64_t out_features, int64_t in_features,
                     int64_t group_size, int64_t axis) {
    return at::_ops::_weight_int4pack_mm::call(input, packed_weight_raw, group_size, scale_shift);
}
"""
    source_v1 = """
torch::Tensor launch(torch::Tensor input, torch::Tensor packed_weight,
                     torch::Tensor scale_shift, c10::optional<torch::Tensor> bias,
                     int64_t out_features, int64_t in_features,
                     int64_t group_size, int64_t axis) {
    return input;
}
"""

    assert quantized.detect_kernel_source_abi("torch.nn.functional.linear", source_v2) == quantized.TINY_GEMM_LINEAR_ABI
    assert (
        quantized.detect_kernel_source_abi("torch.nn.functional.linear", source_v1)
        == quantized.TINY_GEMM_LINEAR_ABI_V1
    )
    assert quantized.detect_kernel_source_abi("torch.nn.functional.softmax", source_v2) is None


def test_tinygemm_linear_uses_scoped_validation_tolerance() -> None:
    assert quantized.validation_tolerances_for_kernel_abi(quantized.TINY_GEMM_LINEAR_ABI) == (
        quantized.TINY_GEMM_LINEAR_ATOL,
        quantized.TINY_GEMM_LINEAR_RTOL,
    )
    assert quantized.validation_tolerances_for_kernel_abi(None) == (
        quantized.DEFAULT_FLOAT_ATOL,
        quantized.DEFAULT_FLOAT_RTOL,
    )


def test_prompt_leaves_dense_linear_signature_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    calls = [
        {
            "function_name": "torch.nn.functional.linear",
            "args": [
                torch.empty((2, 8), dtype=torch.bfloat16),
                torch.empty((4, 8), dtype=torch.bfloat16),
                None,
            ],
            "kwargs": {},
            "output": torch.empty((2, 4), dtype=torch.bfloat16),
            "signature": {"params": ["input", "weight", "bias"], "defaults": {"bias": None}},
        }
    ]

    spec = prompts.generate_function_spec_from_calls(calls, "torch.nn.functional.linear")
    prompt = prompts.format_operator_prompt(spec)

    assert "kernel_abi" not in spec
    assert "Kernel Forge Special ABI" not in prompt
    assert "Must accept ALL parameters listed above in the exact order" in prompt
