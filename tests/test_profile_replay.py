from __future__ import annotations

from src.optimizer.profile_replay import normalize_args_kwargs, normalize_profile_call_args


def test_legacy_grouped_mm_profile_replay_keeps_keyword_only_args() -> None:
    payload = {
        "function_name": "torch.nn.functional.grouped_mm",
        "args": [],
        "kwargs": {
            "mat_a": "a",
            "mat_b": "b",
            "offs": "offsets",
            "bias": None,
            "out_dtype": None,
        },
        "signature": {
            "params": ["mat_a", "mat_b", "offs", "bias", "out_dtype"],
            "defaults": {"offs": None, "bias": None, "out_dtype": None},
        },
    }

    args, kwargs = normalize_profile_call_args(payload)

    assert args == ["a", "b"]
    assert kwargs == {"offs": "offsets", "bias": None, "out_dtype": None}


def test_generated_kernel_launch_can_flatten_grouped_mm_profile() -> None:
    payload = {
        "function_name": "torch.nn.functional.grouped_mm",
        "args": [],
        "kwargs": {
            "mat_a": "a",
            "mat_b": "b",
            "offs": "offsets",
            "bias": None,
            "out_dtype": None,
        },
        "signature": {
            "params": ["mat_a", "mat_b", "offs", "bias", "out_dtype"],
            "defaults": {"offs": None, "bias": None, "out_dtype": None},
        },
    }

    args, kwargs = normalize_profile_call_args(payload, preserve_keyword_only=False)

    assert args == ["a", "b", "offsets", None, None]
    assert kwargs == {}


def test_signature_kinds_preserve_keyword_only_for_any_op() -> None:
    args, kwargs = normalize_args_kwargs(
        [],
        {"x": 1, "y": 2},
        {
            "params": ["x", "y"],
            "defaults": {"y": 0},
            "kinds": {"x": "POSITIONAL_OR_KEYWORD", "y": "KEYWORD_ONLY"},
        },
        function_name="example.op",
    )

    assert args == [1]
    assert kwargs == {"y": 2}


def test_linear_profile_replay_is_unchanged() -> None:
    args, kwargs = normalize_args_kwargs(
        [],
        {"input": "input", "weight": "weight", "bias": None},
        {
            "params": ["input", "weight", "bias"],
            "defaults": {"bias": None},
            "kinds": {
                "input": "POSITIONAL_OR_KEYWORD",
                "weight": "POSITIONAL_OR_KEYWORD",
                "bias": "POSITIONAL_OR_KEYWORD",
            },
        },
        function_name="torch.nn.functional.linear",
    )

    assert args == ["input", "weight", None]
    assert kwargs == {}
