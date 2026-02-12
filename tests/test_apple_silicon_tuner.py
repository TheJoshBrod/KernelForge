from __future__ import annotations

import pytest

from src.apple_silicon.runtime_args import sanitize_runtime_args, split_runtime_args_and_env
from src.apple_silicon.tuner import _extract_json_payload, _normalize_kernel_overrides


def test_extract_json_payload_fenced() -> None:
    response = """```json
{"candidate_name":"c1","kernel_overrides":{"quant_matvec_decode":{"threadgroup":128}}}
```"""
    payload = _extract_json_payload(response)
    assert payload["candidate_name"] == "c1"


def test_extract_json_payload_rejects_non_json() -> None:
    with pytest.raises(Exception):
        _extract_json_payload("no json here")


def test_normalize_kernel_overrides_filters_invalid() -> None:
    overrides = _normalize_kernel_overrides(
        {
            "quant_matvec_decode": {
                "variant_name": "v1",
                "threadgroup": 128,
                "tile": [32, 16],
                "use_simdgroup": True,
                "notes": "ok",
            },
            "bad": "value",
        }
    )
    assert "quant_matvec_decode" in overrides
    assert overrides["quant_matvec_decode"]["threadgroup"] == 128
    assert "bad" not in overrides


def test_sanitize_runtime_args_allowlist() -> None:
    args = sanitize_runtime_args(["-fa", "-ub", "512", "--threads", "8", "--rm", "-rf"])
    assert args == ["--flash-attn", "on", "-ub", "512", "--threads", "8"]


def test_internal_long_vector_flag_maps_to_env() -> None:
    cli_args, env = split_runtime_args_and_env(
        ["--cgins-long-vector-schedule", "on", "--flash-attn", "on", "-ub", "512"]
    )
    assert cli_args == ["--flash-attn", "on", "-ub", "512"]
    assert env == {"CGINS_MUL_MV_LONG_VECTOR": "1"}
