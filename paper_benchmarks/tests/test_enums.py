from __future__ import annotations

from paper_benchmarks.paper_bench.schema import BenchmarkMode, Stage, Variant


def test_variant_enum_values_are_frozen():
    assert {member.value for member in Variant} == {"eager", "torch_compile", "kf_cast"}


def test_stage_enum_values_are_frozen():
    assert {member.value for member in Stage} == {
        "operator",
        "prefill",
        "decode",
        "total_generate",
        "load",
        "compile",
        "warmup",
    }


def test_benchmark_mode_enum_values_are_frozen():
    assert {member.value for member in BenchmarkMode} == {
        "operator",
        "e2e_model",
        "deployment",
    }
