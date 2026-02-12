from __future__ import annotations

from pathlib import Path

import pytest

from src.apple_silicon.kernel_patch import (
    KernelPatchError,
    build_kernel_patch_candidate,
    classify_compile_record,
    classify_correctness_record,
    variant_family_patch_set,
)
from src.apple_silicon.types import BenchmarkMetrics, BenchmarkResult, WorkloadProfile


def _write_fake_llamacpp_root(root: Path) -> None:
    metal_dir = root / "ggml" / "src" / "ggml-metal"
    metal_dir.mkdir(parents=True, exist_ok=True)
    (root / "ggml" / "src").mkdir(parents=True, exist_ok=True)

    (metal_dir / "ggml-metal.metal").write_text(
        "\n".join(
            [
                "#define N_SIMDWIDTH 32 // assuming SIMD group size is 32",
                "kernel void k() {",
                "  int needle = 1;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (metal_dir / "ggml-metal-impl.h").write_text(
        "\n".join(
            [
                "#define N_R0_Q4_K 2",
                "#define N_R0_Q5_K 2",
                "#define N_R0_Q6_K 2",
            ]
        ),
        encoding="utf-8",
    )
    (root / "ggml" / "src" / "ggml-common.h").write_text("// common", encoding="utf-8")


def _mk_result(*, rc: int, prefill: float | None, decode: float | None, stderr: str = "") -> BenchmarkResult:
    profile = WorkloadProfile(name="chat", ctx=8192, prompt_tokens_target=1024, generate_tokens=16, repeats=1)
    metrics = BenchmarkMetrics(
        prefill_tokens_per_sec=prefill,
        decode_tokens_per_sec=decode,
        ttft_ms=10.0 if decode is not None else None,
        p50_token_latency_ms=1.0,
        p95_token_latency_ms=1.0,
        peak_memory_mib=10.0,
    )
    return BenchmarkResult(
        profile=profile,
        metrics=metrics,
        elapsed_seconds=0.01,
        runs=[
            {
                "return_code": rc,
                "prefill_tokens_per_sec": prefill,
                "decode_tokens_per_sec": decode,
                "stdout": "",
                "stderr": stderr,
            }
        ],
    )


def test_build_kernel_patch_candidate_applies_mutations(tmp_path: Path) -> None:
    llama_root = tmp_path / "llama.cpp"
    _write_fake_llamacpp_root(llama_root)
    cache_dir = tmp_path / "cache"

    cand_a = build_kernel_patch_candidate(
        llamacpp_root=llama_root,
        candidate_cache_dir=cache_dir,
        candidate_id="cand",
        template_mutations={"n_r0_q4_k": 3, "n_r0_q5_k": 4},
        source_patches=[
            {
                "patch_id": "p1",
                "file": "ggml-metal.metal",
                "find": "int needle = 1;",
                "replace": "int needle = 2;",
            }
        ],
    )
    cand_b = build_kernel_patch_candidate(
        llamacpp_root=llama_root,
        candidate_cache_dir=cache_dir,
        candidate_id="cand",
        template_mutations={"n_r0_q4_k": 3, "n_r0_q5_k": 4},
        source_patches=[
            {
                "patch_id": "p1",
                "file": "ggml-metal.metal",
                "find": "int needle = 1;",
                "replace": "int needle = 2;",
            }
        ],
    )

    assert cand_a.patch_hash == cand_b.patch_hash
    impl_text = (Path(cand_a.resources_dir) / "ggml-metal-impl.h").read_text(encoding="utf-8")
    assert "#define N_R0_Q4_K 3" in impl_text
    assert "#define N_R0_Q5_K 4" in impl_text
    metal_text = (Path(cand_a.resources_dir) / "ggml-metal.metal").read_text(encoding="utf-8")
    assert "int needle = 2;" in metal_text


def test_build_kernel_patch_candidate_rejects_unsafe_patch(tmp_path: Path) -> None:
    llama_root = tmp_path / "llama.cpp"
    _write_fake_llamacpp_root(llama_root)
    with pytest.raises(KernelPatchError):
        build_kernel_patch_candidate(
            llamacpp_root=llama_root,
            candidate_cache_dir=tmp_path / "cache",
            candidate_id="bad",
            template_mutations={},
            source_patches=[
                {
                    "patch_id": "bad",
                    "file": "ggml-metal.metal",
                    "find": "not-present",
                    "replace": "x",
                }
            ],
        )


def test_compile_and_correctness_classification() -> None:
    ok = _mk_result(rc=0, prefill=100.0, decode=50.0)
    fail = _mk_result(rc=1, prefill=None, decode=None, stderr="error: failed to initialize the Metal library")

    compile_ok = classify_compile_record(ok)
    assert compile_ok.success
    assert compile_ok.classification == "compiled_or_loaded"

    compile_fail = classify_compile_record(fail)
    assert not compile_fail.success
    assert compile_fail.classification == "metal_compile_error"

    corr_ok = classify_correctness_record(baseline=ok, candidate=ok)
    assert corr_ok.success

    outlier = _mk_result(rc=0, prefill=10000.0, decode=5000.0)
    corr_outlier = classify_correctness_record(baseline=ok, candidate=outlier)
    assert not corr_outlier.success


def test_correctness_strict_parity_detects_mismatch() -> None:
    base = _mk_result(rc=0, prefill=100.0, decode=50.0)
    cand = _mk_result(rc=0, prefill=100.0, decode=50.0)
    base.runs[0]["stdout"] = "> prompt\n\nhello world\n[ Prompt: 100.0 t/s | Generation: 50.0 t/s ]"
    cand.runs[0]["stdout"] = "> prompt\n\nhello mars\n[ Prompt: 100.0 t/s | Generation: 50.0 t/s ]"
    result = classify_correctness_record(
        baseline=base,
        candidate=cand,
        strict_parity=True,
        similarity_threshold=0.99,
    )
    assert not result.success
    assert result.classification == "semantic_mismatch"


def test_variant_family_patch_set_exists() -> None:
    payload = variant_family_patch_set("mul_mv_q5k_decode_aggr")
    assert payload["name"] == "mul_mv_q5k_decode_aggr"
    assert payload["template_mutations"]["n_r0_q5_k"] == 4
