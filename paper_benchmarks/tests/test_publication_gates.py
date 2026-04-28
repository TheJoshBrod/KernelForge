from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from paper_benchmarks.paper_bench.publication import certify_llm_publication_plan
from paper_benchmarks.paper_bench.schema import (
    BenchmarkMode,
    CorrectnessStatus,
    DeviceAuditArtifact,
    EnvironmentArtifact,
    RunManifestArtifact,
    Stage,
    Variant,
    validate_artifact_payload,
)
from paper_benchmarks.paper_bench.validator import validate_benchmark_artifact

from .helpers import build_common_payload


def _manifest(sample_paths, **updates):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "run_manifest",
            "benchmark_mode": BenchmarkMode.e2e_model,
            "variant": None,
            "stage": None,
            "warmup_count": 5,
            "timed_run_count": 20,
            "latency_samples_ms": [],
            "correctness_status": CorrectnessStatus.not_applicable,
            "run_dir": str(Path(sample_paths["suite_path"]).parent / "run"),
            "variants_requested": [Variant.eager, Variant.torch_compile, Variant.kf_cast],
            "stages_requested": [Stage.warmup, Stage.prefill, Stage.decode, Stage.total_generate],
            "description": "paper plan",
            "placement_profile": "single_cuda",
            "cache_mode": "kv_cache_on",
            "model_license": "Apache-2.0",
            "dataset_license": "CC-BY-4.0",
            "cast_package_path": sample_paths["cast_path"],
            "cast_package_hash": sample_paths["cast_hash"],
            "toolchain_status": {"jit_ready": True},
            "kf_settings": {
                "fail_on_fallback": True,
                "record_runtime_stats": True,
                "require_precompiled": True,
            },
        }
    )
    payload.update(updates)
    return RunManifestArtifact.model_validate(payload)


def _env(sample_paths):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "environment_snapshot",
            "benchmark_mode": BenchmarkMode.e2e_model,
            "variant": None,
            "stage": None,
            "warmup_count": 5,
            "timed_run_count": 20,
            "latency_samples_ms": [],
            "correctness_status": CorrectnessStatus.not_applicable,
            "platform": "Linux-test",
            "platform_release": "6.0",
            "machine": "x86_64",
            "torch_cuda_available": True,
            "torch_mps_available": False,
        }
    )
    return EnvironmentArtifact.model_validate(payload)


def _model_spec():
    return SimpleNamespace(
        model_id="unit_test_model",
        model_path="/tmp/model",
        model_config_path="/tmp/model/config.json",
        quantization="bf16",
        device_map="single_cuda",
        paper_eligible=True,
        benchmark_expectation="paper",
        baselines_required=[Variant.eager, Variant.torch_compile, Variant.kf_cast],
    )


def _suite():
    return SimpleNamespace(
        suite_id="paper_suite",
        benchmark_mode=BenchmarkMode.e2e_model,
        warmup_count=5,
        timed_run_count=20,
        max_new_tokens=128,
    )


def _metric_artifact(sample_paths, *, variant, correctness_status, details=None):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "benchmark_result",
            "variant": variant,
            "stage": Stage.total_generate,
            "correctness_status": correctness_status,
            "steady_state_time_ms": 10.0,
            "latency_summary": {
                "count": 2,
                "mean_ms": 1.5,
                "median_ms": 1.5,
                "p05_ms": 1.05,
                "p95_ms": 1.95,
                "min_ms": 1.0,
                "max_ms": 2.0,
                "stddev_ms": 0.5,
            },
            "sample_records": [
                {"prompt_id": "p0", "output_token_hashes": ["hash-0"]},
                {"prompt_id": "p0", "output_token_hashes": ["hash-1"]},
            ],
            "cast_package_path": sample_paths["cast_path"] if variant == Variant.kf_cast else None,
            "cast_package_hash": sample_paths["cast_hash"] if variant == Variant.kf_cast else None,
            "kf_settings": {
                "fail_on_fallback": True,
                "record_runtime_stats": True,
                "require_precompiled": True,
            }
            if variant == Variant.kf_cast
            else {},
            "fallback_count": 0 if variant == Variant.kf_cast else None,
            "kernel_hit_count": 10 if variant == Variant.kf_cast else None,
            "details": {
                "per_run_output_hash_verification": variant in {Variant.torch_compile, Variant.kf_cast},
                "correctness_checked_run_count": 2 if variant in {Variant.torch_compile, Variant.kf_cast} else 0,
                **(details or {}),
            },
        }
    )
    return validate_artifact_payload(payload)


def test_validator_rejects_failed_torch_compile_baseline(sample_paths):
    eager = _metric_artifact(sample_paths, variant=Variant.eager, correctness_status=CorrectnessStatus.reference)
    compiled = _metric_artifact(sample_paths, variant=Variant.torch_compile, correctness_status=CorrectnessStatus.failed)
    kf = _metric_artifact(
        sample_paths,
        variant=Variant.kf_cast,
        correctness_status=CorrectnessStatus.passed,
        details={
            "runtime_patch_enabled": True,
            "selected_ops": ["torch_nn_functional_gelu"],
            "per_op_launch_coverage": {
                "torch_nn_functional_gelu": {
                    "patched_calls": 10,
                    "kernel_launches_succeeded": 10,
                    "input_is_cuda": {"true": 10, "false": 0},
                    "fallbacks_to_original": 0,
                }
            },
            "cast_manifest": {
                "export_paper_eligible": False,
                "posthoc_patches": [{"op": "torch_nn_functional_gelu"}],
            },
        },
    )

    validation = validate_benchmark_artifact(kf, eager_baseline=eager, torch_compile_baseline=compiled)

    assert "torch_compile baseline correctness is not usable: failed" in validation["validation_errors"]
    assert "CAST manifest contains posthoc source patches" in validation["validation_warnings"]


def _deployment_cast_inspection(sample_paths):
    return {
        "cast_package_sha256": sample_paths["cast_hash"],
        "checksum_verified": True,
        "loadability_blockers": [],
        "export_paper_eligible": True,
        "uses_non_deployment_evidence": False,
        "selected_kernel_metadata": {
            "torch_nn_functional_linear": {
                "evidence_tier": "deployment",
                "benchmark_reference": {
                    "artifact_path": "/tmp/op_benchmarks.json",
                    "row_ref": "torch_nn_functional_linear#0",
                },
            }
        },
    }


def test_publication_plan_accepts_complete_paper_gate(sample_paths):
    certification = certify_llm_publication_plan(
        manifest=_manifest(sample_paths),
        env_artifact=_env(sample_paths),
        model_spec=_model_spec(),
        suite=_suite(),
        requested_variants=[Variant.eager, Variant.torch_compile, Variant.kf_cast],
        cast_inspection=_deployment_cast_inspection(sample_paths),
    )

    assert certification["paper_ready"] is True
    assert certification["fatal_errors"] == []
    assert certification["paper_blockers"] == []


def test_publication_plan_blocks_unverified_terms_and_manual_cast(sample_paths):
    cast_inspection = _deployment_cast_inspection(sample_paths)
    cast_inspection["export_paper_eligible"] = False
    cast_inspection["selected_kernel_metadata"]["torch_nn_functional_linear"]["evidence_tier"] = "manual_override"

    certification = certify_llm_publication_plan(
        manifest=_manifest(
            sample_paths,
            model_license=None,
            model_access_terms="local mirror; upstream license/access terms must be verified before paper",
            dataset_license=None,
            dataset_access_terms="public prompts; license must be verified",
        ),
        env_artifact=_env(sample_paths),
        model_spec=_model_spec(),
        suite=_suite(),
        requested_variants=[Variant.eager, Variant.torch_compile, Variant.kf_cast],
        cast_inspection=cast_inspection,
    )

    assert certification["paper_ready"] is False
    assert any("model license" in issue for issue in certification["paper_blockers"])
    assert any("dataset license" in issue for issue in certification["paper_blockers"])
    assert any("not deployment-paper eligible" in issue for issue in certification["paper_blockers"])
    assert any("lack deployment-tier evidence" in issue for issue in certification["paper_blockers"])


def test_device_audit_artifact_schema_round_trips(sample_paths):
    payload = build_common_payload(sample_paths)
    payload.update(
        {
            "artifact_type": "device_audit",
            "variant": "kf_cast",
            "stage": "total_generate",
            "audit_stage": "total_generate",
            "audit_status": "passed",
            "selected_ops": ["torch_nn_functional_linear"],
            "runtime_input_device": "cuda:0",
            "tokenizer_output_devices": {"input_ids": "cuda:0", "attention_mask": "cuda:0"},
            "placement_audit": {"first_parameter_device": "cuda:0"},
            "per_op_launch_coverage": {
                "torch_nn_functional_linear": {
                    "patched_calls": 2,
                    "kernel_launches_attempted": 2,
                    "kernel_launches_succeeded": 2,
                    "input_devices": {"cuda:0": 2},
                    "input_is_cuda": {"true": 2},
                }
            },
            "kernel_launches_attempted": 2,
            "kernel_launches_succeeded": 2,
            "kernel_launches_failed": 0,
            "fallback_count": 0,
        }
    )

    artifact = validate_artifact_payload(payload)

    assert isinstance(artifact, DeviceAuditArtifact)
    assert artifact.audit_status == "passed"
