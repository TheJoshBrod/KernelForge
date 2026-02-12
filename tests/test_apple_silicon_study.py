from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apple_silicon import benchmark, op_profile, study
from src.apple_silicon.types import BenchmarkMetrics, BenchmarkResult, DeviceProfile, WorkloadProfile


def _write_gguf_string(buf, value: str) -> None:
    data = value.encode("utf-8")
    buf.write(struct.pack("<Q", len(data)))
    buf.write(data)


def _write_gguf_kv_string(buf, key: str, value: str) -> None:
    _write_gguf_string(buf, key)
    buf.write(struct.pack("<I", 8))
    _write_gguf_string(buf, value)


def _write_gguf_kv_u32(buf, key: str, value: int) -> None:
    _write_gguf_string(buf, key)
    buf.write(struct.pack("<I", 4))
    buf.write(struct.pack("<I", value))


def _write_minimal_gguf(path: Path, *, arch: str = "qwen2", file_type: int = 14) -> None:
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 3))
        _write_gguf_kv_string(f, "general.architecture", arch)
        _write_gguf_kv_string(f, "general.name", path.stem)
        _write_gguf_kv_u32(f, "general.file_type", file_type)


def test_crossover_order_abba() -> None:
    assert study._crossover_orders("baseline", "flash", cycles=1) == [
        ("baseline", "flash"),
        ("flash", "baseline"),
        ("flash", "baseline"),
        ("baseline", "flash"),
    ]


def test_crossover_order_custom_blocks() -> None:
    rows = study._crossover_orders("baseline", "flash", cycles=2)
    assert len(rows) == 8
    assert rows[:4] == [
        ("baseline", "flash"),
        ("flash", "baseline"),
        ("flash", "baseline"),
        ("baseline", "flash"),
    ]


def test_generate_schedule_preview_writes_abba_pattern(tmp_path: Path) -> None:
    m1 = tmp_path / "m1.gguf"
    _write_minimal_gguf(m1)
    matrix = {"models": [{"id": "m1", "path": str(m1)}]}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    preview = study.generate_schedule_preview(
        matrix_path=matrix_path,
        profiles=["chat"],
        arms=["baseline", "flash"],
        abba_cycles=1,
        warmup_blocks=1,
    )
    rows = [r for r in preview["rows"] if not r.get("is_warmup")]
    assert [r["order_label"] for r in rows] == [
        "baseline->flash",
        "flash->baseline",
        "flash->baseline",
        "baseline->flash",
    ]


def test_result_valid_long_prompt_target_miss() -> None:
    profile = WorkloadProfile(name="long", ctx=32768, prompt_tokens_target=4096, generate_tokens=64, repeats=1)
    result = _fake_profile_result(profile=profile, args=[], capture_raw_output=False)
    result.runs[0]["prompt_tokens_target"] = 4096
    result.runs[0]["prompt_tokens_actual"] = 200
    ok, reason = study._result_valid(
        result,
        long_prompt_tolerance=16,
        require_long_prompt_target=True,
    )
    assert not ok
    assert reason == "long_prompt_target_miss"


def test_resolve_arms_alias_to_kernel_names() -> None:
    arms = study._resolve_arms("baseline,flash,oneshot,iterative")
    assert arms == ["baseline", "flash", "oneshot_kernel", "iterative_kernel"]


def test_kernel_budget_allocation_rules() -> None:
    both = study._kernel_budget_allocation(
        arms=["baseline", "oneshot_kernel", "iterative_kernel"],
        kernel_total_budget=20,
    )
    assert both == {"oneshot_kernel": 1, "iterative_kernel": 19}

    iterative_only = study._kernel_budget_allocation(
        arms=["baseline", "iterative_kernel"],
        kernel_total_budget=7,
    )
    assert iterative_only == {"oneshot_kernel": 0, "iterative_kernel": 7}

    default = study._kernel_budget_allocation(
        arms=["baseline", "oneshot_kernel", "iterative_kernel"],
        kernel_total_budget=0,
    )
    assert default == {"oneshot_kernel": 0, "iterative_kernel": 0}


def test_throughput_report_aggregation_handles_null_gates() -> None:
    report = study._build_throughput_report(
        [
            {
                "valid": False,
                "error": "static_feasibility_reject:unsupported_simd_width",
                "gate_a_pass": False,
                "gate_b_pass": None,
                "gate_c_pass": False,
                "gate_d_pass": False,
                "compile_record": {"stderr_hash": "abc"},
                "dispatch_audit": {},
            },
            {
                "valid": True,
                "gate_a_pass": True,
                "gate_b_pass": True,
                "gate_c_pass": True,
                "gate_d_pass": True,
                "compile_record": {"stderr_hash": ""},
                "dispatch_audit": {
                    "metallib_present": True,
                    "selected_dispatch_rule_id": "rule_1",
                    "top_kernels": [{"kernel": "mul_mat_fast", "mentions": 4}],
                },
            },
        ]
    )
    assert report["attempts_total"] == 2
    assert report["compile_success_rate"] == pytest.approx(0.5)
    assert report["gate_b_pass_rate"] == pytest.approx(1.0)
    assert report["gate_c_pass_rate"] == pytest.approx(0.5)
    assert report["gate_d_pass_rate"] == pytest.approx(0.5)
    assert report["top_rejection_reasons"][0]["reason"] == "static_feasibility_reject"
    assert report["top_compile_stderr_hashes"][0]["stderr_hash"] == "abc"
    assert report["top_dispatched_kernels"][0]["kernel"] == "mul_mat_fast"
    assert report["dispatch_metallib_load_rate"] == pytest.approx(1.0)
    assert report["dispatch_rule_coverage"]["unique_rule_count"] == 1


def test_stage1_compare_uses_op_params_and_backend() -> None:
    baseline_ops = [
        {"op": "mul_mat", "op_params": "type_a=q4_K,m=4096,n=3,k=14336", "backend": "MTL0", "time_ms": 1.0},
        {"op": "mul_mat", "op_params": "type_a=q5_K,m=4096,n=2,k=14336", "backend": "MTL0", "time_ms": 3.0},
    ]
    candidate_ops = [
        {"op": "mul_mat", "op_params": "type_a=q4_K,m=4096,n=3,k=14336", "backend": "MTL0", "time_ms": 0.5},
        {"op": "mul_mat", "op_params": "type_a=q5_K,m=4096,n=2,k=14336", "backend": "MTL0", "time_ms": 4.5},
    ]
    cmp_row = op_profile.compare_stage1_op_profiles(
        baseline_ops=baseline_ops,
        candidate_ops=candidate_ops,
    )
    assert cmp_row["status"] == "ok"
    assert cmp_row["compare_key"] == "op+op_params+backend"
    assert cmp_row["common_rows"] == 2
    assert cmp_row["baseline_total_ms"] == pytest.approx(4.0)
    assert cmp_row["candidate_total_ms"] == pytest.approx(5.0)
    assert cmp_row["delta_pct"] == pytest.approx(-25.0)


def test_throughput_report_uses_evaluated_denominator() -> None:
    report = study._build_throughput_report(
        [
            {
                "valid": False,
                "error": "static_feasibility_reject:unsupported_simd_width",
                "gate_a_evaluated": False,
                "gate_a_pass": None,
                "gate_b_evaluated": False,
                "gate_b_pass": None,
                "gate_c_evaluated": False,
                "gate_c_pass": None,
                "gate_d_evaluated": False,
                "gate_d_pass": None,
                "compile_record": {"stderr_hash": "abc"},
                "dispatch_audit": {},
            },
            {
                "valid": False,
                "error": "op_perf_regression:-12.0",
                "gate_a_evaluated": True,
                "gate_a_pass": True,
                "gate_b_evaluated": False,
                "gate_b_pass": None,
                "gate_c_evaluated": False,
                "gate_c_pass": None,
                "gate_d_evaluated": False,
                "gate_d_pass": None,
                "compile_record": {"stderr_hash": ""},
                "dispatch_audit": {},
            },
        ]
    )
    assert report["attempts_total"] == 2
    assert report["gate_a_evaluated"] == 1
    assert report["gate_a_pass_count"] == 1
    assert report["compile_success_rate"] == pytest.approx(1.0)
    assert report["gate_b_evaluated"] == 0
    assert report["gate_b_pass_count"] == 0
    assert report["gate_b_pass_rate"] is None
    assert report["gate_c_pass_rate"] is None
    assert report["gate_d_pass_rate"] is None


def test_backend_dispatch_audit_parser_success(tmp_path: Path) -> None:
    audit_path = tmp_path / "dispatch.json"
    metallib_path = tmp_path / "default.metallib"
    metallib_path.write_text("stub", encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": "attempt_1",
                "dispatch_rule_id": "rule_patch_abc",
                "device_name": "MTL0",
                "resource_dir": str(tmp_path),
                "metallib_path": str(metallib_path),
                "metallib_source": "candidate",
                "kernels": [{"label": "mul_mat_q4_k_variant_7", "count": 12}],
            }
        ),
        encoding="utf-8",
    )

    audit = benchmark._build_dispatch_audit(
        merged_text="",
        resources_path=tmp_path,
        runtime_env={},
        dispatch_audit_path=audit_path,
        candidate_resources_expected=True,
    )
    assert audit["dispatch_audit_status"] == "ok"
    assert audit["candidate_resources_used"] is True
    assert audit["selected_dispatch_rule_id"] == "rule_patch_abc"
    assert audit["metallib_source"] == "candidate"
    assert audit["top_kernels"][0]["kernel"] == "mul_mat_q4_k_variant_7"


def test_backend_dispatch_audit_missing_and_parse_fail(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    missing = benchmark._build_dispatch_audit(
        merged_text="",
        resources_path=tmp_path,
        runtime_env={},
        dispatch_audit_path=missing_path,
        candidate_resources_expected=True,
    )
    assert missing["dispatch_audit_status"] == "missing"

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not-json", encoding="utf-8")
    bad = benchmark._build_dispatch_audit(
        merged_text="",
        resources_path=tmp_path,
        runtime_env={},
        dispatch_audit_path=bad_path,
        candidate_resources_expected=True,
    )
    assert bad["dispatch_audit_status"] == "parse_fail"

    noaudit = benchmark._build_dispatch_audit(
        merged_text="",
        resources_path=tmp_path,
        runtime_env={},
        dispatch_audit_path=None,
        candidate_resources_expected=True,
    )
    assert noaudit["dispatch_audit_status"] == "backend_noaudit"


def test_candidate_executed_gating_in_result_valid() -> None:
    profile = WorkloadProfile(
        name="chat",
        ctx=8192,
        prompt_tokens_target=1024,
        generate_tokens=128,
        repeats=1,
    )
    result = _fake_profile_result(profile=profile, args=[], capture_raw_output=False)
    result.runs[0]["dispatch_audit_status"] = "missing"
    result.runs[0]["candidate_resources_used"] = False
    ok, reason = study._result_valid(
        result,
        candidate_resources_expected=True,
    )
    assert not ok
    assert reason == "dispatch_audit_missing"

    result2 = _fake_profile_result(profile=profile, args=[], capture_raw_output=False)
    result2.runs[0]["dispatch_audit_status"] = "ok"
    result2.runs[0]["candidate_resources_used"] = False
    ok2, reason2 = study._result_valid(
        result2,
        candidate_resources_expected=True,
    )
    assert not ok2
    assert reason2 == "candidate_resources_not_used"

    result3 = _fake_profile_result(profile=profile, args=[], capture_raw_output=False)
    result3.runs[0]["dispatch_audit_status"] = "ok"
    result3.runs[0]["candidate_resources_used"] = True
    ok3, reason3 = study._result_valid(
        result3,
        candidate_resources_expected=True,
    )
    assert ok3
    assert reason3 == "ok"


def test_throughput_report_includes_audit_quality_counts() -> None:
    report = study._build_throughput_report(
        [
            {
                "valid": False,
                "error": "dispatch_audit_missing",
                "gate_a_evaluated": True,
                "gate_a_pass": True,
                "gate_b_evaluated": False,
                "gate_b_pass": None,
                "gate_c_evaluated": False,
                "gate_c_pass": None,
                "gate_d_evaluated": True,
                "gate_d_pass": False,
                "dispatch_audit_status": "missing",
                "candidate_resources_expected": True,
                "candidate_resources_used": False,
                "compile_record": {"stderr_hash": ""},
                "dispatch_audit": {},
            },
            {
                "valid": True,
                "gate_a_evaluated": True,
                "gate_a_pass": True,
                "gate_b_evaluated": True,
                "gate_b_pass": True,
                "gate_c_evaluated": True,
                "gate_c_pass": True,
                "gate_d_evaluated": True,
                "gate_d_pass": True,
                "dispatch_audit_status": "ok",
                "candidate_resources_expected": True,
                "candidate_resources_used": True,
                "dispatch_audit": {
                    "selected_dispatch_rule_id": "rule_1",
                    "top_kernels": [{"kernel": "mul_mat_fast", "mentions": 3}],
                },
                "compile_record": {"stderr_hash": ""},
            },
            {
                "valid": True,
                "gate_a_evaluated": True,
                "gate_a_pass": True,
                "gate_b_evaluated": False,
                "gate_b_pass": None,
                "gate_c_evaluated": False,
                "gate_c_pass": None,
                "gate_d_evaluated": False,
                "gate_d_pass": None,
                "dispatch_audit_status": "backend_noaudit",
                "candidate_resources_expected": False,
                "candidate_resources_used": None,
                "dispatch_audit": {},
                "compile_record": {"stderr_hash": ""},
            },
        ]
    )
    assert report["dispatch_audit_status_counts"]["missing"] == 1
    assert report["dispatch_audit_status_counts"]["ok"] == 1
    assert report["dispatch_audit_status_counts"]["backend_noaudit"] == 1
    assert report["candidate_resources_expected_count"] == 2
    assert report["candidate_resources_used_count"] == 1
    assert report["candidate_resources_used_rate"] == pytest.approx(0.5)
    assert report["audit_missing_count"] == 1
    assert report["backend_noaudit_count"] == 1


def test_bootstrap_ci_reproducible() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    a = study._bootstrap_ci(values, samples=200, seed=7)
    b = study._bootstrap_ci(values, samples=200, seed=7)
    assert a == b
    assert a["mean"] == pytest.approx(2.5)


def test_holm_correction_monotonic() -> None:
    rows = [
        {"p_value": 0.01},
        {"p_value": 0.03},
        {"p_value": 0.20},
    ]
    study._holm_correct(rows, p_key="p_value", out_key="adj")
    vals = [r["adj"] for r in rows]
    assert all(0.0 <= v <= 1.0 for v in vals)


def _fake_profile_result(
    *,
    profile: WorkloadProfile,
    args: list[str],
    capture_raw_output: bool,
) -> BenchmarkResult:
    base_prefill = 1000.0 if profile.name == "chat" else 700.0
    base_decode = 120.0 if profile.name == "chat" else 90.0

    boost_prefill = 0.0
    boost_decode = 0.0
    if "--flash-attn" in args:
        boost_prefill += 40.0
        boost_decode += 8.0
    if "-ub" in args:
        try:
            val = int(args[args.index("-ub") + 1])
        except Exception:
            val = 0
        if val >= 512:
            boost_prefill += 30.0
            boost_decode += 15.0
        elif val >= 256:
            boost_prefill += 18.0
            boost_decode += 10.0

    prefill = base_prefill + boost_prefill
    decode = base_decode + boost_decode

    runs = []
    for i in range(profile.repeats):
        row = {
            "prefill_tokens_per_sec": prefill,
            "decode_tokens_per_sec": decode,
            "ttft_ms": 50.0,
            "token_latency_ms": 1000.0 / decode,
            "peak_memory_mib": 1024.0,
            "prompt_ms": 40.0,
            "eval_ms": 300.0,
            "return_code": 0,
            "command": "fake",
            "stdout": "[ Prompt: 100.0 t/s | Generation: 50.0 t/s ]" if capture_raw_output else "",
            "stderr": "",
            "prompt": f"p{i}",
        }
        runs.append(row)

    return BenchmarkResult(
        profile=profile,
        metrics=BenchmarkMetrics(
            prefill_tokens_per_sec=prefill,
            decode_tokens_per_sec=decode,
            ttft_ms=50.0,
            p50_token_latency_ms=1000.0 / decode,
            p95_token_latency_ms=1000.0 / decode,
            peak_memory_mib=1024.0,
        ),
        elapsed_seconds=0.01,
        runs=runs,
    )


def test_validate_study_writes_artifacts_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1 = tmp_path / "m1.gguf"
    m2 = tmp_path / "m2.gguf"
    _write_minimal_gguf(m1)
    _write_minimal_gguf(m2)

    from src.apple_silicon.model_probe import probe_model

    p1 = probe_model(m1)
    p2 = probe_model(m2)

    matrix = {
        "models": [
            {"id": "m1", "path": str(m1), "sha256": p1.sha256},
            {"id": "m2", "path": str(m2), "sha256": p2.sha256},
        ]
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    monkeypatch.setattr(study.benchmark, "resolve_llama_cli", lambda _root: Path("/tmp/llama-cli"))
    monkeypatch.setattr(study, "ensure_llamacpp_commit", lambda _root, strict=True: (True, "ok"))
    monkeypatch.setattr(study, "get_llamacpp_commit", lambda _root: "deadbeef")
    monkeypatch.setattr(study, "_git_commit", lambda _root: "cafebabe")
    monkeypatch.setattr(
        study,
        "probe_device",
        lambda: DeviceProfile(
            platform="darwin",
            arch="arm64",
            macos_version="14.0",
            is_apple_silicon=True,
            chip="Apple M2",
            gpu_cores=10,
            cpu_cores=8,
            memory_gb=16.0,
            metal_supported=True,
            metal_feature_set="Supported",
            fingerprint="fp123",
        ),
    )
    monkeypatch.setattr(study, "_power_state", lambda: {"on_ac_power": True, "charging": True, "percent": 95})
    monkeypatch.setattr(study, "_resolve_llm_identity", lambda required: {"provider": "openai", "model": "gpt-5.2-codex"})

    call_count = {"n": 0}

    def fake_propose(*, device, model, profile_mode, gate_mode, baseline, previous_attempts, llm_identity):
        call_count["n"] += 1
        # First iterative attempt fails once to validate resilient logging.
        if call_count["n"] == 2:
            raise RuntimeError("bad payload")

        class _Candidate:
            candidate_name = f"cand-{call_count['n']}"
            rationale = "test"
            kernel_overrides = {"quant_matvec_decode": {"threadgroup": 128}}
            runtime_args = ["--flash-attn", "on", "-ub", "512"]

        return _Candidate(), {"provider": "openai", "model": "gpt-5.2-codex"}

    monkeypatch.setattr(study, "_propose_candidate_with_fallback", fake_propose)

    monkeypatch.setattr(
        study.benchmark,
        "run_profile_benchmark",
        lambda **kwargs: _fake_profile_result(
            profile=kwargs["profile"],
            args=kwargs.get("extra_args") or [],
            capture_raw_output=kwargs.get("capture_raw_output", False),
        ),
    )

    out_dir = tmp_path / "study_out"
    cache_root = tmp_path / "custom_cache"
    summary = study.run_validation_study(
        matrix_path=matrix_path,
        output_dir=out_dir,
        profiles=["chat"],
        arms=["baseline", "flash", "oneshot", "iterative"],
        llamacpp_root=tmp_path / "llama.cpp",
        gate_mode="quick",
        cooldown_seconds=0.0,
        bootstrap_samples=200,
        seed=42,
        require_ac_power=True,
        strict_commit=False,
        resume=False,
        cache_root=cache_root,
    )

    assert summary.success
    assert (out_dir / "study_manifest.json").exists()
    assert (out_dir / "runs_raw.jsonl").exists()
    assert (out_dir / "attempts.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "throughput_report.json").exists()
    assert (out_dir / "throughput_report.csv").exists()
    assert (out_dir / "claim_decisions.json").exists()
    assert (out_dir / "schedule.json").exists()
    assert (out_dir / "hotspots.json").exists()
    assert (out_dir / "hotspots_op_perf.json").exists()
    assert (out_dir / "op_profiles.json").exists()
    assert (out_dir / "roofline_analysis.json").exists()
    assert (out_dir / "exclusions.csv").exists()
    assert (out_dir / "methods_note.md").exists()
    assert (out_dir / "metrics_by_block.csv").exists()
    assert (out_dir / "paired_deltas.csv").exists()
    assert (out_dir / "ci_results.csv").exists()
    assert (out_dir / "pvalues_corrected.csv").exists()
    assert (out_dir / "plots" / "decode_effects.svg").exists()
    assert (out_dir / "plots" / "arm_ci_decode.svg").exists()
    assert (out_dir / "plots" / "iterative_convergence.svg").exists()
    assert (out_dir / "plots" / "regression_heatmap.svg").exists()

    manifest = json.loads((out_dir / "study_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("cache_root") == str(cache_root.resolve())

    attempt_lines = (out_dir / "attempts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(attempt_lines) >= 1

    summary2 = study.run_validation_study(
        matrix_path=matrix_path,
        output_dir=out_dir,
        profiles=["chat"],
        arms=["baseline", "flash"],
        llamacpp_root=tmp_path / "llama.cpp",
        gate_mode="quick",
        cooldown_seconds=0.0,
        bootstrap_samples=50,
        seed=1,
        require_ac_power=True,
        strict_commit=False,
        resume=True,
    )
    assert summary2.success
    assert summary2.output_dir != str(out_dir)


def test_validate_study_fails_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1 = tmp_path / "m1.gguf"
    _write_minimal_gguf(m1)
    matrix = {"models": [{"id": "m1", "path": str(m1), "sha256": "deadbeef"}]}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    monkeypatch.setattr(study.benchmark, "resolve_llama_cli", lambda _root: Path("/tmp/llama-cli"))
    monkeypatch.setattr(study, "ensure_llamacpp_commit", lambda _root, strict=True: (True, "ok"))
    monkeypatch.setattr(study, "get_llamacpp_commit", lambda _root: "deadbeef")
    monkeypatch.setattr(study, "_git_commit", lambda _root: "cafebabe")
    monkeypatch.setattr(
        study,
        "probe_device",
        lambda: DeviceProfile(
            platform="darwin",
            arch="arm64",
            macos_version="14.0",
            is_apple_silicon=True,
            chip="Apple M2",
            gpu_cores=10,
            cpu_cores=8,
            memory_gb=16.0,
            metal_supported=True,
            metal_feature_set="Supported",
            fingerprint="fp123",
        ),
    )
    monkeypatch.setattr(study, "_power_state", lambda: {"on_ac_power": True, "charging": True, "percent": 95})

    with pytest.raises(study.StudyError):
        study.run_validation_study(
            matrix_path=matrix_path,
            output_dir=tmp_path / "out",
            profiles=["chat"],
            arms=["baseline", "flash"],
            llamacpp_root=tmp_path / "llama.cpp",
            gate_mode="quick",
            cooldown_seconds=0.0,
            bootstrap_samples=50,
            seed=1,
            require_ac_power=True,
            strict_commit=False,
            resume=False,
        )


def test_validate_study_official_mode_requires_test_backend_ops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1 = tmp_path / "m1.gguf"
    _write_minimal_gguf(m1)
    from src.apple_silicon.model_probe import probe_model

    p1 = probe_model(m1)
    matrix = {"models": [{"id": "m1", "path": str(m1), "sha256": p1.sha256}]}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    monkeypatch.setattr(study.benchmark, "resolve_llama_cli", lambda _root: Path("/tmp/llama-cli"))
    monkeypatch.setattr(study, "resolve_test_backend_ops", lambda _root: None)
    monkeypatch.setattr(study, "ensure_llamacpp_commit", lambda _root, strict=True: (True, "ok"))
    monkeypatch.setattr(study, "get_llamacpp_commit", lambda _root: "deadbeef")
    monkeypatch.setattr(study, "_git_commit", lambda _root: "cafebabe")
    monkeypatch.setattr(
        study,
        "probe_device",
        lambda: DeviceProfile(
            platform="darwin",
            arch="arm64",
            macos_version="14.0",
            is_apple_silicon=True,
            chip="Apple M2",
            gpu_cores=10,
            cpu_cores=8,
            memory_gb=16.0,
            metal_supported=True,
            metal_feature_set="Supported",
            fingerprint="fp123",
        ),
    )
    monkeypatch.setattr(study, "_power_state", lambda: {"on_ac_power": True, "charging": True, "percent": 95})

    with pytest.raises(study.StudyError, match="test-backend-ops"):
        study.run_validation_study(
            matrix_path=matrix_path,
            output_dir=tmp_path / "out_missing_ops",
            profiles=["chat"],
            arms=["baseline", "flash"],
            llamacpp_root=tmp_path / "llama.cpp",
            gate_mode="quick",
            cooldown_seconds=0.0,
            bootstrap_samples=20,
            seed=1,
            require_ac_power=True,
            strict_commit=False,
            resume=False,
            profiling_mode="op_perf_required",
        )


def test_iterative_ladder_quick_then_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1 = tmp_path / "m1.gguf"
    _write_minimal_gguf(m1)
    from src.apple_silicon.model_probe import probe_model

    p1 = probe_model(m1)
    matrix = {"models": [{"id": "m1", "path": str(m1), "sha256": p1.sha256}]}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    monkeypatch.setattr(study.benchmark, "resolve_llama_cli", lambda _root: Path("/tmp/llama-cli"))
    monkeypatch.setattr(study, "ensure_llamacpp_commit", lambda _root, strict=True: (True, "ok"))
    monkeypatch.setattr(study, "get_llamacpp_commit", lambda _root: "deadbeef")
    monkeypatch.setattr(study, "_git_commit", lambda _root: "cafebabe")
    monkeypatch.setattr(
        study,
        "probe_device",
        lambda: DeviceProfile(
            platform="darwin",
            arch="arm64",
            macos_version="14.0",
            is_apple_silicon=True,
            chip="Apple M2",
            gpu_cores=10,
            cpu_cores=8,
            memory_gb=16.0,
            metal_supported=True,
            metal_feature_set="Supported",
            fingerprint="fp123",
        ),
    )
    monkeypatch.setattr(study, "_power_state", lambda: {"on_ac_power": True, "charging": True, "percent": 95})
    monkeypatch.setattr(study, "_resolve_llm_identity", lambda required: {"provider": "openai", "model": "gpt-5.2-codex"})
    monkeypatch.setattr(
        study.benchmark,
        "run_profile_benchmark",
        lambda **kwargs: _fake_profile_result(
            profile=kwargs["profile"],
            args=kwargs.get("extra_args") or [],
            capture_raw_output=kwargs.get("capture_raw_output", False),
        ),
    )

    seen_gate_modes: list[str] = []

    def fake_propose(*, device, model, profile_mode, gate_mode, baseline, previous_attempts, llm_identity):
        seen_gate_modes.append(gate_mode)

        class _Candidate:
            candidate_name = "cand"
            rationale = "ok"
            kernel_overrides = {}
            runtime_args = ["--flash-attn", "on"]

        return _Candidate(), {"provider": "openai", "model": "gpt-5.2-codex"}

    monkeypatch.setattr(study, "_propose_candidate_with_fallback", fake_propose)

    summary = study.run_validation_study(
        matrix_path=matrix_path,
        output_dir=tmp_path / "out",
        profiles=["chat"],
        arms=["baseline", "iterative"],
        llamacpp_root=tmp_path / "llama.cpp",
        gate_mode="full",
        cooldown_seconds=0.0,
        bootstrap_samples=20,
        seed=1,
        require_ac_power=True,
        strict_commit=False,
        resume=False,
    )
    assert summary.success
    assert seen_gate_modes
    assert seen_gate_modes[0] == "quick"
    assert all(mode == "full" for mode in seen_gate_modes[1:])


def test_invalid_block_logged_on_failed_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1 = tmp_path / "m1.gguf"
    _write_minimal_gguf(m1)
    from src.apple_silicon.model_probe import probe_model

    p1 = probe_model(m1)
    matrix = {"models": [{"id": "m1", "path": str(m1), "sha256": p1.sha256}]}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    monkeypatch.setattr(study.benchmark, "resolve_llama_cli", lambda _root: Path("/tmp/llama-cli"))
    monkeypatch.setattr(study, "ensure_llamacpp_commit", lambda _root, strict=True: (True, "ok"))
    monkeypatch.setattr(study, "get_llamacpp_commit", lambda _root: "deadbeef")
    monkeypatch.setattr(study, "_git_commit", lambda _root: "cafebabe")
    monkeypatch.setattr(
        study,
        "probe_device",
        lambda: DeviceProfile(
            platform="darwin",
            arch="arm64",
            macos_version="14.0",
            is_apple_silicon=True,
            chip="Apple M2",
            gpu_cores=10,
            cpu_cores=8,
            memory_gb=16.0,
            metal_supported=True,
            metal_feature_set="Supported",
            fingerprint="fp123",
        ),
    )
    monkeypatch.setattr(study, "_power_state", lambda: {"on_ac_power": True, "charging": True, "percent": 95})
    monkeypatch.setattr(study, "_resolve_llm_identity", lambda required: {"provider": "openai", "model": "gpt-5.2-codex"})

    def fake_run_profile(**kwargs):
        profile = kwargs["profile"]
        args = kwargs.get("extra_args") or []
        res = _fake_profile_result(profile=profile, args=args, capture_raw_output=kwargs.get("capture_raw_output", False))
        if "--flash-attn" in args:
            res.runs[0]["return_code"] = 1
            res.runs[0]["prefill_tokens_per_sec"] = None
            res.runs[0]["decode_tokens_per_sec"] = None
            res.metrics.prefill_tokens_per_sec = None
            res.metrics.decode_tokens_per_sec = None
        return res

    monkeypatch.setattr(study.benchmark, "run_profile_benchmark", fake_run_profile)

    summary = study.run_validation_study(
        matrix_path=matrix_path,
        output_dir=tmp_path / "out_fail",
        profiles=["chat"],
        arms=["baseline", "flash"],
        llamacpp_root=tmp_path / "llama.cpp",
        gate_mode="quick",
        cooldown_seconds=0.0,
        bootstrap_samples=20,
        seed=1,
        require_ac_power=True,
        strict_commit=False,
        resume=False,
    )
    assert summary.success
    assert summary.invalid_blocks > 0
    assert summary.invalid_runs > 0
