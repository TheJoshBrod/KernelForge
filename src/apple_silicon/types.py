from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DeviceProfile:
    platform: str
    arch: str
    macos_version: str
    is_apple_silicon: bool
    chip: str
    gpu_cores: int | None
    cpu_cores: int | None
    memory_gb: float | None
    metal_supported: bool
    metal_feature_set: str
    fingerprint: str
    metal_thread_execution_width: int | None = None
    metal_max_threads_per_threadgroup: int | None = None
    metal_max_threadgroup_memory_bytes: int | None = None


@dataclass(slots=True)
class AllowedParams:
    thread_execution_width: int
    allowed_simd_widths: list[int] = field(default_factory=list)
    max_threads_per_threadgroup: int = 1024
    max_threadgroup_memory_bytes: int = 32 * 1024
    source: str = "default"


@dataclass(slots=True)
class ModelProfile:
    path: Path
    name: str
    architecture: str
    quant: str
    file_type_id: int | None
    size_bytes: int
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkloadProfile:
    name: str
    ctx: int
    prompt_tokens_target: int
    generate_tokens: int
    repeats: int


@dataclass(slots=True)
class BenchmarkMetrics:
    prefill_tokens_per_sec: float | None
    decode_tokens_per_sec: float | None
    ttft_ms: float | None
    p50_token_latency_ms: float | None
    p95_token_latency_ms: float | None
    peak_memory_mib: float | None


@dataclass(slots=True)
class BenchmarkResult:
    profile: WorkloadProfile
    metrics: BenchmarkMetrics
    elapsed_seconds: float
    runs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TuningCandidate:
    candidate_name: str
    rationale: str
    kernel_overrides: dict[str, Any] = field(default_factory=dict)
    runtime_args: list[str] = field(default_factory=list)
    template_mutations: dict[str, int] = field(default_factory=dict)
    source_patches: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


@dataclass(slots=True)
class KernelPatchCandidate:
    candidate_id: str
    template_version: str
    resources_dir: str
    patch_hash: str
    source_hash: str
    template_mutations: dict[str, int] = field(default_factory=dict)
    source_patches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class KernelVariantSpec:
    variant_id: str
    kernel_family: str
    template_mutations: dict[str, int] = field(default_factory=dict)
    source_patches: list[dict[str, Any]] = field(default_factory=list)
    runtime_args: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class DispatchRule:
    rule_id: str
    model_arch: str
    quant_family: str
    profile: str
    shape_bucket: str
    variant_id: str
    priority: int = 0


@dataclass(slots=True)
class KernelCompileRecord:
    attempted: bool
    success: bool
    classification: str
    stderr_hash: str = ""
    error: str = ""
    compile_warmup_done: bool = False
    pipeline_cache_key: str = ""
    compile_time_ms: float | None = None
    toolchain_fingerprint: str = ""


@dataclass(slots=True)
class KernelCorrectnessRecord:
    attempted: bool
    success: bool
    classification: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CorrectnessSuiteResult:
    strict_parity_enabled: bool
    success: bool
    classification: str
    min_similarity: float | None = None
    max_similarity: float | None = None
    checked_runs: int = 0
    failed_runs: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParityGateRecord:
    stage: str
    attempted: bool
    success: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeasibilityCheckRecord:
    attempted: bool
    success: bool
    classification: str
    reasons: list[str] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolchainFingerprint:
    fingerprint: str
    xcrun_version: str = ""
    xcode_version: str = ""
    macos_version: str = ""


@dataclass(slots=True)
class OpPerfRecord:
    op: str
    backend: str
    op_params: str = ""
    time_ms: float | None = None
    flops: float | None = None
    bandwidth_gb_s: float | None = None


@dataclass(slots=True)
class OpPerfTableRecord:
    table_name: str
    columns: list[str] = field(default_factory=list)
    rows: list[OpPerfRecord] = field(default_factory=list)
    rank_metric: str = "time"
    backend: str = ""
    source_sqlite_path: str = ""


@dataclass(slots=True)
class OpPerfStatusRecord:
    status: str
    elapsed_ms: float | None = None
    rows_emitted: int = 0
    backend_requested: str = ""
    backend_resolved: str = ""
    cache_hit: bool = False
    command: str = ""
    timeout_sec: float | None = None
    reason: str = ""
    sqlite_path: str = ""
    sql_path: str = ""
    support_csv_path: str = ""
    profiling_mode_effective: str = ""


@dataclass(slots=True)
class DebugCandidateRecord:
    debug_run_id: str
    command: str
    mode: str
    op: str
    resources_dir: str = ""
    validation_env: dict[str, str] = field(default_factory=dict)
    return_code: int = 0
    stderr_hash: str = ""
    stdout_hash: str = ""
    wall_clock_utc: str = ""


@dataclass(slots=True)
class OptimizeReport:
    success: bool
    reason: str
    model_path: str
    profile_mode: str
    gate_mode: str
    pack_id: str
    pack_dir: str
    baseline: dict[str, Any]
    optimized: dict[str, Any]
    delta: dict[str, Any]
    pass_gate: bool
    tuning: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudyRunRecord:
    run_id: str
    block_id: str
    is_warmup: bool
    model_id: str
    model_path: str
    model_sha256: str
    profile: str
    arm_id: str
    attempt_id: str
    order_index: int
    wall_clock_utc: str
    git_commit: str
    llamacpp_commit: str
    kernel_template_version: str = ""
    patch_hash: str = ""
    candidate_source_hash: str = ""
    compile_result: str = ""
    compile_stderr_hash: str = ""
    compile_warmup_done: bool = False
    pipeline_cache_key: str = ""
    compile_time_ms: float | None = None
    prompt_tokens_target: int | None = None
    prompt_tokens_actual: int | None = None
    prompt_tokens_target_met: bool | None = None
    runtime_args: list[str] = field(default_factory=list)
    power_state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_run: dict[str, Any] = field(default_factory=dict)
    profiling_mode: str = ""
    op_perf_status: str = ""
    op_perf_cache_hit: bool | None = None
    op_perf_rows_emitted: int | None = None
    op_perf_cache_key: str = ""
    prompt_cache_mode: str = ""
    prompt_cache_file: str = ""
    prompt_cache_build_elapsed_ms: float | None = None
    prompt_cache_isolated: bool | None = None
    dispatch_audit_status: str = ""
    candidate_resources_expected: bool | None = None
    candidate_resources_used: bool | None = None
    dispatch_audit_path: str = ""
    dispatch_audit_source: str = ""
    metallib_source: str = ""
    dispatch_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudyAttemptRecord:
    model_id: str
    model_sha256: str
    profile: str
    arm_id: str
    attempt_id: str
    wall_clock_utc: str
    git_commit: str
    llamacpp_commit: str
    order_index: int
    provider: str
    model: str
    candidate_name: str
    rationale: str
    stage: str
    constraint_repairs: list[str] = field(default_factory=list)
    allowed_params: dict[str, Any] = field(default_factory=dict)
    kernel_template_version: str = ""
    patch_hash: str = ""
    candidate_source_hash: str = ""
    compile_warmup_done: bool = False
    pipeline_cache_key: str = ""
    compile_time_ms: float | None = None
    compile_record: dict[str, Any] = field(default_factory=dict)
    correctness_record: dict[str, Any] = field(default_factory=dict)
    feasibility_record: dict[str, Any] = field(default_factory=dict)
    runtime_args: list[str] = field(default_factory=list)
    kernel_overrides: dict[str, Any] = field(default_factory=dict)
    power_state: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    error: str = ""
    score: float | None = None
    delta: dict[str, Any] = field(default_factory=dict)
    op_perf_status: str = ""
    op_perf_cache_hit: bool | None = None
    op_perf_rows_emitted: int | None = None
    op_perf_cache_key: str = ""
    op_perf_delta_pct: float | None = None
    op_perf_common_rows: int = 0
    op_perf_baseline_total_ms: float | None = None
    op_perf_candidate_total_ms: float | None = None
    op_perf_compare_key: str = ""
    op_perf_decision: str = ""
    op_perf_promoted: bool = False
    gate_a_evaluated: bool | None = None
    gate_b_evaluated: bool | None = None
    gate_c_evaluated: bool | None = None
    gate_d_evaluated: bool | None = None
    gate_a_pass: bool | None = None
    gate_b_pass: bool | None = None
    gate_c_pass: bool | None = None
    gate_d_pass: bool | None = None
    dispatch_audit_status: str = ""
    candidate_resources_expected: bool | None = None
    candidate_resources_used: bool | None = None
    dispatch_audit_path: str = ""
    dispatch_audit_source: str = ""
    metallib_source: str = ""
    dispatch_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HotspotAttributionRecord:
    run_id: str
    model_id: str
    profile: str
    arm_id: str
    order_index: int
    decode_tokens_per_sec: float | None
    prefill_tokens_per_sec: float | None
    runtime_args: list[str] = field(default_factory=list)
    hotspot_ops: list[str] = field(default_factory=list)
    source: str = "heuristic"
    profiling_mode: str = "heuristic"
    wall_clock_utc: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudySummary:
    success: bool
    output_dir: str
    generated_at_utc: str
    models_tested: int
    profiles_tested: list[str] = field(default_factory=list)
    arms_tested: list[str] = field(default_factory=list)
    total_blocks: int = 0
    total_runs: int = 0
    invalid_blocks: int = 0
    invalid_runs: int = 0
    block_failure_rate: float | None = None
    run_failure_rate: float | None = None
    claim_matrix: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScheduleRecord:
    model_id: str
    profile: str
    arm_id: str
    stage: str
    block_id: str
    cycle_index: int
    order_index: int
    first_arm: str
    second_arm: str
    order_label: str
    is_warmup: bool = False


@dataclass(slots=True)
class PromptTargetRecord:
    model_id: str
    profile: str
    block_id: str
    run_id: str
    prompt_tokens_target: int | None
    prompt_tokens_actual: int | None
    tolerance: int
    met: bool
    reason: str = ""


@dataclass(slots=True)
class StudyClaimRecord:
    model_id: str
    profile: str
    arm_id: str
    claim_faster: bool
    primary_positive: bool
    guardrail_pass: bool
    decode_ci95_low: float | None = None
    decode_ci95_high: float | None = None


@dataclass(slots=True)
class ClaimDecision:
    scope: str
    model_id: str
    arm_id: str
    profile: str = ""
    metric: str = "decode"
    threshold_pct: float = 30.0
    n_blocks_total: int = 0
    n_blocks_valid: int = 0
    mean_delta_pct: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    p_value: float | None = None
    p_value_holm: float | None = None
    guardrail_pass: bool = False
    claim_faster: bool = False
    profiling_mode: str = ""
    toolchain_fingerprint: str = ""
    schedule_proof_id: str = ""
    reason: str = ""
