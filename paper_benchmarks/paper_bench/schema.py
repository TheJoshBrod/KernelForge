from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import HARNESS_VERSION

SCHEMA_VERSION = "1.0"


class Variant(str, Enum):
    eager = "eager"
    torch_compile = "torch_compile"
    kf_cast = "kf_cast"


class Stage(str, Enum):
    operator = "operator"
    prefill = "prefill"
    decode = "decode"
    total_generate = "total_generate"
    load = "load"
    compile = "compile"
    warmup = "warmup"


class BenchmarkMode(str, Enum):
    operator = "operator"
    e2e_model = "e2e_model"
    deployment = "deployment"


class CorrectnessStatus(str, Enum):
    reference = "reference"
    passed = "passed"
    failed = "failed"
    skipped = "skipped"
    not_applicable = "not_applicable"


PERFORMANCE_STAGES = frozenset({
    Stage.operator,
    Stage.prefill,
    Stage.decode,
    Stage.total_generate,
})
MODEL_LEVEL_STAGES = frozenset({
    Stage.prefill,
    Stage.decode,
    Stage.total_generate,
})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _coerce_enum(value: Any, enum_type):
    if value is None or isinstance(value, enum_type):
        return value
    return enum_type(value)


def _coerce_enum_list(value: Any, enum_type):
    if value is None:
        return value
    if not isinstance(value, list):
        raise TypeError(f"Expected list for {enum_type.__name__} values")
    return [_coerce_enum(item, enum_type) for item in value]


class LatencySummary(StrictModel):
    count: int = Field(ge=0)
    mean_ms: float | None = None
    median_ms: float | None = None
    p05_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    stddev_ms: float | None = None


class ArtifactBase(StrictModel):
    artifact_type: str = Field(min_length=1)
    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1)
    benchmark_harness_version: str = Field(default=HARNESS_VERSION, min_length=1)
    timestamp_utc: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    benchmark_mode: BenchmarkMode | None = None
    variant: Variant | None = None
    stage: Stage | None = None
    git_available: bool = True
    git_commit: str = Field(min_length=1)
    git_dirty: bool
    git_branch: str | None = None
    git_dirty_summary: list[str] = Field(default_factory=list)
    git_untracked_summary: list[str] = Field(default_factory=list)
    command_line: list[str] = Field(min_length=1)
    command_line_text: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    os_name: str = Field(min_length=1)
    os_release: str | None = None
    os_version: str | None = None
    python_version: str = Field(min_length=1)
    pytorch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_names: list[str] = Field(default_factory=list)
    gpu_count: int = Field(ge=0)
    gpu_properties: list[dict[str, Any]] = Field(default_factory=list)
    driver_info: dict[str, Any] = Field(default_factory=dict)
    relevant_env: dict[str, str] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    determinism_controls: dict[str, Any] = Field(default_factory=dict)
    model_id: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    model_path_hash: str | None = None
    model_config_path: str | None = None
    model_config_hash: str | None = None
    quantization: str | None = None
    quantization_config_hash: str | None = None
    placement_profile: str | None = None
    model_license: str | None = None
    model_access_terms: str | None = None
    suite_id: str = Field(min_length=1)
    suite_path: str = Field(min_length=1)
    suite_hash: str = Field(min_length=1)
    workload_path: str = Field(min_length=1)
    workload_hash: str | None = None
    workload_slug: str | None = None
    dataset_license: str | None = None
    dataset_access_terms: str | None = None
    cache_mode: str | None = None
    config_hashes: dict[str, str] = Field(default_factory=dict)
    cast_package_path: str | None = None
    cast_package_hash: str | None = None
    kf_artifact_path: str | None = None
    kf_artifact_hash: str | None = None
    kf_artifact_kind: str | None = None
    exported_kernel_hashes: dict[str, str] = Field(default_factory=dict)
    compile_settings: dict[str, Any] = Field(default_factory=dict)
    kf_settings: dict[str, Any] = Field(default_factory=dict)
    toolchain_status: dict[str, Any] = Field(default_factory=dict)
    cache_reuse_status: str | None = None
    paper_claim_status: str | None = None
    claim_category: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    comparison_group: str | None = None
    configured_batch_size: int | None = Field(default=None, ge=1)
    prompt_bucket_id: str | None = None
    reused: bool = False
    reused_from_artifact: str | None = None
    reused_from_artifact_hash: str | None = None
    paper_eligible: bool
    paper_eligibility_issues: list[str] = Field(default_factory=list)
    synthetic_workload: bool
    warmup_count: int = Field(ge=0)
    timed_run_count: int = Field(ge=0)
    latency_samples_ms: list[float] = Field(default_factory=list)
    correctness_status: CorrectnessStatus = CorrectnessStatus.not_applicable
    correctness_message: str | None = None
    fallback_count: int | None = Field(default=None, ge=0)
    kernel_hit_count: int | None = Field(default=None, ge=0)
    compile_time_ms: float | None = Field(default=None, ge=0.0)
    steady_state_time_ms: float | None = Field(default=None, ge=0.0)
    notes: list[str] = Field(default_factory=list)

    @field_validator("benchmark_mode", mode="before")
    @classmethod
    def _parse_benchmark_mode(cls, value: Any):
        return _coerce_enum(value, BenchmarkMode)

    @field_validator("variant", mode="before")
    @classmethod
    def _parse_variant(cls, value: Any):
        return _coerce_enum(value, Variant)

    @field_validator("stage", mode="before")
    @classmethod
    def _parse_stage(cls, value: Any):
        return _coerce_enum(value, Stage)

    @field_validator("correctness_status", mode="before")
    @classmethod
    def _parse_correctness_status(cls, value: Any):
        return _coerce_enum(value, CorrectnessStatus)

    @model_validator(mode="after")
    def _validate_hash_requirements(self) -> ArtifactBase:
        if self.model_config_path and not self.model_config_hash:
            raise ValueError("model_config_hash is required when model_config_path is set")
        if self.cast_package_path and not self.cast_package_hash:
            raise ValueError("cast_package_hash is required when cast_package_path is set")
        if self.kf_artifact_path and not self.kf_artifact_hash:
            raise ValueError("kf_artifact_hash is required when kf_artifact_path is set")
        if not self.model_path_hash:
            raise ValueError("model_path_hash is required")
        issues = list(self.paper_eligibility_issues)
        if not self.git_available:
            issues.append("git metadata unavailable")
        if not self.git_commit or self.git_commit == "unknown":
            issues.append("git commit unknown")
        if not self.workload_hash:
            issues.append("workload hash missing")
        if not self.model_id:
            issues.append("model_id missing")
        if not self.model_path:
            issues.append("model path missing")
        if not self.model_path_hash:
            issues.append("model path hash missing")
        if self.synthetic_workload:
            issues.append("synthetic workload used")
        if self.model_license is None and self.paper_claim_status == "paper_eligible":
            issues.append("model license/access terms missing")
        if self.dataset_license is None and self.paper_claim_status == "paper_eligible":
            issues.append("dataset license/access terms missing")
        if not self.command_line:
            issues.append("command line missing")
        if not self.command_line_text:
            issues.append("command line text missing")
        if not self.package_versions.get("torch"):
            issues.append("torch package version missing")
        if self.artifact_type == "benchmark_result" and self.correctness_status == CorrectnessStatus.failed:
            issues.append("benchmark artifact failed correctness or execution")
        for validation_error in self.validation_errors:
            issues.append(f"run validation error: {validation_error}")
        if self.stage in PERFORMANCE_STAGES:
            if self.correctness_status not in {CorrectnessStatus.reference, CorrectnessStatus.passed}:
                issues.append("correctness did not pass")
            if self.timed_run_count <= 0 or len(self.latency_samples_ms) <= 0:
                issues.append("timed latency samples missing")
            if self.timed_run_count != len(self.latency_samples_ms):
                issues.append("timed latency sample count mismatch")
            if self.steady_state_time_ms is None:
                issues.append("steady_state_time_ms missing")
            if self.variant == Variant.kf_cast:
                if self.fallback_count is None:
                    issues.append("fallback count missing")
                if self.kernel_hit_count is None:
                    issues.append("kernel hit count missing")
                if not (
                    (self.cast_package_path and self.cast_package_hash)
                    or (self.kf_artifact_path and self.kf_artifact_hash)
                ):
                    issues.append("Kernel Forge artifact path/hash missing")
                if self.kf_settings.get("fail_on_fallback", True) and (self.fallback_count or 0) > 0:
                    issues.append("fallback observed while fail_on_fallback is enabled")
        deduped = list(dict.fromkeys(issues))
        self.paper_eligibility_issues = deduped
        self.paper_eligible = not deduped
        return self


class RunManifestArtifact(ArtifactBase):
    artifact_type: Literal["run_manifest"] = "run_manifest"
    run_dir: str = Field(min_length=1)
    variants_requested: list[Variant] = Field(default_factory=list)
    stages_requested: list[Stage] = Field(default_factory=list)
    description: str | None = None

    @field_validator("variants_requested", mode="before")
    @classmethod
    def _parse_variants_requested(cls, value: Any):
        return _coerce_enum_list(value, Variant)

    @field_validator("stages_requested", mode="before")
    @classmethod
    def _parse_stages_requested(cls, value: Any):
        return _coerce_enum_list(value, Stage)


class EnvironmentArtifact(ArtifactBase):
    artifact_type: Literal["environment_snapshot"] = "environment_snapshot"
    platform: str = Field(min_length=1)
    platform_release: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str | None = None
    torch_cuda_available: bool
    torch_mps_available: bool
    torch_device_capability: str | None = None
    nvcc_version: str | None = None
    nvidia_smi_output: str | None = None


class DeviceAuditArtifact(ArtifactBase):
    artifact_type: Literal["device_audit"] = "device_audit"
    audit_stage: str = Field(min_length=1)
    audit_status: str = Field(min_length=1)
    audit_errors: list[str] = Field(default_factory=list)
    audit_warnings: list[str] = Field(default_factory=list)
    selected_ops: list[str] = Field(default_factory=list)
    runtime_input_device: str | None = None
    tokenizer_output_devices: dict[str, str] = Field(default_factory=dict)
    placement_audit: dict[str, Any] = Field(default_factory=dict)
    per_op_launch_coverage: dict[str, Any] = Field(default_factory=dict)
    fallback_reasons_by_op: dict[str, dict[str, int]] = Field(default_factory=dict)
    kernel_launches_attempted: int | None = Field(default=None, ge=0)
    kernel_launches_succeeded: int | None = Field(default=None, ge=0)
    kernel_launches_failed: int | None = Field(default=None, ge=0)
    fallback_count: int | None = Field(default=None, ge=0)


class BenchmarkArtifact(ArtifactBase):
    artifact_type: Literal["benchmark_result"] = "benchmark_result"
    variant: Variant
    stage: Stage
    latency_samples_ms: list[float] = Field(min_length=1)
    latency_summary: LatencySummary
    sample_records: list[dict[str, Any]] = Field(default_factory=list)
    prompt_id: str | None = None
    prompt_hash: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_benchmark_specifics(self) -> BenchmarkArtifact:
        if self.stage in {Stage.operator, Stage.prefill, Stage.decode, Stage.total_generate, Stage.load, Stage.warmup}:
            if self.steady_state_time_ms is None:
                raise ValueError("steady_state_time_ms is required for benchmark stages")
        if self.stage == Stage.compile and self.compile_time_ms is None:
            raise ValueError("compile_time_ms is required for compile stage")
        if self.variant == Variant.kf_cast:
            if not (
                (self.cast_package_path and self.cast_package_hash)
                or (self.kf_artifact_path and self.kf_artifact_hash)
            ):
                raise ValueError("kf_cast artifacts require a cast package hash or Kernel Forge artifact hash")
        issues = list(self.paper_eligibility_issues)
        if self.stage in MODEL_LEVEL_STAGES:
            selection_policy = self.details.get("selection_policy")
            if isinstance(selection_policy, dict):
                method = str(selection_policy.get("method", "")).strip().lower()
                if bool(selection_policy.get("post_hoc")) or method in {
                    "longest_prompt_only",
                    "post_hoc_longest_prompt_only",
                    "longest_prompt_only_post_hoc",
                }:
                    issues.append("post-hoc longest-prompt-only selection is not paper eligible")
            missing_output_hashes = [
                index
                for index, record in enumerate(self.sample_records)
                if not isinstance(record.get("output_token_hashes"), list) or not record.get("output_token_hashes")
            ]
            if missing_output_hashes:
                issues.append("per-run output token hashes missing")
            if self.variant in {Variant.torch_compile, Variant.kf_cast}:
                if self.details.get("per_run_output_hash_verification") is not True:
                    issues.append("per-run output hash verification missing")
                checked_count = self.details.get("correctness_checked_run_count")
                if checked_count is None or int(checked_count) != self.timed_run_count:
                    issues.append("per-run correctness check count mismatch")
        deduped = list(dict.fromkeys(issues))
        self.paper_eligibility_issues = deduped
        self.paper_eligible = not deduped
        return self


class SummaryRow(StrictModel):
    variant: Variant
    stage: Stage
    comparison_group: str | None = None
    configured_batch_size: int | None = Field(default=None, ge=1)
    prompt_bucket_id: str | None = None
    artifact_path: str | None = None
    reused: bool = False
    reused_from_artifact: str | None = None
    reused_from_artifact_hash: str | None = None
    correctness_status: CorrectnessStatus
    sample_count: int = Field(ge=0)
    p05_ms: float | None = None
    median_ms: float | None = None
    mean_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    stddev_ms: float | None = None
    steady_state_time_ms: float | None = None
    compile_time_ms: float | None = None
    runtime_load_time_ms: float | None = None
    setup_time_ms: float | None = None
    jit_compile_time_ms: float | None = None
    precompiled_load_time_ms: float | None = None
    total_tokens_per_second: float | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    prompt_count: int | None = Field(default=None, ge=0)
    prompt_suite_hash: str | None = None
    generated_token_equality_status: str | None = None
    speedup_vs_eager: float | None = None
    speedup_vs_torch_compile: float | None = None
    paper_eligible: bool
    claim_eligible: bool
    fallback_count: int | None = Field(default=None, ge=0)
    kernel_hit_count: int | None = Field(default=None, ge=0)
    exception_fallback_count: int | None = Field(default=None, ge=0)
    contiguous_copy_count: int | None = Field(default=None, ge=0)
    adaptation_count: int | None = Field(default=None, ge=0)
    cast_package_path: str | None = None
    cast_package_hash: str | None = None
    project_ref: str | None = None
    export_selection_policy: str | None = None
    loaded_kernels: list[str] = Field(default_factory=list)
    precompiled_vs_jit_path: dict[str, str] = Field(default_factory=dict)
    selected_ops: list[str] = Field(default_factory=list)
    selected_kernel_ids: dict[str, str] = Field(default_factory=dict)
    selected_kernel_paths: dict[str, str] = Field(default_factory=dict)
    selected_source_hashes: dict[str, str] = Field(default_factory=dict)
    evidence_tiers: dict[str, str] = Field(default_factory=dict)
    benchmark_evidence_refs: dict[str, Any] = Field(default_factory=dict)
    rejected_export_candidate_summary: dict[str, Any] = Field(default_factory=dict)
    export_paper_eligible: bool | None = None
    uses_non_deployment_evidence: bool | None = None
    coverage: dict[str, Any] = Field(default_factory=dict)
    paper_claim_status: str | None = None
    claim_category: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    placement_profile: str | None = None
    cache_mode: str | None = None
    workload_slug: str | None = None
    quantization_config_hash: str | None = None
    model_license: str | None = None
    dataset_license: str | None = None
    toolchain_status: dict[str, Any] = Field(default_factory=dict)
    cache_reuse_status: str | None = None
    per_op_launch_coverage: dict[str, Any] = Field(default_factory=dict)

    @field_validator("variant", mode="before")
    @classmethod
    def _parse_variant(cls, value: Any):
        return _coerce_enum(value, Variant)

    @field_validator("stage", mode="before")
    @classmethod
    def _parse_stage(cls, value: Any):
        return _coerce_enum(value, Stage)

    @field_validator("correctness_status", mode="before")
    @classmethod
    def _parse_correctness_status(cls, value: Any):
        return _coerce_enum(value, CorrectnessStatus)


class SummaryArtifact(ArtifactBase):
    artifact_type: Literal["summary_report"] = "summary_report"
    rows: list[SummaryRow] = Field(default_factory=list)
    summary_markdown_path: str | None = None
    summary_csv_path: str | None = None
    sections: dict[str, Any] = Field(default_factory=dict)
    paper_eligible_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    failure_regressions: list[str] = Field(default_factory=list)
    raw_artifact_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_summary_paper_eligibility(self) -> SummaryArtifact:
        issues = list(self.paper_eligibility_issues)
        steady_rows = [row for row in self.rows if row.stage in PERFORMANCE_STAGES]
        claim_rows = [row for row in steady_rows if row.claim_eligible]

        if not steady_rows:
            issues.append("no steady-state rows found")
        if not claim_rows:
            issues.append("no claim-eligible steady-state rows found")

        if self.benchmark_mode in {BenchmarkMode.e2e_model, BenchmarkMode.deployment}:
            required_variants = {Variant.eager, Variant.torch_compile, Variant.kf_cast}
            variants_present = {row.variant for row in steady_rows if row.stage in MODEL_LEVEL_STAGES}
            for required in required_variants:
                if required not in variants_present:
                    issues.append(f"missing {required.value} variant for model-level comparison")
            stage_to_variants: dict[tuple[Stage, str | None], set[Variant]] = {}
            for row in claim_rows:
                if row.stage not in MODEL_LEVEL_STAGES:
                    continue
                stage_to_variants.setdefault((row.stage, row.comparison_group), set()).add(row.variant)
            if not any(required_variants.issubset(stage_variants) for stage_variants in stage_to_variants.values()):
                issues.append("no comparable model-level stage across eager, torch_compile, and kf_cast")
        elif self.benchmark_mode == BenchmarkMode.operator:
            operator_rows = [row for row in steady_rows if row.stage == Stage.operator]
            variants_present = {row.variant for row in operator_rows}
            if Variant.eager not in variants_present:
                issues.append("missing eager baseline for operator comparison")
            if not any(variant != Variant.eager for variant in variants_present):
                issues.append("missing comparable non-eager variant for operator comparison")

        deduped = list(dict.fromkeys(issues))
        self.paper_eligibility_issues = deduped
        self.paper_eligible = not deduped
        return self


ArtifactModel = RunManifestArtifact | EnvironmentArtifact | DeviceAuditArtifact | BenchmarkArtifact | SummaryArtifact


def artifact_model_for_payload(payload: dict[str, Any]) -> type[ArtifactModel]:
    artifact_type = payload.get("artifact_type")
    mapping: dict[str, type[ArtifactModel]] = {
        "run_manifest": RunManifestArtifact,
        "environment_snapshot": EnvironmentArtifact,
        "device_audit": DeviceAuditArtifact,
        "benchmark_result": BenchmarkArtifact,
        "summary_report": SummaryArtifact,
    }
    if artifact_type not in mapping:
        raise ValueError(f"Unknown artifact_type: {artifact_type!r}")
    return mapping[artifact_type]


def validate_artifact_payload(payload: dict[str, Any]) -> ArtifactModel:
    model_type = artifact_model_for_payload(payload)
    return model_type.model_validate(payload)


def artifact_schema_bundle() -> dict[str, Any]:
    return {
        "run_manifest": RunManifestArtifact.model_json_schema(),
        "environment_snapshot": EnvironmentArtifact.model_json_schema(),
        "device_audit": DeviceAuditArtifact.model_json_schema(),
        "benchmark_result": BenchmarkArtifact.model_json_schema(),
        "summary_report": SummaryArtifact.model_json_schema(),
    }
