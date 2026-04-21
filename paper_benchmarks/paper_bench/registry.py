from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from .provenance import compute_suite_hash, sha256_bytes, sha256_file, sha256_path
from .schema import BenchmarkMode, Stage, StrictModel, Variant


class ModelRegistryEntry(StrictModel):
    model_id: str = Field(min_length=1)
    loader_kind: str = Field(default="transformers_causal_lm", min_length=1)
    model_path: str | None = None
    model_resolver: str | None = None
    tokenizer_path: str | None = None
    model_config_path: str | None = None
    cast_package_path: str | None = None
    deployment_artifact_path: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    device_map: str | dict[str, Any] | None = None
    device: str | None = None
    max_memory: dict[str, str] | None = None
    local_files_only: bool = True
    attn_implementation: str | None = None
    expected_model_config_hash: str | None = None
    task_type: str | None = None
    category: str | None = None
    validation_suite_path: str | None = None
    workload_hash: str | None = None
    batch_sizes: list[int] = Field(default_factory=list)
    shape_or_length_buckets: list[dict[str, Any]] = Field(default_factory=list)
    correctness_comparator: str | None = None
    baselines_required: list[Variant] = Field(default_factory=list)
    expected_memory_footprint_notes: list[str] = Field(default_factory=list)
    paper_eligible: bool = False
    benchmark_expectation: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_payload(cls, value: Any):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "dtype" in payload and "torch_dtype" not in payload:
            payload["torch_dtype"] = payload.pop("dtype")
        if "device_placement" in payload and "device" not in payload:
            payload["device"] = payload.pop("device_placement")
        if "resolver" in payload and "model_resolver" not in payload:
            payload["model_resolver"] = payload.pop("resolver")
        if "suite_config_path" in payload and "validation_suite_path" not in payload:
            payload["validation_suite_path"] = payload.pop("suite_config_path")
        if "deployment_artifact_path" not in payload and payload.get("cast_package_path"):
            payload["deployment_artifact_path"] = payload.get("cast_package_path")
        return payload

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Any):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("expected_memory_footprint_notes", mode="before")
    @classmethod
    def _normalize_expected_memory_notes(cls, value: Any):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("batch_sizes", mode="before")
    @classmethod
    def _normalize_batch_sizes(cls, value: Any):
        if value in (None, []):
            return []
        if not isinstance(value, list):
            raise TypeError("batch_sizes must be a list")
        return [int(item) for item in value]

    @field_validator("baselines_required", mode="before")
    @classmethod
    def _parse_baselines_required(cls, value: Any):
        if value in (None, []):
            return []
        if not isinstance(value, list):
            raise TypeError("baselines_required must be a list")
        return [item if isinstance(item, Variant) else Variant(item) for item in value]

    @model_validator(mode="after")
    def _validate_model_locator(self) -> ModelRegistryEntry:
        if not self.model_path and not self.model_resolver:
            raise ValueError("model_path or model_resolver is required")
        if self.batch_sizes:
            self.batch_sizes = [int(item) for item in self.batch_sizes]
        return self


class ModelRegistryFile(StrictModel):
    version: int = Field(default=1, ge=1)
    models: list[ModelRegistryEntry] = Field(default_factory=list)


class PromptLengthBucket(StrictModel):
    bucket_id: str = Field(min_length=1)
    min_tokens: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_bucket_payload(cls, value: Any):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "id" in payload and "bucket_id" not in payload:
            payload["bucket_id"] = payload.pop("id")
        return payload

    @model_validator(mode="after")
    def _validate_bounds(self) -> PromptLengthBucket:
        if self.min_tokens is None and self.max_tokens is None:
            raise ValueError("prompt length bucket requires min_tokens or max_tokens")
        if self.min_tokens is not None and self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError("prompt length bucket min_tokens cannot exceed max_tokens")
        return self


class SuiteConfig(StrictModel):
    suite_id: str = Field(min_length=1)
    benchmark_mode: BenchmarkMode
    description: str | None = None
    workload_type: str = Field(default="prompt_records", min_length=1)
    workload_path: str = Field(min_length=1)
    synthetic_workload: bool = False
    variants: list[Variant] = Field(min_length=1)
    stages: list[Stage] = Field(min_length=1)
    warmup_count: int = Field(default=1, ge=0)
    timed_run_count: int = Field(default=3, ge=1)
    batch_size: int = Field(default=1, ge=1)
    batch_sizes: list[int] = Field(default_factory=list)
    max_new_tokens: int | None = Field(default=None, ge=1)
    prompt_length_buckets: list[PromptLengthBucket] = Field(default_factory=list)
    shape_or_length_buckets: list[dict[str, Any]] = Field(default_factory=list)
    generation_mode: str = Field(default="greedy", min_length=1)
    include_tokenization_in_timing: bool = False
    measure_prefill_decode_separately: bool = True
    callable_name: str | None = None
    device: str = Field(default="cuda", min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_suite_payload(cls, value: Any):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "prompt_file" in payload and "workload_path" not in payload:
            payload["workload_path"] = payload.pop("prompt_file")
        if "warmup_runs" in payload and "warmup_count" not in payload:
            payload["warmup_count"] = payload.pop("warmup_runs")
        if "timed_runs" in payload and "timed_run_count" not in payload:
            payload["timed_run_count"] = payload.pop("timed_runs")
        if "batch_sizes" in payload and "batch_size" not in payload:
            batch_sizes = payload.get("batch_sizes") or []
            if len(batch_sizes) == 1:
                payload["batch_size"] = batch_sizes[0]
        if "shape_buckets" in payload and "shape_or_length_buckets" not in payload:
            payload["shape_or_length_buckets"] = payload.pop("shape_buckets")
        if "length_buckets" in payload and "shape_or_length_buckets" not in payload:
            payload["shape_or_length_buckets"] = payload.pop("length_buckets")
        if payload.get("benchmark_mode") == BenchmarkMode.operator.value and "workload_type" not in payload:
            payload["workload_type"] = "operator_entries"
        return payload

    @field_validator("benchmark_mode", mode="before")
    @classmethod
    def _parse_benchmark_mode(cls, value):
        if isinstance(value, BenchmarkMode):
            return value
        return BenchmarkMode(value)

    @field_validator("variants", mode="before")
    @classmethod
    def _parse_variants(cls, value):
        if not isinstance(value, list):
            raise TypeError("variants must be a list")
        return [item if isinstance(item, Variant) else Variant(item) for item in value]

    @field_validator("stages", mode="before")
    @classmethod
    def _parse_stages(cls, value):
        if not isinstance(value, list):
            raise TypeError("stages must be a list")
        return [item if isinstance(item, Stage) else Stage(item) for item in value]

    @field_validator("batch_sizes", mode="before")
    @classmethod
    def _parse_batch_sizes(cls, value):
        if value in (None, []):
            return []
        if not isinstance(value, list):
            raise TypeError("batch_sizes must be a list")
        return [int(item) for item in value]

    @field_validator("generation_mode", mode="before")
    @classmethod
    def _parse_generation_mode(cls, value):
        if value is None:
            return "greedy"
        return str(value).strip().lower()

    @field_validator("notes", mode="before")
    @classmethod
    def _parse_notes(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate_mode(self) -> SuiteConfig:
        if self.benchmark_mode == BenchmarkMode.operator and self.workload_type != "operator_entries":
            raise ValueError("operator benchmark_mode requires workload_type=operator_entries")
        if self.benchmark_mode != BenchmarkMode.operator and self.workload_type not in {
            "prompt_jsonl",
            "prompt_yaml",
            "prompt_records",
            "dataset_records",
            "tensor_corpus",
            "image_folder",
            "encoder_inputs",
            "vision_batch",
            "seq2seq_records",
            "embedding_batch",
            "dynamic_shape_records",
        }:
            raise ValueError("Unsupported non-operator workload_type")
        if self.benchmark_mode == BenchmarkMode.operator and not self.callable_name:
            raise ValueError("operator suites require callable_name")
        if self.benchmark_mode != BenchmarkMode.operator and self.generation_mode != "greedy":
            raise ValueError("Paper LLM harness only supports generation_mode=greedy")
        if not self.batch_sizes:
            self.batch_sizes = [int(self.batch_size)]
        else:
            self.batch_sizes = [int(item) for item in self.batch_sizes]
        if not self.batch_sizes:
            raise ValueError("Suite requires at least one batch size")
        self.batch_size = int(self.batch_sizes[0])
        return self


class SyntheticWorkloadError(RuntimeError):
    pass


class ModelSuiteRegistryEntry(StrictModel):
    model_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    model_config_ref: str = Field(min_length=1, alias="model_config")
    suite_config_ref: str = Field(min_length=1, alias="suite_config")
    benchmark_expectation: str = Field(default="unknown", min_length=1)
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Any):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class ModelSuiteRegistryFile(StrictModel):
    version: int = Field(default=1, ge=1)
    models: list[ModelSuiteRegistryEntry] = Field(default_factory=list)


class ModelRegistryValidationItem(StrictModel):
    model_id: str
    category: str
    benchmark_expectation: str
    model_config_path: str
    suite_config_path: str
    workload_path: str | None = None
    workload_hash_declared: str | None = None
    workload_hash_actual: str | None = None
    baselines_required: list[Variant] = Field(default_factory=list)
    paper_eligible_declared: bool = False
    paper_eligible_effective: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)

    @field_validator("baselines_required", mode="before")
    @classmethod
    def _parse_baselines_required(cls, value: Any):
        if value in (None, []):
            return []
        if not isinstance(value, list):
            raise TypeError("baselines_required must be a list")
        return [item if isinstance(item, Variant) else Variant(item) for item in value]


class ModelRegistryValidationReport(StrictModel):
    registry_path: str
    model_count: int = Field(ge=0)
    runnable_count: int = Field(ge=0)
    paper_eligible_count: int = Field(ge=0)
    incomplete_count: int = Field(ge=0)
    entries: list[ModelRegistryValidationItem] = Field(default_factory=list)
    missing_fields_by_model: dict[str, list[str]] = Field(default_factory=dict)
    issue_count_by_model: dict[str, int] = Field(default_factory=dict)


def _read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _resolve_path_like(base_dir: Path, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _resolve_generic_paths(base_dir: Path, payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    updated = dict(payload)
    for key in keys:
        updated[key] = _resolve_path_like(base_dir, updated.get(key))
    return updated


def load_model_registry(path: str | Path) -> ModelRegistryFile:
    payload = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    if isinstance(payload.get("models"), list):
        normalized_models = []
        for entry in payload["models"]:
            if not isinstance(entry, dict):
                normalized_models.append(entry)
                continue
            model_payload = dict(entry)
            for key in (
                "model_path",
                "tokenizer_path",
                "model_config_path",
                "cast_package_path",
                "deployment_artifact_path",
                "validation_suite_path",
            ):
                model_payload[key] = _resolve_path_like(base_dir, model_payload.get(key))
            normalized_models.append(model_payload)
        payload["models"] = normalized_models
    return ModelRegistryFile.model_validate(payload)


def resolve_model(registry_path: str | Path, model_id: str) -> ModelRegistryEntry:
    registry = load_model_registry(registry_path)
    for entry in registry.models:
        if entry.model_id == model_id:
            return entry
    raise KeyError(f"Model {model_id!r} not found in registry {registry_path}")


def load_model_config(path: str | Path) -> ModelRegistryEntry:
    payload = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    payload = _resolve_generic_paths(
        base_dir,
        payload,
        (
            "model_path",
            "tokenizer_path",
            "model_config_path",
            "cast_package_path",
            "deployment_artifact_path",
            "validation_suite_path",
        ),
    )
    return ModelRegistryEntry.model_validate(payload)


def load_suite_config(path: str | Path) -> SuiteConfig:
    payload = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    if "prompt_file" in payload and "workload_path" not in payload:
        payload["workload_path"] = payload["prompt_file"]
        payload.pop("prompt_file", None)
    payload["workload_path"] = _resolve_path_like(base_dir, payload.get("workload_path"))
    return SuiteConfig.model_validate(payload)


def load_model_suite_registry(path: str | Path) -> ModelSuiteRegistryFile:
    payload = _read_yaml(path)
    base_dir = Path(path).resolve().parent
    if isinstance(payload.get("models"), list):
        normalized = []
        for entry in payload["models"]:
            if not isinstance(entry, dict):
                normalized.append(entry)
                continue
            normalized.append(
                _resolve_generic_paths(
                    base_dir,
                    entry,
                    ("model_config", "suite_config"),
                )
            )
        payload["models"] = normalized
    return ModelSuiteRegistryFile.model_validate(payload)


def _required_missing(entry: ModelRegistryEntry, suite_path: str, suite: SuiteConfig | None) -> list[str]:
    missing: list[str] = []
    if not entry.model_id:
        missing.append("model_id")
    if not (entry.model_path or entry.model_resolver):
        missing.append("model_path_or_model_resolver")
    if not entry.task_type:
        missing.append("task_type")
    if not entry.torch_dtype:
        missing.append("dtype")
    if entry.device is None and entry.device_map is None:
        missing.append("device_placement")
    if not entry.validation_suite_path and not suite_path:
        missing.append("validation_suite_path")
    if not entry.workload_hash:
        missing.append("workload_hash")
    if not entry.batch_sizes:
        missing.append("batch_sizes")
    if not entry.shape_or_length_buckets:
        missing.append("shape_or_length_buckets")
    if not entry.correctness_comparator:
        missing.append("correctness_comparator")
    if not entry.baselines_required:
        missing.append("baselines_required")
    if not entry.expected_memory_footprint_notes:
        missing.append("expected_memory_footprint_notes")
    if suite is None:
        missing.append("suite_config")
    return missing


def _baseline_requirement_issues(entry: ModelRegistryEntry) -> list[str]:
    issues: list[str] = []
    baselines = set(entry.baselines_required)
    if Variant.eager not in baselines:
        issues.append("baselines_required missing eager")
    if Variant.torch_compile not in baselines:
        issues.append("baselines_required missing torch_compile")
    return issues


def validate_model_suite_registry(path: str | Path) -> ModelRegistryValidationReport:
    registry = load_model_suite_registry(path)
    entries: list[ModelRegistryValidationItem] = []
    for registry_entry in registry.models:
        model_config_path = registry_entry.model_config_ref
        suite_config_path = registry_entry.suite_config_ref
        validation_issues: list[str] = []
        suite = None
        try:
            model = load_model_config(model_config_path)
        except Exception as exc:
            entries.append(
                ModelRegistryValidationItem(
                    model_id=registry_entry.model_id,
                    category=registry_entry.category,
                    benchmark_expectation=registry_entry.benchmark_expectation,
                    model_config_path=model_config_path,
                    suite_config_path=suite_config_path,
                    missing_fields=["model_config"],
                    validation_issues=[f"model config load failed: {type(exc).__name__}: {exc}"],
                )
            )
            continue

        try:
            suite = load_suite_config(suite_config_path)
        except Exception as exc:
            validation_issues.append(f"suite config load failed: {type(exc).__name__}: {exc}")

        missing_fields = _required_missing(model, suite_config_path, suite)
        validation_issues.extend(_baseline_requirement_issues(model))
        if not model.correctness_comparator:
            validation_issues.append("correctness comparator missing")

        actual_workload_hash = None
        workload_path = None
        if suite is not None:
            workload_path = suite.workload_path
            if not workload_path or not Path(workload_path).exists():
                validation_issues.append("workload file missing")
            else:
                actual_workload_hash = sha256_path(workload_path)
                if model.workload_hash and model.workload_hash != actual_workload_hash:
                    validation_issues.append("workload hash mismatch")

        if model.validation_suite_path and Path(model.validation_suite_path).resolve() != Path(suite_config_path).resolve():
            validation_issues.append("validation suite path does not match registry suite_config")

        if model.paper_eligible and ("workload file missing" in validation_issues or any(issue.startswith("baselines_required") for issue in validation_issues)):
            validation_issues.append("paper benchmark blocked by missing workload or incomplete baseline requirements")

        paper_effective = bool(model.paper_eligible and not missing_fields and not validation_issues)
        entries.append(
            ModelRegistryValidationItem(
                model_id=model.model_id,
                category=registry_entry.category,
                benchmark_expectation=registry_entry.benchmark_expectation,
                model_config_path=model_config_path,
                suite_config_path=suite_config_path,
                workload_path=workload_path,
                workload_hash_declared=model.workload_hash,
                workload_hash_actual=actual_workload_hash,
                baselines_required=model.baselines_required,
                paper_eligible_declared=bool(model.paper_eligible),
                paper_eligible_effective=paper_effective,
                missing_fields=list(dict.fromkeys(missing_fields)),
                validation_issues=list(dict.fromkeys(validation_issues)),
            )
        )

    missing_by_model = {entry.model_id: entry.missing_fields for entry in entries if entry.missing_fields}
    issues_by_model = {entry.model_id: len(entry.validation_issues) for entry in entries}
    return ModelRegistryValidationReport(
        registry_path=str(Path(path).resolve()),
        model_count=len(entries),
        runnable_count=sum(1 for entry in entries if not entry.missing_fields and not entry.validation_issues),
        paper_eligible_count=sum(1 for entry in entries if entry.paper_eligible_effective),
        incomplete_count=sum(1 for entry in entries if entry.missing_fields or entry.validation_issues or not entry.paper_eligible_effective),
        entries=entries,
        missing_fields_by_model=missing_by_model,
        issue_count_by_model=issues_by_model,
    )

def enforce_workload_policy(suite: SuiteConfig, allow_synthetic_demo: bool) -> bool:
    if not suite.synthetic_workload:
        return True
    if not allow_synthetic_demo:
        raise SyntheticWorkloadError(
            "Synthetic workloads are forbidden for paper runs. Pass --allow-synthetic-demo to proceed."
        )
    return False


def dump_config_json(path: str | Path, payload: StrictModel) -> None:
    Path(path).write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
