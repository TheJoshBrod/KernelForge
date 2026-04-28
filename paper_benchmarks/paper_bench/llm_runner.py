from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import yaml

from .artifacts import RunLayout, write_json_artifact
from .baselines import compile_model, compile_settings_from_dict, load_transformers_causal_lm, sync_device, timed_call
from .cache import CacheRequest, copy_reused_artifact_set, find_matching_reusable_artifact, make_cache_request
from .correctness import reference_correctness, skipped_correctness
from .kf_runtime import (
    build_kf_runtime_details,
    get_cast_runtime_stats,
    load_cast_model,
    reset_cast_runtime_stats,
    runtime_settings_from_dict,
)
from .schema import BenchmarkArtifact, CorrectnessStatus, DeviceAuditArtifact, EnvironmentArtifact, RunManifestArtifact, Stage, Variant
from .stats import build_latency_summary
from .validator import validated_artifact_update


@dataclass(frozen=True)
class GenerationConfig:
    batch_size: int
    max_new_tokens: int
    pad_token_id: int
    eos_token_id: int | None
    do_sample: bool = False


@dataclass
class GenerationRunResult:
    prompt_ids: list[str]
    batch_size: int
    prompt_lengths: list[int]
    prompt_token_count: int
    generated_lengths: list[int]
    generated_token_count: int
    decode_generated_token_count: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    decode_step_samples_ms: list[float]
    input_batch_hash: str
    output_token_hashes: list[str]
    batch_output_hash: str
    generated_tokens: list[list[int]]
    tokenization_ms: float = 0.0


@dataclass(frozen=True)
class PromptSuiteData:
    path: str
    source_format: str
    records: list[dict[str, Any]]
    synthetic_workload: bool
    manifest_path: str | None = None
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromptWorkloadSlice:
    batch_size: int
    prompt_bucket_id: str
    comparison_group: str
    prompt_batches: list[list[dict[str, Any]]]
    selected_prompt_ids: list[str]


def _load_prompt_suite_manifest(prompt_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    manifest_candidates = [
        prompt_path.with_suffix(".manifest.json"),
        Path(f"{prompt_path}.manifest.json"),
    ]
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), None)
    if manifest_path is None:
        return None, None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt suite manifest must be a JSON object in {manifest_path}")
    expected_hash = payload.get("prompt_file_hash")
    if expected_hash:
        actual_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if str(expected_hash) != actual_hash:
            raise ValueError(
                f"Prompt suite hash mismatch for {prompt_path}: "
                f"manifest={expected_hash} actual={actual_hash}"
            )
    return payload, str(manifest_path)


def load_prompt_records(prompt_path: str | Path) -> PromptSuiteData:
    path = Path(prompt_path)
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    synthetic_workload = False
    manifest, manifest_path = _load_prompt_suite_manifest(path)
    if manifest is not None:
        synthetic_workload = bool(manifest.get("synthetic_workload", False))

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Prompt record {idx} is not a JSON object")
                text = payload.get("text", payload.get("prompt"))
                prompt_id = payload.get("id", payload.get("prompt_id"))
                if not isinstance(prompt_id, str) or not prompt_id:
                    raise ValueError(f"Prompt record {idx} missing non-empty 'id'")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Prompt record {idx} missing non-empty 'text'")
                record = dict(payload)
                record["id"] = prompt_id
                record["text"] = text
                records.append(record)
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            synthetic_workload = bool(
                synthetic_workload or payload.get("synthetic_workload") or payload.get("synthetic")
            )
            raw_records = payload.get("prompts") or payload.get("records") or []
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise ValueError(f"Prompt YAML payload must be a mapping or list in {path}")
        if not isinstance(raw_records, list):
            raise ValueError(f"Prompt YAML records must be a list in {path}")
        for idx, item in enumerate(raw_records):
            if not isinstance(item, dict):
                raise ValueError(f"Prompt record {idx} is not a mapping")
            text = item.get("text", item.get("prompt"))
            prompt_id = item.get("id", item.get("prompt_id"))
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"Prompt record {idx} missing non-empty 'id'")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Prompt record {idx} missing non-empty 'text'")
            record = dict(item)
            record["id"] = prompt_id
            record["text"] = text
            records.append(record)
    else:
        raise ValueError(f"Unsupported prompt file format for {prompt_path}; expected .jsonl, .yaml, or .yml")

    if not records:
        raise ValueError(f"No prompt records found in {prompt_path}")
    return PromptSuiteData(
        path=str(path),
        source_format="jsonl" if suffix == ".jsonl" else "yaml",
        records=records,
        synthetic_workload=synthetic_workload,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _chunk_prompt_records(records: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        list(records[index : index + batch_size])
        for index in range(0, len(records), batch_size)
    ]


def _sanitize_label(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    return sanitized or "default"


def _suite_batch_sizes(suite) -> list[int]:
    batch_sizes = getattr(suite, "batch_sizes", None)
    if batch_sizes:
        return [int(item) for item in batch_sizes]
    return [int(getattr(suite, "batch_size", 1))]


def _measure_prefill_decode_separately(suite) -> bool:
    return bool(getattr(suite, "measure_prefill_decode_separately", True))


def _include_tokenization_in_timing(suite) -> bool:
    return bool(getattr(suite, "include_tokenization_in_timing", False))


def _suite_uses_kv_cache(suite) -> bool:
    cache_mode = str(getattr(suite, "cache_mode", "kv_cache_on") or "kv_cache_on").strip().lower()
    return cache_mode != "kv_cache_off"


def _resolve_suite_device(suite, model_spec) -> str:
    return str(getattr(suite, "device", None) or getattr(model_spec, "device", None) or "cuda")


def _resolve_input_device(model, requested_device: str) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.device)
    except Exception:
        return requested_device


def _normalize_tokenized_mapping(tokenized: Any) -> dict[str, Any]:
    if isinstance(tokenized, dict):
        return dict(tokenized)
    if isinstance(tokenized, Mapping):
        return dict(tokenized.items())
    if hasattr(tokenized, "items"):
        return dict(tokenized.items())
    raise ValueError("Tokenizer must return a mapping with input_ids")


def _compute_prompt_token_length(tokenizer, text: str) -> int:
    tokenized = _normalize_tokenized_mapping(tokenizer(text, return_tensors="pt", padding=True))
    if "input_ids" not in tokenized:
        raise ValueError("Tokenizer must return input_ids when computing prompt lengths")
    attention_mask = tokenized.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask.sum().item())
    input_ids = tokenized["input_ids"]
    if torch.is_tensor(input_ids):
        if input_ids.ndim == 1:
            return int(input_ids.shape[0])
        return int(input_ids.shape[-1])
    raise ValueError("Tokenizer returned unsupported input_ids for prompt length computation")


def _prompt_matches_bucket(token_length: int, bucket) -> bool:
    min_tokens = getattr(bucket, "min_tokens", None)
    max_tokens = getattr(bucket, "max_tokens", None)
    if min_tokens is not None and token_length < int(min_tokens):
        return False
    if max_tokens is not None and token_length > int(max_tokens):
        return False
    return True


def _select_prompt_workload_slices(prompt_suite: PromptSuiteData, tokenizer, suite) -> list[PromptWorkloadSlice]:
    records = list(prompt_suite.records)
    token_lengths = [_compute_prompt_token_length(tokenizer, record["text"]) for record in records]
    bucket_specs = list(getattr(suite, "prompt_length_buckets", []) or [])
    if not bucket_specs:
        bucket_specs = [type("PromptBucket", (), {"bucket_id": "all_prompts", "min_tokens": None, "max_tokens": None})()]

    slices: list[PromptWorkloadSlice] = []
    timed_run_count = int(getattr(suite, "timed_run_count", 1))
    for bucket in bucket_specs:
        filtered_records = [
            record
            for record, token_length in zip(records, token_lengths, strict=True)
            if _prompt_matches_bucket(token_length, bucket)
        ]
        if not filtered_records:
            raise ValueError(f"No prompts matched bucket {bucket.bucket_id!r}")
        for batch_size in _suite_batch_sizes(suite):
            prompt_batches = _chunk_prompt_records(filtered_records, int(batch_size))
            if len(prompt_batches) < timed_run_count:
                raise ValueError(
                    f"Bucket {bucket.bucket_id!r} with batch size {batch_size} only provides {len(prompt_batches)} timed batches; "
                    f"suite requires {timed_run_count}"
                )
            comparison_group = _sanitize_label(f"{bucket.bucket_id}_bs{batch_size}")
            selected_prompt_ids = [
                record["id"]
                for batch in prompt_batches[:timed_run_count]
                for record in batch
            ]
            slices.append(
                PromptWorkloadSlice(
                    batch_size=int(batch_size),
                    prompt_bucket_id=str(bucket.bucket_id),
                    comparison_group=comparison_group,
                    prompt_batches=prompt_batches,
                    selected_prompt_ids=selected_prompt_ids,
                )
            )
    return slices


def _resolve_generation_config(tokenizer, model, suite, *, batch_size: int) -> GenerationConfig:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None and hasattr(model, "config"):
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = int(eos_token_id[0]) if eos_token_id else None

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and hasattr(model, "config"):
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = int(pad_token_id[0]) if pad_token_id else None
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    try:
        tokenizer.pad_token_id = int(pad_token_id)
    except Exception:
        pass

    return GenerationConfig(
        batch_size=int(batch_size),
        max_new_tokens=int(suite.max_new_tokens or 1),
        pad_token_id=int(pad_token_id),
        eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
        do_sample=False,
    )


def _generation_settings_payload(generation_config: GenerationConfig) -> dict[str, Any]:
    return {
        "generation_mode": "greedy",
        "do_sample": generation_config.do_sample,
        "max_new_tokens": generation_config.max_new_tokens,
        "pad_token_id": generation_config.pad_token_id,
        "eos_token_id": generation_config.eos_token_id,
        "batch_size": generation_config.batch_size,
    }


def _tokenize_prompt_batch(
    tokenizer,
    records: Sequence[dict[str, Any]],
    device: str,
    *,
    measure_timing: bool = False,
) -> tuple[dict[str, torch.Tensor], float]:
    prompts = [record["text"] for record in records]
    start = time.perf_counter()
    tokenized = _normalize_tokenized_mapping(tokenizer(prompts, return_tensors="pt", padding=True))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if "input_ids" not in tokenized:
        raise ValueError("Tokenizer batch missing input_ids")
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in tokenized.items()
    }
    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return batch, float(elapsed_ms if measure_timing else 0.0)


def _tokenizer_output_devices(tokenized: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value.device)
        for key, value in tokenized.items()
        if torch.is_tensor(value)
    }


def _summarize_tokenized_batch(tokenized: dict[str, torch.Tensor]) -> tuple[int, list[int], int, str]:
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    batch_size = int(input_ids.shape[0])
    prompt_lengths = [int(item) for item in attention_mask.sum(dim=1).tolist()]
    prompt_token_count = int(sum(prompt_lengths))
    input_batch_hash = _hash_json_payload(
        {
            "input_ids": input_ids.detach().cpu().tolist(),
            "attention_mask": attention_mask.detach().cpu().tolist(),
        }
    )
    return batch_size, prompt_lengths, prompt_token_count, input_batch_hash


def _execute_greedy_generation(
    model,
    tokenized: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    device: str,
    *,
    measure_timings: bool,
    tokenization_ms: float = 0.0,
    include_tokenization_in_timing: bool = False,
    use_kv_cache: bool = True,
) -> GenerationRunResult:
    batch_size, prompt_lengths, prompt_token_count, input_batch_hash = _summarize_tokenized_batch(tokenized)

    if not use_kv_cache:
        current_input_ids = tokenized["input_ids"]
        current_attention_mask = tokenized["attention_mask"]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=current_input_ids.device)
        generated_tensors: list[torch.Tensor] = []
        generated_lengths = torch.zeros(batch_size, dtype=torch.int64, device=current_input_ids.device)
        prefill_ms = 0.0
        decode_step_samples_ms: list[float] = []

        for step_index in range(generation_config.max_new_tokens):
            def run_full_context_step():
                return model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=False,
                )

            if measure_timings:
                step_call = timed_call(device, run_full_context_step, inference_mode=True)
                outputs = step_call.value
                if step_index == 0:
                    prefill_ms = float(step_call.elapsed_ms)
                else:
                    decode_step_samples_ms.append(float(step_call.elapsed_ms))
            else:
                with torch.inference_mode():
                    outputs = run_full_context_step()

            active_mask = ~finished
            if active_mask.any():
                generated_lengths[active_mask] += 1

            raw_next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if generation_config.eos_token_id is not None:
                finished = finished | raw_next_token.squeeze(-1).eq(generation_config.eos_token_id)
            next_token = raw_next_token.clone()
            if finished.any():
                next_token[finished] = generation_config.pad_token_id
            generated_tensors.append(next_token)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(next_token, dtype=current_attention_mask.dtype)],
                dim=1,
            )

        generated_tensor = torch.cat(generated_tensors, dim=1)
        generated_tokens = [
            [int(token) for token in row]
            for row in generated_tensor.detach().cpu().tolist()
        ]
        output_token_hashes = [_hash_json_payload(row) for row in generated_tokens]
        batch_output_hash = _hash_json_payload(generated_tokens)
        generated_length_list = [int(length) for length in generated_lengths.detach().cpu().tolist()]
        generated_token_count = int(sum(generated_length_list))
        decode_generated_token_count = int(max(generated_token_count - batch_size, 0))
        decode_ms = float(sum(decode_step_samples_ms))
        total_ms = float(prefill_ms + decode_ms + (tokenization_ms if include_tokenization_in_timing else 0.0))

        return GenerationRunResult(
            prompt_ids=[],
            batch_size=batch_size,
            prompt_lengths=prompt_lengths,
            prompt_token_count=prompt_token_count,
            generated_lengths=generated_length_list,
            generated_token_count=generated_token_count,
            decode_generated_token_count=decode_generated_token_count,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            decode_step_samples_ms=decode_step_samples_ms,
            input_batch_hash=input_batch_hash,
            output_token_hashes=output_token_hashes,
            batch_output_hash=batch_output_hash,
            generated_tokens=generated_tokens,
            tokenization_ms=float(tokenization_ms),
        )

    def run_prefill():
        return model(**tokenized, use_cache=True)

    if measure_timings:
        prefill_call = timed_call(device, run_prefill, inference_mode=True)
        outputs = prefill_call.value
        prefill_ms = float(prefill_call.elapsed_ms)
    else:
        with torch.inference_mode():
            outputs = run_prefill()
        prefill_ms = 0.0

    past_key_values = getattr(outputs, "past_key_values", None)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tensors = [next_token]
    generated_lengths = torch.ones(batch_size, dtype=torch.int64, device=next_token.device)
    if generation_config.eos_token_id is None:
        finished = torch.zeros(batch_size, dtype=torch.bool, device=next_token.device)
    else:
        finished = next_token.squeeze(-1).eq(generation_config.eos_token_id)

    decode_step_samples_ms: list[float] = []
    for _ in range(generation_config.max_new_tokens - 1):
        step_input = next_token.clone()
        if finished.any():
            step_input[finished] = generation_config.pad_token_id

        def run_decode_step():
            return model(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

        if measure_timings:
            decode_call = timed_call(device, run_decode_step, inference_mode=True)
            outputs = decode_call.value
            decode_step_samples_ms.append(float(decode_call.elapsed_ms))
        else:
            with torch.inference_mode():
                outputs = run_decode_step()

        active_mask = ~finished
        if active_mask.any():
            generated_lengths[active_mask] += 1

        past_key_values = getattr(outputs, "past_key_values", None)
        raw_next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if generation_config.eos_token_id is not None:
            finished = finished | raw_next_token.squeeze(-1).eq(generation_config.eos_token_id)
        next_token = raw_next_token.clone()
        if finished.any():
            next_token[finished] = generation_config.pad_token_id
        generated_tensors.append(next_token)

    generated_tensor = torch.cat(generated_tensors, dim=1)
    generated_tokens = [
        [int(token) for token in row]
        for row in generated_tensor.detach().cpu().tolist()
    ]
    output_token_hashes = [_hash_json_payload(row) for row in generated_tokens]
    batch_output_hash = _hash_json_payload(generated_tokens)
    generated_length_list = [int(length) for length in generated_lengths.detach().cpu().tolist()]
    generated_token_count = int(sum(generated_length_list))
    decode_generated_token_count = int(max(generated_token_count - batch_size, 0))
    decode_ms = float(sum(decode_step_samples_ms))
    total_ms = float(prefill_ms + decode_ms + (tokenization_ms if include_tokenization_in_timing else 0.0))

    return GenerationRunResult(
        prompt_ids=[],
        batch_size=batch_size,
        prompt_lengths=prompt_lengths,
        prompt_token_count=prompt_token_count,
        generated_lengths=generated_length_list,
        generated_token_count=generated_token_count,
        decode_generated_token_count=decode_generated_token_count,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        decode_step_samples_ms=decode_step_samples_ms,
        input_batch_hash=input_batch_hash,
        output_token_hashes=output_token_hashes,
        batch_output_hash=batch_output_hash,
        generated_tokens=generated_tokens,
        tokenization_ms=float(tokenization_ms),
    )


def _safe_rate(tokens: int, latency_ms: float) -> float | None:
    if tokens <= 0 or latency_ms <= 0:
        return None
    return float(tokens * 1000.0 / latency_ms)


def _build_stage_sample_record(
    sample_index: int,
    result: GenerationRunResult,
    *,
    stage: Stage,
    prompt_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "stage": stage.value,
        "prompt_ids": list(prompt_ids),
        "batch_size": result.batch_size,
        "prompt_lengths": result.prompt_lengths,
        "prompt_token_count": result.prompt_token_count,
        "generated_lengths": result.generated_lengths,
        "generated_token_count": result.generated_token_count,
        "decode_generated_token_count": result.decode_generated_token_count,
        "generated_tokens": result.generated_tokens,
        "latency_ms": {
            Stage.prefill: result.prefill_ms,
            Stage.decode: result.decode_ms,
            Stage.total_generate: result.total_ms,
        }[stage],
        "tokenization_ms": result.tokenization_ms,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "total_generate_ms": result.total_ms,
        "decode_step_samples_ms": result.decode_step_samples_ms,
        "input_batch_hash": result.input_batch_hash,
        "output_token_hashes": result.output_token_hashes,
        "batch_output_hash": result.batch_output_hash,
        "prefill_tokens_per_second": _safe_rate(result.prompt_token_count, result.prefill_ms),
        "decode_tokens_per_second": _safe_rate(result.decode_generated_token_count, result.decode_ms),
        "total_generated_tokens_per_second": _safe_rate(result.generated_token_count, result.total_ms),
        "latency_per_generated_token_ms": (
            float(result.total_ms / result.generated_token_count)
            if result.generated_token_count > 0
            else None
        ),
    }


def _build_failure_sample_record(
    sample_index: int,
    *,
    stage: Stage,
    prompt_ids: Sequence[str],
    batch_size: int,
    prompt_lengths: Sequence[int],
    input_batch_hash: str | None,
    tokenization_ms: float,
    latency_ms: float,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "stage": stage.value,
        "prompt_ids": list(prompt_ids),
        "batch_size": int(batch_size),
        "prompt_lengths": [int(item) for item in prompt_lengths],
        "prompt_token_count": int(sum(int(item) for item in prompt_lengths)),
        "generated_lengths": [],
        "generated_token_count": 0,
        "decode_generated_token_count": 0,
        "generated_tokens": [],
        "latency_ms": float(latency_ms),
        "tokenization_ms": float(tokenization_ms),
        "prefill_ms": None,
        "decode_ms": None,
        "total_generate_ms": float(latency_ms) if stage == Stage.total_generate else None,
        "decode_step_samples_ms": [],
        "input_batch_hash": input_batch_hash,
        "output_token_hashes": [],
        "batch_output_hash": None,
        "execution_status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _build_stage_details(
    stage: Stage,
    sample_records: Sequence[dict[str, Any]],
    generation_config: GenerationConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency_key = {
        Stage.prefill: "prefill_ms",
        Stage.decode: "decode_ms",
        Stage.total_generate: "total_generate_ms",
    }.get(stage)
    aggregate = {
        "prompt_token_count_total": int(sum(record["prompt_token_count"] for record in sample_records)),
        "generated_token_count_total": int(sum(record["generated_token_count"] for record in sample_records)),
        "decode_generated_token_count_total": int(sum(record["decode_generated_token_count"] for record in sample_records)),
        "tokenization_ms_total": float(sum(float(record.get("tokenization_ms", 0.0) or 0.0) for record in sample_records)),
    }
    if latency_key:
        total_latency_ms = float(sum(float(record.get(latency_key, 0.0) or 0.0) for record in sample_records))
        aggregate["stage_tokens_per_second"] = _safe_rate(
            aggregate["prompt_token_count_total"] if stage == Stage.prefill else aggregate["decode_generated_token_count_total"] if stage == Stage.decode else aggregate["generated_token_count_total"],
            total_latency_ms,
        )
    details = {
        "generation_settings": _generation_settings_payload(generation_config),
        "aggregate_metrics": aggregate,
    }
    if extra:
        details.update(extra)
    return details


def _write_stage_artifact(
    layout: RunLayout,
    common: dict[str, Any],
    *,
    variant: Variant,
    stage: Stage,
    samples_ms: list[float],
    warmup_count: int,
    timed_run_count: int,
    correctness_status: CorrectnessStatus,
    correctness_message: str | None,
    compile_time_ms: float | None,
    steady_state_time_ms: float | None,
    fallback_count: int | None = None,
    kernel_hit_count: int | None = None,
    sample_records: list[dict[str, Any]] | None = None,
    details: dict[str, Any] | None = None,
    token_count: int | None = None,
    comparison_group: str | None = None,
    configured_batch_size: int | None = None,
    prompt_bucket_id: str | None = None,
    file_suffix: str | None = None,
) -> Path:
    artifact_common = dict(common)
    for key in (
        "artifact_type",
        "variant",
        "stage",
        "warmup_count",
        "timed_run_count",
        "latency_samples_ms",
        "latency_summary",
        "sample_records",
        "correctness_status",
        "correctness_message",
        "fallback_count",
        "kernel_hit_count",
        "compile_time_ms",
        "steady_state_time_ms",
        "prompt_id",
        "prompt_hash",
        "token_count",
        "details",
        "comparison_group",
        "configured_batch_size",
        "prompt_bucket_id",
    ):
        artifact_common.pop(key, None)
    artifact = BenchmarkArtifact(
        **artifact_common,
        artifact_type="benchmark_result",
        variant=variant,
        stage=stage,
        warmup_count=warmup_count,
        timed_run_count=timed_run_count,
        latency_samples_ms=samples_ms,
        latency_summary=build_latency_summary(samples_ms),
        sample_records=sample_records or [],
        correctness_status=correctness_status,
        correctness_message=correctness_message,
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        compile_time_ms=compile_time_ms,
        steady_state_time_ms=steady_state_time_ms,
        details=details or {},
        token_count=token_count,
        comparison_group=comparison_group,
        configured_batch_size=configured_batch_size,
        prompt_bucket_id=prompt_bucket_id,
    )
    artifact = validated_artifact_update(artifact)
    suffix = f"__{file_suffix}" if file_suffix else ""
    return write_json_artifact(layout.metrics_dir / f"{variant.value}_{stage.value}{suffix}.json", artifact)


def _metric_artifact_path(
    layout: RunLayout,
    variant: Variant,
    stage: Stage,
    *,
    file_suffix: str | None = None,
) -> Path:
    suffix = f"__{file_suffix}" if file_suffix else ""
    return layout.metrics_dir / f"{variant.value}_{stage.value}{suffix}.json"


def _common_fields_for_variant(common: dict[str, Any], variant: Variant) -> dict[str, Any]:
    if variant == Variant.kf_cast:
        return common
    normalized = dict(common)
    normalized["cast_package_path"] = None
    normalized["cast_package_hash"] = None
    normalized["kf_artifact_path"] = None
    normalized["kf_artifact_hash"] = None
    normalized["kf_artifact_kind"] = None
    normalized["exported_kernel_hashes"] = {}
    normalized["kf_settings"] = {}
    normalized["toolchain_status"] = {}
    return normalized


def _device_audit_artifact_path(
    layout: RunLayout,
    variant: Variant,
    stage: Stage,
    *,
    file_suffix: str | None = None,
) -> Path:
    suffix = f"__{file_suffix}" if file_suffix else ""
    audit_dir = layout.run_dir / "device_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / f"{variant.value}_{stage.value}{suffix}.json"


def _device_audit_per_op(runtime_stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = runtime_stats or {}
    per_op = stats.get("per_op")
    if not isinstance(per_op, dict):
        return {}
    out: dict[str, Any] = {}
    for op_name, raw_op_stats in per_op.items():
        if not isinstance(raw_op_stats, dict):
            continue
        input_devices = raw_op_stats.get("input_devices") if isinstance(raw_op_stats.get("input_devices"), dict) else {}
        input_dtypes = raw_op_stats.get("input_dtypes") if isinstance(raw_op_stats.get("input_dtypes"), dict) else {}
        input_is_cuda = raw_op_stats.get("input_is_cuda") if isinstance(raw_op_stats.get("input_is_cuda"), dict) else {}
        fallback_reasons = raw_op_stats.get("fallback_reasons") if isinstance(raw_op_stats.get("fallback_reasons"), dict) else {}
        out[str(op_name)] = {
            "patched_calls": int(raw_op_stats.get("patched_calls", 0) or 0),
            "kernel_launches_attempted": int(raw_op_stats.get("kernel_launches_attempted", 0) or 0),
            "kernel_launches_succeeded": int(raw_op_stats.get("kernel_launches_succeeded", 0) or 0),
            "kernel_launches_failed": int(raw_op_stats.get("kernel_launches_failed", 0) or 0),
            "fallbacks_to_original": int(raw_op_stats.get("fallbacks_to_original", 0) or 0),
            "fallback_reasons": {str(key): int(value) for key, value in fallback_reasons.items()},
            "input_devices": {str(key): int(value) for key, value in input_devices.items()},
            "input_dtypes": {str(key): int(value) for key, value in input_dtypes.items()},
            "input_is_cuda": {str(key): int(value) for key, value in input_is_cuda.items()},
            "last_exception": raw_op_stats.get("last_exception"),
        }
    return out


def _device_audit_issues(
    *,
    selected_ops: Sequence[str],
    per_op_launch_coverage: dict[str, Any],
    require_runtime_coverage: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for op_name in selected_ops:
        stats = per_op_launch_coverage.get(op_name)
        if not isinstance(stats, dict):
            if require_runtime_coverage:
                errors.append(f"selected op {op_name} has no runtime coverage")
            else:
                warnings.append(f"selected op {op_name} has no runtime coverage yet")
            continue
        input_devices = stats.get("input_devices") if isinstance(stats.get("input_devices"), dict) else {}
        input_is_cuda = stats.get("input_is_cuda") if isinstance(stats.get("input_is_cuda"), dict) else {}
        bad_devices = [
            str(device)
            for device in input_devices
            if str(device).startswith("cpu") or str(device).startswith("meta")
        ]
        if bad_devices:
            errors.append(f"selected op {op_name} saw non-CUDA devices: {', '.join(sorted(bad_devices))}")
        if int(input_is_cuda.get("false", 0) or 0) > 0:
            errors.append(f"selected op {op_name} saw non-CUDA tensor inputs")
        if require_runtime_coverage:
            if int(stats.get("patched_calls", 0) or 0) <= 0:
                errors.append(f"selected op {op_name} was not patched during audited stage")
            if int(stats.get("kernel_launches_succeeded", 0) or 0) <= 0:
                errors.append(f"selected op {op_name} had zero successful kernel launches during audited stage")
            if int(stats.get("fallbacks_to_original", 0) or 0) > 0:
                errors.append(f"selected op {op_name} fell back to original implementation")
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def _write_device_audit_artifact(
    layout: RunLayout,
    common: dict[str, Any],
    *,
    variant: Variant,
    stage: Stage,
    runtime_meta: dict[str, Any],
    runtime_stats: dict[str, Any] | None,
    runtime_input_device: str,
    tokenizer_output_devices: dict[str, str],
    require_runtime_coverage: bool,
    file_suffix: str | None = None,
) -> Path:
    selected_ops = [str(item) for item in (runtime_meta.get("selected_ops") or [])]
    per_op_launch_coverage = _device_audit_per_op(runtime_stats)
    audit_errors, audit_warnings = _device_audit_issues(
        selected_ops=selected_ops,
        per_op_launch_coverage=per_op_launch_coverage,
        require_runtime_coverage=require_runtime_coverage,
    )
    stats = runtime_stats or {}
    artifact_common = dict(common)
    for key in (
        "artifact_type",
        "variant",
        "stage",
        "warmup_count",
        "timed_run_count",
        "audit_stage",
        "audit_status",
        "audit_errors",
        "audit_warnings",
        "selected_ops",
        "runtime_input_device",
        "tokenizer_output_devices",
        "placement_audit",
        "per_op_launch_coverage",
        "fallback_reasons_by_op",
        "kernel_launches_attempted",
        "kernel_launches_succeeded",
        "kernel_launches_failed",
        "fallback_count",
    ):
        artifact_common.pop(key, None)
    audit = DeviceAuditArtifact(
        **artifact_common,
        artifact_type="device_audit",
        variant=variant,
        stage=stage,
        warmup_count=int(common.get("warmup_count", 0) or 0),
        timed_run_count=int(common.get("timed_run_count", 0) or 0),
        audit_stage=stage.value,
        audit_status="failed" if audit_errors else "passed" if require_runtime_coverage else "pending_runtime",
        audit_errors=audit_errors,
        audit_warnings=audit_warnings,
        selected_ops=selected_ops,
        runtime_input_device=str(runtime_input_device),
        tokenizer_output_devices=dict(tokenizer_output_devices),
        placement_audit=dict(runtime_meta.get("placement_audit") or {}),
        per_op_launch_coverage=per_op_launch_coverage,
        fallback_reasons_by_op={
            op_name: dict(stats.get("fallback_reasons") or {})
            for op_name, stats in per_op_launch_coverage.items()
            if isinstance(stats, dict)
        },
        kernel_launches_attempted=int(stats.get("kernel_launches_attempted", runtime_meta.get("kernel_launches_attempted", 0)) or 0),
        kernel_launches_succeeded=int(stats.get("kernel_launches_succeeded", runtime_meta.get("kernel_hit_count", 0)) or 0),
        kernel_launches_failed=int(stats.get("kernel_launches_failed", runtime_meta.get("kernel_launches_failed", 0)) or 0),
        fallback_count=int(stats.get("fallbacks_to_original", runtime_meta.get("fallback_count", 0)) or 0),
    )
    return write_json_artifact(
        _device_audit_artifact_path(layout, variant, stage, file_suffix=file_suffix),
        audit,
    )


def _partial_prompt_sample_matrix(
    tokenizer,
    prompt_batches: Sequence[Sequence[dict[str, Any]]],
    device: str,
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for batch in prompt_batches:
        tokenized, _ = _tokenize_prompt_batch(tokenizer, batch, device, measure_timing=False)
        attention_mask = tokenized["attention_mask"]
        prompt_lengths = [int(item) for item in attention_mask.sum(dim=1).detach().cpu().tolist()]
        matrix.append(
            {
                "prompt_ids": [record["id"] for record in batch],
                "batch_size": len(batch),
                "prompt_lengths": prompt_lengths,
            }
        )
    return matrix


def _load_stage_sample_record(
    load_ms: float,
    *,
    load_source: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_index": 0,
        "stage": Stage.load.value,
        "latency_ms": float(load_ms),
        "load_source": load_source,
        "prompt_length_buckets": details.get("prompt_length_buckets", []),
        "generation_settings": details.get("generation_settings"),
    }


def _compile_failure_sample_record(
    compile_time_ms: float,
    *,
    execution_status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_index": 0,
        "stage": Stage.compile.value,
        "latency_ms": float(compile_time_ms),
        "execution_status": execution_status,
        "error_type": error_type,
        "error_message": error_message,
    }


def _format_failure_message(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {type(exc).__name__}: {exc}"


def _write_failed_generation_artifact(
    layout: RunLayout,
    common: dict[str, Any],
    *,
    variant: Variant,
    stage: Stage,
    generation_config: GenerationConfig,
    slice_details: dict[str, Any],
    compile_time_ms: float | None,
    fallback_count: int | None,
    kernel_hit_count: int | None,
    sample_index: int,
    prompt_ids: Sequence[str],
    prompt_lengths: Sequence[int],
    batch_size: int,
    tokenization_ms: float,
    latency_ms: float,
    input_batch_hash: str | None,
    exc: Exception,
    sample_records: list[dict[str, Any]] | None = None,
    samples_ms: list[float] | None = None,
    comparison_group: str | None = None,
    configured_batch_size: int | None = None,
    prompt_bucket_id: str | None = None,
    file_suffix: str | None = None,
) -> None:
    failed_sample_records = list(sample_records or [])
    failed_latency_samples = list(samples_ms or [])
    failed_sample_records.append(
        _build_failure_sample_record(
            sample_index,
            stage=stage,
            prompt_ids=prompt_ids,
            batch_size=batch_size,
            prompt_lengths=prompt_lengths,
            input_batch_hash=input_batch_hash,
            tokenization_ms=tokenization_ms,
            latency_ms=latency_ms,
            exc=exc,
        )
    )
    failed_latency_samples.append(float(latency_ms))
    _write_stage_artifact(
        layout,
        common,
        variant=variant,
        stage=stage,
        samples_ms=failed_latency_samples,
        warmup_count=int(common.get("warmup_count", 0) or 0),
        timed_run_count=len(failed_latency_samples),
        correctness_status=CorrectnessStatus.failed,
        correctness_message=_format_failure_message(f"{variant.value} {stage.value} failed", exc),
        compile_time_ms=compile_time_ms,
        steady_state_time_ms=sum(failed_latency_samples) / len(failed_latency_samples),
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        sample_records=failed_sample_records,
        details={
            "generation_settings": _generation_settings_payload(generation_config),
            **slice_details,
            "execution_status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failed_sample_index": int(sample_index),
            "failed_prompt_ids": list(prompt_ids),
        },
        token_count=int(sum(int(record.get("generated_token_count", 0) or 0) for record in failed_sample_records)),
        comparison_group=comparison_group,
        configured_batch_size=configured_batch_size,
        prompt_bucket_id=prompt_bucket_id,
        file_suffix=file_suffix,
    )


def run_llm_benchmark(
    *,
    layout: RunLayout,
    common_fields: dict[str, Any],
    env_artifact: EnvironmentArtifact,
    manifest_artifact: RunManifestArtifact,
    model_spec,
    suite,
    variant: Variant,
    model_loader: Callable[..., tuple[Any, Any, float]] = load_transformers_causal_lm,
    compile_model_fn: Callable[..., tuple[Any, float]] = compile_model,
    cast_loader: Callable[..., tuple[Any, dict[str, Any]]] = load_cast_model,
    store_prompts: bool = False,
    reuse_cache: bool = False,
    cache_search_root: str | Path | None = None,
) -> RunLayout:
    prompt_suite = load_prompt_records(suite.workload_path)
    compile_settings = compile_settings_from_dict(common_fields.get("compile_settings"))
    kf_settings = runtime_settings_from_dict(common_fields.get("kf_settings"))
    common = {
        **common_fields,
        "compile_settings": compile_settings.as_dict(),
        "kf_settings": kf_settings.as_dict(),
    }
    common = _common_fields_for_variant(common, variant)
    if prompt_suite.synthetic_workload and not common.get("synthetic_workload", False):
        issues = list(common.get("paper_eligibility_issues", []) or [])
        issues.append("synthetic workload used")
        common["synthetic_workload"] = True
        common["paper_eligibility_issues"] = list(dict.fromkeys(issues))
        common["paper_eligible"] = False
        manifest_artifact = manifest_artifact.model_copy(
            update={
                "synthetic_workload": True,
                "paper_eligibility_issues": common["paper_eligibility_issues"],
                "paper_eligible": False,
            }
        )
        env_artifact = env_artifact.model_copy(
            update={
                "synthetic_workload": True,
                "paper_eligibility_issues": common["paper_eligibility_issues"],
                "paper_eligible": False,
            }
        )

    write_json_artifact(layout.run_dir / "manifest.json", manifest_artifact)
    write_json_artifact(layout.run_dir / "env.json", env_artifact)

    device = _resolve_suite_device(suite, model_spec)

    compile_time_ms: float | None = None
    fallback_count: int | None = None
    kernel_hit_count: int | None = None
    reference_model = None
    compile_wrap_ms: float | None = None
    runtime_meta: dict[str, Any] | None = None

    if variant == Variant.kf_cast:
        if not (kf_settings.cast_package_path or model_spec.cast_package_path):
            raise ValueError("kf_cast benchmarking requires a cast package path.")
        model, runtime_meta = cast_loader(
            kf_settings.cast_package_path or model_spec.cast_package_path,
            device=device,
            settings=kf_settings,
        )
        reference_model, tokenizer, _ = model_loader(model_spec, device=device)
        load_ms = float(runtime_meta.get("load_time_ms", 0.0))
        compile_time_ms = (
            float(runtime_meta.get("jit_compile_time_ms", 0.0))
            if runtime_meta.get("jit_compile_time_ms") is not None
            else None
        )
        if kf_settings.record_runtime_stats:
            fallback_count = int(runtime_meta.get("fallback_count", 0))
            kernel_hit_count = int(runtime_meta.get("kernel_hit_count", 0))
        else:
            fallback_count = None
            kernel_hit_count = None
        common["cast_package_path"] = runtime_meta.get("cast_package_path", common.get("cast_package_path"))
        common["cast_package_hash"] = runtime_meta.get("cast_package_hash", common.get("cast_package_hash"))
        kernel_hashes = runtime_meta.get("kernel_source_hashes") or {}
        if kernel_hashes:
            merged_kernel_hashes = dict(common.get("exported_kernel_hashes") or {})
            merged_kernel_hashes.update(kernel_hashes)
            common["exported_kernel_hashes"] = dict(sorted(merged_kernel_hashes.items()))
        kf_common_settings = dict(common.get("kf_settings") or {})
        if runtime_meta.get("selected_source_hashes"):
            kf_common_settings["selected_source_hashes"] = dict(runtime_meta.get("selected_source_hashes") or {})
        cast_manifest = runtime_meta.get("cast_manifest")
        if isinstance(cast_manifest, dict) and cast_manifest.get("project_ref") and not kf_common_settings.get("project_ref"):
            kf_common_settings["project_ref"] = str(cast_manifest.get("project_ref"))
        common["kf_settings"] = kf_common_settings
        reference_status, reference_message = skipped_correctness(
            "Reference correctness is computed against eager outputs for every timed run."
        )
    else:
        model, tokenizer, load_ms = model_loader(model_spec, device=device)
        reference_status, reference_message = reference_correctness() if variant == Variant.eager else skipped_correctness(
            "Reference correctness is computed against eager outputs for every timed run."
        )
        if variant == Variant.torch_compile:
            reference_model = model

    runtime_input_device = _resolve_input_device(model, device)
    workload_slices = _select_prompt_workload_slices(prompt_suite, tokenizer, suite)
    use_group_suffix = len(workload_slices) > 1
    load_generation_config = _resolve_generation_config(
        tokenizer,
        model,
        suite,
        batch_size=int(workload_slices[0].batch_size),
    )
    load_details = {
        "load_source": "cast_package" if variant == Variant.kf_cast else "transformers",
        "generation_settings": {
            **_generation_settings_payload(load_generation_config),
            "batch_sizes": _suite_batch_sizes(suite),
        },
        "prompt_source_format": prompt_suite.source_format,
        "prompt_length_buckets": [slice_data.prompt_bucket_id for slice_data in workload_slices],
        "placement_profile": common.get("placement_profile"),
        "runtime_input_device": runtime_input_device,
    }
    if runtime_meta is not None:
        load_details.update(build_kf_runtime_details(runtime_meta))
    first_slice = workload_slices[0]
    timed_batches = first_slice.prompt_batches[: int(suite.timed_run_count)]
    first_generation_config = _resolve_generation_config(
        tokenizer,
        model,
        suite,
        batch_size=int(first_slice.batch_size),
    )
    audit_probe_tokenized, _ = _tokenize_prompt_batch(
        tokenizer,
        timed_batches[0],
        runtime_input_device,
        measure_timing=False,
    )
    tokenizer_output_devices = _tokenizer_output_devices(audit_probe_tokenized)
    load_stage_records = [
        _load_stage_sample_record(
            float(load_ms),
            load_source=str(load_details["load_source"]),
            details=load_details,
        )
    ]

    cache_root = Path(cache_search_root) if cache_search_root else layout.run_dir.parent
    load_request = CacheRequest(
        signature=make_cache_request(
            common,
            variant=variant,
            stage=Stage.load,
            details=load_details,
            sample_matrix=[{"sample_index": 0, "load_source": load_details["load_source"]}],
            warmup_count=0,
            timed_run_count=1,
        ),
        target_path=_metric_artifact_path(layout, variant, Stage.load),
    )
    cache_requests: list[CacheRequest] = [load_request]
    torch_compile_request: CacheRequest | None = None
    if variant == Variant.torch_compile:
        compile_probe_batch = timed_batches[0]
        torch_compile_request = CacheRequest(
            signature=make_cache_request(
                common,
                variant=variant,
                stage=Stage.compile,
                sample_matrix=_partial_prompt_sample_matrix(
                    tokenizer,
                    [compile_probe_batch],
                    runtime_input_device,
                ),
                warmup_count=0,
                timed_run_count=1,
            ),
            target_path=_metric_artifact_path(layout, variant, Stage.compile),
        )
        cache_requests.append(torch_compile_request)
    elif variant == Variant.kf_cast and compile_time_ms is not None and runtime_meta is not None:
        cache_requests.append(
            CacheRequest(
                signature=make_cache_request(
                    common,
                    variant=variant,
                    stage=Stage.compile,
                    details=build_kf_runtime_details(runtime_meta),
                    sample_matrix=[{"prompt_ids": [], "batch_size": None, "prompt_lengths": []}],
                    warmup_count=0,
                    timed_run_count=1,
                ),
                target_path=_metric_artifact_path(layout, variant, Stage.compile),
            )
        )

    for workload_slice in workload_slices:
        generation_config = _resolve_generation_config(
            tokenizer,
            model,
            suite,
            batch_size=int(workload_slice.batch_size),
        )
        slice_details = {
            "comparison_group": workload_slice.comparison_group,
            "prompt_bucket_id": workload_slice.prompt_bucket_id,
            "selected_prompt_ids": workload_slice.selected_prompt_ids,
            "selected_prompt_ids_hash": _hash_json_payload(workload_slice.selected_prompt_ids),
            "prompt_suite_hash": common.get("workload_hash"),
            "prompt_source_format": prompt_suite.source_format,
            "placement_profile": common.get("placement_profile"),
            "runtime_input_device": runtime_input_device,
            "selection_policy": {
                "method": "frozen_input_order",
                "post_hoc": False,
                "prompt_bucket_id": workload_slice.prompt_bucket_id,
                "batch_size": int(workload_slice.batch_size),
                "timed_run_count": int(suite.timed_run_count),
            },
        }
        file_suffix = workload_slice.comparison_group if use_group_suffix else None
        timed_prompt_batches = workload_slice.prompt_batches[: int(suite.timed_run_count)]
        timed_matrix = _partial_prompt_sample_matrix(tokenizer, timed_prompt_batches, runtime_input_device)
        stage_details = {
            "generation_settings": _generation_settings_payload(generation_config),
            **slice_details,
        }
        if int(suite.warmup_count) > 0:
            warmup_batches = [
                workload_slice.prompt_batches[index % len(workload_slice.prompt_batches)]
                for index in range(int(suite.warmup_count))
            ]
            cache_requests.append(
                CacheRequest(
                    signature=make_cache_request(
                        common,
                        variant=variant,
                        stage=Stage.warmup,
                        details=stage_details,
                        configured_batch_size=int(workload_slice.batch_size),
                        prompt_bucket_id=workload_slice.prompt_bucket_id,
                        comparison_group=workload_slice.comparison_group,
                        sample_matrix=_partial_prompt_sample_matrix(tokenizer, warmup_batches, runtime_input_device),
                        warmup_count=int(suite.warmup_count),
                        timed_run_count=len(warmup_batches),
                    ),
                    target_path=_metric_artifact_path(layout, variant, Stage.warmup, file_suffix=file_suffix),
                )
            )
        if _measure_prefill_decode_separately(suite):
            cache_requests.append(
                CacheRequest(
                    signature=make_cache_request(
                        common,
                        variant=variant,
                        stage=Stage.prefill,
                        details=stage_details,
                        configured_batch_size=int(workload_slice.batch_size),
                        prompt_bucket_id=workload_slice.prompt_bucket_id,
                        comparison_group=workload_slice.comparison_group,
                        sample_matrix=timed_matrix,
                        warmup_count=int(suite.warmup_count),
                        timed_run_count=len(timed_prompt_batches),
                    ),
                    target_path=_metric_artifact_path(layout, variant, Stage.prefill, file_suffix=file_suffix),
                )
            )
            cache_requests.append(
                CacheRequest(
                    signature=make_cache_request(
                        common,
                        variant=variant,
                        stage=Stage.decode,
                        details=stage_details,
                        configured_batch_size=int(workload_slice.batch_size),
                        prompt_bucket_id=workload_slice.prompt_bucket_id,
                        comparison_group=workload_slice.comparison_group,
                        sample_matrix=timed_matrix,
                        warmup_count=int(suite.warmup_count),
                        timed_run_count=len(timed_prompt_batches),
                    ),
                    target_path=_metric_artifact_path(layout, variant, Stage.decode, file_suffix=file_suffix),
                )
            )
        cache_requests.append(
            CacheRequest(
                signature=make_cache_request(
                    common,
                    variant=variant,
                    stage=Stage.total_generate,
                    details=stage_details,
                    configured_batch_size=int(workload_slice.batch_size),
                    prompt_bucket_id=workload_slice.prompt_bucket_id,
                    comparison_group=workload_slice.comparison_group,
                    sample_matrix=timed_matrix,
                    warmup_count=int(suite.warmup_count),
                    timed_run_count=len(timed_prompt_batches),
                ),
                target_path=_metric_artifact_path(layout, variant, Stage.total_generate, file_suffix=file_suffix),
            )
        )

    if reuse_cache:
        if variant == Variant.torch_compile and torch_compile_request is not None:
            compile_match = find_matching_reusable_artifact(
                cache_root,
                torch_compile_request.signature,
                exclude_run_dir=layout.run_dir,
            )
            if compile_match is not None:
                _, compile_artifact = compile_match
                compile_failed = (
                    compile_artifact.correctness_status == CorrectnessStatus.failed
                    or str((compile_artifact.details or {}).get("execution_status", "")).lower() == "failed"
                )
                reusable_requests = cache_requests[:2] if compile_failed else cache_requests
                if copy_reused_artifact_set(
                    reusable_requests,
                    search_root=cache_root,
                    run_id=common["run_id"],
                    timestamp_utc=common["timestamp_utc"],
                    exclude_run_dir=layout.run_dir,
                ):
                    return layout
        elif copy_reused_artifact_set(
            cache_requests,
            search_root=cache_root,
            run_id=common["run_id"],
            timestamp_utc=common["timestamp_utc"],
            exclude_run_dir=layout.run_dir,
        ):
            return layout

    if variant == Variant.kf_cast and runtime_meta is not None:
        load_audit_path = _write_device_audit_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.load,
            runtime_meta=runtime_meta,
            runtime_stats=runtime_meta.get("runtime_stats") if isinstance(runtime_meta.get("runtime_stats"), dict) else None,
            runtime_input_device=runtime_input_device,
            tokenizer_output_devices=tokenizer_output_devices,
            require_runtime_coverage=False,
        )
        load_details["device_audit_artifact_path"] = str(load_audit_path.resolve())

    _write_stage_artifact(
        layout,
        common,
        variant=variant,
        stage=Stage.load,
        samples_ms=[float(load_ms)],
        warmup_count=0,
        timed_run_count=1,
        correctness_status=reference_status,
        correctness_message=reference_message,
        compile_time_ms=None,
        steady_state_time_ms=float(load_ms),
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        sample_records=load_stage_records,
        details=load_details,
    )

    if variant == Variant.kf_cast and compile_time_ms is not None and runtime_meta is not None:
        compile_details = build_kf_runtime_details(runtime_meta)
        compile_audit_path = _write_device_audit_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.compile,
            runtime_meta=runtime_meta,
            runtime_stats=runtime_meta.get("runtime_stats") if isinstance(runtime_meta.get("runtime_stats"), dict) else None,
            runtime_input_device=runtime_input_device,
            tokenizer_output_devices=tokenizer_output_devices,
            require_runtime_coverage=False,
        )
        compile_details["device_audit_artifact_path"] = str(compile_audit_path.resolve())
        _write_stage_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.compile,
            samples_ms=[float(compile_time_ms)],
            warmup_count=0,
            timed_run_count=1,
            correctness_status=CorrectnessStatus.not_applicable,
            correctness_message="Kernel Forge deployment setup/JIT stage.",
            compile_time_ms=float(compile_time_ms),
            steady_state_time_ms=None,
            fallback_count=int(runtime_meta.get("fallback_count", 0)),
            kernel_hit_count=int(runtime_meta.get("kernel_hit_count", 0)),
            sample_records=[
                {
                    "sample_index": 0,
                    "stage": Stage.compile.value,
                    "latency_ms": float(compile_time_ms),
                    "execution_status": "ok",
                }
            ],
            details=compile_details,
        )

    if variant == Variant.torch_compile:
        use_kv_cache = _suite_uses_kv_cache(suite)
        compile_wrap_start = time.perf_counter()
        try:
            model, compile_wrap_ms = compile_model_fn(model, compile_settings)
        except Exception as exc:
            compile_wrap_ms = (time.perf_counter() - compile_wrap_start) * 1000.0
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.compile,
                samples_ms=[float(compile_wrap_ms)],
                warmup_count=0,
                timed_run_count=1,
                correctness_status=CorrectnessStatus.failed,
                correctness_message=_format_failure_message("torch.compile wrap failed", exc),
                compile_time_ms=float(compile_wrap_ms),
                steady_state_time_ms=None,
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                sample_records=[
                    _compile_failure_sample_record(
                        float(compile_wrap_ms),
                        execution_status="failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                ],
                details={
                    "execution_status": "failed",
                    "generation_settings": _generation_settings_payload(first_generation_config),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "compile_wrap_time_ms": float(compile_wrap_ms),
                    "compile_first_run_time_ms": None,
                },
            )
            return layout

        compile_probe_batch = timed_batches[0]
        compile_probe_tokenized, compile_probe_tokenization_ms = _tokenize_prompt_batch(
            tokenizer,
            compile_probe_batch,
            runtime_input_device,
            measure_timing=_include_tokenization_in_timing(suite),
        )
        compile_probe_start = time.perf_counter()
        try:
            compile_probe_result = _execute_greedy_generation(
                model,
                compile_probe_tokenized,
                first_generation_config,
                runtime_input_device,
                measure_timings=True,
                tokenization_ms=compile_probe_tokenization_ms,
                include_tokenization_in_timing=_include_tokenization_in_timing(suite),
                use_kv_cache=use_kv_cache,
            )
            compile_probe_result.prompt_ids = [record["id"] for record in compile_probe_batch]
            compile_first_run_ms = float(compile_probe_result.total_ms)
            compile_time_ms = float((compile_wrap_ms or 0.0) + compile_first_run_ms)
            compile_correctness_status = CorrectnessStatus.passed
            compile_correctness_message = None
            if reference_model is not None:
                reference_probe_result = _execute_greedy_generation(
                    reference_model,
                    compile_probe_tokenized,
                    first_generation_config,
                    runtime_input_device,
                    measure_timings=False,
                    tokenization_ms=compile_probe_tokenization_ms,
                    include_tokenization_in_timing=_include_tokenization_in_timing(suite),
                    use_kv_cache=use_kv_cache,
                )
                if reference_probe_result.output_token_hashes != compile_probe_result.output_token_hashes:
                    compile_correctness_status = CorrectnessStatus.failed
                    compile_correctness_message = "torch.compile first-run output hash mismatch against eager reference."
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.compile,
                samples_ms=[compile_time_ms],
                warmup_count=0,
                timed_run_count=1,
                correctness_status=compile_correctness_status,
                correctness_message=compile_correctness_message,
                compile_time_ms=compile_time_ms,
                steady_state_time_ms=None,
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                sample_records=[
                    _build_stage_sample_record(
                        0,
                        compile_probe_result,
                        stage=Stage.total_generate,
                        prompt_ids=compile_probe_result.prompt_ids,
                    )
                ],
                details={
                    "execution_status": "ok",
                    "generation_settings": _generation_settings_payload(first_generation_config),
                    "compile_wrap_time_ms": float(compile_wrap_ms or 0.0),
                    "compile_first_run_time_ms": compile_first_run_ms,
                    "compile_probe_prompt_ids": compile_probe_result.prompt_ids,
                    "compile_probe_batch_output_hash": compile_probe_result.batch_output_hash,
                },
                token_count=compile_probe_result.generated_token_count,
            )
        except Exception as exc:
            sync_device(device)
            compile_first_run_ms = float((time.perf_counter() - compile_probe_start) * 1000.0)
            compile_time_ms = float((compile_wrap_ms or 0.0) + compile_first_run_ms)
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.compile,
                samples_ms=[compile_time_ms],
                warmup_count=0,
                timed_run_count=1,
                correctness_status=CorrectnessStatus.failed,
                correctness_message=_format_failure_message("torch.compile first run failed", exc),
                compile_time_ms=compile_time_ms,
                steady_state_time_ms=None,
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                sample_records=[
                    _compile_failure_sample_record(
                        float(compile_time_ms),
                        execution_status="failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                ],
                details={
                    "execution_status": "failed",
                    "generation_settings": _generation_settings_payload(first_generation_config),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "compile_wrap_time_ms": float(compile_wrap_ms or 0.0),
                    "compile_first_run_time_ms": compile_first_run_ms,
                },
            )
            return layout

    for workload_slice in workload_slices:
        generation_config = _resolve_generation_config(
            tokenizer,
            model,
            suite,
            batch_size=int(workload_slice.batch_size),
        )
        slice_details = {
            "comparison_group": workload_slice.comparison_group,
            "prompt_bucket_id": workload_slice.prompt_bucket_id,
            "selected_prompt_ids": workload_slice.selected_prompt_ids,
            "selected_prompt_ids_hash": _hash_json_payload(workload_slice.selected_prompt_ids),
            "prompt_suite_hash": common.get("workload_hash"),
            "prompt_source_format": prompt_suite.source_format,
            "selection_policy": {
                "method": "frozen_input_order",
                "post_hoc": False,
                "prompt_bucket_id": workload_slice.prompt_bucket_id,
                "batch_size": int(workload_slice.batch_size),
                "timed_run_count": int(suite.timed_run_count),
            },
        }
        file_suffix = workload_slice.comparison_group if use_group_suffix else None
        timed_batches = workload_slice.prompt_batches[: int(suite.timed_run_count)]
        use_kv_cache = _suite_uses_kv_cache(suite)

        if variant == Variant.kf_cast and runtime_meta is not None and kf_settings.record_runtime_stats:
            reset_cast_runtime_stats(model)

        warmup_samples: list[float] = []
        warmup_records: list[dict[str, Any]] = []
        warmup_batches = [
            workload_slice.prompt_batches[index % len(workload_slice.prompt_batches)]
            for index in range(int(suite.warmup_count))
        ]
        for sample_index, batch in enumerate(warmup_batches):
            prompt_ids = [record["id"] for record in batch]
            tokenized = None
            tokenization_ms = 0.0
            failure_start = time.perf_counter()
            try:
                tokenized, tokenization_ms = _tokenize_prompt_batch(
                    tokenizer,
                    batch,
                    runtime_input_device,
                    measure_timing=_include_tokenization_in_timing(suite),
                )
                warmup_result = _execute_greedy_generation(
                    model,
                    tokenized,
                    generation_config,
                    runtime_input_device,
                    measure_timings=True,
                    tokenization_ms=tokenization_ms,
                    include_tokenization_in_timing=_include_tokenization_in_timing(suite),
                    use_kv_cache=use_kv_cache,
                )
            except Exception as exc:
                sync_device(runtime_input_device)
                failure_latency_ms = float((time.perf_counter() - failure_start) * 1000.0)
                prompt_lengths: list[int] = []
                input_batch_hash = None
                batch_size = len(batch)
                if tokenized is not None:
                    batch_size, prompt_lengths, _, input_batch_hash = _summarize_tokenized_batch(tokenized)
                _write_failed_generation_artifact(
                    layout,
                    common,
                    variant=variant,
                    stage=Stage.warmup,
                    generation_config=generation_config,
                    slice_details=slice_details,
                    compile_time_ms=compile_time_ms,
                    fallback_count=fallback_count,
                    kernel_hit_count=kernel_hit_count,
                    sample_index=sample_index,
                    prompt_ids=prompt_ids,
                    prompt_lengths=prompt_lengths,
                    batch_size=batch_size,
                    tokenization_ms=tokenization_ms,
                    latency_ms=(time.perf_counter() - failure_start) * 1000.0,
                    input_batch_hash=input_batch_hash,
                    exc=exc,
                    sample_records=warmup_records,
                    samples_ms=warmup_samples,
                    comparison_group=workload_slice.comparison_group,
                    configured_batch_size=int(workload_slice.batch_size),
                    prompt_bucket_id=workload_slice.prompt_bucket_id,
                    file_suffix=file_suffix,
                )
                return layout
            warmup_result.prompt_ids = prompt_ids
            warmup_samples.append(float(warmup_result.total_ms))
            warmup_records.append(
                _build_stage_sample_record(
                    sample_index,
                    warmup_result,
                    stage=Stage.total_generate,
                    prompt_ids=warmup_result.prompt_ids,
                )
            )

        warmup_fallback_count = fallback_count
        warmup_kernel_hit_count = kernel_hit_count
        warmup_details: dict[str, Any] | None = None
        if variant == Variant.kf_cast and runtime_meta is not None:
            warmup_stats = get_cast_runtime_stats(model) if kf_settings.record_runtime_stats else runtime_meta.get("runtime_stats", {})
            if kf_settings.record_runtime_stats:
                warmup_fallback_count = int(warmup_stats.get("fallbacks_to_original", runtime_meta.get("fallback_count", 0)) or 0)
                warmup_kernel_hit_count = int(warmup_stats.get("kernel_launches_succeeded", runtime_meta.get("kernel_hit_count", 0)) or 0)
            else:
                warmup_fallback_count = None
                warmup_kernel_hit_count = None
            warmup_details = build_kf_runtime_details(runtime_meta, warmup_stats)
            warmup_audit_path = _write_device_audit_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.warmup,
                runtime_meta=runtime_meta,
                runtime_stats=warmup_stats if isinstance(warmup_stats, dict) else None,
                runtime_input_device=runtime_input_device,
                tokenizer_output_devices=tokenizer_output_devices,
                require_runtime_coverage=True,
                file_suffix=file_suffix,
            )
            warmup_details["device_audit_artifact_path"] = str(warmup_audit_path.resolve())
            if kf_settings.record_runtime_stats:
                reset_cast_runtime_stats(model)

        if warmup_samples:
            warmup_stage_details = {
                "generation_settings": _generation_settings_payload(generation_config),
                **slice_details,
            }
            if warmup_details is not None:
                warmup_stage_details.update(warmup_details)
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.warmup,
                samples_ms=warmup_samples,
                warmup_count=int(suite.warmup_count),
                timed_run_count=len(warmup_samples),
                correctness_status=reference_status,
                correctness_message=reference_message,
                compile_time_ms=compile_time_ms,
                steady_state_time_ms=sum(warmup_samples) / len(warmup_samples),
                fallback_count=warmup_fallback_count,
                kernel_hit_count=warmup_kernel_hit_count,
                sample_records=warmup_records,
                details=warmup_stage_details,
                comparison_group=workload_slice.comparison_group,
                configured_batch_size=int(workload_slice.batch_size),
                prompt_bucket_id=workload_slice.prompt_bucket_id,
                file_suffix=file_suffix,
            )

        prefill_samples: list[float] = []
        decode_samples: list[float] = []
        total_samples: list[float] = []
        prefill_records: list[dict[str, Any]] = []
        decode_records: list[dict[str, Any]] = []
        total_records: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        correctness_failures: list[str] = []

        for sample_index, batch in enumerate(timed_batches):
            prompt_ids = [record["id"] for record in batch]
            tokenized = None
            tokenization_ms = 0.0
            reference_result = None
            failure_start = time.perf_counter()
            try:
                tokenized, tokenization_ms = _tokenize_prompt_batch(
                    tokenizer,
                    batch,
                    runtime_input_device,
                    measure_timing=_include_tokenization_in_timing(suite),
                )
                candidate_result = _execute_greedy_generation(
                    model,
                    tokenized,
                    generation_config,
                    runtime_input_device,
                    measure_timings=True,
                    tokenization_ms=tokenization_ms,
                    include_tokenization_in_timing=_include_tokenization_in_timing(suite),
                    use_kv_cache=use_kv_cache,
                )
                candidate_result.prompt_ids = prompt_ids

                if variant in {Variant.torch_compile, Variant.kf_cast} and reference_model is not None:
                    reference_result = _execute_greedy_generation(
                        reference_model,
                        tokenized,
                        generation_config,
                        runtime_input_device,
                        measure_timings=False,
                        tokenization_ms=tokenization_ms,
                        include_tokenization_in_timing=_include_tokenization_in_timing(suite),
                        use_kv_cache=use_kv_cache,
                    )
                    if reference_result.output_token_hashes != candidate_result.output_token_hashes:
                        correctness_failures.append(
                            f"timed run {sample_index} output hash mismatch for prompts {prompt_ids}"
                        )
            except Exception as exc:
                sync_device(runtime_input_device)
                failure_latency_ms = float((time.perf_counter() - failure_start) * 1000.0)
                prompt_lengths: list[int] = []
                input_batch_hash = None
                batch_size = len(batch)
                if tokenized is not None:
                    batch_size, prompt_lengths, _, input_batch_hash = _summarize_tokenized_batch(tokenized)
                raw_rows.append(
                    {
                        "sample_index": sample_index,
                        "comparison_group": workload_slice.comparison_group,
                        "prompt_bucket_id": workload_slice.prompt_bucket_id,
                        "batch_size": int(workload_slice.batch_size),
                        "prompt_ids": prompt_ids,
                        "input_batch_hash": input_batch_hash,
                        "prompt_lengths": prompt_lengths,
                        "generated_lengths": [],
                        "tokenization_ms": float(tokenization_ms),
                        "prefill_ms": None,
                        "decode_ms": None,
                        "decode_step_samples_ms": [],
                        "total_generate_ms": failure_latency_ms,
                        "generated_tokens": [],
                        "output_token_hashes": [],
                        "batch_output_hash": None,
                        "reference_output_token_hashes": None,
                        "execution_status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                raw_name = f"{variant.value}_llm_measurements"
                if file_suffix:
                    raw_name = f"{raw_name}__{file_suffix}"
                (layout.raw_dir / f"{raw_name}.json").write_text(
                    json.dumps(raw_rows, indent=2),
                    encoding="utf-8",
                )
                _write_failed_generation_artifact(
                    layout,
                    common,
                    variant=variant,
                    stage=Stage.total_generate,
                    generation_config=generation_config,
                    slice_details=slice_details,
                    compile_time_ms=compile_time_ms,
                    fallback_count=fallback_count,
                    kernel_hit_count=kernel_hit_count,
                    sample_index=sample_index,
                    prompt_ids=prompt_ids,
                    prompt_lengths=prompt_lengths,
                    batch_size=batch_size,
                    tokenization_ms=tokenization_ms,
                    latency_ms=failure_latency_ms,
                    input_batch_hash=input_batch_hash,
                    exc=exc,
                    sample_records=total_records,
                    samples_ms=total_samples,
                    comparison_group=workload_slice.comparison_group,
                    configured_batch_size=int(workload_slice.batch_size),
                    prompt_bucket_id=workload_slice.prompt_bucket_id,
                    file_suffix=file_suffix,
                )
                return layout

            prefill_samples.append(float(candidate_result.prefill_ms))
            decode_samples.append(float(candidate_result.decode_ms))
            total_samples.append(float(candidate_result.total_ms))
            prefill_records.append(
                _build_stage_sample_record(sample_index, candidate_result, stage=Stage.prefill, prompt_ids=prompt_ids)
            )
            decode_records.append(
                _build_stage_sample_record(sample_index, candidate_result, stage=Stage.decode, prompt_ids=prompt_ids)
            )
            total_records.append(
                _build_stage_sample_record(sample_index, candidate_result, stage=Stage.total_generate, prompt_ids=prompt_ids)
            )
            raw_row = {
                "sample_index": sample_index,
                "comparison_group": workload_slice.comparison_group,
                "prompt_bucket_id": workload_slice.prompt_bucket_id,
                "batch_size": int(workload_slice.batch_size),
                "prompt_ids": prompt_ids,
                "input_batch_hash": candidate_result.input_batch_hash,
                "prompt_lengths": candidate_result.prompt_lengths,
                "generated_lengths": candidate_result.generated_lengths,
                "tokenization_ms": candidate_result.tokenization_ms,
                "prefill_ms": candidate_result.prefill_ms,
                "decode_ms": candidate_result.decode_ms,
                "decode_step_samples_ms": candidate_result.decode_step_samples_ms,
                "total_generate_ms": candidate_result.total_ms,
                "generated_tokens": candidate_result.generated_tokens,
                "output_token_hashes": candidate_result.output_token_hashes,
                "batch_output_hash": candidate_result.batch_output_hash,
                "reference_output_token_hashes": reference_result.output_token_hashes if reference_result is not None else None,
            }
            if store_prompts:
                raw_row["prompt_texts"] = [record["text"] for record in batch]
            raw_rows.append(raw_row)

        raw_name = f"{variant.value}_llm_measurements"
        if file_suffix:
            raw_name = f"{raw_name}__{file_suffix}"
        (layout.raw_dir / f"{raw_name}.json").write_text(
            json.dumps(raw_rows, indent=2),
            encoding="utf-8",
        )

        if variant == Variant.kf_cast and runtime_meta is not None:
            timed_runtime_stats = get_cast_runtime_stats(model) if kf_settings.record_runtime_stats else runtime_meta.get("runtime_stats", {})
            if kf_settings.record_runtime_stats:
                fallback_count = int(timed_runtime_stats.get("fallbacks_to_original", runtime_meta.get("fallback_count", 0)) or 0)
                kernel_hit_count = int(timed_runtime_stats.get("kernel_launches_succeeded", runtime_meta.get("kernel_hit_count", 0)) or 0)
            else:
                fallback_count = None
                kernel_hit_count = None
            timed_runtime_details = build_kf_runtime_details(runtime_meta, timed_runtime_stats)
            timed_audit_path = _write_device_audit_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.total_generate,
                runtime_meta=runtime_meta,
                runtime_stats=timed_runtime_stats if isinstance(timed_runtime_stats, dict) else None,
                runtime_input_device=runtime_input_device,
                tokenizer_output_devices=tokenizer_output_devices,
                require_runtime_coverage=True,
                file_suffix=file_suffix,
            )
            timed_runtime_details["device_audit_artifact_path"] = str(timed_audit_path.resolve())
        else:
            timed_runtime_details = None

        if variant == Variant.eager:
            timed_correctness_status, timed_correctness_message = reference_correctness()
        elif variant in {Variant.torch_compile, Variant.kf_cast}:
            if correctness_failures:
                timed_correctness_status = CorrectnessStatus.failed
                timed_correctness_message = "; ".join(correctness_failures[:8])
            else:
                timed_correctness_status = CorrectnessStatus.passed
                timed_correctness_message = None
        else:
            timed_correctness_status, timed_correctness_message = skipped_correctness(
                "Kernel Forge LLM correctness reference is not implemented in this scaffold."
            )

        prefill_details = _build_stage_details(Stage.prefill, prefill_records, generation_config, extra=dict(slice_details))
        decode_details = _build_stage_details(
            Stage.decode,
            decode_records,
            generation_config,
            extra={
                **slice_details,
                "decode_step_count_total": int(sum(len(record["decode_step_samples_ms"]) for record in decode_records)),
            },
        )
        total_details = _build_stage_details(Stage.total_generate, total_records, generation_config, extra=dict(slice_details))
        correctness_evidence = {
            "per_run_output_hash_verification": True,
            "correctness_checked_run_count": len(total_records),
        }
        prefill_details.update(correctness_evidence)
        decode_details.update(correctness_evidence)
        total_details.update(correctness_evidence)
        if timed_runtime_details is not None:
            prefill_details.update(timed_runtime_details)
            decode_details.update(timed_runtime_details)
            total_details.update(timed_runtime_details)

        if _measure_prefill_decode_separately(suite):
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.prefill,
                samples_ms=prefill_samples,
                warmup_count=int(suite.warmup_count),
                timed_run_count=len(prefill_samples),
                correctness_status=timed_correctness_status,
                correctness_message=timed_correctness_message,
                compile_time_ms=compile_time_ms,
                steady_state_time_ms=sum(prefill_samples) / len(prefill_samples),
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                sample_records=prefill_records,
                details=prefill_details,
                token_count=int(sum(record["prompt_token_count"] for record in prefill_records)),
                comparison_group=workload_slice.comparison_group,
                configured_batch_size=int(workload_slice.batch_size),
                prompt_bucket_id=workload_slice.prompt_bucket_id,
                file_suffix=file_suffix,
            )
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.decode,
                samples_ms=decode_samples,
                warmup_count=int(suite.warmup_count),
                timed_run_count=len(decode_samples),
                correctness_status=timed_correctness_status,
                correctness_message=timed_correctness_message,
                compile_time_ms=compile_time_ms,
                steady_state_time_ms=sum(decode_samples) / len(decode_samples),
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                sample_records=decode_records,
                details=decode_details,
                token_count=int(sum(record["decode_generated_token_count"] for record in decode_records)),
                comparison_group=workload_slice.comparison_group,
                configured_batch_size=int(workload_slice.batch_size),
                prompt_bucket_id=workload_slice.prompt_bucket_id,
                file_suffix=file_suffix,
            )
        _write_stage_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.total_generate,
            samples_ms=total_samples,
            warmup_count=int(suite.warmup_count),
            timed_run_count=len(total_samples),
            correctness_status=timed_correctness_status,
            correctness_message=timed_correctness_message,
            compile_time_ms=compile_time_ms,
            steady_state_time_ms=sum(total_samples) / len(total_samples),
            fallback_count=fallback_count,
            kernel_hit_count=kernel_hit_count,
            sample_records=total_records,
            details=total_details,
            token_count=int(sum(record["generated_token_count"] for record in total_records)),
            comparison_group=workload_slice.comparison_group,
            configured_batch_size=int(workload_slice.batch_size),
            prompt_bucket_id=workload_slice.prompt_bucket_id,
            file_suffix=file_suffix,
        )
    return layout
