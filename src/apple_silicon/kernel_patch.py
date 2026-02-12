from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import (
    BenchmarkResult,
    KernelCompileRecord,
    KernelCorrectnessRecord,
    KernelPatchCandidate,
)

TEMPLATE_VERSION = "kernel-template-v1"

ALLOWED_SOURCE_FILES = {
    "ggml-metal.metal",
    "ggml-metal-impl.h",
}

VARIANT_FAMILY_PATCH_SETS: dict[str, dict[str, Any]] = {
    "mul_mv_q4k_decode_aggr": {
        "description": "Q4_K decode-heavy matvec tuning on Apple M-class GPUs",
        "template_mutations": {"n_r0_q4_k": 3, "n_simdwidth": 32},
        "source_patches": [],
    },
    "mul_mv_q5k_decode_aggr": {
        "description": "Q5_K decode-heavy matvec tuning on Apple M-class GPUs",
        "template_mutations": {"n_r0_q5_k": 4, "n_simdwidth": 32},
        "source_patches": [],
    },
    "mul_mv_ext_q4q5_balance": {
        "description": "Balanced ext matvec throughput for mixed q4/q5 workloads",
        "template_mutations": {"n_r0_q4_k": 3, "n_r0_q5_k": 3, "n_simdwidth": 32},
        "source_patches": [],
    },
    "softmax_rmsnorm_decode_safe": {
        "description": "Decode-path conservative softmax/rmsnorm layout variant",
        "template_mutations": {"n_r0_q4_k": 2, "n_r0_q5_k": 2, "n_simdwidth": 32},
        "source_patches": [],
    },
    "long_vector_schedule_q4q5": {
        "description": "Long-context scheduling bias for q4/q5 matvec kernels",
        "template_mutations": {"n_r0_q4_k": 4, "n_r0_q5_k": 4, "n_simdwidth": 32},
        "source_patches": [],
    },
    "dequant_matvec_fuse_conservative": {
        "description": "Conservative dequant+matvec accumulation fusion surrogate via template tuning",
        "template_mutations": {"n_r0_q4_k": 5, "n_r0_q5_k": 3, "n_simdwidth": 32},
        "source_patches": [],
    },
    "rmsnorm_residual_fuse_conservative": {
        "description": "Conservative rmsnorm+residual throughput variant for decode path",
        "template_mutations": {"n_r0_q4_k": 3, "n_r0_q5_k": 5, "n_simdwidth": 32},
        "source_patches": [],
    },
}

TEMPLATE_MUTATION_SPECS: dict[str, dict[str, Any]] = {
    "n_r0_q4_k": {
        "file": "ggml-metal-impl.h",
        "regex": r"(?m)^#define\s+N_R0_Q4_K\s+\d+\s*$",
        "format": "#define N_R0_Q4_K {value}",
        "min": 1,
        "max": 8,
    },
    "n_r0_q5_k": {
        "file": "ggml-metal-impl.h",
        "regex": r"(?m)^#define\s+N_R0_Q5_K\s+\d+\s*$",
        "format": "#define N_R0_Q5_K {value}",
        "min": 1,
        "max": 8,
    },
    "n_r0_q6_k": {
        "file": "ggml-metal-impl.h",
        "regex": r"(?m)^#define\s+N_R0_Q6_K\s+\d+\s*$",
        "format": "#define N_R0_Q6_K {value}",
        "min": 1,
        "max": 8,
    },
    "n_simdwidth": {
        "file": "ggml-metal.metal",
        "regex": r"(?m)^#define\s+N_SIMDWIDTH\s+\d+.*$",
        "format": "#define N_SIMDWIDTH {value} // assuming SIMD group size is 32",
        "min": 16,
        "max": 64,
    },
}


class KernelPatchError(RuntimeError):
    pass


def variant_family_patch_set(name: str) -> dict[str, Any]:
    key = (name or "").strip()
    if key not in VARIANT_FAMILY_PATCH_SETS:
        raise KernelPatchError(f"Unknown variant family: {name}")
    payload = VARIANT_FAMILY_PATCH_SETS[key]
    return {
        "name": key,
        "description": str(payload.get("description", "")),
        "template_mutations": dict(payload.get("template_mutations") or {}),
        "source_patches": list(payload.get("source_patches") or []),
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_payload(
    *,
    template_mutations: dict[str, int],
    source_patches: list[dict[str, Any]],
) -> str:
    payload = {
        "template_mutations": template_mutations,
        "source_patches": source_patches,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _copy_base_resources(llamacpp_root: Path, resources_dir: Path) -> None:
    metal_dir = llamacpp_root / "ggml" / "src" / "ggml-metal"
    common_h = llamacpp_root / "ggml" / "src" / "ggml-common.h"
    required = {
        "ggml-metal.metal": metal_dir / "ggml-metal.metal",
        "ggml-metal-impl.h": metal_dir / "ggml-metal-impl.h",
        "ggml-common.h": common_h,
    }
    resources_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name, src in required.items() if not src.exists()]
    if missing:
        raise KernelPatchError(
            f"Missing required Metal resource files under {llamacpp_root}: {', '.join(missing)}"
        )
    for name, src in required.items():
        shutil.copy2(src, resources_dir / name)


def _apply_template_mutations(resources_dir: Path, template_mutations: dict[str, int]) -> None:
    for key, value in template_mutations.items():
        spec = TEMPLATE_MUTATION_SPECS.get(key)
        if spec is None:
            raise KernelPatchError(f"Unsupported template mutation key: {key}")
        if not isinstance(value, int):
            raise KernelPatchError(f"Template mutation '{key}' must be an integer")
        if value < int(spec["min"]) or value > int(spec["max"]):
            raise KernelPatchError(
                f"Template mutation '{key}' out of range: {value} not in [{spec['min']}, {spec['max']}]"
            )

        target = resources_dir / str(spec["file"])
        text = target.read_text(encoding="utf-8")
        replacement = str(spec["format"]).format(value=value)
        updated, n = re.subn(str(spec["regex"]), replacement, text, count=1)
        if n != 1:
            raise KernelPatchError(
                f"Failed to apply template mutation '{key}' to {target.name}; expected one match"
            )
        target.write_text(updated, encoding="utf-8")


def _normalize_source_patch(patch: dict[str, Any]) -> dict[str, Any]:
    file_name = str(patch.get("file", "")).strip()
    find = str(patch.get("find", ""))
    replace = str(patch.get("replace", ""))
    patch_id = str(patch.get("patch_id", "")).strip() or "patch"
    if file_name not in ALLOWED_SOURCE_FILES:
        raise KernelPatchError(f"Source patch file not allowed: {file_name}")
    if not find:
        raise KernelPatchError("Source patch must include non-empty 'find'")
    if len(find) > 600 or len(replace) > 600:
        raise KernelPatchError("Source patch find/replace length exceeds 600 characters")
    return {
        "patch_id": patch_id,
        "file": file_name,
        "find": find,
        "replace": replace,
    }


def _apply_source_patches(resources_dir: Path, source_patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for patch in source_patches:
        if not isinstance(patch, dict):
            raise KernelPatchError("Every source patch entry must be an object")
        p = _normalize_source_patch(patch)
        target = resources_dir / p["file"]
        text = target.read_text(encoding="utf-8")
        occurrences = text.count(p["find"])
        if occurrences != 1:
            raise KernelPatchError(
                f"Patch {p['patch_id']} expected one occurrence in {p['file']}, found {occurrences}"
            )
        target.write_text(text.replace(p["find"], p["replace"], 1), encoding="utf-8")
        normalized.append(p)
    return normalized


def _source_hash(resources_dir: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(ALLOWED_SOURCE_FILES | {"ggml-common.h"}):
        path = resources_dir / name
        if not path.exists():
            continue
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def build_kernel_patch_candidate(
    *,
    llamacpp_root: Path,
    candidate_cache_dir: Path,
    candidate_id: str,
    template_mutations: dict[str, int],
    source_patches: list[dict[str, Any]],
) -> KernelPatchCandidate:
    normalized_mutations = {
        str(k).strip().lower(): int(v)
        for k, v in (template_mutations or {}).items()
        if str(k).strip()
    }
    normalized_source_patches = [
        patch for patch in (source_patches or []) if isinstance(patch, dict)
    ]

    payload_str = _canonical_payload(
        template_mutations=normalized_mutations,
        source_patches=normalized_source_patches,
    )
    patch_hash = _sha256_text(payload_str)
    effective_candidate_id = f"{candidate_id}-{patch_hash[:10]}"

    candidate_dir = candidate_cache_dir / effective_candidate_id
    resources_dir = candidate_dir / "resources"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    _copy_base_resources(llamacpp_root, resources_dir)
    _apply_template_mutations(resources_dir, normalized_mutations)
    applied_source_patches = _apply_source_patches(resources_dir, normalized_source_patches)
    source_hash = _source_hash(resources_dir)

    manifest = {
        "candidate_id": effective_candidate_id,
        "template_version": TEMPLATE_VERSION,
        "patch_hash": patch_hash,
        "source_hash": source_hash,
        "template_mutations": normalized_mutations,
        "source_patches": applied_source_patches,
    }
    (candidate_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return KernelPatchCandidate(
        candidate_id=effective_candidate_id,
        template_version=TEMPLATE_VERSION,
        resources_dir=str(resources_dir),
        patch_hash=patch_hash,
        source_hash=source_hash,
        template_mutations=normalized_mutations,
        source_patches=applied_source_patches,
    )


def classify_compile_record(
    result: BenchmarkResult,
    *,
    compile_meta: dict[str, Any] | None = None,
) -> KernelCompileRecord:
    meta = compile_meta or {}
    compile_warmup_done = bool(meta.get("compile_warmup_done", False))
    pipeline_cache_key = str(meta.get("pipeline_cache_key", ""))
    compile_time_ms = (
        float(meta["compile_time_ms"]) if isinstance(meta.get("compile_time_ms"), (int, float)) else None
    )
    toolchain_fingerprint = str(meta.get("toolchain_fingerprint", ""))
    stderr_merged = "\n".join(str(run.get("stderr", "")) for run in result.runs if run.get("stderr"))
    stdout_merged = "\n".join(str(run.get("stdout", "")) for run in result.runs if run.get("stdout"))
    all_text = f"{stdout_merged}\n{stderr_merged}".strip()
    stderr_hash = _sha256_text(stderr_merged) if stderr_merged else ""
    all_zero = all(int(run.get("return_code", 1)) == 0 for run in result.runs)
    if all_zero:
        return KernelCompileRecord(
            attempted=True,
            success=True,
            classification="compiled_or_loaded",
            stderr_hash=stderr_hash,
            compile_warmup_done=compile_warmup_done,
            pipeline_cache_key=pipeline_cache_key,
            compile_time_ms=compile_time_ms,
            toolchain_fingerprint=toolchain_fingerprint,
        )

    text = all_text.lower()
    if "failed to initialize the metal library" in text or "newlibrarywithsource" in text:
        return KernelCompileRecord(
            attempted=True,
            success=False,
            classification="metal_compile_error",
            stderr_hash=stderr_hash,
            error="metal_compile_error",
            compile_warmup_done=compile_warmup_done,
            pipeline_cache_key=pipeline_cache_key,
            compile_time_ms=compile_time_ms,
            toolchain_fingerprint=toolchain_fingerprint,
        )
    if "error:" in text and "metal" in text:
        return KernelCompileRecord(
            attempted=True,
            success=False,
            classification="metal_compile_error",
            stderr_hash=stderr_hash,
            error="metal_compile_error",
            compile_warmup_done=compile_warmup_done,
            pipeline_cache_key=pipeline_cache_key,
            compile_time_ms=compile_time_ms,
            toolchain_fingerprint=toolchain_fingerprint,
        )
    return KernelCompileRecord(
        attempted=True,
        success=False,
        classification="runtime_error",
        stderr_hash=stderr_hash,
        error="runtime_error",
        compile_warmup_done=compile_warmup_done,
        pipeline_cache_key=pipeline_cache_key,
        compile_time_ms=compile_time_ms,
        toolchain_fingerprint=toolchain_fingerprint,
    )


def classify_correctness_record(
    *,
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    strict_parity: bool = False,
    similarity_threshold: float = 0.98,
) -> KernelCorrectnessRecord:
    details: dict[str, Any] = {}
    metrics = [
        ("prefill_tokens_per_sec", baseline.metrics.prefill_tokens_per_sec, candidate.metrics.prefill_tokens_per_sec),
        ("decode_tokens_per_sec", baseline.metrics.decode_tokens_per_sec, candidate.metrics.decode_tokens_per_sec),
    ]

    for name, base_v, cand_v in metrics:
        if base_v is None or cand_v is None:
            return KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification="missing_metric",
                details={"metric": name},
            )
        if not math.isfinite(float(base_v)) or not math.isfinite(float(cand_v)):
            return KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification="non_finite_metric",
                details={"metric": name},
            )
        ratio = float(cand_v) / float(base_v) if float(base_v) != 0 else 0.0
        details[f"{name}_ratio"] = ratio
        # Hard sanity fence: reject pathological outliers that usually indicate broken runs.
        if ratio > 5.0 or ratio < 0.1:
            return KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification="outlier_metric_ratio",
                details={"metric": name, "ratio": ratio},
            )

    text_chunks: list[str] = []
    for run in candidate.runs:
        text_chunks.append(str(run.get("stdout", "")))
        text_chunks.append(str(run.get("stderr", "")))
    joined_text = "\n".join(text_chunks).lower()
    if re.search(r"(^|[^a-z])nan([^a-z]|$)", joined_text) or re.search(
        r"(^|[^a-z])inf([^a-z]|$)", joined_text
    ):
        return KernelCorrectnessRecord(
            attempted=True,
            success=False,
            classification="nan_or_inf_detected",
            details={"scan": "detected nan/inf token in candidate output"},
        )

    if strict_parity:
        ratios: list[float] = []
        failed: list[dict[str, Any]] = []
        pair_count = min(len(baseline.runs), len(candidate.runs))
        for i in range(pair_count):
            left = _normalize_generation_text(baseline.runs[i])
            right = _normalize_generation_text(candidate.runs[i])
            ratio = difflib.SequenceMatcher(None, left, right).ratio()
            ratios.append(ratio)
            if ratio < similarity_threshold:
                failed.append(
                    {
                        "run_index": i,
                        "similarity": ratio,
                        "threshold": similarity_threshold,
                    }
                )
        details["semantic_similarity_min"] = min(ratios) if ratios else None
        details["semantic_similarity_max"] = max(ratios) if ratios else None
        details["semantic_checked_runs"] = pair_count
        details["semantic_failed_runs"] = len(failed)
        if failed:
            details["semantic_failures"] = failed[:5]
            return KernelCorrectnessRecord(
                attempted=True,
                success=False,
                classification="semantic_mismatch",
                details=details,
            )

    return KernelCorrectnessRecord(
        attempted=True,
        success=True,
        classification="metric_sanity_ok",
        details=details,
    )


def kernel_candidate_dict(candidate: KernelPatchCandidate) -> dict[str, Any]:
    return asdict(candidate)


def _normalize_generation_text(run: dict[str, Any]) -> str:
    # llama.cpp prints interactive-style transcripts with carriage/backspace controls.
    text = f"{run.get('stdout', '')}\n{run.get('stderr', '')}"
    if not text.strip():
        return ""
    text = text.replace("\r", "")
    for _ in range(4):
        text = re.sub(r".\x08", "", text)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

    # Keep the response body before summary throughput line.
    if "[ Prompt:" in text:
        text = text.split("[ Prompt:", 1)[0]
    if "available commands:" in text:
        text = text.split("available commands:", 1)[-1]
    if "\n> " in text:
        text = text.split("\n> ", 1)[-1]
    lines = text.splitlines()
    if lines:
        # First line is usually the prompt itself.
        lines = lines[1:]
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("/"):
            continue
        if s.lower().startswith("exiting"):
            continue
        if s.startswith("build      :") or s.startswith("model      :") or s.startswith("modalities"):
            continue
        cleaned.append(s)
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()
