from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from .constants import (
    CHAT_PROFILE,
    LONG_CLAIM_PROFILE,
    current_cache_root,
    LONG_PROFILE,
    LONG_SMOKE_PROFILE,
    PROMPT_SUITE_PATH,
)
from .runtime_args import sanitize_runtime_args, split_runtime_args_and_env
from .types import BenchmarkMetrics, BenchmarkResult, WorkloadProfile

PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*\(([\d.]+)\s*tokens per second\)",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs?.*\(([\d.]+)\s*tokens per second\)",
    re.IGNORECASE,
)
MEM_RE = re.compile(r"mem required\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)
PROMPT_GEN_RE = re.compile(
    r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
    re.IGNORECASE,
)
KERNEL_MENTION_RE = re.compile(r"\bkernel[_\s:'\"]+([A-Za-z0-9_]+)\b", re.IGNORECASE)
CGINS_METAL_AUDIT_RE = re.compile(r"CGINS_METAL_AUDIT\s+(.+)$", re.IGNORECASE | re.MULTILINE)

_METAL_ENV_CACHE: dict[str, str] | None = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_capture(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=merged_env)
    except Exception as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _extract_kernel_mentions(text: str) -> list[str]:
    counts: dict[str, int] = {}
    for match in KERNEL_MENTION_RE.finditer(text or ""):
        name = str(match.group(1) or "").strip()
        if not name:
            continue
        counts[name] = int(counts.get(name, 0)) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ordered]


def _safe_audit_token(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip())
    token = token.strip("_.-")
    return token[:160] or fallback


def _parse_metal_audit_sentinel(text: str) -> dict[str, str]:
    match = CGINS_METAL_AUDIT_RE.search(text or "")
    if not match:
        return {}
    payload = str(match.group(1) or "").strip()
    out: dict[str, str] = {}
    for chunk in payload.split():
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        key = str(k or "").strip()
        if not key:
            continue
        out[key] = str(v or "").strip()
    return out


def _normalize_backend_dispatch_audit(payload: dict[str, Any]) -> dict[str, Any]:
    kernels_raw = payload.get("kernels")
    kernels_out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    if isinstance(kernels_raw, list):
        for row in kernels_raw:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("kernel") or "").strip()
            if not label:
                continue
            count_raw = row.get("count")
            try:
                count = int(count_raw) if count_raw is not None else 0
            except Exception:
                count = 0
            if count < 0:
                count = 0
            counts[label] = int(counts.get(label, 0)) + count
    for label, count in counts.items():
        kernels_out.append({"kernel": label, "mentions": int(count)})
    kernels_out.sort(key=lambda row: (-int(row.get("mentions") or 0), str(row.get("kernel") or "")))

    return {
        "schema_version": int(payload.get("schema_version") or 1),
        "attempt_id": str(payload.get("attempt_id") or ""),
        "selected_dispatch_rule_id": str(payload.get("dispatch_rule_id") or ""),
        "device_name": str(payload.get("device_name") or ""),
        "resource_dir": str(payload.get("resource_dir") or ""),
        "metallib_path": str(payload.get("metallib_path") or ""),
        "metallib_source": str(payload.get("metallib_source") or ""),
        "top_kernels": kernels_out[:5],
        "kernel_mention_counts": {str(k): int(v) for k, v in counts.items()},
    }


def _read_backend_dispatch_audit(audit_path: Path | None) -> tuple[dict[str, Any] | None, bool]:
    if audit_path is None:
        return None, False
    if not audit_path.exists():
        return None, False
    try:
        raw = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return None, True
    if not isinstance(raw, dict):
        return None, True
    required = {"attempt_id", "dispatch_rule_id", "metallib_source", "metallib_path", "kernels"}
    if not required.issubset(set(raw.keys())):
        return None, True
    if not isinstance(raw.get("kernels"), list):
        return None, True
    return _normalize_backend_dispatch_audit(raw), False


def _resolve_dispatch_audit_status(
    *,
    audit_path: Path | None,
    backend_payload: dict[str, Any] | None,
    parse_failed: bool,
) -> str:
    if backend_payload is not None:
        return "ok"
    if audit_path is None:
        return "backend_noaudit"
    if parse_failed:
        return "parse_fail"
    if not audit_path.exists():
        return "missing"
    return "parse_fail"


def _build_dispatch_audit(
    *,
    merged_text: str,
    resources_path: Path | None,
    runtime_env: dict[str, str],
    dispatch_audit_path: Path | None,
    candidate_resources_expected: bool,
) -> dict[str, Any]:
    backend_payload, parse_failed = _read_backend_dispatch_audit(dispatch_audit_path)
    status = _resolve_dispatch_audit_status(
        audit_path=dispatch_audit_path,
        backend_payload=backend_payload,
        parse_failed=parse_failed,
    )
    sentinel = _parse_metal_audit_sentinel(merged_text)

    counts: dict[str, int] = {}
    ordered: list[tuple[str, int]] = []
    mentions: list[str] = []
    dispatch_audit_source = "backend_json" if backend_payload is not None else "heuristic_fallback"
    metallib_source = ""
    metallib_path = ""

    if backend_payload is not None:
        counts = {
            str(k): int(v)
            for k, v in dict(backend_payload.get("kernel_mention_counts") or {}).items()
            if str(k).strip()
        }
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        mentions = [name for name, _ in ordered]
        metallib_source = str(backend_payload.get("metallib_source") or "")
        metallib_path = str(backend_payload.get("metallib_path") or "")
    else:
        for match in KERNEL_MENTION_RE.finditer(merged_text or ""):
            name = str(match.group(1) or "").strip()
            if not name:
                continue
            counts[name] = int(counts.get(name, 0)) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        mentions = [name for name, _ in ordered]
        if sentinel:
            dispatch_audit_source = "stderr_sentinel"
            metallib_source = str(sentinel.get("metallib_source") or "")
            metallib_path = str(sentinel.get("metallib_path") or "")

    selected_dispatch_rule_id = (
        str((backend_payload or {}).get("selected_dispatch_rule_id") or "").strip()
        or str(runtime_env.get("CGINS_DISPATCH_RULE_ID", "")).strip()
    )

    if not metallib_path and resources_path is not None:
        metallib_path = str((Path(resources_path) / "default.metallib").resolve())

    metallib_present = bool(Path(metallib_path).exists()) if metallib_path else False
    candidate_resources_used = bool(status == "ok" and metallib_source == "candidate")

    return {
        "selected_dispatch_rule_id": selected_dispatch_rule_id,
        "resources_path": str(resources_path) if resources_path is not None else "",
        "metallib_path": str((backend_payload or {}).get("metallib_path") or sentinel.get("metallib_path") or metallib_path),
        "metallib_present": bool(metallib_present),
        "metallib_source": str((backend_payload or {}).get("metallib_source") or sentinel.get("metallib_source") or metallib_source),
        "dispatch_audit_status": status,
        "candidate_resources_expected": bool(candidate_resources_expected),
        "candidate_resources_used": bool(candidate_resources_used),
        "dispatch_audit_path": str(dispatch_audit_path) if dispatch_audit_path is not None else "",
        "dispatch_audit_source": dispatch_audit_source,
        "device_name": str((backend_payload or {}).get("device_name") or ""),
        "resource_dir": str((backend_payload or {}).get("resource_dir") or sentinel.get("resource_dir") or ""),
        "kernel_mentions": mentions[:20],
        "kernel_mention_counts": counts,
        "top_kernels": [{"kernel": name, "mentions": int(cnt)} for name, cnt in ordered[:5]],
        "dispatch_source": "candidate_resources" if resources_path is not None else "baseline",
    }


def metal_toolchain_env() -> dict[str, str]:
    global _METAL_ENV_CACHE
    if _METAL_ENV_CACHE is not None:
        return dict(_METAL_ENV_CACHE)

    # First prefer the currently-selected developer dir/toolchain.
    rc, _, _ = _run_capture(["xcrun", "-f", "metal"])
    if rc == 0:
        _METAL_ENV_CACHE = {}
        return {}

    # If CLT is selected without Metal tools, fall back to full Xcode when available.
    xcode_dev = Path("/Applications/Xcode.app/Contents/Developer")
    if xcode_dev.exists():
        candidate_env = {"DEVELOPER_DIR": str(xcode_dev)}
        rc2, _, _ = _run_capture(["xcrun", "-f", "metal"], env=candidate_env)
        if rc2 == 0:
            _METAL_ENV_CACHE = dict(candidate_env)
            return dict(candidate_env)

    _METAL_ENV_CACHE = {}
    return {}


def resolve_metal_toolchain_paths() -> dict[str, Any]:
    tool_env = metal_toolchain_env()
    rc_metal, out_metal, err_metal = _run_capture(["xcrun", "-f", "metal"], env=tool_env)
    rc_metallib, out_metallib, err_metallib = _run_capture(["xcrun", "-f", "metallib"], env=tool_env)
    return {
        "success": (rc_metal == 0 and rc_metallib == 0),
        "developer_dir": tool_env.get("DEVELOPER_DIR", ""),
        "metal_path": out_metal.strip(),
        "metallib_path": out_metallib.strip(),
        "metal_rc": int(rc_metal),
        "metallib_rc": int(rc_metallib),
        "metal_err": err_metal.strip(),
        "metallib_err": err_metallib.strip(),
    }


def metal_toolchain_fingerprint() -> str:
    tool_env = metal_toolchain_env()
    parts: list[str] = []
    for cmd in (
        ["xcrun", "--version"],
        ["xcrun", "-f", "metal"],
        ["xcodebuild", "-version"],
        ["sw_vers", "-productVersion"],
    ):
        try:
            rc, out, err = _run_capture(cmd, env=tool_env)
        except Exception:
            rc, out, err = 1, "", ""
        payload = f"{' '.join(cmd)}|rc={rc}|out={out.strip()}|err={err.strip()}"
        parts.append(payload)
    parts.append(f"developer_dir={tool_env.get('DEVELOPER_DIR', '')}")
    return _sha256_text("\n".join(parts))


def pipeline_cache_key(
    *,
    llamacpp_commit: str,
    chip_family: str,
    macos_version: str,
    source_hash: str,
    candidate_hash: str,
    toolchain_fingerprint: str,
    compile_defines: list[str] | tuple[str, ...] | None = None,
) -> str:
    defines = [str(d).strip() for d in (compile_defines or []) if str(d).strip()]
    payload = {
        "llamacpp_commit": (llamacpp_commit or "").strip(),
        "chip_family": (chip_family or "").strip().lower(),
        "macos_version": (macos_version or "").strip().lower(),
        "source_hash": (source_hash or "").strip().lower(),
        "candidate_hash": (candidate_hash or "").strip().lower(),
        "toolchain_fingerprint": (toolchain_fingerprint or "").strip().lower(),
        "compile_defines": sorted(set(defines)),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_text(text)


def prepare_candidate_resources_for_benchmark(
    *,
    resources_path: Path,
    llamacpp_commit: str,
    chip_family: str,
    macos_version: str,
    source_hash: str,
    candidate_hash: str,
    compile_cache_root: Path | None = None,
) -> dict[str, Any]:
    toolchain = metal_toolchain_fingerprint()
    # Mirror the runtime source-compile macro gate so precompiled candidate metallibs
    # expose the same BF16 entry points used by MUL_MAT correctness checks.
    compile_defines = ["GGML_METAL_HAS_BF16=1"]
    cache_key = pipeline_cache_key(
        llamacpp_commit=llamacpp_commit,
        chip_family=chip_family,
        macos_version=macos_version,
        source_hash=source_hash,
        candidate_hash=candidate_hash,
        toolchain_fingerprint=toolchain,
        compile_defines=compile_defines,
    )
    cache_root = (compile_cache_root or (current_cache_root() / "compile_cache")).expanduser().resolve()
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    metallib_dst = resources_path / "default.metallib"
    metallib_dst.parent.mkdir(parents=True, exist_ok=True)
    metallib_cache = cache_dir / "default.metallib"
    src_metal = resources_path / "ggml-metal.metal"
    if not src_metal.exists():
        return {
            "compile_warmup_done": False,
            "pipeline_cache_key": cache_key,
            "compile_time_ms": None,
            "toolchain_fingerprint": toolchain,
            "cache_hit": False,
            "success": False,
            "classification": "missing_metal_source",
            "stderr_hash": "",
            "error": f"Missing source file: {src_metal}",
        }

    if metallib_cache.exists():
        shutil.copy2(metallib_cache, metallib_dst)
        return {
            "compile_warmup_done": True,
            "pipeline_cache_key": cache_key,
            "compile_time_ms": 0.0,
            "toolchain_fingerprint": toolchain,
            "cache_hit": True,
            "success": True,
            "classification": "cache_hit",
            "stderr_hash": "",
            "error": "",
        }

    start_t = time.perf_counter()
    tmp_dir = cache_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    air = tmp_dir / "ggml-metal.air"
    metallib_tmp = tmp_dir / "default.metallib"

    tool_env = metal_toolchain_env()

    cmd_compile = [
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
        "-I",
        str(resources_path),
        *[f"-D{define}" for define in compile_defines],
        "-c",
        str(src_metal),
        "-o",
        str(air),
    ]
    rc1, out1, err1 = _run_capture(cmd_compile, env=tool_env)
    if rc1 != 0:
        all_err = "\n".join([out1, err1]).strip()
        return {
            "compile_warmup_done": False,
            "pipeline_cache_key": cache_key,
            "compile_time_ms": (time.perf_counter() - start_t) * 1000.0,
            "toolchain_fingerprint": toolchain,
            "cache_hit": False,
            "success": False,
            "classification": "metal_compile_error",
            "stderr_hash": _sha256_text(all_err) if all_err else "",
            "error": all_err,
        }

    cmd_link = [
        "xcrun",
        "-sdk",
        "macosx",
        "metallib",
        str(air),
        "-o",
        str(metallib_tmp),
    ]
    rc2, out2, err2 = _run_capture(cmd_link, env=tool_env)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    if rc2 != 0 or not metallib_tmp.exists():
        all_err = "\n".join([out2, err2]).strip()
        return {
            "compile_warmup_done": False,
            "pipeline_cache_key": cache_key,
            "compile_time_ms": elapsed_ms,
            "toolchain_fingerprint": toolchain,
            "cache_hit": False,
            "success": False,
            "classification": "metallib_link_error",
            "stderr_hash": _sha256_text(all_err) if all_err else "",
            "error": all_err,
        }

    shutil.copy2(metallib_tmp, metallib_cache)
    shutil.copy2(metallib_tmp, metallib_dst)
    return {
        "compile_warmup_done": True,
        "pipeline_cache_key": cache_key,
        "compile_time_ms": elapsed_ms,
        "toolchain_fingerprint": toolchain,
        "cache_hit": False,
        "success": True,
        "classification": "compiled",
        "stderr_hash": "",
        "error": "",
    }


def load_prompt_suite(extra_prompt_file: str | None = None) -> list[str]:
    prompts: list[str] = []
    try:
        data = json.loads(PROMPT_SUITE_PATH.read_text(encoding="utf-8"))
        prompts.extend([str(p) for p in data.get("prompts", []) if p])
    except Exception:
        pass

    if extra_prompt_file:
        path = Path(extra_prompt_file).expanduser().resolve()
        if path.exists():
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            prompts.extend([line for line in lines if line])

    if not prompts:
        prompts = ["Summarize why reproducible benchmarking matters for llama.cpp performance tuning."]
    return prompts


def _approx_token_count(text: str) -> int:
    # Coarse deterministic estimate for prompt expansion when tokenizer is unavailable.
    return max(1, int(len(text) / 4))


def _expand_prompt_to_target(base_prompt: str, target_tokens: int) -> str:
    prompt = base_prompt.strip()
    if target_tokens <= 0:
        return prompt
    est_tokens = _approx_token_count(prompt)
    parts: list[str] = [prompt]
    while est_tokens < target_tokens:
        parts.append(prompt)
        est_tokens += _approx_token_count(prompt)
        if len(parts) > 4096:
            break
    return "\n\n".join(parts)


def workload_profiles(profile_mode: str, gate_mode: str) -> list[WorkloadProfile]:
    mode = (profile_mode or "both").lower()
    gate = (gate_mode or "quick").lower()

    def build(spec: dict[str, Any]) -> WorkloadProfile:
        repeats = spec["repeats_full"] if gate == "full" else spec["repeats_quick"]
        return WorkloadProfile(
            name=spec["name"],
            ctx=int(spec["ctx"]),
            prompt_tokens_target=int(spec["prompt_tokens_target"]),
            generate_tokens=int(spec["generate_tokens"]),
            repeats=int(repeats),
        )

    # Comma-separated explicit profile list is supported for staged optimization flows.
    if "," in mode:
        mapping = {
            "chat": CHAT_PROFILE,
            "long": LONG_PROFILE,
            "long_smoke": LONG_SMOKE_PROFILE,
            "long_claim": LONG_CLAIM_PROFILE,
        }
        rows: list[WorkloadProfile] = []
        seen: set[str] = set()
        for item in [x.strip().lower() for x in mode.split(",") if x.strip()]:
            if item in seen:
                continue
            spec = mapping.get(item)
            if spec is None:
                continue
            rows.append(build(spec))
            seen.add(item)
        if rows:
            return rows

    if mode == "chat":
        return [build(CHAT_PROFILE)]
    if mode == "long":
        return [build(LONG_PROFILE)]
    if mode == "long_smoke":
        return [build(LONG_SMOKE_PROFILE)]
    if mode == "long_claim":
        return [build(LONG_CLAIM_PROFILE)]
    return [build(CHAT_PROFILE), build(LONG_PROFILE)]


def resolve_llama_cli(llamacpp_root: Path) -> Path:
    candidates = [
        llamacpp_root / "build" / "bin" / "llama-cli",
        llamacpp_root / "bin" / "llama-cli",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"llama-cli not found under {llamacpp_root}. Run bootstrap first."
    )


def _parse_run_output(text: str) -> dict[str, Any]:
    prompt_match = PROMPT_RE.search(text)
    eval_match = EVAL_RE.search(text)
    mem_match = MEM_RE.search(text)

    prompt_ms = float(prompt_match.group(1)) if prompt_match else None
    prompt_tokens = int(prompt_match.group(2)) if prompt_match else None
    prompt_tps = float(prompt_match.group(3)) if prompt_match else None

    eval_ms = float(eval_match.group(1)) if eval_match else None
    eval_tokens = int(eval_match.group(2)) if eval_match else None
    eval_tps = float(eval_match.group(3)) if eval_match else None

    if prompt_tps is None or eval_tps is None:
        prompt_gen_match = PROMPT_GEN_RE.search(text)
        if prompt_gen_match:
            if prompt_tps is None:
                prompt_tps = float(prompt_gen_match.group(1))
            if eval_tps is None:
                eval_tps = float(prompt_gen_match.group(2))

    ttft_ms = None
    if prompt_ms is not None and eval_ms is not None and eval_tokens:
        ttft_ms = prompt_ms + (eval_ms / max(eval_tokens, 1))

    token_latency_ms = None
    if eval_tps and eval_tps > 0:
        token_latency_ms = 1000.0 / eval_tps

    peak_mem = float(mem_match.group(1)) if mem_match else None

    return {
        "prefill_tokens_per_sec": prompt_tps,
        "decode_tokens_per_sec": eval_tps,
        "ttft_ms": ttft_ms,
        "token_latency_ms": token_latency_ms,
        "peak_memory_mib": peak_mem,
        "prompt_ms": prompt_ms,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_actual": prompt_tokens,
        "eval_ms": eval_ms,
        "eval_tokens": eval_tokens,
    }


def _aggregate(profile: WorkloadProfile, runs: list[dict[str, Any]], elapsed: float) -> BenchmarkResult:
    prefill = [r["prefill_tokens_per_sec"] for r in runs if r.get("prefill_tokens_per_sec")]
    decode = [r["decode_tokens_per_sec"] for r in runs if r.get("decode_tokens_per_sec")]
    ttft = [r["ttft_ms"] for r in runs if r.get("ttft_ms")]
    lat = [r["token_latency_ms"] for r in runs if r.get("token_latency_ms")]
    mem = [r["peak_memory_mib"] for r in runs if r.get("peak_memory_mib")]

    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else None

    metrics = BenchmarkMetrics(
        prefill_tokens_per_sec=mean(prefill) if prefill else None,
        decode_tokens_per_sec=mean(decode) if decode else None,
        ttft_ms=mean(ttft) if ttft else None,
        p50_token_latency_ms=p50,
        p95_token_latency_ms=p95,
        peak_memory_mib=max(mem) if mem else None,
    )
    return BenchmarkResult(profile=profile, metrics=metrics, elapsed_seconds=elapsed, runs=runs)


def run_profile_benchmark(
    *,
    llama_cli: Path,
    model_path: Path,
    profile: WorkloadProfile,
    prompts: list[str],
    resources_path: Path | None = None,
    extra_args: list[str] | None = None,
    capture_raw_output: bool = False,
    force_source_compile: bool = False,
    enforce_long_prompt_target: bool = False,
    long_prompt_token_tolerance: int = 0,
    prompt_cache_path: Path | None = None,
    prompt_cache_ro: bool = False,
    prompt_cache_all: bool = False,
    build_prompt_cache_first: bool = False,
    dispatch_attempt_id: str = "",
    dispatch_rule_id: str = "",
    dispatch_audit_dir: Path | None = None,
    candidate_resources_expected: bool = False,
) -> BenchmarkResult:
    runs: list[dict[str, Any]] = []
    safe_extra_args = sanitize_runtime_args(extra_args or [])
    cli_extra_args, runtime_env = split_runtime_args_and_env(safe_extra_args)
    runtime_env = dict(runtime_env)
    effective_candidate_resources_expected = bool(
        candidate_resources_expected or resources_path is not None
    )
    temp_cli_root: Path | None = None
    cli_for_run = llama_cli
    prompt_cache_file = (
        Path(prompt_cache_path).expanduser().resolve() if prompt_cache_path is not None else None
    )
    prompt_cache_build_elapsed_ms: float | None = None
    prompt_cache_build_rc: int | None = None

    if force_source_compile and resources_path is not None:
        temp_cli_root = Path(tempfile.mkdtemp(prefix="cgins_as_llama_src_"))
        cli_for_run = temp_cli_root / "llama-cli"
        shutil.copy2(llama_cli, cli_for_run)
        cli_for_run.chmod(0o755)

    try:
        if prompt_cache_file is not None:
            prompt_cache_file.parent.mkdir(parents=True, exist_ok=True)
            if build_prompt_cache_first:
                base_prompt = prompts[0] if prompts else "hello"
                seed_prompt = base_prompt
                if profile.name.startswith("long") and profile.prompt_tokens_target > 0:
                    seed_prompt = _expand_prompt_to_target(base_prompt, profile.prompt_tokens_target)
                build_cmd = [
                    str(cli_for_run),
                    "-m",
                    str(model_path),
                    "-c",
                    str(profile.ctx),
                    "-n",
                    "0",
                    "-p",
                    seed_prompt,
                    "--single-turn",
                    "--seed",
                    "42",
                    "--temp",
                    "0",
                    "-ngl",
                    "99",
                    "--prompt-cache",
                    str(prompt_cache_file),
                ]
                if prompt_cache_all:
                    build_cmd.append("--prompt-cache-all")
                if cli_extra_args:
                    build_cmd.extend(cli_extra_args)
                build_env = dict(os.environ)
                if resources_path is not None:
                    build_env["GGML_METAL_PATH_RESOURCES"] = str(resources_path)
                if runtime_env:
                    build_env.update(runtime_env)
                build_started = time.perf_counter()
                build_proc = subprocess.run(
                    build_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=build_env,
                )
                prompt_cache_build_elapsed_ms = (time.perf_counter() - build_started) * 1000.0
                prompt_cache_build_rc = int(build_proc.returncode)

        start_t = time.perf_counter()
        for i in range(profile.repeats):
            dispatch_audit_path: Path | None = None
            base_prompt = prompts[i % len(prompts)]
            prompt = base_prompt
            if profile.name.startswith("long") and profile.prompt_tokens_target > 0:
                prompt = _expand_prompt_to_target(base_prompt, profile.prompt_tokens_target)
            cmd = [
                str(cli_for_run),
                "-m",
                str(model_path),
                "-c",
                str(profile.ctx),
                "-n",
                str(profile.generate_tokens),
                "-p",
                prompt,
                "-st",
                "--single-turn",
                "--seed",
                "42",
                "--temp",
                "0",
                "-ngl",
                "99",
            ]
            if prompt_cache_file is not None:
                cmd.extend(["--prompt-cache", str(prompt_cache_file)])
                if prompt_cache_ro:
                    cmd.append("--prompt-cache-ro")
                if prompt_cache_all:
                    cmd.append("--prompt-cache-all")
            if cli_extra_args:
                cmd.extend(cli_extra_args)
            needs_local_env = bool(
                resources_path is not None
                or runtime_env
                or dispatch_audit_dir is not None
            )
            local_env = dict(os.environ) if needs_local_env else None
            if resources_path is not None:
                local_env["GGML_METAL_PATH_RESOURCES"] = str(resources_path)
            if runtime_env:
                local_env.update(runtime_env)
            if dispatch_audit_dir is not None:
                if local_env is None:
                    local_env = dict(os.environ)
                dispatch_audit_dir.mkdir(parents=True, exist_ok=True)
                base_attempt_token = _safe_audit_token(
                    dispatch_attempt_id or profile.name,
                    fallback="attempt",
                )
                profile_token = _safe_audit_token(profile.name, fallback="profile")
                dispatch_audit_path = dispatch_audit_dir / (
                    f"{base_attempt_token}_{profile_token}_r{i + 1:03d}.json"
                )
                if dispatch_audit_path.exists():
                    dispatch_audit_path.unlink()
                local_env["CGINS_ATTEMPT_ID"] = str(
                    dispatch_attempt_id or f"{profile.name}_repeat_{i + 1}"
                )
                local_env["CGINS_DISPATCH_RULE_ID"] = str(dispatch_rule_id or "")
                local_env["CGINS_DISPATCH_AUDIT_PATH"] = str(dispatch_audit_path)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=local_env,
            )
            merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
            parsed = _parse_run_output(merged)
            parsed["return_code"] = proc.returncode
            parsed["command"] = " ".join(shlex.quote(c) for c in cmd)
            parsed["runtime_env"] = dict(runtime_env)
            parsed["profile_name"] = profile.name
            parsed["ctx_used"] = int(profile.ctx)
            parsed["prompt_tokens_target"] = int(profile.prompt_tokens_target)
            parsed["prompt_cache_mode"] = (
                "ro" if (prompt_cache_file is not None and prompt_cache_ro) else ("rw" if prompt_cache_file is not None else "off")
            )
            parsed["prompt_cache_file"] = str(prompt_cache_file) if prompt_cache_file is not None else ""
            parsed["prompt_cache_build_elapsed_ms"] = prompt_cache_build_elapsed_ms
            parsed["prompt_cache_build_return_code"] = prompt_cache_build_rc
            parsed["prompt_cache_isolated"] = prompt_cache_file is not None
            dispatch_audit = _build_dispatch_audit(
                merged_text=merged,
                resources_path=resources_path,
                runtime_env=runtime_env,
                dispatch_audit_path=dispatch_audit_path,
                candidate_resources_expected=effective_candidate_resources_expected,
            )
            parsed["dispatch_audit"] = dispatch_audit
            parsed["dispatch_rule_id"] = str(dispatch_audit.get("selected_dispatch_rule_id", ""))
            parsed["metallib_path"] = str(dispatch_audit.get("metallib_path", ""))
            parsed["metallib_present"] = bool(dispatch_audit.get("metallib_present", False))
            parsed["metallib_source"] = str(dispatch_audit.get("metallib_source", ""))
            parsed["dispatch_audit_status"] = str(dispatch_audit.get("dispatch_audit_status", ""))
            parsed["candidate_resources_expected"] = bool(
                dispatch_audit.get("candidate_resources_expected", False)
            )
            parsed["candidate_resources_used"] = bool(
                dispatch_audit.get("candidate_resources_used", False)
            )
            parsed["dispatch_audit_path"] = str(dispatch_audit.get("dispatch_audit_path", ""))
            parsed["dispatch_audit_source"] = str(dispatch_audit.get("dispatch_audit_source", ""))
            parsed["top_dispatched_kernels"] = list(dispatch_audit.get("top_kernels") or [])
            if profile.name.startswith("long") and profile.prompt_tokens_target > 0:
                actual = parsed.get("prompt_tokens_actual")
                met_target = False
                if isinstance(actual, (int, float)):
                    met_target = float(actual) + float(long_prompt_token_tolerance) >= float(
                        profile.prompt_tokens_target
                    )
                parsed["prompt_tokens_target_met"] = met_target
                parsed["prompt_tokens_target_tolerance"] = int(long_prompt_token_tolerance)
                if enforce_long_prompt_target and not met_target:
                    parsed["validation_error"] = "long_prompt_target_miss"
            if capture_raw_output:
                parsed["stdout"] = proc.stdout or ""
                parsed["stderr"] = proc.stderr or ""
                parsed["prompt"] = prompt
            runs.append(parsed)
    finally:
        if temp_cli_root is not None:
            shutil.rmtree(temp_cli_root, ignore_errors=True)

    elapsed = time.perf_counter() - start_t
    return _aggregate(profile, runs, elapsed)


def run_benchmarks(
    *,
    llama_cli: Path,
    model_path: Path,
    profile_mode: str,
    gate_mode: str,
    prompts: list[str],
    resources_path: Path | None = None,
    extra_args: list[str] | None = None,
    capture_raw_output: bool = False,
    force_source_compile: bool = False,
    enforce_long_prompt_target: bool = False,
    long_prompt_token_tolerance: int = 0,
    prompt_cache_path: Path | None = None,
    prompt_cache_ro: bool = False,
    prompt_cache_all: bool = False,
    build_prompt_cache_first: bool = False,
    dispatch_attempt_id: str = "",
    dispatch_rule_id: str = "",
    dispatch_audit_dir: Path | None = None,
    candidate_resources_expected: bool = False,
) -> list[BenchmarkResult]:
    profiles = workload_profiles(profile_mode, gate_mode)
    out: list[BenchmarkResult] = []
    for profile in profiles:
        out.append(
            run_profile_benchmark(
                llama_cli=llama_cli,
                model_path=model_path,
                profile=profile,
                prompts=prompts,
                resources_path=resources_path,
                extra_args=extra_args,
                capture_raw_output=capture_raw_output,
                force_source_compile=force_source_compile,
                enforce_long_prompt_target=enforce_long_prompt_target,
                long_prompt_token_tolerance=long_prompt_token_tolerance,
                prompt_cache_path=prompt_cache_path,
                prompt_cache_ro=prompt_cache_ro,
                prompt_cache_all=prompt_cache_all,
                build_prompt_cache_first=build_prompt_cache_first,
                dispatch_attempt_id=dispatch_attempt_id,
                dispatch_rule_id=dispatch_rule_id,
                dispatch_audit_dir=dispatch_audit_dir,
                candidate_resources_expected=candidate_resources_expected,
            )
        )
    return out


def benchmark_results_to_dict(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]
