#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apple_silicon import constants
from src.apple_silicon.benchmark import resolve_llama_cli, resolve_metal_toolchain_paths
from src.apple_silicon.compat import ensure_llamacpp_commit, get_llamacpp_commit
from src.apple_silicon.device_probe import probe_device
from src.apple_silicon.feasibility import derive_allowed_params
from src.apple_silicon.model_probe import probe_model
from src.apple_silicon.model_store import ensure_default_model
from src.apple_silicon.op_profile import resolve_backend_filter, resolve_test_backend_ops
from src.apple_silicon.optimize import optimize_for_apple_silicon
from src.apple_silicon.runtime_args import sanitize_runtime_args
from src.apple_silicon.study import StudyError, generate_schedule_preview, run_validation_study
from src.apple_silicon.pack import (
    create_pack,
    disable_pack,
    export_pack,
    get_active_pack,
    set_active_pack,
)
from src.apple_silicon.runner import run_with_optional_pack
from src.auth.codex_capabilities import inspect_codex_cli
from src.auth.credentials import apply_auth_env, resolve_auth
from src.config import apply_llm_config, load_config_data


def _llamacpp_root(args: argparse.Namespace) -> Path:
    return Path(args.llamacpp_root).expanduser().resolve()


def _project_report_path(project: str) -> Path:
    return REPO_ROOT / "projects" / project / "benchmarks" / "apple_silicon_report.json"


def _write_report_if_project(project: str | None, payload: dict[str, Any]) -> None:
    if not project:
        return
    out = _project_report_path(project)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_profiles_for_set(profile_set: str, raw_profiles: str) -> list[str]:
    mode = (profile_set or "mixed").strip().lower()
    explicit = [x.strip() for x in str(raw_profiles or "").split(",") if x.strip()]
    if mode == "smoke":
        return ["chat", "long_smoke"]
    if mode == "claim":
        return ["chat", "long_claim"]
    if explicit:
        return explicit
    return ["chat", "long"]


def cmd_doctor(args: argparse.Namespace) -> int:
    device = probe_device()
    ll_root = _llamacpp_root(args)
    llm_applied = apply_llm_config()
    cfg, _ = load_config_data()
    auth_status = resolve_auth(
        config=cfg,
        env=dict(os.environ),
        runtime_context={"in_container": bool(os.environ.get("CGINS_PROJECT_DIR"))},
    )
    apply_auth_env(auth_status, os.environ)
    codex_caps = inspect_codex_cli()

    commit_ok, commit_msg = ensure_llamacpp_commit(ll_root, strict=False)
    llama_cli_exists = False
    llama_cli_path = ""
    test_backend_ops_path = ""
    test_backend_ops_exists = False
    try:
        llama_cli = resolve_llama_cli(ll_root)
        llama_cli_exists = llama_cli.exists()
        llama_cli_path = str(llama_cli)
    except Exception:
        pass
    try:
        test_backend_ops = resolve_test_backend_ops(ll_root)
        if test_backend_ops is not None:
            test_backend_ops_path = str(test_backend_ops)
            test_backend_ops_exists = test_backend_ops.exists()
    except Exception:
        pass

    result = {
        "success": True,
        "device": asdict(device),
        "allowed_params": asdict(derive_allowed_params(device)),
        "llamacpp_root": str(ll_root),
        "llamacpp_commit": get_llamacpp_commit(ll_root),
        "llamacpp_commit_ok": commit_ok,
        "llamacpp_commit_message": commit_msg,
        "llama_cli": llama_cli_path,
        "llama_cli_exists": llama_cli_exists,
        "test_backend_ops": test_backend_ops_path,
        "test_backend_ops_exists": test_backend_ops_exists,
        "default_model_cache": str(constants.MODELS_CACHE_DIR),
        "packs_cache": str(constants.PACKS_CACHE_DIR),
        "llm_config_applied": llm_applied,
        "llm_provider": os.environ.get("LLM_PROVIDER", ""),
        "metal_toolchain": resolve_metal_toolchain_paths(),
        "auth": auth_status.to_dict(),
        "codex_capabilities": {
            "binary_found": codex_caps.binary_found,
            "version": codex_caps.version,
            "supports_exec_subcommand": codex_caps.supports_exec_subcommand,
            "supports_login_subcommand": codex_caps.supports_login_subcommand,
            "supports_prompt_mode": codex_caps.supports_prompt_mode,
            "reason": codex_caps.reason,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    cfg, _ = load_config_data()
    auth_status = resolve_auth(config=cfg, env=dict(os.environ), runtime_context={"in_container": False})
    apply_auth_env(auth_status, os.environ)
    if auth_status.mode_effective == "unconfigured":
        payload = {"success": False, "error": f"No usable auth configured: {auth_status.reason}"}
        _write_report_if_project(args.project, payload)
        print(json.dumps(payload, indent=2))
        return 1
    try:
        report = optimize_for_apple_silicon(
            model_path=args.model,
            profile_mode=args.profile,
            gate_mode=("full" if args.full else "quick"),
            prompts_path=args.prompts,
            llamacpp_root=_llamacpp_root(args),
            cgins_version="v1",
            strict_commit=True,
            attempt_budget=args.attempt_budget if args.attempt_budget > 0 else None,
            study_tag=args.study_tag,
            emit_attempt_log=args.emit_attempt_log,
            kernel_mode=args.kernel_mode,
            candidate_cache_dir=args.candidate_cache_dir,
            strict_parity=bool(getattr(args, "strict_parity", False)),
            reuse_policy=args.reuse_policy,
            profile_set="mixed",
            profiling_mode="heuristic",
            stage0_feasibility=True,
            stage1_op_sql=False,
            stage2_logits_gate=True,
        )
    except Exception as exc:
        payload = {"success": False, "error": str(exc)}
        _write_report_if_project(args.project, payload)
        print(json.dumps(payload, indent=2))
        return 1

    payload = asdict(report)
    _write_report_if_project(args.project, payload)

    print(json.dumps(payload, indent=2))

    # Signed delta summary for UX.
    deltas = payload.get("delta", {}).get("profiles", {})
    summary = []
    for name, row in deltas.items():
        p = row.get("prefill_uplift_pct")
        d = row.get("decode_uplift_pct")
        summary.append(f"{name}: prefill={p if p is not None else 'n/a'}% decode={d if d is not None else 'n/a'}%")
    if summary:
        print("\nDelta summary:")
        for line in summary:
            print(f"- {line}")
    print(f"\nPass gate: {'yes' if payload.get('pass_gate') else 'no'}")
    return 0


def cmd_optimize_kernels(args: argparse.Namespace) -> int:
    cfg, _ = load_config_data()
    auth_status = resolve_auth(config=cfg, env=dict(os.environ), runtime_context={"in_container": False})
    apply_auth_env(auth_status, os.environ)
    if auth_status.mode_effective == "unconfigured":
        print(json.dumps({"success": False, "error": f"No usable auth configured: {auth_status.reason}"}, indent=2))
        return 1
    _normalize_bool_stage_flags(args)
    gate_mode = args.stage
    profile_set = _normalize_profile_set(args.profile_set)
    budget = int(args.budget) if int(args.budget) > 0 else None
    try:
        report = optimize_for_apple_silicon(
            model_path=args.model,
            profile_mode=args.profile,
            gate_mode=gate_mode,
            prompts_path=args.prompts,
            llamacpp_root=_llamacpp_root(args),
            cgins_version="v1",
            strict_commit=True,
            attempt_budget=budget,
            study_tag=args.study_tag,
            emit_attempt_log=args.attempt_log,
            kernel_mode=args.kernel_mode,
            candidate_cache_dir=args.candidate_cache_dir,
            strict_parity=bool(args.strict_parity),
            reuse_policy=args.reuse_policy,
            profile_set=profile_set,
            profiling_mode=args.profiling_mode,
            stage0_feasibility=bool(args.stage0_feasibility),
            stage1_op_sql=bool(args.stage1_op_sql),
            stage2_logits_gate=bool(args.stage2_logits_gate),
            op_perf_timeout_sec=float(args.op_perf_timeout_sec),
            op_perf_cache=str(args.op_perf_cache),
            op_perf_min_rows=int(args.op_perf_min_rows),
            op_perf_op_filter=str(args.op_perf_op_filter),
            op_perf_case_limit=int(args.op_perf_case_limit),
            op_perf_case_seed=int(args.op_perf_case_seed),
            op_perf_warmup_iters=int(args.op_perf_warmup_iters),
            op_perf_bench_iters=int(args.op_perf_bench_iters),
            op_perf_reject_regression_pct=float(args.op_perf_reject_regression_pct),
            op_perf_promote_topk=int(args.op_perf_promote_topk),
            op_test_timeout_sec=float(args.op_test_timeout_sec),
            op_test_cache=str(args.op_test_cache),
            op_test_min_rows=int(args.op_test_min_rows),
            op_test_case_limit=int(args.op_test_case_limit),
            op_test_case_seed=int(args.op_test_case_seed),
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(asdict(report), indent=2))
    return 0


def cmd_build_pack(args: argparse.Namespace) -> int:
    ll_root = _llamacpp_root(args)
    commit_ok, commit_msg = ensure_llamacpp_commit(ll_root, strict=True)
    if not commit_ok:
        print(json.dumps({"success": False, "error": commit_msg}, indent=2))
        return 1

    model = probe_model(_resolve_model_or_default(args.model))
    device = probe_device()

    src = Path(args.from_candidate).expanduser().resolve()
    if (src / "resources").exists():
        resources_dir = src / "resources"
        candidate_manifest = src / "candidate_manifest.json"
    elif src.name == "resources":
        resources_dir = src
        candidate_manifest = src.parent / "candidate_manifest.json"
    else:
        print(json.dumps({"success": False, "error": f"Invalid candidate path: {src}"}, indent=2))
        return 1

    if not resources_dir.exists():
        print(json.dumps({"success": False, "error": f"Resources directory missing: {resources_dir}"}, indent=2))
        return 1

    kernel_patch_metadata: dict[str, Any] = {}
    if candidate_manifest.exists():
        try:
            cm = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            kernel_patch_metadata = {
                "template_version": cm.get("template_version", ""),
                "patch_hash": cm.get("patch_hash", ""),
                "source_hash": cm.get("source_hash", ""),
                "template_mutations": cm.get("template_mutations", {}),
                "source_patches": cm.get("source_patches", []),
            }
        except Exception:
            kernel_patch_metadata = {}

    runtime_args = sanitize_runtime_args(shlex.split(args.runtime_args or ""))
    llama_commit = get_llamacpp_commit(ll_root)
    pack_id, pack_dir, manifest = create_pack(
        llamacpp_root=ll_root,
        device=device,
        model=model,
        profile_mode=args.profile,
        gate_mode=args.stage,
        cgins_version="v1",
        llamacpp_commit=llama_commit,
        bench_before={"results": []},
        bench_after={"results": []},
        kernel_overrides={},
        runtime_args=runtime_args,
        strict_guardrails={
            "min_primary_uplift_pct": constants.PASS_PRIMARY_UPLIFT_PCT,
            "max_allowed_regression_pct": constants.PASS_MAX_REGRESSION_PCT,
            "correctness": {"strict_parity": bool(args.strict_parity)},
        },
        resources_source_dir=resources_dir,
        kernel_patch_metadata=kernel_patch_metadata,
        tuning_session={
            "source": "build-pack",
            "candidate_path": str(src),
        },
        reuse_policy=args.reuse_policy,
        os_compat_override=args.os_compat,
    )

    if args.activate:
        set_active_pack(model_sha=model.sha256, device_fingerprint=device.fingerprint, pack_id=pack_id)

    payload = {
        "success": True,
        "pack_id": pack_id,
        "pack_dir": str(pack_dir),
        "activated": bool(args.activate),
        "manifest": manifest,
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_validate_study(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    resolved_cache_root = constants.configure_cache_root(args.cache_root)
    profiles = _resolve_profiles_for_set(args.profile_set, args.profiles)
    if args.dry_run_schedule:
        try:
            preview = generate_schedule_preview(
                matrix_path=Path(args.matrix).expanduser().resolve(),
                profiles=profiles,
                arms=[x.strip() for x in args.arms.split(",") if x.strip()],
                abba_cycles=int(args.abba_cycles if int(args.abba_blocks) <= 0 else args.abba_blocks),
                warmup_blocks=int(args.warmup_blocks),
            )
        except StudyError as exc:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2))
            return 1
        except Exception as exc:
            print(json.dumps({"success": False, "error": f"Unexpected error: {exc}"}, indent=2))
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        schedule_path = out_dir / "schedule.json"
        schedule_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(json.dumps({"success": True, "schedule_path": str(schedule_path), "schedule": preview}, indent=2))
        return 0

    ll_root = _llamacpp_root(args)
    try:
        parity_stage = str(args.parity_stage).strip().lower()
        if bool(getattr(args, "strict_parity", False)) and parity_stage != "claim":
            parity_stage = "claim"
        summary = run_validation_study(
            matrix_path=Path(args.matrix).expanduser().resolve(),
            output_dir=out_dir,
            profiles=profiles,
            arms=[x.strip() for x in args.arms.split(",") if x.strip()],
            llamacpp_root=ll_root,
            gate_mode=args.gate_mode,
            cooldown_seconds=float(args.cooldown_sec),
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed),
            require_ac_power=bool(args.strict_power or (not args.allow_battery)),
            strict_commit=bool(not args.allow_unpinned_llamacpp),
            resume=bool(args.resume),
            cache_root=resolved_cache_root,
            kernel_mode=args.kernel_mode,
            kernel_total_budget=int(args.kernel_total_budget),
            candidate_cache_dir=Path(args.candidate_cache_dir).expanduser().resolve() if args.candidate_cache_dir else None,
            attempt_log_path=Path(args.attempt_log).expanduser().resolve() if args.attempt_log else None,
            abba_cycles=int(args.abba_cycles),
            abba_blocks_legacy=int(args.abba_blocks),
            warmup_blocks=int(args.warmup_blocks),
            strict_parity=bool(args.strict_parity),
            parity_stage=parity_stage,
            profiling_mode=str(args.profiling_mode).strip().lower(),
            long_token_tolerance=int(args.long_token_tolerance),
            decode_claim_threshold_pct=float(args.decode_claim_threshold_pct),
            op_perf_timeout_sec=float(args.op_perf_timeout_sec),
            op_perf_cache=str(args.op_perf_cache),
            op_perf_min_rows=int(args.op_perf_min_rows),
            op_perf_op_filter=str(args.op_perf_op_filter),
            op_perf_case_limit=int(args.op_perf_case_limit),
            op_perf_case_seed=int(args.op_perf_case_seed),
            op_perf_warmup_iters=int(args.op_perf_warmup_iters),
            op_perf_bench_iters=int(args.op_perf_bench_iters),
            op_perf_reject_regression_pct=float(args.op_perf_reject_regression_pct),
            op_perf_promote_topk=int(args.op_perf_promote_topk),
            op_test_timeout_sec=float(args.op_test_timeout_sec),
            op_test_cache=str(args.op_test_cache),
            op_test_min_rows=int(args.op_test_min_rows),
            op_test_case_limit=int(args.op_test_case_limit),
            op_test_case_seed=int(args.op_test_case_seed),
        )
    except StudyError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"Unexpected error: {exc}"}, indent=2))
        return 1

    payload = {"success": True, "summary": asdict(summary)}
    print(json.dumps(payload, indent=2))
    return 0


def cmd_debug_candidate(args: argparse.Namespace) -> int:
    ll_root = _llamacpp_root(args)
    bin_path = resolve_test_backend_ops(ll_root)
    if bin_path is None:
        print(json.dumps({"success": False, "error": "test-backend-ops binary not found"}, indent=2))
        return 1

    mode = str(args.mode).strip().lower()
    if mode not in {"perf", "test"}:
        print(json.dumps({"success": False, "error": "--mode must be perf or test"}, indent=2))
        return 2
    op = str(args.op or "").strip()
    if not op:
        print(json.dumps({"success": False, "error": "--op is required"}, indent=2))
        return 2

    backend_resolution = resolve_backend_filter(
        llamacpp_root=ll_root,
        requested_backend="Metal",
    )
    resolved_backend = str(backend_resolution.get("resolved_backend") or "Metal").strip()
    available_backends = [
        str(x).strip() for x in (backend_resolution.get("backend_names") or []) if str(x).strip()
    ]
    if not any(resolved_backend.lower() == b.lower() for b in available_backends):
        payload = {
            "success": False,
            "status": "backend_unavailable",
            "error": f"Resolved backend '{resolved_backend}' is not available on this host",
            "backend_requested": "Metal",
            "backend_resolution": backend_resolution,
            "available_backends": available_backends,
        }
        print(json.dumps(payload, indent=2))
        return 1

    cmd = [str(bin_path), mode, "-b", resolved_backend]
    cmd.extend(["-o", op])

    env = dict(os.environ)
    env["MTL_DEBUG_LAYER"] = "1"
    env["MTL_SHADER_VALIDATION"] = "1"
    env["MTL_SHADER_VALIDATION_REPORT_TO_STDERR"] = "1"
    env["MTL_SHADER_VALIDATION_FAIL_MODE"] = "allow"
    env["MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS"] = "1"
    env["MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS"] = "1"
    if args.resources_dir:
        env["GGML_METAL_PATH_RESOURCES"] = str(Path(args.resources_dir).expanduser().resolve())

    print(f"[debug-candidate] command: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=max(1.0, float(args.timeout_sec)),
        )
        return_code = int(proc.returncode)
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout_text = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr_text = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stdout_hash = hashlib.sha256(stdout_text.encode("utf-8")).hexdigest() if stdout_text else ""
    stderr_hash = hashlib.sha256(stderr_text.encode("utf-8")).hexdigest() if stderr_text else ""
    validation_env = {
        key: env[key]
        for key in [
            "MTL_DEBUG_LAYER",
            "MTL_SHADER_VALIDATION",
            "MTL_SHADER_VALIDATION_REPORT_TO_STDERR",
            "MTL_SHADER_VALIDATION_FAIL_MODE",
            "MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS",
            "MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS",
            "GGML_METAL_PATH_RESOURCES",
        ]
        if key in env
    }
    debug_run_id = f"dbg_{int(time.time())}_{stderr_hash[:8] if stderr_hash else 'ok'}"
    status = "ok" if return_code == 0 else ("timeout" if timed_out else "failed")
    payload = {
        "success": return_code == 0,
        "status": status,
        "return_code": return_code,
        "command": " ".join(shlex.quote(c) for c in cmd),
        "mode": mode,
        "op": op,
        "backend_requested": "Metal",
        "backend": resolved_backend,
        "backend_resolution": backend_resolution,
        "debug_run_id": debug_run_id,
        "resources_dir": str(Path(args.resources_dir).expanduser().resolve()) if args.resources_dir else "",
        "validation_env": validation_env,
        "stdout_hash": stdout_hash,
        "stderr_hash": stderr_hash,
        "elapsed_ms": elapsed_ms,
        "timeout_sec": float(args.timeout_sec),
        "attempt_id": str(args.attempt_id or "").strip(),
        "stdout": stdout_text,
        "stderr": stderr_text,
    }
    emit_path = str(args.emit_debug_record or "").strip()
    if emit_path:
        record_path = Path(emit_path).expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
        payload["emit_debug_record"] = str(record_path)
    if args.emit_attempt_log:
        attempt_path = Path(args.emit_attempt_log).expanduser().resolve()
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        with attempt_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"debug_candidate": payload}, ensure_ascii=False))
            f.write("\n")
        payload["emit_attempt_log"] = str(attempt_path)
    if args.study_out:
        study_out = Path(args.study_out).expanduser().resolve()
        study_out.mkdir(parents=True, exist_ok=True)
        debug_runs = study_out / "debug_candidate_runs.jsonl"
        with debug_runs.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
        payload["study_debug_log"] = str(debug_runs)
    payload["wall_clock_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(payload, indent=2))
    return 0 if return_code == 0 else 1


def _normalize_bool_stage_flags(args: argparse.Namespace) -> None:
    # Explicit --no-* flags override default-on behavior.
    if not hasattr(args, "stage0_feasibility"):
        return
    args.stage0_feasibility = bool(args.stage0_feasibility)
    args.stage1_op_sql = bool(args.stage1_op_sql)
    args.stage2_logits_gate = bool(args.stage2_logits_gate)


def _normalize_profile_set(value: str) -> str:
    mode = str(value or "mixed").strip().lower()
    if mode not in {"smoke", "claim", "mixed"}:
        return "mixed"
    return mode


def _resolve_model_or_default(path_value: str | None) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    return ensure_default_model()


def cmd_run(args: argparse.Namespace, passthrough: list[str]) -> int:
    ll_root = _llamacpp_root(args)
    llama_cli = resolve_llama_cli(ll_root)
    model_path = _resolve_model_or_default(args.model)
    model = probe_model(model_path)
    device = probe_device()
    commit = get_llamacpp_commit(ll_root)

    pack_id = get_active_pack(model_sha=model.sha256, device_fingerprint=device.fingerprint)

    extra_args = passthrough
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    return run_with_optional_pack(
        llama_cli=llama_cli,
        model=model,
        extra_args=extra_args,
        pack_id=pack_id,
        device=device,
        llamacpp_commit=commit,
    )


def cmd_export_pack(args: argparse.Namespace) -> int:
    model = probe_model(_resolve_model_or_default(args.model))
    device = probe_device()
    pack_id = get_active_pack(model_sha=model.sha256, device_fingerprint=device.fingerprint)
    if not pack_id:
        print(json.dumps({"success": False, "error": "No active pack for model/device"}, indent=2))
        return 1

    pack_dir = constants.PACKS_CACHE_DIR / pack_id
    out_path = Path(args.out).expanduser().resolve()
    if out_path.suffix.lower() != ".cginspack":
        out_path = out_path.with_suffix(".cginspack")

    export_pack(pack_dir, out_path)
    print(json.dumps({"success": True, "pack_id": pack_id, "path": str(out_path)}, indent=2))
    return 0


def cmd_disable_pack(args: argparse.Namespace) -> int:
    model = probe_model(_resolve_model_or_default(args.model))
    device = probe_device()
    disable_pack(model_sha=model.sha256, device_fingerprint=device.fingerprint)
    print(json.dumps({"success": True, "model_sha256": model.sha256, "device": device.fingerprint}, indent=2))
    return 0


def cmd_torch_optimize(args: argparse.Namespace) -> int:
    apply_llm_config()
    project = (args.project or "").strip()
    if not project:
        print(json.dumps({"success": False, "error": "--project is required"}, indent=2))
        return 2

    io_dir = (
        Path(args.io_dir).expanduser().resolve()
        if args.io_dir
        else (REPO_ROOT / "projects" / project / "io" / "individual_ops")
    )
    kernel_dir = (
        Path(args.kernel_dir).expanduser().resolve()
        if args.kernel_dir
        else (REPO_ROOT / "projects" / project / "kernels" / "generated" / "individual_op_kernels")
    )

    if not io_dir.exists():
        print(json.dumps({"success": False, "error": f"io dir missing: {io_dir}"}, indent=2))
        return 1
    if not kernel_dir.exists():
        print(json.dumps({"success": False, "error": f"kernel dir missing: {kernel_dir}"}, indent=2))
        return 1

    env = dict(os.environ)
    env["CGINS_TARGET_DEVICE"] = "mps"
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.optimize_ops",
        str(io_dir),
        project,
        "--kernel-dir",
        str(kernel_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode == 0:
        print(
            json.dumps(
                {
                    "success": True,
                    "project": project,
                    "target_device": "mps",
                    "io_dir": str(io_dir),
                    "kernel_dir": str(kernel_dir),
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "success": False,
                    "project": project,
                    "target_device": "mps",
                    "return_code": proc.returncode,
                },
                indent=2,
            )
        )
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cgins-as", description="CGinS Apple Silicon llama.cpp optimizer")
    parser.add_argument(
        "--llamacpp-root",
        default=str(REPO_ROOT / ".vendor" / "llama.cpp"),
        help="Path to pinned llama.cpp checkout",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check environment readiness")

    opt = sub.add_parser("optimize", help="Benchmark and build Apple Silicon kernel pack")
    opt.add_argument("--model", default="", help="Path to GGUF model (default: pinned tiny Qwen download)")
    opt.add_argument("--profile", choices=["chat", "long", "both"], default="both")
    opt.add_argument("--prompts", default="", help="Optional newline-delimited prompt file")
    opt.add_argument("--quick", action="store_true", help="Use quick benchmark gate (default)")
    opt.add_argument("--full", action="store_true", help="Use full benchmark gate")
    opt.add_argument("--project", default="", help="Project name to persist apple_silicon_report.json")
    opt.add_argument("--attempt-budget", type=int, default=0, help="Override LLM tuning attempt budget")
    opt.add_argument("--study-tag", default="", help="Optional tag to include in tuning artifacts")
    opt.add_argument(
        "--kernel-mode",
        choices=["none", "oneshot", "iterative"],
        default="none",
        help="Kernel candidate strategy: none (runtime args only) or kernel-source tuning modes",
    )
    opt.add_argument(
        "--candidate-cache-dir",
        default="",
        help="Directory for temporary kernel candidate resources",
    )
    opt.add_argument(
        "--emit-attempt-log",
        default="",
        help="Optional JSONL path to append per-attempt tuning records",
    )
    opt.add_argument(
        "--strict-parity",
        action="store_true",
        help="Enable strict semantic parity checks for candidate correctness",
    )
    opt.add_argument(
        "--reuse-policy",
        choices=["machine", "chip_family", "chip_family+os_minor"],
        default="chip_family",
        help="Pack reuse policy for selected optimized candidate",
    )

    optk = sub.add_parser(
        "optimize-kernels",
        help="Run kernel-focused optimization with explicit budget/stage controls",
    )
    optk.add_argument("--model", default="", help="Path to GGUF model (default: pinned tiny Qwen download)")
    optk.add_argument("--profile", choices=["chat", "long", "long_smoke", "long_claim", "both"], default="both")
    optk.add_argument(
        "--profile-set",
        choices=["smoke", "claim", "mixed"],
        default="mixed",
        help="Profile mapping for staged runs: smoke=chat+long_smoke, claim=chat+long_claim, mixed=use --profile",
    )
    optk.add_argument("--prompts", default="", help="Optional newline-delimited prompt file")
    optk.add_argument("--budget", type=int, default=120, help="Candidate evaluation budget")
    optk.add_argument("--stage", choices=["quick", "full"], default="full")
    optk.add_argument(
        "--profiling-mode",
        choices=["op_perf_required", "heuristic"],
        default="op_perf_required",
        help="Stage-1 profiling mode for candidate pruning",
    )
    optk.add_argument(
        "--stage0-feasibility",
        dest="stage0_feasibility",
        action="store_true",
        default=True,
        help="Enable static feasibility filter before compile (default: on)",
    )
    optk.add_argument(
        "--no-stage0-feasibility",
        dest="stage0_feasibility",
        action="store_false",
        help="Disable static feasibility filter",
    )
    optk.add_argument(
        "--stage1-op-sql",
        dest="stage1_op_sql",
        action="store_true",
        default=True,
        help="Enable SQL-based test-backend-ops stage-1 profiling (default: on)",
    )
    optk.add_argument(
        "--no-stage1-op-sql",
        dest="stage1_op_sql",
        action="store_false",
        help="Disable SQL op profiling stage",
    )
    optk.add_argument(
        "--stage2-logits-gate",
        dest="stage2_logits_gate",
        action="store_true",
        default=True,
        help="Enable stage-2 numeric gate before full ABBA (default: on)",
    )
    optk.add_argument(
        "--no-stage2-logits-gate",
        dest="stage2_logits_gate",
        action="store_false",
        help="Disable stage-2 numeric gate",
    )
    optk.add_argument("--op-perf-timeout-sec", type=float, default=90.0, help="Stage-1 op perf timeout in seconds")
    optk.add_argument(
        "--op-perf-cache",
        choices=["on", "off", "refresh"],
        default="on",
        help="Stage-1 op perf cache mode",
    )
    optk.add_argument("--op-perf-min-rows", type=int, default=1, help="Minimum SQL rows required for Stage-1 success")
    optk.add_argument("--op-perf-op-filter", default="MUL_MAT", help="Stage-1 op filter passed to test-backend-ops -o")
    optk.add_argument("--op-perf-case-limit", type=int, default=64, help="Stage-1 case limit (requires patched test-backend-ops)")
    optk.add_argument("--op-perf-case-seed", type=int, default=0, help="Stage-1 deterministic case selection seed (0 disables shuffling)")
    optk.add_argument("--op-perf-warmup-iters", type=int, default=1, help="Stage-1 warmup iterations (requires patched test-backend-ops)")
    optk.add_argument("--op-perf-bench-iters", type=int, default=3, help="Stage-1 benchmark iterations (requires patched test-backend-ops)")
    optk.add_argument(
        "--op-perf-reject-regression-pct",
        type=float,
        default=10.0,
        help="Reject candidate when Stage-1 delta drops below this negative percent threshold",
    )
    optk.add_argument(
        "--op-perf-promote-topk",
        type=int,
        default=3,
        help="Allow up to K hard-regression candidates through Stage-1 for downstream validation",
    )
    optk.add_argument("--op-test-timeout-sec", type=float, default=45.0, help="Gate-B op correctness timeout in seconds")
    optk.add_argument(
        "--op-test-cache",
        choices=["on", "off", "refresh"],
        default="on",
        help="Gate-B op correctness cache mode",
    )
    optk.add_argument("--op-test-min-rows", type=int, default=1, help="Minimum SQL rows required for Gate-B success")
    optk.add_argument("--op-test-case-limit", type=int, default=32, help="Gate-B test case limit (requires patched test-backend-ops)")
    optk.add_argument("--op-test-case-seed", type=int, default=0, help="Gate-B deterministic case selection seed (0 disables shuffling)")
    optk.add_argument(
        "--kernel-mode",
        choices=["oneshot", "iterative"],
        default="iterative",
        help="Kernel candidate strategy",
    )
    optk.add_argument("--candidate-cache-dir", default="", help="Candidate cache directory")
    optk.add_argument("--attempt-log", default="", help="Optional JSONL path for attempts")
    optk.add_argument("--study-tag", default="", help="Optional run tag for provenance")
    optk.add_argument(
        "--strict-parity",
        action="store_true",
        help="Enable strict semantic parity checks for candidate correctness",
    )
    optk.add_argument(
        "--reuse-policy",
        choices=["machine", "chip_family", "chip_family+os_minor"],
        default="chip_family",
        help="Pack reuse policy for selected optimized candidate",
    )

    run = sub.add_parser("run", help="Run llama-cli with active compatible pack + fallback")
    run.add_argument("--model", default="", help="Path to GGUF model (default: pinned tiny Qwen download)")

    bpack = sub.add_parser("build-pack", help="Build reusable pack from an evaluated candidate resources dir")
    bpack.add_argument("--model", default="", help="Path to GGUF model (default: pinned tiny Qwen download)")
    bpack.add_argument("--from-candidate", required=True, help="Path to candidate dir or resources dir")
    bpack.add_argument("--profile", choices=["chat", "long", "both"], default="both")
    bpack.add_argument("--stage", choices=["quick", "full"], default="full")
    bpack.add_argument(
        "--reuse-policy",
        choices=["machine", "chip_family", "chip_family+os_minor"],
        default="chip_family",
        help="Pack reuse policy",
    )
    bpack.add_argument("--os-compat", default="", help="Optional OS compatibility override (e.g., 26.2)")
    bpack.add_argument("--runtime-args", default="", help="Optional runtime args string to store in pack manifest")
    bpack.add_argument("--activate", action="store_true", help="Set built pack active for current model/device")
    bpack.add_argument("--strict-parity", action="store_true", help="Record strict parity policy in manifest guardrails")

    exp = sub.add_parser("export-pack", help="Export active pack for model/device")
    exp.add_argument("--model", default="", help="Path to GGUF model (default: pinned tiny Qwen download)")
    exp.add_argument("--out", required=True, help="Output .cginspack path")

    dis = sub.add_parser("disable-pack", help="Disable active pack for model/device")
    dis.add_argument("--model", default="", help="Path to GGUF model (default: pinned tiny Qwen download)")

    torch_opt = sub.add_parser(
        "torch-optimize",
        help="Run existing CGinS optimizer pipeline on Apple Silicon via PyTorch MPS",
    )
    torch_opt.add_argument("--project", required=True, help="Project name under projects/<name>")
    torch_opt.add_argument("--io-dir", default="", help="Optional explicit io/individual_ops path")
    torch_opt.add_argument("--kernel-dir", default="", help="Optional explicit generated kernels root")

    study = sub.add_parser(
        "validate-study",
        help="Run rigorous crossover study for Apple Silicon llama.cpp optimization",
    )
    study.add_argument("--matrix", required=True, help="JSON matrix file listing model paths/checksums")
    study.add_argument(
        "--profiles",
        default="chat,long",
        help="Comma-separated profile list (chat,long,long_smoke,long_claim)",
    )
    study.add_argument(
        "--profile-set",
        choices=["smoke", "claim", "mixed"],
        default="mixed",
        help="Override profile mapping for study runs",
    )
    study.add_argument(
        "--arms",
        default="baseline,flash,oneshot_kernel,iterative_kernel",
        help="Comma-separated arm list",
    )
    study.add_argument("--gate-mode", choices=["quick", "full"], default="full")
    study.add_argument("--cooldown-sec", type=float, default=5.0, help="Cooldown between crossover blocks")
    study.add_argument("--bootstrap-samples", type=int, default=10000, help="Bootstrap resamples for CI")
    study.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    study.add_argument("--abba-cycles", type=int, default=8, help="Number of ABBA cycles per arm (each cycle is ABBA)")
    study.add_argument("--abba-blocks", type=int, default=0, help="Deprecated alias; interpreted as abba-cycles when >0")
    study.add_argument("--warmup-blocks", type=int, default=2, help="Warmup blocks per arm/profile")
    study.add_argument("--dry-run-schedule", action="store_true", help="Only emit the exact schedule artifact and exit")
    study.add_argument(
        "--profiling-mode",
        choices=["op_perf_required", "heuristic"],
        default="op_perf_required",
        help="Hotspot attribution mode. op_perf_required blocks official runs if test-backend-ops perf is unavailable.",
    )
    study.add_argument(
        "--long-token-tolerance",
        type=int,
        default=128,
        help="Allowed shortfall in long profile prompt token target before invalidation",
    )
    study.add_argument(
        "--parity-stage",
        choices=["none", "numeric", "semantic", "claim"],
        default="numeric",
        help="Correctness gate stage: none/numeric/semantic/claim",
    )
    study.add_argument("--op-perf-timeout-sec", type=float, default=90.0, help="Stage-1 op perf timeout in seconds")
    study.add_argument(
        "--op-perf-cache",
        choices=["on", "off", "refresh"],
        default="on",
        help="Stage-1 op perf cache mode",
    )
    study.add_argument("--op-perf-min-rows", type=int, default=1, help="Minimum SQL rows required for Stage-1 success")
    study.add_argument("--op-perf-op-filter", default="MUL_MAT", help="Stage-1 op filter passed to test-backend-ops -o")
    study.add_argument("--op-perf-case-limit", type=int, default=64, help="Stage-1 case limit (requires patched test-backend-ops)")
    study.add_argument("--op-perf-case-seed", type=int, default=0, help="Stage-1 deterministic case selection seed (0 disables shuffling)")
    study.add_argument("--op-perf-warmup-iters", type=int, default=1, help="Stage-1 warmup iterations (requires patched test-backend-ops)")
    study.add_argument("--op-perf-bench-iters", type=int, default=3, help="Stage-1 benchmark iterations (requires patched test-backend-ops)")
    study.add_argument(
        "--op-perf-reject-regression-pct",
        type=float,
        default=10.0,
        help="Reject candidate when Stage-1 delta drops below this negative percent threshold",
    )
    study.add_argument(
        "--op-perf-promote-topk",
        type=int,
        default=3,
        help="Allow up to K hard-regression candidates through Stage-1 for downstream validation",
    )
    study.add_argument("--op-test-timeout-sec", type=float, default=45.0, help="Gate-B op correctness timeout in seconds")
    study.add_argument(
        "--op-test-cache",
        choices=["on", "off", "refresh"],
        default="on",
        help="Gate-B op correctness cache mode",
    )
    study.add_argument("--op-test-min-rows", type=int, default=1, help="Minimum SQL rows required for Gate-B success")
    study.add_argument("--op-test-case-limit", type=int, default=32, help="Gate-B test case limit (requires patched test-backend-ops)")
    study.add_argument("--op-test-case-seed", type=int, default=0, help="Gate-B deterministic case selection seed (0 disables shuffling)")
    study.add_argument("--out", required=True, help="Output directory for study artifacts")
    study.add_argument(
        "--kernel-mode",
        choices=["none", "oneshot", "iterative"],
        default="none",
        help="Kernel candidate mode used during study tuning arms",
    )
    study.add_argument(
        "--candidate-cache-dir",
        default="",
        help="Directory for temporary kernel candidate resources",
    )
    study.add_argument(
        "--kernel-total-budget",
        type=int,
        default=0,
        help="Optional total kernel attempt budget for study tuning (0 keeps default dynamic strategy)",
    )
    study.add_argument(
        "--cache-root",
        default="",
        help="Cache root override (precedence: --cache-root > CGINS_CACHE_ROOT > XDG_CACHE_HOME > ~/.cache)",
    )
    study.add_argument(
        "--attempt-log",
        default="",
        help="Optional JSONL path for study attempt-level records",
    )
    study.add_argument("--resume", action="store_true", help="Resume safely into a nested resume directory")
    study.add_argument("--strict-power", action="store_true", help="Require AC power and non-discharging state")
    study.add_argument("--strict-parity", action="store_true", help="Require strict semantic parity checks")
    study.add_argument("--decode-claim-threshold-pct", type=float, default=30.0, help="Decode CI lower-bound threshold for claims")
    study.add_argument("--allow-battery", action="store_true", help="Disable strict AC power requirement")
    study.add_argument(
        "--allow-unpinned-llamacpp",
        action="store_true",
        help="Allow study run when local llama.cpp commit differs from pinned commit",
    )

    dbg = sub.add_parser("debug-candidate", help="Run isolated Metal backend-op debug with shader validation")
    dbg.add_argument("--mode", choices=["perf", "test"], default="test", help="test-backend-ops mode")
    dbg.add_argument("--op", default="MUL_MAT", help="Optional operation filter (e.g., MUL_MAT)")
    dbg.add_argument("--resources-dir", default="", help="Optional candidate resources dir for GGML_METAL_PATH_RESOURCES")
    dbg.add_argument("--emit-debug-record", default="", help="Optional JSONL path for debug provenance records")
    dbg.add_argument("--emit-attempt-log", default="", help="Optional JSONL attempt log append path")
    dbg.add_argument("--timeout-sec", type=float, default=120.0, help="Subprocess timeout in seconds")
    dbg.add_argument("--attempt-id", default="", help="Optional related attempt_id for provenance linkage")
    dbg.add_argument("--study-out", default="", help="Optional study output root; appends to debug_candidate_runs.jsonl")

    return parser


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "optimize":
        if args.full and args.quick:
            print("Choose one of --quick or --full", file=sys.stderr)
            return 2
        return cmd_optimize(args)
    if args.command == "optimize-kernels":
        return cmd_optimize_kernels(args)
    if args.command == "run":
        return cmd_run(args, passthrough)
    if args.command == "build-pack":
        return cmd_build_pack(args)
    if args.command == "export-pack":
        return cmd_export_pack(args)
    if args.command == "disable-pack":
        return cmd_disable_pack(args)
    if args.command == "torch-optimize":
        return cmd_torch_optimize(args)
    if args.command == "validate-study":
        if args.strict_power and args.allow_battery:
            print("Choose one of --strict-power or --allow-battery", file=sys.stderr)
            return 2
        if int(getattr(args, "abba_blocks", 0)) > 0:
            print("Warning: --abba-blocks is deprecated; use --abba-cycles.", file=sys.stderr)
        return cmd_validate_study(args)
    if args.command == "debug-candidate":
        return cmd_debug_candidate(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
