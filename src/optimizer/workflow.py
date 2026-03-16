from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.progress import check_cancelled, update_job_progress, wait_if_paused
from src.optimizer.tree_store import publish_generated_root
from src.optimizer.pipeline import update_queue_state


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_dir(project: str) -> Path:
    return _repo_root() / "kernels" / "projects" / project


def _normalize_device(device: str) -> str:
    d = (device or "").strip().lower()
    valid_devices = {"cuda", "metal", "triton", "mps", "cpu"}
    if d not in valid_devices:
        raise ValueError(f"Unsupported target device: {d}")
    return d


def _parse_sm_to_capability(arch: str) -> tuple[int, int] | None:
    if not arch or not arch.startswith("sm_"):
        return None
    suffix = arch[3:]
    digits = ""
    for ch in suffix:
        if ch.isdigit():
            digits += ch
    if not digits:
        return None
    value = int(digits)
    if value <= 0:
        return None
    return value // 10, value % 10


def _max_supported_cuda_capability() -> tuple[int, int] | None:
    try:
        import torch
    except Exception:
        return None
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return None
    best: tuple[int, int] | None = None
    for arch in arch_list:
        capability = _parse_sm_to_capability(str(arch))
        if capability is None:
            continue
        if best is None or capability > best:
            best = capability
    return best


def _preflight_cuda_target() -> tuple[bool, str]:
    try:
        import torch
    except Exception as e:
        return False, f"CUDA preflight failed: unable to import torch ({e})."

    if not torch.cuda.is_available():
        return False, (
            "CUDA preflight failed: target device is CUDA but "
            "torch.cuda.is_available() returned False."
        )

    try:
        dev_idx = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        dev_cap = torch.cuda.get_device_capability(dev_idx)
    except Exception as e:
        return False, f"CUDA preflight failed: unable to query CUDA device ({e})."

    max_cap = _max_supported_cuda_capability()
    if max_cap is not None and dev_cap > max_cap:
        torch_cuda = getattr(torch.version, "cuda", None) or "unknown"
        torch_cuda_major = 0
        try:
            torch_cuda_major = int(str(torch_cuda).split(".")[0])
        except Exception:
            torch_cuda_major = 0

        forward_compatible = (
            dev_cap[0] == max_cap[0]
            and dev_cap[1] == (max_cap[1] + 1)
            and torch_cuda_major >= 12
        )
        if forward_compatible:
            return True, (
                "CUDA preflight note: detected GPU "
                f"'{dev_name}' capability {dev_cap[0]}.{dev_cap[1]} is one step newer "
                f"than this PyTorch build max {max_cap[0]}.{max_cap[1]}; "
                "continuing in compatibility mode."
            )

        return False, (
            "CUDA preflight failed: detected GPU "
            f"'{dev_name}' capability {dev_cap[0]}.{dev_cap[1]} exceeds this "
            f"PyTorch build maximum {max_cap[0]}.{max_cap[1]} "
            f"(torch CUDA {torch_cuda})."
        )

    return True, ""


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"[workflow] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def _ops_from_csv(raw: str) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for op in raw.split(","):
        name = str(op).strip()
        if name:
            out.append(name)
    return out


def _discover_ops(io_dir: Path) -> list[str]:
    if not io_dir.exists():
        return []
    ops: list[str] = []
    for child in sorted(io_dir.iterdir()):
        if child.is_dir():
            ops.append(child.name)
    return ops


def _has_success_marker(op_dir: Path) -> bool:
    markers = (
        op_dir / "success.cuda",
        op_dir / "success.triton",
        op_dir / "success.mps",
        op_dir / "success.cpu",
    )
    for marker in markers:
        if marker.exists():
            return True
    return False


def _load_kernel_benchmark(project_dir: Path, op_name: str) -> tuple[float | None, str]:
    bench_path = project_dir / "benchmarks" / "op_benchmarks.json"
    if not bench_path.exists():
        return None, ""
    try:
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
    except Exception:
        return None, ""

    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return None, ""

    for row in results:
        if not isinstance(row, dict):
            continue
        if str(row.get("op", "")) != op_name:
            continue
        if str(row.get("kernel_status", "")) != "ok":
            continue
        kernel_ms = row.get("kernel_ms")
        try:
            parsed_ms = float(kernel_ms)
        except Exception:
            return None, ""
        if parsed_ms <= 0.0:
            return None, ""
        return parsed_ms, str(row.get("backend", "") or "")
    return None, ""


def run_profile(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = dict(os.environ)
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.pipeline",
        "--project",
        args.project,
    ]
    if args.weights_b64_path:
        cmd += ["--weights-b64-path", args.weights_b64_path]
    if args.validation_b64_path:
        cmd += ["--validation-b64-path", args.validation_b64_path]
    if args.validation_name_path:
        cmd += ["--validation-name-path", args.validation_name_path]
    return _run(cmd, root, env)


def run_benchmark(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = dict(os.environ)
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.benchmark_ops",
        "--project",
        args.project,
    ]
    return _run(cmd, root, env)


def run_generate(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = dict(os.environ)

    if hasattr(args, "llm_provider") and args.llm_provider:
        env["LLM_PROVIDER"] = args.llm_provider
    if hasattr(args, "llm_model") and args.llm_model:
        provider = getattr(args, "llm_provider", "") or ""
        if provider == "openai":
            env["OPENAI_MODEL"] = args.llm_model
        elif provider == "anthropic":
            env["ANTHROPIC_MODEL"] = args.llm_model
        elif provider in ("google", "gemini"):
            env["GEMINI_MODEL"] = args.llm_model
        else:
            env["OPENAI_MODEL"] = args.llm_model

    target_device = _normalize_device(args.target_device)
    env["KFORGE_TARGET_DEVICE"] = target_device
    if target_device == "cuda":
        ok, reason = _preflight_cuda_target()
        if not ok:
            print(f"[workflow] {reason}")
            return 2
        if reason:
            print(f"[workflow] {reason}")

    project_dir = _project_dir(args.project)
    io_dir = project_dir / "io" / "individual_ops"
    out_dir = project_dir / "kernels" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = _ops_from_csv(args.ops)
    if not ops:
        ops = _discover_ops(io_dir)
    if not ops:
        print("[workflow] No operators discovered for generation.")
        return 1

    total_ops = len(ops)
    progress_total = total_ops
    update_job_progress(0, progress_total, "Starting kernel generation.")
    subprocess_env = dict(env)
    subprocess_env.pop("KFORGE_STATE_PATH", None)
    subprocess_env.pop("KFORGE_JOB_KEY", None)

    failed_ops: list[tuple[str, str]] = []

    for idx, op_name in enumerate(ops):
        if not wait_if_paused():
            return 130
        if check_cancelled():
            return 130

        update_job_progress(
            idx,
            progress_total,
            f"Generating kernel for {op_name} ({idx + 1}/{total_ops})",
        )

        generated_op_dir = out_dir / "individual_op_kernels" / op_name
        task_key = "gen_" + op_name
        task_meta = {
            "tag": "[GEN]",
            "op_name": op_name,
        }
        op_failure: str | None = None
        can_benchmark = False
        reused_existing = False

        # C3/C4: advance current_operator and shrink pending_operators in the queue
        update_queue_state(project_dir, {
            "current_operator": op_name,
            "pending_operators": ops[idx + 1:],
        })

        if _has_success_marker(generated_op_dir):
            reused_existing = True
            can_benchmark = True
            publish_result = publish_generated_root(
                project_dir,
                op_name,
                kernel_ms=None,
                backend=target_device,
                description="Generated baseline kernel (existing)",
            )
            if not publish_result.get("ok", False):
                print(
                    "[workflow] Warning: failed to publish existing root for "
                    f"{op_name}: {publish_result.get('reason', 'unknown')}"
                )
        else:
            stale_markers = (
                generated_op_dir / "success.cuda",
                generated_op_dir / "success.triton",
                generated_op_dir / "success.mps",
                generated_op_dir / "success.cpu",
            )
            for marker in stale_markers:
                try:
                    if marker.exists():
                        marker.unlink()
                except Exception:
                    pass

            gen_cmd = [
                sys.executable,
                "-m",
                "src.generator.main",
                "--io-dir",
                str(io_dir),
                "--out-dir",
                str(out_dir),
                "--only-ops",
                op_name,
            ]
            if args.remote:
                gen_cmd += ["--remote", args.remote]
            update_queue_state(project_dir, {"active_tasks": {task_key: {
                **task_meta,
                "current_step": "Generating",
                "status": "In Progress",
            }}})
            # Pass KFORGE state env through so attempt-level progress messages
            # from generator/main.py reach the dashboard.
            rc = _run(gen_cmd, root, env)
            if rc != 0:
                op_failure = f"generation command failed (exit {rc})"
            elif not _has_success_marker(generated_op_dir):
                op_failure = "kernel failed validation/compile (missing success marker)"
            else:
                can_benchmark = True
                publish_result = publish_generated_root(
                    project_dir,
                    op_name,
                    kernel_ms=None,
                    backend=target_device,
                    description="Generated baseline kernel",
                )
                if not publish_result.get("ok", False):
                    print(
                        "[workflow] Warning: failed to publish generated root for "
                        f"{op_name}: {publish_result.get('reason', 'unknown')}"
                    )

        if args.optimize and can_benchmark:
            update_job_progress(
                idx,
                total_ops,
                f"Optimizing kernel for {op_name} ({idx + 1}/{total_ops})",
            )
            opt_cmd = [
                sys.executable,
                "-m",
                "src.optimizer.pipeline",
                str(io_dir),
                args.project,
                "--op",
                op_name,
            ]
            if args.iterations and args.iterations > 0:
                opt_cmd += ["--max-iterations", str(args.iterations)]
            if args.workers and args.workers > 1:
                opt_cmd += ["--parallel", "--workers", str(args.workers)]
            if args.remote:
                opt_cmd += ["--remote", args.remote]
            rc = _run(opt_cmd, root, subprocess_env)
            if rc != 0:
                failure_msg = f"optimization failed (exit {rc})"
                op_failure = f"{op_failure}; {failure_msg}" if op_failure else failure_msg

        if args.benchmark and can_benchmark:
            if not wait_if_paused():
                return 130
            if check_cancelled():
                return 130

            update_job_progress(
                idx,
                progress_total,
                (
                    f"Benchmarking kernel for {op_name} ({idx + 1}/{total_ops})"
                    if not reused_existing
                    else (
                        f"Skipping generation for {op_name} (already generated); "
                        f"benchmarking ({idx + 1}/{total_ops})"
                    )
                ),
            )
            bench_cmd = [
                sys.executable,
                "-m",
                "src.optimizer.benchmarking.benchmark_ops",
                "--project",
                args.project,
                "--ops",
                op_name,
            ]
            # C2: signal benchmarking phase so the UI progress bar advances
            update_queue_state(project_dir, {"active_tasks": {task_key: {
                **task_meta,
                "current_step": "Benchmarking",
                "status": "In Progress",
            }}})
            rc = _run(bench_cmd, root, subprocess_env)
            if rc != 0:
                failure_msg = f"benchmark failed (exit {rc})"
                op_failure = f"{op_failure}; {failure_msg}" if op_failure else failure_msg
                update_queue_state(project_dir, {"active_tasks": {task_key: {
                    **task_meta,
                    "current_step": "Failed",
                    "status": "Failed",
                    "result": failure_msg[:200],
                }}})
            else:
                kernel_ms, backend = _load_kernel_benchmark(project_dir, op_name)
                # C1: write benchmark timing so completed section can show value_ms
                update_queue_state(project_dir, {"active_tasks": {task_key: {
                    **task_meta,
                    "current_step": "Done",
                    "status": "Done",
                    "value_ms": kernel_ms,
                }}})
                publish_result = publish_generated_root(
                    project_dir,
                    op_name,
                    kernel_ms=kernel_ms,
                    backend=backend or target_device,
                    description=(
                        "Generated baseline kernel (existing, benchmarked)"
                        if reused_existing
                        else "Generated baseline kernel (benchmarked)"
                    ),
                )
                if not publish_result.get("ok", False):
                    print(
                        "[workflow] Warning: failed to publish benchmarked root for "
                        f"{op_name}: {publish_result.get('reason', 'unknown')}"
                    )

        if op_failure:
            update_queue_state(project_dir, {"active_tasks": {task_key: {
                **task_meta,
                "current_step": "Failed",
                "status": "Failed",
                "result": op_failure[:200],
            }}})
            failed_ops.append((op_name, op_failure))
            print(f"[workflow] Failed {op_name}: {op_failure}. Continuing.")
            update_job_progress(
                idx + 1,
                progress_total,
                (
                    f"Unable to forge {op_name}; continuing "
                    f"({idx + 1}/{total_ops})."
                ),
            )
            continue

        if args.benchmark:
            if reused_existing:
                msg = f"Reused + benchmarked {idx + 1}/{total_ops} operators."
            else:
                msg = f"Generated + benchmarked {idx + 1}/{total_ops} operators."
            update_job_progress(idx + 1, progress_total, msg)
        else:
            if reused_existing:
                msg = f"Reused existing kernel {idx + 1}/{total_ops} operators."
            else:
                msg = f"Generated {idx + 1}/{total_ops} operators."
            update_job_progress(idx + 1, progress_total, msg)

    # Clear operator tracking once the generation loop finishes
    update_queue_state(project_dir, {
        "pending_operators": [],
        "current_operator": "",
    })

    if failed_ops:
        failed_count = len(failed_ops)
        success_count = total_ops - failed_count
        listed = ", ".join(op for op, _ in failed_ops[:6])
        if failed_count > 6:
            listed = f"{listed} (+{failed_count - 6} more)"
        final_msg = (
            f"Forged {success_count}/{total_ops} operators. "
            f"Unable to forge {failed_count} kernel(s): {listed}. "
            "Try a more powerful model or run again."
        )
        print(f"[workflow] {final_msg}")
        for op_name, reason in failed_ops:
            print(f"[workflow]   - {op_name}: {reason}")
        update_job_progress(total_ops, total_ops, final_msg)
        return 0

    update_job_progress(total_ops, total_ops, "Kernel generation completed.")
    return 0


def run_optimize(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = dict(os.environ)

    if hasattr(args, "llm_provider") and args.llm_provider:
        env["LLM_PROVIDER"] = args.llm_provider
    if hasattr(args, "llm_model") and args.llm_model:
        provider = getattr(args, "llm_provider", "") or ""
        if provider == "openai":
            env["OPENAI_MODEL"] = args.llm_model
        elif provider == "anthropic":
            env["ANTHROPIC_MODEL"] = args.llm_model
        elif provider in ("google", "gemini"):
            env["GEMINI_MODEL"] = args.llm_model
        else:
            env["OPENAI_MODEL"] = args.llm_model

    # Backend is inferred per-operator in the pipeline from success markers;
    # do not set KFORGE_TARGET_DEVICE here as it would incorrectly override
    # downstream device detection for mixed-backend projects.

    project_dir = _project_dir(args.project)
    io_dir = project_dir / "io" / "individual_ops"
    ops = _ops_from_csv(args.ops)
    if not ops:
        ops = _discover_ops(io_dir)

    total = len(ops)
    update_job_progress(0, total or 1, "Starting kernel optimization.")

    for idx, op_name in enumerate(ops):
        if not wait_if_paused():
            update_queue_state(project_dir, {
                "pending_operators": [],
                "current_operator": "",
            })
            return 130
        if check_cancelled():
            update_queue_state(project_dir, {
                "pending_operators": [],
                "current_operator": "",
            })
            return 130

        update_queue_state(project_dir, {
            "current_operator": op_name,
            "pending_operators": ops[idx + 1:],
        })
        update_job_progress(
            idx,
            total,
            f"Optimizing {op_name} ({idx + 1}/{total})",
        )

        opt_cmd = [
            sys.executable,
            "-m",
            "src.optimizer.pipeline",
            str(io_dir),
            args.project,
            "--op",
            op_name,
        ]
        if args.iterations and args.iterations > 0:
            opt_cmd += ["--max-iterations", str(args.iterations)]
        if args.workers and args.workers > 1:
            opt_cmd += ["--parallel", "--workers", str(args.workers)]
        if args.remote:
            opt_cmd += ["--remote", args.remote]
        rc = _run(opt_cmd, root, env)
        if rc != 0:
            reason = f"pipeline_exit_{rc}"
            print(
                f"[workflow-optimize-result] op={op_name} status=hard_error "
                f"new_nodes=0 last_reason={json.dumps(reason)}"
            )
            update_queue_state(project_dir, {
                "pending_operators": [],
                "current_operator": "",
            })
            update_job_progress(
                idx + 1,
                total,
                f"Optimization failed for {op_name} (exit {rc}).",
            )
            return rc
        print(
            f"[workflow-optimize-result] op={op_name} status=completed "
            f"pipeline_exit={rc}"
        )
        update_job_progress(
            idx + 1,
            total,
            f"Optimized {idx + 1}/{total} operators.",
        )

    if args.benchmark:
        update_job_progress(total, total, "Benchmarking optimized kernels.")
        bench_cmd = [
            sys.executable,
            "-m",
            "src.optimizer.benchmarking.benchmark_ops",
            "--project",
            args.project,
        ]
        rc = _run(bench_cmd, root, env)
        if rc != 0:
            update_queue_state(project_dir, {
                "pending_operators": [],
                "current_operator": "",
            })
            return rc

    update_queue_state(project_dir, {
        "pending_operators": [],
        "current_operator": "",
    })
    update_job_progress(total, total, "Kernel optimization completed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel Forge workflow runner")
    sub = parser.add_subparsers(dest="action", required=True)

    profile = sub.add_parser("profile")
    profile.add_argument("--project", required=True)
    profile.add_argument("--weights-b64-path", default="")
    profile.add_argument("--validation-b64-path", default="")
    profile.add_argument("--validation-name-path", default="")

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--project", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--project", required=True)
    generate.add_argument("--ops", default="")
    generate.add_argument("--optimize", action="store_true")
    generate.add_argument("--benchmark", action="store_true")
    generate.add_argument("--iterations", type=int, default=0)
    generate.add_argument("--target-device", default="cuda")
    generate.add_argument("--remote", default="")
    generate.add_argument("--llm-provider", default="")
    generate.add_argument("--llm-model", default="")
    generate.add_argument("--workers", type=int, default=1)

    optimize = sub.add_parser("optimize")
    optimize.add_argument("--project", required=True)
    optimize.add_argument("--ops", default="")
    optimize.add_argument("--iterations", type=int, default=0)
    optimize.add_argument("--benchmark", action="store_true")
    optimize.add_argument("--target-device", default="cuda")
    optimize.add_argument("--remote", default="")
    optimize.add_argument("--llm-provider", default="")
    optimize.add_argument("--llm-model", default="")
    optimize.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    if args.action == "profile":
        return run_profile(args)
    if args.action == "benchmark":
        return run_benchmark(args)
    if args.action == "generate":
        return run_generate(args)
    if args.action == "optimize":
        return run_optimize(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
