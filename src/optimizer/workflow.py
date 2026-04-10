from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.progress import check_cancelled, update_job_progress, wait_if_paused
from src.optimizer.tree_store import publish_generated_root
from src.optimizer.pipeline import update_queue_state
from src.optimizer.config.settings import MIN_CUDA_VERSION

_RESULT_TRUNCATE = 300


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_dir(project: str) -> Path:
    return _repo_root() / "kernels" / "projects" / project


def _with_python_bin_on_path(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(env) if env is not None else dict(os.environ)
    python_bin = str(Path(sys.executable).parent)
    current_path = merged.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    if python_bin not in path_parts:
        merged["PATH"] = (
            python_bin
            if not current_path
            else python_bin + os.pathsep + current_path
        )
    return merged


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


def _fail_all_active_tasks(project_dir: Path, reason: str) -> None:
    """Mark every active queue task as Failed with a given reason.

    Called when a preflight check fails before per-task processing begins,
    so tasks don't get stuck in 'In Progress' forever.
    """
    queue_path = project_dir / "queue.json"
    if not queue_path.exists():
        return
    from src.optimizer.benchmarking.locks import file_lock
    lock_path = queue_path.with_suffix(".json.lock")
    try:
        with file_lock(lock_path):
            state = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return
    active = state.get("active_tasks", {})
    if not active:
        return
    updates = {
        k: {**v, "current_step": "Failed", "status": "Failed", "result": reason[:_RESULT_TRUNCATE]}
        for k, v in active.items()
        if v.get("current_step") not in ("Done", "Failed")
    }
    if updates:
        update_queue_state(project_dir, {"active_tasks": updates})


def _preflight_nvcc_version() -> tuple[bool, str]:
    """Check that nvcc meets the minimum required version (11.8)."""
    _MIN = MIN_CUDA_VERSION
    nvcc = shutil.which("nvcc")
    cuda_home = os.environ.get("CUDA_HOME", "")
    if not nvcc and cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.exists():
            nvcc = str(candidate)
    if not nvcc:
        return False, (
            "CUDA preflight failed: nvcc not found on PATH or in CUDA_HOME. "
            "Ensure CUDA is installed and CUDA_HOME is set correctly."
        )
    try:
        out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT).decode()
    except Exception as e:
        return False, f"CUDA preflight failed: could not run nvcc --version ({e})."
    match = re.search(r"release (\d+)\.(\d+)", out)
    if not match:
        return False, f"CUDA preflight failed: could not parse nvcc version from: {out!r}"
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < _MIN:
        return False, (
            f"CUDA preflight failed: nvcc {major}.{minor} is too old. "
            f"Kernel Forge requires CUDA {_MIN[0]}.{_MIN[1]} or newer."
        )
    return True, ""


def _preflight_cuda_target() -> tuple[bool, str]:
    ok, reason = _preflight_nvcc_version()
    if not ok:
        return False, reason

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
                "continuing with forward-compatible CUDA support."
            )

        return False, (
            "CUDA preflight failed: detected GPU "
            f"'{dev_name}' capability {dev_cap[0]}.{dev_cap[1]} exceeds this "
            f"PyTorch build maximum {max_cap[0]}.{max_cap[1]} "
            f"(torch CUDA {torch_cuda})."
        )

    return True, ""


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    print(f"[workflow] Running: {' '.join(cmd)}")
    # Use spool files instead of capture_output=True so spawned helper processes
    # cannot keep PIPE FDs open and wedge the parent workflow process.
    popen_env = dict(env)
    popen_env.setdefault("PYTHONUNBUFFERED", "1")
    with tempfile.TemporaryDirectory(prefix="kforge_workflow_run_") as tmpdir:
        combined_path = Path(tmpdir) / "combined.log"
        with combined_path.open("w", encoding="utf-8", buffering=1) as combined:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=popen_env,
                stdout=combined,
                stderr=combined,
                text=True,
            )
        chunks: list[str] = []
        with combined_path.open("r", encoding="utf-8", errors="replace") as combined:
            while True:
                chunk = combined.read()
                if chunk:
                    print(chunk, end="")
                    chunks.append(chunk)
                result = proc.poll()
                if result is not None:
                    break
                time.sleep(0.1)
            tail = combined.read()
            if tail:
                print(tail, end="")
                chunks.append(tail)
        combined_text = "".join(chunks)

    if combined_text:
        return int(result), combined_text.strip()
    return int(result), ""


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
        if not child.is_dir():
            continue
        has_entries = any(entry.name.startswith("entry_") and entry.suffix == ".pt" for entry in child.iterdir())
        if has_entries:
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


def _find_kernel_source(op_dir: Path) -> Path | None:
    for name in ("kernel.cu", "kernel.py", "kernel.metal", "kernel.mps", "kernel.cpu"):
        candidate = op_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(op_dir.glob("kernel.*")):
        if candidate.is_file():
            return candidate
    return None


def _load_tree_root_kernel_path(project_dir: Path, op_name: str) -> Path | None:
    tree_op_dir = project_dir / "trees" / op_name
    meta_path = tree_op_dir / "generated_root.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            relpath = str(payload.get("kernel_relpath") or "").strip()
            if relpath:
                candidate = project_dir / relpath
                if candidate.exists() and candidate.is_file():
                    return candidate
        except Exception:
            pass

    kernels_dir = tree_op_dir / "kernels"
    if not kernels_dir.exists():
        return None
    candidates = sorted(kernels_dir.glob("kernel_0.*"))
    if not candidates:
        candidates = sorted(kernels_dir.glob("kernel_*.*"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_saved_benchmark_for_existing_kernel(
    project_dir: Path,
    op_name: str,
    generated_op_dir: Path,
) -> tuple[float | None, str]:
    generated_kernel = _find_kernel_source(generated_op_dir)
    if generated_kernel is None:
        return None, ""

    tree_kernel = _load_tree_root_kernel_path(project_dir, op_name)
    if tree_kernel is None:
        return None, ""

    try:
        if _file_sha256(generated_kernel) != _file_sha256(tree_kernel):
            return None, ""
    except Exception:
        return None, ""

    return _load_kernel_benchmark(project_dir, op_name)


def _load_kernel_benchmark(project_dir: Path, op_name: str) -> tuple[float | None, str]:
    bench_path = project_dir / "benchmarks" / "op_benchmarks.json"
    if not bench_path.exists():
        return None, ""
    try:
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
    except Exception:
        return None, ""

    results = []
    if isinstance(payload, dict):
        for key in ("benchmarks", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                results = value
                break
    if not isinstance(results, list):
        return None, ""

    for row in results:
        if not isinstance(row, dict):
            continue
        if str(row.get("op", "")) != op_name:
            continue
        micro = row.get("micro") if isinstance(row.get("micro"), dict) else {}
        micro_status = str(micro.get("status", "") or "").lower()
        legacy_status = str(row.get("kernel_status", "") or "").lower()
        if micro:
            if micro_status not in {"ready", "ok"}:
                continue
            kernel_entry_latencies = micro.get("entry_latencies_ms")
            kernel_ms = micro.get("kernel_ms")
            backend = str(micro.get("backend", "") or row.get("backend", "") or "")
        else:
            if legacy_status != "ok":
                continue
            kernel_entry_latencies = row.get("kernel_entry_latencies_ms")
            kernel_ms = row.get("kernel_ms")
            backend = str(row.get("backend", "") or "")
        if not isinstance(kernel_entry_latencies, list) or not kernel_entry_latencies:
            return None, ""
        try:
            parsed_ms = float(kernel_ms)
        except Exception:
            return None, ""
        if parsed_ms <= 0.0:
            return None, ""
        return parsed_ms, backend
    return None, ""


def run_profile(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _with_python_bin_on_path()
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
    rc, _ = _run(cmd, root, env)
    return rc


def run_benchmark(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _with_python_bin_on_path()
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.benchmarking.benchmark_ops",
        "--project",
        args.project,
    ]
    rc, _ = _run(cmd, root, env)
    return rc


def run_generate(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = _with_python_bin_on_path()

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

    project_dir = _project_dir(args.project)
    io_dir = project_dir / "io" / "individual_ops"
    out_dir = project_dir / "kernels" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_device == "cuda" and not getattr(args, "remote", ""):
        ok, reason = _preflight_cuda_target()
        if not ok:
            print(f"[workflow] {reason}")
            _fail_all_active_tasks(project_dir, reason)
            return 2
        if reason:
            print(f"[workflow] {reason}")

    discovered_ops = _discover_ops(io_dir)
    ops = _ops_from_csv(args.ops)
    if ops:
        invalid_ops = [op for op in ops if op not in discovered_ops]
        ops = [op for op in ops if op in discovered_ops]
        if invalid_ops:
            available = ", ".join(discovered_ops) if discovered_ops else "(none)"
            reason = (
                "No captured inputs found for requested ops: "
                + ", ".join(invalid_ops)
                + ". Available captured ops: "
                + available
            )
            print(f"[workflow] {reason}")
            if not ops:
                _fail_all_active_tasks(project_dir, reason)
                return 1
    else:
        ops = discovered_ops
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
        reused_saved_benchmark = False
        saved_kernel_ms: float | None = None
        saved_backend = ""

        # C3/C4: advance current_operator and shrink pending_operators in the queue
        update_queue_state(project_dir, {
            "current_operator": op_name,
            "pending_operators": ops[idx + 1:],
        })

        if _has_success_marker(generated_op_dir):
            reused_existing = True
            can_benchmark = True
            if args.benchmark:
                saved_kernel_ms, saved_backend = _load_saved_benchmark_for_existing_kernel(
                    project_dir,
                    op_name,
                    generated_op_dir,
                )
                reused_saved_benchmark = saved_kernel_ms is not None
            publish_result = publish_generated_root(
                project_dir,
                op_name,
                kernel_ms=saved_kernel_ms,
                backend=saved_backend or target_device,
                description=(
                    "Generated baseline kernel (existing, benchmark reused)"
                    if reused_saved_benchmark
                    else "Generated baseline kernel (existing)"
                ),
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
            rc, stderr = _run(gen_cmd, root, env)
            if rc != 0:
                detail = f": {stderr[:_RESULT_TRUNCATE]}" if stderr else ""
                op_failure = f"generation command failed (exit {rc}){detail}"
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
            rc, stderr = _run(opt_cmd, root, subprocess_env)
            if rc != 0:
                detail = f": {stderr[:_RESULT_TRUNCATE]}" if stderr else ""
                failure_msg = f"optimization failed (exit {rc}){detail}"
                op_failure = f"{op_failure}; {failure_msg}" if op_failure else failure_msg

        if args.benchmark and can_benchmark:
            if not wait_if_paused():
                return 130
            if check_cancelled():
                return 130

            if reused_saved_benchmark:
                update_job_progress(
                    idx,
                    progress_total,
                    f"Reusing saved benchmark for {op_name} ({idx + 1}/{total_ops})",
                )
                update_queue_state(project_dir, {"active_tasks": {task_key: {
                    **task_meta,
                    "current_step": "Done",
                    "status": "Done",
                    "value_ms": saved_kernel_ms,
                    "kernel_id": "0",
                    "op_name": op_name,
                    "tag": "[GEN]",
                }}})
            else:
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
                    "op_name": op_name,
                    "tag": "[GEN]",
                }}})
                rc, stderr = _run(bench_cmd, root, subprocess_env)
                if rc != 0:
                    detail = f": {stderr[:_RESULT_TRUNCATE]}" if stderr else ""
                    failure_msg = f"benchmark failed (exit {rc}){detail}"
                    op_failure = f"{op_failure}; {failure_msg}" if op_failure else failure_msg
                    update_queue_state(project_dir, {"active_tasks": {task_key: {
                        **task_meta,
                        "current_step": "Failed",
                        "status": "Failed",
                        "result": failure_msg[:_RESULT_TRUNCATE],
                        "op_name": op_name,
                        "tag": "[GEN]",
                    }}})
                else:
                    kernel_ms, backend = _load_kernel_benchmark(project_dir, op_name)
                    # C1: write benchmark timing so completed section can show value_ms
                    update_queue_state(project_dir, {"active_tasks": {task_key: {
                        **task_meta,
                        "current_step": "Done",
                        "status": "Done",
                        "value_ms": kernel_ms,
                        "kernel_id": "0",
                        "op_name": op_name,
                        "tag": "[GEN]",
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
                "result": op_failure[:_RESULT_TRUNCATE],
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
            if reused_saved_benchmark:
                msg = f"Reused saved benchmark for {idx + 1}/{total_ops} operators."
            elif reused_existing:
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
    env = _with_python_bin_on_path()

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

    target_device = _normalize_device(getattr(args, "target_device", "cuda") or "cuda")
    if target_device == "cuda" and not getattr(args, "remote", ""):
        ok, reason = _preflight_cuda_target()
        if not ok:
            print(f"[workflow] {reason}")
            _fail_all_active_tasks(project_dir, reason)
            return 2
        if reason:
            print(f"[workflow] {reason}")

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
        rc, stderr = _run(opt_cmd, root, env)
        if rc != 0:
            reason = f"pipeline_exit_{rc}" + (f": {stderr[:_RESULT_TRUNCATE]}" if stderr else "")
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
        rc, _ = _run(bench_cmd, root, env)
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
