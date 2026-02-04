#!/usr/bin/env python3
"""
Advanced per-op generation using Codex CLI inside a container.

- One-shot Codex generation for selected ops
- Optional optimization loop using Codex CLI
- Optional benchmarking against PyTorch
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.progress import update_job_progress, wait_if_paused, check_cancelled
from src.config import ensure_llm_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("CGINS_PROJECT_DIR"))


def _ensure_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    if key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    if key and not os.environ.get("CODEX_API_KEY"):
        os.environ["CODEX_API_KEY"] = key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY for Codex CLI.")
        return False
    return True


def _prepare_codex_env() -> None:
    if not _in_container():
        return
    base = os.environ.get("CGINS_CODEX_HOME")
    if not base:
        project_dir = os.environ.get("CGINS_PROJECT_DIR")
        if project_dir:
            base = str(Path(project_dir) / ".codex")
        else:
            base = "/tmp/cgins-codex"
    os.environ["HOME"] = base
    os.environ.setdefault("CODEX_HOME", base)
    os.environ.setdefault("XDG_CONFIG_HOME", str(Path(base) / ".config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(base) / ".cache"))
    os.environ.setdefault("XDG_STATE_HOME", str(Path(base) / ".local" / "state"))
    for key in ("HOME", "CODEX_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        try:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
        except Exception:
            continue


def _normalize_op_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _parse_ops(value: str | None) -> list[str]:
    if not value:
        return []
    ops = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        ops.append(_normalize_op_name(item))
    return ops


def _project_dir(project: str) -> Path:
    return _repo_root() / "projects" / project


def _generated_root(project: str) -> Path:
    return _project_dir(project) / "kernels" / "generated" / "individual_op_kernels"


def _run(cmd: list[str], *, env: dict | None = None) -> int:
    return subprocess.run(cmd, cwd=str(_repo_root()), env=env).returncode


def _benchmark(project: str, ops: list[str]) -> dict:
    if not ops:
        return {}
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "benchmark_project_ops.py"),
        "--project",
        project,
        "--only-ops",
        ",".join(ops),
    ]
    code = _run(cmd)
    if code != 0:
        return {}

    out_path = _project_dir(project) / "benchmarks" / "op_benchmarks.json"
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return {}
    return {r.get("op"): r for r in results if isinstance(r, dict)}


def _verify(project: str, op: str) -> bool:
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "verify_one_op.py"),
        "--project",
        project,
        "--op",
        op,
    ]
    return _run(cmd) == 0


def _prepare_workspace(project: str, op: str, work_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "codex_prepare_workspace.py"),
        "--project",
        project,
        "--op",
        op,
        "--work-dir",
        str(work_dir),
        "--overwrite",
    ]
    return _run(cmd)


def _run_codex(work_dir: Path, prompt: str, *, model: str | None, sandbox: str | None) -> bool:
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "codex_exec.py"),
        "--work-dir",
        str(work_dir),
        "--prompt",
        prompt,
    ]
    if model:
        cmd += ["--model", model]
    if sandbox:
        cmd += ["--sandbox", sandbox]
    return _run(cmd) == 0


def _read_kernel(kernel_path: Path) -> str:
    try:
        return kernel_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_kernel(kernel_path: Path, contents: str) -> None:
    kernel_path.write_text(contents, encoding="utf-8")


def _missing_kernels(project: str, ops: list[str]) -> list[str]:
    missing: list[str] = []
    root = _generated_root(project)
    for op in ops:
        op_dir = root / op
        kernel_path = op_dir / "kernel.cu"
        success_path = op_dir / "success"
        if not kernel_path.exists() or not success_path.exists():
            missing.append(op)
    return missing


def _optimize_op(
    project: str,
    op: str,
    *,
    iterations: int,
    patience: int,
    min_improve_ms: float,
    model: str | None,
    sandbox: str | None,
) -> None:
    kernel_dir = _generated_root(project) / op
    kernel_path = kernel_dir / "kernel.cu"
    if not kernel_path.exists():
        print(f"[optimize] kernel.cu not found for {op}")
        return

    work_dir = kernel_dir / "work"
    if _prepare_workspace(project, op, work_dir) != 0:
        print(f"[optimize] failed to prepare workspace for {op}")
        return

    best_kernel = _read_kernel(kernel_path)
    best_ms = None

    baseline = _benchmark(project, [op]).get(op, {})
    if baseline:
        best_ms = baseline.get("kernel_ms")

    no_improve = 0
    for attempt in range(1, iterations + 1):
        if not wait_if_paused() or check_cancelled():
            print("Optimization cancelled.")
            return

        prompt = "\n".join(
            [
                f"Task: Optimize kernel.cu for op {op}.",
                "Rules:",
                "- Edit kernel.cu only.",
                "- Keep torch::Tensor launch(...) signature exactly unchanged.",
                "- Preserve correctness; run verify after edits.",
                "Suggested checks:",
                f"- python scripts/verify_one_op.py --project {project} --op {op}",
                f"- python scripts/benchmark_project_ops.py --project {project} --only-ops {op}",
            ]
        )
        if best_ms:
            prompt += f"\nCurrent best kernel_ms: {best_ms}" \
                + " (lower is better)."

        ok = _run_codex(work_dir, prompt, model=model, sandbox=sandbox)
        if not ok:
            print(f"[optimize] codex exec failed for {op} attempt {attempt}")
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[optimize] no improvement in {patience} attempts; stopping.")
                break
            continue

        # Sync edited kernel back into generated kernel dir
        edited_kernel = work_dir / "kernel.cu"
        if edited_kernel.exists():
            shutil.copy2(edited_kernel, kernel_path)

        if not _verify(project, op):
            print(f"[optimize] verify failed for {op} attempt {attempt}")
            _write_kernel(kernel_path, best_kernel)
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[optimize] no improvement in {patience} attempts; stopping.")
                break
            continue

        results = _benchmark(project, [op]).get(op, {})
        kernel_ms = results.get("kernel_ms") if results else None
        if kernel_ms is None:
            no_improve += 1
            _write_kernel(kernel_path, best_kernel)
        else:
            improved = best_ms is None or (best_ms - kernel_ms) >= min_improve_ms
            if improved:
                best_ms = kernel_ms
                best_kernel = _read_kernel(kernel_path)
                no_improve = 0
                print(f"[optimize] new best for {op}: {best_ms} ms")
            else:
                _write_kernel(kernel_path, best_kernel)
                no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"[optimize] no improvement in {patience} attempts; stopping.")
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Advanced Codex generate + optimize")
    parser.add_argument("--project", required=True)
    parser.add_argument("--ops", default=None, help="Comma-separated op list")
    parser.add_argument("--ops-file", default=None, help="JSON file containing ops list")
    parser.add_argument("--optimize", action="store_true", help="Run Codex optimization loop")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after generation")
    parser.add_argument("--opt-iterations", type=int, default=3)
    parser.add_argument("--opt-patience", type=int, default=2)
    parser.add_argument("--opt-min-improve-ms", type=float, default=0.0)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-sandbox", default="workspace-write")
    parser.add_argument("--generate-attempts", type=int, default=3)
    args = parser.parse_args()

    ensure_llm_config()
    _prepare_codex_env()
    if _in_container():
        os.environ.setdefault("CGINS_CODEX_AUTO_INSTALL", "1")
    if not _ensure_openai_key():
        return 2

    project = str(args.project).strip()
    ops = []
    if args.ops_file:
        try:
            ops_payload = json.loads(Path(args.ops_file).read_text(encoding="utf-8"))
            if isinstance(ops_payload, list):
                ops = [_normalize_op_name(str(o)) for o in ops_payload if o]
        except Exception:
            ops = []
    if not ops:
        ops = _parse_ops(args.ops)
    if not ops:
        io_root = _project_dir(project) / "io" / "individual_ops"
        if io_root.exists():
            ops = sorted([p.name for p in io_root.iterdir() if p.is_dir()])
    if not ops:
        print("No ops provided.")
        return 2

    # One-shot Codex generation
    update_job_progress(0, len(ops), "Starting Codex generation")
    io_dir = _project_dir(project) / "io" / "individual_ops"
    out_dir = _project_dir(project) / "kernels" / "generated"

    env = os.environ.copy()
    env["CGINS_CODEX_GENERATE"] = "1"
    env["CGINS_CODEX_REPAIR"] = "1"
    env["CGINS_CODEX_MAX_ATTEMPTS"] = str(max(args.generate_attempts, 1))
    env["CGINS_MAX_ATTEMPTS"] = str(max(args.generate_attempts, 1))
    env["CGINS_CODEX_GENERATE_OPS"] = ",".join(ops)
    env["CGINS_CODEX_REPAIR_OPS"] = ",".join(ops)
    if args.codex_model:
        env["CGINS_CODEX_MODEL"] = args.codex_model
    if args.codex_sandbox:
        env["CGINS_CODEX_SANDBOX"] = args.codex_sandbox

    gen_cmd = [
        sys.executable,
        "-m",
        "src.generator.main",
        "--io-dir",
        str(io_dir),
        "--out-dir",
        str(out_dir),
        "--only-ops",
        ",".join(ops),
    ]

    if _run(gen_cmd, env=env) != 0:
        print("Generation failed.")
        update_job_progress(0, len(ops), "Generation failed")
        return 1

    missing = _missing_kernels(project, ops)
    if missing:
        preview = ", ".join(missing[:6])
        suffix = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
        print(f"Missing kernels for {len(missing)} ops: {preview}{suffix}")
        update_job_progress(
            0, len(ops), f"Missing kernels for {len(missing)} ops"
        )
        return 1

    update_job_progress(len(ops), len(ops), "Generation complete")

    if args.benchmark:
        update_job_progress(0, len(ops), "Benchmarking generated kernels")
        _benchmark(project, ops)
        update_job_progress(len(ops), len(ops), "Benchmark complete")

    if args.optimize:
        update_job_progress(0, len(ops), "Starting Codex optimization")
        for idx, op in enumerate(ops, start=1):
            if not wait_if_paused() or check_cancelled():
                print("Optimization cancelled.")
                return 1
            update_job_progress(idx - 1, len(ops), f"Optimizing {op}")
            _optimize_op(
                project,
                op,
                iterations=max(args.opt_iterations, 1),
                patience=max(args.opt_patience, 0),
                min_improve_ms=max(args.opt_min_improve_ms, 0.0),
                model=args.codex_model,
                sandbox=args.codex_sandbox,
            )
        update_job_progress(len(ops), len(ops), "Optimization complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
