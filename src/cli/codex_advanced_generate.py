#!/usr/bin/env python3
"""Advanced generation entrypoint backed by optimizer pipeline modules."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"[codex_advanced_generate] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def _load_ops(ops_file: Path) -> list[str]:
    raw = json.loads(ops_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if item:
            out.append(str(item))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--ops-file", required=True)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--optimize-only", action="store_true")
    parser.add_argument("--opt-iterations", type=int, default=0)
    parser.add_argument("--opt-patience", type=int, default=0)
    parser.add_argument("--generate-attempts", type=int, default=0)
    parser.add_argument("--codex-model", default="")
    parser.add_argument("--opt-min-improve-ms", type=float, default=0.0)
    parser.add_argument("--opt-min-speedup-pct", type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ops_path = Path(args.ops_file)
    if not ops_path.is_absolute():
        ops_path = repo_root / ops_path
    if not ops_path.exists():
        print(f"[codex_advanced_generate] Missing ops file: {ops_path}")
        return 1

    ops = _load_ops(ops_path)
    if not ops:
        print("[codex_advanced_generate] No ops provided.")
        return 1

    io_dir = repo_root / "kernels" / "projects" / args.project / "io" / "individual_ops"
    if not io_dir.exists():
        print(f"[codex_advanced_generate] Missing io dir: {io_dir}")
        return 1

    env = dict(os.environ)
    if args.codex_model:
        env["OPTIMIZER_LLM_MODEL_NAME"] = str(args.codex_model)

    if args.optimize or args.optimize_only:
        for op_name in ops:
            cmd = [
                sys.executable,
                "-m",
                "src.optimizer.pipeline",
                str(io_dir),
                args.project,
                "--op",
                op_name,
            ]
            if args.opt_iterations and args.opt_iterations > 0:
                cmd += ["--max-iterations", str(args.opt_iterations)]
            rc = _run(cmd, repo_root, env)
            if rc != 0:
                return rc

    if args.benchmark:
        bench_cmd = [
            sys.executable,
            "-m",
            "src.optimizer.benchmarking.benchmark_ops",
            "--project",
            args.project,
        ]
        rc = _run(bench_cmd, repo_root, env)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

