from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_dir(project: str) -> Path:
    return _repo_root() / "kernels" / "projects" / project


def _normalize_device(device: str) -> str:
    d = (device or "").strip().lower()
    if d in {"gpu", "cuda", "rocm", "amd"}:
        return "cuda"
    if d in {"mps", "metal", "apple"}:
        return "mps"
    if d in {"xpu", "intel"}:
        return "cpu"
    if d == "cpu":
        return "cpu"
    return "cuda"


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

    target_device = _normalize_device(args.target_device)
    env["CGINS_TARGET_DEVICE"] = target_device
    if args.gen_attempts and args.gen_attempts > 0:
        env["CGINS_MAX_ATTEMPTS"] = str(args.gen_attempts)

    project_dir = _project_dir(args.project)
    io_dir = project_dir / "io" / "individual_ops"
    out_dir = project_dir / "kernels" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = _ops_from_csv(args.ops)
    if not ops:
        ops = _discover_ops(io_dir)

    gen_cmd = [
        sys.executable,
        "-m",
        "src.generator.main",
        "--io-dir",
        str(io_dir),
        "--out-dir",
        str(out_dir),
    ]
    if ops:
        gen_cmd += ["--only-ops", ",".join(ops)]
    rc = _run(gen_cmd, root, env)
    if rc != 0:
        return rc

    if args.optimize:
        for op_name in ops:
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
            rc = _run(opt_cmd, root, env)
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
        rc = _run(bench_cmd, root, env)
        if rc != 0:
            return rc

    return 0


def run_optimize(args: argparse.Namespace) -> int:
    root = _repo_root()
    env = dict(os.environ)
    target_device = _normalize_device(args.target_device)
    env["CGINS_TARGET_DEVICE"] = target_device

    project_dir = _project_dir(args.project)
    io_dir = project_dir / "io" / "individual_ops"
    ops = _ops_from_csv(args.ops)
    if not ops:
        ops = _discover_ops(io_dir)

    for op_name in ops:
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
        rc = _run(opt_cmd, root, env)
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
        rc = _run(bench_cmd, root, env)
        if rc != 0:
            return rc

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
    generate.add_argument("--gen-attempts", type=int, default=0)
    generate.add_argument("--target-device", default="cuda")

    optimize = sub.add_parser("optimize")
    optimize.add_argument("--project", required=True)
    optimize.add_argument("--ops", default="")
    optimize.add_argument("--iterations", type=int, default=0)
    optimize.add_argument("--benchmark", action="store_true")
    optimize.add_argument("--target-device", default="cuda")

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
