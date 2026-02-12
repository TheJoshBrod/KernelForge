import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, name: str) -> bool:
    print(f"[{name}] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[{name}] failed with code {result.returncode}")
        return False
    return True


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_llm_config(repo_root: Path) -> tuple[str, str, str, str]:
    config_path = repo_root / "frontend" / "config.json"
    if config_path.exists():
        data = _load_json(config_path)
        llm = data.get("llm_info") if isinstance(data, dict) else None
        auth = data.get("auth") if isinstance(data, dict) else None
        provider = str(os.environ.get("LLM_PROVIDER", "")).strip().lower()
        if not provider and isinstance(auth, dict):
            provider = str(auth.get("provider", "")).strip().lower()
        if not provider and isinstance(llm, dict):
            provider = str(llm.get("provider", "")).strip().lower()

        model = ""
        if provider == "openai":
            model = str(os.environ.get("OPENAI_MODEL", "")).strip()
        elif provider == "anthropic":
            model = str(os.environ.get("ANTHROPIC_MODEL", "")).strip()
        elif provider == "gemini":
            model = str(os.environ.get("GEMINI_MODEL", "")).strip()
        if not model and isinstance(auth, dict):
            model = str(auth.get("model", "")).strip()
        if not model and isinstance(llm, dict):
            model = str(llm.get("model", "")).strip()

        apikey = ""
        if provider == "openai":
            apikey = str(os.environ.get("OPENAI_API_KEY", "")).strip()
        elif provider == "anthropic":
            apikey = str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        elif provider == "gemini":
            apikey = str(os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not apikey and isinstance(llm, dict):
            apikey = str(llm.get("apikey", "")).strip()

        mode = str(os.environ.get("CGINS_AUTH_MODE", "")).strip().lower()
        if not mode and isinstance(auth, dict):
            mode = str(auth.get("mode", "")).strip().lower()
        if not mode:
            mode = "auto"
        return provider, model, apikey, mode
    provider = str(os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if provider == "openai":
        model = str(os.environ.get("OPENAI_MODEL", "")).strip()
    elif provider == "anthropic":
        model = str(os.environ.get("ANTHROPIC_MODEL", "")).strip()
    elif provider == "gemini":
        model = str(os.environ.get("GEMINI_MODEL", "")).strip()
    else:
        model = ""
    if provider == "openai":
        apikey = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    elif provider == "anthropic":
        apikey = str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    elif provider == "gemini":
        apikey = str(os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")).strip()
    else:
        apikey = ""
    mode = str(os.environ.get("CGINS_AUTH_MODE", "auto")).strip().lower() or "auto"
    return provider, model, apikey, mode


def _check_profile_outputs(project_dir: Path, enforce_skiplist: bool) -> bool:
    summary_path = project_dir / "io" / "summary.json"
    if not summary_path.exists():
        print("[profile] summary.json missing")
        return False
    summary = _load_json(summary_path)
    op_counts = summary.get("op_counts", {}) if isinstance(summary, dict) else {}
    if not op_counts:
        print("[profile] op_counts empty")
        return False
    if enforce_skiplist:
        skip_terms = [
            "dropout",
            "rand",
            "bernoulli",
            "multinomial",
            "view",
            "reshape",
            "permute",
            "transpose",
            "squeeze",
            "unsqueeze",
            "expand",
            "as_strided",
            "size",
            "stride",
            "numel",
            "contiguous",
            "clone",
            "copy_",
            "empty",
            "zeros",
            "ones",
            "full",
            "arange",
        ]
        for op_name in op_counts.keys():
            name = str(op_name).lower()
            if any(term in name for term in skip_terms):
                print(f"[profile] skipped op still present: {op_name}")
                return False
    return True


def _check_generation_outputs(project_dir: Path, io_dir: Path) -> bool:
    ops_dir = io_dir
    generated_dir = project_dir / "kernels" / "generated" / "individual_op_kernels"
    if not ops_dir.exists():
        print("[generate] io dir missing")
        return False
    ok = True
    for op_dir in ops_dir.iterdir():
        if not op_dir.is_dir():
            continue
        op_name = op_dir.name
        out_dir = generated_dir / op_name
        if not out_dir.exists():
            print(f"[generate] missing output for {op_name}")
            ok = False
            continue
        success = out_dir / "kernel.cu"
        failure = out_dir / "attempts" / "failure.json"
        if not success.exists() and not failure.exists():
            print(f"[generate] no success or failure report for {op_name}")
            ok = False
    return ok


def _check_benchmark_outputs(project_dir: Path) -> bool:
    bench_path = project_dir / "benchmarks" / "op_benchmarks.json"
    if not bench_path.exists():
        print("[benchmark] op_benchmarks.json missing")
        return False
    data = _load_json(bench_path)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        print("[benchmark] results missing or invalid")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CGinS automated smoke tests.")
    parser.add_argument("--project", default="pipeline_test")
    parser.add_argument("--force", action="store_true", help="Overwrite test project")
    parser.add_argument("--with-llm", action="store_true", help="Run LLM connectivity + generation")
    parser.add_argument("--require-llm", action="store_true", help="Fail if LLM not configured")
    parser.add_argument("--with-benchmark", action="store_true", help="Run benchmarks (requires CUDA)")
    parser.add_argument("--enforce-skiplist", action="store_true", help="Fail if skipped ops appear in profile")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    project_dir = repo_root / "projects" / args.project

    if not project_dir.exists() or args.force:
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "create_test_project.py"),
            "--name",
            args.project,
        ]
        if args.force:
            cmd.append("--force")
        if not _run(cmd, repo_root, "create_project"):
            return 1

    if not _run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "profiler" / "profile_project.py"),
            "--project",
            args.project,
        ],
        repo_root,
        "profile",
    ):
        return 1

    if not _check_profile_outputs(project_dir, args.enforce_skiplist):
        return 1

    if args.with_llm:
        provider, model, apikey, mode = _resolve_llm_config(repo_root)
        if not provider or not model:
            msg = "[llm] missing provider/model for LLM tests"
            if args.require_llm:
                print(msg)
                return 1
            print(msg + " (skipping)")
        else:
            llm_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "test_llm_connection.py"),
                "--provider",
                provider,
                "--model",
                model,
            ]
            if apikey:
                llm_cmd += ["--apikey", apikey]
            if not _run(
                llm_cmd,
                repo_root,
                f"llm_test({mode})",
            ):
                return 1

            io_dir = project_dir / "io" / "individual_ops"
            out_dir = project_dir / "kernels" / "generated"
            out_dir.mkdir(parents=True, exist_ok=True)
            if not _run(
                [
                    sys.executable,
                    "-m",
                    "src.generator.main",
                    "--io-dir",
                    str(io_dir),
                    "--out-dir",
                    str(out_dir),
                ],
                repo_root,
                "generate",
            ):
                return 1

            if not _check_generation_outputs(project_dir, io_dir):
                return 1

    if args.with_benchmark:
        try:
            import torch
        except Exception:
            print("[benchmark] torch not available")
            return 1
        if not torch.cuda.is_available():
            print("[benchmark] CUDA not available")
            return 1
        if not _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "benchmark_project_ops.py"),
                "--project",
                args.project,
            ],
            repo_root,
            "benchmark",
        ):
            return 1
        if not _check_benchmark_outputs(project_dir):
            return 1

    print("[ok] smoke tests completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
