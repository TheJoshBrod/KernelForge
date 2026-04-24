from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "gemma4-e2b-int4-gb10"
MODEL_SLUG = "gemma4-e2b-int4-gb10"
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-opus-4-7"
TARGET_DEVICE = "cuda"
ARMS = [
    ("zero_shot", 0),
    ("optimize_5", 5),
    ("optimize_10", 10),
    ("optimize_20", 20),
    ("optimize_50", 50),
]


def project_dir() -> Path:
    return REPO_ROOT / "kernels" / "projects" / PROJECT


def state_path() -> Path:
    return project_dir() / "state.json"


def queue_path() -> Path:
    return project_dir() / "queue.json"


def logs_dir() -> Path:
    path = project_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def update_state(job_key: str, updates: dict[str, Any]) -> None:
    state = read_json(state_path(), {})
    if not isinstance(state, dict):
        state = {}
    job = dict(state.get(job_key, {}))
    job.update(updates)
    state[job_key] = job
    write_json(state_path(), state)


def profiled_ops() -> list[str]:
    io_root = project_dir() / "io" / "individual_ops"
    if not io_root.exists():
        return []
    return sorted(child.name for child in io_root.iterdir() if child.is_dir())


def generated_ops() -> set[str]:
    gen_root = project_dir() / "kernels" / "generated" / "individual_op_kernels"
    if not gen_root.exists():
        return set()
    ops = set()
    for child in gen_root.iterdir():
        if not child.is_dir():
            continue
        if any((child / marker).exists() for marker in ("success.cuda", "success.triton", "success.mps", "success.cpu")):
            ops.add(child.name)
    return ops


def max_node_id(op_name: str) -> int:
    db_path = project_dir() / "trees" / op_name / "nodes.db"
    if not db_path.exists():
        return -1
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) FROM nodes").fetchone()
    if not row or row[0] is None:
        return -1
    return int(row[0])


def arm_manifest_exists(arm: str) -> bool:
    artifact_root = REPO_ROOT / "paper_benchmarks" / "data_collection" / "artifacts" / MODEL_SLUG
    if not artifact_root.exists():
        return False
    for child in artifact_root.iterdir():
        if child.is_dir() and f"__{arm}__" in child.name and (child / "collection_manifest.json").exists():
            return True
    return False


def command_env(job_key: str) -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = str(Path(sys.executable).parent)
    path_parts = env.get("PATH", "").split(os.pathsep)
    if venv_bin and venv_bin not in path_parts:
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["KFORGE_STATE_PATH"] = str(state_path())
    env["KFORGE_JOB_KEY"] = job_key
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(job_key: str, log_name: str, cmd: list[str]) -> None:
    log_path = logs_dir() / log_name
    run_id = f"{job_key}:{int(time.time() * 1000)}"
    now = datetime.now().isoformat()
    update_state(
        job_key,
        {
            "status": "running",
            "control": "running",
            "active": True,
            "phase": "generating" if job_key == "generate" else "optimizing",
            "message": "Kernel generation started." if job_key == "generate" else "Kernel optimization started.",
            "progress_percent": 0.05,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "log": str(log_path),
            "command": cmd,
            "run_id": run_id,
        },
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[job] Started at {now}\n")
        log.write(f"[job] Command: {cmd}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=command_env(job_key),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        update_state(job_key, {"pid": proc.pid})
        assert proc.stdout is not None
        last_echo = 0.0
        for line in proc.stdout:
            log.write(line)
            log.flush()
            now = time.time()
            important = (
                line.startswith("[workflow")
                or line.startswith("[optimize-result]")
                or line.startswith("[attempts-summary]")
                or line.startswith("[n-kernels")
                or line.startswith("Optimizing:")
                or "Kernel generation completed" in line
                or "Kernel optimization completed" in line
                or "Traceback" in line
                or "ERROR" in line
                or "Failed" in line
            )
            if important or now - last_echo > 60:
                print(line[:500], end="" if line.endswith("\n") else "\n")
                last_echo = now
        rc = proc.wait()

    finished = datetime.now().isoformat()
    if rc != 0:
        update_state(
            job_key,
            {
                "status": "error",
                "control": "idle",
                "active": False,
                "phase": "error",
                "message": f"Job failed with exit code {rc}.",
                "progress_percent": 1.0,
                "updated_at": finished,
                "finished_at": finished,
                "returncode": rc,
            },
        )
        raise RuntimeError(f"{job_key} failed with exit code {rc}")

    update_state(
        job_key,
        {
            "status": "completed",
            "control": "idle",
            "active": False,
            "phase": "done",
            "message": "Kernel generation completed." if job_key == "generate" else "Kernel optimization completed.",
            "progress_percent": 1.0,
            "updated_at": finished,
            "finished_at": finished,
            "returncode": rc,
        },
    )


def collect_arm(arm: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "paper_benchmarks.data_collection.collect_zero_shot",
        "--project",
        PROJECT,
        "--model-slug",
        MODEL_SLUG,
        "--arm",
        arm,
    ]
    print(f"[collector] collecting {arm}")
    log_path = logs_dir() / f"collect_{arm}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[collector] Started at {datetime.now().isoformat()}\n")
        log.write(f"[collector] Command: {cmd}\n")
        log.flush()
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=command_env("collect"),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    print(f"[collector] wrote {log_path}")


def run_generate() -> None:
    ops = profiled_ops()
    missing = [op for op in ops if op not in generated_ops()]
    if not missing:
        print("[runner] zero-shot kernels already exist for all profiled ops.")
        return
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.workflow",
        "generate",
        "--project",
        PROJECT,
        "--ops",
        ",".join(missing),
        "--target-device",
        TARGET_DEVICE,
        "--benchmark",
        "--llm-model",
        LLM_MODEL,
        "--llm-provider",
        LLM_PROVIDER,
    ]
    run_logged("generate", "generate.log", cmd)


def run_optimize_to(target: int, workers: int) -> None:
    ops = profiled_ops()
    missing = [op for op in ops if max_node_id(op) < target]
    if not missing:
        print(f"[runner] optimization tree already reaches node {target} for all ops.")
        return

    min_existing = min(max_node_id(op) for op in missing)
    incremental_iterations = max(1, target - min_existing)
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.workflow",
        "optimize",
        "--project",
        PROJECT,
        "--ops",
        ",".join(missing),
        "--iterations",
        str(incremental_iterations),
        "--workers",
        str(workers),
        "--llm-model",
        LLM_MODEL,
        "--llm-provider",
        LLM_PROVIDER,
    ]
    run_logged("optimize", "optimize.log", cmd)

    still_missing = [op for op in ops if max_node_id(op) < target]
    if still_missing:
        details = {op: max_node_id(op) for op in still_missing}
        raise RuntimeError(f"optimization did not reach target {target}: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma INT4 collection arms.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force-collect", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not project_dir().exists():
        raise FileNotFoundError(project_dir())

    print(f"[runner] project={PROJECT} model_slug={MODEL_SLUG}")
    print(f"[runner] profiled_ops={profiled_ops()}")

    run_generate()
    if args.force_collect or not arm_manifest_exists("zero_shot"):
        collect_arm("zero_shot")
    else:
        print("[runner] zero_shot collection already exists.")

    for arm, target in ARMS[1:]:
        run_optimize_to(target, workers=args.workers)
        if args.force_collect or not arm_manifest_exists(arm):
            collect_arm(arm)
        else:
            print(f"[runner] {arm} collection already exists.")

    print("[runner] all requested Gemma INT4 arms are complete and collected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
