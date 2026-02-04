import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _update_state(state_path: Path, job_key: str, updates: dict) -> None:
    state = _load_state(state_path)
    job_state = dict(state.get(job_key, {}))
    job_state.update(updates)
    state[job_key] = job_state
    _save_state(state_path, state)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_docker_cmd(
    cmd: list[str],
    *,
    state_path: Path,
    job_key: str,
    project_dir: Path,
    repo_root: Path,
    image_override: str | None = None,
) -> list[str]:
    image = image_override or os.environ.get("CGINS_DOCKER_IMAGE", "cgins-worker:latest")
    gpus = os.environ.get("CGINS_DOCKER_GPUS", "all")
    container_repo = os.environ.get("CGINS_DOCKER_REPO_MOUNT", "/work/cgins")
    container_project = os.environ.get("CGINS_DOCKER_PROJECT_MOUNT", "/work/project")
    container_datasets = os.environ.get("CGINS_DOCKER_DATASETS_MOUNT", "/work/datasets")

    project_name = project_dir.name
    container_project_in_repo = f"{container_repo}/projects/{project_name}"

    rel_state = None
    try:
        rel_state = state_path.relative_to(project_dir)
    except Exception:
        rel_state = Path(state_path.name)
    container_state_path = str(Path(container_project_in_repo) / rel_state)

    docker_cmd: list[str] = ["docker", "run", "--rm", "--gpus", gpus]

    if _bool_env("CGINS_DOCKER_READ_ONLY", True):
        docker_cmd.append("--read-only")
    docker_cmd += ["--tmpfs", os.environ.get("CGINS_DOCKER_TMPFS", "/tmp:rw,noexec,nosuid,size=4g")]

    if _bool_env("CGINS_DOCKER_DROP_CAPS", True):
        docker_cmd += ["--cap-drop=ALL", "--security-opt=no-new-privileges"]

    network = os.environ.get("CGINS_DOCKER_NETWORK", "").strip()
    if network:
        docker_cmd += ["--network", network]

    memory = os.environ.get("CGINS_DOCKER_MEMORY", "").strip()
    if memory:
        docker_cmd += ["--memory", memory]

    cpus = os.environ.get("CGINS_DOCKER_CPUS", "").strip()
    if cpus:
        docker_cmd += ["--cpus", cpus]

    pids_limit = os.environ.get("CGINS_DOCKER_PIDS_LIMIT", "").strip()
    if pids_limit:
        docker_cmd += ["--pids-limit", pids_limit]

    docker_cmd += [
        "--workdir",
        container_repo,
        "-v",
        f"{project_dir}:{container_project}:rw",
        "-v",
        f"{repo_root}:{container_repo}:{'ro' if _bool_env('CGINS_DOCKER_REPO_RO', True) else 'rw'}",
        "-v",
        f"{project_dir}:{container_project_in_repo}:rw",
    ]

    datasets_dir = os.environ.get("CGINS_DOCKER_DATASETS_DIR") or os.environ.get("CGINS_DATASETS_DIR")
    if datasets_dir:
        docker_cmd += ["-v", f"{datasets_dir}:{container_datasets}:ro"]

    env_allowlist = {
        "LLM_PROVIDER",
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_USE_RESPONSES",
        "OPENAI_MAX_OUTPUT_TOKENS",
        "OPENAI_MAX_TOKENS",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "ANTHROPIC_API_KEY",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    }

    docker_env = {
        "CGINS_STATE_PATH": container_state_path,
        "CGINS_JOB_KEY": job_key,
        "CGINS_PROJECT_DIR": container_project_in_repo,
    }
    project_tmp = project_dir / ".tmp"
    try:
        project_tmp.mkdir(parents=True, exist_ok=True)
    except Exception:
        project_tmp = None
    if project_tmp is not None:
        container_tmp = f"{container_project_in_repo}/.tmp"
        docker_env["TMPDIR"] = container_tmp
        docker_env["TMP"] = container_tmp
        docker_env["TEMP"] = container_tmp

    for key in env_allowlist:
        val = os.environ.get(key)
        if val:
            docker_env[key] = val

    for key, val in os.environ.items():
        if key.startswith("CGINS_") and key not in docker_env:
            docker_env[key] = val

    for key, val in docker_env.items():
        if val is None:
            continue
        docker_cmd += ["-e", f"{key}={val}"]

    extra_args = os.environ.get("CGINS_DOCKER_EXTRA_ARGS", "").strip()
    if extra_args:
        docker_cmd += shlex.split(extra_args)

    docker_cmd.append(image)
    docker_cmd += cmd
    return docker_cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a job and update state.json")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--use-container", action="store_true")
    parser.add_argument("--container-image", default=None)
    args, cmd = parser.parse_known_args()

    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    state_path = Path(args.state_path).resolve()
    log_path = Path(args.log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    project_dir = state_path.parent
    repo_root = Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve()

    started_at = time.time()
    _update_state(
        state_path,
        args.job_key,
        {
            "status": "running",
            "started_at": started_at,
            "log": str(log_path),
        },
    )

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            env = os.environ.copy()
            env["CGINS_STATE_PATH"] = str(state_path)
            env["CGINS_JOB_KEY"] = str(args.job_key)
            project_tmp = project_dir / ".tmp"
            try:
                project_tmp.mkdir(parents=True, exist_ok=True)
            except Exception:
                project_tmp = None
            if project_tmp is not None:
                env["TMPDIR"] = str(project_tmp)
                env["TMP"] = str(project_tmp)
                env["TEMP"] = str(project_tmp)

            use_container = args.use_container or _bool_env("CGINS_USE_CONTAINER", False)
            if sys.platform == "darwin":
                use_container = False

            if use_container:
                docker_cmd = _build_docker_cmd(
                    cmd,
                    state_path=state_path,
                    job_key=str(args.job_key),
                    project_dir=project_dir,
                    repo_root=repo_root,
                    image_override=args.container_image,
                )
                proc = subprocess.Popen(
                    docker_cmd,
                    cwd=str(repo_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            else:
                if sys.platform == "darwin":
                    print("macOS detected; running job locally (Docker disabled).")
                proc = subprocess.Popen(
                    cmd,
                    cwd=args.cwd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            return_code = proc.wait()
    except Exception as exc:
        _update_state(
            state_path,
            args.job_key,
            {
                "status": "error",
                "finished_at": time.time(),
                "error": str(exc),
            },
        )
        return 1

    status = "success" if return_code == 0 else "error"
    try:
        job_state = _load_state(state_path).get(args.job_key, {})
        if str(job_state.get("control", "")).lower() == "cancelled":
            status = "cancelled"
    except Exception:
        pass
    _update_state(
        state_path,
        args.job_key,
        {
            "status": status,
            "finished_at": time.time(),
            "return_code": return_code,
        },
    )
    return return_code


if __name__ == "__main__":
    sys.exit(main())
