import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except Exception:
        return


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _update_state(path: Path, job_key: str, updates: dict) -> None:
    state = _load_state(path)
    job_state = dict(state.get(job_key, {}))
    job_state.update(updates)
    state[job_key] = job_state
    _save_state(path, state)


def _load_config(project_dir: Path) -> dict:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(project_dir: Path, config: dict) -> None:
    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _decode_b64_file(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    data = _read_text(src)
    if not data:
        return False
    try:
        decoded = base64.b64decode(data)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(decoded)
        return True
    except Exception:
        return False


def _extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    if not zip_path.exists():
        return False
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as zipf:
            zipf.extractall(dest_dir)
        return True
    except Exception:
        return False


def _start_profile(project_dir: Path, repo_root: Path) -> None:
    state_path = project_dir / "state.json"
    state = _load_state(state_path)
    profile_state = state.get("profile", {})
    if str(profile_state.get("status", "")).lower() != "queued":
        return

    runner = repo_root / "scripts" / "run_job.py"
    log_path = project_dir / "logs" / "profile.log"
    cmd = [
        sys.executable,
        str(runner),
        "--state-path",
        str(state_path),
        "--job-key",
        "profile",
        "--log-path",
        str(log_path),
        "--cwd",
        str(repo_root),
        "--",
        sys.executable,
        str(repo_root / "benchmarks" / "profiler" / "profile_project.py"),
        "--project",
        project_dir.name,
    ]
    subprocess.Popen(cmd, cwd=str(repo_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare project assets asynchronously.")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument("--weights-b64-path", type=str, default="")
    parser.add_argument("--validation-b64-path", type=str, default="")
    parser.add_argument("--validation-name-path", type=str, default="")
    args = parser.parse_args()

    if args.project_dir:
        project_dir = Path(args.project_dir).resolve()
    elif args.project:
        project_dir = (Path.cwd() / "projects" / args.project).resolve()
    else:
        raise RuntimeError("Missing --project or --project-dir")

    repo_root = project_dir.parent.parent
    state_path = project_dir / "state.json"

    weights_b64 = Path(args.weights_b64_path) if args.weights_b64_path else None
    validation_b64 = Path(args.validation_b64_path) if args.validation_b64_path else None
    validation_name_path = Path(args.validation_name_path) if args.validation_name_path else None

    _update_state(state_path, "prepare", {"message": "Decoding weights"})
    if weights_b64 and weights_b64.exists():
        weights_path = project_dir / "weights.pt"
        if _decode_b64_file(weights_b64, weights_path):
            _safe_unlink(weights_b64)

    _update_state(state_path, "prepare", {"message": "Unpacking validation"})
    if validation_b64 and validation_b64.exists():
        data_dir = project_dir / "data"
        zip_path = data_dir / "validation.zip"
        if _decode_b64_file(validation_b64, zip_path):
            _safe_unlink(validation_b64)
            validation_dir = data_dir / "validation"
            if _extract_zip(zip_path, validation_dir):
                config = _load_config(project_dir)
                config["validation_dir"] = os.path.relpath(validation_dir, project_dir)
                _save_config(project_dir, config)
        _safe_unlink(zip_path)
    if validation_name_path and validation_name_path.exists():
        _safe_unlink(validation_name_path)

    _update_state(state_path, "prepare", {"message": "Starting profile"})
    _start_profile(project_dir, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
