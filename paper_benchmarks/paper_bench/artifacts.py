from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from .schema import ArtifactModel, validate_artifact_payload


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    run_dir: Path
    raw_dir: Path
    metrics_dir: Path
    correctness_dir: Path
    reports_dir: Path
    logs_dir: Path


def default_runs_root() -> Path:
    return Path(__file__).resolve().parents[1] / "runs"


def make_run_id(timestamp_utc: str, model_id: str, suite_id: str) -> str:
    compact_ts = (
        timestamp_utc.replace(":", "")
        .replace("-", "")
        .replace("T", "_")
        .replace("Z", "")
    )
    return f"{compact_ts}_{model_id}_{suite_id}"


def create_run_layout(runs_root: str | Path, timestamp_utc: str, model_id: str, suite_id: str) -> RunLayout:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(timestamp_utc, model_id, suite_id)
    run_dir = root / run_id
    raw_dir = run_dir / "raw"
    metrics_dir = run_dir / "metrics"
    correctness_dir = run_dir / "correctness"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    for path in (run_dir, raw_dir, metrics_dir, correctness_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        run_id=run_id,
        run_dir=run_dir,
        raw_dir=raw_dir,
        metrics_dir=metrics_dir,
        correctness_dir=correctness_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )


def create_run_layout_for_dir(run_dir: str | Path) -> RunLayout:
    root = Path(run_dir)
    raw_dir = root / "raw"
    metrics_dir = root / "metrics"
    correctness_dir = root / "correctness"
    reports_dir = root / "reports"
    logs_dir = root / "logs"
    for path in (root, raw_dir, metrics_dir, correctness_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        run_id=root.name,
        run_dir=root,
        raw_dir=raw_dir,
        metrics_dir=metrics_dir,
        correctness_dir=correctness_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )


def write_commands_txt(run_dir: str | Path, command_line: list[str]) -> Path:
    path = Path(run_dir) / "commands.txt"
    path.write_text(shlex.join(command_line) + "\n", encoding="utf-8")
    return path


def write_json_artifact(path: str | Path, artifact: ArtifactModel) -> Path:
    payload = artifact.model_dump(mode="json")
    validated = validate_artifact_payload(payload)
    Path(path).write_text(
        json.dumps(validated.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return Path(path)


def load_json_artifact(path: str | Path) -> ArtifactModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return validate_artifact_payload(payload)
