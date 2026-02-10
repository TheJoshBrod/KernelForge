import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a test CGinS project.")
    parser.add_argument("--name", default="pipeline_test", help="Project name")
    parser.add_argument("--force", action="store_true", help="Overwrite if exists")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_src = repo_root / "benchmarks" / "models" / "cgins_mini.py"
    weights_src = repo_root / "benchmarks" / "models" / "cgins_mini_weights.pt"

    project_dir = repo_root / "projects" / args.name
    if project_dir.exists():
        if not args.force:
            print(f"Project already exists: {project_dir}")
            return 1
        shutil.rmtree(project_dir)

    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, project_dir / "model.py")
    if weights_src.exists():
        shutil.copy2(weights_src, project_dir / "weights.pt")

    config = {
        "validation_dir": "",
        "created_at": datetime.now().isoformat(),
        "generator": {
            "skip_ops": [],
            "only_ops": [],
            "max_ops": 0,
            "use_baseline_kernels": True,
            "use_baseline_as_template": True,
            "max_attempts": 8,
            "extra_validation_cases": 1,
        },
        "profile": {
            "allow_ops": [],
            "skip_ops": [],
            "skip_prefixes": [],
        },
    }
    (project_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"Created test project: {project_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
