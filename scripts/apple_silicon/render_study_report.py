#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apple_silicon.study import StudyError, render_study_plots


def main() -> int:
    parser = argparse.ArgumentParser(description="Render plot bundle for Apple Silicon study output")
    parser.add_argument(
        "--study-dir",
        required=True,
        help="Path to study output directory (contains ci_results.csv and paired_deltas.csv)",
    )
    args = parser.parse_args()

    study_dir = Path(args.study_dir).expanduser().resolve()
    try:
        result = render_study_plots(study_dir)
    except StudyError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"Unexpected error: {exc}"}, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
