#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill sha256 fields in an Apple Silicon study matrix JSON.")
    parser.add_argument("--matrix", required=True, help="Input matrix JSON path")
    parser.add_argument("--out", default="", help="Output path (defaults to overwrite input)")
    args = parser.parse_args()

    matrix_path = Path(args.matrix).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else matrix_path
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        models = payload.get("models")
    else:
        models = payload
    if not isinstance(models, list):
        raise SystemExit("Matrix must be a list or object with 'models' list.")

    updated = 0
    for item in models:
        if not isinstance(item, dict):
            continue
        model_path = Path(str(item.get("path", "")).strip()).expanduser().resolve()
        if not model_path.exists():
            raise SystemExit(f"Model missing: {model_path}")
        item["sha256"] = sha256_file(model_path)
        updated += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "models_updated": updated, "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
