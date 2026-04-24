from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper_benchmarks.paper_bench.cast_export import (
    export_cast_package,
    inspect_cast_package,
)
from paper_benchmarks.paper_bench.provenance import safe_sha256_path


SECRET_PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_root() -> Path:
    return _REPO_ROOT


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.upper() in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"}:
                clean[key_text] = "[REDACTED_SECRET]"
            else:
                clean[key_text] = redact(item)
        return clean
    return value


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return redact(json.loads(raw))
    except Exception as exc:
        return {"parse_error": str(exc), "raw": redact_text(raw)}


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return redact_text(path.read_text(encoding="utf-8", errors="replace"))


def sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path, *, root: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    try:
        relpath = path.relative_to(root).as_posix()
    except ValueError:
        relpath = str(path)
    return {
        "path": str(path),
        "repo_relpath": relpath,
        "exists": exists,
        "size_bytes": int(stat.st_size) if stat else None,
        "mtime": (
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            if stat
            else None
        ),
        "sha256": sha256_path(path) if exists and path.is_file() else None,
    }


def run_git(args: list[str], *, root: Path) -> Any:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "returncode": proc.returncode,
        "stdout": redact_text(proc.stdout).splitlines(),
        "stderr": redact_text(proc.stderr).splitlines(),
    }


def sqlite_dump(path: Path, *, max_rows_per_table: int = 10000) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "tables": {}}
    result: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
        "tables": {},
    }
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for table in tables:
            table_name = str(table["name"])
            if table_name.startswith("sqlite_"):
                continue
            count = conn.execute(f'SELECT COUNT(*) AS n FROM "{table_name}"').fetchone()["n"]
            rows = conn.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (max_rows_per_table,)).fetchall()
            result["tables"][table_name] = {
                "schema": table["sql"],
                "row_count": int(count),
                "rows_truncated": int(count) > max_rows_per_table,
                "rows": [redact(dict(row)) for row in rows],
            }
    finally:
        conn.close()
    return result


def normalize_op_name(op_name: str) -> str:
    return (
        op_name.replace("torch.nn.functional.", "torch_nn_functional_")
        .replace(".", "_")
        .replace("::", "_")
    )


def op_display_name(op_dir_name: str) -> str:
    prefix = "torch_nn_functional_"
    if op_dir_name.startswith(prefix):
        return "torch.nn.functional." + op_dir_name[len(prefix):]
    return op_dir_name


def captured_input_refs(project_dir: Path, *, root: Path) -> dict[str, list[dict[str, Any]]]:
    io_root = project_dir / "io" / "individual_ops"
    captured: dict[str, list[dict[str, Any]]] = {}
    if not io_root.exists():
        return captured
    for op_dir in sorted(child for child in io_root.iterdir() if child.is_dir()):
        captured[op_dir.name] = [
            file_ref(path, root=root)
            for path in sorted(op_dir.glob("entry_*.pt"))
        ]
    return captured


def generated_op_artifacts(project_dir: Path, *, root: Path) -> dict[str, Any]:
    gen_root = project_dir / "kernels" / "generated" / "individual_op_kernels"
    tree_root = project_dir / "trees"
    artifacts: dict[str, Any] = {}
    op_names: set[str] = set()
    if gen_root.exists():
        op_names.update(child.name for child in gen_root.iterdir() if child.is_dir())
    if tree_root.exists():
        op_names.update(child.name for child in tree_root.iterdir() if child.is_dir())

    for op_name in sorted(op_names):
        gen_dir = gen_root / op_name
        tree_dir = tree_root / op_name
        generated_kernel = gen_dir / "kernel.cu"
        tree_kernel_refs = []
        if (tree_dir / "kernels").exists():
            tree_kernel_refs = [
                file_ref(path, root=root)
                for path in sorted((tree_dir / "kernels").glob("*.cu"))
            ]
        attempt_files = []
        attempts_dir = gen_dir / "attempts"
        if attempts_dir.exists():
            for path in sorted(attempts_dir.iterdir()):
                if path.is_file():
                    item = file_ref(path, root=root)
                    if path.suffix in {".json", ".txt"}:
                        item["content"] = read_json(path) if path.suffix == ".json" else read_text(path)
                    attempt_files.append(item)

        artifacts[op_name] = {
            "op": op_name,
            "display_op": op_display_name(op_name),
            "generated_dir": str(gen_dir) if gen_dir.exists() else None,
            "success_cuda": (gen_dir / "success.cuda").exists(),
            "generated_kernel": file_ref(generated_kernel, root=root),
            "generated_kernel_content": read_text(generated_kernel),
            "cuda_cu": file_ref(gen_dir / "cuda.cu", root=root),
            "main_cpp": file_ref(gen_dir / "main.cpp", root=root),
            "attempt_files": attempt_files,
            "per_op_llm_usage_db": sqlite_dump(gen_dir / "llm_usage.db"),
            "tree_generated_root": read_json(tree_dir / "generated_root.json"),
            "tree_kernel_files": tree_kernel_refs,
            "tree_nodes_db": sqlite_dump(tree_dir / "nodes.db"),
        }
    return artifacts


def attempted_generation_ops(project_dir: Path) -> list[str]:
    gen_root = project_dir / "kernels" / "generated" / "individual_op_kernels"
    if not gen_root.exists():
        return []
    return sorted(child.name for child in gen_root.iterdir() if child.is_dir())


def successful_zero_shot_kernel_map(project_dir: Path) -> dict[str, str]:
    gen_root = project_dir / "kernels" / "generated" / "individual_op_kernels"
    if not gen_root.exists():
        return {}
    selected: dict[str, str] = {}
    for op_dir in sorted(child for child in gen_root.iterdir() if child.is_dir()):
        kernel = op_dir / "kernel.cu"
        if kernel.exists() and (op_dir / "success.cuda").exists():
            selected[op_dir.name] = str(kernel.resolve())
    return selected


def profiled_ops(project_dir: Path) -> list[str]:
    ops: set[str] = set()
    summary = read_json(project_dir / "io" / "summary.json")
    if isinstance(summary, dict):
        for op_name in (summary.get("op_counts") or {}).keys():
            ops.add(normalize_op_name(str(op_name)))
    io_root = project_dir / "io" / "individual_ops"
    if io_root.exists():
        ops.update(child.name for child in io_root.iterdir() if child.is_dir())
    return sorted(ops)


def export_variant(
    *,
    project_dir: Path,
    root: Path,
    output_path: Path,
    variant: str,
    selected_kernels: dict[str, str] | None,
    allow_native_package: bool,
) -> dict[str, Any]:
    result = export_cast_package(
        project_dir,
        selected_kernels=selected_kernels or {},
        allow_operator_only=True,
        allow_micro_only=False,
        unsafe_override=False,
        allow_native_package=allow_native_package,
        repo_root=root,
        output_path=output_path,
    )
    inspection = inspect_cast_package(output_path)
    return redact(
        {
            "variant": variant,
            "export_result": result,
            "inspection": inspection,
            "cast_file": file_ref(output_path, root=root),
        }
    )


def dispatch_table(*, profiled: list[str], export_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = export_record.get("export_result", {}).get("manifest", {})
    selected_by_op = manifest.get("selected_kernel_by_op", {}) if isinstance(manifest, dict) else {}
    selected_metadata = manifest.get("selected_kernel_metadata", {}) if isinstance(manifest, dict) else {}
    table: dict[str, dict[str, Any]] = {}
    for op_name in profiled:
        if op_name in selected_by_op:
            table[op_name] = {
                "dispatch": "forge",
                "kernel": selected_by_op.get(op_name),
                "metadata": selected_metadata.get(op_name, {}),
            }
        else:
            table[op_name] = {
                "dispatch": "torch",
                "kernel": None,
                "metadata": None,
            }
    return table


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(redact(record), sort_keys=True) + "\n")


def build_common(
    *,
    record_type: str,
    model_slug: str,
    project_name: str,
    arm: str,
    run_id: str,
    root: Path,
    source_paths: list[Path],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": record_type,
        "model_slug": model_slug,
        "project_name": project_name,
        "arm": arm,
        "run_id": run_id,
        "created_at": utc_now(),
        "source_paths": [str(path) for path in source_paths],
        "source_hashes": {
            str(path): safe_sha256_path(path) if path.exists() and path.is_file() else None
            for path in source_paths
        },
        "collector": {
            "script": str(Path(__file__).resolve()),
            "git_commit": (run_git(["rev-parse", "HEAD"], root=root).get("stdout") or [""])[0],
        },
    }


def assert_generation_finished(project_dir: Path) -> None:
    state = read_json(project_dir / "state.json")
    if not isinstance(state, dict):
        return
    generate = state.get("generate")
    if isinstance(generate, dict) and generate.get("active"):
        raise RuntimeError(
            "Generation is still active; refusing to collect zero-shot exports before it finishes."
        )


def collect(args: argparse.Namespace) -> Path:
    root = repo_root()
    project_dir = root / "kernels" / "projects" / args.project
    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_dir}")
    if not args.allow_running:
        assert_generation_finished(project_dir)

    model_slug = args.model_slug or args.project
    arm = "zero_shot"
    run_id = args.run_id or f"{model_slug}__{arm}__{compact_timestamp()}"
    model_file = root / args.output_file if args.output_file else root / "paper_benchmarks" / "data_collection" / "models" / f"{model_slug}.jsonl"
    artifact_dir = root / args.artifact_dir if args.artifact_dir else root / "paper_benchmarks" / "data_collection" / "artifacts" / model_slug / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [
        project_dir / "config.json",
        project_dir / "state.json",
        project_dir / "queue.json",
        project_dir / "io" / "summary.json",
        project_dir / "benchmarks" / "op_benchmarks.json",
        project_dir / "benchmarks" / "torch_baseline_cache.json",
        project_dir / "logs" / "profile.log",
        project_dir / "logs" / "generate.log",
    ]

    profiled = profiled_ops(project_dir)
    attempted_ops = attempted_generation_ops(project_dir)
    generated_map = successful_zero_shot_kernel_map(project_dir)
    missing_full_forge_ops = [op for op in profiled if op not in generated_map]
    failed_attempted_ops = [op for op in attempted_ops if op not in generated_map]
    not_attempted_profiled_ops = [op for op in profiled if op not in attempted_ops]
    config_payload = read_json(project_dir / "config.json")
    generation_config = (
        config_payload.get("generation", {})
        if isinstance(config_payload, dict) and isinstance(config_payload.get("generation"), dict)
        else {}
    )

    full_export = export_variant(
        project_dir=project_dir,
        root=root,
        output_path=artifact_dir / f"{model_slug}__{arm}__full_forge.cast",
        variant="full_forge",
        selected_kernels=generated_map,
        allow_native_package=True,
    )
    mixed_export = export_variant(
        project_dir=project_dir,
        root=root,
        output_path=artifact_dir / f"{model_slug}__{arm}__mixed_forge.cast",
        variant="mixed_forge",
        selected_kernels=None,
        allow_native_package=True,
    )

    full_dispatch = dispatch_table(profiled=profiled, export_record=full_export)
    mixed_dispatch = dispatch_table(profiled=profiled, export_record=mixed_export)
    exports_differ = (
        full_export["cast_file"]["sha256"] != mixed_export["cast_file"]["sha256"]
        or full_dispatch != mixed_dispatch
    )

    common_kwargs = {
        "model_slug": model_slug,
        "project_name": args.project,
        "arm": arm,
        "run_id": run_id,
        "root": root,
    }

    records: list[dict[str, Any]] = []
    records.append(
        {
            **build_common(record_type="model_project_snapshot", source_paths=source_paths[:3], **common_kwargs),
            "payload": {
                "project_dir": str(project_dir),
                "config": read_json(project_dir / "config.json"),
                "state": read_json(project_dir / "state.json"),
                "queue": read_json(project_dir / "queue.json"),
                "git": {
                    "rev_parse_head": run_git(["rev-parse", "HEAD"], root=root),
                    "branch": run_git(["branch", "--show-current"], root=root),
                    "status_short": run_git(["status", "--short"], root=root),
                    "diff_stat": run_git(["diff", "--stat", "--compact-summary", "HEAD"], root=root),
                },
            },
        }
    )
    records.append(
        {
            **build_common(record_type="profile_snapshot", source_paths=[project_dir / "io" / "summary.json", project_dir / "logs" / "profile.log"], **common_kwargs),
            "payload": {
                "summary": read_json(project_dir / "io" / "summary.json"),
                "profile_log": read_text(project_dir / "logs" / "profile.log"),
                "profiled_ops": profiled,
                "captured_inputs": captured_input_refs(project_dir, root=root),
            },
        }
    )
    records.append(
        {
            **build_common(record_type="forge_operator_benchmark", source_paths=[project_dir / "benchmarks" / "op_benchmarks.json", project_dir / "benchmarks" / "torch_baseline_cache.json"], **common_kwargs),
            "payload": {
                "op_benchmarks": read_json(project_dir / "benchmarks" / "op_benchmarks.json"),
                "torch_baseline_cache": read_json(project_dir / "benchmarks" / "torch_baseline_cache.json"),
            },
        }
    )
    records.append(
        {
            **build_common(record_type="generation_attempt", source_paths=[project_dir / "logs" / "generate.log"], **common_kwargs),
            "payload": {
                "generate_log": read_text(project_dir / "logs" / "generate.log"),
                "generated_artifacts": generated_op_artifacts(project_dir, root=root),
            },
        }
    )
    records.append(
        {
            **build_common(record_type="llm_usage", source_paths=[project_dir / "llm_usage.db"], **common_kwargs),
            "payload": {
                "project_llm_usage_db": sqlite_dump(project_dir / "llm_usage.db"),
                "per_op_usage_dbs": {
                    op: sqlite_dump(project_dir / "kernels" / "generated" / "individual_op_kernels" / op / "llm_usage.db")
                    for op in sorted(generated_op_artifacts(project_dir, root=root))
                },
                "generation_config": generation_config,
                "reasoning_effort": generation_config.get("reasoning_effort"),
            },
        }
    )
    for export_record, export_kind, dispatch in [
        (full_export, "full_forge", full_dispatch),
        (mixed_export, "mixed_forge", mixed_dispatch),
    ]:
        records.append(
            {
                **build_common(record_type="cast_export", source_paths=[Path(export_record["cast_file"]["path"])], **common_kwargs),
                "cast_export_id": f"{run_id}__{export_kind}",
                "payload": {
                    "export_kind": export_kind,
                    "export": export_record,
                    "dispatch_by_profiled_op": dispatch,
                    "forge_ops": [op for op, item in dispatch.items() if item["dispatch"] == "forge"],
                    "torch_fallback_ops": [op for op, item in dispatch.items() if item["dispatch"] == "torch"],
                    "missing_full_forge_ops": missing_full_forge_ops if export_kind == "full_forge" else [],
                    "zero_shot_generated_kernel_map": generated_map,
                },
            }
        )
    records.append(
        {
            **build_common(record_type="arm_summary", source_paths=source_paths, **common_kwargs),
            "payload": {
                "arm_complete": not missing_full_forge_ops,
                "arm_collection_complete": True,
                "all_profiled_ops_have_zero_shot_kernel": not missing_full_forge_ops,
                "zero_shot_generated_kernel_count": len(generated_map),
                "zero_shot_failed_attempted_ops": failed_attempted_ops,
                "zero_shot_not_attempted_profiled_ops": not_attempted_profiled_ops,
                "optimization_started": False,
                "profiled_ops": profiled,
                "attempted_zero_shot_ops": attempted_ops,
                "zero_shot_generated_ops": sorted(generated_map),
                "missing_full_forge_ops": missing_full_forge_ops,
                "full_forge_cast": full_export["cast_file"],
                "mixed_forge_cast": mixed_export["cast_file"],
                "exports_differ": exports_differ,
                "full_forge_dispatch_by_profiled_op": full_dispatch,
                "mixed_forge_dispatch_by_profiled_op": mixed_dispatch,
                "artifact_dir": str(artifact_dir),
            },
        }
    )

    append_jsonl(model_file, records)
    manifest_path = artifact_dir / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            redact(
                {
                    "ok": True,
                    "model_file": str(model_file),
                    "artifact_dir": str(artifact_dir),
                    "run_id": run_id,
                    "records_appended": len(records),
                    "full_forge_cast": full_export["cast_file"],
                    "mixed_forge_cast": mixed_export["cast_file"],
                    "exports_differ": exports_differ,
                    "full_forge_dispatch_by_profiled_op": full_dispatch,
                    "mixed_forge_dispatch_by_profiled_op": mixed_dispatch,
                    "missing_full_forge_ops": missing_full_forge_ops,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(redact(read_json(manifest_path)), indent=2, sort_keys=True))
    return model_file


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect zero-shot Forge export data for one model.")
    parser.add_argument("--project", required=True, help="Kernel Forge project directory name.")
    parser.add_argument("--model-slug", default="", help="Canonical model slug for data_collection/models.")
    parser.add_argument("--run-id", default="", help="Stable run id. Defaults to timestamped zero-shot id.")
    parser.add_argument("--output-file", default="", help="Repo-relative or absolute JSONL output file.")
    parser.add_argument("--artifact-dir", default="", help="Repo-relative or absolute artifact directory.")
    parser.add_argument("--allow-running", action="store_true", help="Collect even if state.json says generation is active.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        collect(args)
    except Exception as exc:
        print(f"[collect_zero_shot] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
