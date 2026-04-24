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


def _empty_usage_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }


def _add_usage(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    input_tokens = int(row.get("input_tokens") or 0)
    output_tokens = int(row.get("output_tokens") or 0)
    reasoning_tokens = int(row.get("reasoning_tokens") or 0)
    bucket["calls"] += 1
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["reasoning_tokens"] += reasoning_tokens
    bucket["total_tokens"] += input_tokens + output_tokens + reasoning_tokens
    bucket["input_cost_usd"] += float(row.get("input_cost_usd") or 0.0)
    bucket["output_cost_usd"] += float(row.get("output_cost_usd") or 0.0)
    bucket["total_cost_usd"] += float(row.get("total_cost_usd") or 0.0)


def _round_usage(bucket: dict[str, Any]) -> dict[str, Any]:
    out = dict(bucket)
    for key in ["input_cost_usd", "output_cost_usd", "total_cost_usd"]:
        out[key] = round(float(out.get(key) or 0.0), 12)
    return out


def llm_usage_summary(
    path: Path,
    *,
    operator_hint: str | None = None,
    row_filter: Any | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    result: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
        "totals": _empty_usage_bucket(),
        "by_op": {},
        "by_op_and_step": {},
        "by_provider_model": {},
        "per_call_rows": [],
    }
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(llm_calls)").fetchall()
            if row["name"]
        }
        if not columns:
            return {**result, "error": "missing llm_calls table"}
        desired_columns = [
            "id",
            "ts",
            "job_key",
            "operator",
            "step_type",
            "iteration",
            "attempt",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
        ]
        select_columns = [column for column in desired_columns if column in columns]
        order_column = "id" if "id" in columns else "ts"
        rows = conn.execute(
            f"SELECT {', '.join(select_columns)} FROM llm_calls ORDER BY {order_column}"
        ).fetchall()
    finally:
        conn.close()

    for raw_row in rows:
        row = redact(dict(raw_row))
        if row_filter is not None and not row_filter(row):
            continue
        operator = str(row.get("operator") or operator_hint or "unknown")
        row["operator"] = operator
        step = str(row.get("step_type") or "unknown")
        provider = str(row.get("provider") or "unknown")
        model = str(row.get("model") or "unknown")
        provider_model = f"{provider}/{model}"
        row["total_tokens"] = (
            int(row.get("input_tokens") or 0)
            + int(row.get("output_tokens") or 0)
            + int(row.get("reasoning_tokens") or 0)
        )
        result["per_call_rows"].append(row)
        _add_usage(result["totals"], row)

        by_op = result["by_op"].setdefault(operator, _empty_usage_bucket())
        _add_usage(by_op, row)

        by_step_root = result["by_op_and_step"].setdefault(operator, {})
        by_step = by_step_root.setdefault(step, _empty_usage_bucket())
        _add_usage(by_step, row)

        by_provider_model = result["by_provider_model"].setdefault(
            provider_model,
            _empty_usage_bucket(),
        )
        _add_usage(by_provider_model, row)

    result["totals"] = _round_usage(result["totals"])
    result["by_op"] = {
        key: _round_usage(value)
        for key, value in sorted(result["by_op"].items())
    }
    result["by_op_and_step"] = {
        op: {
            step: _round_usage(bucket)
            for step, bucket in sorted(steps.items())
        }
        for op, steps in sorted(result["by_op_and_step"].items())
    }
    result["by_provider_model"] = {
        key: _round_usage(value)
        for key, value in sorted(result["by_provider_model"].items())
    }
    return result


def arm_usage_filter(arm: str) -> Any:
    if arm == "zero_shot":
        return lambda row: str(row.get("job_key") or "") == "generate"

    match = re.fullmatch(r"optimize_(\d+)", arm)
    if match:
        max_iteration = int(match.group(1))

        def _filter(row: dict[str, Any]) -> bool:
            if str(row.get("job_key") or "") != "optimize":
                return False
            iteration = row.get("iteration")
            try:
                return int(iteration) <= max_iteration
            except Exception:
                return True

        return _filter

    return lambda row: True


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


def optimize_arm_limit(arm: str) -> int | None:
    match = re.fullmatch(r"optimize_(\d+)", arm)
    return int(match.group(1)) if match else None


def _tree_kernel_path(project_dir: Path, op_name: str, raw_code: str | None) -> Path | None:
    if not raw_code:
        return None
    candidate = Path(str(raw_code))
    if candidate.is_absolute():
        return candidate
    tree_root = project_dir / "trees"
    for base in (tree_root, project_dir):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return tree_root / candidate


def tree_node_rows(project_dir: Path, op_name: str) -> list[dict[str, Any]]:
    db_path = project_dir / "trees" / op_name / "nodes.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
    finally:
        conn.close()
    return [redact(dict(row)) for row in rows]


def selected_tree_kernel_map(project_dir: Path, *, arm: str) -> tuple[dict[str, str], dict[str, Any]]:
    max_node_id = optimize_arm_limit(arm)
    tree_root = project_dir / "trees"
    selected: dict[str, str] = {}
    details: dict[str, Any] = {}
    if not tree_root.exists():
        return selected, details

    for tree_dir in sorted(child for child in tree_root.iterdir() if child.is_dir()):
        op_name = tree_dir.name
        rows = tree_node_rows(project_dir, op_name)
        eligible = []
        for row in rows:
            value = row.get("value")
            code = row.get("code")
            node_id = row.get("id")
            try:
                node_id_int = int(node_id)
            except Exception:
                continue
            if max_node_id is not None and node_id_int > max_node_id:
                continue
            if value is None or code is None:
                continue
            try:
                value_float = float(value)
            except Exception:
                continue
            kernel_path = _tree_kernel_path(project_dir, op_name, str(code))
            if kernel_path is None or not kernel_path.exists():
                continue
            eligible.append((value_float, node_id_int, row, kernel_path))

        if not eligible:
            details[op_name] = {
                "selected": None,
                "reason": "no valid tree kernel found for arm",
                "node_count": len(rows),
                "max_node_id": max_node_id,
            }
            continue

        value_float, node_id_int, row, kernel_path = min(eligible, key=lambda item: (item[0], item[1]))
        selected[op_name] = str(kernel_path.resolve())
        latest_valid = max(eligible, key=lambda item: item[1])
        details[op_name] = {
            "selected": {
                "node_id": node_id_int,
                "value_ms": value_float,
                "code": row.get("code"),
                "kernel_path": str(kernel_path.resolve()),
                "kernel_file": file_ref(kernel_path, root=repo_root()),
                "selection_reason": "best valid tree kernel at arm checkpoint",
            },
            "latest_valid": {
                "node_id": latest_valid[1],
                "value_ms": latest_valid[0],
                "code": latest_valid[2].get("code"),
                "kernel_path": str(latest_valid[3].resolve()),
            },
            "valid_node_count": len(eligible),
            "node_count": len(rows),
            "max_node_id": max_node_id,
        }
    return selected, details


def optimize_queue_summary(project_dir: Path) -> dict[str, Any]:
    queue = read_json(project_dir / "queue.json")
    tasks = queue.get("active_tasks", {}) if isinstance(queue, dict) else {}
    by_op: dict[str, list[dict[str, Any]]] = {}
    if isinstance(tasks, dict):
        for key, task in tasks.items():
            if not isinstance(task, dict) or task.get("tag") != "[OPT]":
                continue
            op_name = str(task.get("op_name") or "unknown")
            by_op.setdefault(op_name, []).append({"task_key": key, **task})
    return {
        "tasks_by_op": {
            op: sorted(items, key=lambda item: str(item.get("task_key") or ""))
            for op, items in sorted(by_op.items())
        },
        "failed_tasks": [
            {"task_key": item.get("task_key"), **item}
            for items in by_op.values()
            for item in items
            if str(item.get("status") or "") == "Failed"
        ],
    }


def optimization_tree_summary(project_dir: Path, *, arm: str) -> dict[str, Any]:
    _, selection_details = selected_tree_kernel_map(project_dir, arm=arm)
    tree_root = project_dir / "trees"
    ops = sorted(child.name for child in tree_root.iterdir() if child.is_dir()) if tree_root.exists() else []
    return {
        "arm": arm,
        "max_node_id": optimize_arm_limit(arm),
        "selection_by_op": selection_details,
        "queue_summary": optimize_queue_summary(project_dir),
        "tree_nodes_by_op": {
            op: tree_node_rows(project_dir, op)
            for op in ops
        },
    }


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


def assert_jobs_finished(project_dir: Path) -> None:
    state = read_json(project_dir / "state.json")
    if not isinstance(state, dict):
        return
    generate = state.get("generate")
    if isinstance(generate, dict) and generate.get("active"):
        raise RuntimeError(
            "Generation is still active; refusing to collect exports before it finishes."
        )
    optimize = state.get("optimize")
    if isinstance(optimize, dict) and optimize.get("active"):
        raise RuntimeError(
            "Optimization is still active; refusing to collect exports before it finishes."
        )


def collect(args: argparse.Namespace) -> Path:
    root = repo_root()
    project_dir = root / "kernels" / "projects" / args.project
    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_dir}")
    if not args.allow_running:
        assert_jobs_finished(project_dir)

    model_slug = args.model_slug or args.project
    arm = args.arm or "zero_shot"
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
        project_dir / "logs" / "optimize.log",
        project_dir / "logs" / "queue_debug.log",
    ]

    profiled = profiled_ops(project_dir)
    attempted_ops = attempted_generation_ops(project_dir)
    zero_shot_generated_map = successful_zero_shot_kernel_map(project_dir)
    if arm == "zero_shot":
        selected_kernel_map = zero_shot_generated_map
        optimization_summary = None
    else:
        selected_kernel_map, _selection_details = selected_tree_kernel_map(project_dir, arm=arm)
        optimization_summary = optimization_tree_summary(project_dir, arm=arm)
    missing_full_forge_ops = [op for op in profiled if op not in selected_kernel_map]
    failed_attempted_ops = [op for op in attempted_ops if op not in zero_shot_generated_map]
    not_attempted_profiled_ops = [op for op in profiled if op not in attempted_ops]
    config_payload = read_json(project_dir / "config.json")
    generation_config = (
        config_payload.get("generation", {})
        if isinstance(config_payload, dict) and isinstance(config_payload.get("generation"), dict)
        else {}
    )
    project_llm_usage_summary = llm_usage_summary(project_dir / "llm_usage.db")
    arm_llm_usage_summary = llm_usage_summary(
        project_dir / "llm_usage.db",
        row_filter=arm_usage_filter(arm),
    )

    full_export = export_variant(
        project_dir=project_dir,
        root=root,
        output_path=artifact_dir / f"{model_slug}__{arm}__full_forge.cast",
        variant="full_forge",
        selected_kernels=selected_kernel_map,
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
                "optimize_log": read_text(project_dir / "logs" / "optimize.log"),
                "queue_debug_log": read_text(project_dir / "logs" / "queue_debug.log"),
                "generated_artifacts": generated_op_artifacts(project_dir, root=root),
                "optimization_summary": optimization_summary,
            },
        }
    )
    records.append(
        {
            **build_common(record_type="llm_usage", source_paths=[project_dir / "llm_usage.db"], **common_kwargs),
            "payload": {
                "project_llm_usage_db": sqlite_dump(project_dir / "llm_usage.db"),
                "project_llm_usage_summary": project_llm_usage_summary,
                "arm_llm_usage_summary": arm_llm_usage_summary,
                "usage_by_op": project_llm_usage_summary.get("by_op", {}),
                "arm_usage_by_op": arm_llm_usage_summary.get("by_op", {}),
                "per_op_usage_dbs": {
                    op: sqlite_dump(project_dir / "kernels" / "generated" / "individual_op_kernels" / op / "llm_usage.db")
                    for op in sorted(generated_op_artifacts(project_dir, root=root))
                },
                "per_op_usage_summaries": {
                    op: llm_usage_summary(
                        project_dir / "kernels" / "generated" / "individual_op_kernels" / op / "llm_usage.db",
                        operator_hint=op,
                    )
                    for op in sorted(generated_op_artifacts(project_dir, root=root))
                },
                "per_op_tree_usage_dbs": {
                    op: sqlite_dump(project_dir / "trees" / op / "llm_usage.db")
                    for op in sorted(child.name for child in (project_dir / "trees").iterdir() if child.is_dir())
                } if (project_dir / "trees").exists() else {},
                "per_op_tree_usage_summaries": {
                    op: llm_usage_summary(
                        project_dir / "trees" / op / "llm_usage.db",
                        operator_hint=op,
                    )
                    for op in sorted(child.name for child in (project_dir / "trees").iterdir() if child.is_dir())
                } if (project_dir / "trees").exists() else {},
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
                    "zero_shot_generated_kernel_map": zero_shot_generated_map,
                    "selected_kernel_map_for_arm": selected_kernel_map,
                    "optimization_summary": optimization_summary,
                },
            }
        )
    records.append(
        {
            **build_common(record_type="arm_summary", source_paths=source_paths, **common_kwargs),
            "payload": {
                "arm_complete": not missing_full_forge_ops,
                "arm_collection_complete": True,
                "all_profiled_ops_have_selected_kernel": not missing_full_forge_ops,
                "zero_shot_generated_kernel_count": len(zero_shot_generated_map),
                "zero_shot_failed_attempted_ops": failed_attempted_ops,
                "zero_shot_not_attempted_profiled_ops": not_attempted_profiled_ops,
                "optimization_started": arm != "zero_shot",
                "optimization_summary": optimization_summary,
                "profiled_ops": profiled,
                "attempted_zero_shot_ops": attempted_ops,
                "zero_shot_generated_ops": sorted(zero_shot_generated_map),
                "selected_kernel_ops_for_arm": sorted(selected_kernel_map),
                "selected_kernel_map_for_arm": selected_kernel_map,
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
                    "llm_usage_summary": project_llm_usage_summary,
                    "arm_llm_usage_summary": arm_llm_usage_summary,
                    "usage_by_op": project_llm_usage_summary.get("by_op", {}),
                    "arm_usage_by_op": arm_llm_usage_summary.get("by_op", {}),
                    "full_forge_dispatch_by_profiled_op": full_dispatch,
                    "mixed_forge_dispatch_by_profiled_op": mixed_dispatch,
                    "missing_full_forge_ops": missing_full_forge_ops,
                    "selected_kernel_map_for_arm": selected_kernel_map,
                    "optimization_summary": optimization_summary,
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
    parser.add_argument("--arm", default="zero_shot", help="Collection arm, for example zero_shot, optimize_5, optimize_10, optimize_20, optimize_50.")
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
