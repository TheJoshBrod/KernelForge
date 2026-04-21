from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote


class ProjectResolutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        project_ref: str,
        candidate_refs: list[str],
        searched_roots: list[Path],
    ) -> None:
        super().__init__(message)
        self.project_ref = project_ref
        self.candidate_refs = list(candidate_refs)
        self.searched_roots = list(searched_roots)


class ProjectNotFoundError(ProjectResolutionError):
    pass


class AmbiguousProjectError(ProjectResolutionError):
    def __init__(
        self,
        message: str,
        *,
        project_ref: str,
        candidate_refs: list[str],
        searched_roots: list[Path],
        matches: list[Path],
    ) -> None:
        super().__init__(
            message,
            project_ref=project_ref,
            candidate_refs=candidate_refs,
            searched_roots=searched_roots,
        )
        self.matches = list(matches)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _unique_paths(values: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for value in values:
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _project_container_roots() -> list[Path]:
    repo_root = _repo_root()
    raw_roots: list[Path] = []
    data_dir = os.environ.get("KFORGE_DATA_DIR")
    if data_dir:
        data_root = Path(data_dir).expanduser()
        raw_roots.extend([
            data_root / "projects",
            data_root / "project",
            data_root,
        ])
    raw_roots.extend([
        repo_root / "kernels" / "projects",
        repo_root / "kernels" / "private" / "projects",
        repo_root / "projects",
        repo_root / "project",
        repo_root / "data" / "projects",
        repo_root / "data" / "project",
        repo_root / "data",
        repo_root / ".kernelforge" / "projects",
        repo_root / ".kernelforge",
        repo_root / "paper_benchmarks" / "runs",
        Path.home() / ".kernelforge" / "projects",
        Path.home() / ".kernelforge",
    ])
    return _unique_paths([root.resolve(strict=False) for root in raw_roots])


def _normalize_project_ref(project_ref: str) -> str:
    value = str(project_ref or "").strip()
    if not value:
        raise ValueError("project_ref must be non-empty")
    while value.endswith("/"):
        value = value[:-1]
    return value


def decode_project_ref(project_ref: str) -> list[str]:
    normalized = _normalize_project_ref(project_ref).replace("\\", "/")
    decoded = unquote(normalized)
    basename_normalized = Path(normalized).name
    basename_decoded = Path(decoded).name
    encoded = quote(decoded, safe="/-._")
    encoded_basename = Path(encoded).name
    candidates = [
        normalized,
        decoded,
        encoded,
        basename_normalized,
        basename_decoded,
        encoded_basename,
    ]
    if basename_decoded and f"project/{basename_decoded}" not in candidates:
        candidates.append(f"project/{basename_decoded}")
    if encoded_basename and f"project/{encoded_basename}" not in candidates:
        candidates.append(f"project/{encoded_basename}")
    return _unique_strings([candidate for candidate in candidates if candidate])


def _looks_like_project_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = [
        path / "config.json",
        path / "model.py",
        path / "io",
        path / "trees",
        path / "benchmarks",
        path / "exports",
    ]
    return any(marker.exists() for marker in markers)


def _candidate_locations(root: Path, candidate: str) -> list[Path]:
    candidate_path = Path(candidate)
    basename = candidate_path.name
    locations = [
        root / candidate,
        root / basename,
    ]
    if root.name not in {"project", "projects"}:
        locations.extend([
            root / "project" / basename,
            root / "projects" / basename,
        ])
    return _unique_paths([location.resolve(strict=False) for location in locations])


def find_project(project_ref: str, search_roots: list[Path] | None = None) -> Path:
    normalized = _normalize_project_ref(project_ref)
    candidate_refs = decode_project_ref(normalized)
    searched_roots = _unique_paths(
        [Path(root).expanduser().resolve(strict=False) for root in (search_roots or _project_container_roots())]
    )

    matches: list[Path] = []
    for candidate in candidate_refs:
        direct = Path(candidate).expanduser()
        if direct.exists() and _looks_like_project_root(direct):
            matches.append(direct.resolve())

    for root in searched_roots:
        for candidate in candidate_refs:
            for location in _candidate_locations(root, candidate):
                if location.exists() and _looks_like_project_root(location):
                    matches.append(location.resolve())

    unique_matches = _unique_paths(matches)
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        roots_text = ", ".join(str(root) for root in searched_roots)
        candidates_text = ", ".join(candidate_refs)
        raise ProjectNotFoundError(
            f"Unable to resolve project ref {project_ref!r}. Candidates: [{candidates_text}]. "
            f"Searched roots: [{roots_text}]",
            project_ref=project_ref,
            candidate_refs=candidate_refs,
            searched_roots=searched_roots,
        )
    matches_text = ", ".join(str(match) for match in unique_matches)
    roots_text = ", ".join(str(root) for root in searched_roots)
    raise AmbiguousProjectError(
        f"Project ref {project_ref!r} is ambiguous. Matches: [{matches_text}]. "
        f"Searched roots: [{roots_text}]",
        project_ref=project_ref,
        candidate_refs=candidate_refs,
        searched_roots=searched_roots,
        matches=unique_matches,
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return None


def _relative_or_none(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _first_existing_file(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def _project_config_summary(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "config.json"
    payload = _load_json(config_path) or {}
    return {
        "path": str(config_path) if config_path.exists() else None,
        "base_name": payload.get("base_name"),
        "validation_dir": payload.get("validation_dir"),
        "backend": payload.get("backend"),
        "weights": list(payload.get("artifacts", {}).get("weights", []) or []),
    }


def _model_recorded_path(project_root: Path, config_summary: dict[str, Any]) -> str | None:
    weights = list(config_summary.get("weights", []) or [])
    if weights:
        return str(weights[0])
    model_py = project_root / "model.py"
    if not model_py.exists():
        return None
    for line in model_py.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "/home/" in stripped or stripped.startswith(("MODEL_PATH =", "LOCAL_MODEL_PATH =", "DEFAULT_MODEL_PATH =")):
            if '"' in stripped:
                parts = stripped.split('"')
                if len(parts) >= 2 and parts[1]:
                    return parts[1]
            if "'" in stripped:
                parts = stripped.split("'")
                if len(parts) >= 2 and parts[1]:
                    return parts[1]
            return stripped
    return str(model_py)


def load_project_export_candidates(project_root: str | Path) -> dict[str, Any]:
    from .cast_selection import NoEligibleCastKernelsError, select_project_export_kernels

    root = Path(project_root).expanduser().resolve()
    benchmarks_dir = root / "benchmarks"
    exports_dir = root / "exports"
    entries_root = root / "io" / "individual_ops"
    generated_root = root / "kernels" / "generated" / "individual_op_kernels"
    trees_root = root / "trees"
    runtime_kernels_root = benchmarks_dir / "runtime_kernels"
    op_benchmarks_path = benchmarks_dir / "op_benchmarks.json"
    tps_compare_path = benchmarks_dir / "qwen_tps_compare.json"
    op_benchmarks = _load_json(op_benchmarks_path) or {}
    results = op_benchmarks.get("results", [])
    benchmark_rows: list[dict[str, Any]] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            op_name = str(item.get("op") or "")
            if not op_name:
                continue
            entries_dir = entries_root / op_name
            runtime_dir = runtime_kernels_root / op_name
            generated_dir = generated_root / op_name
            tree_dir = trees_root / op_name
            runtime_kernel_file = _first_existing_file([
                runtime_dir / "kernel" / "kernel.cu",
                runtime_dir / f"kernel_{item.get('winner_index', 0)}" / "kernel.cu",
            ])
            benchmark_rows.append({
                "op": op_name,
                "winner": item.get("winner"),
                "kernel_status": item.get("kernel_status"),
                "pytorch_ms": item.get("pytorch_ms"),
                "kernel_ms": item.get("kernel_ms"),
                "integrated_kernel_ms": item.get("integrated_kernel_ms"),
                "entries": item.get("entries"),
                "available_entries": item.get("available_entries"),
                "benchmarked_entry_count": item.get("benchmarked_entry_count"),
                "benchmarked_entry_files": list(item.get("benchmarked_entry_files", []) or []),
                "entries_dir": str(entries_dir) if entries_dir.exists() else None,
                "generated_kernel_dir": str(generated_dir) if generated_dir.exists() else None,
                "tree_dir": str(tree_dir) if tree_dir.exists() else None,
                "runtime_kernel_dir": str(runtime_dir) if runtime_dir.exists() else None,
                "runtime_kernel_file": str(runtime_kernel_file) if runtime_kernel_file else None,
            })

    export_paths = sorted(
        [path.resolve() for path in exports_dir.glob("*.cast") if path.is_file()],
        key=lambda path: path.name,
    )
    try:
        selection_report = select_project_export_kernels(root, fail_if_empty=False)
    except NoEligibleCastKernelsError as exc:
        selection_report = exc.report

    return {
        "project_root": str(root),
        "benchmarks_dir": str(benchmarks_dir) if benchmarks_dir.exists() else None,
        "exports_dir": str(exports_dir) if exports_dir.exists() else None,
        "entries_root": str(entries_root) if entries_root.exists() else None,
        "generated_kernels_root": str(generated_root) if generated_root.exists() else None,
        "trees_root": str(trees_root) if trees_root.exists() else None,
        "runtime_kernels_root": str(runtime_kernels_root) if runtime_kernels_root.exists() else None,
        "op_benchmarks_path": str(op_benchmarks_path) if op_benchmarks_path.exists() else None,
        "qwen_tps_compare_path": str(tps_compare_path) if tps_compare_path.exists() else None,
        "export_candidate_paths": [str(path) for path in export_paths],
        "benchmark_rows": benchmark_rows,
        "auto_best_fastest_valid": selection_report,
    }


def describe_project(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_summary = _project_config_summary(root)
    candidates = load_project_export_candidates(root)
    entries_root = root / "io" / "individual_ops"
    trees_root = root / "trees"
    generated_root = root / "kernels" / "generated" / "individual_op_kernels"
    exports_dir = root / "exports"
    captured_entries: list[dict[str, Any]] = []
    if entries_root.exists():
        for op_dir in sorted(child for child in entries_root.iterdir() if child.is_dir()):
            entry_files = sorted(op_dir.glob("entry_*.pt"))
            captured_entries.append({
                "op": op_dir.name,
                "entries_dir": str(op_dir),
                "entry_count": len(entry_files),
            })
    optimization_results: list[dict[str, Any]] = []
    if trees_root.exists():
        for op_dir in sorted(child for child in trees_root.iterdir() if child.is_dir()):
            optimization_results.append({
                "op": op_dir.name,
                "tree_dir": str(op_dir),
                "nodes_db": str(op_dir / "nodes.db") if (op_dir / "nodes.db").exists() else None,
                "generated_root": str(op_dir / "generated_root.json") if (op_dir / "generated_root.json").exists() else None,
                "kernels_dir": str(op_dir / "kernels") if (op_dir / "kernels").exists() else None,
            })
    generated_kernel_dirs: list[dict[str, Any]] = []
    if generated_root.exists():
        for op_dir in sorted(child for child in generated_root.iterdir() if child.is_dir()):
            generated_kernel_dirs.append({
                "op": op_dir.name,
                "dir": str(op_dir),
                "kernel_source": str(op_dir / "kernel.cu") if (op_dir / "kernel.cu").exists() else None,
            })
    missing_artifacts: list[str] = []
    for label, path in [
        ("config", root / "config.json"),
        ("model_module", root / "model.py"),
        ("captured_entries", entries_root),
        ("trees", trees_root),
        ("generated_kernels", generated_root),
        ("benchmarks", root / "benchmarks"),
        ("exports", exports_dir),
    ]:
        if not path.exists():
            missing_artifacts.append(label)

    return {
        "project_root": str(root),
        "project_id": root.name,
        "project_name": root.name,
        "project_slug": root.name,
        "encoded_project_ref": f"project/{quote(root.name, safe='-._')}/",
        "decoded_project_ref": f"project/{root.name}/",
        "config": config_summary,
        "model_recorded_path": _model_recorded_path(root, config_summary),
        "model_module_path": str(root / "model.py") if (root / "model.py").exists() else None,
        "summary_path": str(root / "io" / "summary.json") if (root / "io" / "summary.json").exists() else None,
        "captured_operator_entries": captured_entries,
        "optimization_results": optimization_results,
        "generated_kernel_dirs": generated_kernel_dirs,
        "benchmark_artifacts": {
            "op_benchmarks_path": candidates["op_benchmarks_path"],
            "qwen_tps_compare_path": candidates["qwen_tps_compare_path"],
            "runtime_kernels_root": candidates["runtime_kernels_root"],
        },
        "export_candidate_paths": candidates["export_candidate_paths"],
        "export_candidates": candidates,
        "auto_best_fastest_valid": candidates["auto_best_fastest_valid"],
        "missing_artifacts": missing_artifacts,
        "relative_paths": {
            "entries_root": _relative_or_none(entries_root, root) if entries_root.exists() else None,
            "trees_root": _relative_or_none(trees_root, root) if trees_root.exists() else None,
            "generated_kernels_root": _relative_or_none(generated_root, root) if generated_root.exists() else None,
            "exports_dir": _relative_or_none(exports_dir, root) if exports_dir.exists() else None,
        },
    }
