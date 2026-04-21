from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from paper_benchmarks.paper_bench.cli import main
from paper_benchmarks.paper_bench.kf_project import (
    AmbiguousProjectError,
    ProjectNotFoundError,
    decode_project_ref,
    describe_project,
    find_project,
    load_project_export_candidates,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_fake_project(project_root: Path) -> Path:
    project_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        project_root / "config.json",
        {
            "base_name": "test_qwen",
            "validation_dir": "data/validation",
            "backend": "cuda",
            "artifacts": {
                "weights": ["/models/Qwen3.5-35B-A3B"],
            },
        },
    )
    (project_root / "model.py").write_text(
        'LOCAL_MODEL_PATH = "/models/Qwen3.5-35B-A3B"\n',
        encoding="utf-8",
    )
    _write_json(project_root / "io" / "summary.json", {"ops": ["torch_nn_functional_linear"]})
    entry_path = project_root / "io" / "individual_ops" / "torch_nn_functional_linear" / "entry_000000.pt"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_bytes(b"pt")
    nodes_db_path = project_root / "trees" / "torch_nn_functional_linear" / "nodes.db"
    nodes_db_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_db_path.write_text("sqlite", encoding="utf-8")
    _write_json(project_root / "trees" / "torch_nn_functional_linear" / "generated_root.json", {"root": 1})
    tree_kernel_path = project_root / "trees" / "torch_nn_functional_linear" / "kernels" / "kernel_4.cu"
    tree_kernel_path.parent.mkdir(parents=True, exist_ok=True)
    tree_kernel_path.write_text(
        "// tree kernel\n",
        encoding="utf-8",
    )
    generated_kernel_path = project_root / "kernels" / "generated" / "individual_op_kernels" / "torch_nn_functional_linear" / "kernel.cu"
    generated_kernel_path.parent.mkdir(parents=True, exist_ok=True)
    generated_kernel_path.write_text(
        "// generated kernel\n",
        encoding="utf-8",
    )
    runtime_kernel_path = project_root / "benchmarks" / "runtime_kernels" / "torch_nn_functional_linear" / "kernel" / "kernel.cu"
    runtime_kernel_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_kernel_path.write_text(
        "// runtime kernel\n",
        encoding="utf-8",
    )
    _write_json(
        project_root / "benchmarks" / "op_benchmarks.json",
        {
            "project": project_root.name,
            "results": [
                {
                    "op": "torch_nn_functional_linear",
                    "winner": "optimized",
                    "kernel_status": "ok",
                    "pytorch_ms": 1.0,
                    "kernel_ms": 0.5,
                    "integrated_kernel_ms": 0.6,
                    "entries": 1,
                    "available_entries": 1,
                    "benchmarked_entry_count": 1,
                    "benchmarked_entry_files": ["entry_000000.pt"],
                    "kernel_source_path": str(generated_kernel_path),
                    "kernel_correctness": {"strict_pass": True, "errors": []},
                }
            ],
        },
    )
    _write_json(project_root / "benchmarks" / "qwen_tps_compare.json", {"project": project_root.name})
    export_path = project_root / "exports" / f"{project_root.name}.cast"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text("cast", encoding="utf-8")
    return project_root


def test_encoded_and_decoded_refs_resolve_to_same_project(tmp_path: Path):
    projects_root = tmp_path / "projects"
    project_root = _write_fake_project(projects_root / "test_qwen - NVIDIA GB10")

    encoded = "project/test_qwen%20-%20NVIDIA%20GB10/"
    decoded = "project/test_qwen - NVIDIA GB10/"
    candidates = decode_project_ref(encoded)

    assert "test_qwen%20-%20NVIDIA%20GB10" in candidates
    assert "test_qwen - NVIDIA GB10" in candidates
    assert find_project(encoded, [projects_root]) == project_root.resolve()
    assert find_project(decoded, [projects_root]) == project_root.resolve()


def test_missing_project_reports_searched_roots(tmp_path: Path):
    roots = [tmp_path / "a" / "projects", tmp_path / "b" / "projects"]

    with pytest.raises(ProjectNotFoundError) as exc_info:
        find_project("project/missing%20project/", roots)

    message = str(exc_info.value)
    assert str(roots[0].resolve(strict=False)) in message
    assert str(roots[1].resolve(strict=False)) in message
    assert "missing project" in message


def test_ambiguous_project_fails_loudly(tmp_path: Path):
    root_a = tmp_path / "root_a" / "projects"
    root_b = tmp_path / "root_b" / "projects"
    _write_fake_project(root_a / "test_qwen - NVIDIA GB10")
    _write_fake_project(root_b / "test_qwen - NVIDIA GB10")

    with pytest.raises(AmbiguousProjectError) as exc_info:
        find_project("project/test_qwen%20-%20NVIDIA%20GB10/", [root_a, root_b])

    message = str(exc_info.value)
    assert str((root_a / "test_qwen - NVIDIA GB10").resolve()) in message
    assert str((root_b / "test_qwen - NVIDIA GB10").resolve()) in message


def test_project_description_includes_benchmark_kernel_and_export_paths(tmp_path: Path):
    project_root = _write_fake_project(tmp_path / "projects" / "test_qwen - NVIDIA GB10")

    description = describe_project(project_root)
    candidates = load_project_export_candidates(project_root)

    assert description["encoded_project_ref"] == "project/test_qwen%20-%20NVIDIA%20GB10/"
    assert description["model_recorded_path"] == "/models/Qwen3.5-35B-A3B"
    assert description["captured_operator_entries"][0]["entries_dir"].endswith("io/individual_ops/torch_nn_functional_linear")
    assert description["optimization_results"][0]["nodes_db"].endswith("trees/torch_nn_functional_linear/nodes.db")
    assert description["generated_kernel_dirs"][0]["kernel_source"].endswith(
        "kernels/generated/individual_op_kernels/torch_nn_functional_linear/kernel.cu"
    )
    assert description["benchmark_artifacts"]["op_benchmarks_path"].endswith("benchmarks/op_benchmarks.json")
    assert description["export_candidate_paths"][0].endswith(".cast")
    assert description["auto_best_fastest_valid"]["policy_name"] == "auto_best_fastest_valid"
    assert description["auto_best_fastest_valid"]["selected_ops"]["torch_nn_functional_linear"]["candidate_id"] == "torch_nn_functional_linear:operator"
    assert candidates["benchmark_rows"][0]["runtime_kernel_file"].endswith(
        "benchmarks/runtime_kernels/torch_nn_functional_linear/kernel/kernel.cu"
    )


def test_inspect_project_cli_resolves_project_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    data_root = tmp_path / "kf_data"
    project_name = "test_qwen - CLI Unique"
    project_root = _write_fake_project(data_root / "projects" / project_name)
    monkeypatch.setenv("KFORGE_DATA_DIR", str(data_root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "inspect-project",
            "--project-ref",
            "project/test_qwen%20-%20CLI%20Unique/",
        ],
    )

    exit_code = main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["project_root"] == str(project_root.resolve())
