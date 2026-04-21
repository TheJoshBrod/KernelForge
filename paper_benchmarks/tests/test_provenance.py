from __future__ import annotations

from pathlib import Path

from paper_benchmarks.paper_bench.provenance import collect_common_fields, sha256_directory


def test_collect_common_fields_includes_hashes_and_versions(sample_paths):
    repo_root = Path(__file__).resolve().parents[2]
    payload = collect_common_fields(
        repo_root=repo_root,
        model_id="unit_test_model",
        model_path=sample_paths["model_path"],
        model_config_path=sample_paths["model_config_path"],
        suite_id="test_suite",
        suite_path=sample_paths["suite_path"],
        workload_path=sample_paths["workload_path"],
        command_line=["python", "-m", "paper_benchmarks.paper_bench.cli", "preflight"],
        paper_eligible=True,
        synthetic_workload=False,
        cast_package_path=sample_paths["cast_path"],
    )

    assert payload["git_commit"]
    assert payload["timestamp_utc"]
    assert payload["hostname"]
    assert payload["python_version"]
    assert payload["pytorch_version"]
    assert payload["model_path_hash"]
    assert payload["suite_hash"]
    assert payload["cast_package_hash"]


def test_missing_git_marks_non_paper(sample_paths, tmp_path: Path):
    payload = collect_common_fields(
        repo_root=tmp_path,
        model_id="unit_test_model",
        model_path=sample_paths["model_path"],
        model_config_path=sample_paths["model_config_path"],
        suite_id="test_suite",
        suite_path=sample_paths["suite_path"],
        workload_path=sample_paths["workload_path"],
        command_line=["python", "-m", "paper_benchmarks.paper_bench.cli", "preflight"],
        paper_eligible=True,
        synthetic_workload=False,
        cast_package_path=sample_paths["cast_path"],
    )

    assert payload["git_available"] is False
    assert payload["paper_eligible"] is False
    assert "git metadata unavailable" in payload["paper_eligibility_issues"]


def test_directory_hash_is_stable_under_file_ordering(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    (left / "b.txt").write_text("b", encoding="utf-8")
    (left / "a.txt").write_text("a", encoding="utf-8")
    (left / "nested").mkdir()
    (left / "nested" / "c.txt").write_text("c", encoding="utf-8")

    (right / "nested").mkdir()
    (right / "nested" / "c.txt").write_text("c", encoding="utf-8")
    (right / "a.txt").write_text("a", encoding="utf-8")
    (right / "b.txt").write_text("b", encoding="utf-8")

    assert sha256_directory(left) == sha256_directory(right)
