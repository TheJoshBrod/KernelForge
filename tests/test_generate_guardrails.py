from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import main as generator_main
from src.optimizer import workflow


def test_discover_ops_only_returns_directories_with_entries(tmp_path: Path):
    io_dir = tmp_path / "io"
    valid_dir = io_dir / "torch_nn_functional_softmax"
    valid_dir.mkdir(parents=True)
    (valid_dir / "entry_000001.pt").write_text("x", encoding="utf-8")

    empty_dir = io_dir / "torch_nn_functional_silu"
    empty_dir.mkdir()
    (io_dir / "not_a_dir.txt").write_text("x", encoding="utf-8")

    assert workflow._discover_ops(io_dir) == ["torch_nn_functional_softmax"]


def test_sample_entry_files_keeps_first_and_last_for_large_sets():
    entry_files = [f"entry_{idx:06d}.pt" for idx in range(10)]

    sampled = generator_main._sample_entry_files(entry_files, 4)

    assert sampled[0] == entry_files[0]
    assert sampled[-1] == entry_files[-1]
    assert len(sampled) == 4


def test_validate_kernel_passes_selected_entry_files_to_backend(
    tmp_path: Path, monkeypatch
):
    entries = [
        tmp_path / "entry_000001.pt",
        tmp_path / "entry_000002.pt",
    ]
    for entry in entries:
        entry.write_text("x", encoding="utf-8")

    captured: dict = {}

    def fake_validate(self, code, paths, ssh_config=None):
        captured["entry_files"] = paths["entry_files"]
        return True, "[Success] All 2 tests passed"

    monkeypatch.setattr(generator_main, "_is_triton", lambda: False)
    monkeypatch.setattr(generator_main.CUDABackend, "validate_kernel", fake_validate)

    log_file = tmp_path / "log.txt"
    call_success, exec_success, feedback = generator_main._validate_kernel(
        "// kernel",
        [str(entry) for entry in entries],
        log_file,
        str(tmp_path / "tmp"),
    )

    assert call_success is True
    assert exec_success is True
    assert "All 2 tests passed" in feedback
    assert captured["entry_files"] == entries


def test_run_generate_rejects_invalid_requested_ops_and_marks_tasks_failed(
    tmp_path: Path, monkeypatch
):
    repo_root = tmp_path / "repo"
    project_dir = repo_root / "kernels" / "projects" / "demo"
    io_dir = project_dir / "io" / "individual_ops" / "torch_nn_functional_softmax"
    io_dir.mkdir(parents=True)
    (io_dir / "entry_000001.pt").write_text("x", encoding="utf-8")

    queue_path = project_dir / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "active_tasks": {
                    "gen_aten__softmax": {
                        "current_step": "Generating",
                        "status": "In Progress",
                        "result": "",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(workflow, "update_job_progress", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        project="demo",
        ops="aten__softmax",
        optimize=False,
        benchmark=False,
        iterations=0,
        target_device="cpu",
        remote="",
        llm_provider="",
        llm_model="",
        workers=1,
    )

    assert workflow.run_generate(args) == 1

    queue_state = json.loads(queue_path.read_text(encoding="utf-8"))
    failed = queue_state["active_tasks"]["gen_aten__softmax"]
    assert failed["status"] == "Failed"
    assert "No captured inputs found for requested ops: aten__softmax" in failed["result"]
    assert "torch_nn_functional_softmax" in failed["result"]


def test_generator_main_reports_available_ops_when_requested_op_is_missing(
    tmp_path: Path, monkeypatch, capsys
):
    io_dir = tmp_path / "io"
    softmax_dir = io_dir / "torch_nn_functional_softmax"
    softmax_dir.mkdir(parents=True)
    (softmax_dir / "entry_000001.pt").write_text("x", encoding="utf-8")

    silu_dir = io_dir / "torch_nn_functional_silu"
    silu_dir.mkdir()
    (silu_dir / "entry_000001.pt").write_text("x", encoding="utf-8")

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generator.main",
            "--io-dir",
            str(io_dir),
            "--out-dir",
            str(out_dir),
            "--only-ops",
            "aten__softmax",
        ],
    )
    monkeypatch.setattr(generator_main, "update_job_progress", lambda *args, **kwargs: None)

    assert generator_main.main() == 2

    output = capsys.readouterr().out
    assert "No captured operator directories matched the requested ops: aten__softmax" in output
    assert "torch_nn_functional_softmax" in output
    assert "torch_nn_functional_silu" in output


def test_generator_main_shuts_down_cuda_verifier_on_exit(tmp_path: Path, monkeypatch):
    io_dir = tmp_path / "io"
    softmax_dir = io_dir / "torch_nn_functional_softmax"
    softmax_dir.mkdir(parents=True)
    (softmax_dir / "entry_000001.pt").write_text("x", encoding="utf-8")

    out_dir = tmp_path / "out"
    calls: list[str] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generator.main",
            "--io-dir",
            str(io_dir),
            "--out-dir",
            str(out_dir),
            "--only-ops",
            "aten__softmax",
        ],
    )
    monkeypatch.setattr(generator_main, "update_job_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        generator_main.cuda_verifier,
        "shutdown_worker",
        lambda *args, **kwargs: calls.append("shutdown"),
    )

    assert generator_main.main() == 2
    assert calls == ["shutdown"]


def test_load_saved_benchmark_for_existing_kernel_reuses_verified_row_when_sources_match(
    tmp_path: Path,
):
    project_dir = tmp_path / "project"
    generated_op_dir = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / "torch_nn_functional_softplus"
    )
    generated_op_dir.mkdir(parents=True)
    (generated_op_dir / "kernel.cu").write_text("// same kernel\n", encoding="utf-8")

    tree_kernel = (
        project_dir
        / "trees"
        / "torch_nn_functional_softplus"
        / "kernels"
        / "kernel_0.cu"
    )
    tree_kernel.parent.mkdir(parents=True)
    tree_kernel.write_text("// same kernel\n", encoding="utf-8")

    (tree_kernel.parent.parent / "generated_root.json").write_text(
        json.dumps(
            {
                "kernel_relpath": "trees/torch_nn_functional_softplus/kernels/kernel_0.cu"
            }
        ),
        encoding="utf-8",
    )

    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "op": "torch_nn_functional_softplus",
                        "kernel_status": "ok",
                        "kernel_ms": 0.42,
                        "backend": "cuda",
                        "kernel_entry_latencies_ms": [0.41, 0.43],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert workflow._load_saved_benchmark_for_existing_kernel(
        project_dir,
        "torch_nn_functional_softplus",
        generated_op_dir,
    ) == (0.42, "cuda")


def test_load_saved_benchmark_for_existing_kernel_rejects_changed_sources(
    tmp_path: Path,
):
    project_dir = tmp_path / "project"
    generated_op_dir = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / "torch_nn_functional_softplus"
    )
    generated_op_dir.mkdir(parents=True)
    (generated_op_dir / "kernel.cu").write_text("// new kernel\n", encoding="utf-8")

    tree_kernel = (
        project_dir
        / "trees"
        / "torch_nn_functional_softplus"
        / "kernels"
        / "kernel_0.cu"
    )
    tree_kernel.parent.mkdir(parents=True)
    tree_kernel.write_text("// old kernel\n", encoding="utf-8")

    (tree_kernel.parent.parent / "generated_root.json").write_text(
        json.dumps(
            {
                "kernel_relpath": "trees/torch_nn_functional_softplus/kernels/kernel_0.cu"
            }
        ),
        encoding="utf-8",
    )

    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "op": "torch_nn_functional_softplus",
                        "kernel_status": "ok",
                        "kernel_ms": 0.42,
                        "backend": "cuda",
                        "kernel_entry_latencies_ms": [0.41, 0.43],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert workflow._load_saved_benchmark_for_existing_kernel(
        project_dir,
        "torch_nn_functional_softplus",
        generated_op_dir,
    ) == (None, "")
