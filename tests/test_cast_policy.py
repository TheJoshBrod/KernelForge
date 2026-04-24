from __future__ import annotations

import json

from src.optimizer.cast_policy import collect_cast_policy_metadata


def test_cast_policy_marks_full_forged_unpublishable_with_fallback(tmp_path) -> None:
    project_dir = tmp_path / "project"
    io_dir = project_dir / "io"
    generated_linear = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / "torch_nn_functional_linear"
    )
    generated_linear.mkdir(parents=True)
    io_dir.mkdir(parents=True)
    (io_dir / "summary.json").write_text(
        json.dumps(
            {
                "op_counts": {
                    "torch_nn_functional_linear": 10,
                    "torch_nn_functional_gelu": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    (generated_linear / "fallback_required.json").write_text(
        json.dumps({"reason": "quantized_linear_packed_weight"}),
        encoding="utf-8",
    )

    policy = collect_cast_policy_metadata(
        project_dir,
        selected_kernel_map={"torch_nn_functional_gelu": "kernel.cu"},
    )

    assert policy["full_forged_publishable"] is False
    assert policy["mixed_forged_publishable"] is True
    assert policy["cast_policy"] == "mixed_forged"
    assert policy["torch_fallback_ops"] == ["torch_nn_functional_linear"]
    assert policy["fallback_reasons"] == {
        "torch_nn_functional_linear": "quantized_linear_packed_weight"
    }


def test_cast_policy_records_missing_required_ops_as_mixed_fallback(tmp_path) -> None:
    project_dir = tmp_path / "project"
    io_dir = project_dir / "io"
    io_dir.mkdir(parents=True)
    (io_dir / "summary.json").write_text(
        json.dumps({"op_counts": {"torch_nn_functional_softmax": 3}}),
        encoding="utf-8",
    )

    policy = collect_cast_policy_metadata(project_dir, selected_kernel_map={})

    assert policy["full_forged_publishable"] is False
    assert policy["mixed_forged_publishable"] is True
    assert policy["torch_fallback_ops"] == ["torch_nn_functional_softmax"]
    assert policy["fallback_reasons"]["torch_nn_functional_softmax"] == (
        "missing_forged_kernel"
    )
