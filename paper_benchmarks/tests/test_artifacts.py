from __future__ import annotations

from pathlib import Path

from paper_benchmarks.paper_bench.artifacts import create_run_layout, load_json_artifact, write_commands_txt, write_json_artifact
from paper_benchmarks.paper_bench.schema import EnvironmentArtifact, RunManifestArtifact

from .helpers import build_common_payload


def test_run_layout_and_artifact_writing(sample_paths, tmp_path: Path):
    common = build_common_payload(sample_paths)
    layout = create_run_layout(tmp_path, common["timestamp_utc"], common["model_id"], common["suite_id"])
    manifest_common = {**common}
    manifest_common.pop("stage")
    manifest_common["variant"] = None
    manifest_common["correctness_status"] = "not_applicable"
    manifest_common["latency_samples_ms"] = []

    manifest = RunManifestArtifact(
        **manifest_common,
        artifact_type="run_manifest",
        run_dir=str(layout.run_dir),
        variants_requested=["eager", "torch_compile"],
        stages_requested=["prefill", "decode"],
        description="unit test",
    )
    env_common = {**manifest_common}
    env = EnvironmentArtifact(
        **env_common,
        artifact_type="environment_snapshot",
        platform="Linux-test",
        platform_release="6.0-test",
        machine="x86_64",
        processor="test",
        torch_cuda_available=False,
        torch_mps_available=False,
        torch_device_capability=None,
        nvcc_version=None,
        nvidia_smi_output=None,
    )

    commands_path = write_commands_txt(layout.run_dir, common["command_line"])
    manifest_path = write_json_artifact(layout.run_dir / "manifest.json", manifest)
    env_path = write_json_artifact(layout.run_dir / "env.json", env)

    assert commands_path.exists()
    assert manifest_path.exists()
    assert env_path.exists()
    assert layout.metrics_dir.exists()
    assert "paper_benchmarks.paper_bench.cli" in commands_path.read_text(encoding="utf-8")
    assert load_json_artifact(manifest_path).artifact_type == "run_manifest"
    assert load_json_artifact(env_path).artifact_type == "environment_snapshot"
