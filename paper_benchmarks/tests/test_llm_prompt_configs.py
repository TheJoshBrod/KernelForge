from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from paper_benchmarks.paper_bench.artifacts import create_run_layout, load_json_artifact
from paper_benchmarks.paper_bench.cli import _prepare_llm_run_context
from paper_benchmarks.paper_bench.llm_runner import load_prompt_records, run_llm_benchmark
from paper_benchmarks.paper_bench.provenance import build_environment_artifact_fields, collect_common_fields, sha256_path
from paper_benchmarks.paper_bench.registry import SyntheticWorkloadError, load_model_config, load_suite_config
from paper_benchmarks.paper_bench.schema import EnvironmentArtifact, RunManifestArtifact, Stage, Variant
from paper_benchmarks.tests.test_llm_baselines import _make_compile_fn, _toy_model_loader


def _write_prompt_yaml(tmp_path: Path, *, synthetic_workload: bool = False) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt_path = tmp_path / "paper_prompts.yaml"
    payload = {
        "synthetic_workload": synthetic_workload,
        "prompts": [
            {"id": "p2", "text": "alpha"},
            {"id": "p0", "text": "beta beta"},
            {"id": "p1", "text": "gamma gamma gamma"},
        ],
    }
    prompt_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(prompt_path)


def _write_prompt_jsonl(
    tmp_path: Path,
    *,
    synthetic_workload: bool = False,
    hash_override: str | None = None,
) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt_path = tmp_path / "paper_prompts.jsonl"
    records = [
        {"id": "j2", "text": "alpha", "source": "unit-test", "bucket": "short"},
        {"id": "j0", "text": "beta beta", "source": "unit-test", "bucket": "short"},
        {"id": "j1", "text": "gamma gamma gamma", "source": "unit-test", "bucket": "medium"},
    ]
    prompt_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest_path = prompt_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_file_hash": hash_override or sha256_path(prompt_path),
                "synthetic_workload": synthetic_workload,
                "synthetic": synthetic_workload,
                "prompt_text_may_be_stored_in_raw_artifacts": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(prompt_path)


def _write_model_config(tmp_path: Path, sample_paths: dict[str, str]) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "toy_model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_id": "toy_model_cfg",
                "loader_kind": "transformers_causal_lm",
                "model_path": sample_paths["model_path"],
                "tokenizer_path": sample_paths["model_path"],
                "model_config_path": sample_paths["model_config_path"],
                "dtype": "fp32",
                "device": "cpu",
                "local_files_only": True,
                "trust_remote_code": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _write_suite_config(tmp_path: Path, prompt_path: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "toy_suite_cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "suite_id": "toy_prompt_suite",
                "benchmark_mode": "e2e_model",
                "prompt_file": prompt_path,
                "workload_type": "prompt_records",
                "synthetic_workload": False,
                "variants": ["eager", "torch_compile"],
                "stages": ["prefill", "decode", "total_generate"],
                "prompt_length_buckets": [{"bucket_id": "all", "min_tokens": 1, "max_tokens": 32}],
                "batch_sizes": [1],
                "max_new_tokens": 3,
                "warmup_runs": 1,
                "timed_runs": 2,
                "generation_mode": "greedy",
                "include_tokenization_in_timing": False,
                "measure_prefill_decode_separately": True,
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _make_config_context(tmp_path: Path, sample_paths: dict[str, str], variant: Variant):
    prompt_path = _write_prompt_yaml(tmp_path)
    model_config_path = _write_model_config(tmp_path, sample_paths)
    suite_path = _write_suite_config(tmp_path, prompt_path)

    model_spec = load_model_config(model_config_path)
    suite = load_suite_config(suite_path)
    common = collect_common_fields(
        repo_root=Path(__file__).resolve().parents[2],
        model_id=model_spec.model_id,
        model_path=model_spec.model_path,
        model_config_path=model_spec.model_config_path,
        suite_id=suite.suite_id,
        suite_path=suite_path,
        workload_path=suite.workload_path,
        command_line=["python", "-m", "paper_benchmarks.paper_bench.cli", "run-llm"],
        paper_eligible=True,
        synthetic_workload=False,
    )
    common["compile_settings"] = {
        "backend": "inductor",
        "mode": "default",
        "fullgraph": False,
        "dynamic": False,
    }
    common["kf_settings"] = {
        "cast_package_path": None,
        "require_precompiled": False,
        "allow_jit": True,
        "fail_on_fallback": True,
        "record_runtime_stats": True,
    }
    layout = create_run_layout(tmp_path / "runs", common["timestamp_utc"], model_spec.model_id, suite.suite_id)
    common["run_id"] = layout.run_id
    manifest = RunManifestArtifact(
        **common,
        artifact_type="run_manifest",
        benchmark_mode="e2e_model",
        variant=variant,
        stage=None,
        warmup_count=suite.warmup_count,
        timed_run_count=suite.timed_run_count,
        latency_samples_ms=[],
        correctness_status="not_applicable",
        run_dir=str(layout.run_dir),
        variants_requested=["eager", "torch_compile"],
        stages_requested=["prefill", "decode", "total_generate"],
        description="toy prompt config test",
    )
    env = EnvironmentArtifact(
        **common,
        artifact_type="environment_snapshot",
        benchmark_mode="e2e_model",
        variant=variant,
        stage=None,
        warmup_count=suite.warmup_count,
        timed_run_count=suite.timed_run_count,
        latency_samples_ms=[],
        correctness_status="not_applicable",
        **build_environment_artifact_fields(),
    )
    common_fields = manifest.model_dump(
        mode="json",
        exclude={"artifact_type", "run_dir", "variants_requested", "stages_requested", "description"},
    )
    return layout, common_fields, env, manifest, model_spec, suite, prompt_path, model_config_path, suite_path


def test_prompt_suite_hash_is_recorded(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite, prompt_path, _, _ = _make_config_context(tmp_path, sample_paths, Variant.eager)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert total_artifact.workload_hash == sha256_path(prompt_path)
    assert total_artifact.details["prompt_suite_hash"] == sha256_path(prompt_path)


def test_prompt_selection_order_is_deterministic(sample_paths, tmp_path: Path):
    first = _make_config_context(tmp_path / "first", sample_paths, Variant.eager)
    second = _make_config_context(tmp_path / "second", sample_paths, Variant.eager)

    for context in (first, second):
        layout, common_fields, env, manifest, model_spec, suite, *_ = context
        run_llm_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env,
            manifest_artifact=manifest,
            model_spec=model_spec,
            suite=suite,
            variant=Variant.eager,
            model_loader=_toy_model_loader,
        )

    first_artifact = load_json_artifact(first[0].metrics_dir / "eager_total_generate.json")
    second_artifact = load_json_artifact(second[0].metrics_dir / "eager_total_generate.json")
    assert first_artifact.details["selected_prompt_ids"] == ["p2", "p0"]
    assert second_artifact.details["selected_prompt_ids"] == first_artifact.details["selected_prompt_ids"]


def test_yaml_and_jsonl_prompt_loading_work(sample_paths, tmp_path: Path):
    yaml_prompt_path = _write_prompt_yaml(tmp_path / "yaml")
    jsonl_prompt_path = _write_prompt_jsonl(tmp_path / "jsonl")

    yaml_suite = load_prompt_records(yaml_prompt_path)
    jsonl_suite = load_prompt_records(jsonl_prompt_path)

    assert yaml_suite.source_format == "yaml"
    assert [record["id"] for record in yaml_suite.records] == ["p2", "p0", "p1"]
    assert jsonl_suite.source_format == "jsonl"
    assert [record["id"] for record in jsonl_suite.records] == ["j2", "j0", "j1"]
    assert jsonl_suite.manifest_path is not None
    assert jsonl_suite.manifest is not None
    assert jsonl_suite.manifest["prompt_file_hash"] == sha256_path(jsonl_prompt_path)


def test_prompt_id_prompt_aliases_load_for_external_manifests(tmp_path: Path):
    jsonl_prompt_path = tmp_path / "external_prompts.jsonl"
    jsonl_prompt_path.write_text(
        json.dumps({"prompt_id": "qgqa_000000", "prompt": "Question: one plus one?"}) + "\n",
        encoding="utf-8",
    )
    yaml_prompt_path = tmp_path / "external_prompts.yaml"
    yaml_prompt_path.write_text(
        yaml.safe_dump(
            {"prompts": [{"prompt_id": "synthetic_000000", "prompt": "short prompt"}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    jsonl_suite = load_prompt_records(jsonl_prompt_path)
    yaml_suite = load_prompt_records(yaml_prompt_path)

    assert jsonl_suite.records[0]["id"] == "qgqa_000000"
    assert jsonl_suite.records[0]["text"] == "Question: one plus one?"
    assert yaml_suite.records[0]["id"] == "synthetic_000000"
    assert yaml_suite.records[0]["text"] == "short prompt"


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (".jsonl", '{"text":"missing-id"}\n'),
        (".jsonl", '{"id":"missing-text"}\n'),
        (".yaml", yaml.safe_dump({"prompts": [{"text": "missing-id"}]}, sort_keys=False)),
        (".yaml", yaml.safe_dump({"prompts": [{"id": "missing-text"}]}, sort_keys=False)),
    ],
)
def test_prompt_records_require_id_and_text(tmp_path: Path, suffix: str, payload: str):
    prompt_path = tmp_path / f"invalid_prompts{suffix}"
    prompt_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="missing non-empty"):
        load_prompt_records(prompt_path)


def test_prompt_manifest_hash_mismatch_fails(tmp_path: Path):
    prompt_path = _write_prompt_jsonl(tmp_path, hash_override="0" * 64)

    with pytest.raises(ValueError, match="Prompt suite hash mismatch"):
        load_prompt_records(prompt_path)


def test_output_token_hashes_recorded_for_every_timed_run(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite, *_ = _make_config_context(tmp_path, sample_paths, Variant.eager)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert len(total_artifact.sample_records) == 2
    assert all(record["output_token_hashes"] for record in total_artifact.sample_records)


def test_token_mismatch_marks_variant_incorrect(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite, *_ = _make_config_context(tmp_path, sample_paths, Variant.torch_compile)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=_make_compile_fn(mismatch_prefill_runs={4}),
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_total_generate.json")
    assert total_artifact.correctness_status.value == "failed"
    assert total_artifact.paper_eligible is False


def test_missing_prompt_file_fails(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite, *_ = _make_config_context(tmp_path, sample_paths, Variant.eager)
    suite_payload = suite.model_dump(mode="json")
    suite_payload["workload_path"] = str(tmp_path / "missing.yaml")
    suite = SimpleNamespace(**suite_payload)

    with pytest.raises(FileNotFoundError):
        run_llm_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env,
            manifest_artifact=manifest,
            model_spec=model_spec,
            suite=suite,
            variant=Variant.eager,
            model_loader=_toy_model_loader,
        )


def test_synthetic_prompt_requires_allow_flag(sample_paths, tmp_path: Path):
    prompt_path = _write_prompt_yaml(tmp_path, synthetic_workload=True)
    model_config_path = _write_model_config(tmp_path, sample_paths)
    suite_path = _write_suite_config(tmp_path, prompt_path)

    base_args = SimpleNamespace(
        registry="paper_benchmarks/configs/models/registry.yaml",
        model_id=None,
        model_config=model_config_path,
        suite=None,
        suite_config=suite_path,
        variant=None,
        variants=["eager"],
        runs_root=str(tmp_path / "runs"),
        out=None,
        allow_synthetic_demo=False,
        store_prompts=False,
        fail_if_not_paper_eligible=False,
        compile_backend="inductor",
        compile_mode=None,
        compile_fullgraph=False,
        compile_dynamic=False,
        cast_package=None,
        kf_require_precompiled=False,
        kf_allow_jit=True,
        kf_fail_on_fallback=True,
        kf_record_runtime_stats=True,
    )

    with pytest.raises(SyntheticWorkloadError):
        _prepare_llm_run_context(base_args, materialize_run_dir=False)

    allowed_args = SimpleNamespace(**{**base_args.__dict__, "allow_synthetic_demo": True})
    _, manifest, _, _, _, _ = _prepare_llm_run_context(allowed_args, materialize_run_dir=False)
    assert manifest.synthetic_workload is True
    assert manifest.paper_eligible is False


def test_prompt_ids_recorded_and_prompt_texts_redacted_by_default(sample_paths, tmp_path: Path):
    without_prompt_text = _make_config_context(tmp_path / "without_text", sample_paths, Variant.eager)
    layout, common_fields, env, manifest, model_spec, suite, *_ = without_prompt_text
    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        store_prompts=False,
    )
    redacted_rows = json.loads((layout.raw_dir / "eager_llm_measurements.json").read_text(encoding="utf-8"))
    assert redacted_rows
    assert all(row["prompt_ids"] for row in redacted_rows)
    assert all("prompt_texts" not in row for row in redacted_rows)

    with_prompt_text = _make_config_context(tmp_path / "with_text", sample_paths, Variant.eager)
    layout, common_fields, env, manifest, model_spec, suite, *_ = with_prompt_text
    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
        store_prompts=True,
    )
    unredacted_rows = json.loads((layout.raw_dir / "eager_llm_measurements.json").read_text(encoding="utf-8"))
    assert unredacted_rows
    assert all(row["prompt_ids"] for row in unredacted_rows)
    assert all("prompt_texts" in row for row in unredacted_rows)


def test_prefill_decode_total_metrics_are_separate(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite, *_ = _make_config_context(tmp_path, sample_paths, Variant.eager)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
    )

    prefill = load_json_artifact(layout.metrics_dir / "eager_prefill.json")
    decode = load_json_artifact(layout.metrics_dir / "eager_decode.json")
    total = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert prefill.stage == Stage.prefill
    assert decode.stage == Stage.decode
    assert total.stage == Stage.total_generate
    assert prefill.latency_samples_ms != total.latency_samples_ms
    assert decode.latency_samples_ms != total.latency_samples_ms
