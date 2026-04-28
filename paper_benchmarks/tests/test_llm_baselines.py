from __future__ import annotations

import json
from statistics import median
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from paper_benchmarks.paper_bench.artifacts import create_run_layout, load_json_artifact
from paper_benchmarks.paper_bench.cli import _cmd_preflight, _resolve_compile_settings
from paper_benchmarks.paper_bench.llm_runner import run_llm_benchmark
from paper_benchmarks.paper_bench.provenance import build_environment_artifact_fields, collect_common_fields
from paper_benchmarks.paper_bench.schema import EnvironmentArtifact, RunManifestArtifact, Stage, Variant
from paper_benchmarks.paper_bench.stats import percentile


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 29

    def __call__(self, prompts, return_tensors="pt", padding=True):
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        token_rows: list[list[int]] = []
        for prompt in prompt_list:
            row = [((ord(char) % 7) + 1) for char in prompt][:4]
            token_rows.append(row or [1])
        max_len = max(len(row) for row in token_rows)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for row in token_rows:
            pad_width = max_len - len(row)
            input_ids.append(row + ([self.pad_token_id] * pad_width))
            attention_mask.append(([1] * len(row)) + ([0] * pad_width))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ToyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self._extra_offset = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        next_token = ((input_ids[:, -1] + 1 + int(self._extra_offset)) % (self.vocab_size - 1)) + 1
        logits[torch.arange(batch, device=input_ids.device), -1, next_token] = 10.0 + self.anchor
        return SimpleNamespace(logits=logits, past_key_values=("kv", seq_len))


class CompiledToyModel(torch.nn.Module):
    def __init__(
        self,
        base_model: ToyLM,
        *,
        fail_on_prefill_run: int | None = None,
        mismatch_prefill_runs: set[int] | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.fail_on_prefill_run = fail_on_prefill_run
        self.mismatch_prefill_runs = set(mismatch_prefill_runs or set())
        self.prefill_run_count = 0
        self.active_run_index = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        if attention_mask is not None:
            self.prefill_run_count += 1
            self.active_run_index = self.prefill_run_count
            if self.fail_on_prefill_run == self.active_run_index:
                raise RuntimeError("synthetic compile failure")
        previous_offset = self.base_model._extra_offset
        self.base_model._extra_offset = 1 if self.active_run_index in self.mismatch_prefill_runs else 0
        try:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        finally:
            self.base_model._extra_offset = previous_offset


def _write_prompt_workload(tmp_path: Path) -> tuple[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    workload_path = tmp_path / "toy_prompts.jsonl"
    workload_path.write_text(
        "\n".join(
            [
                '{"id":"p0","prompt":"alpha"}',
                '{"id":"p1","prompt":"beta"}',
                '{"id":"p2","prompt":"gamma"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "toy_suite.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "suite_id: toy_suite",
                "benchmark_mode: e2e_model",
                "workload_type: prompt_jsonl",
                f"workload_path: {workload_path}",
                "synthetic_workload: false",
                "variants: [eager, torch_compile]",
                "stages: [prefill, decode, total_generate]",
                "warmup_count: 1",
                "timed_run_count: 2",
                "batch_size: 1",
                "max_new_tokens: 3",
                "device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(workload_path), str(suite_path)


def _write_model_config(tmp_path: Path, sample_paths: dict[str, str]) -> str:
    config_path = tmp_path / "toy_model.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_id: toy_model_cfg",
                "loader_kind: transformers_causal_lm",
                f"model_path: {sample_paths['model_path']}",
                f"tokenizer_path: {sample_paths['model_path']}",
                f"model_config_path: {sample_paths['model_config_path']}",
                "torch_dtype: fp32",
                "device: cpu",
                "local_files_only: true",
                "trust_remote_code: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(config_path)


def _make_context(tmp_path: Path, sample_paths: dict[str, str], variant: Variant):
    workload_path, suite_path = _write_prompt_workload(tmp_path)
    common = collect_common_fields(
        repo_root=Path(__file__).resolve().parents[2],
        model_id="toy_model",
        model_path=sample_paths["model_path"],
        model_config_path=sample_paths["model_config_path"],
        suite_id="toy_suite",
        suite_path=suite_path,
        workload_path=workload_path,
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
    layout = create_run_layout(tmp_path / "runs", common["timestamp_utc"], "toy_model", "toy_suite")
    common["run_id"] = layout.run_id
    manifest = RunManifestArtifact(
        **common,
        artifact_type="run_manifest",
        benchmark_mode="e2e_model",
        variant=variant,
        stage=None,
        warmup_count=1,
        timed_run_count=2,
        latency_samples_ms=[],
        correctness_status="not_applicable",
        run_dir=str(layout.run_dir),
        variants_requested=["eager", "torch_compile"],
        stages_requested=["prefill", "decode", "total_generate"],
        description="toy llm baseline test",
    )
    env = EnvironmentArtifact(
        **common,
        artifact_type="environment_snapshot",
        benchmark_mode="e2e_model",
        variant=variant,
        stage=None,
        warmup_count=1,
        timed_run_count=2,
        latency_samples_ms=[],
        correctness_status="not_applicable",
        **build_environment_artifact_fields(),
    )
    common_fields = manifest.model_dump(
        mode="json",
        exclude={"artifact_type", "run_dir", "variants_requested", "stages_requested", "description"},
    )
    model_spec = SimpleNamespace(
        model_id="toy_model",
        model_path=sample_paths["model_path"],
        tokenizer_path=None,
        trust_remote_code=False,
        torch_dtype=None,
        cast_package_path=None,
    )
    suite = SimpleNamespace(
        workload_path=workload_path,
        device="cpu",
        max_new_tokens=3,
        warmup_count=1,
        timed_run_count=2,
        batch_size=1,
    )
    return layout, common_fields, env, manifest, model_spec, suite


def _toy_model_loader(model_spec, device=None):
    model = ToyLM()
    if device:
        model.to(device)
    return model, ToyTokenizer(), 5.0


def _hf_tokenizer_model_loader(model_spec, device=None):
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "<bos>",
        "<eos>",
        "alpha",
        "beta",
        "gamma",
        "hello",
        "world",
        "kernel",
        "forge",
        "benchmark",
    ]
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    model = ToyLM(vocab_size=len(vocab_tokens) + 8)
    if device:
        model.to(device)
    return model, tokenizer, 5.0


def _make_compile_fn(*, fail_on_prefill_run: int | None = None, mismatch_prefill_runs: set[int] | None = None):
    observed_settings: list[dict[str, object]] = []

    def _compile(model, settings):
        observed_settings.append(settings.as_dict())
        return CompiledToyModel(
            model,
            fail_on_prefill_run=fail_on_prefill_run,
            mismatch_prefill_runs=mismatch_prefill_runs,
        ), 7.0

    _compile.observed_settings = observed_settings
    return _compile


def test_compile_failure_is_recorded_and_not_hidden(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.torch_compile)
    compile_fn = _make_compile_fn(fail_on_prefill_run=1)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=compile_fn,
    )

    compile_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_compile.json")
    assert compile_artifact.correctness_status.value == "failed"
    assert compile_artifact.paper_eligible is False
    assert compile_artifact.details["execution_status"] == "failed"
    assert compile_artifact.details["error_type"] == "RuntimeError"
    assert compile_artifact.compile_settings["backend"] == "inductor"
    assert not (layout.metrics_dir / "torch_compile_total_generate.json").exists()


def test_compile_time_is_separated_from_steady_state_time(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.torch_compile)
    compile_fn = _make_compile_fn()

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=compile_fn,
    )

    compile_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_compile.json")
    total_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_total_generate.json")
    assert compile_fn.observed_settings == [common_fields["compile_settings"]]
    assert compile_artifact.compile_time_ms is not None
    assert compile_artifact.compile_time_ms == compile_artifact.latency_samples_ms[0]
    assert total_artifact.steady_state_time_ms == sum(total_artifact.latency_samples_ms) / len(total_artifact.latency_samples_ms)
    assert all(sample != compile_artifact.compile_time_ms for sample in total_artifact.latency_samples_ms)


def test_torch_compile_steady_state_failure_is_recorded_and_not_hidden(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.torch_compile)
    compile_fn = _make_compile_fn(fail_on_prefill_run=3)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=compile_fn,
    )

    compile_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_compile.json")
    total_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_total_generate.json")
    raw_rows = json.loads((layout.raw_dir / "torch_compile_llm_measurements.json").read_text(encoding="utf-8"))

    assert compile_artifact.correctness_status.value == "passed"
    assert total_artifact.correctness_status.value == "failed"
    assert total_artifact.paper_eligible is False
    assert total_artifact.details["execution_status"] == "failed"
    assert total_artifact.details["error_type"] == "RuntimeError"
    assert total_artifact.details["failed_sample_index"] == 0
    assert total_artifact.sample_records[-1]["execution_status"] == "failed"
    assert raw_rows[-1]["execution_status"] == "failed"
    assert raw_rows[-1]["error_type"] == "RuntimeError"


def test_warmup_is_excluded_from_steady_state_metrics(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.eager)

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

    warmup_artifact = load_json_artifact(layout.metrics_dir / "eager_warmup.json")
    total_artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert warmup_artifact.stage == Stage.warmup
    assert len(warmup_artifact.latency_samples_ms) == 1
    assert len(total_artifact.latency_samples_ms) == 2
    assert total_artifact.steady_state_time_ms == sum(total_artifact.latency_samples_ms) / 2


def test_eager_and_compile_use_identical_inputs(sample_paths, tmp_path: Path):
    eager_layout, eager_common, eager_env, eager_manifest, model_spec, suite = _make_context(tmp_path / "eager", sample_paths, Variant.eager)
    compile_layout, compile_common, compile_env, compile_manifest, model_spec_compile, suite_compile = _make_context(tmp_path / "compile", sample_paths, Variant.torch_compile)

    run_llm_benchmark(
        layout=eager_layout,
        common_fields=eager_common,
        env_artifact=eager_env,
        manifest_artifact=eager_manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_toy_model_loader,
    )
    run_llm_benchmark(
        layout=compile_layout,
        common_fields=compile_common,
        env_artifact=compile_env,
        manifest_artifact=compile_manifest,
        model_spec=model_spec_compile,
        suite=suite_compile,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=_make_compile_fn(),
    )

    eager_total = load_json_artifact(eager_layout.metrics_dir / "eager_total_generate.json")
    compile_total = load_json_artifact(compile_layout.metrics_dir / "torch_compile_total_generate.json")
    eager_hashes = [record["input_batch_hash"] for record in eager_total.sample_records]
    compile_hashes = [record["input_batch_hash"] for record in compile_total.sample_records]
    assert eager_hashes == compile_hashes


def test_all_timed_output_hashes_are_checked(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.torch_compile)
    compile_fn = _make_compile_fn(mismatch_prefill_runs={4})

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=_toy_model_loader,
        compile_model_fn=compile_fn,
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "torch_compile_total_generate.json")
    raw_rows = json.loads((layout.raw_dir / "torch_compile_llm_measurements.json").read_text(encoding="utf-8"))
    assert total_artifact.correctness_status.value == "failed"
    assert total_artifact.paper_eligible is False
    assert len(total_artifact.sample_records) == 2
    assert raw_rows[1]["output_token_hashes"] != raw_rows[1]["reference_output_token_hashes"]


def test_metrics_use_raw_samples(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.eager)

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
    samples = total_artifact.latency_samples_ms
    assert total_artifact.latency_summary.count == len(samples)
    assert total_artifact.latency_summary.mean_ms == sum(samples) / len(samples)
    assert total_artifact.latency_summary.median_ms == median(samples)
    assert total_artifact.latency_summary.p05_ms == percentile(samples, 5)
    assert total_artifact.latency_summary.p95_ms == percentile(samples, 95)


def test_sample_records_capture_generation_inputs_and_outputs(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.eager)

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
    assert total_artifact.details["generation_settings"]["generation_mode"] == "greedy"
    assert total_artifact.details["generation_settings"]["do_sample"] is False
    assert total_artifact.details["generation_settings"]["max_new_tokens"] == 3
    assert total_artifact.details["generation_settings"]["pad_token_id"] == ToyTokenizer.pad_token_id
    assert total_artifact.details["generation_settings"]["eos_token_id"] == ToyTokenizer.eos_token_id
    assert all(record["batch_size"] == 1 for record in total_artifact.sample_records)
    assert all(record["prompt_lengths"] for record in total_artifact.sample_records)
    assert all(record["generated_tokens"] for record in total_artifact.sample_records)


def test_huggingface_batchencoding_tokenizer_is_supported(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.eager)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=_hf_tokenizer_model_loader,
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "eager_total_generate.json")
    assert total_artifact.correctness_status.value == "reference"
    assert len(total_artifact.sample_records) == 2


def test_preflight_surfaces_compile_settings(sample_paths, tmp_path: Path, capsys):
    _, suite_path = _write_prompt_workload(tmp_path)
    model_config_path = _write_model_config(tmp_path, sample_paths)
    args = SimpleNamespace(
        registry="paper_benchmarks/configs/models/registry.yaml",
        model_id=None,
        model_config=model_config_path,
        suite=None,
        suite_config=suite_path,
        variant=None,
        variants=["torch_compile"],
        runs_root=str(tmp_path / "runs"),
        out=None,
        allow_synthetic_demo=False,
        store_prompts=False,
        reuse_cache=False,
        fail_if_not_paper_eligible=False,
        compile_backend="inductor",
        compile_mode="reduce-overhead",
        compile_fullgraph=True,
        compile_dynamic=True,
        cast_package=None,
        project_ref=None,
        kf_require_precompiled=False,
        kf_allow_jit=True,
        kf_fail_on_fallback=True,
        kf_record_runtime_stats=True,
    )

    rc = _cmd_preflight(args)
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["compile_settings"] == {
        "backend": "inductor",
        "mode": "reduce-overhead",
        "fullgraph": True,
        "dynamic": True,
    }


def test_compile_settings_use_model_defaults_with_cli_overrides():
    args = SimpleNamespace(
        compile_backend=None,
        compile_mode=None,
        compile_fullgraph=None,
        compile_dynamic=None,
    )
    model_spec = SimpleNamespace(
        compile_settings={
            "backend": "inductor",
            "mode": "max-autotune",
            "fullgraph": True,
            "dynamic": True,
        }
    )

    assert _resolve_compile_settings(args, model_spec) == {
        "backend": "inductor",
        "mode": "max-autotune",
        "fullgraph": True,
        "dynamic": True,
    }

    args.compile_backend = "aot_eager"
    args.compile_dynamic = False

    assert _resolve_compile_settings(args, model_spec) == {
        "backend": "aot_eager",
        "mode": "max-autotune",
        "fullgraph": True,
        "dynamic": False,
    }
