from __future__ import annotations

import json
import hashlib
import os
import zipfile
from pathlib import Path

import torch
import pytest

from paper_benchmarks.paper_bench.artifacts import load_json_artifact
from paper_benchmarks.paper_bench.kf_runtime import (
    KfRuntimeSettings,
    inspect_cast_package,
    load_cast_model,
    validate_runtime_stats_api,
)
from paper_benchmarks.paper_bench.llm_runner import run_llm_benchmark
from paper_benchmarks.paper_bench.report import summarize_run
from paper_benchmarks.paper_bench.schema import Variant

from .test_llm_baselines import ToyLM, ToyTokenizer, _make_compile_fn, _make_context


def test_gelu_launch_arg_completion_uses_pytorch_default():
    from kernelforge.run_cast import _complete_functional_launch_args

    completed = _complete_functional_launch_args(
        op_name="torch_nn_functional_gelu",
        call_args=["tensor"],
        resolved_args={},
        kwargs={},
        n_launch=2,
    )

    assert completed == ["tensor", "none"]


def test_cuda_toolkit_env_exposes_python_env_ninja(monkeypatch, tmp_path):
    from kernelforge import run_cast

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    ninja = venv_bin / "ninja"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    ninja.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    ninja.chmod(0o755)

    cuda_root = tmp_path / "cuda"
    cuda_bin = cuda_root / "bin"
    cuda_bin.mkdir(parents=True)
    nvcc = cuda_bin / "nvcc"
    nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    nvcc.chmod(0o755)

    clean_path = tmp_path / "clean-path"
    clean_path.mkdir()
    monkeypatch.setattr(run_cast.sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(clean_path))
    monkeypatch.setenv("CUDA_HOME", str(cuda_root))

    assert run_cast.ensure_cuda_toolkit_env() == str(cuda_root)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert str(venv_bin) in path_entries
    assert str(cuda_bin) in path_entries


def test_provenance_toolchain_status_exposes_python_env_ninja(monkeypatch, tmp_path):
    from paper_benchmarks.paper_bench import provenance

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    real_python = tmp_path / "python-real"
    python = venv_bin / "python"
    ninja = venv_bin / "ninja"
    real_python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.symlink_to(real_python)
    ninja.write_text("#!/bin/sh\necho 1.13.0\n", encoding="utf-8")
    real_python.chmod(0o755)
    ninja.chmod(0o755)

    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    nvcc = cuda_bin / "nvcc"
    nvcc.write_text("#!/bin/sh\necho nvcc test\n", encoding="utf-8")
    nvcc.chmod(0o755)

    monkeypatch.setattr(provenance.sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(cuda_bin))

    status = provenance.collect_toolchain_status()

    assert status["nvcc_path"] == str(nvcc)
    assert status["ninja_path"] == str(ninja)
    assert status["path_contains_python_env_bin"] is True
    assert status["jit_ready"] is True


def test_gelu_launch_arg_completion_preserves_explicit_approximation():
    from kernelforge.run_cast import _complete_functional_launch_args

    completed = _complete_functional_launch_args(
        op_name="torch_nn_functional_gelu",
        call_args=["tensor"],
        resolved_args={"approximate": "tanh"},
        kwargs={},
        n_launch=2,
    )

    assert completed == ["tensor", "tanh"]


def test_softmax_launch_arg_completion_uses_pytorch_stacklevel_default():
    from kernelforge.run_cast import _complete_functional_launch_args

    completed = _complete_functional_launch_args(
        op_name="torch_nn_functional_softmax",
        call_args=["tensor", -1, None, torch.float32],
        resolved_args={"dim": -1, "dtype": torch.float32},
        kwargs={},
        n_launch=4,
    )

    assert completed == ["tensor", -1, 3, torch.float32]


def test_softmax_launch_arg_completion_preserves_explicit_stacklevel():
    from kernelforge.run_cast import _complete_functional_launch_args

    completed = _complete_functional_launch_args(
        op_name="torch_nn_functional_softmax",
        call_args=["tensor", -1],
        resolved_args={"dim": -1, "_stacklevel": 5, "dtype": torch.float32},
        kwargs={},
        n_launch=4,
    )

    assert completed == ["tensor", -1, 5, torch.float32]


def _make_cast_zip(path: Path) -> Path:
    kernel_bytes = b"// fake kernel\n"
    kernel_hash = hashlib.sha256(kernel_bytes).hexdigest()
    manifest = {
        "project_name": "toy",
        "project_id": "toy",
        "project_ref": "project/toy/",
        "model_class": "",
        "model_entrypoints": {"build_model": True, "load_weights": False, "sample_inputs": False},
        "model_init_args": {},
        "weight_file": "",
        "selection_policy": "auto_best_fastest_valid",
        "selected_ops": ["torch_nn_functional_linear"],
        "selected_kernel_metadata": {
            "torch_nn_functional_linear": {
                "selected_source_hash": kernel_hash,
                "evidence_tier": "deployment",
            }
        },
        "ops": [
            {
                "name": "torch_nn_functional_linear",
                "cuda_source": "kernels/linear.cu",
                "precompiled": {},
            }
        ],
    }
    model_py = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Tiny(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "def build_model():\n"
        "    return Tiny()\n"
    ).encode("utf-8")
    return _write_cast_archive(
        path,
        {
            "manifest.json": json.dumps(manifest).encode("utf-8"),
            "model.py": model_py,
            "kernels/linear.cu": kernel_bytes,
        },
    )


def _write_cast_archive(path: Path, file_map: dict[str, bytes]) -> Path:
    checksum_lines = [
        f"{hashlib.sha256(data).hexdigest()}  {relpath}"
        for relpath, data in sorted(file_map.items())
    ]
    checksums_bytes = "\n".join(checksum_lines).encode("utf-8")
    header = {
        "format_version": "1.0",
        "file_type": "kernelforge_inference",
        "project_name": "toy",
        "archive_checksum": hashlib.sha256(checksums_bytes).hexdigest(),
        "runtime": {
            "target_sm_versions": [],
            "gpu_name": None,
            "gpu_capability": None,
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("HEADER.json", json.dumps(header).encode("utf-8"))
        for relpath, data in sorted(file_map.items()):
            archive.writestr(relpath, data)
        archive.writestr("checksums.sha256", checksums_bytes)
    return path


def _empty_runtime_stats() -> dict:
    return {
        "patched_calls": 0,
        "kernel_launches_attempted": 0,
        "kernel_launches_succeeded": 0,
        "kernel_launches_failed": 0,
        "fallbacks_to_original": 0,
        "exception_fallback_count": 0,
        "contiguous_copy_count": 0,
        "adaptation_count": 0,
        "per_op": {},
    }


class FakeKfRuntimeModel(torch.nn.Module):
    def __init__(
        self,
        *,
        fallback_prefill_runs: set[int] | None = None,
        mismatch_prefill_runs: set[int] | None = None,
    ):
        super().__init__()
        self.base_model = ToyLM()
        self.fallback_prefill_runs = set(fallback_prefill_runs or set())
        self.mismatch_prefill_runs = set(mismatch_prefill_runs or set())
        self.prefill_run_count = 0
        self.active_run_index = 0
        self._cast_functional_patches = {"linear": object()}
        self._kf_runtime_stats = _empty_runtime_stats()
        self._kf_runtime_report = {
            "runtime_patch_enabled": True,
            "selected_ops": ["torch_nn_functional_linear"],
            "loaded_kernels": [{"op_name": "torch_nn_functional_linear", "load_mode": "jit"}],
            "load_modes": {"torch_nn_functional_linear": "jit"},
            "jit_compile_time_ms": 7.0,
            "precompiled_load_time_ms": 0.0,
            "runtime_load_time_ms": 12.0,
            "setup_time_ms": 5.0,
            "op_reports": [{"op_name": "torch_nn_functional_linear", "load_mode": "jit", "patch_registered": True}],
        }

    def _op_stats(self) -> dict:
        return self._kf_runtime_stats["per_op"].setdefault(
            "torch_nn_functional_linear",
            {
                "patched_calls": 0,
                "kernel_launches_attempted": 0,
                "kernel_launches_succeeded": 0,
                "kernel_launches_failed": 0,
                "fallbacks_to_original": 0,
                "exception_fallback_count": 0,
                "contiguous_copy_count": 0,
                "adaptation_count": 0,
                "last_exception": None,
                "fallback_reasons": {},
            },
        )

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        stats = self._kf_runtime_stats
        op_stats = self._op_stats()
        stats["patched_calls"] += 1
        op_stats["patched_calls"] += 1
        stats["kernel_launches_attempted"] += 1
        op_stats["kernel_launches_attempted"] += 1

        if attention_mask is not None:
            self.prefill_run_count += 1
            self.active_run_index = self.prefill_run_count

        if self.active_run_index in self.fallback_prefill_runs:
            stats["kernel_launches_failed"] += 1
            stats["fallbacks_to_original"] += 1
            stats["exception_fallback_count"] += 1
            op_stats["kernel_launches_failed"] += 1
            op_stats["fallbacks_to_original"] += 1
            op_stats["exception_fallback_count"] += 1
            op_stats["last_exception"] = "RuntimeError: synthetic kernel failure"
            op_stats["fallback_reasons"]["kernel_exception"] = int(op_stats["fallback_reasons"].get("kernel_exception", 0)) + 1
            previous_offset = self.base_model._extra_offset
            self.base_model._extra_offset = 0
            try:
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            finally:
                self.base_model._extra_offset = previous_offset

        stats["kernel_launches_succeeded"] += 1
        op_stats["kernel_launches_succeeded"] += 1
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


def _make_fake_cast_loader(
    *,
    fallback_prefill_runs: set[int] | None = None,
    mismatch_prefill_runs: set[int] | None = None,
    selected_source_hash: str = "fake-kernel-hash",
    project_ref: str | None = None,
):
    def _loader(cast_path, *, device=None, settings=None):
        record_runtime_stats = True
        if isinstance(settings, dict):
            record_runtime_stats = bool(settings.get("record_runtime_stats", True))
        elif settings is not None:
            record_runtime_stats = bool(getattr(settings, "record_runtime_stats", True))
        model = FakeKfRuntimeModel(
            fallback_prefill_runs=fallback_prefill_runs,
            mismatch_prefill_runs=mismatch_prefill_runs,
        )
        if device:
            model.to(device)
        runtime_meta = {
            "cast_package_path": str(cast_path),
            "cast_package_hash": "fake-cast-hash",
            "cast_manifest": {
                "project_ref": project_ref,
                "ops": [{"name": "torch_nn_functional_linear", "cuda_source": "kernels/linear.cu"}],
            },
            "manifest": {
                "project_ref": project_ref,
                "ops": [{"name": "torch_nn_functional_linear", "cuda_source": "kernels/linear.cu"}],
            },
            "project_ref": project_ref,
            "kernel_source_hashes": {"kernels/linear.cu": selected_source_hash},
            "selected_source_hashes": {"torch_nn_functional_linear": selected_source_hash},
            "precompiled_binary_hashes": {},
            "runtime_report": model._kf_runtime_report,
            "runtime_stats": model._kf_runtime_stats,
            "runtime_load_time_ms": 12.0,
            "setup_time_ms": 5.0,
            "load_time_ms": 5.0,
            "jit_compile_time_ms": 7.0,
            "precompiled_load_time_ms": 0.0,
            "precompiled_vs_jit_path": {"torch_nn_functional_linear": "jit"},
            "selected_ops": ["torch_nn_functional_linear"],
            "loaded_kernels": [{"op_name": "torch_nn_functional_linear", "load_mode": "jit"}],
            "runtime_patch_enabled": True,
            "runtime_stats_enabled": record_runtime_stats,
            "patched_call_count": 0,
            "kernel_hit_count": 0,
            "kernel_launches_attempted": 0,
            "kernel_launches_succeeded": 0,
            "kernel_launches_failed": 0,
            "fallback_count": 0,
            "exception_fallback_count": 0,
            "contiguous_copy_count": 0,
            "adaptation_count": 0,
        }
        return model, runtime_meta

    return _loader


def _configure_kf_context(common_fields: dict, model_spec, sample_paths: dict[str, str]) -> None:
    common_fields["kf_settings"] = {
        "cast_package_path": sample_paths["cast_path"],
        "require_precompiled": False,
        "allow_jit": True,
        "fail_on_fallback": True,
        "record_runtime_stats": True,
    }
    common_fields["cast_package_path"] = sample_paths["cast_path"]
    common_fields["cast_package_hash"] = sample_paths["cast_hash"]
    model_spec.cast_package_path = sample_paths["cast_path"]


def test_load_cast_model_records_runtime_metadata(tmp_path: Path):
    cast_path = _make_cast_zip(tmp_path / "toy.cast")

    class FakeLoadedModel:
        def __init__(self):
            self._kf_runtime_stats = {
                **_empty_runtime_stats(),
                "patched_calls": 5,
                "kernel_launches_attempted": 3,
                "kernel_launches_succeeded": 3,
            }
            self._kf_runtime_report = {
                "runtime_patch_enabled": True,
                "selected_ops": ["torch_nn_functional_linear"],
                "loaded_kernels": [{"op_name": "torch_nn_functional_linear", "load_mode": "precompiled"}],
                "load_modes": {"torch_nn_functional_linear": "precompiled"},
                "jit_compile_time_ms": 0.0,
                "precompiled_load_time_ms": 1.0,
                "runtime_load_time_ms": 4.0,
                "setup_time_ms": 4.0,
            }

    model, meta = load_cast_model(
        cast_path,
        loader=lambda *args, **kwargs: FakeLoadedModel(),
        stats_getter=lambda model: model._kf_runtime_stats,
    )

    assert meta["runtime_patch_enabled"] is True
    assert meta["selected_ops"] == ["torch_nn_functional_linear"]
    assert meta["kernel_hit_count"] == 3
    assert meta["precompiled_vs_jit_path"] == {"torch_nn_functional_linear": "precompiled"}
    assert meta["kernel_source_hashes"]["kernels/linear.cu"]
    assert meta["runtime_stats_api"]["fallback_count_after_reset"] == 0
    assert model is not None


def test_load_cast_model_applies_and_restores_loader_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cast_path = _make_cast_zip(tmp_path / "toy.cast")
    monkeypatch.setenv("KFORGE_DEVICE_MAP", "previous")
    monkeypatch.delenv("KFORGE_QWEN35_MAX_MEMORY", raising=False)
    monkeypatch.delenv("KFORGE_TARGET_DEVICE", raising=False)
    seen: dict[str, str | None] = {}

    class FakeLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self._kf_runtime_stats = _empty_runtime_stats()
            self._kf_runtime_report = {
                "runtime_patch_enabled": True,
                "selected_ops": ["torch_nn_functional_linear"],
                "loaded_kernels": [{"op_name": "torch_nn_functional_linear", "load_mode": "jit"}],
                "load_modes": {"torch_nn_functional_linear": "jit"},
                "jit_compile_time_ms": 0.0,
                "precompiled_load_time_ms": 0.0,
                "runtime_load_time_ms": 1.0,
                "setup_time_ms": 1.0,
            }

    def _loader(*args, **kwargs):
        seen["device"] = kwargs.get("device")
        seen["device_map"] = os.environ.get("KFORGE_DEVICE_MAP")
        seen["max_memory"] = os.environ.get("KFORGE_QWEN35_MAX_MEMORY")
        seen["target_device"] = os.environ.get("KFORGE_TARGET_DEVICE")
        return FakeLoadedModel()

    _, meta = load_cast_model(
        cast_path,
        device="cuda",
        settings=KfRuntimeSettings(
            placement_profile="single_cuda",
            device_map="single_cuda",
            max_memory={"0": "120GiB", "cpu": "8GiB"},
        ),
        loader=_loader,
        stats_getter=lambda model: model._kf_runtime_stats,
    )

    assert seen["device"] == "cuda"
    assert seen["device_map"] == "single_cuda"
    assert seen["max_memory"] == '{"0": "120GiB", "cpu": "8GiB"}'
    assert seen["target_device"] == "cuda"
    assert os.environ.get("KFORGE_DEVICE_MAP") == "previous"
    assert "KFORGE_QWEN35_MAX_MEMORY" not in os.environ
    assert "KFORGE_TARGET_DEVICE" not in os.environ
    assert meta["loader_env_overrides"]["KFORGE_DEVICE_MAP"] == "single_cuda"


def test_inspect_cast_package_records_manifest_and_source_hashes(tmp_path: Path):
    cast_path = _make_cast_zip(tmp_path / "toy.cast")

    inspected = inspect_cast_package(cast_path)

    assert inspected["manifest"]["selection_policy"] == "auto_best_fastest_valid"
    assert inspected["selected_ops"] == ["torch_nn_functional_linear"]
    assert inspected["kernel_source_hashes"]["kernels/linear.cu"] == inspected["selected_source_hashes"]["torch_nn_functional_linear"]


def test_precompiled_required_mode_fails_when_no_precompiled_binary_exists(tmp_path: Path):
    cast_path = _make_cast_zip(tmp_path / "toy.cast")
    loader_called = {"value": False}

    def _forbidden_loader(*args, **kwargs):
        loader_called["value"] = True
        raise AssertionError("loader should not be called when precompiled enforcement fails early")

    with pytest.raises(RuntimeError, match="Precompiled kernels were required"):
        load_cast_model(
            cast_path,
            settings=KfRuntimeSettings(require_precompiled=True),
            loader=_forbidden_loader,
        )

    assert loader_called["value"] is False


def test_jit_allowed_mode_records_jit_time(tmp_path: Path):
    cast_path = _make_cast_zip(tmp_path / "toy.cast")

    class FakeLoadedModel:
        def __init__(self):
            self._kf_runtime_stats = _empty_runtime_stats()
            self._kf_runtime_report = {
                "runtime_patch_enabled": True,
                "selected_ops": ["torch_nn_functional_linear"],
                "loaded_kernels": [{"op_name": "torch_nn_functional_linear", "load_mode": "jit"}],
                "load_modes": {"torch_nn_functional_linear": "jit"},
                "jit_compile_time_ms": 9.5,
                "precompiled_load_time_ms": 0.0,
                "runtime_load_time_ms": 13.0,
                "setup_time_ms": 3.5,
            }

    _, meta = load_cast_model(
        cast_path,
        settings=KfRuntimeSettings(allow_jit=True),
        loader=lambda *args, **kwargs: FakeLoadedModel(),
        stats_getter=lambda model: model._kf_runtime_stats,
    )

    assert meta["jit_compile_time_ms"] == 9.5
    assert meta["precompiled_vs_jit_path"] == {"torch_nn_functional_linear": "jit"}


def test_fallback_count_starts_at_zero_after_reset():
    class FakeLoadedModel:
        def __init__(self):
            self._kf_runtime_stats = _empty_runtime_stats()
            self._kf_runtime_stats["fallbacks_to_original"] = 4
            self._kf_runtime_stats["per_op"] = {
                "torch_nn_functional_linear": {
                    **_empty_runtime_stats(),
                    "fallback_reasons": {"kernel_exception": 4},
                    "last_exception": "RuntimeError: boom",
                }
            }

    model = FakeLoadedModel()

    def _stats_getter(current):
        return current._kf_runtime_stats

    def _stats_reset(current):
        current._kf_runtime_stats = _empty_runtime_stats()
        return current._kf_runtime_stats

    api = validate_runtime_stats_api(model, stats_getter=_stats_getter, stats_reset=_stats_reset)

    assert api["get_runtime_stats_exists"] is True
    assert api["reset_runtime_stats_exists"] is True
    assert api["fallback_count_after_reset"] == 0
    assert api["stats_after_reset"]["per_op"] == {}


def test_missing_manifest_fails(tmp_path: Path):
    cast_path = _write_cast_archive(
        tmp_path / "missing-manifest.cast",
        {
            "model.py": b"import torch.nn as nn\nclass Tiny(nn.Module):\n    pass\n",
        },
    )

    with pytest.raises(RuntimeError, match="manifest.json"):
        inspect_cast_package(cast_path)


def test_missing_selected_source_fails(tmp_path: Path):
    manifest = {
        "project_name": "toy",
        "project_id": "toy",
        "project_ref": "project/toy/",
        "model_class": "",
        "model_entrypoints": {"build_model": True},
        "model_init_args": {},
        "weight_file": "",
        "selected_ops": ["torch_nn_functional_linear"],
        "selected_kernel_metadata": {
            "torch_nn_functional_linear": {
                "selected_source_hash": "deadbeef",
                "evidence_tier": "deployment",
            }
        },
        "ops": [
            {
                "name": "torch_nn_functional_linear",
                "cuda_source": "kernels/missing.cu",
                "precompiled": {},
            }
        ],
    }
    cast_path = _write_cast_archive(
        tmp_path / "missing-source.cast",
        {
            "manifest.json": json.dumps(manifest).encode("utf-8"),
            "model.py": (
                "import torch.nn as nn\n"
                "class Tiny(nn.Module):\n"
                "    def forward(self, x):\n"
                "        return x\n"
                "def build_model():\n"
                "    return Tiny()\n"
            ).encode("utf-8"),
        },
    )

    with pytest.raises(RuntimeError, match="missing selected kernel source"):
        inspect_cast_package(cast_path)


def test_kf_cast_without_cast_package_fails_cleanly(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    common_fields["kf_settings"] = {
        "cast_package_path": None,
        "require_precompiled": False,
        "allow_jit": True,
        "fail_on_fallback": True,
        "record_runtime_stats": True,
    }
    model_spec.cast_package_path = None

    try:
        run_llm_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env,
            manifest_artifact=manifest,
            model_spec=model_spec,
            suite=suite,
            variant=Variant.kf_cast,
            model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
            cast_loader=_make_fake_cast_loader(),
        )
    except ValueError as exc:
        assert "cast package path" in str(exc)
    else:
        raise AssertionError("Expected kf_cast run to fail without a cast package path")


def test_kf_cast_with_missing_cast_file_fails(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    missing_cast = tmp_path / "missing.cast"
    common_fields["kf_settings"] = {
        "cast_package_path": str(missing_cast),
        "require_precompiled": False,
        "allow_jit": True,
        "fail_on_fallback": True,
        "record_runtime_stats": True,
    }
    common_fields["cast_package_path"] = str(missing_cast)
    common_fields["cast_package_hash"] = None
    model_spec.cast_package_path = str(missing_cast)

    with pytest.raises(FileNotFoundError, match="Cast package not found"):
        run_llm_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env,
            manifest_artifact=manifest,
            model_spec=model_spec,
            suite=suite,
            variant=Variant.kf_cast,
            model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        )


def test_fake_runtime_records_kernel_hit_and_load_compile_split(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    _configure_kf_context(common_fields, model_spec, sample_paths)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.kf_cast,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        cast_loader=_make_fake_cast_loader(),
    )

    load_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_load.json")
    compile_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_compile.json")
    total_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_total_generate.json")
    assert load_artifact.latency_samples_ms == [5.0]
    assert compile_artifact.compile_time_ms == 7.0
    assert total_artifact.steady_state_time_ms not in {5.0, 7.0}
    assert total_artifact.kernel_hit_count == 6
    assert total_artifact.fallback_count == 0
    assert total_artifact.details["runtime_patch_enabled"] is True
    assert total_artifact.details["runtime_load_time_ms"] == 12.0
    assert total_artifact.details["setup_time_ms"] == 5.0
    assert total_artifact.details["jit_compile_time_ms"] == 7.0
    assert total_artifact.details["precompiled_vs_jit_path"] == {"torch_nn_functional_linear": "jit"}
    assert total_artifact.details["cast_manifest"]["ops"][0]["name"] == "torch_nn_functional_linear"
    assert total_artifact.details["kernel_source_hashes"] == {"kernels/linear.cu": "fake-kernel-hash"}
    assert total_artifact.details["selected_source_hashes"] == {"torch_nn_functional_linear": "fake-kernel-hash"}
    assert total_artifact.details["exception_fallback_count"] == 0
    assert total_artifact.details["contiguous_copy_count"] == 0
    assert total_artifact.details["adaptation_count"] == 0
    audit_path = Path(total_artifact.details["device_audit_artifact_path"])
    audit_artifact = load_json_artifact(audit_path)
    assert audit_artifact.artifact_type == "device_audit"
    assert audit_artifact.audit_status == "passed"
    assert audit_artifact.runtime_input_device == "cpu"
    assert audit_artifact.kernel_launches_succeeded == 6


def test_fake_kernel_exception_marks_paper_ineligible(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    _configure_kf_context(common_fields, model_spec, sample_paths)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.kf_cast,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        cast_loader=_make_fake_cast_loader(fallback_prefill_runs={2}),
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_total_generate.json")
    assert total_artifact.fallback_count > 0
    assert total_artifact.paper_eligible is False
    assert total_artifact.details["exception_fallback_count"] > 0


def test_kf_cast_mismatch_against_eager_marks_incorrect(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    _configure_kf_context(common_fields, model_spec, sample_paths)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.kf_cast,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        cast_loader=_make_fake_cast_loader(mismatch_prefill_runs={2}),
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_total_generate.json")
    raw_rows = json.loads((layout.raw_dir / "kf_cast_llm_measurements.json").read_text(encoding="utf-8"))

    assert total_artifact.correctness_status.value == "failed"
    assert total_artifact.paper_eligible is False
    assert "output hash mismatch" in (total_artifact.correctness_message or "")
    assert raw_rows[0]["output_token_hashes"] != raw_rows[0]["reference_output_token_hashes"]


def test_kf_variant_cannot_be_summarized_as_win_without_correctness(sample_paths, tmp_path: Path):
    eager_layout, eager_common, eager_env, eager_manifest, eager_model_spec, suite = _make_context(tmp_path / "shared", sample_paths, Variant.eager)
    eager_common["run_id"] = eager_layout.run_id
    run_llm_benchmark(
        layout=eager_layout,
        common_fields=eager_common,
        env_artifact=eager_env,
        manifest_artifact=eager_manifest,
        model_spec=eager_model_spec,
        suite=suite,
        variant=Variant.eager,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
    )

    compile_manifest = eager_manifest.model_copy(update={"variant": Variant.torch_compile})
    compile_common = dict(eager_common)
    compile_common["variant"] = "torch_compile"
    run_llm_benchmark(
        layout=eager_layout,
        common_fields=compile_common,
        env_artifact=eager_env.model_copy(update={"variant": Variant.torch_compile}),
        manifest_artifact=compile_manifest,
        model_spec=eager_model_spec,
        suite=suite,
        variant=Variant.torch_compile,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        compile_model_fn=_make_compile_fn(),
    )

    kf_manifest = eager_manifest.model_copy(update={"variant": Variant.kf_cast})
    kf_common = dict(eager_common)
    kf_common["variant"] = "kf_cast"
    _configure_kf_context(kf_common, eager_model_spec, sample_paths)
    run_llm_benchmark(
        layout=eager_layout,
        common_fields=kf_common,
        env_artifact=eager_env.model_copy(update={"variant": Variant.kf_cast}),
        manifest_artifact=kf_manifest,
        model_spec=eager_model_spec,
        suite=suite,
        variant=Variant.kf_cast,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        cast_loader=_make_fake_cast_loader(mismatch_prefill_runs={3}),
    )

    summary = summarize_run(eager_layout.run_dir)
    kf_rows = [row for row in summary.rows if row.variant == Variant.kf_cast and row.stage.value == "total_generate"]
    assert len(kf_rows) == 1
    assert kf_rows[0].claim_eligible is False
    assert kf_rows[0].speedup_vs_eager is not None
    assert not any("Kernel Forge improves model throughput on this workload." in claim for claim in summary.paper_eligible_claims)
    assert any("output tokens do not exactly match eager" in claim for claim in summary.forbidden_claims)


def test_paper_run_fails_if_fallback_nonzero_when_enabled(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    _configure_kf_context(common_fields, model_spec, sample_paths)

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.kf_cast,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        cast_loader=_make_fake_cast_loader(fallback_prefill_runs={2}),
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_total_generate.json")
    assert total_artifact.paper_eligible is False
    assert "fallback observed while fail_on_fallback is enabled" in total_artifact.paper_eligibility_issues


def test_disabled_runtime_stats_marks_kf_cast_non_paper(sample_paths, tmp_path: Path):
    layout, common_fields, env, manifest, model_spec, suite = _make_context(tmp_path, sample_paths, Variant.kf_cast)
    _configure_kf_context(common_fields, model_spec, sample_paths)
    common_fields["kf_settings"]["record_runtime_stats"] = False

    run_llm_benchmark(
        layout=layout,
        common_fields=common_fields,
        env_artifact=env,
        manifest_artifact=manifest,
        model_spec=model_spec,
        suite=suite,
        variant=Variant.kf_cast,
        model_loader=lambda model_spec, device=None: (ToyLM(), ToyTokenizer(), 0.0),
        cast_loader=_make_fake_cast_loader(),
    )

    total_artifact = load_json_artifact(layout.metrics_dir / "kf_cast_total_generate.json")
    assert total_artifact.details["runtime_stats_enabled"] is False
    assert total_artifact.fallback_count is None
    assert total_artifact.kernel_hit_count is None
    assert total_artifact.paper_eligible is False
    assert "fallback count missing" in total_artifact.paper_eligibility_issues
