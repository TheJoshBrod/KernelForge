from __future__ import annotations

import pytest

from src.optimizer.benchmarking import benchmark_ops
from src.optimizer.benchmarking import profile_project
from src.optimizer.benchmarking.profile_project import get_samples


class _ListSampleModule:
    def sample_inputs(self):
        return ["a", "b", "c"]


class _ValidationPreferredModule:
    def __init__(self) -> None:
        self.sample_called = False
        self.validation_path = None

    def sample_inputs(self):
        self.sample_called = True
        return ["sample"]

    def get_validation_dataloader(self, validation_path=None):
        self.validation_path = validation_path
        return ["validation-a", "validation-b"]


def test_profile_project_limits_list_samples() -> None:
    assert get_samples(_ListSampleModule(), 2, None) == ["a", "b"]


def test_profile_project_prefers_validation_loader_when_path_is_configured() -> None:
    module = _ValidationPreferredModule()

    samples = get_samples(module, 1, "/tmp/validation")

    assert samples == ["validation-a"]
    assert module.validation_path == "/tmp/validation"
    assert module.sample_called is False


def test_profile_project_explicit_cuda_does_not_fallback_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    monkeypatch.setattr(profile_project.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="KFORGE_TARGET_DEVICE=cuda"):
        profile_project._resolve_device()


def test_benchmark_ops_explicit_cuda_does_not_fallback_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    monkeypatch.setattr(benchmark_ops.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="KFORGE_TARGET_DEVICE=cuda"):
        benchmark_ops._resolve_device()
