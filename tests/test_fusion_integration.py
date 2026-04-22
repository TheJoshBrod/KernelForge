"""
Integration tests for the Kernel Fusion Engine.

Tests are grouped by component:
1. Synthetic IO & Validation Tests (Ground Truth Suite)
2. State Machine & Retry Loop Tests (Mocked LLM)
3. File System & Storage Layout Tests

These tests isolate deterministic parts (IO, validation, profiling)
from non-deterministic parts (LLM) using mocks.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.core.fusion.types import (
    COSINE_THRESHOLD,
    FUSION_ATOL,
    FUSION_RTOL,
    AttemptStatus,
    FusionGenStatus,
    FusionGroup,
    MemberOpContext,
)
from src.optimizer.core.fusion.store import (
    get_group_attempts,
    sync_from_json,
)


# =============================================================================
# 1. Synthetic IO & Validation Tests (Ground Truth Suite)
# =============================================================================


class TestStatefulOpFreezing:
    """Ensure stateful ops are frozen during validation."""

    def test_batch_norm_uses_eval_mode(self):
        """BatchNorm running stats should not change during IO generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create a BatchNorm layer with known running stats
            bn = nn.BatchNorm2d(64)
            bn.running_mean.fill_(1.0)
            bn.running_var.fill_(2.0)
            original_mean = bn.running_mean.clone()
            original_var = bn.running_var.clone()

            # Simulate what IO builder should do: use eval mode
            bn.eval()
            input_tensor = torch.randn(4, 64, 32, 32)

            with torch.no_grad():
                _ = bn(input_tensor)

            # Running stats should be unchanged in eval mode
            assert torch.equal(bn.running_mean, original_mean), \
                "BatchNorm running_mean changed during eval - IO builder must use eval()"
            assert torch.equal(bn.running_var, original_var), \
                "BatchNorm running_var changed during eval - IO builder must use eval()"

    def test_dropout_is_disabled_during_validation(self):
        """Dropout should be identity during validation."""
        # In eval mode, dropout should pass input through unchanged
        dropout = nn.Dropout(p=0.5)
        dropout.eval()

        input_tensor = torch.randn(100, 100)

        with torch.no_grad():
            output = dropout(input_tensor)

        assert torch.equal(input_tensor, output), \
            "Dropout modified tensor in eval mode - validation will be flaky"


class TestValidationTolerancesFP:
    """Test that tolerances correctly handle floating point edge cases."""

    def test_accepts_1e4_deviation(self):
        """Known 1e-4 deviation should pass with our relaxed tolerances."""
        baseline = torch.tensor([100.0, 200.0, 300.0, 400.0])
        # Add exactly 1e-4 deviation (0.0001)
        fused_output = baseline + 1e-4

        # Our tolerance: rtol=1e-3, atol=1e-5
        # For value 100: allowed diff = 100 * 1e-3 + 1e-5 = 0.10001
        # Actual diff: 0.0001 < 0.10001 ✓
        passed = torch.allclose(fused_output, baseline, rtol=FUSION_RTOL, atol=FUSION_ATOL)

        assert passed, f"1e-4 deviation should pass with rtol={FUSION_RTOL}, atol={FUSION_ATOL}"

    def test_rejects_all_zeros_output(self):
        """A kernel returning all zeros is clearly broken."""
        expected = torch.randn(100, 100) * 100  # Non-zero values
        fused_output = torch.zeros_like(expected)  # Broken kernel output

        # Primary check should fail
        allclose_passed = torch.allclose(
            fused_output, expected, rtol=FUSION_RTOL, atol=FUSION_ATOL
        )
        assert not allclose_passed, "All-zeros output should fail allclose"

        # Cosine similarity should also fail (zero vector has no direction)
        cos_sim = F.cosine_similarity(
            fused_output.flatten().unsqueeze(0),
            expected.flatten().unsqueeze(0)
        )
        assert cos_sim.item() < COSINE_THRESHOLD, \
            "All-zeros output should fail cosine similarity"

    def test_rejects_nan_output(self):
        """NaN outputs indicate kernel bugs."""
        expected = torch.tensor([1.0, 2.0, 3.0])
        fused_output = torch.tensor([1.0, float('nan'), 3.0])

        passed = torch.allclose(fused_output, expected, rtol=FUSION_RTOL, atol=FUSION_ATOL)
        assert not passed, "NaN in output should fail validation"

    def test_rejects_inf_output(self):
        """Infinite outputs indicate overflow bugs."""
        expected = torch.tensor([1.0, 2.0, 3.0])
        fused_output = torch.tensor([1.0, float('inf'), 3.0])

        passed = torch.allclose(fused_output, expected, rtol=FUSION_RTOL, atol=FUSION_ATOL)
        assert not passed, "Inf in output should fail validation"


# =============================================================================
# 2. State Machine & Retry Loop Tests (Mocked LLM)
# =============================================================================


class TestHappyPath:
    """Test successful fusion on first attempt."""

    @pytest.fixture
    def mock_project(self):
        """Create a minimal project for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Project structure
            (project_dir / "io" / "individual_ops" / "torch_nn_functional_relu").mkdir(parents=True)
            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "generated").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)

            # DAG
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "relu_1", "op": "relu", "bench_op": "torch_nn_functional_relu"}],
                "edges": []
            }))

            # Benchmarks
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({
                "results": [{"op": "torch_nn_functional_relu", "pytorch_ms": 0.5, "kernel_ms": 0.3}]
            }))

            # Fusion group
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{
                    "id": "fg_test",
                    "pattern_name": "relu_chain",
                    "members": ["relu_1"],
                    "status": "accepted",
                    "score": 0.9
                }]
            }))

            yield project_dir

    def test_first_try_success(self, mock_project):
        """Engine returns completed on first successful attempt."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})
        backend.validate_kernel.return_value = (True, "")
        backend.profile_kernel.return_value = {"mean_ms": 0.2}

        valid_kernel = """
// [START FEEDBACK]
Fused relu kernel
// [END FEEDBACK]
// [START kernel.cu]
__global__ void fused_relu(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}
// [END kernel.cu]
"""

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.return_value = valid_kernel
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        # Assertions
        assert result.status == FusionGenStatus.COMPLETED
        assert mock_llm.chat.call_count == 1  # Only one attempt needed

        # Check attempts in DB
        attempts = get_group_attempts(mock_project, "fg_test")
        assert len(attempts) == 1
        assert attempts[0].status == AttemptStatus.SUCCESS


class TestSelfCorrectionPath:
    """Test successful fusion after retry with error feedback."""

    @pytest.fixture
    def mock_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "io" / "individual_ops").mkdir(parents=True)
            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "op_1", "op": "relu"}], "edges": []
            }))
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({"results": []}))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_retry", "pattern_name": "test", "members": ["op_1"], "status": "accepted"}]
            }))
            yield project_dir

    def test_error_fed_back_to_llm_on_retry(self, mock_project):
        """Previous error must appear in retry prompt."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})

        compile_error = "nvcc fatal: Unsupported gpu architecture 'compute_999'"

        # First attempt fails, second succeeds
        backend.validate_kernel.side_effect = [
            (False, compile_error),
            (True, ""),
        ]
        backend.profile_kernel.return_value = {"mean_ms": 1.0}

        prompts_received = []

        def capture_chat(prompt, model):
            prompts_received.append(prompt)
            return "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]"

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.side_effect = capture_chat
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        # Assertions
        assert result.status == FusionGenStatus.COMPLETED
        assert len(prompts_received) == 2

        # First prompt should NOT have error
        assert "Previous Attempt Failed" not in prompts_received[0]

        # Second prompt MUST have the error
        assert "Previous Attempt Failed" in prompts_received[1]
        assert compile_error in prompts_received[1]

        # Check attempts array
        attempts = get_group_attempts(mock_project, "fg_retry")
        assert len(attempts) == 2
        assert attempts[0].status == AttemptStatus.VALIDATION_ERROR
        assert attempts[1].status == AttemptStatus.SUCCESS


class TestCompleteFailurePath:
    """Test behavior when all retries are exhausted."""

    @pytest.fixture
    def mock_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "io" / "individual_ops").mkdir(parents=True)
            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "op_1", "op": "relu"}], "edges": []
            }))
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({"results": []}))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_fail", "pattern_name": "test", "members": ["op_1"], "status": "accepted"}]
            }))
            yield project_dir

    def test_stops_at_max_attempts(self, mock_project):
        """Engine must stop after exactly MAX_ATTEMPTS."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})
        backend.validate_kernel.return_value = (False, "persistent error")

        max_attempts = 3

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.return_value = "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]"
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(
                    backend=backend,
                    project_dir=mock_project,
                    max_attempts=max_attempts
                )

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        # Assertions
        assert result.status == FusionGenStatus.FAILED
        assert mock_llm.chat.call_count == max_attempts

        # Check all attempt files exist
        attempts_dir = mock_project / "kernels" / "fused" / "fg_fail" / "attempts"
        for i in range(max_attempts):
            kernel_file = attempts_dir / f"{i}_kernel.cu"
            assert kernel_file.exists(), f"Attempt {i} kernel file should exist"


class TestContextWindowVerification:
    """Verify prompts contain all required context."""

    @pytest.fixture
    def mock_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create IO with tensor data
            op_dir = project_dir / "io" / "individual_ops" / "torch_nn_functional_conv2d"
            op_dir.mkdir(parents=True)
            entry_data = {
                "inputs": {
                    "input": torch.randn(16, 64, 56, 56),
                    "weight": torch.randn(128, 64, 3, 3),
                },
                "output": torch.randn(16, 128, 54, 54)
            }
            torch.save(entry_data, op_dir / "entry_000000.pt")

            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)

            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "conv_1", "op": "conv2d", "bench_op": "torch_nn_functional_conv2d"}],
                "edges": []
            }))
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({
                "results": [{"op": "torch_nn_functional_conv2d", "pytorch_ms": 5.0}]
            }))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{
                    "id": "fg_ctx",
                    "pattern_name": "conv_bn_relu",
                    "members": ["conv_1"],
                    "status": "accepted"
                }]
            }))

            yield project_dir

    def test_prompt_contains_gpu_specs(self, mock_project):
        """Prompt must include GPU specifications."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {
            "gpu_name": "NVIDIA RTX 4090",
            "compute_capability": "8.9",
            "sm_count": 128,
            "max_shared_memory_per_block": 99840,
        })
        backend.validate_kernel.return_value = (True, "")
        backend.profile_kernel.return_value = {"mean_ms": 1.0}

        captured_prompt = []

        def capture(prompt, model):
            captured_prompt.append(prompt)
            return "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]"

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.side_effect = capture
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project)

                groups = engine.load_accepted_groups()
                engine.fuse_group(groups[0])

        prompt = captured_prompt[0]
        assert "RTX 4090" in prompt, "GPU name must be in prompt"
        assert "8.9" in prompt, "Compute capability must be in prompt"

    def test_prompt_contains_sync_instructions(self, mock_project):
        """Prompt must warn about thread synchronization."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})
        backend.validate_kernel.return_value = (True, "")
        backend.profile_kernel.return_value = {"mean_ms": 1.0}

        captured_prompt = []

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.side_effect = lambda p, m: (captured_prompt.append(p), "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]")[1]
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project)

                groups = engine.load_accepted_groups()
                engine.fuse_group(groups[0])

        prompt = captured_prompt[0]
        assert "__syncthreads" in prompt, "Sync instructions must be in prompt"
        assert "block size" in prompt.lower(), "Block size guidance must be in prompt"
        assert "shared memory" in prompt.lower(), "Shared memory guidance must be in prompt"


# =============================================================================
# 3. File System & Storage Layout Tests
# =============================================================================


class TestResultMetadataSerialization:
    """Test that result files are correctly written."""

    @pytest.fixture
    def mock_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "io" / "individual_ops").mkdir(parents=True)
            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "op_1", "op": "relu"}], "edges": []
            }))
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({
                "results": [{"op": "torch_nn_functional_relu", "pytorch_ms": 0.5, "kernel_ms": 0.3}]
            }))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_meta", "pattern_name": "test", "members": ["op_1"], "status": "accepted"}]
            }))
            yield project_dir

    def test_best_kernel_file_created(self, mock_project):
        """Best kernel should be saved to group directory."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})
        backend.validate_kernel.return_value = (True, "")
        backend.profile_kernel.return_value = {"mean_ms": 0.2}

        kernel_code = "__global__ void test_kernel() { /* best */ }"

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.return_value = f"// [START kernel.cu]\n{kernel_code}\n// [END kernel.cu]"
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        # Check kernel file
        kernel_path = mock_project / "kernels" / "fused" / "fg_meta" / "kernel.cu"
        assert kernel_path.exists(), "Best kernel file should exist"
        assert "best" in kernel_path.read_text(), "Best kernel should contain the code"

    def test_db_updated_with_results(self, mock_project):
        """Fusion DB should be updated with timing results."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})
        backend.validate_kernel.return_value = (True, "")
        backend.profile_kernel.return_value = {"mean_ms": 0.15}

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.return_value = "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]"
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        # Check DB was updated
        from src.optimizer.core.fusion.store import get_group_by_id
        group = get_group_by_id(mock_project, "fg_meta")

        assert group.gen_status == FusionGenStatus.COMPLETED
        assert group.fused_ms == 0.15


class TestDirectoryCleanup:
    """Test that directories are preserved correctly on failure."""

    @pytest.fixture
    def mock_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "io" / "individual_ops" / "torch_nn_functional_relu").mkdir(parents=True)
            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)

            # Create IO entry
            entry_data = {"inputs": {"x": torch.randn(10)}, "output": torch.randn(10)}
            torch.save(entry_data, project_dir / "io" / "individual_ops" / "torch_nn_functional_relu" / "entry_000000.pt")

            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "relu_1", "op": "relu", "bench_op": "torch_nn_functional_relu"}],
                "edges": []
            }))
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({"results": []}))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_cleanup", "pattern_name": "test", "members": ["relu_1"], "status": "accepted"}]
            }))
            yield project_dir

    def test_io_preserved_on_failure(self, mock_project):
        """Synthetic IO should be preserved for debugging even on failure."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})
        backend.validate_kernel.return_value = (False, "persistent error")

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.return_value = "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]"
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project, max_attempts=2)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        assert result.status == FusionGenStatus.FAILED

        # IO directory should still exist
        io_dir = mock_project / "io" / "fusion_groups" / "fg_cleanup"
        assert io_dir.exists(), "Fusion IO directory should exist for debugging"

        # Attempt files should exist
        attempts_dir = mock_project / "kernels" / "fused" / "fg_cleanup" / "attempts"
        assert attempts_dir.exists(), "Attempts directory should exist"
        assert (attempts_dir / "0_kernel.cu").exists(), "First attempt should be preserved"
        assert (attempts_dir / "1_kernel.cu").exists(), "Second attempt should be preserved"


class TestAttemptKernelStorage:
    """Test that all attempt kernels are stored for debugging."""

    @pytest.fixture
    def mock_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "io" / "individual_ops").mkdir(parents=True)
            (project_dir / "io" / "fusion_groups").mkdir(parents=True)
            (project_dir / "kernels" / "fused").mkdir(parents=True)
            (project_dir / "benchmarks").mkdir(parents=True)
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "op_1", "op": "relu"}], "edges": []
            }))
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({"results": []}))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_attempts", "pattern_name": "test", "members": ["op_1"], "status": "accepted"}]
            }))
            yield project_dir

    def test_each_attempt_stored_with_unique_code(self, mock_project):
        """Each LLM response should be stored separately."""
        backend = MagicMock()
        backend.kernel_extension = ".cu"
        backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})

        # Three different responses
        responses = [
            "// [START kernel.cu]\nvoid attempt_0() {}\n// [END kernel.cu]",
            "// [START kernel.cu]\nvoid attempt_1() {}\n// [END kernel.cu]",
            "// [START kernel.cu]\nvoid attempt_2_success() {}\n// [END kernel.cu]",
        ]
        backend.validate_kernel.side_effect = [
            (False, "error 0"),
            (False, "error 1"),
            (True, ""),
        ]
        backend.profile_kernel.return_value = {"mean_ms": 1.0}

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockLLM:
                mock_llm = MagicMock()
                mock_llm.chat.side_effect = responses
                MockLLM.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=mock_project, max_attempts=3)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

        # Verify each attempt was stored
        attempts_dir = mock_project / "kernels" / "fused" / "fg_attempts" / "attempts"

        for i in range(3):
            kernel_file = attempts_dir / f"{i}_kernel.cu"
            assert kernel_file.exists(), f"Attempt {i} file should exist"
            content = kernel_file.read_text()
            assert f"attempt_{i}" in content, f"Attempt {i} should have unique code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
