"""
Behavioral tests for the kernel fusion engine.
Tests focus on what the code DOES, not what objects contain.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer.core.fusion.types import (
    COSINE_THRESHOLD,
    FUSION_ATOL,
    FUSION_RTOL,
    MAX_ATTEMPTS,
    AttemptStatus,
    FusionGenStatus,
    FusionGroup,
    FusionUIStatus,
    MemberOpContext,
)
from src.optimizer.core.fusion.store import (
    ensure_fusion_schema,
    get_accepted_groups,
    get_group_attempts,
    get_group_by_id,
    record_attempt,
    sync_from_json,
    update_group_status,
)
from src.optimizer.core.fusion.context import compute_baseline_timing


# =============================================================================
# Test Store Behavior
# =============================================================================


class TestStoreFiltersAcceptedGroups:
    """Store should only return groups that need fusion generation."""

    @pytest.fixture
    def project_with_mixed_groups(self):
        """Create project with groups in various states."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            json_data = {
                "groups": [
                    {"id": "fg_accepted", "pattern_name": "p1", "members": ["a"], "status": "accepted"},
                    {"id": "fg_rejected", "pattern_name": "p2", "members": ["b"], "status": "rejected"},
                    {"id": "fg_proposed", "pattern_name": "p3", "members": ["c"], "status": "proposed"},
                ]
            }
            (project_dir / "fusion_groups.json").write_text(json.dumps(json_data))
            sync_from_json(project_dir)
            yield project_dir

    def test_only_accepted_groups_returned(self, project_with_mixed_groups):
        """Should filter out rejected and proposed groups."""
        groups = get_accepted_groups(project_with_mixed_groups)

        assert len(groups) == 1
        assert all(g.ui_status == FusionUIStatus.ACCEPTED for g in groups)

    def test_completed_groups_excluded_from_pending(self, project_with_mixed_groups):
        """Groups already completed should not be returned for re-processing."""
        # Mark one as completed
        update_group_status(
            project_with_mixed_groups,
            "fg_accepted",
            gen_status=FusionGenStatus.COMPLETED.value
        )

        groups = get_accepted_groups(project_with_mixed_groups)
        assert len(groups) == 0

    def test_failed_groups_can_be_retried(self, project_with_mixed_groups):
        """Failed groups should be eligible for retry."""
        update_group_status(
            project_with_mixed_groups,
            "fg_accepted",
            gen_status=FusionGenStatus.FAILED.value
        )

        groups = get_accepted_groups(project_with_mixed_groups)
        assert len(groups) == 1
        assert groups[0].gen_status == FusionGenStatus.FAILED


class TestStoreTracksAttempts:
    """Store should track all fusion attempts for debugging."""

    @pytest.fixture
    def project_with_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            json_data = {
                "groups": [{"id": "fg_0", "pattern_name": "test", "members": ["op_1"], "status": "accepted"}]
            }
            (project_dir / "fusion_groups.json").write_text(json.dumps(json_data))
            sync_from_json(project_dir)
            yield project_dir

    def test_multiple_attempts_preserved_in_order(self, project_with_group):
        """All attempts should be stored and retrievable in order."""
        # Simulate retry loop with failures then success
        record_attempt(project_with_group, "fg_0", {
            "attempt_num": 0, "status": "compile_error", "error_message": "syntax error"
        })
        record_attempt(project_with_group, "fg_0", {
            "attempt_num": 1, "status": "validation_error", "error_message": "output mismatch"
        })
        record_attempt(project_with_group, "fg_0", {
            "attempt_num": 2, "status": "success", "fused_ms": 1.5
        })

        attempts = get_group_attempts(project_with_group, "fg_0")

        assert len(attempts) == 3
        assert attempts[0].status == AttemptStatus.COMPILE_ERROR
        assert attempts[1].status == AttemptStatus.VALIDATION_ERROR
        assert attempts[2].status == AttemptStatus.SUCCESS
        assert attempts[2].fused_ms == 1.5

    def test_error_messages_stored_for_feedback(self, project_with_group):
        """Error messages must be stored so they can be fed back to LLM."""
        error_msg = "CUDA error: invalid memory access at line 42"
        record_attempt(project_with_group, "fg_0", {
            "attempt_num": 0, "status": "validation_error", "error_message": error_msg
        })

        attempts = get_group_attempts(project_with_group, "fg_0")
        assert attempts[0].error_message == error_msg


class TestStoreSyncsFromJson:
    """Store should import from UI's fusion_groups.json."""

    def test_sync_is_idempotent(self):
        """Running sync multiple times should not duplicate groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            json_data = {"groups": [{"id": "fg_0", "pattern_name": "test", "members": ["a"], "status": "accepted"}]}
            (project_dir / "fusion_groups.json").write_text(json.dumps(json_data))

            # Sync twice
            first_import = sync_from_json(project_dir)
            second_import = sync_from_json(project_dir)

            assert first_import == 1
            assert second_import == 0  # Already exists, not re-imported

            # Only one group in DB
            groups = get_accepted_groups(project_dir)
            assert len(groups) == 1


# =============================================================================
# Test Baseline Timing Calculation
# =============================================================================


class TestBaselineTimingPreference:
    """Baseline should prefer forged kernel timing over PyTorch timing."""

    def test_uses_kernel_ms_when_available(self):
        """Forged kernel timing is more accurate than PyTorch."""
        contexts = [
            MemberOpContext(node_id="a", op_type="conv2d", kernel_ms=1.0, pytorch_ms=5.0),
            MemberOpContext(node_id="b", op_type="relu", kernel_ms=0.1, pytorch_ms=0.5),
        ]

        baseline = compute_baseline_timing(contexts)

        # Should use kernel_ms (1.0 + 0.1) not pytorch_ms (5.0 + 0.5)
        assert baseline == 1.1

    def test_falls_back_to_pytorch_when_no_kernel(self):
        """Use PyTorch timing if kernel not yet generated."""
        contexts = [
            MemberOpContext(node_id="a", op_type="conv2d", pytorch_ms=5.0),
            MemberOpContext(node_id="b", op_type="relu", pytorch_ms=0.5),
        ]

        baseline = compute_baseline_timing(contexts)
        assert baseline == 5.5

    def test_mixes_sources_per_op(self):
        """Each op can use different timing source."""
        contexts = [
            MemberOpContext(node_id="a", op_type="conv2d", kernel_ms=1.0),  # Has kernel
            MemberOpContext(node_id="b", op_type="relu", pytorch_ms=0.5),   # No kernel yet
        ]

        baseline = compute_baseline_timing(contexts)
        assert baseline == 1.5

    def test_returns_none_when_no_timing_data(self):
        """Cannot compute baseline without any timing data."""
        contexts = [
            MemberOpContext(node_id="a", op_type="conv2d"),
            MemberOpContext(node_id="b", op_type="relu"),
        ]

        baseline = compute_baseline_timing(contexts)
        assert baseline is None


# =============================================================================
# Test Validation Tolerances
# =============================================================================


class TestValidationTolerances:
    """Validation should use relaxed tolerances suitable for GPU computation."""

    def test_accepts_small_numerical_drift(self):
        """Small floating point differences should pass."""
        # Use deterministic values within tolerance
        output = torch.tensor([100.0, 200.0, 300.0, 400.0])
        # Add noise well within rtol=1e-3 (0.1% of value)
        expected = torch.tensor([100.01, 200.02, 300.03, 400.04])  # ~0.01% diff

        passed = torch.allclose(output, expected, rtol=FUSION_RTOL, atol=FUSION_ATOL)
        assert passed

    def test_rejects_significant_differences(self):
        """Actual bugs (wrong values) should fail."""
        output = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([1.0, 2.0, 30.0])  # 10x error in one element

        passed = torch.allclose(output, expected, rtol=FUSION_RTOL, atol=FUSION_ATOL)
        assert not passed


class TestCosineSimilarityFallback:
    """Cosine similarity catches structurally correct outputs with numerical drift."""

    def test_passes_scaled_outputs(self):
        """Outputs that are scaled versions should pass cosine check."""
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        output = expected * 1.0001  # Tiny scale difference

        cos_sim = F.cosine_similarity(
            output.flatten().unsqueeze(0),
            expected.flatten().unsqueeze(0)
        )

        assert cos_sim.item() >= COSINE_THRESHOLD

    def test_fails_structurally_wrong_outputs(self):
        """Completely wrong outputs should fail cosine check."""
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        output = torch.tensor([4.0, 3.0, 2.0, 1.0])  # Reversed!

        cos_sim = F.cosine_similarity(
            output.flatten().unsqueeze(0),
            expected.flatten().unsqueeze(0)
        )

        assert cos_sim.item() < COSINE_THRESHOLD

    def test_fails_random_outputs(self):
        """Random garbage should definitely fail."""
        expected = torch.randn(100)
        output = torch.randn(100)  # Completely different random tensor

        cos_sim = F.cosine_similarity(
            output.flatten().unsqueeze(0),
            expected.flatten().unsqueeze(0)
        )

        # Random vectors unlikely to be similar
        assert cos_sim.item() < 0.99


# =============================================================================
# Test FusionEngine Behavior
# =============================================================================


class TestEngineRetryLoop:
    """Engine should retry on failure with error feedback."""

    @pytest.fixture
    def mock_engine_deps(self):
        """Mock external dependencies for engine testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Setup minimal project structure
            (project_dir / "io" / "dag.json").parent.mkdir(parents=True)
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "relu_1", "op": "relu"}],
                "edges": []
            }))
            (project_dir / "benchmarks").mkdir()
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({"results": []}))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_0", "pattern_name": "test", "members": ["relu_1"], "status": "accepted"}]
            }))

            # Mock backend
            backend = MagicMock()
            backend.kernel_extension = ".cu"
            backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {"gpu_name": "Test"})

            yield project_dir, backend

    def test_feeds_error_back_to_llm_on_retry(self, mock_engine_deps):
        """Previous error should be included in next attempt's prompt."""
        project_dir, backend = mock_engine_deps

        # First attempt fails validation
        backend.validate_kernel.side_effect = [
            (False, "CUDA error: race condition at line 42"),  # Attempt 1
            (True, ""),  # Attempt 2 succeeds
        ]
        backend.profile_kernel.return_value = {"mean_ms": 1.0}

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockGenModel:
                # Track what prompts are sent to LLM
                prompts_received = []
                def capture_prompt(prompt, model):
                    prompts_received.append(prompt)
                    return "// [START kernel.cu]\nvoid kernel() {}\n// [END kernel.cu]"

                mock_llm = MagicMock()
                mock_llm.chat.side_effect = capture_prompt
                MockGenModel.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=project_dir)

                groups = engine.load_accepted_groups()
                engine.fuse_group(groups[0])

                # Second prompt should contain the error from first attempt
                assert len(prompts_received) == 2
                assert "race condition" in prompts_received[1]

    def test_stops_after_max_attempts(self, mock_engine_deps):
        """Should not retry forever."""
        project_dir, backend = mock_engine_deps

        # All attempts fail
        backend.validate_kernel.return_value = (False, "persistent error")

        with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
            with patch("src.optimizer.core.fusion.engine.GenModel") as MockGenModel:
                mock_llm = MagicMock()
                mock_llm.chat.return_value = "// [START kernel.cu]\nvoid k() {}\n// [END kernel.cu]"
                MockGenModel.return_value = mock_llm

                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=project_dir, max_attempts=3)

                groups = engine.load_accepted_groups()
                result = engine.fuse_group(groups[0])

                # Should have exactly MAX_ATTEMPTS calls
                assert mock_llm.chat.call_count == 3
                assert result.status == FusionGenStatus.FAILED


class TestEngineContextGathering:
    """Engine should gather correct context from project files."""

    def test_finds_existing_kernel_code(self):
        """Should load existing kernel code to include in prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create kernel file
            kernel_dir = project_dir / "kernels" / "generated" / "individual_op_kernels" / "torch_nn_functional_relu"
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "kernel.cu").write_text("__global__ void relu_kernel() { /* existing */ }")

            # Create project structure
            (project_dir / "io" / "dag.json").parent.mkdir(parents=True)
            (project_dir / "io" / "dag.json").write_text(json.dumps({
                "nodes": [{"id": "relu_1", "op": "relu", "bench_op": "torch_nn_functional_relu"}],
                "edges": []
            }))
            (project_dir / "benchmarks").mkdir()
            (project_dir / "benchmarks" / "op_benchmarks.json").write_text(json.dumps({"results": []}))
            (project_dir / "fusion_groups.json").write_text(json.dumps({
                "groups": [{"id": "fg_0", "pattern_name": "test", "members": ["relu_1"], "status": "accepted"}]
            }))

            backend = MagicMock()
            backend.kernel_extension = ".cu"
            backend.get_device_specs.return_value = MagicMock(model_dump=lambda: {})

            with patch("src.optimizer.core.fusion.engine.ensure_llm_config"):
                from src.optimizer.core.fusion.engine import FusionEngine
                engine = FusionEngine(backend=backend, project_dir=project_dir)

                groups = engine.load_accepted_groups()
                contexts = engine.gather_member_context(groups[0])

                assert len(contexts) == 1
                assert "existing" in contexts[0].existing_kernel_code


# =============================================================================
# Test Prompt Generation Behavior
# =============================================================================


class TestPromptIncludesSyncGuidance:
    """Prompts must include thread synchronization guidance."""

    def test_cuda_prompt_warns_about_syncthreads(self):
        """CUDA prompt should emphasize __syncthreads() requirements."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cuda_fusion",
            Path(__file__).resolve().parents[1] / "src" / "optimizer" / "backends" / "cuda" / "fusion.py"
        )
        cuda_fusion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cuda_fusion)

        group = FusionGroup(id="fg_0", pattern_name="conv_bn_relu", members=["a", "b", "c"])
        contexts = [MemberOpContext(node_id=m, op_type="op") for m in group.members]

        prompt = cuda_fusion.generate_fusion_prompt({"gpu_name": "Test"}, group, contexts)

        # Must warn about synchronization
        assert "__syncthreads" in prompt
        assert "shared memory" in prompt.lower()
        assert "block size" in prompt.lower()

    def test_prompt_includes_previous_error_on_retry(self):
        """Retry prompts must include the error for LLM to fix."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cuda_fusion",
            Path(__file__).resolve().parents[1] / "src" / "optimizer" / "backends" / "cuda" / "fusion.py"
        )
        cuda_fusion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cuda_fusion)

        error = "nvcc error: undefined reference to 'missing_function'"
        prompt = cuda_fusion.generate_fusion_prompt(
            {"gpu_name": "Test"},
            FusionGroup(id="fg_0", pattern_name="test", members=["a"]),
            [MemberOpContext(node_id="a", op_type="relu")],
            previous_error=error
        )

        assert "Previous Attempt Failed" in prompt
        assert "missing_function" in prompt


class TestPromptIncludesTensorShapes:
    """Prompts must include tensor shapes for kernel sizing."""

    def test_includes_input_output_shapes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cuda_fusion",
            Path(__file__).resolve().parents[1] / "src" / "optimizer" / "backends" / "cuda" / "fusion.py"
        )
        cuda_fusion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cuda_fusion)

        contexts = [
            MemberOpContext(
                node_id="conv_1",
                op_type="conv2d",
                tensor_shapes={"input_0": [16, 64, 56, 56], "output": [16, 128, 28, 28]}
            )
        ]

        prompt = cuda_fusion.generate_fusion_prompt(
            {"gpu_name": "Test"},
            FusionGroup(id="fg_0", pattern_name="test", members=["conv_1"]),
            contexts
        )

        assert "[16, 64, 56, 56]" in prompt
        assert "conv2d" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
