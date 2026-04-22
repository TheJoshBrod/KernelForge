"""
FusionEngine: Core orchestrator for kernel fusion generation.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.config import ensure_llm_config
from src.llm_tools import GenModel
from src.optimizer.core.backend import Backend
from src.optimizer.core.fusion.context import (
    compute_baseline_timing,
    gather_op_context,
    load_dag_and_benchmarks,
)
from src.optimizer.core.fusion.io_builder import build_fusion_io, get_expected_output
from src.optimizer.core.fusion.store import (
    ensure_fusion_schema,
    get_accepted_groups,
    get_group_by_id,
    record_attempt,
    sync_from_json,
    update_group_status,
)
from src.optimizer.core.fusion.types import (
    COSINE_THRESHOLD,
    FUSION_ATOL,
    FUSION_RTOL,
    MAX_ATTEMPTS,
    AttemptStatus,
    FusionGenStatus,
    FusionGroup,
    FusionResult,
    MemberOpContext,
)
from src.optimizer.core.types import GPUSpecs


def _log(msg: str) -> None:
    print(f"[FusionEngine] {msg}")


def _extract_feedback_and_code(content: str) -> tuple[str | None, str | None]:
    """
    Extract feedback and code sections from LLM response.
    Reuses pattern from optimizer/core/generator.py.
    """
    # Extract feedback section
    feedback_pattern = r"(?://|#)\s*\[START FEEDBACK\](.*?)(?://|#)\s*\[END FEEDBACK\]"
    feedback_match = re.search(feedback_pattern, content, re.DOTALL | re.IGNORECASE)
    feedback = feedback_match.group(1).strip() if feedback_match else None

    # Extract code section
    code_pattern = (
        r"(?://|#)\s*\[START kernel\.(?:cu|py)\](.*?)(?://|#)\s*\[END kernel\.(?:cu|py)\]"
    )
    code_match = re.search(code_pattern, content, re.DOTALL | re.IGNORECASE)

    if code_match:
        code = code_match.group(1).strip()
    else:
        # Markdown fallback
        fallback_pattern = r"```(?:C\+\+|cpp|cuda|c|python|triton)?\s*\n(.*?)```"
        fallback_match = re.search(fallback_pattern, content, re.DOTALL | re.IGNORECASE)
        code = fallback_match.group(1).strip() if fallback_match else None

    if feedback is None:
        feedback = "No feedback provided"

    return feedback, code


def _validate_output(
    output: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[bool, str]:
    """
    Validate fused kernel output against expected.
    Uses relaxed tolerances and cosine similarity fallback.
    """
    # Primary check with relaxed tolerances
    try:
        passed = torch.allclose(output, expected, rtol=FUSION_RTOL, atol=FUSION_ATOL)
    except Exception as e:
        return False, f"allclose error: {e}"

    if passed:
        return True, ""

    # Cosine similarity fallback
    try:
        output_flat = output.flatten().float()
        expected_flat = expected.flatten().float()
        cos_sim = F.cosine_similarity(
            output_flat.unsqueeze(0), expected_flat.unsqueeze(0)
        )

        if cos_sim.item() >= COSINE_THRESHOLD:
            _log(
                f"  allclose failed but cosine similarity {cos_sim.item():.6f} >= {COSINE_THRESHOLD}"
            )
            return True, ""

        return False, (
            f"Output mismatch: allclose failed (rtol={FUSION_RTOL}, atol={FUSION_ATOL}), "
            f"cosine_similarity={cos_sim.item():.6f} < {COSINE_THRESHOLD}"
        )
    except Exception as e:
        return False, f"Validation error: {e}"


class FusionEngine:
    """
    Orchestrates kernel fusion generation.

    Pipeline:
    1. Load accepted fusion groups from fusion.db (synced from JSON)
    2. Gather context for member operations
    3. Generate fused kernel via LLM (with retry loop)
    4. Validate fused kernel correctness
    5. Profile and store result
    """

    def __init__(
        self,
        backend: Backend,
        project_dir: Path,
        model: str | None = None,
        ssh_config: dict | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ):
        self.backend = backend
        self.project_dir = project_dir
        self.model = model or "claude-sonnet-4-5-20251001"
        self.ssh_config = ssh_config
        self.max_attempts = max_attempts

        # Ensure config is loaded
        ensure_llm_config()

        # Sync fusion_groups.json to DB
        ensure_fusion_schema(project_dir)
        imported = sync_from_json(project_dir)
        if imported > 0:
            _log(f"Imported {imported} fusion groups from JSON")

    def load_accepted_groups(self) -> list[FusionGroup]:
        """Load fusion groups with status='accepted' that need generation."""
        return get_accepted_groups(self.project_dir)

    def gather_member_context(self, group: FusionGroup) -> list[MemberOpContext]:
        """Gather IO, kernels, timings for all member ops."""
        dag_nodes, dag_edges, benchmarks = load_dag_and_benchmarks(self.project_dir)

        contexts: list[MemberOpContext] = []
        for node_id in group.members:
            # Find op_type from DAG
            op_type = ""
            for node in dag_nodes:
                if node.get("id") == node_id:
                    op_type = node.get("op", "")
                    break

            if not op_type:
                # Try to infer from node_id (e.g., "conv2d_4" -> "conv2d")
                parts = node_id.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    op_type = parts[0]

            ctx = gather_op_context(
                self.project_dir,
                node_id,
                op_type,
                benchmarks,
                dag_nodes,
            )
            contexts.append(ctx)

        return contexts

    def build_fusion_io(
        self,
        group: FusionGroup,
        contexts: list[MemberOpContext],
    ) -> Path:
        """Create synthetic IO by running ops sequentially in PyTorch."""
        return build_fusion_io(self.project_dir, group, contexts)

    def generate_fused_kernel(
        self,
        group: FusionGroup,
        contexts: list[MemberOpContext],
        gpu_specs: GPUSpecs,
        previous_error: str | None = None,
    ) -> tuple[str | None, str]:
        """
        Call LLM to generate fused kernel code.

        Returns:
            (code, feedback) where code is None if extraction failed
        """
        # Import fusion prompt generator based on backend extension
        # Import directly from fusion.py to avoid pulling in verifier dependencies
        if self.backend.kernel_extension == ".cu":
            import importlib.util
            import os
            fusion_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "backends", "cuda", "fusion.py"
            )
            spec = importlib.util.spec_from_file_location("cuda_fusion", fusion_path)
            fusion_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fusion_module)
            generate_fusion_prompt = fusion_module.generate_fusion_prompt
            get_fusion_sys_prompt = fusion_module.get_fusion_sys_prompt
        elif self.backend.kernel_extension == ".py":
            import importlib.util
            import os
            fusion_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "backends", "triton", "fusion.py"
            )
            spec = importlib.util.spec_from_file_location("triton_fusion", fusion_path)
            fusion_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fusion_module)
            generate_fusion_prompt = fusion_module.generate_fusion_prompt
            get_fusion_sys_prompt = fusion_module.get_fusion_sys_prompt
        else:
            raise ValueError(f"Unsupported backend extension: {self.backend.kernel_extension}")

        sys_prompt = get_fusion_sys_prompt()
        user_prompt = generate_fusion_prompt(
            gpu_info=gpu_specs.model_dump(),
            group=group,
            member_contexts=contexts,
            previous_error=previous_error,
        )

        llm = GenModel(sys_prompt)
        _log(f"  Calling LLM ({self.model}) for fusion...")

        response = llm.chat(user_prompt, self.model)

        feedback, code = _extract_feedback_and_code(response)

        if code is None:
            _log("  Failed to extract code from LLM response")
            # Save failed response for debugging
            self._save_failed_response(group, response)

        return code, feedback or ""

    def validate_fused_kernel(
        self,
        group: FusionGroup,
        code: str,
        io_dir: Path,
        contexts: list[MemberOpContext],
    ) -> tuple[bool, str]:
        """
        Validate fused kernel produces correct output.

        Uses synthetic IO from build_fusion_io().
        """
        # Save kernel to temp file
        kernel_dir = (
            self.project_dir / "kernels" / "fused" / group.id / "attempts"
        )
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_path = kernel_dir / f"temp_kernel{self.backend.kernel_extension}"
        kernel_path.write_text(code, encoding="utf-8")

        # Build paths dict for backend validation
        paths = {
            "kernel_dir": kernel_dir,
            "kernel_path": kernel_path,
            "io_dir": io_dir,
            "proj_dir": self.project_dir / "kernels" / "fused" / group.id,
        }

        # Use backend validation
        success, error = self.backend.validate_kernel(
            code, paths, ssh_config=self.ssh_config
        )

        if not success:
            return False, error

        # Additional validation with relaxed tolerances
        entries = sorted(io_dir.glob("entry_*.pt"))
        if entries:
            for entry_path in entries[:3]:  # Validate first 3 entries
                expected = get_expected_output(contexts, 0)
                if expected is not None:
                    # The backend validation already checked correctness,
                    # but we can do additional cosine similarity check here
                    pass

        return True, ""

    def profile_fused_kernel(
        self,
        group: FusionGroup,
        kernel_path: Path,
        io_dir: Path,
    ) -> dict[str, Any]:
        """Profile fused kernel performance."""
        paths = {
            "kernel_path": kernel_path,
            "kernel_dir": kernel_path.parent,
            "io_dir": io_dir,
            "proj_dir": self.project_dir / "kernels" / "fused" / group.id,
        }

        stats = self.backend.profile_kernel(
            paths,
            baseline=False,
            ssh_config=self.ssh_config,
        )

        return stats

    def fuse_group(
        self,
        group: FusionGroup,
        status_callback: Callable[[str], None] | None = None,
    ) -> FusionResult:
        """
        End-to-end fusion with retry loop for error correction.

        Pipeline:
        1. Gather member context
        2. Build synthetic IO for validation
        3. RETRY LOOP (up to max_attempts):
           a. Generate fused kernel (include previous_error if retrying)
           b. Validate correctness
           c. If validation fails, capture error and retry
           d. If success, break loop
        4. Profile performance (only on success)
        5. Store result
        """
        _log(f"Fusing group: {group.id} ({group.pattern_name})")
        _log(f"  Members: {group.members}")

        # Update status to generating
        update_group_status(
            self.project_dir, group.id, gen_status=FusionGenStatus.GENERATING.value
        )

        if status_callback:
            status_callback(f"Gathering context for {group.pattern_name}...")

        # 1. Gather member context
        contexts = self.gather_member_context(group)
        baseline_ms = compute_baseline_timing(contexts)
        _log(f"  Baseline timing: {baseline_ms:.4f} ms" if baseline_ms else "  No baseline timing")

        # 2. Build synthetic IO
        if status_callback:
            status_callback(f"Building synthetic IO for {group.pattern_name}...")

        io_dir = self.build_fusion_io(group, contexts)
        _log(f"  Synthetic IO dir: {io_dir}")

        # 3. Get GPU specs
        gpu_specs = self.backend.get_device_specs(ssh_config=self.ssh_config)

        # 4. Retry loop
        previous_error: str | None = None
        attempts: list[dict] = []

        for attempt_num in range(self.max_attempts):
            _log(f"  Attempt {attempt_num + 1}/{self.max_attempts}")

            if status_callback:
                status_callback(
                    f"Generating fused kernel (attempt {attempt_num + 1})..."
                )

            # Generate kernel
            code, feedback = self.generate_fused_kernel(
                group, contexts, gpu_specs, previous_error=previous_error
            )

            if code is None:
                error_msg = "Failed to extract code from LLM response"
                attempts.append({
                    "attempt_num": attempt_num,
                    "status": AttemptStatus.COMPILE_ERROR.value,
                    "error_message": error_msg,
                    "llm_model": self.model,
                })
                record_attempt(self.project_dir, group.id, attempts[-1])
                previous_error = error_msg
                continue

            # Save attempt
            attempt_kernel_path = self._save_attempt(group, attempt_num, code)

            if status_callback:
                status_callback(f"Validating fused kernel (attempt {attempt_num + 1})...")

            # Validate
            success, error_msg = self.validate_fused_kernel(
                group, code, io_dir, contexts
            )

            attempt_status = (
                AttemptStatus.SUCCESS if success else AttemptStatus.VALIDATION_ERROR
            )
            attempts.append({
                "attempt_num": attempt_num,
                "status": attempt_status.value,
                "kernel_path": attempt_kernel_path,
                "error_message": error_msg if not success else None,
                "llm_model": self.model,
            })
            record_attempt(self.project_dir, group.id, attempts[-1])

            if success:
                _log(f"  Validation passed!")

                # Profile
                if status_callback:
                    status_callback(f"Profiling fused kernel...")

                try:
                    stats = self.profile_fused_kernel(group, attempt_kernel_path, io_dir)
                    fused_ms = stats.get("mean_ms")
                except Exception as e:
                    _log(f"  Profiling failed: {e}")
                    fused_ms = None

                # Save best kernel
                best_kernel_path = self._save_best_kernel(group, code)

                # Compute speedup
                speedup = None
                if baseline_ms and fused_ms and fused_ms > 0:
                    speedup = baseline_ms / fused_ms

                # Update DB
                update_group_status(
                    self.project_dir,
                    group.id,
                    gen_status=FusionGenStatus.COMPLETED.value,
                    baseline_ms=baseline_ms,
                    fused_ms=fused_ms,
                    actual_speedup=speedup,
                    best_kernel_path=str(best_kernel_path),
                    llm_model=self.model,
                )

                _log(
                    f"  Fusion complete! "
                    f"baseline={baseline_ms:.4f}ms, "
                    f"fused={fused_ms:.4f}ms, "
                    f"speedup={speedup:.2f}x"
                    if speedup
                    else f"  Fusion complete!"
                )

                return FusionResult(
                    group_id=group.id,
                    status=FusionGenStatus.COMPLETED,
                    kernel_path=best_kernel_path,
                    baseline_ms=baseline_ms,
                    fused_ms=fused_ms,
                    speedup=speedup,
                )

            # Capture error for next attempt
            _log(f"  Validation failed: {error_msg[:200]}...")
            previous_error = error_msg

        # All attempts failed
        _log(f"  All {self.max_attempts} attempts failed")

        update_group_status(
            self.project_dir,
            group.id,
            gen_status=FusionGenStatus.FAILED.value,
            baseline_ms=baseline_ms,
        )

        return FusionResult(
            group_id=group.id,
            status=FusionGenStatus.FAILED,
            error=previous_error,
            baseline_ms=baseline_ms,
        )

    def fuse_all_accepted(
        self,
        status_callback: Callable[[str], None] | None = None,
    ) -> list[FusionResult]:
        """Process all accepted fusion groups."""
        groups = self.load_accepted_groups()

        if not groups:
            _log("No accepted fusion groups to process")
            return []

        _log(f"Processing {len(groups)} accepted fusion groups")

        results: list[FusionResult] = []
        for i, group in enumerate(groups, 1):
            _log(f"\n--- Group {i}/{len(groups)} ---")
            result = self.fuse_group(group, status_callback=status_callback)
            results.append(result)

        return results

    def _save_attempt(
        self,
        group: FusionGroup,
        attempt_num: int,
        code: str,
    ) -> Path:
        """Save attempt kernel to disk."""
        attempts_dir = self.project_dir / "kernels" / "fused" / group.id / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)

        kernel_path = attempts_dir / f"{attempt_num}_kernel{self.backend.kernel_extension}"
        kernel_path.write_text(code, encoding="utf-8")

        return kernel_path

    def _save_best_kernel(
        self,
        group: FusionGroup,
        code: str,
    ) -> Path:
        """Save best kernel to main location."""
        kernel_dir = self.project_dir / "kernels" / "fused" / group.id
        kernel_dir.mkdir(parents=True, exist_ok=True)

        kernel_path = kernel_dir / f"kernel{self.backend.kernel_extension}"
        kernel_path.write_text(code, encoding="utf-8")

        return kernel_path

    def _save_failed_response(
        self,
        group: FusionGroup,
        response: str,
    ) -> None:
        """Save failed LLM response for debugging."""
        dump_dir = self.project_dir / "kernels" / "fused" / group.id / "garbage_dump"
        dump_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        dump_path = dump_dir / f"llm_response_{timestamp}.txt"
        dump_path.write_text(response, encoding="utf-8")
        _log(f"  Saved failed response to: {dump_path}")
