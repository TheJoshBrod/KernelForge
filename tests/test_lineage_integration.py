"""Integration test: drive the optimizer's generate() with fakes and verify
that history_{node_id}.json is written with the expected chain."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeBackend:
    kernel_extension = ".cu"

    def get_sys_prompt(self):
        return "FAKE_SYS_PROMPT"

    def generate_optimization_prompt(self, *a, **kw):
        return "INITIAL_PROMPT"

    def validate_kernel(self, code, paths, ssh_config=None):
        # Deterministic: first attempt compile-fail, second succeeds.
        attempt = paths.get("attempt", 0)
        if attempt == 0:
            return False, "[Compilation Failed] expected ';' at line 5"
        return True, ""


class FakeLLM:
    """Stands in for GenModel. .chat() returns canned kernel source."""

    def __init__(self, sys_prompt):
        self.sys_prompt = sys_prompt
        self.calls = []

    def chat(self, msg, model):
        self.calls.append(msg)
        # Return a valid [START kernel.cu] block so extract_feedback_and_code works
        attempt_idx = len(self.calls) - 1
        return (
            "// [START FEEDBACK]\n"
            f"// attempt {attempt_idx} notes\n"
            "// [END FEEDBACK]\n"
            "// [START kernel.cu]\n"
            f"__global__ void k{attempt_idx}() {{}}\n"
            "// [END kernel.cu]\n"
        )


def test_optimizer_generate_writes_history(tmp_path):
    from src.optimizer.core import generator as gen_mod

    proj_dir = tmp_path / "testop"
    (proj_dir / "kernels").mkdir(parents=True)

    paths = {
        "proj_dir": proj_dir,
        "iteration": 0,
    }

    backend = FakeBackend()

    # Patch GenModel constructor used inside generate() and get_next_node_id
    with patch.object(gen_mod, "GenModel", FakeLLM), \
         patch.object(gen_mod.mcts, "get_next_node_id", return_value=42), \
         patch.object(gen_mod, "ensure_llm_config", return_value="openai"):
        feedback, ok, err, attempts = gen_mod.generate(
            backend=backend,
            best_kernel_code="// dummy\n",
            gpu_specs=None,
            improvement_log=[],
            paths=paths,
            model="fake-model",
        )

    assert ok, f"generate should succeed; err={err}"
    assert attempts == 2  # initial failed + one retry succeeded

    history_path = proj_dir / "kernels" / "history_42.json"
    assert history_path.exists()

    data = json.loads(history_path.read_text())
    assert data["node_id"] == 42
    assert data["phase"] == "optimizer"
    assert data["op_name"] == "testop"
    assert data["final_outcome"] == "success"
    assert data["attempts_to_correct"] == 2
    assert data["system_prompt"] == "FAKE_SYS_PROMPT"

    chain = data["chain"]
    assert len(chain) == 2

    # iter 0 = initial prompt, failed with compilation error
    assert chain[0]["role"] == "initial"
    assert chain[0]["prompt"] == "INITIAL_PROMPT"
    assert chain[0]["is_valid"] is False
    assert chain[0]["error_type"] == "CompilationFailed"
    assert "[Compilation Failed]" in chain[0]["error_details"]
    assert "k0" in chain[0]["llm_response_code"]

    # iter 1 = fix prompt; optimizer path sends the raw error back verbatim
    assert chain[1]["role"] == "fix"
    assert chain[1]["prompt"] == chain[0]["error_details"]  # lineage linkage
    assert chain[1]["is_valid"] is True
    assert chain[1]["error_type"] is None
    assert "k1" in chain[1]["llm_response_code"]
