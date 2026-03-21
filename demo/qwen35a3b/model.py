from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

# For a live demo, point this at a pre-downloaded snapshot to avoid waiting on a
# 70+ GB Hub download during project creation. If the directory does not exist,
# the wrapper falls back to MODEL_ID and lets transformers download from HF.
LOCAL_MODEL_DIR = os.environ.get(
    "KFORGE_QWEN35_DIR", "/home/gb10/model-cache/Qwen3.5-35B-A3B"
)

DTYPE = torch.bfloat16
MAX_PROMPTS = 2
MAX_LENGTH = 1024

DEFAULT_PROMPTS = [
    "Summarize why mixture-of-experts models can improve serving efficiency.",
    (
        "Explain the tradeoffs between flash attention, eager attention, and "
        "kernel autotuning in one paragraph."
    ),
    (
        "You are presenting a systems paper. Draft a short abstract that "
        "motivates GPU kernel generation, validation, and search."
    ),
    (
        "List the dominant operators in a decoder-only transformer forward pass "
        "and explain which ones are memory bound versus compute bound."
    ),
    (
        "Write a concise benchmarking plan for inference latency. Cover warmup, "
        "batch size, prompt length, decode length, and baseline comparisons."
    ),
    (
        "Repeat the sentence 'Operator-level optimization can move the latency "
        "needle when the forward path is exercised with representative data.' "
        "forty times."
    ),
    (
        "Repeat the sentence 'Prefill and decode stress different kernels and "
        "must both be measured.' one hundred times."
    ),
    (
        "Repeat the sentence 'A convincing ASPLOS demo needs deterministic "
        "inputs, cached weights, and pre-selected hot operators.' two hundred "
        "times."
    ),
]


def _local_snapshot_ready(local: Path) -> bool:
    manifest = local / "model.safetensors.index.json"
    required = [
        local / "config.json",
        local / "tokenizer.json",
        manifest,
    ]
    if not all(path.exists() for path in required):
        return False

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False

    weight_map = payload["weight_map"] if "weight_map" in payload else {}
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    shard_names = {str(name) for name in weight_map.values() if name}
    if not shard_names:
        return False

    return all((local / shard_name).exists() for shard_name in shard_names)


def _model_source() -> str:
    local = Path(LOCAL_MODEL_DIR).expanduser()
    if _local_snapshot_ready(local):
        return str(local)
    return MODEL_ID


def _build_from_pretrained() -> Qwen3_5MoeForConditionalGeneration:
    source = _model_source()
    return Qwen3_5MoeForConditionalGeneration.from_pretrained(
        source,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )


def build_model() -> Qwen3_5MoeForConditionalGeneration:
    return _build_from_pretrained()


def load_weights(weights_path: str, device: str = "cpu") -> Qwen3_5MoeForConditionalGeneration:
    # The official checkpoint is distributed as sharded safetensors, not a
    # single .pt/.pth file. For this demo wrapper we always load from the local
    # snapshot or Hub ID and ignore the uploaded file path if one exists.
    _ = weights_path
    _ = device
    return _build_from_pretrained()


def _read_prompts(validation_path: str | None) -> list[str]:
    if not validation_path:
        return list(DEFAULT_PROMPTS)

    root = Path(validation_path)
    candidates = [
        root / "prompts.jsonl",
        root / "texts.jsonl",
        root / "prompts.txt",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".txt":
            items = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines()]
            prompts = [item for item in items if item]
            if prompts:
                return prompts
            continue

        prompts: list[str] = []
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row["text"]).strip() if "text" in row and row["text"] else ""
            if text:
                prompts.append(text)
        if prompts:
            return prompts

    return list(DEFAULT_PROMPTS)


def _tokenizer():
    tok = AutoTokenizer.from_pretrained(_model_source(), trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_validation_dataloader(validation_path: str | None = None):
    tok = _tokenizer()
    prompts = _read_prompts(validation_path)[:MAX_PROMPTS]
    batches = []
    for prompt in prompts:
        encoded = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        batch = {key: value for key, value in encoded.items()}
        batch["use_cache"] = False
        batches.append(batch)
    return batches


def sample_inputs():
    return get_validation_dataloader(None)
