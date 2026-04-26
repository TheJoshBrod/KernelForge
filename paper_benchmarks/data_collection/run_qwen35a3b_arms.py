from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
PROJECTS_ROOT = REPO_ROOT / "kernels" / "projects"
CATALOG_DB = PROJECTS_ROOT / "catalog.db"
VALIDATION_ZIP = (
    WORKSPACE_ROOT
    / "datacollection"
    / "forge_inputs"
    / "validation_hf_HuggingFaceH4_ultrachat_200k_forge_v1.zip"
)
BENCH_CAST_ROOT = WORKSPACE_ROOT / "datacollection" / "benchmarking" / "cast_exports" / "qwen35a3b"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
LOCAL_MODEL_DIR = "/home/gb10/model-cache/Qwen3.5-35B-A3B"
MODEL_SLUG = "qwen35a3b"
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-opus-4-7"
TARGET_DEVICE = "cuda"
BUDGET_TARGETS = {
    "zeroshot": 0,
    "opt05": 5,
    "opt10": 10,
    "opt20": 20,
    "opt50": 50,
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_benchmarks.data_collection.collect_zero_shot import (  # noqa: E402
    arm_usage_filter,
    llm_usage_summary,
    profiled_ops,
    selected_tree_kernel_map,
    successful_zero_shot_kernel_map,
)
from paper_benchmarks.paper_bench.cast_export import (  # noqa: E402
    export_cast_package,
    inspect_cast_package,
)
from paper_benchmarks.paper_bench.provenance import safe_sha256_path  # noqa: E402
from src.llm.key_store import load_config  # noqa: E402
from src.llm.runtime_config import resolve_runtime_env  # noqa: E402


MODEL_TEMPLATE = r'''
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
@@QUANT_IMPORT@@

@@INT4_GROUPED_MM_PATCH@@

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
LOCAL_MODEL_DIR = os.environ.get(
    "KFORGE_QWEN35_DIR",
    "/home/gb10/model-cache/Qwen3.5-35B-A3B",
)
OFFLOAD_DIR = os.environ.get(
    "KFORGE_QWEN35_OFFLOAD_DIR",
    "@@OFFLOAD_DIR@@",
)

DTYPE = torch.bfloat16
MAX_PROMPTS = int(os.environ.get("KFORGE_VALIDATION_MAX_PROMPTS", "16"))
PROMPT_OFFSET = int(os.environ.get("KFORGE_VALIDATION_PROMPT_OFFSET", "0"))
MAX_LENGTH = int(os.environ.get("KFORGE_VALIDATION_MAX_LENGTH", "2048"))

DEFAULT_PROMPTS = [
    "Summarize why mixture-of-experts models can improve serving efficiency.",
    "Explain why prefill and decode should be benchmarked separately for LLM inference.",
    (
        "Draft a concise systems-paper paragraph about profiling transformer "
        "operators, generating GPU kernels, validating correctness, and selecting "
        "the fastest safe deployment path."
    ),
    (
        "Describe how expert routing, shared experts, routed experts, token "
        "dispatch, and gather operations affect the runtime behavior of a sparse "
        "mixture-of-experts decoder model."
    ),
    (
        "Repeat the sentence 'Representative validation prompts are required "
        "because operator shapes depend on the executed forward path.' forty times."
    ),
    (
        "Repeat the sentence 'KV cache, prompt length, batch size, and backend "
        "choice all affect latency and memory on GB10.' sixty times."
    ),
]


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _local_snapshot_ready(local: Path) -> bool:
    if not local.exists():
        return False
    if not (local / "config.json").exists():
        return False
    if not ((local / "tokenizer.json").exists() or (local / "tokenizer.model").exists()):
        return False

    manifest = local / "model.safetensors.index.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = payload.get("weight_map", {})
        shard_names = {str(name) for name in weight_map.values() if name}
        return bool(shard_names) and all((local / shard).exists() for shard in shard_names)

    return any(local.glob("*.safetensors")) or (local / "pytorch_model.bin").exists()


def _model_source() -> str:
    local = Path(LOCAL_MODEL_DIR).expanduser()
    if _local_snapshot_ready(local):
        return str(local)
    if not _truthy_env("KFORGE_ALLOW_HF_DOWNLOAD", "0"):
        raise FileNotFoundError(
            f"Qwen snapshot is not ready at {local}. Set KFORGE_QWEN35_DIR "
            "or KFORGE_ALLOW_HF_DOWNLOAD=1 to permit a Hugging Face download."
        )
    return MODEL_ID


def _target_device(default: str = "cuda") -> str:
    return os.environ.get("KFORGE_TARGET_DEVICE", default).strip().lower()


def _text_layer_count() -> int:
    config_path = Path(LOCAL_MODEL_DIR).expanduser() / "config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            text_config = payload.get("text_config")
            if isinstance(text_config, dict):
                return int(text_config.get("num_hidden_layers") or 40)
        except Exception:
            pass
    return int(os.environ.get("KFORGE_QWEN35_TEXT_LAYERS", "40"))


def _explicit_disk_device_map() -> dict:
    execution_device = 0
    device_map = {
        "model.embed_tokens": execution_device,
        "model.rotary_emb": execution_device,
        "model.norm": execution_device,
        "lm_head": execution_device,
    }
    for layer_index in range(_text_layer_count()):
        device_map[f"model.layers.{layer_index}"] = "disk"
    return device_map


def _from_pretrained_kwargs(device: str | None = None) -> dict:
    target_device = (device or _target_device()).strip().lower()
    kwargs = {
        "torch_dtype": DTYPE,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "trust_remote_code": _truthy_env("KFORGE_TRUST_REMOTE_CODE", "0"),
@@EXPERTS_KWARG@@
@@QUANT_KWARG@@
    }
    if target_device in {"", "cuda", "gpu"} and torch.cuda.is_available():
        device_map = os.environ.get("KFORGE_DEVICE_MAP", "auto").strip() or "auto"
        if device_map.lower() in {"none", "unset", "no_device_map"}:
            return kwargs
        if device_map.lower() in {"explicit_disk", "qwen_explicit_disk", "qwen35_explicit_disk"}:
            kwargs["device_map"] = _explicit_disk_device_map()
        elif device_map.lower() in {"cuda", "cuda:0", "single_cuda"}:
            kwargs["device_map"] = {"": "cuda:0"}
        else:
            max_memory_raw = os.environ.get("KFORGE_QWEN35_MAX_MEMORY", "").strip()
            if max_memory_raw:
                parsed = json.loads(max_memory_raw)
                kwargs["max_memory"] = {
                    int(key) if str(key).isdigit() else key: value
                    for key, value in parsed.items()
                }
            else:
                kwargs["max_memory"] = {
                    0: os.environ.get("KFORGE_QWEN35_GPU_MAX_MEMORY", "32GiB"),
                    "cpu": os.environ.get("KFORGE_QWEN35_CPU_MAX_MEMORY", "12GiB"),
                    "disk": os.environ.get("KFORGE_QWEN35_DISK_MAX_MEMORY", "200GiB"),
                }
            kwargs["device_map"] = device_map
        kwargs["offload_state_dict"] = True
        kwargs["offload_folder"] = OFFLOAD_DIR
        kwargs["offload_buffers"] = _truthy_env("KFORGE_QWEN35_OFFLOAD_BUFFERS", "1")
        Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)
    return kwargs


def _build_from_pretrained(device: str | None = None):
    source = _model_source()
    kwargs = _from_pretrained_kwargs(device)
    try:
        return AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" not in str(exc):
            raise
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(source, **kwargs)


def build_model():
    return _build_from_pretrained()


def load_weights(weights_path: str, device: str = "cpu"):
    _ = weights_path
    return _build_from_pretrained(device)


def _read_prompt_rows(validation_path: str | None) -> list[dict[str, str]]:
    if not validation_path:
        return [{"text": prompt, "bucket": "default"} for prompt in DEFAULT_PROMPTS]

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
            rows = []
            for line in candidate.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if text:
                    rows.append({"text": text, "bucket": "txt"})
            if rows:
                return rows
            continue

        rows = []
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if text:
                rows.append({"text": text, "bucket": str(row.get("bucket") or "")})
        if rows:
            return rows

    return [{"text": prompt, "bucket": "default"} for prompt in DEFAULT_PROMPTS]


def _selected_prompts(validation_path: str | None) -> list[str]:
    rows = _read_prompt_rows(validation_path)
    bucket_filter_raw = os.environ.get("KFORGE_VALIDATION_BUCKETS", "").strip()
    if bucket_filter_raw:
        allowed = {bucket.strip() for bucket in bucket_filter_raw.split(",") if bucket.strip()}
        rows = [row for row in rows if row.get("bucket", "") in allowed]

    offset = max(0, PROMPT_OFFSET)
    if _truthy_env("KFORGE_VALIDATION_STRATIFIED", "1"):
        bucket_order = []
        by_bucket: dict[str, list[dict[str, str]]] = {}
        for row in rows[offset:]:
            bucket = row.get("bucket", "") or "unknown"
            if bucket not in by_bucket:
                by_bucket[bucket] = []
                bucket_order.append(bucket)
            by_bucket[bucket].append(row)
        indexes = {bucket: 0 for bucket in bucket_order}
        selected: list[str] = []
        while len(selected) < MAX_PROMPTS:
            progressed = False
            for bucket in bucket_order:
                index = indexes[bucket]
                bucket_rows = by_bucket[bucket]
                if index >= len(bucket_rows):
                    continue
                selected.append(bucket_rows[index]["text"])
                indexes[bucket] = index + 1
                progressed = True
                if len(selected) >= MAX_PROMPTS:
                    break
            if not progressed:
                break
        return selected

    return [row["text"] for row in rows[offset : offset + MAX_PROMPTS]]


def _tokenizer():
    tok = AutoTokenizer.from_pretrained(
        _model_source(),
        trust_remote_code=_truthy_env("KFORGE_TRUST_REMOTE_CODE", "0"),
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_validation_dataloader(validation_path: str | None = None):
    tok = _tokenizer()
    prompts = _selected_prompts(validation_path)
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
'''.lstrip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def project_name_for_quant(quantization: str) -> str:
    return f"qwen35a3b-{quantization}-gb10"


def project_dir(project: str) -> Path:
    return PROJECTS_ROOT / project


def state_path(project: str) -> Path:
    return project_dir(project) / "state.json"


def logs_dir(project: str) -> Path:
    path = project_dir(project) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def validation_prompt_count() -> int:
    with zipfile.ZipFile(VALIDATION_ZIP, "r") as zf:
        with zf.open("prompts.jsonl") as handle:
            return sum(1 for line in handle if line.strip())


def validation_zip_b64(project: str) -> tuple[Path, Path]:
    uploads = project_dir(project) / ".uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    b64_path = uploads / "validation.b64"
    name_path = uploads / "validation.name"
    if not b64_path.exists():
        b64_path.write_text(base64.b64encode(VALIDATION_ZIP.read_bytes()).decode("ascii"), encoding="utf-8")
    if not name_path.exists():
        name_path.write_text(VALIDATION_ZIP.name, encoding="utf-8")
    return b64_path, name_path


def extract_validation(project: str) -> None:
    validation_dir = project_dir(project) / "data" / "validation"
    if (validation_dir / "prompts.jsonl").exists():
        return
    validation_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(VALIDATION_ZIP, "r") as zf:
        zf.extractall(validation_dir)


def gpu_info() -> dict[str, str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            name = parts[0] if parts else "NVIDIA GB10"
            uuid = parts[1] if len(parts) > 1 else ""
            return {
                "uuid": uuid,
                "name": name,
                "connection": "local",
                "identity": uuid or f"{name}:local",
            }
    except Exception:
        pass
    return {
        "uuid": "",
        "name": "NVIDIA GB10",
        "connection": "local",
        "identity": "NVIDIA GB10:local",
    }


def model_py_text(quantization: str) -> str:
    if quantization == "int4":
        quant_import = "from transformers import QuantoConfig"
        quant_kwarg = '        "quantization_config": QuantoConfig(weights="int4", modules_to_not_convert=["shared_expert_gate"]),'
        experts_kwarg = '        "experts_implementation": os.environ.get("KFORGE_QWEN35_EXPERTS_IMPL", "grouped_mm"),'
        int4_grouped_mm_patch = r'''
def _install_int4_grouped_mm_alignment_patch() -> None:
    import transformers.integrations.moe as moe

    if getattr(moe, "_kforge_int4_grouped_mm_alignment_patch", False):
        return

    def _aligned_grouped_mm(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        if hasattr(weight, "dequantize"):
            weight = weight.dequantize()
        weight = weight.to(device=input.device, dtype=input.dtype)
        input = input.to(dtype=weight.dtype)
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if offs.device != input.device or offs.dtype != torch.int32 or not offs.is_contiguous():
            offs = offs.to(device=input.device, dtype=torch.int32).contiguous()
        return torch.nn.functional.grouped_mm(input, weight, offs=offs)

    moe._grouped_mm = _aligned_grouped_mm
    moe._kforge_int4_grouped_mm_alignment_patch = True


_install_int4_grouped_mm_alignment_patch()
'''
    else:
        quant_import = ""
        quant_kwarg = ""
        experts_kwarg = ""
        int4_grouped_mm_patch = ""
    return (
        MODEL_TEMPLATE.replace("@@QUANT_IMPORT@@", quant_import)
        .replace("@@QUANT_KWARG@@", quant_kwarg)
        .replace("@@EXPERTS_KWARG@@", experts_kwarg)
        .replace("@@INT4_GROUPED_MM_PATCH@@", int4_grouped_mm_patch)
        .replace("@@OFFLOAD_DIR@@", f"/tmp/kforge_qwen35a3b_{quantization}_offload")
    )


def config_payload(project: str, quantization: str) -> dict[str, Any]:
    int4 = quantization == "int4"
    profile_cfg: dict[str, Any] = {
        "allow_ops": [],
        "max_batches": 1 if int4 else 16,
        "max_calls_per_op": 4 if int4 else 16,
        "max_tensor_elements": 200000000,
        "skip_ops": [],
        "skip_prefixes": [],
        "max_saved_entries_per_op": 4 if int4 else 5000,
        "capture_policy": "full_trace_up_to_cap",
    }
    metadata: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_slug": MODEL_SLUG,
        "quantization": quantization,
        "validation_prompt_count": validation_prompt_count(),
        "validation_zip": str(VALIDATION_ZIP),
        "dataset_role": "forge_validation_calibration",
        "target_device": "gb10",
        "profile_replay_entry_cap": 5000,
        "profile_capture_policy": "full_trace_up_to_cap",
        "profile_device_map": "none_cpu_profile" if int4 else "explicit_disk",
        "profile_gpu_max_memory": "32GiB",
        "profile_cpu_max_memory": "12GiB",
        "profile_disk_max_memory": "200GiB",
        "profile_offload": "disk_enabled",
        "profile_max_prompts": 1,
        "profile_prompt_offset": 4,
        "profile_max_length": 128 if int4 else 512,
        "profile_target_device": "cpu" if int4 else "cuda",
        "profile_experts_implementation": "grouped_mm",
        "profile_max_saved_entries_per_op": 4 if int4 else 5000,
        "profile_note": (
            "INT4 profiling uses experts_implementation=grouped_mm and captures "
            "torch.nn.functional.grouped_mm. CUDA full-model Quanto loading was "
            "attempted first but failed because on-the-fly Quanto forbids CPU/disk "
            "device maps and all-CUDA loading saturated GB10 memory. The project "
            "model installs a grouped_mm alignment shim that dequantizes and "
            "contiguizes grouped-mm operands before dispatching "
            "torch.nn.functional.grouped_mm; grouped-mm shape capture is therefore "
            "performed on CPU, while Forge generation/optimization targets CUDA."
            if int4
            else "BF16 profiling uses CUDA with explicit disk offload."
        ),
    }
    if int4:
        metadata.update(
            {
                "quantization_backend": "optimum-quanto",
                "quantization_config": "QuantoConfig(weights=int4, modules_to_not_convert=[shared_expert_gate])",
                "weight_bits": 4,
                "activation_dtype": "bfloat16",
            }
        )
    return {
        "artifacts": {"weights": []},
        "backend": "cuda",
        "base_name": project,
        "benchmarked_gpu": gpu_info(),
        "created_at": utc_now(),
        "generator": {
            "extra_validation_cases": 1,
            "max_attempts": 8,
            "max_ops": 0,
            "only_ops": [],
            "skip_ops": [],
            "target_device": "cuda",
            "use_baseline_as_template": False,
            "use_baseline_kernels": False,
        },
        "llm": {
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
        },
        "paper_metadata": metadata,
        "profile": profile_cfg,
        "validation_dir": "data/validation",
    }


def ensure_project(project: str, quantization: str) -> None:
    root = project_dir(project)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_text(model_py_text(quantization), encoding="utf-8")
    write_json(root / "config.json", config_payload(project, quantization))
    validation_zip_b64(project)
    extract_validation(project)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "exports").mkdir(parents=True, exist_ok=True)
    update_catalog(project)


def update_catalog(project: str) -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    with sqlite3.connect(CATALOG_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                name TEXT PRIMARY KEY,
                project_path TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_opened_at TEXT
            )
            """
        )
        existing = conn.execute(
            "SELECT created_at, last_opened_at FROM projects WHERE name = ?",
            (project,),
        ).fetchone()
        created_at = str(existing[0]) if existing and existing[0] else now
        last_opened_at = str(existing[1]) if existing and len(existing) > 1 and existing[1] else None
        conn.execute(
            """
            INSERT OR REPLACE INTO projects(name, project_path, created_at, updated_at, last_opened_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project, str(project_dir(project)), created_at, now, last_opened_at),
        )
        conn.commit()


def global_config() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "frontend" / "config.json"
    if cfg_path.exists():
        return load_config(cfg_path)
    return {}


def command_env(project: str, job_key: str) -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = str(Path(sys.executable).parent)
    if venv_bin and venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    project_cfg = read_json(project_dir(project) / "config.json", {})
    env.update(
        resolve_runtime_env(
            global_config=global_config(),
            project_config=project_cfg if isinstance(project_cfg, dict) else {},
            override_provider=LLM_PROVIDER,
            override_model=LLM_MODEL,
        )
    )
    env["KFORGE_STATE_PATH"] = str(state_path(project))
    env["KFORGE_JOB_KEY"] = job_key
    int4_profile = "-int4-" in project and job_key == "profile"
    env["KFORGE_TARGET_DEVICE"] = "cpu" if int4_profile else TARGET_DEVICE
    if "-int4-" in project:
        env.setdefault("KFORGE_DEVICE_MAP", "none" if int4_profile else "single_cuda")
        env.setdefault("KFORGE_QWEN35_EXPERTS_IMPL", "grouped_mm")
    else:
        env.setdefault("KFORGE_DEVICE_MAP", "explicit_disk")
    env.setdefault("KFORGE_QWEN35_GPU_MAX_MEMORY", "32GiB")
    env.setdefault("KFORGE_QWEN35_CPU_MAX_MEMORY", "12GiB")
    env.setdefault("KFORGE_QWEN35_DISK_MAX_MEMORY", "200GiB")
    env.setdefault("KFORGE_QWEN35_OFFLOAD_BUFFERS", "1")
    env["KFORGE_VALIDATION_MAX_PROMPTS"] = "1"
    env["KFORGE_VALIDATION_PROMPT_OFFSET"] = "4"
    env["KFORGE_VALIDATION_MAX_LENGTH"] = "128" if int4_profile else "512"
    env["KFORGE_VALIDATION_STRATIFIED"] = "1"
    env["KFORGE_PROFILE_MAX_PER_OP"] = "4" if int4_profile else "5000"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def update_state(project: str, job_key: str, updates: dict[str, Any]) -> None:
    path = state_path(project)
    state = read_json(path, {})
    if not isinstance(state, dict):
        state = {}
    job = dict(state.get(job_key, {}))
    job.update(updates)
    state[job_key] = job
    write_json(path, state)


def run_logged(project: str, job_key: str, log_name: str, cmd: list[str]) -> None:
    log_path = logs_dir(project) / log_name
    now = datetime.now().isoformat()
    phase = "profiling" if job_key == "profile" else "generating" if job_key == "generate" else "optimizing"
    update_state(
        project,
        job_key,
        {
            "status": "running",
            "control": "running",
            "active": True,
            "phase": phase,
            "message": f"Kernel {phase} started.",
            "progress_percent": 0.05,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "log": str(log_path),
            "command": cmd,
            "run_id": f"{project}__{job_key}__{int(time.time() * 1000)}",
        },
    )
    scoped_cmd = maybe_scope_command(cmd)
    print(f"[runner] {project} {job_key}: {' '.join(scoped_cmd)}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[job] Started at {now}\n")
        log.write(f"[job] Command: {scoped_cmd}\n")
        log.flush()
        proc = subprocess.Popen(
            scoped_cmd,
            cwd=str(REPO_ROOT),
            env=command_env(project, job_key),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        update_state(project, job_key, {"pid": proc.pid})
        assert proc.stdout is not None
        last_echo = 0.0
        for line in proc.stdout:
            log.write(line)
            log.flush()
            now_ts = time.time()
            important = (
                line.startswith("[workflow")
                or line.startswith("[benchmarking.")
                or line.startswith("[generator")
                or line.startswith("[optimize-result]")
                or line.startswith("[attempts-summary]")
                or line.startswith("[n-kernels")
                or line.startswith("Optimizing:")
                or "Saved profiling entries" in line
                or "Summary written" in line
                or "Kernel generation completed" in line
                or "Kernel optimization completed" in line
                or "Traceback" in line
                or "ERROR" in line
                or "Failed" in line
            )
            if important or now_ts - last_echo > 60:
                print(line[:600], end="" if line.endswith("\n") else "\n")
                last_echo = now_ts
        rc = proc.wait()

    finished = datetime.now().isoformat()
    if rc != 0:
        update_state(
            project,
            job_key,
            {
                "status": "error",
                "control": "idle",
                "active": False,
                "phase": "error",
                "message": f"Job failed with exit code {rc}.",
                "progress_percent": 1.0,
                "updated_at": finished,
                "finished_at": finished,
                "returncode": rc,
            },
        )
        raise RuntimeError(f"{project} {job_key} failed with exit code {rc}")

    update_state(
        project,
        job_key,
        {
            "status": "completed",
            "control": "idle",
            "active": False,
            "phase": "done",
            "message": f"Kernel {phase} completed.",
            "progress_percent": 1.0,
            "updated_at": finished,
            "finished_at": finished,
            "returncode": rc,
        },
    )


def maybe_scope_command(cmd: list[str]) -> list[str]:
    if os.environ.get("KFORGE_NO_SYSTEMD_SCOPE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return cmd
    if not shutil.which("systemd-run"):
        return cmd
    return ["systemd-run", "--user", "--scope", "--quiet", "--same-dir", *cmd]


def run_profile(project: str, force: bool) -> None:
    if not force and (project_dir(project) / "io" / "summary.json").exists():
        print(f"[runner] {project}: profile already exists.")
        return
    b64_path, name_path = validation_zip_b64(project)
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.workflow",
        "profile",
        "--project",
        project,
        "--validation-b64-path",
        str(b64_path),
        "--validation-name-path",
        str(name_path),
    ]
    run_logged(project, "profile", "profile.log", cmd)


def op_dirs(project: str) -> list[str]:
    root = project_dir(project) / "io" / "individual_ops"
    if not root.exists():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def generated_ops(project: str) -> set[str]:
    gen_root = project_dir(project) / "kernels" / "generated" / "individual_op_kernels"
    if not gen_root.exists():
        return set()
    out = set()
    for child in gen_root.iterdir():
        if not child.is_dir():
            continue
        if any((child / marker).exists() for marker in ("success.cuda", "success.triton", "success.mps", "success.cpu")):
            out.add(child.name)
    return out


def max_node_id(project: str, op_name: str) -> int:
    db_path = project_dir(project) / "trees" / op_name / "nodes.db"
    if not db_path.exists():
        return -1
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) FROM nodes").fetchone()
    if not row or row[0] is None:
        return -1
    return int(row[0])


def run_generate(project: str) -> None:
    ops = op_dirs(project)
    done = generated_ops(project)
    missing = [op for op in ops if op not in done]
    if not missing:
        print(f"[runner] {project}: zero-shot kernels already exist for all profiled ops.")
        return
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.workflow",
        "generate",
        "--project",
        project,
        "--ops",
        ",".join(missing),
        "--target-device",
        TARGET_DEVICE,
        "--benchmark",
        "--llm-model",
        LLM_MODEL,
        "--llm-provider",
        LLM_PROVIDER,
    ]
    run_logged(project, "generate", "generate.log", cmd)


def run_optimize_to(project: str, target: int, workers: int) -> None:
    candidates = sorted(generated_ops(project))
    missing = [op for op in candidates if max_node_id(project, op) < target]
    if not missing:
        print(f"[runner] {project}: optimization already reaches node {target} for generated ops.")
        return
    min_existing = min(max_node_id(project, op) for op in missing)
    incremental_iterations = max(1, target - min_existing)
    cmd = [
        sys.executable,
        "-m",
        "src.optimizer.workflow",
        "optimize",
        "--project",
        project,
        "--ops",
        ",".join(missing),
        "--iterations",
        str(incremental_iterations),
        "--workers",
        str(workers),
        "--llm-model",
        LLM_MODEL,
        "--llm-provider",
        LLM_PROVIDER,
    ]
    run_logged(project, "optimize", "optimize.log", cmd)
    still_missing = [op for op in candidates if max_node_id(project, op) < target]
    if still_missing:
        details = {op: max_node_id(project, op) for op in still_missing}
        raise RuntimeError(f"{project}: optimization did not reach target {target}: {details}")


def tree_node_rows(project: str, op_name: str) -> list[dict[str, Any]]:
    db_path = project_dir(project) / "trees" / op_name / "nodes.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def node_value_for_kernel(project: str, op_name: str, kernel_path: str) -> float | None:
    path = Path(kernel_path)
    node_id = None
    if path.stem.startswith("kernel_"):
        try:
            node_id = int(path.stem.split("_", 1)[1])
        except Exception:
            node_id = None
    for row in tree_node_rows(project, op_name):
        try:
            row_id = int(row.get("id"))
        except Exception:
            continue
        if node_id is not None and row_id != node_id:
            continue
        value = row.get("value")
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            return None
    return None


def pytorch_ms_by_op(project: str) -> dict[str, float]:
    payload = read_json(project_dir(project) / "benchmarks" / "op_benchmarks.json", {})
    out: dict[str, float] = {}
    for row in payload.get("results", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict) or not row.get("op"):
            continue
        try:
            out[str(row["op"])] = float(row.get("pytorch_ms") or 0.0)
        except Exception:
            out[str(row["op"])] = 0.0
    return out


def budget_target(budget: str) -> int:
    try:
        return BUDGET_TARGETS[budget]
    except KeyError as exc:
        raise ValueError(f"unsupported budget {budget!r}") from exc


def budget_arm(budget: str) -> str:
    target = budget_target(budget)
    return "zero_shot" if target == 0 else f"optimize_{target}"


def budget_selection(project: str, budget: str) -> tuple[dict[str, str], dict[str, Any]]:
    if budget == "zeroshot":
        selected = successful_zero_shot_kernel_map(project_dir(project))
        details = {
            op: {
                "selected": {
                    "node_id": 0,
                    "value_ms": node_value_for_kernel(project, op, kernel),
                    "kernel_path": kernel,
                }
            }
            for op, kernel in selected.items()
        }
        return selected, details
    selected, details = selected_tree_kernel_map(project_dir(project), arm=budget_arm(budget))
    return selected, details


def mixed_selection(
    project: str,
    budget: str,
    selected: dict[str, str],
    all_profiled_ops: list[str],
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    baselines = pytorch_ms_by_op(project)
    mixed: dict[str, str] = {}
    fallback_reasons: dict[str, Any] = {}
    fallback_ops: list[str] = []
    for op_name in all_profiled_ops:
        kernel_path = selected.get(op_name)
        pytorch_ms = baselines.get(op_name)
        if not kernel_path:
            fallback_ops.append(op_name)
            fallback_reasons[op_name] = {
                "reason": f"kernel_not_usable_for_{budget}",
                "kernel_status": "missing_generated",
                "kernel_source_path": None,
                "kernel_ms": None,
                "pytorch_ms": pytorch_ms,
                "speedup": None,
            }
            continue
        kernel_ms = node_value_for_kernel(project, op_name, kernel_path)
        if kernel_ms is None:
            mixed[op_name] = kernel_path
            continue
        if pytorch_ms and kernel_ms >= pytorch_ms:
            fallback_ops.append(op_name)
            fallback_reasons[op_name] = {
                "reason": f"pytorch_faster_than_{budget}_kernel",
                "kernel_status": "ok",
                "kernel_source_path": kernel_path,
                "kernel_ms": kernel_ms,
                "pytorch_ms": pytorch_ms,
                "speedup": (pytorch_ms / kernel_ms) if kernel_ms > 0 else None,
            }
            continue
        mixed[op_name] = kernel_path
    return mixed, fallback_reasons, sorted(fallback_ops)


def selected_override_map(selected: dict[str, str], fallback_ops: list[str]) -> dict[str, str]:
    out = dict(selected)
    for op_name in fallback_ops:
        out[op_name] = "__PYTORCH__"
    return out


def summarize_cast(path: Path, selected_ops: list[str], fallback_ops: list[str], fallback_reasons: dict[str, Any], failed_ops: list[str], mixed_equals_full: bool | None) -> dict[str, Any]:
    inspection = inspect_cast_package(path)
    manifest = inspection.get("manifest") if isinstance(inspection, dict) else {}
    selected_by_op: dict[str, Any] = {}
    if isinstance(manifest, dict):
        for op in manifest.get("ops", []) or []:
            if not isinstance(op, dict):
                continue
            evidence = op.get("selection_evidence")
            if isinstance(evidence, dict) and op.get("name"):
                selected_by_op[str(op["name"])] = {
                    "candidate_id": evidence.get("candidate_id"),
                    "evidence_tier": evidence.get("evidence_tier"),
                    "kernel_source_repo_relpath": evidence.get("kernel_source_repo_relpath"),
                    "selected_kernel_file": evidence.get("selected_kernel_file"),
                    "selected_kernel_id": evidence.get("selected_kernel_id"),
                    "selected_kernel_node_id": evidence.get("selected_kernel_node_id"),
                    "selected_kernel_source_kind": evidence.get("selected_kernel_source_kind"),
                    "selected_source_hash": evidence.get("selected_source_hash"),
                    "selection_reason": evidence.get("selection_reason"),
                }
    return {
        "file": path.name,
        "cast_file": path.name,
        "path": str(path),
        "sha256": safe_sha256_path(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "checksum_verified": bool(inspection.get("checksum_verified")) if isinstance(inspection, dict) else False,
        "loadability_blockers": list(inspection.get("loadability_blockers") or []) if isinstance(inspection, dict) else [],
        "precompiled_binaries": list(inspection.get("precompiled_binaries") or []) if isinstance(inspection, dict) else [],
        "target_sm_versions": list(inspection.get("target_sm_versions") or []) if isinstance(inspection, dict) else [],
        "selected_ops": selected_ops,
        "selected_kernel_by_op": selected_by_op,
        "selected_source_hashes": {
            op: item.get("selected_source_hash")
            for op, item in selected_by_op.items()
        },
        "torch_fallback_ops": fallback_ops,
        "fallback_reasons": fallback_reasons,
        "failed_ops": failed_ops,
        "mixed_equals_full": mixed_equals_full,
    }


def copy_to_benchmarking(path: Path) -> Path:
    BENCH_CAST_ROOT.mkdir(parents=True, exist_ok=True)
    dst = BENCH_CAST_ROOT / path.name
    shutil.copy2(path, dst)
    return dst


def update_exports_manifest(entries: list[dict[str, Any]]) -> None:
    manifest_path = BENCH_CAST_ROOT / "EXPORTS_MANIFEST.json"
    current = read_json(manifest_path, {})
    if not isinstance(current, dict):
        current = {}
    existing = current.get("exports")
    if not isinstance(existing, list):
        existing = []
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in existing:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("quantization") or ""),
            str(item.get("budget") or ""),
            str(item.get("policy") or ""),
        )
        keyed[key] = item
    for item in entries:
        key = (
            str(item.get("quantization") or ""),
            str(item.get("budget") or ""),
            str(item.get("policy") or ""),
        )
        keyed[key] = item
    write_json(
        manifest_path,
        {
            "model_slug": MODEL_SLUG,
            "updated_at_utc": utc_now(),
            "exports": [keyed[key] for key in sorted(keyed)],
        },
    )


def export_budget(project: str, quantization: str, budget: str) -> dict[str, Any]:
    selected, selection_details = budget_selection(project, budget)
    profiled = profiled_ops(project_dir(project))
    missing_full = sorted(op for op in profiled if op not in selected)
    full_fallback_overrides = selected_override_map(selected, missing_full)
    mixed_selected, fallback_reasons, mixed_fallback_ops = mixed_selection(project, budget, selected, profiled)
    mixed_equals_full = sorted(mixed_selected) == sorted(selected) and not missing_full
    mixed_policy = "mixed_equals_full" if mixed_equals_full else "mixed_forged"
    all_export_entries: list[dict[str, Any]] = []

    prefix = f"{MODEL_SLUG}__gb10__{quantization}__ultrachat200k_forge_v1__{budget}"
    export_root = project_dir(project) / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    full_path = export_root / f"{prefix}__full_forged.cast"
    export_cast_package(
        project_dir(project),
        selected_kernels=full_fallback_overrides,
        allow_operator_only=True,
        allow_micro_only=False,
        unsafe_override=False,
        allow_native_package=True,
        repo_root=REPO_ROOT,
        output_path=full_path,
    )
    full_summary = summarize_cast(
        full_path,
        selected_ops=sorted(selected),
        fallback_ops=[],
        fallback_reasons={},
        failed_ops=missing_full,
        mixed_equals_full=None,
    )

    mixed_path = export_root / f"{prefix}__{mixed_policy}.cast"
    mixed_overrides = selected_override_map(mixed_selected, mixed_fallback_ops)
    export_cast_package(
        project_dir(project),
        selected_kernels=mixed_overrides,
        allow_operator_only=True,
        allow_micro_only=False,
        unsafe_override=False,
        allow_native_package=True,
        repo_root=REPO_ROOT,
        output_path=mixed_path,
    )
    mixed_summary = summarize_cast(
        mixed_path,
        selected_ops=sorted(mixed_selected),
        fallback_ops=mixed_fallback_ops,
        fallback_reasons=fallback_reasons,
        failed_ops=[],
        mixed_equals_full=mixed_equals_full,
    )

    copied_full = copy_to_benchmarking(full_path)
    copied_mixed = copy_to_benchmarking(mixed_path)
    for policy, cast_summary, copied in [
        ("full_forged", full_summary, copied_full),
        (mixed_policy, mixed_summary, copied_mixed),
    ]:
        all_export_entries.append(
            {
                "model_slug": MODEL_SLUG,
                "model_id": MODEL_ID,
                "quantization": quantization,
                "budget": budget,
                "policy": policy,
                "source_project": project,
                "project_export_path": cast_summary["path"],
                "path": str(copied),
                "sha256": safe_sha256_path(copied),
                "selected_op_count": len(cast_summary["selected_ops"]),
                "fallback_count": len(cast_summary["torch_fallback_ops"]),
                "mixed_equals_full": cast_summary["mixed_equals_full"],
                "selected_ops": cast_summary["selected_ops"],
                "torch_fallback_ops": cast_summary["torch_fallback_ops"],
                "failed_ops": cast_summary["failed_ops"],
                "selected_kernel_by_op": cast_summary["selected_kernel_by_op"],
                "created_at_utc": utc_now(),
            }
        )
    update_exports_manifest(all_export_entries)

    generation_record_path = export_root / f"{prefix}__generation_record.json"
    operator_correctness_path = export_root / f"{prefix}__operator_correctness.json"
    manifest_path = export_root / f"{prefix}__export_manifest.json"
    llm_usage_path = export_root / f"{prefix}__llm_usage.jsonl"

    project_usage = llm_usage_summary(project_dir(project) / "llm_usage.db")
    target = budget_target(budget)
    arm = budget_arm(budget)
    arm_usage = llm_usage_summary(project_dir(project) / "llm_usage.db", row_filter=arm_usage_filter(arm))
    write_llm_usage_jsonl(project, llm_usage_path, arm)

    generation_record = {
        "record_type": "forge_generation_record",
        "created_at_utc": utc_now(),
        "project": project,
        "model_id": MODEL_ID,
        "model_slug": MODEL_SLUG,
        "quantization": quantization,
        "budget": budget,
        "optimization_budget": target,
        "validation_zip": str(VALIDATION_ZIP),
        "validation_dataset_sha256": safe_sha256_path(VALIDATION_ZIP),
        "profiled_ops": profiled,
        "selected_kernel_map": selected,
        "selection_details": selection_details,
        "logs": {
            "profile_log": str(logs_dir(project) / "profile.log"),
            "generate_log": str(logs_dir(project) / "generate.log"),
            "optimize_log": str(logs_dir(project) / "optimize.log"),
        },
        "llm_usage": {
            "project_total": project_usage.get("totals", {}),
            "arm_total": arm_usage.get("totals", {}),
            "per_op": arm_usage.get("by_op", {}),
            "project_call_log_jsonl": str(llm_usage_path.relative_to(REPO_ROOT)),
        },
    }
    write_json(generation_record_path, generation_record)

    operator_correctness = {
        "all_passed": None,
        "budget": budget,
        "created_at_utc": utc_now(),
        "project": project,
        "records": [],
        "tolerance": {
            "atol": 1e-2,
            "rtol": 1e-1,
        },
        "notes": "Candidate kernels were validated by Forge generation/optimization; no extra posthoc replay was run by this collection script.",
    }
    write_json(operator_correctness_path, operator_correctness)

    manifest = {
        "record_type": "forge_generation_export",
        "created_at_utc": utc_now(),
        "project": project,
        "model_id": MODEL_ID,
        "model_slug": MODEL_SLUG,
        "quantization": quantization,
        "budget": budget,
        "optimization_budget": target,
        "validation_dataset_sha256": safe_sha256_path(VALIDATION_ZIP),
        "generation_record_path": str(generation_record_path),
        "operator_correctness_path": str(operator_correctness_path),
        "cast_exports": {
            "full_forged": full_summary,
            mixed_policy: mixed_summary,
        },
        "full_forged": {
            "selected_ops": full_summary["selected_ops"],
            "torch_fallback_ops": [],
            "failed_ops": full_summary["failed_ops"],
        },
        "mixed_forged": {
            "selected_ops": mixed_summary["selected_ops"],
            "torch_fallback_ops": mixed_summary["torch_fallback_ops"],
            "fallback_reasons": mixed_summary["fallback_reasons"],
            "mixed_equals_full": mixed_summary["mixed_equals_full"],
        },
        "benchmark_plan_alignment": {
            "forge_generation_requirements_logged": True,
            "external_prompt_benchmark_metrics_logged_here": False,
            "external_prompt_benchmark_metrics_reason": "Out of scope for Forge generation/export; external held-out benchmark runs are intentionally separate.",
        },
    }
    write_json(manifest_path, manifest)

    model_file = REPO_ROOT / "paper_benchmarks" / "data_collection" / "models" / f"{project}.jsonl"
    record = {
        "record_type": "forge_generation_export",
        "created_at_utc": utc_now(),
        "project": project,
        "model_id": MODEL_ID,
        "model_slug": MODEL_SLUG,
        "quantization": quantization,
        "budget": budget,
        "optimization_budget": target,
        "validation_dataset_sha256": safe_sha256_path(VALIDATION_ZIP),
        "generation_record_path": str(generation_record_path),
        "generation_record_sha256": safe_sha256_path(generation_record_path),
        "operator_correctness_path": str(operator_correctness_path),
        "operator_correctness_sha256": safe_sha256_path(operator_correctness_path),
        "export_manifest_path": str(manifest_path),
        "export_manifest_sha256": safe_sha256_path(manifest_path),
        "llm_usage": arm_usage.get("totals", {}),
        "casts": {
            "full_forged": compact_cast_record(full_summary),
            mixed_policy: compact_cast_record(mixed_summary),
        },
        "operators": [],
    }
    append_jsonl(model_file, record)
    print(f"[runner] exported {project} {budget}: {full_path.name}, {mixed_path.name}")
    return manifest


def compact_cast_record(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": summary.get("file"),
        "sha256": summary.get("sha256"),
        "selected_ops": summary.get("selected_ops", []),
        "torch_fallback_ops": summary.get("torch_fallback_ops", []),
        "fallback_reasons": summary.get("fallback_reasons", {}),
        "failed_ops": summary.get("failed_ops", []),
        "loadability_blockers": summary.get("loadability_blockers", []),
        "precompiled_binaries": summary.get("precompiled_binaries", []),
        "mixed_equals_full": summary.get("mixed_equals_full"),
    }


def write_llm_usage_jsonl(project: str, out_path: Path, arm: str) -> None:
    db_path = project_dir(project) / "llm_usage.db"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        out_path.write_text("", encoding="utf-8")
        return
    filt = arm_usage_filter(arm)
    with sqlite3.connect(db_path) as conn, out_path.open("w", encoding="utf-8") as handle:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM llm_calls ORDER BY id").fetchall()
        except Exception:
            rows = []
        for row in rows:
            payload = dict(row)
            if filt(payload):
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_quantization(quantization: str, args: argparse.Namespace) -> None:
    project = project_name_for_quant(quantization)
    print(f"[runner] preparing {project}")
    ensure_project(project, quantization)
    run_profile(project, force=args.force_profile)
    print(f"[runner] {project}: profiled_ops={op_dirs(project)}")
    run_generate(project)
    for budget in args.budgets:
        target = budget_target(budget)
        if target > 0:
            run_optimize_to(project, target, workers=args.workers)
        export_budget(project, quantization, budget)


def normalize_budgets(raw_budgets: list[str] | None) -> list[str]:
    budgets = raw_budgets or ["zeroshot", "opt05"]
    seen: set[str] = set()
    normalized: list[str] = []
    for budget in budgets:
        budget_target(budget)
        if budget in seen:
            continue
        seen.add(budget)
        normalized.append(budget)
    return sorted(normalized, key=budget_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and collect Qwen3.5 A3B BF16/INT4 Forge arms by budget.")
    parser.add_argument(
        "--quantization",
        action="append",
        choices=["bf16", "int4"],
        help="Quantization to run. Repeatable. Defaults to bf16 and int4.",
    )
    parser.add_argument(
        "--budget",
        action="append",
        choices=sorted(BUDGET_TARGETS, key=budget_target),
        dest="budgets",
        help="Budget to run/export. Repeatable. Defaults to zeroshot and opt05.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force-profile", action="store_true")
    args = parser.parse_args()
    args.budgets = normalize_budgets(args.budgets)
    return args


def main() -> int:
    args = parse_args()
    quantizations = args.quantization or ["bf16", "int4"]
    if not VALIDATION_ZIP.exists():
        raise FileNotFoundError(VALIDATION_ZIP)
    if not Path(LOCAL_MODEL_DIR).exists():
        raise FileNotFoundError(LOCAL_MODEL_DIR)
    for quantization in quantizations:
        run_quantization(quantization, args)
    print(f"[runner] Qwen BF16/INT4 collection complete for budgets: {', '.join(args.budgets)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
