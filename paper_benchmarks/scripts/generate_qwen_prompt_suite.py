from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import urllib.request
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

BUCKET_RANGES = {
    "short": (1, 128),
    "medium": (129, 512),
    "long": (513, 1024),
}

MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
LONGBENCH_TEMPLATE_URL = "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json"
SHAREGPT_REPO = "lewtun/sharegpt_prompts_annotated"
LONGBENCH_REPO = "THUDM/LongBench"


def _default_tokenizer_path() -> Path:
    candidates = [
        Path("/home/gb10/model-cache/Qwen3.5-35B-A3B"),
        Path("/home/gb10/model-cache/Qwen3.6-35B-A3B"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a local Qwen tokenizer snapshot under /home/gb10/model-cache. "
        "Pass --tokenizer-path explicitly."
    )


def _load_json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _bucket_for_length(token_length: int) -> str | None:
    for bucket, (min_tokens, max_tokens) in BUCKET_RANGES.items():
        if min_tokens <= token_length <= max_tokens:
            return bucket
    return None


def _prompt_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _creation_command() -> str:
    return " ".join(
        [shlex.quote(str(Path(sys.executable).resolve()))]
        + [shlex.quote(part) for part in sys.argv]
    )


def _build_short_prompts(tokenizer, *, target_count: int) -> list[dict[str, Any]]:
    rows = _load_json_lines(urllib.request.urlopen(MT_BENCH_URL).read().decode("utf-8"))
    selected: list[dict[str, Any]] = []
    for row in rows:
        question_id = int(row["question_id"])
        category = str(row.get("category") or "unknown")
        for turn_index, text in enumerate(row["turns"], start=1):
            token_length = len(tokenizer.encode(text, add_special_tokens=True))
            if _bucket_for_length(token_length) != "short":
                continue
            selected.append(
                {
                    "id": f"mt_bench_q{question_id:03d}_t{turn_index}",
                    "text": text,
                    "source": (
                        "MT-Bench / lm-sys/FastChat / fastchat/llm_judge/data/mt_bench/question.jsonl "
                        f"(question_id={question_id}, turn={turn_index})"
                    ),
                    "bucket": "short",
                    "notes": f"category={category}; qwen_token_count={token_length}",
                }
            )
            if len(selected) >= target_count:
                return selected
    raise RuntimeError(f"Unable to collect {target_count} short prompts from MT-Bench")


def _build_medium_prompts(tokenizer, *, target_count: int) -> list[dict[str, Any]]:
    parquet_path = hf_hub_download(
        repo_id=SHAREGPT_REPO,
        repo_type="dataset",
        filename="data/no_code-00000-of-00001-5253f60a2fdb7a13.parquet",
    )
    frame = pd.read_parquet(parquet_path)
    unique = frame.drop_duplicates(subset=["prompt_id"])
    selected: list[dict[str, Any]] = []
    for _, row in unique.iterrows():
        if str(row.get("rating") or "").strip().lower() != "good":
            continue
        prompt = str(row["prompt"])
        token_length = len(tokenizer.encode(prompt, add_special_tokens=True))
        if _bucket_for_length(token_length) != "medium":
            continue
        tags = row.get("tags")
        if hasattr(tags, "tolist"):
            tags = tags.tolist()
        tags_text = ",".join(str(item) for item in (tags or []))
        selected.append(
            {
                "id": f"sharegpt_{row['prompt_id']}",
                "text": prompt,
                "source": (
                    "sharegpt_prompts_annotated / lewtun/sharegpt_prompts_annotated "
                    f"(prompt_id={row['prompt_id']})"
                ),
                "bucket": "medium",
                "notes": (
                    f"rating={row.get('rating')}; annotator_dataset=no_code; "
                    f"tags={tags_text or '-'}; qwen_token_count={token_length}"
                ),
            }
        )
        if len(selected) >= target_count:
            return selected
    raise RuntimeError(f"Unable to collect {target_count} medium prompts from ShareGPT")


def _build_long_prompts(tokenizer, *, target_count: int) -> list[dict[str, Any]]:
    zip_path = Path(
        hf_hub_download(
            repo_id=LONGBENCH_REPO,
            repo_type="dataset",
            filename="data.zip",
        )
    )
    templates = json.loads(urllib.request.urlopen(LONGBENCH_TEMPLATE_URL).read().decode("utf-8"))
    template = str(templates["multi_news"])
    selected: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        lines = archive.read("data/multi_news.jsonl").decode("utf-8").splitlines()
    for line in lines:
        row = json.loads(line)
        mapping = {
            key: (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
            for key, value in row.items()
        }
        prompt = template.format(**mapping)
        token_length = len(tokenizer.encode(prompt, add_special_tokens=True))
        if _bucket_for_length(token_length) != "long":
            continue
        selected.append(
            {
                "id": f"longbench_multi_news_{row['_id']}",
                "text": prompt,
                "source": (
                    "LongBench / THUDM/LongBench / data/multi_news.jsonl "
                    f"(_id={row['_id']})"
                ),
                "bucket": "long",
                "notes": f"dataset=multi_news; qwen_token_count={token_length}",
            }
        )
        if len(selected) >= target_count:
            return selected
    raise RuntimeError(f"Unable to collect {target_count} long prompts from LongBench multi_news")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_manifest(
    *,
    prompt_path: Path,
    rows: list[dict[str, Any]],
    tokenizer_path: Path,
    tokenizer,
) -> dict[str, Any]:
    bucket_counts: OrderedDict[str, int] = OrderedDict()
    for bucket in ("short", "medium", "long"):
        bucket_counts[bucket] = sum(1 for row in rows if row["bucket"] == bucket)

    return {
        "version": 1,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_hash": _prompt_hash(prompt_path),
        "prompt_count": len(rows),
        "source_dataset_provenance": [
            {
                "name": "MT-Bench",
                "repository": "lm-sys/FastChat",
                "source_type": "benchmark_dataset",
                "file": "fastchat/llm_judge/data/mt_bench/question.jsonl",
                "url": MT_BENCH_URL,
                "selected_bucket": "short",
                "selected_count": bucket_counts["short"],
            },
            {
                "name": "ShareGPT Prompts Annotated",
                "repository": SHAREGPT_REPO,
                "source_type": "public_prompt_dataset",
                "file": "data/no_code-00000-of-00001-5253f60a2fdb7a13.parquet",
                "url": f"https://huggingface.co/datasets/{SHAREGPT_REPO}",
                "selected_bucket": "medium",
                "selected_count": bucket_counts["medium"],
            },
            {
                "name": "LongBench multi_news",
                "repository": LONGBENCH_REPO,
                "source_type": "benchmark_dataset",
                "file": "data/multi_news.jsonl",
                "url": f"https://huggingface.co/datasets/{LONGBENCH_REPO}",
                "selected_bucket": "long",
                "selected_count": bucket_counts["long"],
            },
        ],
        "selection_method": {
            "name": "stable_source_order_by_bucket",
            "description": (
                "Use stable source order with no random sampling. Select the first prompts whose "
                "Qwen tokenizer lengths fall into the configured bucket ranges."
            ),
            "bucket_ranges_tokens": BUCKET_RANGES,
            "target_counts": bucket_counts,
            "sampling": "none",
            "deduplication": {
                "sharegpt_prompts_annotated": "drop duplicate prompt_id rows before selection",
            },
            "order": [
                "short prompts from MT-Bench in source order by question_id and turn index",
                "medium prompts from ShareGPT no_code in stable parquet order after prompt_id deduplication",
                "long prompts from LongBench multi_news in source file order",
            ],
        },
        "bucket_counts": bucket_counts,
        "tokenizer_used": {
            "path": str(tokenizer_path.resolve()),
            "class": tokenizer.__class__.__name__,
        },
        "creation_command": _creation_command(),
        "creation_timestamp": _utc_now(),
        "prompt_text_may_be_stored_in_raw_artifacts": False,
        "synthetic_workload": False,
        "notes": [
            "Prompt text comes from public benchmark and public prompt datasets only.",
            "Raw benchmark artifacts should redact prompt text unless --store-prompts is explicitly enabled.",
            "This suite is frozen in file order and must not be sampled or reshuffled at benchmark time.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the frozen Qwen paper prompt suite")
    parser.add_argument(
        "--tokenizer-path",
        default=str(_default_tokenizer_path()),
        help="Local tokenizer path used for bucket assignment.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.jsonl",
    )
    parser.add_argument(
        "--output-manifest",
        default="paper_benchmarks/workloads/qwen35a3b/qwen_paper_prompts_v1.manifest.json",
    )
    parser.add_argument("--prompts-per-bucket", type=int, default=20)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )

    target_count = int(args.prompts_per_bucket)
    rows = (
        _build_short_prompts(tokenizer, target_count=target_count)
        + _build_medium_prompts(tokenizer, target_count=target_count)
        + _build_long_prompts(tokenizer, target_count=target_count)
    )

    prompt_path = Path(args.output_jsonl).expanduser().resolve()
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    _write_jsonl(prompt_path, rows)

    manifest = _build_manifest(
        prompt_path=prompt_path,
        rows=rows,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "prompt_file": str(prompt_path),
                "prompt_file_hash": manifest["prompt_file_hash"],
                "manifest_file": str(manifest_path),
                "bucket_counts": manifest["bucket_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
