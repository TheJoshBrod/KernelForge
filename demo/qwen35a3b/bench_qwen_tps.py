#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.optimizer.benchmarking.integration import (
    empty_adapter_stats,
    invoke_kernel_launch,
    launch_params_for_runtime_kernel,
    merge_adapter_stats,
)


RESULT_SENTINEL = "__KFORGE_QWEN_TPS_RESULT__"
OP_PREFIX = "torch_nn_functional_"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_project_module(project_dir: Path):
    model_path = project_dir / "model.py"
    spec = importlib.util.spec_from_file_location("kforge_project_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import project model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kforge_project_model"] = module
    spec.loader.exec_module(module)
    return module


def _project_prompts(project_dir: Path) -> list[str]:
    validation_path = project_dir / "data" / "validation" / "prompts.jsonl"
    prompts: list[str] = []
    if not validation_path.exists():
        return prompts
    for line in validation_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        if text:
            prompts.append(text)
    return prompts


def _get_tokenizer(module):
    if hasattr(module, "_tokenizer"):
        return module._tokenizer()

    from transformers import AutoTokenizer

    if hasattr(module, "_model_source"):
        source = module._model_source()
    else:
        source = os.environ.get("KFORGE_QWEN35_DIR", "Qwen/Qwen3.5-35B-A3B")

    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _build_model(module, device: str):
    if device == "cuda" and hasattr(module, "Qwen3_5MoeForConditionalGeneration") and hasattr(module, "_model_source"):
        import torch
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from transformers import AutoConfig

        dtype = getattr(module, "DTYPE", torch.bfloat16)
        source = module._model_source()
        offload_dir = Path("/tmp/kforge_qwen_offload")
        offload_dir.mkdir(parents=True, exist_ok=True)
        print("[bench] building empty model and dispatching checkpoint to cuda", flush=True)
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        with init_empty_weights():
            model = module.Qwen3_5MoeForConditionalGeneration(config)
        model = load_checkpoint_and_dispatch(
            model,
            source,
            device_map={"": "cuda:0"},
            dtype=dtype,
            offload_folder=str(offload_dir),
            offload_state_dict=True,
        )
        return model, True

    print("[bench] loading model via project wrapper", flush=True)
    model = module.build_model()
    return model, False


def _select_prompts(
    prompts: list[str],
    tokenizer,
    *,
    max_prompts: int,
    selection: str,
    max_input_length: int,
) -> list[str]:
    if not prompts:
        raise RuntimeError("No prompts found in project validation set.")
    if selection == "first":
        return prompts[:max_prompts]

    scored: list[tuple[int, str]] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        scored.append((int(encoded["input_ids"].shape[1]), prompt))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [prompt for _, prompt in scored[:max_prompts]]


def _load_benchmark_results(project_dir: Path) -> list[dict[str, Any]]:
    bench_path = project_dir / "benchmarks" / "op_benchmarks.json"
    if not bench_path.exists():
        return []
    payload = json.loads(bench_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def _benchmark_results_by_op(project_dir: Path) -> dict[str, dict[str, Any]]:
    result_map: dict[str, dict[str, Any]] = {}
    for row in _load_benchmark_results(project_dir):
        if not isinstance(row, dict):
            continue
        op = row.get("op")
        if op:
            result_map[str(op)] = row
    return result_map


def _default_forged_ops(project_dir: Path) -> list[str]:
    results = _load_benchmark_results(project_dir)
    winners = []
    for row in results:
        if not isinstance(row, dict):
            continue
        deployment_safe_winner = str(row.get("deployment_safe_winner") or "")
        deployment_winner = str(row.get("deployment_winner") or "")
        integrated_status = str(row.get("integrated_kernel_status") or "")
        direct_winner = str(row.get("winner") or "")
        direct_status = str(row.get("kernel_status") or "")
        correctness = (
            row.get("deployment_correctness")
            if isinstance(row.get("deployment_correctness"), dict)
            else {}
        )
        strict_pass = correctness.get("strict_pass") if isinstance(correctness, dict) else None
        if deployment_safe_winner:
            if (
                deployment_safe_winner != "optimized"
                or integrated_status != "ok"
                or strict_pass is not True
            ):
                continue
        elif deployment_winner:
            if deployment_winner != "optimized" or integrated_status != "ok":
                continue
        elif direct_winner != "optimized" or direct_status != "ok":
            continue
        op = row.get("op")
        if op:
            winners.append(str(op))
    return winners


def _resolve_ops(project_dir: Path, ops_arg: str) -> list[str]:
    if not ops_arg or ops_arg == "winners":
        return _default_forged_ops(project_dir)
    return [item.strip() for item in ops_arg.split(",") if item.strip()]


def _resolve_generated_kernel(project_dir: Path, op_name: str) -> Path | None:
    kernel_path = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / op_name
        / "kernel.cu"
    )
    return kernel_path if kernel_path.exists() else None


def _best_tree_kernel(project_dir: Path, op_name: str) -> Path | None:
    nodes_db = project_dir / "trees" / op_name / "nodes.db"
    if not nodes_db.exists():
        return None
    with sqlite3.connect(nodes_db) as conn:
        row = conn.execute(
            "select code, value from nodes order by value asc limit 1"
        ).fetchone()
    if not row or not row[0]:
        return None
    kernel_path = project_dir / "trees" / str(row[0])
    return kernel_path if kernel_path.exists() else None


def _resolve_kernel_source(
    project_dir: Path,
    op_name: str,
    *,
    prefer_tree_best: bool,
    preferred_source: str = "",
) -> Path | None:
    generated = _resolve_generated_kernel(project_dir, op_name)
    tree_best = _best_tree_kernel(project_dir, op_name)
    if preferred_source == "optimized_tree":
        return tree_best or generated
    if preferred_source == "generated":
        return generated or tree_best
    if not prefer_tree_best:
        return generated
    if tree_best is not None:
        return tree_best
    return generated


def _make_kernel_patch(
    ext,
    orig_fn,
    launch_params: list[tuple[str, str]],
    stats: dict[str, Any],
    *,
    op_name: str,
):
    def patched(*args, **kwargs):
        stats["calls"] += 1
        try:
            call_adapter_stats = empty_adapter_stats()
            output = invoke_kernel_launch(
                ext,
                args=args,
                kwargs=kwargs,
                launch_params=launch_params,
                op_name=op_name,
                func=orig_fn,
                ensure_device=None,
                force_contiguous=False,
                adapter_stats=call_adapter_stats,
            )
            merge_adapter_stats(stats["adapter_stats"], call_adapter_stats)
            stats["kernel_success"] += 1
            return output
        except Exception as exc:
            stats["fallback"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            return orig_fn(*args, **kwargs)

    return patched


def _load_runtime_kernel_source(
    project_dir: Path,
    op_name: str,
    source_path: Path,
):
    from src.optimizer.backends.cuda import loader

    runtime_dir = (
        project_dir
        / "benchmarks"
        / "runtime_kernels"
        / op_name
        / source_path.stem
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_kernel = runtime_dir / "kernel.cu"
    runtime_kernel.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    ext = loader.load_kernel(
        runtime_dir,
        name=f"kforge_runtime_{op_name}_{source_path.stem}",
        build_dir=runtime_dir / ".runtime_build",
    )
    return ext, runtime_kernel


def _patch_forged_ops(
    project_dir: Path,
    selected_ops: list[str],
    *,
    prefer_tree_best: bool,
) -> tuple[dict[str, Any], dict[str, str], callable]:
    import torch.nn.functional as F

    stats: dict[str, Any] = {}
    sources: dict[str, str] = {}
    originals: list[tuple[str, Any]] = []
    benchmark_rows = _benchmark_results_by_op(project_dir)

    for op_name in selected_ops:
        if not op_name.startswith(OP_PREFIX):
            continue
        fn_attr = op_name[len(OP_PREFIX):]
        orig_fn = getattr(F, fn_attr, None)
        if orig_fn is None:
            continue

        benchmark_row = benchmark_rows.get(op_name, {})
        preferred_source = (
            str(benchmark_row.get("deployment_kernel_source") or "")
            if isinstance(benchmark_row, dict)
            else ""
        )
        generated_kernel = _resolve_generated_kernel(project_dir, op_name)
        tree_kernel = _best_tree_kernel(project_dir, op_name)
        kernel_path = _resolve_kernel_source(
            project_dir,
            op_name,
            prefer_tree_best=prefer_tree_best,
            preferred_source=preferred_source,
        )
        if kernel_path is None:
            continue
        actual_source = preferred_source
        if tree_kernel is not None and kernel_path == tree_kernel:
            actual_source = "optimized_tree"
        elif generated_kernel is not None and kernel_path == generated_kernel:
            actual_source = "generated"

        ext, runtime_kernel = _load_runtime_kernel_source(project_dir, op_name, kernel_path)
        launch_params = launch_params_for_runtime_kernel(runtime_kernel, ext.launch)
        op_stats = {
            "calls": 0,
            "kernel_success": 0,
            "fallback": 0,
            "last_error": "",
            "adapter_stats": empty_adapter_stats(),
            "deployment_kernel_source": actual_source,
        }
        stats[op_name] = op_stats
        sources[op_name] = str(kernel_path)
        originals.append((fn_attr, orig_fn))
        setattr(
            F,
            fn_attr,
            _make_kernel_patch(
                ext,
                orig_fn,
                launch_params,
                op_stats,
                op_name=op_name,
            ),
        )

    def restore():
        for fn_attr, orig_fn in originals:
            setattr(F, fn_attr, orig_fn)

    return stats, sources, restore


def _sync_if_needed(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


def _clone_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone().to(device)
        else:
            cloned[key] = value
    return cloned


def _prepare_prompt_batch(tokenizer, prompts: list[str], max_input_length: int) -> dict[str, Any]:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    return {key: value for key, value in batch.items()}


def _benchmark_once(model, batch: dict[str, Any], *, max_new_tokens: int, device: str):
    import torch

    run_batch = _clone_batch(batch, device)
    attention_mask = run_batch.get("attention_mask")
    input_ids = run_batch["input_ids"]
    total_prompt_tokens = int(attention_mask.sum().item()) if attention_mask is not None else int(input_ids.numel())
    batch_size = int(input_ids.shape[0])

    with torch.inference_mode():
        _sync_if_needed(device)
        prefill_start = time.perf_counter()
        outputs = model(**run_batch, use_cache=True)
        _sync_if_needed(device)
        prefill_end = time.perf_counter()

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token] if max_new_tokens > 0 else []

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        _sync_if_needed(device)
        decode_start = time.perf_counter()
        for _ in range(max(max_new_tokens - 1, 0)):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )
            model_inputs = model.prepare_inputs_for_generation(
                next_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
            outputs = model(**model_inputs)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
        _sync_if_needed(device)
        decode_end = time.perf_counter()

    generated_ids = (
        torch.cat(generated, dim=1).detach().cpu()
        if generated
        else input_ids.new_empty((batch_size, 0)).cpu()
    )
    prefill_ms = (prefill_end - prefill_start) * 1000.0
    decode_ms = (decode_end - decode_start) * 1000.0
    total_ms = prefill_ms + decode_ms
    generated_tokens = batch_size * max_new_tokens
    decode_tokens = batch_size * max(max_new_tokens - 1, 0)

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "generated_ids": generated_ids.tolist(),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    import statistics

    prefill_ms = [float(run["prefill_ms"]) for run in runs]
    decode_ms = [float(run["decode_ms"]) for run in runs]
    total_ms = [float(run["total_ms"]) for run in runs]
    prompt_tokens = int(runs[0]["prompt_tokens"])
    generated_tokens = int(runs[0]["generated_tokens"])
    decode_tokens = int(runs[0]["decode_tokens"])

    med_prefill_ms = statistics.median(prefill_ms)
    med_decode_ms = statistics.median(decode_ms)
    med_total_ms = statistics.median(total_ms)

    summary = {
        "prefill_ms_median": med_prefill_ms,
        "decode_ms_median": med_decode_ms,
        "total_ms_median": med_total_ms,
        "prefill_tokens_per_s": (prompt_tokens / (med_prefill_ms / 1000.0)) if med_prefill_ms > 0 else None,
        "decode_tokens_per_s": (decode_tokens / (med_decode_ms / 1000.0)) if med_decode_ms > 0 else None,
        "generated_tokens_per_s": (generated_tokens / (med_total_ms / 1000.0)) if med_total_ms > 0 else None,
        "total_tokens_per_s": ((prompt_tokens + generated_tokens) / (med_total_ms / 1000.0)) if med_total_ms > 0 else None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "runs": runs,
    }
    return summary


def _scenario_result(args) -> dict[str, Any]:
    import torch

    project_dir = (_repo_root() / "kernels" / "projects" / args.project).resolve()
    if not project_dir.exists():
        raise RuntimeError(f"Project not found: {project_dir}")

    module = _load_project_module(project_dir)
    print(f"[bench] preparing tokenizer and prompts for {args.mode}", flush=True)
    tokenizer = _get_tokenizer(module)
    prompts = _select_prompts(
        _project_prompts(project_dir),
        tokenizer,
        max_prompts=args.max_prompts,
        selection=args.prompt_selection,
        max_input_length=args.max_input_length,
    )
    batch = _prepare_prompt_batch(tokenizer, prompts, args.max_input_length)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    selected_ops: list[str] = []
    patch_stats: dict[str, Any] = {}
    patch_sources: dict[str, str] = {}
    restore = lambda: None

    if args.mode == "forged":
        print("[bench] patching forged ops", flush=True)
        selected_ops = _resolve_ops(project_dir, args.ops)
        patch_stats, patch_sources, restore = _patch_forged_ops(
            project_dir,
            selected_ops,
            prefer_tree_best=args.prefer_tree_best,
        )

    try:
        model, already_on_device = _build_model(module, args.device)
        if not already_on_device:
            model = model.to(args.device)
        model.eval()
        if args.mode == "compiled":
            print(
                f"[bench] compiling model backend={args.compile_backend} mode={args.compile_mode}",
                flush=True,
            )
            model = torch.compile(
                model,
                backend=args.compile_backend,
                mode=args.compile_mode,
                fullgraph=False,
                dynamic=True,
            )

        warmup_tokens = min(args.max_new_tokens, max(2, args.warmup_new_tokens))
        print(f"[bench] warmup runs={args.warmup_runs} tokens={warmup_tokens}", flush=True)
        for _ in range(args.warmup_runs):
            _benchmark_once(model, batch, max_new_tokens=warmup_tokens, device=args.device)

        timed_runs = []
        print(f"[bench] timed runs={args.timed_runs} max_new_tokens={args.max_new_tokens}", flush=True)
        for _ in range(args.timed_runs):
            timed_runs.append(
                _benchmark_once(model, batch, max_new_tokens=args.max_new_tokens, device=args.device)
            )
        summary = _summarize_runs(timed_runs)
        print("[bench] scenario complete", flush=True)
    finally:
        restore()

    result = {
        "mode": args.mode,
        "project": args.project,
        "device": args.device,
        "prompts": prompts,
        "selected_ops": selected_ops,
        "patch_stats": patch_stats,
        "patch_sources": patch_sources,
        "compile_backend": args.compile_backend if args.mode == "compiled" else "",
        "compile_mode": args.compile_mode if args.mode == "compiled" else "",
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "timed_runs": args.timed_runs,
        "warmup_runs": args.warmup_runs,
        "summary": summary,
    }
    return result


def _run_subprocess(args, mode: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--project",
        args.project,
        "--mode",
        mode,
        "--device",
        args.device,
        "--max-prompts",
        str(args.max_prompts),
        "--max-input-length",
        str(args.max_input_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--warmup-new-tokens",
        str(args.warmup_new_tokens),
        "--timed-runs",
        str(args.timed_runs),
        "--prompt-selection",
        args.prompt_selection,
        "--compile-backend",
        args.compile_backend,
        "--compile-mode",
        args.compile_mode,
    ]
    if args.ops:
        cmd.extend(["--ops", args.ops])
    if args.prefer_tree_best:
        cmd.append("--prefer-tree-best")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_repo_root()))
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        print(line, end="")

    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{mode} scenario failed with exit code {return_code}")

    for line in captured:
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    raise RuntimeError(f"Missing scenario result for {mode}")


def _compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    base_tps = float(base_summary["generated_tokens_per_s"])
    candidate_tps = float(candidate_summary["generated_tokens_per_s"])
    base_total_tps = float(base_summary["total_tokens_per_s"])
    candidate_total_tps = float(candidate_summary["total_tokens_per_s"])

    baseline_ids = baseline["summary"]["runs"][0]["generated_ids"]
    candidate_ids = candidate["summary"]["runs"][0]["generated_ids"]
    exact_match = baseline_ids == candidate_ids

    return {
        "generated_tokens_per_s_speedup": (candidate_tps / base_tps) if base_tps > 0 else None,
        "total_tokens_per_s_speedup": (candidate_total_tps / base_total_tps) if base_total_tps > 0 else None,
        "prefill_speedup": (
            float(base_summary["prefill_ms_median"]) / float(candidate_summary["prefill_ms_median"])
            if float(candidate_summary["prefill_ms_median"]) > 0
            else None
        ),
        "decode_speedup": (
            float(base_summary["decode_ms_median"]) / float(candidate_summary["decode_ms_median"])
            if float(candidate_summary["decode_ms_median"]) > 0
            else None
        ),
        "exact_generated_token_match": exact_match,
    }


def _write_compare_result(project: str, payload: dict[str, Any]) -> Path:
    out_path = _repo_root() / "kernels" / "projects" / project / "benchmarks" / "qwen_tps_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen TPS with and without forged kernels.")
    parser.add_argument("--project", default="test_qwen - NVIDIA GB10")
    parser.add_argument("--mode", choices=["compare", "compare3", "baseline", "compiled", "forged"], default="compare")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ops", default="")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument("--timed-runs", type=int, default=2)
    parser.add_argument("--prompt-selection", choices=["first", "longest"], default="longest")
    parser.add_argument("--prefer-tree-best", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()

    try:
        if args.mode in {"baseline", "compiled", "forged"}:
            result = _scenario_result(args)
            print(RESULT_SENTINEL + json.dumps(result), flush=True)
            return 0

        if args.mode == "compare3":
            baseline = _run_subprocess(args, "baseline")
            compiled = _run_subprocess(args, "compiled")
            forged = _run_subprocess(args, "forged")
            payload = {
                "project": args.project,
                "device": args.device,
                "ops": forged["selected_ops"],
                "baseline": baseline,
                "compiled": compiled,
                "forged": forged,
                "compiled_vs_baseline": _compare_results(baseline, compiled),
                "forged_vs_baseline": _compare_results(baseline, forged),
                "forged_vs_compiled": _compare_results(compiled, forged),
            }
            out_path = _write_compare_result(args.project, payload)
            print(json.dumps({"results_path": str(out_path)}, indent=2))
            return 0

        baseline = _run_subprocess(args, "baseline")
        forged = _run_subprocess(args, "forged")
        compare = _compare_results(baseline, forged)
        payload = {
            "project": args.project,
            "device": args.device,
            "ops": forged["selected_ops"],
            "baseline": baseline,
            "forged": forged,
            "compare": compare,
        }
        out_path = _write_compare_result(args.project, payload)
        print(json.dumps({"compare": compare, "results_path": str(out_path)}, indent=2))
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
