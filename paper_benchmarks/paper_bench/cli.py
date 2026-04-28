from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .artifacts import create_run_layout, create_run_layout_for_dir, default_runs_root, load_json_artifact, make_run_id, write_commands_txt
from .cast_export import copy_cast_artifact, export_cast_package, inspect_cast_package
from .kf_project import describe_project, find_project
from .kf_runtime import load_cast_model
from .llm_runner import load_prompt_records, run_llm_benchmark
from .op_runner import (
    load_operator_entries,
    resolve_operator_callable,
    resolve_project_operator_entries_dir,
    run_operator_benchmark,
)
from .provenance import build_environment_artifact_fields, collect_common_fields, safe_sha256_path, utc_now_iso
from .publication import build_llm_plan_payload, write_plan_payload
from .registry import (
    SyntheticWorkloadError,
    enforce_workload_policy,
    load_model_config,
    load_suite_config,
    resolve_model,
    validate_model_suite_registry,
)
from .report import summarize_run
from .schema import BenchmarkMode, CorrectnessStatus, EnvironmentArtifact, PERFORMANCE_STAGES, RunManifestArtifact, Stage, Variant, artifact_schema_bundle


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "models" / "registry.yaml"


def _default_model_suite_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "model_suite_10" / "registry.yaml"


def _hash_json_payload(payload) -> str | None:
    if not payload:
        return None
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_llm_shared(run_parser: argparse.ArgumentParser) -> None:
        run_parser.add_argument("--registry", default=str(_default_registry_path()))
        run_parser.add_argument("--model-id")
        run_parser.add_argument("--model-config")
        run_parser.add_argument("--suite")
        run_parser.add_argument("--suite-config")
        run_parser.add_argument("--variant", choices=[v.value for v in Variant])
        run_parser.add_argument("--variants", nargs="+", choices=[v.value for v in Variant])
        run_parser.add_argument("--runs-root", default=str(default_runs_root()))
        run_parser.add_argument("--out")
        run_parser.add_argument("--allow-synthetic-demo", action="store_true")
        run_parser.add_argument("--store-prompts", action="store_true")
        run_parser.add_argument("--reuse-cache", action="store_true")
        run_parser.add_argument("--fail-if-not-paper-eligible", action="store_true")
        run_parser.add_argument("--compile-backend")
        run_parser.add_argument("--compile-mode")
        run_parser.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=None)
        run_parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=None)
        run_parser.add_argument("--cast-package")
        run_parser.add_argument("--project-ref")
        run_parser.add_argument("--kf-require-precompiled", action=argparse.BooleanOptionalAction, default=False)
        run_parser.add_argument("--kf-allow-jit", action=argparse.BooleanOptionalAction, default=True)
        run_parser.add_argument("--kf-fail-on-fallback", action=argparse.BooleanOptionalAction, default=True)
        run_parser.add_argument("--kf-record-runtime-stats", action=argparse.BooleanOptionalAction, default=True)
        run_parser.add_argument("--certify-paper-ready", action="store_true")

    plan_llm = sub.add_parser("plan-llm")
    add_llm_shared(plan_llm)
    plan_llm.add_argument("--write-plan")
    plan_llm.add_argument("--fail-if-not-paper-ready", action="store_true")
    plan_llm.add_argument("--skip-cast-inspection", action="store_true")

    preflight = sub.add_parser("preflight")
    add_llm_shared(preflight)

    run_llm = sub.add_parser("run-llm")
    add_llm_shared(run_llm)

    run_ops = sub.add_parser("run-ops")
    run_ops.add_argument("--entries-dir", required=True)
    run_ops.add_argument("--op", required=True)
    run_ops.add_argument("--kernel-source-or-cast")
    run_ops.add_argument("--model-id")
    run_ops.add_argument("--model-path")
    run_ops.add_argument("--model-config-path")
    run_ops.add_argument("--callable-name")
    run_ops.add_argument("--variant", choices=[v.value for v in Variant])
    run_ops.add_argument("--variants", nargs="+", choices=[v.value for v in Variant], required=True)
    run_ops.add_argument("--runs-root", default=str(default_runs_root()))
    run_ops.add_argument("--out")
    run_ops.add_argument("--allow-synthetic-demo", action="store_true")
    run_ops.add_argument("--reuse-cache", action="store_true")
    run_ops.add_argument("--fail-if-not-paper-eligible", action="store_true")
    run_ops.add_argument("--warmup-runs", type=int, default=1)
    run_ops.add_argument("--timed-runs", type=int, default=3)
    run_ops.add_argument("--device", default="cuda")
    run_ops.add_argument("--project-ref")
    run_ops.add_argument("--compile-backend")
    run_ops.add_argument("--compile-mode")
    run_ops.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=None)
    run_ops.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=None)
    run_ops.add_argument("--kf-require-precompiled", action=argparse.BooleanOptionalAction, default=False)
    run_ops.add_argument("--kf-allow-jit", action=argparse.BooleanOptionalAction, default=True)
    run_ops.add_argument("--kf-fail-on-fallback", action=argparse.BooleanOptionalAction, default=True)
    run_ops.add_argument("--kf-record-runtime-stats", action=argparse.BooleanOptionalAction, default=True)

    validate = sub.add_parser("validate-artifact")
    validate.add_argument("artifact")
    validate.add_argument("--print-schema", action="store_true")

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--run-dir", required=True)

    validate_run = sub.add_parser("validate-run")
    validate_run.add_argument("--run-dir", required=True)
    validate_run.add_argument("--fail-if-not-paper-eligible", action="store_true")

    inspect_project = sub.add_parser("inspect-project")
    inspect_project.add_argument("--project-ref", required=True)

    export_cast = sub.add_parser("export-cast")
    export_cast.add_argument("--project-ref", required=True)
    export_cast.add_argument("--artifact-dir")
    export_cast.add_argument("--artifact-name")
    export_cast.add_argument("--allow-operator-only", action=argparse.BooleanOptionalAction, default=True)
    export_cast.add_argument("--allow-micro-only", action=argparse.BooleanOptionalAction, default=False)
    export_cast.add_argument("--unsafe-override", action=argparse.BooleanOptionalAction, default=False)
    export_cast.add_argument("--allow-native-package", action=argparse.BooleanOptionalAction, default=False)

    inspect_cast = sub.add_parser("inspect-cast")
    inspect_cast.add_argument("--cast-package", required=True)

    validate_registry = sub.add_parser("validate-model-registry")
    validate_registry.add_argument("--registry", default=str(_default_model_suite_registry_path()))

    return parser.parse_args()


def _resolve_llm_model(args: argparse.Namespace):
    if args.model_config:
        model_spec = load_model_config(args.model_config)
        if args.model_id and args.model_id != model_spec.model_id:
            raise ValueError(f"--model-id {args.model_id!r} does not match model config id {model_spec.model_id!r}")
        return model_spec, str(args.model_config), "model_config"
    if not args.model_id:
        raise ValueError("run-llm requires either --model-config or --model-id")
    return resolve_model(args.registry, args.model_id), str(args.registry), "registry"


def _resolve_llm_suite(args: argparse.Namespace):
    suite_path = args.suite_config or args.suite
    if not suite_path:
        raise ValueError("run-llm requires --suite-config or --suite")
    return load_suite_config(suite_path), str(suite_path)


def _requested_variants(
    args: argparse.Namespace,
    suite,
    *,
    default_to_suite: bool = False,
) -> list[Variant]:
    raw_variants = args.variants or ([args.variant] if args.variant else [])
    if not raw_variants:
        if default_to_suite:
            return list(suite.variants)
        raise ValueError("Specify --variant or --variants for run-llm/preflight")
    requested = [Variant(item) for item in raw_variants]
    for requested_variant in requested:
        if requested_variant not in suite.variants:
            raise ValueError(f"Variant {requested_variant.value} is not enabled by suite {suite.suite_id}")
    return requested


def _resolve_project_details(project_ref: str | None) -> dict | None:
    if not project_ref:
        return None
    return describe_project(find_project(project_ref))


def _cache_search_root(args: argparse.Namespace, layout) -> Path:
    runs_root = Path(args.runs_root).expanduser()
    if getattr(args, "out", None) and runs_root.resolve() == default_runs_root().resolve():
        return layout.run_dir.parent
    return runs_root


def _resolve_compile_settings(args: argparse.Namespace, model_spec=None) -> dict[str, Any]:
    defaults = dict(getattr(model_spec, "compile_settings", None) or {})
    return {
        "backend": args.compile_backend if args.compile_backend is not None else str(defaults.get("backend") or "inductor"),
        "mode": args.compile_mode if args.compile_mode is not None else defaults.get("mode"),
        "fullgraph": bool(args.compile_fullgraph) if args.compile_fullgraph is not None else bool(defaults.get("fullgraph", False)),
        "dynamic": bool(args.compile_dynamic) if args.compile_dynamic is not None else bool(defaults.get("dynamic", False)),
    }


def _prepare_llm_run_context(
    args: argparse.Namespace,
    *,
    materialize_run_dir: bool,
    default_variants_to_suite: bool = False,
):
    repo_root = Path(__file__).resolve().parents[2]
    model_spec, config_source_path, config_source_kind = _resolve_llm_model(args)
    suite, suite_path = _resolve_llm_suite(args)
    requested_variants = _requested_variants(args, suite, default_to_suite=default_variants_to_suite)
    cast_package_path = args.cast_package or model_spec.cast_package_path
    if Variant.kf_cast in requested_variants and not cast_package_path:
        raise ValueError("kf_cast benchmarking requires --cast-package or a cast_package_path in the model config.")
    if cast_package_path:
        model_spec = model_spec.model_copy(update={"cast_package_path": cast_package_path})

    prompt_suite = load_prompt_records(suite.workload_path)
    synthetic_workload = bool(suite.synthetic_workload or prompt_suite.synthetic_workload)
    suite_for_policy = suite.model_copy(update={"synthetic_workload": synthetic_workload})
    paper_eligible = enforce_workload_policy(suite_for_policy, args.allow_synthetic_demo)
    common = collect_common_fields(
        repo_root=repo_root,
        model_id=model_spec.model_id,
        model_path=model_spec.model_path,
        model_config_path=model_spec.model_config_path,
        suite_id=suite.suite_id,
        suite_path=suite_path,
        workload_path=suite.workload_path,
        command_line=sys.argv,
        paper_eligible=paper_eligible,
        synthetic_workload=synthetic_workload,
        cast_package_path=cast_package_path,
        registry_path=None if args.model_config else args.registry,
        quantization=getattr(model_spec, "quantization", None),
        quantization_config_hash=getattr(model_spec, "quantization_config_hash", None)
        or _hash_json_payload(getattr(model_spec, "quantization_config", None)),
        placement_profile=getattr(model_spec, "placement_profile", None),
        model_license=getattr(model_spec, "model_license", None),
        model_access_terms=getattr(model_spec, "model_access_terms", None),
        workload_slug=getattr(suite, "workload_slug", None),
        dataset_license=getattr(suite, "dataset_license", None),
        dataset_access_terms=getattr(suite, "dataset_access_terms", None),
        cache_mode=getattr(suite, "cache_mode", None),
    )
    common["compile_settings"] = _resolve_compile_settings(args, model_spec)
    common["kf_settings"] = {
        "cast_package_path": cast_package_path,
        "require_precompiled": bool(args.kf_require_precompiled),
        "allow_jit": bool(args.kf_allow_jit),
        "fail_on_fallback": bool(args.kf_fail_on_fallback),
        "record_runtime_stats": bool(args.kf_record_runtime_stats),
        "placement_profile": getattr(model_spec, "placement_profile", None),
        "device_map": getattr(model_spec, "device_map", None),
        "max_memory": getattr(model_spec, "max_memory", None),
    }
    project_details = _resolve_project_details(getattr(args, "project_ref", None))
    if project_details:
        common["kf_settings"]["project_ref"] = str(args.project_ref)
        common["kf_settings"]["project_root"] = project_details["project_root"]
        common["kf_settings"]["project_export_candidate_paths"] = list(project_details["export_candidate_paths"])
        common["kf_settings"]["project_op_benchmarks_path"] = project_details["benchmark_artifacts"]["op_benchmarks_path"]
        common["kf_settings"]["project_qwen_tps_compare_path"] = project_details["benchmark_artifacts"]["qwen_tps_compare_path"]
        common["kf_settings"]["project_cast_selection_policy"] = project_details["auto_best_fastest_valid"]["policy_name"]
        common["kf_settings"]["project_selected_kernel_map"] = project_details["auto_best_fastest_valid"]["selected_kernel_map"]
        common["kf_settings"]["project_export_paper_eligible"] = project_details["auto_best_fastest_valid"]["export_paper_eligible"]
    config_source_hash = safe_sha256_path(config_source_path)
    if config_source_hash:
        common["config_hashes"][config_source_kind] = config_source_hash
    if common.get("workload_hash"):
        common["config_hashes"]["prompt_file"] = common["workload_hash"]
    if getattr(model_spec, "expected_model_config_hash", None):
        if common.get("model_config_hash") != model_spec.expected_model_config_hash:
            issues = list(common.get("paper_eligibility_issues", []) or [])
            issues.append("model config hash mismatch against expected_model_config_hash")
            common["paper_eligibility_issues"] = list(dict.fromkeys(issues))
            common["paper_eligible"] = False

    if materialize_run_dir:
        if args.out:
            layout = create_run_layout_for_dir(args.out)
        else:
            layout = create_run_layout(args.runs_root, common["timestamp_utc"], model_spec.model_id, suite.suite_id)
        run_dir = layout.run_dir
        run_id = layout.run_id
        write_commands_txt(layout.run_dir, sys.argv)
    else:
        run_id = make_run_id(common["timestamp_utc"], model_spec.model_id, suite.suite_id)
        run_dir = Path(args.out) if args.out else Path(args.runs_root) / run_id
        layout = None

    common["run_id"] = run_id
    manifest = RunManifestArtifact(
        **common,
        artifact_type="run_manifest",
        benchmark_mode=suite.benchmark_mode,
        variant=None,
        stage=None,
        warmup_count=suite.warmup_count,
        timed_run_count=suite.timed_run_count,
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        run_dir=str(run_dir),
        variants_requested=requested_variants,
        stages_requested=suite.stages,
        description=suite.description,
        notes=list(suite.notes),
    )
    env_artifact = EnvironmentArtifact(
        **common,
        artifact_type="environment_snapshot",
        benchmark_mode=suite.benchmark_mode,
        variant=None,
        stage=None,
        warmup_count=suite.warmup_count,
        timed_run_count=suite.timed_run_count,
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        notes=list(suite.notes),
        **build_environment_artifact_fields(),
    )
    return layout, manifest, env_artifact, model_spec, suite, requested_variants


def _build_llm_plan_from_args(args: argparse.Namespace) -> dict:
    _, manifest, env_artifact, model_spec, suite, requested_variants = _prepare_llm_run_context(
        args,
        materialize_run_dir=False,
        default_variants_to_suite=True,
    )
    cast_inspection = None
    if Variant.kf_cast in requested_variants and manifest.cast_package_path and not getattr(args, "skip_cast_inspection", False):
        cast_inspection = inspect_cast_package(manifest.cast_package_path)
    return build_llm_plan_payload(
        manifest=manifest,
        env_artifact=env_artifact,
        model_spec=model_spec,
        suite=suite,
        requested_variants=requested_variants,
        cast_inspection=cast_inspection,
    )


def _requested_op_variants(args: argparse.Namespace) -> list[Variant]:
    raw_variants = args.variants or ([args.variant] if args.variant else [])
    if not raw_variants:
        raise ValueError("Specify --variants for run-ops")
    return [Variant(item) for item in raw_variants]


def _prepare_ops_run_context(args: argparse.Namespace):
    requested_variants = _requested_op_variants(args)
    if Variant.kf_cast in requested_variants and not args.kernel_source_or_cast:
        raise ValueError("kf_cast operator benchmarking requires --kernel-source-or-cast")
    project_details = _resolve_project_details(getattr(args, "project_ref", None))
    if project_details:
        canonical_entries_dir = resolve_project_operator_entries_dir(project_details["project_root"], args.op)
        if canonical_entries_dir is None:
            raise ValueError(
                f"No captured operator entries for {args.op!r} were found under project {args.project_ref!r}"
            )
        requested_entries_dir = Path(args.entries_dir).expanduser().resolve()
        if requested_entries_dir != canonical_entries_dir:
            raise ValueError(
                f"--entries-dir {requested_entries_dir} did not match canonical project entries "
                f"{canonical_entries_dir} for {args.op!r} under {args.project_ref!r}"
            )

    timestamp_utc = utc_now_iso()
    suite_id = f"operator_{args.op.replace('.', '_').replace('/', '_')}"
    model_id = args.model_id or suite_id
    if args.out:
        layout = create_run_layout_for_dir(args.out)
    else:
        layout = create_run_layout(args.runs_root, timestamp_utc, model_id, suite_id)
    write_commands_txt(layout.run_dir, sys.argv)

    entries, entry_summary = load_operator_entries(args.entries_dir, requested_op_name=args.op)
    synthetic_workload = bool(entry_summary["synthetic_workload"])
    suite_for_policy = SimpleNamespace(synthetic_workload=synthetic_workload)
    paper_eligible = enforce_workload_policy(suite_for_policy, args.allow_synthetic_demo)

    suite_spec_path = layout.run_dir / "raw" / "operator_suite.json"
    suite_spec_payload = {
        "suite_id": suite_id,
        "benchmark_mode": BenchmarkMode.operator.value,
        "workload_type": "operator_entries",
        "workload_path": str(Path(args.entries_dir).resolve()),
        "synthetic_workload": synthetic_workload,
        "variants": [variant.value for variant in requested_variants],
        "stages": [Stage.load.value, Stage.compile.value, Stage.warmup.value, Stage.operator.value],
        "warmup_count": int(args.warmup_runs),
        "timed_run_count": int(args.timed_runs),
        "device": args.device,
        "op_name": args.op,
        "callable_name": args.callable_name,
        "kernel_source_or_cast": args.kernel_source_or_cast,
        "description": f"Captured operator paper benchmark for {args.op}",
    }
    suite_spec_path.write_text(json.dumps(suite_spec_payload, indent=2, sort_keys=True), encoding="utf-8")

    model_path = args.model_path or args.kernel_source_or_cast or args.entries_dir
    common = collect_common_fields(
        repo_root=Path(__file__).resolve().parents[2],
        model_id=model_id,
        model_path=str(model_path),
        model_config_path=args.model_config_path,
        suite_id=suite_id,
        suite_path=str(suite_spec_path),
        workload_path=str(Path(args.entries_dir).resolve()),
        command_line=sys.argv,
        paper_eligible=paper_eligible,
        synthetic_workload=synthetic_workload,
        cast_package_path=args.kernel_source_or_cast if args.kernel_source_or_cast and str(args.kernel_source_or_cast).endswith(".cast") else None,
        exported_kernel_paths=[args.kernel_source_or_cast] if args.kernel_source_or_cast and not str(args.kernel_source_or_cast).endswith(".cast") else None,
        cache_mode="operator",
    )
    common["run_id"] = layout.run_id
    common["compile_settings"] = _resolve_compile_settings(args)
    common["kf_settings"] = {
        "cast_package_path": args.kernel_source_or_cast if args.kernel_source_or_cast and str(args.kernel_source_or_cast).endswith(".cast") else None,
        "kernel_source_or_cast": args.kernel_source_or_cast,
        "require_precompiled": bool(args.kf_require_precompiled),
        "allow_jit": bool(args.kf_allow_jit),
        "fail_on_fallback": bool(args.kf_fail_on_fallback),
        "record_runtime_stats": bool(args.kf_record_runtime_stats),
    }
    if project_details:
        common["kf_settings"]["project_ref"] = str(args.project_ref)
        common["kf_settings"]["project_root"] = project_details["project_root"]
        common["kf_settings"]["project_export_candidate_paths"] = list(project_details["export_candidate_paths"])
        common["kf_settings"]["project_op_benchmarks_path"] = project_details["benchmark_artifacts"]["op_benchmarks_path"]
        common["kf_settings"]["project_qwen_tps_compare_path"] = project_details["benchmark_artifacts"]["qwen_tps_compare_path"]
        common["kf_settings"]["project_cast_selection_policy"] = project_details["auto_best_fastest_valid"]["policy_name"]
        common["kf_settings"]["project_selected_kernel_map"] = project_details["auto_best_fastest_valid"]["selected_kernel_map"]
        common["kf_settings"]["project_export_paper_eligible"] = project_details["auto_best_fastest_valid"]["export_paper_eligible"]
    common["config_hashes"]["operator_suite"] = safe_sha256_path(suite_spec_path) or ""
    common["config_hashes"]["entry_set"] = entry_summary["entry_set_hash"]

    manifest = RunManifestArtifact(
        **common,
        artifact_type="run_manifest",
        benchmark_mode=BenchmarkMode.operator,
        variant=None,
        stage=None,
        warmup_count=int(args.warmup_runs),
        timed_run_count=int(args.timed_runs),
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        run_dir=str(layout.run_dir),
        variants_requested=requested_variants,
        stages_requested=[Stage.load, Stage.compile, Stage.warmup, Stage.operator],
        description=f"Captured operator paper benchmark for {args.op}",
        notes=[
            "Captured operator entries only. Synthetic entries are not paper-eligible.",
            "Direct source operator results are micro/operator only and must not be presented as deployment wins.",
        ],
    )
    env_artifact = EnvironmentArtifact(
        **common,
        artifact_type="environment_snapshot",
        benchmark_mode=BenchmarkMode.operator,
        variant=None,
        stage=None,
        warmup_count=int(args.warmup_runs),
        timed_run_count=int(args.timed_runs),
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        notes=list(manifest.notes),
        **build_environment_artifact_fields(),
    )
    suite = SimpleNamespace(
        suite_id=suite_id,
        benchmark_mode=BenchmarkMode.operator,
        workload_type="operator_entries",
        workload_path=str(Path(args.entries_dir).resolve()),
        synthetic_workload=synthetic_workload,
        variants=requested_variants,
        stages=[Stage.load, Stage.compile, Stage.warmup, Stage.operator],
        warmup_count=int(args.warmup_runs),
        timed_run_count=int(args.timed_runs),
        device=args.device,
        callable_name=args.callable_name,
        op_name=args.op,
        kernel_source_or_cast=args.kernel_source_or_cast,
    )
    return layout, manifest, env_artifact, suite, requested_variants


def _build_run_artifacts(
    args: argparse.Namespace,
    *,
    materialize_run_dir: bool,
) -> tuple[RunManifestArtifact, EnvironmentArtifact, object, object]:
    repo_root = Path(__file__).resolve().parents[2]
    model_spec = resolve_model(args.registry, args.model_id)
    suite = load_suite_config(args.suite)
    requested_variant = Variant(args.variant)
    if requested_variant not in suite.variants:
        raise ValueError(f"Variant {requested_variant.value} is not enabled by suite {suite.suite_id}")
    cast_package_path = args.cast_package or model_spec.cast_package_path
    if requested_variant == Variant.kf_cast and not cast_package_path:
        raise ValueError("kf_cast benchmarking requires --cast-package or a cast_package_path in the model registry.")
    if cast_package_path:
        model_spec = model_spec.model_copy(update={"cast_package_path": cast_package_path})

    paper_eligible = enforce_workload_policy(suite, args.allow_synthetic_demo)
    common = collect_common_fields(
        repo_root=repo_root,
        model_id=model_spec.model_id,
        model_path=model_spec.model_path,
        model_config_path=model_spec.model_config_path,
        suite_id=suite.suite_id,
        suite_path=args.suite,
        workload_path=suite.workload_path,
        command_line=sys.argv,
        paper_eligible=paper_eligible,
        synthetic_workload=suite.synthetic_workload,
        cast_package_path=cast_package_path,
        registry_path=args.registry,
        quantization=getattr(model_spec, "quantization", None),
        quantization_config_hash=getattr(model_spec, "quantization_config_hash", None)
        or _hash_json_payload(getattr(model_spec, "quantization_config", None)),
        placement_profile=getattr(model_spec, "placement_profile", None),
        model_license=getattr(model_spec, "model_license", None),
        model_access_terms=getattr(model_spec, "model_access_terms", None),
        workload_slug=getattr(suite, "workload_slug", None),
        dataset_license=getattr(suite, "dataset_license", None),
        dataset_access_terms=getattr(suite, "dataset_access_terms", None),
        cache_mode=getattr(suite, "cache_mode", None),
    )
    common["compile_settings"] = _resolve_compile_settings(args, model_spec)
    common["kf_settings"] = {
        "cast_package_path": cast_package_path,
        "require_precompiled": bool(args.kf_require_precompiled),
        "allow_jit": bool(args.kf_allow_jit),
        "fail_on_fallback": bool(args.kf_fail_on_fallback),
        "record_runtime_stats": bool(args.kf_record_runtime_stats),
        "placement_profile": getattr(model_spec, "placement_profile", None),
    }

    run_id = make_run_id(common["timestamp_utc"], model_spec.model_id, suite.suite_id)
    run_dir = Path(args.runs_root) / run_id
    if materialize_run_dir:
        layout = create_run_layout(args.runs_root, common["timestamp_utc"], model_spec.model_id, suite.suite_id)
        run_dir = layout.run_dir
        run_id = layout.run_id
        write_commands_txt(layout.run_dir, sys.argv)
    common["run_id"] = run_id

    manifest = RunManifestArtifact(
        **common,
        artifact_type="run_manifest",
        benchmark_mode=suite.benchmark_mode,
        variant=requested_variant,
        stage=None,
        warmup_count=suite.warmup_count,
        timed_run_count=suite.timed_run_count,
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        run_dir=str(run_dir),
        variants_requested=suite.variants,
        stages_requested=suite.stages,
        description=suite.description,
        notes=list(suite.notes),
    )
    env_artifact = EnvironmentArtifact(
        **common,
        artifact_type="environment_snapshot",
        benchmark_mode=suite.benchmark_mode,
        variant=requested_variant,
        stage=None,
        warmup_count=suite.warmup_count,
        timed_run_count=suite.timed_run_count,
        latency_samples_ms=[],
        correctness_status=CorrectnessStatus.not_applicable,
        notes=list(suite.notes),
        **build_environment_artifact_fields(),
    )
    return manifest, env_artifact, model_spec, suite


def _cmd_preflight(args: argparse.Namespace) -> int:
    _, manifest, env_artifact, model_spec, suite, requested_variants = _prepare_llm_run_context(
        args,
        materialize_run_dir=False,
        default_variants_to_suite=True,
    )
    payload = {
        "ok": True,
        "model_id": model_spec.model_id,
        "suite_id": suite.suite_id,
        "variants": [variant.value for variant in requested_variants],
        "compile_settings": dict(manifest.compile_settings),
        "paper_eligible": manifest.paper_eligible,
        "synthetic_workload": manifest.synthetic_workload,
        "schema_bundle_keys": sorted(artifact_schema_bundle().keys()),
        "gpu_count": env_artifact.gpu_count,
        "gpu_names": env_artifact.gpu_names,
    }
    project_root = manifest.kf_settings.get("project_root")
    if project_root:
        payload["project_ref"] = manifest.kf_settings.get("project_ref")
        payload["project_root"] = project_root
        payload["project_export_candidate_paths"] = manifest.kf_settings.get("project_export_candidate_paths", [])
        payload["project_op_benchmarks_path"] = manifest.kf_settings.get("project_op_benchmarks_path")
        payload["project_cast_selection_policy"] = manifest.kf_settings.get("project_cast_selection_policy")
        payload["project_selected_kernel_map"] = manifest.kf_settings.get("project_selected_kernel_map", {})
        payload["project_export_paper_eligible"] = manifest.kf_settings.get("project_export_paper_eligible")
    cast_package_path = manifest.kf_settings.get("cast_package_path")
    if cast_package_path:
        cast_inspection = inspect_cast_package(cast_package_path)
        _, runtime_meta = load_cast_model(
            cast_package_path,
            settings=manifest.kf_settings,
        )
        payload["cast_package"] = {
            "cast_path": cast_package_path,
            "cast_package_sha256": cast_inspection.get("cast_package_sha256"),
            "selection_policy": cast_inspection.get("selection_policy"),
            "selected_ops": cast_inspection.get("selected_ops", []),
            "kernel_source_hashes": cast_inspection.get("kernel_source_hashes", {}),
            "selected_source_hashes": cast_inspection.get("selected_source_hashes", {}),
            "precompiled_binaries": cast_inspection.get("precompiled_binaries", []),
            "loadability_blockers": cast_inspection.get("loadability_blockers", []),
            "export_paper_eligible": cast_inspection.get("export_paper_eligible"),
            "uses_non_deployment_evidence": cast_inspection.get("uses_non_deployment_evidence"),
            "runtime_validation": {
                "selected_ops": runtime_meta.get("selected_ops", []),
                "loaded_kernels": runtime_meta.get("loaded_kernels", []),
                "kernel_source_hashes": runtime_meta.get("kernel_source_hashes", {}),
                "selected_source_hashes": runtime_meta.get("selected_source_hashes", {}),
                "precompiled_vs_jit_path": runtime_meta.get("precompiled_vs_jit_path", {}),
                "runtime_load_time_ms": runtime_meta.get("runtime_load_time_ms"),
                "jit_compile_time_ms": runtime_meta.get("jit_compile_time_ms"),
                "precompiled_load_time_ms": runtime_meta.get("precompiled_load_time_ms"),
                "runtime_stats_enabled": runtime_meta.get("runtime_stats_enabled"),
                "runtime_patch_enabled": runtime_meta.get("runtime_patch_enabled"),
                "runtime_stats_api": runtime_meta.get("runtime_stats_api", {}),
                "placement_profile": runtime_meta.get("placement_profile"),
                "requested_device": runtime_meta.get("requested_device"),
                "loader_device": runtime_meta.get("loader_device"),
                "placement_audit": runtime_meta.get("placement_audit", {}),
                "toolchain_status": runtime_meta.get("toolchain_status", {}),
            },
        }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_plan_llm(args: argparse.Namespace) -> int:
    payload = _build_llm_plan_from_args(args)
    if args.write_plan:
        payload["plan_path"] = str(write_plan_payload(args.write_plan, payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_if_not_paper_ready and not payload["certification"]["paper_ready"]:
        return 2
    return 0


def _cmd_run_llm(args: argparse.Namespace) -> int:
    if args.certify_paper_ready:
        plan_payload = _build_llm_plan_from_args(args)
        if not plan_payload["certification"]["paper_ready"]:
            print(json.dumps(plan_payload, indent=2, sort_keys=True))
            return 2
    layout, manifest, env_artifact, model_spec, suite, requested_variants = _prepare_llm_run_context(args, materialize_run_dir=True)
    assert layout is not None
    common_fields = manifest.model_dump(
        mode="json",
        exclude={"artifact_type", "run_dir", "variants_requested", "stages_requested", "description"},
    )
    for requested_variant in requested_variants:
        run_llm_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env_artifact,
            manifest_artifact=manifest,
            model_spec=model_spec,
            suite=suite,
            variant=requested_variant,
            store_prompts=bool(args.store_prompts),
            reuse_cache=bool(args.reuse_cache),
            cache_search_root=_cache_search_root(args, layout),
        )
    summary = summarize_run(layout.run_dir)
    payload = {
        "ok": not args.fail_if_not_paper_eligible or summary.paper_eligible,
        "run_dir": str(layout.run_dir),
        "summary_rows": len(summary.rows),
        "compile_settings": dict(manifest.compile_settings),
        "paper_eligible": summary.paper_eligible,
        "paper_eligibility_issues": summary.paper_eligibility_issues,
    }
    project_root = manifest.kf_settings.get("project_root")
    if project_root:
        payload["project_ref"] = manifest.kf_settings.get("project_ref")
        payload["project_root"] = project_root
        payload["project_export_candidate_paths"] = manifest.kf_settings.get("project_export_candidate_paths", [])
        payload["project_cast_selection_policy"] = manifest.kf_settings.get("project_cast_selection_policy")
        payload["project_selected_kernel_map"] = manifest.kf_settings.get("project_selected_kernel_map", {})
        payload["project_export_paper_eligible"] = manifest.kf_settings.get("project_export_paper_eligible")
    print(json.dumps(payload, indent=2))
    if args.fail_if_not_paper_eligible and not summary.paper_eligible:
        return 2
    return 0


def _cmd_run_ops(args: argparse.Namespace) -> int:
    layout, manifest, env_artifact, suite, requested_variants = _prepare_ops_run_context(args)
    common_fields = manifest.model_dump(
        mode="json",
        exclude={"artifact_type", "run_dir", "variants_requested", "stages_requested", "description"},
    )
    for requested_variant in requested_variants:
        run_operator_benchmark(
            layout=layout,
            common_fields=common_fields,
            env_artifact=env_artifact,
            manifest_artifact=manifest,
            suite=suite,
            variant=requested_variant,
            reuse_cache=bool(args.reuse_cache),
            cache_search_root=_cache_search_root(args, layout),
        )
    summary = summarize_run(layout.run_dir)
    payload = {
        "ok": not args.fail_if_not_paper_eligible or summary.paper_eligible,
        "run_dir": str(layout.run_dir),
        "summary_rows": len(summary.rows),
        "paper_eligible": summary.paper_eligible,
        "paper_eligibility_issues": summary.paper_eligibility_issues,
    }
    print(json.dumps(payload, indent=2))
    if args.fail_if_not_paper_eligible and not summary.paper_eligible:
        return 2
    return 0


def _cmd_validate_artifact(args: argparse.Namespace) -> int:
    artifact = load_json_artifact(args.artifact)
    reasons = list(getattr(artifact, "paper_eligibility_issues", []) or [])
    requires_paper = False
    if getattr(artifact, "artifact_type", "") == "summary_report":
        requires_paper = not artifact.synthetic_workload
    elif getattr(artifact, "artifact_type", "") == "benchmark_result":
        stage = getattr(artifact, "stage", None)
        requires_paper = (stage in PERFORMANCE_STAGES) and not artifact.synthetic_workload

    payload = {
        "ok": not requires_paper or artifact.paper_eligible,
        "artifact_type": artifact.artifact_type,
        "schema_version": artifact.schema_version,
        "paper_eligible": artifact.paper_eligible,
        "paper_eligibility_issues": reasons,
        "paper_claim_status": getattr(artifact, "paper_claim_status", None),
        "claim_category": getattr(artifact, "claim_category", None),
        "validation_errors": getattr(artifact, "validation_errors", []),
        "validation_warnings": getattr(artifact, "validation_warnings", []),
    }
    if args.print_schema:
        payload["schema"] = artifact_schema_bundle()[artifact.artifact_type]
    print(json.dumps(payload, indent=2))
    if requires_paper and not artifact.paper_eligible:
        return 2
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    summary = summarize_run(args.run_dir)
    print(json.dumps({"ok": True, "run_dir": args.run_dir, "rows": len(summary.rows)}, indent=2))
    return 0


def _cmd_validate_run(args: argparse.Namespace) -> int:
    summary = summarize_run(args.run_dir)
    row_errors = [
        {
            "artifact_path": row.artifact_path,
            "variant": row.variant.value,
            "stage": row.stage.value,
            "comparison_group": row.comparison_group,
            "errors": row.validation_errors,
            "warnings": row.validation_warnings,
            "paper_claim_status": row.paper_claim_status,
            "claim_category": row.claim_category,
        }
        for row in summary.rows
        if row.validation_errors or row.validation_warnings
    ]
    payload = {
        "ok": not args.fail_if_not_paper_eligible or summary.paper_eligible,
        "run_dir": args.run_dir,
        "paper_eligible": summary.paper_eligible,
        "paper_eligibility_issues": summary.paper_eligibility_issues,
        "row_count": len(summary.rows),
        "validation_issue_rows": row_errors,
    }
    print(json.dumps(payload, indent=2))
    if args.fail_if_not_paper_eligible and not summary.paper_eligible:
        return 2
    return 0


def _cmd_inspect_project(args: argparse.Namespace) -> int:
    payload = {"ok": True, **describe_project(find_project(args.project_ref))}
    print(json.dumps(payload, indent=2))
    return 0


def _default_cast_artifact_name(*, git_commit: str | None, selection_policy: str, timestamp_utc: str) -> str:
    short_commit = (git_commit or "unknown")[:7]
    date_part = timestamp_utc.split("T", 1)[0]
    return f"qwen35a3b_{selection_policy}_{short_commit}_{date_part}.cast"


def _render_export_report_md(payload: dict[str, object]) -> str:
    selected_ops = list(payload.get("selected_ops", []) or [])
    selected_kernel_metadata = payload.get("selected_kernel_metadata", {}) or {}
    rejected_summary = payload.get("rejected_candidate_summary", {}) or {}
    lines = [
        "# Qwen CAST Export Report",
        "",
        f"- Project ref: `{payload.get('project_ref', '')}`",
        f"- Resolved project root: `{payload.get('project_root', '')}`",
        f"- Selection policy: `{payload.get('selection_policy', '')}`",
        f"- Exported CAST: `{payload.get('cast_package_path', '')}`",
        f"- CAST SHA256: `{payload.get('cast_package_sha256', '')}`",
        f"- Deployment-paper eligible: `{payload.get('export_paper_eligible', False)}`",
        f"- All selected kernels deployment-paper eligible: `{payload.get('all_selected_kernels_deployment_paper_eligible', False)}`",
        f"- Used operator/micro-only evidence: `{payload.get('uses_non_deployment_evidence', False)}`",
        f"- Target SM: `{payload.get('target_sm', '')}`",
        f"- GPU: `{payload.get('gpu_name', '')}`",
        "",
        "## Selected Ops",
    ]
    if not selected_ops:
        lines.append("- None")
    else:
        for op_name in selected_ops:
            meta = selected_kernel_metadata.get(op_name, {}) if isinstance(selected_kernel_metadata, dict) else {}
            lines.append(
                f"- `{op_name}`: `{meta.get('candidate_id', '')}` "
                f"tier=`{meta.get('evidence_tier', '')}` "
                f"hash=`{meta.get('selected_source_hash', '')}`"
            )
    lines.extend(["", "## Rejected Candidates"])
    if not rejected_summary:
        lines.append("- None")
    else:
        for op_name in sorted(rejected_summary):
            summary = rejected_summary[op_name] if isinstance(rejected_summary, dict) else {}
            lines.append(
                f"- `{op_name}`: total={summary.get('total', 0)} reasons={json.dumps(summary.get('reasons', {}), sort_keys=True)}"
            )
    blockers = list(payload.get("loadability_blockers", []) or [])
    lines.extend(["", "## Preflight"])
    lines.append(f"- Checksum verified: `{payload.get('checksum_verified', False)}`")
    lines.append(f"- Loadability blockers: `{'; '.join(blockers) if blockers else 'none recorded'}`")
    manifest_summary = payload.get("cast_manifest_summary", {}) or {}
    lines.extend(["", "## Manifest Summary", "```json", json.dumps(manifest_summary, indent=2, sort_keys=True), "```"])
    return "\n".join(lines) + "\n"


def _cmd_export_cast(args: argparse.Namespace) -> int:
    project_root = find_project(args.project_ref)
    export_result = export_cast_package(
        project_root,
        project_ref=args.project_ref,
        allow_operator_only=bool(args.allow_operator_only),
        allow_micro_only=bool(args.allow_micro_only),
        unsafe_override=bool(args.unsafe_override),
        allow_native_package=bool(args.allow_native_package),
        repo_root=Path(__file__).resolve().parents[2],
    )
    cast_source = Path(export_result["export_path"]).resolve()
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None
    artifact_path = cast_source
    if artifact_dir is not None:
        artifact_name = args.artifact_name or _default_cast_artifact_name(
            git_commit=str(export_result.get("header", {}).get("git_commit") or ""),
            selection_policy=str(export_result.get("manifest", {}).get("selection_policy") or "auto_best_fastest_valid"),
            timestamp_utc=str(export_result.get("manifest", {}).get("timestamp") or utc_now_iso()),
        )
        artifact_path = copy_cast_artifact(cast_source, destination_dir=artifact_dir, filename=artifact_name)

    inspection = inspect_cast_package(artifact_path)
    selected_kernel_metadata = inspection.get("selected_kernel_metadata", {}) or {}
    selected_plan_ops = export_result["selection_manifest"].get("selected_ops", {}) or {}
    selected_kernel_paper_eligibility = {
        op_name: bool(meta.get("paper_eligible"))
        for op_name, meta in selected_plan_ops.items()
        if isinstance(meta, dict)
    }
    payload = {
        "ok": True,
        "project_ref": args.project_ref,
        "project_root": str(project_root.resolve()),
        "selection_policy": export_result["manifest"].get("selection_policy"),
        "selected_ops": inspection.get("selected_ops", []),
        "selected_kernel_metadata": selected_kernel_metadata,
        "selected_kernel_source_hashes": {
            op_name: meta.get("selected_source_hash")
            for op_name, meta in selected_kernel_metadata.items()
            if isinstance(meta, dict)
        },
        "benchmark_evidence_references": {
            op_name: meta.get("benchmark_reference")
            for op_name, meta in selected_kernel_metadata.items()
            if isinstance(meta, dict)
        },
        "evidence_tiers": {
            op_name: meta.get("evidence_tier")
            for op_name, meta in selected_kernel_metadata.items()
            if isinstance(meta, dict)
        },
        "rejected_candidates": export_result["selection_manifest"].get("rejected_candidates", {}),
        "rejected_candidate_summary": export_result["manifest"].get("rejected_candidate_summary", {}),
        "export_paper_eligible": bool(inspection.get("export_paper_eligible")),
        "all_selected_kernels_deployment_paper_eligible": all(selected_kernel_paper_eligibility.values()) if selected_kernel_paper_eligibility else False,
        "selected_kernel_paper_eligibility": selected_kernel_paper_eligibility,
        "uses_non_deployment_evidence": bool(inspection.get("uses_non_deployment_evidence")),
        "cast_package_path": str(artifact_path),
        "cast_package_sha256": inspection.get("cast_package_sha256"),
        "cast_manifest_summary": {
            "project_ref": export_result["manifest"].get("project_ref"),
            "project_root": export_result["manifest"].get("project_root"),
            "model_class": export_result["manifest"].get("model_class"),
            "model_entrypoints": export_result["manifest"].get("model_entrypoints"),
            "weight_file": export_result["manifest"].get("weight_file"),
            "selected_ops": export_result["manifest"].get("selected_ops"),
            "selected_kernel_metadata": export_result["manifest"].get("selected_kernel_metadata"),
            "export_paper_eligible": export_result["manifest"].get("export_paper_eligible"),
        },
        "manifest": export_result["manifest"],
        "header": export_result["header"],
        "selection_manifest": export_result["selection_manifest"],
        "precompiled_binaries": inspection.get("precompiled_binaries", []),
        "target_sm": export_result.get("target_sm"),
        "gpu_name": inspection.get("gpu_name"),
        "gpu_capability": inspection.get("gpu_capability"),
        "checksum_verified": bool(inspection.get("checksum_verified")),
        "loadability_blockers": inspection.get("loadability_blockers", []),
        "product_export_path": str(cast_source),
    }

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = artifact_dir / "export_report.json"
        report_md_path = artifact_dir / "EXPORT_REPORT.md"
        report_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        report_md_path.write_text(_render_export_report_md(payload), encoding="utf-8")
        payload["report_json_path"] = str(report_json_path)
        payload["report_md_path"] = str(report_md_path)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_cast(args: argparse.Namespace) -> int:
    payload = {"ok": True, **inspect_cast_package(args.cast_package)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_validate_model_registry(args: argparse.Namespace) -> int:
    report = validate_model_suite_registry(args.registry)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "plan-llm":
            return _cmd_plan_llm(args)
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "run-llm":
            return _cmd_run_llm(args)
        if args.command == "run-ops":
            return _cmd_run_ops(args)
        if args.command == "validate-artifact":
            return _cmd_validate_artifact(args)
        if args.command == "summarize":
            return _cmd_summarize(args)
        if args.command == "validate-run":
            return _cmd_validate_run(args)
        if args.command == "inspect-project":
            return _cmd_inspect_project(args)
        if args.command == "export-cast":
            return _cmd_export_cast(args)
        if args.command == "inspect-cast":
            return _cmd_inspect_cast(args)
        if args.command == "validate-model-registry":
            return _cmd_validate_model_registry(args)
        return 1
    except SyntheticWorkloadError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[paper_bench] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
