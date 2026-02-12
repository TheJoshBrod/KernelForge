from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .compat import load_manifest, manifest_compatible
from .constants import GGML_METAL_RESOURCES_ENV, PACKS_CACHE_DIR
from .pack import select_best_compatible_pack, update_fallback_rules
from .runtime_args import split_runtime_args_and_env
from .types import DeviceProfile, ModelProfile

KERNEL_RE = re.compile(r"kernel\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)


def _select_dispatch_rule(manifest: dict[str, Any], model: ModelProfile) -> dict[str, Any]:
    rules = manifest.get("dispatch_rules") or []
    if not isinstance(rules, list):
        return {}
    model_arch = (model.architecture or "").strip().lower()
    for row in rules:
        if not isinstance(row, dict):
            continue
        arch = str(row.get("model_arch", "")).strip().lower()
        if arch and arch != model_arch:
            continue
        return row
    return {}


def _extract_kernel_name(stderr: str) -> str | None:
    match = KERNEL_RE.search(stderr or "")
    return match.group(1) if match else None


def _prepare_cli_for_source_compile(llama_cli: Path, resources_dir: Path) -> tuple[Path, Path | None]:
    if (resources_dir / "ggml-metal.metal").exists() and not (resources_dir / "default.metallib").exists():
        temp_root = Path(tempfile.mkdtemp(prefix="cgins_as_pack_run_"))
        copied = temp_root / "llama-cli"
        shutil.copy2(llama_cli, copied)
        copied.chmod(0o755)
        return copied, temp_root
    return llama_cli, None


def run_with_optional_pack(
    *,
    llama_cli: Path,
    model: ModelProfile,
    extra_args: list[str],
    pack_id: str,
    device: DeviceProfile,
    llamacpp_commit: str,
) -> int:
    cmd = [str(llama_cli), "-m", str(model.path)] + list(extra_args)

    if not pack_id:
        pack_id = select_best_compatible_pack(
            model=model,
            device=device,
            llamacpp_commit=llamacpp_commit,
        )
        if not pack_id:
            return subprocess.call(cmd)

    pack_dir = PACKS_CACHE_DIR / pack_id
    resources_dir = pack_dir / "resources"
    manifest = load_manifest(pack_dir)
    ok, _why = manifest_compatible(
        manifest,
        device=device,
        model=model,
        llamacpp_commit=llamacpp_commit,
    )
    if not ok:
        return subprocess.call(cmd)

    runtime_args, runtime_env = split_runtime_args_and_env(manifest.get("runtime_args", []))
    selected_rule = _select_dispatch_rule(manifest, model)
    tuned_cli, temp_cli_root = _prepare_cli_for_source_compile(llama_cli, resources_dir)
    tuned_cmd = [str(tuned_cli), "-m", str(model.path)] + runtime_args + list(extra_args)

    env = dict(os.environ)
    env.update(runtime_env)
    if resources_dir.exists():
        env[GGML_METAL_RESOURCES_ENV] = str(resources_dir)
    if selected_rule:
        env["CGINS_DISPATCH_RULE_ID"] = str(selected_rule.get("rule_id", ""))

    try:
        first = subprocess.run(tuned_cmd, env=env)
    finally:
        if temp_cli_root is not None:
            shutil.rmtree(temp_cli_root, ignore_errors=True)
    if first.returncode == 0:
        return 0

    # Fallback path: mark variant error and retry stock.
    # We don't have stderr content from direct execution, so keep generic message.
    update_fallback_rules(
        pack_dir,
        kernel_name=None,
        error=f"runtime failed with code {first.returncode}",
    )
    second = subprocess.run(cmd)
    return second.returncode


def run_with_pack_capture(
    *,
    llama_cli: Path,
    model: ModelProfile,
    extra_args: list[str],
    pack_id: str,
    device: DeviceProfile,
    llamacpp_commit: str,
) -> tuple[int, dict[str, Any]]:
    cmd = [str(llama_cli), "-m", str(model.path)] + list(extra_args)

    if not pack_id:
        pack_id = select_best_compatible_pack(
            model=model,
            device=device,
            llamacpp_commit=llamacpp_commit,
        )
        if not pack_id:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            return proc.returncode, {
                "used_pack": False,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "fallback": False,
            }

    pack_dir = PACKS_CACHE_DIR / pack_id
    resources_dir = pack_dir / "resources"
    manifest = load_manifest(pack_dir)
    ok, reason = manifest_compatible(
        manifest,
        device=device,
        model=model,
        llamacpp_commit=llamacpp_commit,
    )
    if not ok:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, {
            "used_pack": False,
            "compatibility_error": reason,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "fallback": False,
        }

    runtime_args, runtime_env = split_runtime_args_and_env(manifest.get("runtime_args", []))
    selected_rule = _select_dispatch_rule(manifest, model)
    tuned_cli, temp_cli_root = _prepare_cli_for_source_compile(llama_cli, resources_dir)
    tuned_cmd = [str(tuned_cli), "-m", str(model.path)] + runtime_args + list(extra_args)

    env = dict(os.environ)
    env.update(runtime_env)
    if resources_dir.exists():
        env[GGML_METAL_RESOURCES_ENV] = str(resources_dir)
    if selected_rule:
        env["CGINS_DISPATCH_RULE_ID"] = str(selected_rule.get("rule_id", ""))

    try:
        proc = subprocess.run(tuned_cmd, capture_output=True, text=True, env=env)
    finally:
        if temp_cli_root is not None:
            shutil.rmtree(temp_cli_root, ignore_errors=True)
    if proc.returncode == 0:
        return 0, {
            "used_pack": True,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "fallback": False,
            "selected_dispatch_rule": selected_rule,
        }

    kernel_name = _extract_kernel_name(proc.stderr or "")
    update_fallback_rules(pack_dir, kernel_name=kernel_name, error=proc.stderr or "runtime failure")

    fallback = subprocess.run(cmd, capture_output=True, text=True)
    return fallback.returncode, {
        "used_pack": True,
        "fallback": True,
        "initial_return_code": proc.returncode,
        "stdout": fallback.stdout,
        "stderr": fallback.stderr,
        "selected_dispatch_rule": selected_rule,
    }
