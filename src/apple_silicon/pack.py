from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .compat import chip_family, load_manifest, manifest_compatible, os_minor, quant_family
from .constants import ACTIVE_PACKS_PATH, PACKS_CACHE_DIR
from .types import DeviceProfile, ModelProfile


DEFAULT_FALLBACK_RULES = {
    "disabled": False,
    "disabled_variants": [],
    "errors": [],
}


def _build_dispatch_rules(
    *,
    model_arch: str,
    quant: str,
    profile_mode: str,
    kernel_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for idx, (op_name, config) in enumerate((kernel_overrides or {}).items(), start=1):
        if not isinstance(config, dict):
            continue
        rules.append(
            {
                "rule_id": f"rule_{idx}",
                "model_arch": (model_arch or "").strip().lower(),
                "quant_family": quant_family(quant),
                "profile": (profile_mode or "both").strip().lower(),
                "shape_bucket": "default",
                "variant_id": str(config.get("variant_name", "")).strip() or op_name,
                "op_name": op_name,
                "priority": idx,
            }
        )
    return rules


def _safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pack_id(device: DeviceProfile, model: ModelProfile, profile_mode: str, gate_mode: str) -> str:
    payload = "|".join(
        [
            device.fingerprint,
            model.sha256,
            profile_mode,
            gate_mode,
            str(time.time()),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _copy_resource_payload(llamacpp_root: Path, pack_resources: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    pack_resources.mkdir(parents=True, exist_ok=True)

    candidates = [
        llamacpp_root / "build" / "bin" / "default.metallib",
        llamacpp_root / "bin" / "default.metallib",
        llamacpp_root / "ggml" / "src" / "ggml-metal" / "ggml-metal.metal",
        llamacpp_root / "ggml" / "src" / "ggml-metal" / "ggml-metal-impl.h",
        llamacpp_root / "ggml" / "src" / "ggml-common.h",
        llamacpp_root / "ggml" / "src" / "ggml-metal.metal",
    ]
    for src in candidates:
        if src.exists():
            dst = pack_resources / src.name
            shutil.copy2(src, dst)
            copied[src.name] = str(dst)

    # Keep required public artifact name.
    metallib_src = pack_resources / "default.metallib"
    if metallib_src.exists():
        shutil.copy2(metallib_src, pack_resources.parent / "kernels.metallib")
    else:
        # Placeholder keeps format stable even before custom metal code is emitted.
        placeholder = pack_resources.parent / "kernels.metallib"
        placeholder.write_bytes(b"")

    return copied


def _copy_resource_payload_from_dir(source_dir: Path, pack_resources: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    pack_resources.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        raise FileNotFoundError(f"Kernel resource source directory not found: {source_dir}")

    for item in source_dir.iterdir():
        if not item.is_file():
            continue
        dst = pack_resources / item.name
        shutil.copy2(item, dst)
        copied[item.name] = str(dst)

    metallib_src = pack_resources / "default.metallib"
    if metallib_src.exists():
        shutil.copy2(metallib_src, pack_resources.parent / "kernels.metallib")
    else:
        placeholder = pack_resources.parent / "kernels.metallib"
        placeholder.write_bytes(b"")
    return copied


def create_pack(
    *,
    llamacpp_root: Path,
    device: DeviceProfile,
    model: ModelProfile,
    profile_mode: str,
    gate_mode: str,
    cgins_version: str,
    llamacpp_commit: str,
    bench_before: dict[str, Any],
    bench_after: dict[str, Any],
    kernel_overrides: dict[str, Any],
    runtime_args: list[str] | None,
    strict_guardrails: dict[str, Any],
    resources_source_dir: Path | None = None,
    kernel_patch_metadata: dict[str, Any] | None = None,
    tuning_session: dict[str, Any] | None = None,
    reuse_policy: str = "machine",
    os_compat_override: str = "",
) -> tuple[str, Path, dict[str, Any]]:
    PACKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pack_id = _pack_id(device, model, profile_mode, gate_mode)
    pack_dir = PACKS_CACHE_DIR / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    resources_dir = pack_dir / "resources"
    if resources_source_dir is not None:
        copied_resources = _copy_resource_payload_from_dir(resources_source_dir, resources_dir)
    else:
        copied_resources = _copy_resource_payload(llamacpp_root, resources_dir)
    dispatch_rules = _build_dispatch_rules(
        model_arch=model.architecture,
        quant=model.quant,
        profile_mode=profile_mode,
        kernel_overrides=kernel_overrides,
    )
    compiler_provenance = {
        "toolchain_fingerprint": str((kernel_patch_metadata or {}).get("toolchain_fingerprint", "")),
        "pipeline_cache_key": str((tuning_session or {}).get("compile_record", {}).get("pipeline_cache_key", "")),
    }

    fallback_path = pack_dir / "fallback_rules.json"
    _safe_write_json(fallback_path, DEFAULT_FALLBACK_RULES)

    rp = (reuse_policy or "machine").strip().lower()
    if rp not in {"machine", "chip_family", "chip_family+os_minor"}:
        raise ValueError(
            f"Unsupported reuse policy '{reuse_policy}'. Use machine, chip_family, or chip_family+os_minor."
        )
    os_compat = (os_compat_override or os_minor(device.macos_version)).strip().lower()

    manifest = {
        "pack_id": pack_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cgins_version": cgins_version,
        "llamacpp_commit": llamacpp_commit,
        "macos_version": device.macos_version,
        "chip": device.chip,
        "metal_feature_set": device.metal_feature_set,
        "model_sha256": model.sha256,
        "model_arch": model.architecture,
        "quant": model.quant,
        "profiles": {
            "mode": profile_mode,
            "gate": gate_mode,
        },
        "kernel_overrides": kernel_overrides,
        "dispatch_rules": dispatch_rules,
        "runtime_args": runtime_args or [],
        "compatibility": {
            "reuse_policy": rp,
            "device_fingerprint": device.fingerprint,
            "chip_family": chip_family(device.chip),
            "os_compat": os_compat,
            "model_arch": (model.architecture or "").strip().lower(),
            "quant_family": quant_family(model.quant),
            "supported_arches": ["qwen", "llama"],
        },
        "strict_guardrails": strict_guardrails,
        "resources": copied_resources,
        "kernel_patch": kernel_patch_metadata or {},
        "tuning_session": tuning_session or {},
        "compiler_provenance": compiler_provenance,
    }

    _safe_write_json(pack_dir / "manifest.json", manifest)
    _safe_write_json(pack_dir / "bench_before.json", bench_before)
    _safe_write_json(pack_dir / "bench_after.json", bench_after)

    return pack_id, pack_dir, manifest


def _active_map() -> dict:
    return _read_json(ACTIVE_PACKS_PATH)


def _active_key(*, model_sha: str, device_fingerprint: str) -> str:
    return f"{model_sha}:{device_fingerprint}"


def set_active_pack(*, model_sha: str, device_fingerprint: str, pack_id: str) -> None:
    state = _active_map()
    state[_active_key(model_sha=model_sha, device_fingerprint=device_fingerprint)] = {
        "pack_id": pack_id,
        "updated_at": time.time(),
    }
    _safe_write_json(ACTIVE_PACKS_PATH, state)


def get_active_pack(*, model_sha: str, device_fingerprint: str) -> str:
    state = _active_map()
    item = state.get(_active_key(model_sha=model_sha, device_fingerprint=device_fingerprint), {})
    return str(item.get("pack_id", ""))


def disable_pack(*, model_sha: str, device_fingerprint: str) -> None:
    state = _active_map()
    key = _active_key(model_sha=model_sha, device_fingerprint=device_fingerprint)
    if key in state:
        del state[key]
        _safe_write_json(ACTIVE_PACKS_PATH, state)


def update_fallback_rules(pack_dir: Path, *, kernel_name: str | None, error: str) -> None:
    rules_path = pack_dir / "fallback_rules.json"
    rules = _read_json(rules_path) or dict(DEFAULT_FALLBACK_RULES)
    if kernel_name and kernel_name not in rules.get("disabled_variants", []):
        rules.setdefault("disabled_variants", []).append(kernel_name)
    rules.setdefault("errors", []).append({"ts": time.time(), "kernel": kernel_name or "", "error": error})
    if len(rules.get("errors", [])) >= 2:
        rules["disabled"] = True
    _safe_write_json(rules_path, rules)


def list_compatible_packs(
    *,
    model: ModelProfile,
    device: DeviceProfile,
    llamacpp_commit: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PACKS_CACHE_DIR.exists():
        return rows
    for item in PACKS_CACHE_DIR.iterdir():
        if not item.is_dir():
            continue
        manifest = load_manifest(item)
        if not manifest:
            continue
        ok, reason = manifest_compatible(
            manifest,
            device=device,
            model=model,
            llamacpp_commit=llamacpp_commit,
        )
        if not ok:
            continue
        score = 0.0
        try:
            score = float(((manifest.get("tuning_session") or {}).get("score")) or 0.0)
        except Exception:
            score = 0.0
        rows.append(
            {
                "pack_id": str(manifest.get("pack_id") or item.name),
                "pack_dir": str(item),
                "score": score,
                "created_at": str(manifest.get("created_at", "")),
                "manifest": manifest,
                "reason": reason,
            }
        )
    rows.sort(key=lambda r: (float(r.get("score", 0.0)), str(r.get("created_at", ""))), reverse=True)
    return rows


def select_best_compatible_pack(
    *,
    model: ModelProfile,
    device: DeviceProfile,
    llamacpp_commit: str,
) -> str:
    rows = list_compatible_packs(model=model, device=device, llamacpp_commit=llamacpp_commit)
    if not rows:
        return ""
    return str(rows[0].get("pack_id", ""))


def export_pack(pack_dir: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w") as zf:
        for name in [
            "kernels.metallib",
            "manifest.json",
            "bench_before.json",
            "bench_after.json",
            "fallback_rules.json",
        ]:
            path = pack_dir / name
            if path.exists():
                zf.write(path, name)
        resources = pack_dir / "resources"
        if resources.exists():
            for item in resources.rglob("*"):
                if item.is_file():
                    zf.write(item, str(item.relative_to(pack_dir)))
    return out_path
