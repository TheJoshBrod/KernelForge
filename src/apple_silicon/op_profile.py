from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .benchmark import metal_toolchain_fingerprint
from .compat import chip_family, get_llamacpp_commit
from .constants import current_cache_root
from .device_probe import probe_device
from .types import WorkloadProfile

def _run_capture(cmd: list[str], *, timeout_s: float = 120.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return 124, out, err + "\ncommand_timed_out"
    except Exception as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def resolve_test_backend_ops(llamacpp_root: Path) -> Path | None:
    candidates = [
        llamacpp_root / "build" / "bin" / "test-backend-ops",
        llamacpp_root / "bin" / "test-backend-ops",
        llamacpp_root / "build" / "tests" / "test-backend-ops",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_support_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean = (text or "").strip()
    if not clean:
        return rows
    reader = csv.DictReader(io.StringIO(clean))
    for row in reader:
        entry = {str(k or "").strip().lower(): str(v or "").strip() for k, v in row.items()}
        if entry:
            rows.append(entry)
    return rows


def _parse_discovered_backends(text: str) -> list[str]:
    names: list[str] = []
    for line in (text or "").splitlines():
        match = re.match(r"^\s*Backend\s+\d+/\d+:\s*(.*?)\s*$", line)
        if not match:
            continue
        name = str(match.group(1) or "").strip()
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


def _discover_backend_names(bin_path: Path) -> dict[str, Any]:
    cmd = [str(bin_path), "test", "-b", "__NO_SUCH_BACKEND__"]
    rc, out, err = _run_capture(cmd, timeout_s=60.0)
    text = "\n".join([out or "", err or ""]).strip()
    names = _parse_discovered_backends(text)
    return {
        "success": rc == 0 and bool(names),
        "reason": "" if (rc == 0 and names) else f"backend_discovery_failed_rc={rc}",
        "command": " ".join(shlex.quote(x) for x in cmd),
        "return_code": rc,
        "backend_names": names,
        "stdout": out,
        "stderr": err,
    }


def resolve_backend_filter(
    *,
    llamacpp_root: Path,
    requested_backend: str = "Metal",
) -> dict[str, Any]:
    requested = str(requested_backend or "").strip() or "Metal"
    bin_path = resolve_test_backend_ops(llamacpp_root)
    if bin_path is None:
        return {
            "success": False,
            "reason": "test-backend-ops binary not found",
            "requested_backend": requested,
            "resolved_backend": requested,
            "backend_names": [],
            "discovery_command": "",
            "discovery_rc": 127,
        }

    discovery = _discover_backend_names(bin_path)
    names: list[str] = list(discovery.get("backend_names") or [])

    resolved = requested
    if requested in names:
        resolved = requested
    else:
        requested_l = requested.lower()
        if requested_l in {"metal", "mtl", "gpu", "apple_metal"}:
            for candidate in names:
                if candidate.upper().startswith("MTL") or "metal" in candidate.lower():
                    resolved = candidate
                    break
        else:
            for candidate in names:
                if candidate.lower() == requested_l:
                    resolved = candidate
                    break

    return {
        "success": bool(names),
        "reason": str(discovery.get("reason") or ""),
        "requested_backend": requested,
        "resolved_backend": resolved,
        "backend_names": names,
        "discovery_command": str(discovery.get("command") or ""),
        "discovery_rc": int(discovery.get("return_code", 1)),
    }


def _backend_candidates(
    support_rows: list[dict[str, Any]],
    discovered_names: list[str] | None = None,
) -> list[str]:
    candidates: list[str] = []
    for backend in discovered_names or []:
        value = str(backend or "").strip()
        if not value:
            continue
        if "metal" in value.lower() or value.upper().startswith("MTL"):
            candidates.append(value)
    for row in support_rows:
        backend = str(
            row.get("backend")
            or row.get("backend_name")
            or row.get("backend id")
            or row.get("device")
            or ""
        ).strip()
        if not backend:
            continue
        if "metal" in backend.lower() or backend.upper().startswith("MTL"):
            if backend not in candidates:
                candidates.append(backend)

    for fallback in ("MTL0", "Metal", "metal"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _pick_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and str(row.get(key, "")).strip():
            return row.get(key)
    return None


def _time_ms(row: dict[str, Any]) -> float | None:
    ms = _pick_first(
        row,
        [
            "time_ms",
            "ms_per_run",
            "avg_ms",
            "elapsed_ms",
        ],
    )
    if ms is not None:
        return _as_float(ms)
    us = _pick_first(
        row,
        [
            "time_us",
            "time_us_per_run",
            "us_per_run",
            "avg_us",
            "elapsed_us",
        ],
    )
    value = _as_float(us)
    return None if value is None else value / 1000.0


def _canonical_rows(
    *,
    sql_rows: list[dict[str, Any]],
    rank_metric: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sql_rows:
        op_name = str(
            _pick_first(
                row,
                [
                    "op",
                    "op_name",
                    "operation",
                    "name",
                ],
            )
            or ""
        ).strip()
        backend = str(_pick_first(row, ["backend", "backend_name", "device"]) or "").strip()
        params = str(_pick_first(row, ["op_params", "params", "case", "test_case"]) or "").strip()
        time_ms = _time_ms(row)
        flops = _as_float(_pick_first(row, ["flops", "gflops", "flops_g", "flops_per_s"]))
        bandwidth = _as_float(
            _pick_first(row, ["bandwidth_gb_s", "gb_s", "bandwidth", "mem_bw_gb_s"])
        )
        if not op_name:
            continue
        rows.append(
            {
                "op": op_name.lower(),
                "backend": backend,
                "op_params": params,
                "time_ms": time_ms,
                "flops": flops,
                "bandwidth_gb_s": bandwidth,
            }
        )

    metric = (rank_metric or "time").strip().lower()
    if metric in {"flops", "gflops"}:
        rows.sort(
            key=lambda r: float(r.get("flops")) if isinstance(r.get("flops"), (int, float)) else float("-inf"),
            reverse=True,
        )
    elif metric in {"bandwidth", "bandwidth_gb_s", "gb_s"}:
        rows.sort(
            key=lambda r: float(r.get("bandwidth_gb_s"))
            if isinstance(r.get("bandwidth_gb_s"), (int, float))
            else float("-inf"),
            reverse=True,
        )
    else:
        # Hotspot ranking: higher per-op time first.
        rows.sort(
            key=lambda r: float(r.get("time_ms")) if isinstance(r.get("time_ms"), (int, float)) else float("-inf"),
            reverse=True,
        )
    return rows[: max(1, int(top_k))]


def compare_stage1_op_profiles(
    *,
    baseline_ops: list[dict[str, Any]],
    candidate_ops: list[dict[str, Any]],
    sample_limit: int = 5,
) -> dict[str, Any]:
    compare_key = "op+op_params+backend"

    def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
        op = str(row.get("op", "")).strip().lower()
        params = str(row.get("op_params", "")).strip().lower()
        backend = str(row.get("backend", "")).strip().lower()
        return op, params, backend

    base_time: dict[tuple[str, str, str], float] = {}
    cand_time: dict[tuple[str, str, str], float] = {}

    for row in list(baseline_ops or []):
        t = _as_float(row.get("time_ms"))
        if t is None:
            continue
        key = _row_key(row)
        if not key[0]:
            continue
        base_time[key] = float(t)

    for row in list(candidate_ops or []):
        t = _as_float(row.get("time_ms"))
        if t is None:
            continue
        key = _row_key(row)
        if not key[0]:
            continue
        cand_time[key] = float(t)

    common_keys = sorted(set(base_time.keys()) & set(cand_time.keys()))
    common_rows = int(len(common_keys))
    if common_rows <= 0:
        return {
            "status": "op_perf_uncomparable",
            "compare_key": compare_key,
            "delta_pct": None,
            "common_rows": 0,
            "baseline_total_ms": None,
            "candidate_total_ms": None,
            "common_keys_sample": [],
        }

    baseline_total_ms = float(sum(base_time[k] for k in common_keys))
    candidate_total_ms = float(sum(cand_time[k] for k in common_keys))
    if baseline_total_ms <= 0:
        return {
            "status": "invalid_baseline_total",
            "compare_key": compare_key,
            "delta_pct": None,
            "common_rows": common_rows,
            "baseline_total_ms": baseline_total_ms,
            "candidate_total_ms": candidate_total_ms,
            "common_keys_sample": [
                {"op": k[0], "op_params": k[1], "backend": k[2]}
                for k in common_keys[: max(1, int(sample_limit))]
            ],
        }

    delta_pct = ((baseline_total_ms - candidate_total_ms) / baseline_total_ms) * 100.0
    return {
        "status": "ok",
        "compare_key": compare_key,
        "delta_pct": float(delta_pct),
        "common_rows": common_rows,
        "baseline_total_ms": baseline_total_ms,
        "candidate_total_ms": candidate_total_ms,
        "common_keys_sample": [
            {"op": k[0], "op_params": k[1], "backend": k[2]}
            for k in common_keys[: max(1, int(sample_limit))]
        ],
    }


def _load_sql_rows(sql_text: str, sqlite_path: Path) -> tuple[list[dict[str, Any]], str, list[str], str]:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.executescript(sql_text)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if not tables:
            return [], "", [], "no_sql_tables"

        table_name = ""
        for candidate in ("test_backend_ops", "test_backend_perf", "test_backend_ops_perf"):
            if candidate in tables:
                table_name = candidate
                break
        if not table_name:
            for name in tables:
                if "backend" in name and "ops" in name:
                    table_name = name
                    break
        if not table_name:
            table_name = tables[0]

        columns = [str(r[1]) for r in conn.execute(f"PRAGMA table_info('{table_name}')")]
        cursor = conn.execute(f"SELECT * FROM \"{table_name}\"")
        all_rows: list[dict[str, Any]] = []
        for values in cursor.fetchall():
            row = {columns[idx]: values[idx] for idx in range(min(len(columns), len(values)))}
            all_rows.append(row)
        return all_rows, table_name, columns, ""
    except Exception as exc:
        return [], "", [], f"sql_ingest_error:{exc}"
    finally:
        conn.close()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_op_perf_cache_key(
    *,
    llamacpp_commit: str,
    backend_resolved: str,
    chip_family: str,
    macos_version: str,
    toolchain_fingerprint: str,
    op_filter: str,
    rank_metric: str,
    profile_name: str,
    ctx: int,
    timeout_sec: float = 90.0,
    perf_case_limit: int = 64,
    perf_case_seed: int = 0,
    perf_warmup_iters: int = 1,
    perf_bench_iters: int = 3,
    candidate_hash: str = "",
) -> str:
    payload = {
        "llamacpp_commit": (llamacpp_commit or "").strip(),
        "backend_resolved": (backend_resolved or "").strip().lower(),
        "chip_family": (chip_family or "").strip().lower(),
        "macos_version": (macos_version or "").strip().lower(),
        "toolchain_fingerprint": (toolchain_fingerprint or "").strip().lower(),
        "op_filter": (op_filter or "").strip().upper(),
        "rank_metric": (rank_metric or "").strip().lower(),
        "profile_name": (profile_name or "").strip().lower(),
        "ctx": int(ctx),
        "timeout_sec": float(timeout_sec),
        "perf_case_limit": int(max(0, perf_case_limit)),
        "perf_case_seed": int(max(0, perf_case_seed)),
        "perf_warmup_iters": int(max(0, perf_warmup_iters)),
        "perf_bench_iters": int(max(0, perf_bench_iters)),
        "candidate_hash": (candidate_hash or "").strip().lower(),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _op_perf_cache_root() -> Path:
    return (current_cache_root() / "op_perf").expanduser().resolve()


def load_cached_op_perf(cache_dir: Path, *, min_rows: int = 1) -> dict[str, Any]:
    status_path = cache_dir / "status.json"
    ranked_path = cache_dir / "ranked_ops.json"
    if not status_path.exists() or not ranked_path.exists():
        return {"hit": False, "reason": "cache_missing"}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"hit": False, "reason": f"cache_parse_error:{exc}"}
    rows = list(ranked.get("ops") or [])
    status_value = str(status.get("status", "")).strip().lower()
    rows_emitted = int(status.get("rows_emitted") or len(rows))
    if status_value not in {"ok", "cache_hit_ok"}:
        return {"hit": False, "reason": f"cache_status_{status_value or 'invalid'}"}
    if rows_emitted < max(1, int(min_rows)):
        return {"hit": False, "reason": f"cache_rows_below_min:{rows_emitted}"}
    return {
        "hit": True,
        "status": status,
        "ranked": ranked,
    }


def save_cached_op_perf(
    cache_dir: Path,
    *,
    status_payload: dict[str, Any],
    ranked_ops: list[dict[str, Any]],
    sql_text: str,
    support_csv: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "op_perf.sql").write_text(sql_text or "", encoding="utf-8")
    (cache_dir / "op_support.csv").write_text(support_csv or "", encoding="utf-8")
    (cache_dir / "ranked_ops.json").write_text(
        json.dumps({"ops": ranked_ops}, indent=2),
        encoding="utf-8",
    )
    (cache_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )


def build_op_correctness_cache_key(
    *,
    llamacpp_commit: str,
    backend_resolved: str,
    chip_family: str,
    macos_version: str,
    toolchain_fingerprint: str,
    op_name: str,
    profile_name: str,
    ctx: int,
    case_limit: int,
    case_seed: int = 0,
    timeout_sec: float,
    candidate_hash: str = "",
) -> str:
    payload = {
        "llamacpp_commit": (llamacpp_commit or "").strip(),
        "backend_resolved": (backend_resolved or "").strip().lower(),
        "chip_family": (chip_family or "").strip().lower(),
        "macos_version": (macos_version or "").strip().lower(),
        "toolchain_fingerprint": (toolchain_fingerprint or "").strip().lower(),
        "op_name": (op_name or "").strip().upper(),
        "profile_name": (profile_name or "").strip().lower(),
        "ctx": int(ctx),
        "case_limit": int(max(0, case_limit)),
        "case_seed": int(max(0, case_seed)),
        "timeout_sec": float(timeout_sec),
        "candidate_hash": (candidate_hash or "").strip().lower(),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _op_correctness_cache_root() -> Path:
    return (current_cache_root() / "op_correctness").expanduser().resolve()


def load_cached_op_correctness(cache_dir: Path, *, min_rows: int = 1) -> dict[str, Any]:
    status_path = cache_dir / "status.json"
    rows_path = cache_dir / "rows.json"
    if not status_path.exists() or not rows_path.exists():
        return {"hit": False, "reason": "cache_missing"}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        rows = list(json.loads(rows_path.read_text(encoding="utf-8")).get("rows") or [])
    except Exception as exc:
        return {"hit": False, "reason": f"cache_parse_error:{exc}"}
    status_value = str(status.get("status", "")).strip().lower()
    rows_emitted = int(status.get("rows_emitted") or len(rows))
    if status_value not in {"ok", "cache_hit_ok"}:
        return {"hit": False, "reason": f"cache_status_{status_value or 'invalid'}"}
    if rows_emitted < max(1, int(min_rows)):
        return {"hit": False, "reason": f"cache_rows_below_min:{rows_emitted}"}
    return {"hit": True, "status": status, "rows": rows}


def save_cached_op_correctness(
    cache_dir: Path,
    *,
    status_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    sql_text: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "test_output.sql").write_text(sql_text or "", encoding="utf-8")
    (cache_dir / "rows.json").write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )
    (cache_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )


def _perf_limit_flags_supported(bin_path: Path) -> dict[str, bool]:
    rc, out, err = _run_capture([str(bin_path), "--help"], timeout_s=15.0)
    text = "\n".join([out or "", err or ""])
    return {
        "ok": rc == 0,
        "case_limit": "--case-limit" in text,
        "case_seed": "--case-seed" in text,
        "warmup_iters": "--warmup-iters" in text,
        "bench_iters": "--bench-iters" in text,
    }


def suggest_ggml_ops_from_hotspots(hotspot_ops: list[str], *, max_ops: int = 4) -> list[str]:
    mapping = {
        "mul_mat": "MUL_MAT",
        "mul_mat_id": "MUL_MAT_ID",
        "mul_mv": "MUL_MAT",
        "mul_mv_ext": "MUL_MAT",
        "mul_mv_q4_k": "MUL_MAT",
        "mul_mv_q5_k": "MUL_MAT",
        "softmax": "SOFT_MAX",
        "soft_max": "SOFT_MAX",
        "rms_norm": "RMS_NORM",
    }
    out: list[str] = []
    for item in hotspot_ops:
        key = str(item or "").strip().lower()
        if not key:
            continue
        op = mapping.get(key)
        if not op and key.isupper():
            op = key
        if not op:
            continue
        if op not in out:
            out.append(op)
        if len(out) >= max(1, int(max_ops)):
            break
    if not out:
        out = ["MUL_MAT", "SOFT_MAX", "RMS_NORM"]
    return out


def run_op_correctness_checks(
    *,
    llamacpp_root: Path,
    ops: list[str],
    resources_path: Path | None = None,
    backend: str = "Metal",
    max_ops: int = 3,
    timeout_s: float = 45.0,
    profile_name: str = "",
    ctx: int = 0,
    candidate_hash: str = "",
    cache_mode: str = "on",
    min_rows: int = 1,
    case_limit: int = 32,
    case_seed: int = 0,
    required: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    bin_path = resolve_test_backend_ops(llamacpp_root)
    if bin_path is None:
        return {
            "attempted": False,
            "success": False,
            "classification": "missing_test_backend_ops",
            "rows": [],
            "reason": "test-backend-ops binary not found",
        }

    cache_mode_norm = str(cache_mode or "on").strip().lower()
    if cache_mode_norm not in {"on", "off", "refresh"}:
        cache_mode_norm = "on"

    backend_resolution = resolve_backend_filter(
        llamacpp_root=llamacpp_root,
        requested_backend=backend,
    )
    resolved_backend = str(backend_resolution.get("resolved_backend") or backend).strip()

    flags = _perf_limit_flags_supported(bin_path)
    if int(case_limit) > 0 and not flags.get("case_limit"):
        return {
            "attempted": True,
            "success": False,
            "classification": "op_numeric_unsupported_flags",
            "status": "unsupported_flags",
            "reason": "test-backend-ops missing required flag: --case-limit",
            "rows": [],
            "backend_requested": backend,
            "backend": resolved_backend,
            "backend_resolution": backend_resolution,
        }
    if int(case_seed) > 0 and not flags.get("case_seed"):
        return {
            "attempted": True,
            "success": False,
            "classification": "op_numeric_unsupported_flags",
            "status": "unsupported_flags",
            "reason": "test-backend-ops missing required flag: --case-seed",
            "rows": [],
            "backend_requested": backend,
            "backend": resolved_backend,
            "backend_resolution": backend_resolution,
        }

    selected_ops = suggest_ggml_ops_from_hotspots(ops, max_ops=max_ops)
    env = None
    if resources_path is not None:
        env = dict(os.environ)
        env["GGML_METAL_PATH_RESOURCES"] = str(resources_path)
    if extra_env:
        if env is None:
            env = dict(os.environ)
        env.update(extra_env)

    device = probe_device()
    toolchain = metal_toolchain_fingerprint()
    llama_commit = get_llamacpp_commit(llamacpp_root)

    rows: list[dict[str, Any]] = []
    first_fail_status = ""
    first_fail_op = ""
    status_to_classification = {
        "timeout": "op_numeric_timeout",
        "backend_init_fail": "op_backend_init_fail",
        "zero_rows": "op_numeric_zero_rows",
        "parse_fail": "op_numeric_parse_fail",
        "unsupported_flags": "op_numeric_unsupported_flags",
        "unsupported_only": "op_numeric_no_supported_rows",
        "fail": "op_numeric_mismatch",
    }
    for op in selected_ops:
        cache_key = build_op_correctness_cache_key(
            llamacpp_commit=llama_commit,
            backend_resolved=resolved_backend,
            chip_family=chip_family(device.chip),
            macos_version=device.macos_version,
            toolchain_fingerprint=toolchain,
            op_name=op,
            profile_name=profile_name,
            ctx=int(ctx),
            case_limit=int(max(0, case_limit)),
            case_seed=int(max(0, case_seed)),
            timeout_sec=float(timeout_s),
            candidate_hash=candidate_hash,
        )
        cache_dir = _op_correctness_cache_root() / cache_key

        if cache_mode_norm == "on":
            cached = load_cached_op_correctness(cache_dir, min_rows=min_rows)
            if cached.get("hit"):
                status_raw = dict(cached.get("status") or {})
                row = {
                    "op": op,
                    "status": "cache_hit_ok",
                    "classification": "op_numeric_ok",
                    "reason": "",
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "rows_emitted": int(status_raw.get("rows_emitted") or len(cached.get("rows") or [])),
                    "command": str(status_raw.get("command") or ""),
                    "table": str(status_raw.get("table") or "test_backend_ops"),
                    "sqlite_path": str(cache_dir / "test_output.sqlite"),
                    "sql_path": str(cache_dir / "test_output.sql"),
                    "return_code": int(status_raw.get("return_code") or 0),
                    "stdout": "",
                    "stderr": "",
                }
                rows.append(row)
                continue

        cmd = [str(bin_path), "test", "-b", resolved_backend, "-o", op, "--output", "sql"]
        if int(case_limit) > 0:
            cmd.extend(["--case-limit", str(int(case_limit))])
        if int(case_seed) > 0:
            cmd.extend(["--case-seed", str(int(case_seed))])

        start_t = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(1.0, float(timeout_s)),
                env=env,
            )
            rc = int(proc.returncode)
            out = proc.stdout or ""
            err = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            out = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            err = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
            err = f"{err}\ncommand_timed_out".strip()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        status = "ok"
        reason = ""
        table = ""
        columns: list[str] = []
        sql_rows: list[dict[str, Any]] = []
        sqlite_path = cache_dir / "test_output.sqlite"
        sql_path = cache_dir / "test_output.sql"
        rows_emitted = 0

        if timed_out:
            status = "timeout"
            reason = "op_test_timeout"
        elif rc != 0:
            low = (out + "\n" + err).lower()
            if "failed to create command queue" in low or "failed to allocate context" in low:
                status = "backend_init_fail"
                reason = "metal_backend_init_failed"
            else:
                status = "fail"
                reason = f"op_test_failed_rc:{rc}"
        elif not out.strip():
            status = "zero_rows"
            reason = "empty_test_sql"
        else:
            sql_path.parent.mkdir(parents=True, exist_ok=True)
            sql_path.write_text(out, encoding="utf-8")
            sql_rows, table, columns, sql_err = _load_sql_rows(out, sqlite_path)
            if sql_err:
                status = "parse_fail"
                reason = sql_err
            else:
                relevant = [
                    r
                    for r in sql_rows
                    if str(r.get("test_mode", "")).strip().lower() == "test"
                    and str(r.get("op_name", "")).strip().upper() == op.strip().upper()
                ]
                supported_rows = [
                    r
                    for r in relevant
                    if str(r.get("supported", "")).strip().lower() in {"1", "true", "t", "yes"}
                ]
                rows_emitted = len(supported_rows)
                if rows_emitted < max(1, int(min_rows)):
                    status = "unsupported_only"
                    reason = f"supported_rows_below_min:{rows_emitted}"
                else:
                    failed = [
                        r
                        for r in supported_rows
                        if str(r.get("passed", "")).strip().lower() not in {"1", "true", "t", "yes"}
                    ]
                    if failed:
                        status = "fail"
                        reason = "op_numeric_mismatch"
                    else:
                        status = "ok"
                        reason = ""

        status_payload = {
            "status": status,
            "reason": reason,
            "rows_emitted": int(rows_emitted),
            "cache_hit": False,
            "backend_requested": backend,
            "backend_resolved": resolved_backend,
            "timeout_sec": float(timeout_s),
            "elapsed_ms": float(elapsed_ms),
            "case_seed": int(max(0, case_seed)),
            "command": " ".join(shlex.quote(x) for x in cmd),
            "return_code": int(rc),
            "table": table,
            "columns": columns,
            "op_name": op,
            "toolchain_fingerprint": toolchain,
            "cache_key": cache_key,
        }
        if cache_mode_norm in {"on", "refresh"}:
            save_cached_op_correctness(
                cache_dir,
                status_payload=status_payload,
                rows=sql_rows,
                sql_text=out,
            )

        classification = "op_numeric_ok" if status in {"ok", "cache_hit_ok"} else status_to_classification.get(status, "op_numeric_mismatch")
        row = {
            "op": op,
            "status": status,
            "classification": classification,
            "reason": reason,
            "cache_hit": False,
            "cache_key": cache_key,
            "rows_emitted": int(rows_emitted),
            "command": status_payload["command"],
            "table": table or "test_backend_ops",
            "sqlite_path": str(sqlite_path),
            "sql_path": str(sql_path),
            "return_code": int(rc),
            "stdout": out if status not in {"ok"} else "",
            "stderr": err,
        }
        rows.append(row)
        if status not in {"ok", "cache_hit_ok"} and not first_fail_status:
            first_fail_status = status
            first_fail_op = op

    success = first_fail_status == ""
    if not required and not success:
        # Exploratory mode still rejects this candidate but never stalls the overall run.
        pass

    classification = (
        "op_numeric_ok"
        if success
        else status_to_classification.get(first_fail_status, "op_numeric_mismatch")
    )
    reason = "" if success else f"failed_op={first_fail_op};status={first_fail_status}"
    return {
        "attempted": True,
        "success": success,
        "classification": classification,
        "status": "ok" if success else first_fail_status,
        "reason": reason,
        "rows": rows,
        "backend_requested": backend,
        "backend": resolved_backend,
        "backend_resolution": backend_resolution,
        "timeout_sec": float(timeout_s),
        "cache_mode": cache_mode_norm,
        "required": bool(required),
    }


def run_stage1_op_profile(
    *,
    llamacpp_root: Path,
    model_path: Path,
    profile: WorkloadProfile,
    top_k: int = 12,
    profiling_mode: str = "heuristic",
    rank_metric: str = "time",
    op_filter: str = "MUL_MAT",
    artifacts_dir: Path | None = None,
    artifact_prefix: str = "",
    timeout_sec: float = 90.0,
    cache_mode: str = "on",
    min_rows: int = 1,
    resources_path: Path | None = None,
    candidate_hash: str = "",
    extra_env: dict[str, str] | None = None,
    perf_case_limit: int = 64,
    perf_case_seed: int = 0,
    perf_warmup_iters: int = 1,
    perf_bench_iters: int = 3,
) -> dict[str, Any]:
    profiling_mode_norm = str(profiling_mode or "").strip().lower()
    required = profiling_mode_norm == "op_perf_required"
    if profiling_mode_norm not in {"op_perf_required", "heuristic"}:
        profiling_mode_norm = "heuristic"
        required = False

    bin_path = resolve_test_backend_ops(llamacpp_root)
    if bin_path is None:
        return {
            "success": (not required),
            "profiling_mode": "none",
            "reason": "test-backend-ops binary not found",
            "ops": [],
            "hotspot_ops": [],
            "command": "",
            "required": required,
            "status": "missing_binary",
            "profiling_mode_effective": "heuristic_fallback" if not required else "op_perf_required",
        }

    backend_resolution = resolve_backend_filter(
        llamacpp_root=llamacpp_root,
        requested_backend="Metal",
    )
    backend_requested = str(backend_resolution.get("requested_backend") or "Metal")
    backend_resolved = str(backend_resolution.get("resolved_backend") or backend_requested).strip()
    if not backend_resolved:
        backend_resolved = backend_requested

    device = probe_device()
    toolchain = metal_toolchain_fingerprint()
    llama_commit = get_llamacpp_commit(llamacpp_root)
    cache_mode_norm = str(cache_mode or "on").strip().lower()
    if cache_mode_norm not in {"on", "off", "refresh"}:
        cache_mode_norm = "on"

    cache_key = build_op_perf_cache_key(
        llamacpp_commit=llama_commit,
        backend_resolved=backend_resolved,
        chip_family=chip_family(device.chip),
        macos_version=device.macos_version,
        toolchain_fingerprint=toolchain,
        op_filter=op_filter,
        rank_metric=rank_metric,
        profile_name=profile.name,
        ctx=profile.ctx,
        timeout_sec=float(timeout_sec),
        perf_case_limit=int(max(0, perf_case_limit)),
        perf_case_seed=int(max(0, perf_case_seed)),
        perf_warmup_iters=int(max(0, perf_warmup_iters)),
        perf_bench_iters=int(max(0, perf_bench_iters)),
        candidate_hash=candidate_hash,
    )
    cache_dir = _op_perf_cache_root() / cache_key

    if cache_mode_norm in {"on"}:
        cache_lookup_started = time.perf_counter()
        cached = load_cached_op_perf(cache_dir, min_rows=min_rows)
        if cached.get("hit"):
            cache_lookup_elapsed_ms = (time.perf_counter() - cache_lookup_started) * 1000.0
            ranked = dict(cached.get("ranked") or {})
            status_raw = dict(cached.get("status") or {})
            op_rows = list(ranked.get("ops") or [])
            hotspot_ops = [str(r.get("op", "")).strip() for r in op_rows if str(r.get("op", "")).strip()]
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                for src_name, dst_name in (
                    ("op_support.csv", f"{artifact_prefix}op_support.csv"),
                    ("op_perf.sql", f"{artifact_prefix}op_perf.sql"),
                    ("op_perf.sqlite", f"{artifact_prefix}op_perf.sqlite"),
                ):
                    src = cache_dir / src_name
                    if src.exists():
                        (artifacts_dir / dst_name).write_bytes(src.read_bytes())
            return {
                "success": True,
                "profiling_mode": "test-backend-ops-sql",
                "profiling_mode_effective": "test-backend-ops-sql",
                "status": "cache_hit_ok",
                "reason": "",
                "ops": op_rows[: max(1, int(top_k))],
                "hotspot_ops": hotspot_ops,
                "command": str(status_raw.get("command") or ""),
                "backend": backend_resolved,
                "backend_requested": backend_requested,
                "backend_resolution": backend_resolution,
                "rank_metric": rank_metric,
                "op_filter": str(status_raw.get("op_filter") or op_filter),
                "table": str(status_raw.get("table") or "test_backend_ops"),
                "columns": list(status_raw.get("columns") or []),
                "profile": profile.name,
                "ctx": int(profile.ctx),
                "support_command": str(status_raw.get("support_command") or ""),
                "support_rc": int(status_raw.get("support_rc") or 0),
                "support_stderr": str(status_raw.get("support_stderr") or ""),
                "support_rows_count": int(status_raw.get("support_rows_count") or 0),
                "backend_discovery": status_raw.get("backend_discovery") or {},
                "support_csv_path": str((cache_dir / "op_support.csv")),
                "sqlite_path": str((cache_dir / "op_perf.sqlite")),
                "perf_sql_path": str((cache_dir / "op_perf.sql")),
                "required": required,
                "model_path": str(model_path),
                "cache_hit": True,
                "cache_key": cache_key,
                "rows_emitted": int(status_raw.get("rows_emitted") or len(op_rows)),
                "elapsed_ms": float(cache_lookup_elapsed_ms),
                "timeout_sec": float(status_raw.get("timeout_sec") or timeout_sec),
                "toolchain_fingerprint": toolchain,
                "cached_source_elapsed_ms": float(status_raw.get("elapsed_ms") or 0.0),
            }

    perf_flag_support = _perf_limit_flags_supported(bin_path)
    require_perf_limits = any(
        int(v) > 0 for v in [perf_case_limit, perf_case_seed, perf_warmup_iters, perf_bench_iters]
    )
    if require_perf_limits:
        missing_flags: list[str] = []
        if int(perf_case_limit) > 0 and not perf_flag_support.get("case_limit"):
            missing_flags.append("--case-limit")
        if int(perf_case_seed) > 0 and not perf_flag_support.get("case_seed"):
            missing_flags.append("--case-seed")
        if int(perf_warmup_iters) > 0 and not perf_flag_support.get("warmup_iters"):
            missing_flags.append("--warmup-iters")
        if int(perf_bench_iters) > 0 and not perf_flag_support.get("bench_iters"):
            missing_flags.append("--bench-iters")
        if missing_flags:
            status = {
                "status": "unsupported_flags",
                "reason": "test-backend-ops missing required flags: " + ", ".join(missing_flags),
                "rows_emitted": 0,
                "cache_hit": False,
                "backend_requested": backend_requested,
                "backend_resolved": backend_resolved,
                "timeout_sec": float(timeout_sec),
                "command": f"{shlex.quote(str(bin_path))} --help",
                "support_csv_path": "",
                "sql_path": "",
                "sqlite_path": "",
                "table": "",
                "columns": [],
                "op_filter": op_filter,
                "rank_metric": rank_metric,
                "elapsed_ms": 0.0,
                "toolchain_fingerprint": toolchain,
                "cache_key": cache_key,
                "profiling_mode_effective": "heuristic_fallback" if not required else "op_perf_required",
            }
            if cache_mode_norm in {"on", "refresh"}:
                save_cached_op_perf(
                    cache_dir,
                    status_payload=status,
                    ranked_ops=[],
                    sql_text="",
                    support_csv="",
                )
            return {
                "success": (not required),
                "profiling_mode": "none" if required else "heuristic",
                "profiling_mode_effective": status["profiling_mode_effective"],
                "reason": str(status["reason"]),
                "status": "unsupported_flags",
                "ops": [],
                "hotspot_ops": [],
                "command": status["command"],
                "backend": backend_resolved,
                "backend_requested": backend_requested,
                "backend_resolution": backend_resolution,
                "rank_metric": rank_metric,
                "op_filter": op_filter,
                "table": "",
                "columns": [],
                "profile": profile.name,
                "ctx": int(profile.ctx),
                "support_command": "",
                "support_rc": 0,
                "support_stderr": "",
                "support_rows_count": 0,
                "backend_discovery": backend_resolution,
                "support_csv_path": "",
                "sqlite_path": "",
                "perf_sql_path": "",
                "required": required,
                "model_path": str(model_path),
                "cache_hit": False,
                "cache_key": cache_key,
                "rows_emitted": 0,
                "elapsed_ms": 0.0,
                "timeout_sec": float(timeout_sec),
                "toolchain_fingerprint": toolchain,
            }

    env = dict(os.environ)
    if resources_path is not None:
        env["GGML_METAL_PATH_RESOURCES"] = str(resources_path)
    if extra_env:
        env.update(extra_env)

    support_cmd = [str(bin_path), "support", "-b", backend_resolved, "--output", "csv"]
    rc_support, support_out, support_err = _run_capture(
        support_cmd,
        timeout_s=min(45.0, max(5.0, float(timeout_sec))),
    )
    support_rows = _parse_support_csv(support_out)
    discovery = _discover_backend_names(bin_path)
    discovered_names = list(discovery.get("backend_names") or [])

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        support_path = artifacts_dir / f"{artifact_prefix}op_support.csv"
        support_path.write_text(support_out or "", encoding="utf-8")
    else:
        support_path = cache_dir / "op_support.csv"

    sql_rows: list[dict[str, Any]] = []
    op_rows: list[dict[str, Any]] = []
    chosen_cmd: list[str] = []
    chosen_table = ""
    chosen_cols: list[str] = []
    reason = ""
    status = "zero_rows"
    perf_sql_path = (artifacts_dir / f"{artifact_prefix}op_perf.sql") if artifacts_dir is not None else (cache_dir / "op_perf.sql")
    sqlite_path = (artifacts_dir / f"{artifact_prefix}op_perf.sqlite") if artifacts_dir is not None else (cache_dir / "op_perf.sqlite")
    elapsed_ms = 0.0
    rows_emitted = 0

    perf_filters: list[str] = []
    primary_filter = op_filter.strip()
    if primary_filter:
        perf_filters.append(primary_filter)
        if primary_filter.upper() != "MUL_MAT":
            perf_filters.append("MUL_MAT")
    perf_filters.append("")
    chosen_filter = primary_filter

    backends = _backend_candidates(support_rows, discovered_names=discovered_names)
    if backend_resolved in backends:
        backends = [backend_resolved] + [b for b in backends if b != backend_resolved]

    for backend in backends:
        for perf_filter in perf_filters:
            cmd = [str(bin_path), "perf", "-b", backend, "--output", "sql"]
            if perf_filter:
                cmd.extend(["-o", perf_filter])
            if int(perf_case_limit) > 0:
                cmd.extend(["--case-limit", str(int(perf_case_limit))])
            if int(perf_case_seed) > 0:
                cmd.extend(["--case-seed", str(int(perf_case_seed))])
            if int(perf_warmup_iters) > 0:
                cmd.extend(["--warmup-iters", str(int(perf_warmup_iters))])
            if int(perf_bench_iters) > 0:
                cmd.extend(["--bench-iters", str(int(perf_bench_iters))])

            start_t = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(1.0, float(timeout_sec)),
                    env=env,
                )
                rc = int(proc.returncode)
                out = proc.stdout or ""
                err = proc.stderr or ""
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            except subprocess.TimeoutExpired as exc:
                rc = 124
                out = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
                err = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                status = "timeout"
                reason = "op_perf_timeout"
                chosen_cmd = cmd
                break

            chosen_cmd = cmd
            if rc != 0:
                low = (out + "\n" + err).lower()
                if "failed to create command queue" in low or "failed to allocate context" in low:
                    status = "backend_init_fail"
                    reason = "metal_backend_init_failed"
                else:
                    status = "parse_fail"
                    reason = f"perf_nonzero_exit:{rc}"
                continue

            if not out.strip():
                status = "zero_rows"
                reason = "empty_perf_sql"
                continue

            perf_sql_path.parent.mkdir(parents=True, exist_ok=True)
            perf_sql_path.write_text(out, encoding="utf-8")
            sql_rows, chosen_table, chosen_cols, sql_err = _load_sql_rows(out, sqlite_path)
            if sql_err:
                status = "parse_fail"
                reason = sql_err
                continue

            rows_emitted = len(sql_rows)
            if rows_emitted < max(1, int(min_rows)):
                status = "zero_rows"
                reason = f"rows_emitted_below_min:{rows_emitted}"
                continue

            op_rows = _canonical_rows(sql_rows=sql_rows, rank_metric=rank_metric, top_k=top_k)
            if not op_rows:
                status = "zero_rows"
                reason = "sql_loaded_but_no_ranked_rows"
                continue

            chosen_filter = perf_filter
            backend_resolved = backend
            status = "ok"
            reason = ""
            break
        if status == "ok" or status == "timeout":
            break

    hotspot_ops = [str(r.get("op", "")).strip() for r in op_rows if str(r.get("op", "")).strip()]

    status_payload = {
        "status": status,
        "reason": reason,
        "rows_emitted": int(rows_emitted),
        "cache_hit": False,
        "backend_requested": backend_requested,
        "backend_resolved": backend_resolved,
        "timeout_sec": float(timeout_sec),
        "command": " ".join(shlex.quote(x) for x in chosen_cmd),
        "support_command": " ".join(shlex.quote(x) for x in support_cmd),
        "support_rc": int(rc_support),
        "support_stderr": support_err.strip(),
        "support_rows_count": len(support_rows),
        "support_csv_path": str(support_path),
        "sql_path": str(perf_sql_path),
        "sqlite_path": str(sqlite_path),
        "table": chosen_table,
        "columns": chosen_cols,
        "op_filter": chosen_filter,
        "rank_metric": rank_metric,
        "elapsed_ms": float(elapsed_ms),
        "perf_case_seed": int(max(0, perf_case_seed)),
        "toolchain_fingerprint": toolchain,
        "cache_key": cache_key,
        "backend_discovery": {
            "success": bool(discovery.get("success")),
            "reason": str(discovery.get("reason") or ""),
            "return_code": int(discovery.get("return_code", 1)),
            "command": str(discovery.get("command") or ""),
            "backend_names": discovered_names,
        },
        "profiling_mode_effective": (
            "test-backend-ops-sql"
            if status == "ok"
            else ("heuristic_fallback" if not required else "op_perf_required")
        ),
    }

    if cache_mode_norm in {"on", "refresh"}:
        save_cached_op_perf(
            cache_dir,
            status_payload=status_payload,
            ranked_ops=op_rows[: max(1, int(top_k))] if status == "ok" else [],
            sql_text=perf_sql_path.read_text(encoding="utf-8") if perf_sql_path.exists() else "",
            support_csv=support_out or "",
        )

    success = status == "ok"
    if not required and not success:
        # Exploratory mode can proceed with heuristic fallback.
        success = True

    return {
        "success": success,
        "profiling_mode": "test-backend-ops-sql" if status == "ok" else ("heuristic" if not required else "none"),
        "profiling_mode_effective": status_payload["profiling_mode_effective"],
        "status": status,
        "reason": reason or ("" if status == "ok" else f"sql_op_profile_failed:{status}"),
        "ops": op_rows[: max(1, int(top_k))],
        "hotspot_ops": hotspot_ops,
        "command": status_payload["command"],
        "backend": backend_resolved,
        "backend_requested": backend_requested,
        "backend_resolution": backend_resolution,
        "rank_metric": rank_metric,
        "op_filter": chosen_filter,
        "table": chosen_table,
        "columns": chosen_cols,
        "profile": profile.name,
        "ctx": int(profile.ctx),
        "support_command": status_payload["support_command"],
        "support_rc": int(rc_support),
        "support_stderr": support_err.strip(),
        "support_rows_count": len(support_rows),
        "backend_discovery": status_payload["backend_discovery"],
        "support_csv_path": str(support_path),
        "sqlite_path": str(sqlite_path),
        "perf_sql_path": str(perf_sql_path),
        "required": required,
        "model_path": str(model_path),
        "cache_hit": False,
        "cache_key": cache_key,
        "rows_emitted": int(rows_emitted),
        "elapsed_ms": float(elapsed_ms),
        "timeout_sec": float(timeout_sec),
        "toolchain_fingerprint": toolchain,
    }
