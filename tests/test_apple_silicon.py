from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.apple_silicon import constants
from src.apple_silicon.benchmark import _parse_run_output, pipeline_cache_key, workload_profiles
from src.apple_silicon.compat import manifest_compatible
from src.apple_silicon.device_probe import probe_device
from src.apple_silicon.model_probe import assert_supported_model, probe_model
from src.apple_silicon.op_profile import (
    _canonical_rows,
    _load_sql_rows,
    _parse_discovered_backends,
    build_op_perf_cache_key,
    load_cached_op_perf,
    save_cached_op_perf,
)
from src.apple_silicon.pack import update_fallback_rules


def _write_gguf_string(buf, value: str) -> None:
    data = value.encode("utf-8")
    buf.write(struct.pack("<Q", len(data)))
    buf.write(data)


def _write_gguf_kv_string(buf, key: str, value: str) -> None:
    # key
    _write_gguf_string(buf, key)
    # type = string
    buf.write(struct.pack("<I", 8))
    _write_gguf_string(buf, value)


def _write_gguf_kv_u32(buf, key: str, value: int) -> None:
    _write_gguf_string(buf, key)
    buf.write(struct.pack("<I", 4))
    buf.write(struct.pack("<I", value))


def write_minimal_gguf(path: Path, *, arch: str = "qwen2", file_type: int = 14) -> None:
    with path.open("wb") as f:
        # magic + version
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        # tensor count, kv count
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 3))

        _write_gguf_kv_string(f, "general.architecture", arch)
        _write_gguf_kv_string(f, "general.name", "unit-test-model")
        _write_gguf_kv_u32(f, "general.file_type", file_type)


def test_device_probe_fingerprint_stable() -> None:
    first = probe_device()
    second = probe_device()
    assert first.fingerprint
    assert first.fingerprint == second.fingerprint
    assert first.platform == "darwin"


def test_model_probe_qwen_supported(tmp_path: Path) -> None:
    model_path = tmp_path / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    write_minimal_gguf(model_path, arch="qwen2", file_type=14)

    model = probe_model(model_path)
    assert model.architecture == "qwen2"
    assert model.quant == "Q4_K_M"
    assert_supported_model(model)


def test_model_probe_rejects_unsupported_arch(tmp_path: Path) -> None:
    model_path = tmp_path / "bad.gguf"
    write_minimal_gguf(model_path, arch="mamba", file_type=14)

    model = probe_model(model_path)
    with pytest.raises(RuntimeError):
        assert_supported_model(model)


def test_manifest_compat_mismatch_commit(tmp_path: Path) -> None:
    device = probe_device()
    model_path = tmp_path / "qwen.gguf"
    write_minimal_gguf(model_path, arch="qwen2", file_type=14)
    model = probe_model(model_path)

    manifest = {
        "llamacpp_commit": "deadbeef",
        "model_sha256": model.sha256,
        "compatibility": {"device_fingerprint": device.fingerprint},
    }
    ok, reason = manifest_compatible(
        manifest,
        device=device,
        model=model,
        llamacpp_commit="cafebabe",
    )
    assert not ok
    assert "commit" in reason.lower()


def test_manifest_compat_chip_family_reuse(tmp_path: Path) -> None:
    device = probe_device()
    model_path = tmp_path / "qwen.gguf"
    write_minimal_gguf(model_path, arch="qwen2", file_type=14)
    model = probe_model(model_path)

    manifest = {
        "llamacpp_commit": "cafebabe",
        "model_sha256": "different-model-sha",
        "compatibility": {
            "reuse_policy": "chip_family",
            "chip_family": "m2",
            "model_arch": "qwen2",
            "quant_family": "q4_k",
        },
    }
    ok, _reason = manifest_compatible(
        manifest,
        device=device,
        model=model,
        llamacpp_commit="cafebabe",
    )
    assert ok


def test_benchmark_parser_handles_valid_and_malformed() -> None:
    valid = """
llama_print_timings: prompt eval time = 125.00 ms / 100 tokens (800.00 tokens per second)
llama_print_timings: eval time = 250.00 ms / 50 runs (200.00 tokens per second)
llama_model_load: mem required = 1024.00 MiB
"""
    parsed = _parse_run_output(valid)
    assert parsed["prefill_tokens_per_sec"] == pytest.approx(800.0)
    assert parsed["decode_tokens_per_sec"] == pytest.approx(200.0)
    assert parsed["peak_memory_mib"] == pytest.approx(1024.0)

    malformed = "no timings here"
    parsed_bad = _parse_run_output(malformed)
    assert parsed_bad["prefill_tokens_per_sec"] is None
    assert parsed_bad["decode_tokens_per_sec"] is None


def test_fallback_rules_disable_after_repeated_errors(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    update_fallback_rules(pack_dir, kernel_name="k1", error="first")
    data1 = json.loads((pack_dir / "fallback_rules.json").read_text(encoding="utf-8"))
    assert not data1.get("disabled", False)
    assert "k1" in data1.get("disabled_variants", [])

    update_fallback_rules(pack_dir, kernel_name="k2", error="second")
    data2 = json.loads((pack_dir / "fallback_rules.json").read_text(encoding="utf-8"))
    assert data2.get("disabled", False)
    assert len(data2.get("errors", [])) >= 2


def test_pipeline_cache_key_deterministic() -> None:
    a = pipeline_cache_key(
        llamacpp_commit="abc",
        chip_family="m2",
        macos_version="14.6",
        source_hash="s1",
        candidate_hash="c1",
        toolchain_fingerprint="t1",
    )
    b = pipeline_cache_key(
        llamacpp_commit="abc",
        chip_family="m2",
        macos_version="14.6",
        source_hash="s1",
        candidate_hash="c1",
        toolchain_fingerprint="t1",
    )
    c = pipeline_cache_key(
        llamacpp_commit="abc",
        chip_family="m2",
        macos_version="14.6",
        source_hash="s1",
        candidate_hash="c2",
        toolchain_fingerprint="t1",
    )
    assert a == b
    assert a != c


def test_workload_profiles_support_long_smoke_and_claim() -> None:
    smoke = workload_profiles("long_smoke", "quick")
    claim = workload_profiles("long_claim", "quick")
    mixed = workload_profiles("chat,long_smoke", "quick")
    assert smoke and smoke[0].name == "long_smoke"
    assert claim and claim[0].name == "long_claim"
    assert [p.name for p in mixed] == ["chat", "long_smoke"]


def test_op_profile_sql_loader_and_canonical_ranking(tmp_path: Path) -> None:
    sql = """
CREATE TABLE test_backend_ops (
  backend TEXT,
  op TEXT,
  op_params TEXT,
  time_us_per_run REAL,
  flops REAL,
  bandwidth_gb_s REAL
);
INSERT INTO test_backend_ops VALUES ('MTL0', 'MUL_MAT', 'k=4096', 900.0, 12.0, 150.0);
INSERT INTO test_backend_ops VALUES ('MTL0', 'SOFT_MAX', 'k=4096', 300.0, 3.0, 90.0);
"""
    sql_rows, table_name, columns, err = _load_sql_rows(sql, tmp_path / "op_perf.sqlite")
    assert err == ""
    assert table_name == "test_backend_ops"
    assert "op" in columns
    assert len(sql_rows) == 2

    ranked = _canonical_rows(sql_rows=sql_rows, rank_metric="time", top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["op"] == "mul_mat"
    assert ranked[0]["time_ms"] == pytest.approx(0.9)


def test_parse_discovered_backends_extracts_exact_names() -> None:
    text = """
ggml_metal_device_init: GPU name:   MTL0
Testing 3 devices

Backend 1/3: MTL0
  Skipping
Backend 2/3: BLAS
  Skipping
Backend 3/3: CPU
  Skipping
"""
    names = _parse_discovered_backends(text)
    assert names == ["MTL0", "BLAS", "CPU"]


def test_op_perf_cache_key_deterministic() -> None:
    a = build_op_perf_cache_key(
        llamacpp_commit="abc123",
        backend_resolved="MTL0",
        chip_family="m2",
        macos_version="14.6",
        toolchain_fingerprint="toolchain-fp",
        op_filter="MUL_MAT",
        rank_metric="time",
        profile_name="chat",
        ctx=8192,
        candidate_hash="",
    )
    b = build_op_perf_cache_key(
        llamacpp_commit="abc123",
        backend_resolved="MTL0",
        chip_family="m2",
        macos_version="14.6",
        toolchain_fingerprint="toolchain-fp",
        op_filter="MUL_MAT",
        rank_metric="time",
        profile_name="chat",
        ctx=8192,
        candidate_hash="",
    )
    c = build_op_perf_cache_key(
        llamacpp_commit="abc123",
        backend_resolved="MTL0",
        chip_family="m2",
        macos_version="14.6",
        toolchain_fingerprint="toolchain-fp",
        op_filter="MUL_MAT",
        rank_metric="time",
        profile_name="chat",
        ctx=8192,
        candidate_hash="candidate-a",
    )
    assert a == b
    assert a != c


def test_op_perf_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "opperf_cache"
    save_cached_op_perf(
        cache_dir,
        status_payload={
            "status": "ok",
            "rows_emitted": 2,
            "command": "test-backend-ops perf -b MTL0 --output sql",
            "elapsed_ms": 123.4,
        },
        ranked_ops=[
            {"op": "mul_mat", "time_ms": 0.8, "flops": 12.0, "bandwidth_gb_s": 100.0},
            {"op": "soft_max", "time_ms": 0.3, "flops": 3.0, "bandwidth_gb_s": 80.0},
        ],
        sql_text="CREATE TABLE test_backend_ops(x INT);",
        support_csv="backend,op,supported\nMTL0,MUL_MAT,1\n",
    )
    hit = load_cached_op_perf(cache_dir, min_rows=1)
    assert hit.get("hit") is True
    status = hit.get("status") or {}
    ranked = hit.get("ranked") or {}
    assert status.get("status") == "ok"
    assert len(ranked.get("ops") or []) == 2


def test_op_perf_cache_invalid_status_misses(tmp_path: Path) -> None:
    cache_dir = tmp_path / "opperf_bad"
    save_cached_op_perf(
        cache_dir,
        status_payload={"status": "parse_fail", "rows_emitted": 0},
        ranked_ops=[],
        sql_text="",
        support_csv="",
    )
    miss = load_cached_op_perf(cache_dir, min_rows=1)
    assert miss.get("hit") is False
    assert "cache_status" in str(miss.get("reason", ""))


def test_cache_root_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = constants.current_cache_root()
    try:
        monkeypatch.delenv("CGINS_CACHE_ROOT", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        default_root = constants.configure_cache_root(None)
        assert default_root == (Path.home() / ".cache" / constants.CGINS_APP_NAME / "apple_silicon").resolve()

        xdg_root = tmp_path / "xdg-cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_root))
        xdg_resolved = constants.configure_cache_root(None)
        assert xdg_resolved == (xdg_root / constants.CGINS_APP_NAME / "apple_silicon").resolve()

        env_override = tmp_path / "env-cache"
        monkeypatch.setenv("CGINS_CACHE_ROOT", str(env_override))
        env_resolved = constants.configure_cache_root(None)
        assert env_resolved == env_override.resolve()

        cli_override = tmp_path / "cli-cache"
        cli_resolved = constants.configure_cache_root(cli_override)
        assert cli_resolved == cli_override.resolve()
    finally:
        constants.configure_cache_root(original)


def test_doctor_command_smoke() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "apple_silicon" / "cgins_as.py"),
            "doctor",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert "device" in data
