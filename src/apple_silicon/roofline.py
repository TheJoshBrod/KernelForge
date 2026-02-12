from __future__ import annotations

from typing import Any

from .types import DeviceProfile

CHIP_BANDWIDTH_GB_S = {
    "m1": 68.0,
    "m1 pro": 200.0,
    "m1 max": 400.0,
    "m1 ultra": 800.0,
    "m2": 100.0,
    "m2 pro": 200.0,
    "m2 max": 400.0,
    "m2 ultra": 800.0,
    "m3": 100.0,
    "m3 pro": 150.0,
    "m3 max": 300.0,
    "m4": 120.0,
}


def _chip_bandwidth_gb_s(chip: str) -> float | None:
    chip_key = (chip or "").strip().lower().replace("apple ", "")
    if chip_key in CHIP_BANDWIDTH_GB_S:
        return CHIP_BANDWIDTH_GB_S[chip_key]
    # Prefix fallback (for variants like "m2 pro (10c)").
    for key, value in CHIP_BANDWIDTH_GB_S.items():
        if chip_key.startswith(key):
            return value
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def build_roofline_analysis(
    *,
    device: DeviceProfile,
    metrics_rows: list[dict[str, Any]],
    target_uplift_pct: float = 30.0,
) -> dict[str, Any]:
    bw_gb_s = _chip_bandwidth_gb_s(device.chip)
    rows: list[dict[str, Any]] = []
    if bw_gb_s is None:
        return {
            "success": False,
            "reason": "unknown_chip_bandwidth",
            "chip": device.chip,
            "target_uplift_pct": float(target_uplift_pct),
            "rows": [],
        }

    for row in metrics_rows:
        if bool(row.get("is_warmup")):
            continue
        decode_tps = _as_float(row.get("decode_tokens_per_sec"))
        if decode_tps is None or decode_tps <= 0:
            continue
        target_tps = decode_tps * (1.0 + float(target_uplift_pct) / 100.0)
        # Roofline inversion: if decode t/s approaches memory roofline,
        # the effective bytes/token must shrink to keep improving.
        bytes_per_token_now = (bw_gb_s * 1e9) / decode_tps
        bytes_per_token_for_target = (bw_gb_s * 1e9) / target_tps
        shrink_pct = (
            ((bytes_per_token_now - bytes_per_token_for_target) / bytes_per_token_now) * 100.0
            if bytes_per_token_now > 0
            else None
        )
        rows.append(
            {
                "model_id": row.get("model_id", ""),
                "profile": row.get("profile", ""),
                "arm_id": row.get("arm_id", ""),
                "decode_tokens_per_sec": decode_tps,
                "target_decode_tokens_per_sec": target_tps,
                "chip_bandwidth_gb_s": bw_gb_s,
                "bytes_per_token_roof_now": bytes_per_token_now,
                "bytes_per_token_roof_for_target": bytes_per_token_for_target,
                "required_bytes_per_token_reduction_pct": shrink_pct,
            }
        )

    return {
        "success": True,
        "chip": device.chip,
        "chip_bandwidth_gb_s": bw_gb_s,
        "target_uplift_pct": float(target_uplift_pct),
        "rows": rows,
    }

