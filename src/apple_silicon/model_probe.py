from __future__ import annotations

import hashlib
import io
import os
import re
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .constants import SUPPORTED_ARCHS
from .types import ModelProfile

GGUF_MAGIC = b"GGUF"

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

FILE_TYPE_MAP = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q2_K",
    10: "Q3_K_S",
    11: "Q3_K_M",
    12: "Q3_K_L",
    13: "Q4_K_S",
    14: "Q4_K_M",
    15: "Q5_K_S",
    16: "Q5_K_M",
    17: "Q6_K",
    18: "IQ2_XXS",
    19: "IQ2_XS",
    20: "Q2_K_S",
    21: "IQ3_XS",
    22: "IQ3_XXS",
    23: "IQ1_S",
}


class GGUFParseError(RuntimeError):
    pass


def _read_exact(buf: io.BufferedReader, n: int) -> bytes:
    data = buf.read(n)
    if data is None or len(data) != n:
        raise GGUFParseError("Unexpected EOF while parsing GGUF")
    return data


def _read_u32(buf: io.BufferedReader) -> int:
    return struct.unpack("<I", _read_exact(buf, 4))[0]


def _read_u64(buf: io.BufferedReader) -> int:
    return struct.unpack("<Q", _read_exact(buf, 8))[0]


def _read_i32(buf: io.BufferedReader) -> int:
    return struct.unpack("<i", _read_exact(buf, 4))[0]


def _read_i64(buf: io.BufferedReader) -> int:
    return struct.unpack("<q", _read_exact(buf, 8))[0]


def _read_f32(buf: io.BufferedReader) -> float:
    return struct.unpack("<f", _read_exact(buf, 4))[0]


def _read_f64(buf: io.BufferedReader) -> float:
    return struct.unpack("<d", _read_exact(buf, 8))[0]


def _read_bool(buf: io.BufferedReader) -> bool:
    return struct.unpack("<?", _read_exact(buf, 1))[0]


def _read_string(buf: io.BufferedReader) -> str:
    n = _read_u64(buf)
    raw = _read_exact(buf, n)
    return raw.decode("utf-8", errors="replace")


def _read_value(buf: io.BufferedReader, value_type: int) -> Any:
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", _read_exact(buf, 1))[0]
    if value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", _read_exact(buf, 1))[0]
    if value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(buf, 2))[0]
    if value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", _read_exact(buf, 2))[0]
    if value_type == GGUF_TYPE_UINT32:
        return _read_u32(buf)
    if value_type == GGUF_TYPE_INT32:
        return _read_i32(buf)
    if value_type == GGUF_TYPE_FLOAT32:
        return _read_f32(buf)
    if value_type == GGUF_TYPE_BOOL:
        return _read_bool(buf)
    if value_type == GGUF_TYPE_STRING:
        return _read_string(buf)
    if value_type == GGUF_TYPE_UINT64:
        return _read_u64(buf)
    if value_type == GGUF_TYPE_INT64:
        return _read_i64(buf)
    if value_type == GGUF_TYPE_FLOAT64:
        return _read_f64(buf)
    if value_type == GGUF_TYPE_ARRAY:
        arr_type = _read_u32(buf)
        length = _read_u64(buf)
        return [_read_value(buf, arr_type) for _ in range(length)]
    raise GGUFParseError(f"Unsupported GGUF value type: {value_type}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _guess_quant_from_name(path: Path) -> str:
    name = path.name.lower()
    match = re.search(r"(q\d+_[a-z0-9_]+)", name)
    if not match:
        return "unknown"
    return match.group(1).upper()


def _parse_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError("Not a GGUF file")
        version = _read_u32(f)
        if version < 2:
            raise GGUFParseError(f"Unsupported GGUF version: {version}")

        _tensor_count = _read_u64(f)
        kv_count = _read_u64(f)

        metadata: dict[str, Any] = {"gguf_version": version}
        for _ in range(kv_count):
            key = _read_string(f)
            value_type = _read_u32(f)
            value = _read_value(f, value_type)
            metadata[key] = value
        return metadata


def probe_model(model_path: str | os.PathLike[str]) -> ModelProfile:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    metadata = _parse_metadata(path)

    arch = str(metadata.get("general.architecture", "unknown")).lower()
    if arch == "qwen":
        arch = "qwen2"

    file_type_id: int | None = None
    raw_file_type = metadata.get("general.file_type")
    if isinstance(raw_file_type, int):
        file_type_id = raw_file_type

    quant = FILE_TYPE_MAP.get(file_type_id, _guess_quant_from_name(path))

    model_name = str(metadata.get("general.name") or path.stem)

    return ModelProfile(
        path=path,
        name=model_name,
        architecture=arch,
        quant=quant,
        file_type_id=file_type_id,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        metadata=metadata,
    )


def assert_supported_model(profile: ModelProfile) -> None:
    if profile.architecture not in SUPPORTED_ARCHS:
        raise RuntimeError(
            f"Unsupported GGUF architecture '{profile.architecture}'. "
            f"v1 supports Qwen/Llama text families only ({', '.join(sorted(SUPPORTED_ARCHS))})."
        )


def probe_model_dict(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    return asdict(probe_model(model_path))
