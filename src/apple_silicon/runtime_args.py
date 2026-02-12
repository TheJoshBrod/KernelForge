from __future__ import annotations

from typing import Iterable

BOOL_FLAGS = {
}

FLASH_ATTN_FLAGS = {
    "-fa",
    "--flash-attn",
}

FLASH_ATTN_VALUES = {
    "on",
    "off",
    "auto",
}

VALUE_FLAGS = {
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "-ngl",
    "--n-gpu-layers",
    "-t",
    "--threads",
    "--threads-batch",
}

INTERNAL_VALUE_FLAGS = {
    "--cgins-long-vector-schedule",
}

INTERNAL_FLAG_VALUES = {
    "on",
    "off",
    "auto",
}


def _is_int_like(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    if value[0] in {"+", "-"}:
        value = value[1:]
    return value.isdigit()


def sanitize_runtime_args(raw_args: Iterable[str] | None) -> list[str]:
    if not raw_args:
        return []

    args = [str(v).strip() for v in raw_args if str(v).strip()]
    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token in FLASH_ATTN_FLAGS:
            value = ""
            if i + 1 < len(args):
                nxt = args[i + 1].strip().lower()
                if nxt in FLASH_ATTN_VALUES:
                    value = nxt
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            out.extend(["--flash-attn", value or "on"])
            continue
        if token in BOOL_FLAGS:
            out.append(token)
            i += 1
            continue
        if token in VALUE_FLAGS and i + 1 < len(args):
            value = args[i + 1]
            if _is_int_like(value):
                out.extend([token, value])
            i += 2
            continue
        if token in INTERNAL_VALUE_FLAGS:
            value = "on"
            if i + 1 < len(args):
                nxt = args[i + 1].strip().lower()
                if nxt in INTERNAL_FLAG_VALUES:
                    value = nxt
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            out.extend([token, value])
            continue
        i += 1
    return out


def split_runtime_args_and_env(raw_args: Iterable[str] | None) -> tuple[list[str], dict[str, str]]:
    args = sanitize_runtime_args(raw_args)
    cli_args: list[str] = []
    env: dict[str, str] = {}

    i = 0
    while i < len(args):
        token = args[i]
        if token == "--cgins-long-vector-schedule":
            value = "on"
            if i + 1 < len(args):
                nxt = args[i + 1].strip().lower()
                if nxt in INTERNAL_FLAG_VALUES:
                    value = nxt
                    i += 2
                else:
                    i += 1
            else:
                i += 1

            if value == "on":
                env["CGINS_MUL_MV_LONG_VECTOR"] = "1"
            elif value == "off":
                env["CGINS_MUL_MV_LONG_VECTOR"] = "0"
            else:
                env.pop("CGINS_MUL_MV_LONG_VECTOR", None)
            continue

        cli_args.append(token)
        i += 1

    return cli_args, env
