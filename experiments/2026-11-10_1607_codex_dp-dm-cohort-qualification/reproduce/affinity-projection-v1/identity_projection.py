"""Frozen allowlist projector reused from the verified affinity identity audit."""

import json
from collections.abc import Iterable

FORBIDDEN_FIELDS = (
    "bound",
    "mean",
    "prediction",
    "qualitative_measurement",
    "quantitative_measurement",
    "scale",
    "target",
    "value",
)


def is_forbidden_field(name: str) -> bool:
    token = name.casefold()
    return any(part in token for part in FORBIDDEN_FIELDS)


def validate_identity_fields(fields: Iterable[str]) -> None:
    bad = sorted(field for field in fields if is_forbidden_field(field))
    if bad:
        raise ValueError(f"outcome-like field entered identity allowlist: {bad}")


def _string_end(data: bytes, start: int) -> int:
    if data[start] != 0x22:
        raise ValueError("JSON string did not start with a quote")
    index = start + 1
    escaped = False
    while index < len(data):
        current = data[index]
        if escaped:
            escaped = False
        elif current == 0x5C:
            escaped = True
        elif current == 0x22:
            return index + 1
        index += 1
    raise ValueError("unterminated JSON string")


def _value_end(data: bytes, start: int) -> int:
    index = start
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    if index >= len(data):
        raise ValueError("missing JSON value")
    if data[index] == 0x22:
        return _string_end(data, index)
    if data[index] in (0x7B, 0x5B):
        stack = [data[index]]
        index += 1
        while index < len(data) and stack:
            current = data[index]
            if current == 0x22:
                index = _string_end(data, index)
                continue
            if current in (0x7B, 0x5B):
                stack.append(current)
            elif current == 0x7D:
                if stack[-1] != 0x7B:
                    raise ValueError("mismatched JSON object delimiter")
                stack.pop()
            elif current == 0x5D:
                if stack[-1] != 0x5B:
                    raise ValueError("mismatched JSON array delimiter")
                stack.pop()
            index += 1
        if stack:
            raise ValueError("unterminated JSON container")
        return index
    while index < len(data) and data[index] not in b",}\r\n":
        index += 1
    return index


def project_top_level_json(data: bytes, allowed: frozenset[str]) -> dict:
    """Decode only explicitly allowed top-level values from one JSON object."""
    validate_identity_fields(allowed)
    index = 0
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    if index >= len(data) or data[index] != 0x7B:
        raise ValueError("JSONL record is not an object")
    index += 1
    result = {}
    while True:
        while index < len(data) and data[index] in b" \t\r\n,":
            index += 1
        if index < len(data) and data[index] == 0x7D:
            return result
        key_end = _string_end(data, index)
        key = json.loads(data[index:key_end].decode("utf-8"))
        index = key_end
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index >= len(data) or data[index] != 0x3A:
            raise ValueError("JSON object key lacked colon")
        index += 1
        value_start = index
        value_end = _value_end(data, value_start)
        if key in allowed:
            result[key] = json.loads(data[value_start:value_end].decode("utf-8"))
        index = value_end
