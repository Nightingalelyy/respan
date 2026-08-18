"""Bounded JSON helpers for smolagents telemetry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_BYTES = 16_000
_MAX_ITEMS = 50
_MAX_DEPTH = 8
_SENSITIVE_KEY = re.compile(
    r"(?:^|[._-])(api[_-]?key|authorization|password|secret|token)(?:$|[._-])"
    r"|(?:api[_-]?key|authorization|password|secret|token)$",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)((?:['\"]?)(?:[a-z0-9_-]*[_-])?"
    r"(?:api[_-]?key|authorization|password|secret|token)"
    r"(?:['\"]?)\s*[:=]\s*)(?:['\"]?)[^,;)\s}]+(?:['\"]?)"
)


def redact_text(value: str, *, limit: int = 4_000) -> str:
    normalized = "".join(ch if ch >= " " or ch in "\n\t" else " " for ch in value)
    normalized = _ASSIGNMENT_SECRET.sub(r"\1[REDACTED]", normalized)
    encoded = normalized.encode("utf-8")
    if len(encoded) <= limit:
        return normalized
    return encoded[: limit - 16].decode("utf-8", errors="ignore") + "...[truncated]"


def jsonable(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "length": len(value)}
    if depth >= _MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            iterator = iter(value.items())
            for index in range(_MAX_ITEMS + 1):
                try:
                    key, item = next(iterator)
                except StopIteration:
                    break
                if index >= _MAX_ITEMS:
                    result["__truncated__"] = True
                    break
                key_text = key if isinstance(key, str) else f"<{type(key).__name__}>"
                result[key_text[:128]] = (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(key_text)
                    else jsonable(item, depth=depth + 1)
                )
        except Exception:  # noqa: BLE001 - telemetry must fail open
            return {"type": type(value).__name__, "serialization_error": True}
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result = []
        try:
            iterator = iter(value)
            for index in range(_MAX_ITEMS + 1):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if index >= _MAX_ITEMS:
                    result.append({"truncated": True})
                    break
                result.append(jsonable(item, depth=depth + 1))
        except Exception:  # noqa: BLE001 - telemetry must fail open
            return {"type": type(value).__name__, "serialization_error": True}
        return result
    return {"type": type(value).__name__}


def json_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        encoded = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001 - telemetry must fail open
        encoded = json.dumps(
            {"type": type(value).__name__, "serialization_error": True},
            sort_keys=True,
        )
    if len(encoded.encode("utf-8")) <= _MAX_BYTES:
        return encoded
    preview = encoded.encode("utf-8")[: _MAX_BYTES - 80].decode(
        "utf-8", errors="ignore"
    )
    while True:
        bounded = json.dumps(
            {"preview": preview, "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(bounded.encode("utf-8")) <= _MAX_BYTES:
            return bounded
        preview = preview[:-64]
