"""Bounded, privacy-safe serialization for Semantic Kernel telemetry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

MAX_JSON_BYTES = 16_000
MAX_ITEMS = 50
MAX_DEPTH = 8
_SENSITIVE_KEY = re.compile(
    r"(?:^|[._-])(api[_-]?key|authorization|password|secret|token)(?:$|[._-])",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)((?:['\"]?)(?:api[_-]?key|authorization|password|secret|token)"
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
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "length": len(value)}
    if depth >= MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_ITEMS:
                result["__truncated__"] = True
                break
            key_text = key if isinstance(key, str) else f"<{type(key).__name__}>"
            result[redact_text(key_text, limit=128)] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(key_text)
                else jsonable(item, depth=depth + 1)
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result = []
        for index, item in enumerate(value):
            if index >= MAX_ITEMS:
                result.append({"truncated": True})
                break
            result.append(jsonable(item, depth=depth + 1))
        return result
    return {"type": type(value).__name__}


def json_string(value: Any) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
    if len(encoded.encode("utf-8")) <= MAX_JSON_BYTES:
        return encoded
    preview = encoded.encode("utf-8")[: MAX_JSON_BYTES - 80].decode(
        "utf-8", errors="ignore"
    )
    while True:
        bounded = json.dumps(
            {"preview": preview, "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(bounded.encode("utf-8")) <= MAX_JSON_BYTES:
            return bounded
        preview = preview[:-64]
