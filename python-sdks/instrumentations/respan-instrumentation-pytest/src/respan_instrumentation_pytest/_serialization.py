"""Bounded, privacy-safe JSON helpers for Pytest telemetry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

from respan_instrumentation_pytest._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_COLLECTION_ITEMS,
    MAX_SERIALIZATION_DEPTH,
)

REDACTED = "[REDACTED]"
_MAX_STRING_BYTES = 4_000
_SENSITIVE_KEY_SUFFIXES = ("apikey", "password", "secret", "token")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|token)[\"']?)"
    r"(\s*[:=]\s*)([\"']?)([^\s,;}\"']+)([\"']?)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")


def _truncate_utf8(value: str, limit: int = _MAX_STRING_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    budget = max(0, limit - len(suffix.encode("utf-8")))
    return f"{encoded[:budget].decode('utf-8', errors='ignore')}{suffix}"


def safe_text(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        value = _BEARER_TOKEN.sub(REDACTED, value)
        value = _SECRET_ASSIGNMENT.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}{match.group(3)}"
                f"{REDACTED}{match.group(5)}"
            ),
            value,
        )
        return _truncate_utf8(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return f"<{type(value).__name__}>"


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in {
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "privatekey",
    } or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def _key_text(key: Any) -> str:
    if isinstance(key, str):
        return safe_text(key)[:256]
    if key is None or isinstance(key, bool | int):
        return safe_text(key)[:256]
    if isinstance(key, float) and math.isfinite(key):
        return safe_text(key)[:256]
    return f"<{type(key).__name__}>"


def to_jsonable(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return {"length": len(value), "type": type(value).__name__}
    if isinstance(value, Path):
        return safe_text(value.as_posix())
    if depth >= MAX_SERIALIZATION_DEPTH:
        return {"truncated": "max_depth", "type": type(value).__name__}

    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        return "<cycle>"
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            items = list(islice(value.items(), MAX_COLLECTION_ITEMS + 1))
            for key, item in items[:MAX_COLLECTION_ITEMS]:
                key_text = _key_text(key)
                result[key_text] = (
                    REDACTED
                    if _is_sensitive_key(key)
                    else to_jsonable(item, depth=depth + 1, seen=active)
                )
            if len(items) > MAX_COLLECTION_ITEMS:
                result["__truncated_items__"] = True
            return result
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            items = list(islice(iter(value), MAX_COLLECTION_ITEMS + 1))
            result = [
                to_jsonable(item, depth=depth + 1, seen=active)
                for item in items[:MAX_COLLECTION_ITEMS]
            ]
            if len(items) > MAX_COLLECTION_ITEMS:
                result.append({"__truncated_items__": True})
            return result
        return {"type": type(value).__name__}
    except Exception:  # noqa: BLE001 - hostile test parameters must not break Pytest.
        return {"type": type(value).__name__, "unserializable": True}
    finally:
        active.discard(identity)


def json_dumps(value: Any, *, max_bytes: int = MAX_ATTRIBUTE_CHARS) -> str:
    encoded = json.dumps(
        to_jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded_bytes = len(encoded.encode("utf-8"))
    if encoded_bytes <= max_bytes:
        return encoded
    low = 0
    high = min(len(encoded), max_bytes)
    result = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {
                "original_bytes": encoded_bytes,
                "preview": encoded[:midpoint],
                "truncated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(candidate.encode("utf-8")) <= max_bytes:
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result
