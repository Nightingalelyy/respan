"""Bounded, privacy-safe serialization for Pydantic AI span attributes."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

MAX_ATTRIBUTE_BYTES = 16_000
MAX_COLLECTION_ITEMS = 50
MAX_DEPTH = 8
MAX_STRING_BYTES = 8_000
REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "apikey",
    "authorization",
    "authtoken",
    "clientsecret",
    "cookie",
    "credential",
    "credentials",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "secret",
    "sessioncookie",
    "sessiontoken",
    "token",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|token)[\"']?)"
    r"(\s*[:=]\s*)([\"']?)([^\s,;}\"']+)([\"']?)"
)


def _truncate_utf8(value: str, limit: int = MAX_STRING_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    budget = max(0, limit - len(suffix.encode("utf-8")))
    return f"{encoded[:budget].decode('utf-8', errors='ignore')}{suffix}"


def safe_text(value: Any, *, default: str = "") -> str:
    """Render a safe scalar without calling arbitrary ``str`` or ``repr`` hooks."""
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(REDACTED, text)
        text = _SECRET_ASSIGNMENT.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}{match.group(3)}"
                f"{REDACTED}{match.group(5)}"
            ),
            text,
        )
        return _truncate_utf8(text)
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return f"<{type(value).__name__}>"


def is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("apikey", "password", "secret", "token")
    )


def _key_text(key: Any) -> str:
    if isinstance(key, str):
        return safe_text(key)[:256]
    if key is None or isinstance(key, bool | int):
        return safe_text(key)[:256]
    if isinstance(key, float) and math.isfinite(key):
        return safe_text(key)[:256]
    return f"<{type(key).__name__}>"


def json_value(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Convert untrusted SDK data to bounded JSON without arbitrary conversions."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return {"length": len(value), "type": type(value).__name__}
    if depth >= MAX_DEPTH:
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
                rendered_key = _key_text(key)
                result[rendered_key] = (
                    REDACTED
                    if is_sensitive_key(key)
                    else json_value(item, depth=depth + 1, seen=active)
                )
            if len(items) > MAX_COLLECTION_ITEMS:
                result["__truncated_items__"] = True
            return result

        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            items = list(islice(iter(value), MAX_COLLECTION_ITEMS + 1))
            result = [
                json_value(item, depth=depth + 1, seen=active)
                for item in items[:MAX_COLLECTION_ITEMS]
            ]
            if len(items) > MAX_COLLECTION_ITEMS:
                result.append({"__truncated_items__": True})
            return result

        for method_name in ("model_dump", "to_dict", "dict"):
            try:
                method = getattr(value, method_name, None)
            except Exception:  # noqa: BLE001 - hostile SDK objects are summarized.
                method = None
            if not callable(method):
                continue
            try:
                converted = method()
            except Exception:  # noqa: BLE001,S112 - vendor conversion hooks are untrusted.
                continue
            if isinstance(converted, Mapping):
                return json_value(converted, depth=depth + 1, seen=active)

        return {"type": type(value).__name__}
    except Exception:  # noqa: BLE001 - serialization must never break the SDK call.
        return {"type": type(value).__name__, "unserializable": True}
    finally:
        active.discard(identity)


def json_string(value: Any) -> str:
    """Return valid JSON whose UTF-8 representation fits the attribute budget."""
    normalized = json_value(value)
    encoded = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded_bytes = len(encoded.encode("utf-8"))
    if encoded_bytes <= MAX_ATTRIBUTE_BYTES:
        return encoded

    low = 0
    high = min(len(encoded), MAX_ATTRIBUTE_BYTES)
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
        if len(candidate.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES:
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
