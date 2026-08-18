"""Bounded, privacy-safe serialization for Restate invocation data."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import islice
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_ATTRIBUTE_BYTES = 16_000
MAX_DEPTH = 8
MAX_ITEMS = 50
MAX_STRING_BYTES = 4_000
REDACTED = "[REDACTED]"

_SENSITIVE_SUFFIXES = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "sessiontoken",
    "token",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|token)[\"']?)"
    r"(\s*[:=]\s*)([\"']?)([^\s,;}\"']+)([\"']?)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


def _truncate_utf8(value: str, limit: int = MAX_STRING_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    budget = max(0, limit - len(suffix.encode("utf-8")))
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted-endpoint>"
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname
    if not hostname:
        return "<redacted-endpoint>"
    netloc = (
        f"[{hostname}]"
        if ":" in hostname and not hostname.startswith("[")
        else hostname
    )
    try:
        port = parsed.port
    except ValueError:
        return "<redacted-endpoint>"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def safe_text(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        if "://" in value:
            value = sanitize_url(value)
        value = _BEARER.sub(REDACTED, value)
        value = _SECRET_ASSIGNMENT.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}{match.group(3)}"
                f"{REDACTED}{match.group(5)}"
            ),
            value,
        )
        return _truncate_utf8(_ADDRESS.sub("0x<redacted>", value))
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and math.isfinite(float(value)):
        return str(float(value))
    return f"<{type(value).__name__}>"


def exception_message(exc: BaseException) -> str:
    try:
        arguments = exc.args
    except Exception:  # noqa: BLE001
        arguments = ()
    for argument in arguments:
        if isinstance(argument, str | bool | int | float):
            return safe_text(argument)
    return type(exc).__name__


def exception_status(exc: BaseException, *, default: int = 500) -> int:
    try:
        response = getattr(exc, "response", None)
    except Exception:  # noqa: BLE001
        response = None
    for candidate in (exc, response):
        for name in ("status_code", "status"):
            try:
                value = getattr(candidate, name, None)
            except Exception:  # noqa: BLE001
                value = None
            if isinstance(value, int) and 400 <= value <= 599:
                return value
    return default


def sensitive_key(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _key(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return safe_text(value)[:256]


def json_value(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, bytes | bytearray | memoryview):
        return {"length": len(value), "type": type(value).__name__}
    if isinstance(value, Enum):
        return json_value(value.value, depth=depth + 1, seen=seen)
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
            items = list(islice(value.items(), MAX_ITEMS + 1))
            for key, item in items[:MAX_ITEMS]:
                key_text = _key(key)
                result[key_text] = (
                    REDACTED
                    if sensitive_key(key)
                    else json_value(item, depth=depth + 1, seen=active)
                )
            if len(items) > MAX_ITEMS:
                result["__truncated_items__"] = True
            return result
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            items = list(islice(iter(value), MAX_ITEMS + 1))
            converted = [
                json_value(item, depth=depth + 1, seen=active)
                for item in items[:MAX_ITEMS]
            ]
            if len(items) > MAX_ITEMS:
                return {"items": converted, "truncated": True}
            return converted
        return {"type": type(value).__name__}
    except Exception:  # noqa: BLE001
        return {"type": type(value).__name__, "unserializable": True}
    finally:
        active.discard(identity)


def json_string(value: Any) -> str:
    encoded = json.dumps(
        json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    size = len(encoded.encode("utf-8"))
    if size <= MAX_ATTRIBUTE_BYTES:
        return encoded
    low, high, result = 0, len(encoded), ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {"original_bytes": size, "preview": encoded[:midpoint], "truncated": True},
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
