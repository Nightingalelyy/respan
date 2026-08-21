"""Bounded, privacy-safe serialization for Qdrant calls and results."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from itertools import islice
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from respan_instrumentation_qdrant._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_BYTES,
    SENSITIVE_KEY_PARTS,
)

REDACTED = "[REDACTED]"
_VECTOR_KEY_PARTS = ("embedding", "vector")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|token)[\"']?)"
    r"(\s*[:=]\s*)([\"']?)([^\s,;}\"']+)([\"']?)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_MEMORY_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


def _truncate_utf8(value: str, limit: int = MAX_STRING_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    budget = max(0, limit - len(suffix.encode("utf-8")))
    return f"{encoded[:budget].decode('utf-8', errors='ignore')}{suffix}"


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted-endpoint>"
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname
    if not hostname:
        return "<redacted-endpoint>"
    netloc = hostname
    if ":" in hostname and not hostname.startswith("["):
        netloc = f"[{hostname}]"
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
            value = _sanitize_url(value)
        value = _BEARER_TOKEN.sub(REDACTED, value)
        value = _SECRET_ASSIGNMENT.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}{match.group(3)}"
                f"{REDACTED}{match.group(5)}"
            ),
            value,
        )
        value = _MEMORY_ADDRESS.sub("0x<redacted>", value)
        return _truncate_utf8(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and math.isfinite(float(value)):
        return str(float(value))
    return f"<{type(value).__name__}>"


def _key_text(key: Any) -> str:
    if isinstance(key, Enum):
        return _key_text(key.value)
    return safe_text(key)[:256]


def _sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(
        normalized.endswith(re.sub(r"[^a-z0-9]", "", part.lower()))
        for part in SENSITIVE_KEY_PARTS
    )


def _vector_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _VECTOR_KEY_PARTS)


def _trusted_length(value: Any) -> int | str:
    if type(value) in {list, tuple, dict, range}:
        return len(value)
    return f">{MAX_PREVIEW_ITEMS}"


def _is_numeric_vector(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return False
    try:
        sample = list(islice(iter(value), MAX_PREVIEW_ITEMS + 1))
    except Exception:  # noqa: BLE001 - hostile sequences become summaries.
        return False
    return bool(sample) and all(
        isinstance(item, Real) and not isinstance(item, bool) for item in sample
    )


def json_value(
    value: Any,
    *,
    depth: int = 0,
    vector_context: bool = False,
    seen: set[int] | None = None,
) -> Any:
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
        return json_value(
            value.value,
            depth=depth + 1,
            vector_context=vector_context,
            seen=seen,
        )
    if depth >= MAX_SERIALIZATION_DEPTH:
        return {"truncated": "max_depth", "type": type(value).__name__}

    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        return "<cycle>"
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            selected = list(islice(fields(value), MAX_PREVIEW_ITEMS + 1))
            converted = {
                field.name: getattr(value, field.name)
                for field in selected[:MAX_PREVIEW_ITEMS]
            }
            if len(selected) > MAX_PREVIEW_ITEMS:
                converted["__truncated_items__"] = True
            return json_value(converted, depth=depth + 1, seen=active)

        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            items = list(islice(value.items(), MAX_PREVIEW_ITEMS + 1))
            for key, item in items[:MAX_PREVIEW_ITEMS]:
                key_text = _key_text(key)
                result[key_text] = (
                    REDACTED
                    if _sensitive_key(key)
                    else json_value(
                        item,
                        depth=depth + 1,
                        vector_context=vector_context or _vector_key(key_text),
                        seen=active,
                    )
                )
            if len(items) > MAX_PREVIEW_ITEMS:
                result["__truncated_items__"] = True
            return result

        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            is_vector = vector_context or _is_numeric_vector(value)
            items = list(islice(iter(value), MAX_PREVIEW_ITEMS + 1))
            converted = [
                json_value(
                    item,
                    depth=depth + 1,
                    vector_context=is_vector,
                    seen=active,
                )
                for item in items[:MAX_PREVIEW_ITEMS]
            ]
            if len(items) > MAX_PREVIEW_ITEMS:
                return {
                    "count": _trusted_length(value),
                    "items": converted,
                    "kind": "vector" if is_vector else "sequence",
                    "truncated": True,
                }
            return converted

        for method_name in ("model_dump", "to_dict", "dict"):
            try:
                method = getattr(value, method_name, None)
            except Exception:  # noqa: BLE001 - hostile vendor objects are summarized.
                method = None
            if not callable(method):
                continue
            try:
                converted = method()
            except Exception:  # noqa: BLE001,S112 - conversion hooks are untrusted.
                continue
            if isinstance(converted, Mapping):
                return json_value(converted, depth=depth + 1, seen=active)
        return {"type": type(value).__name__}
    except Exception:  # noqa: BLE001 - serialization must not break Qdrant.
        return {"type": type(value).__name__, "unserializable": True}
    finally:
        active.discard(identity)


def json_dumps(value: Any) -> str:
    encoded = json.dumps(
        json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded_bytes = len(encoded.encode("utf-8"))
    if encoded_bytes <= MAX_ATTRIBUTE_CHARS:
        return encoded
    low = 0
    high = min(len(encoded), MAX_ATTRIBUTE_CHARS)
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
        if len(candidate.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS:
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result


def exception_message(exc: BaseException) -> str:
    try:
        arguments = exc.args
    except Exception:  # noqa: BLE001 - hostile exception attributes are ignored.
        arguments = ()
    for argument in arguments:
        if isinstance(argument, str | bool | int | float):
            return safe_text(argument)
    return type(exc).__name__
