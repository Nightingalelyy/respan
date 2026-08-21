"""Bounded, redacting serializers used by the OpenAI instrumentation."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

MAX_ATTRIBUTE_LENGTH = 16_000
MAX_ERROR_LENGTH = 2_000
MAX_COLLECTION_ITEMS = 50
MAX_DEPTH = 6

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
)
_SENSITIVE_KEYS = (
    "access_token",
    "auth_token",
    "refresh_token",
    "token",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token|access[_-]?token|refresh[_-]?token)\b"
    r"([\"']?)(\s*[:=]\s*)(?![\"'])([^\s,;}]+)"
)
_QUOTED_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token|access[_-]?token|refresh[_-]?token)\b"
    r"(?P<key_quote>[\"']?)(?P<separator>\s*[:=]\s*)"
    r"(?P<value_quote>[\"'])(?P<value>[\s\S]*?)(?P=value_quote)"
)


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def _is_sensitive_key(key: Any) -> bool:
    lowered = str(key).lower().replace("-", "_")
    return (
        lowered in _SENSITIVE_KEYS
        or lowered.endswith("_token")
        or any(marker in lowered for marker in _SENSITIVE_KEY_PARTS)
    )


def redact_text(value: str, *, limit: int = MAX_ATTRIBUTE_LENGTH) -> str:
    redacted = _BEARER_RE.sub(f"Bearer {_REDACTED}", value)
    redacted = _QUOTED_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group(1)}{match.group('key_quote')}"
            f"{match.group('separator')}{match.group('value_quote')}"
            f"{_REDACTED}{match.group('value_quote')}"
        ),
        redacted,
    )
    redacted = _ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}{_REDACTED}",
        redacted,
    )
    return _truncate_utf8(redacted, limit)


def json_value(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _key: str | None = None,
) -> Any:
    """Convert a value without arbitrary ``repr``/``str`` calls."""
    if _key is not None and _is_sensitive_key(_key):
        return _REDACTED
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return {"type": type(value).__name__, "bytes": len(value)}
    if _depth >= MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}

    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return {"type": type(value).__name__, "cycle": True}
    seen.add(value_id)
    try:
        converted: Any
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            converted = {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            }
        elif callable(getattr(value, "model_dump", None)):
            try:
                converted = value.model_dump(exclude_none=True, mode="json")
            except TypeError:
                converted = value.model_dump()
        elif callable(getattr(value, "to_dict", None)):
            converted = value.to_dict()
        elif (
            isinstance(value, Mapping)
            or isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
        ):
            converted = value
        elif isinstance(value, type):
            return {"type": value.__name__}
        else:
            return {"type": type(value).__name__}

        if isinstance(converted, Mapping):
            result: dict[str, Any] = {}
            for index, (key, item) in enumerate(converted.items()):
                if index >= MAX_COLLECTION_ITEMS:
                    result["__truncated_items__"] = True
                    break
                key_text = (
                    str(key)
                    if isinstance(key, str | int | float | bool)
                    else type(key).__name__
                )
                result[key_text] = json_value(
                    item,
                    _depth=_depth + 1,
                    _seen=seen,
                    _key=key_text,
                )
            return result
        if isinstance(converted, Sequence) and not isinstance(converted, str | bytes):
            # Some SDK/user sequences are lazy or report an expensive/hostile
            # length. Consume only the retained items plus one lookahead item.
            iterator = iter(converted)
            result = []
            for _ in range(MAX_COLLECTION_ITEMS):
                try:
                    item = next(iterator)
                except StopIteration:
                    return result
                result.append(json_value(item, _depth=_depth + 1, _seen=seen))
            try:
                next(iterator)
            except StopIteration:
                return result
            result.append({"truncated_items": True})
            return result
        return json_value(converted, _depth=_depth + 1, _seen=seen)
    except Exception:  # noqa: BLE001 - serialization must never break user calls
        return {"type": type(value).__name__, "serialization_error": True}
    finally:
        seen.discard(value_id)


def json_string(value: Any, *, limit: int = MAX_ATTRIBUTE_LENGTH) -> str:
    encoded = json.dumps(
        json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) <= limit:
        return encoded

    # Keep the result valid JSON while bounding the final encoded length.
    low, high = 0, len(encoded)
    best = json.dumps({"preview": "", "truncated": True}, separators=(",", ":"))
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {"preview": encoded[:middle], "truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(candidate.encode("utf-8")) <= limit:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def error_message(exc: BaseException) -> str:
    """Render an exception without invoking a hostile ``__str__`` override."""
    try:
        message = getattr(exc, "message", None)
    except BaseException:  # noqa: BLE001 - never mask the provider error
        message = None
    if not isinstance(message, str):
        try:
            args = getattr(exc, "args", ())
            message = next((arg for arg in args if isinstance(arg, str)), "")
        except BaseException:  # noqa: BLE001 - never mask the provider error
            message = ""
    message = (
        redact_text(message, limit=MAX_ERROR_LENGTH) if message else type(exc).__name__
    )
    return _truncate_utf8(message, MAX_ERROR_LENGTH)
