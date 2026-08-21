"""Small, defensive serializers for Mirascope messages and response values."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

_MAX_DEPTH = 6
_MAX_ITEMS = 50
_MAX_STRING_LENGTH = 8_000
_MAX_JSON_LENGTH = 16_000
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "set-cookie",
    "token",
}
_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)\b(api[-_ ]?key|authorization|cookie|password|refresh[-_ ]?token|"
        r"secret|token)([\"']?\s*[:=]\s*[\"']?)([^\s,;\"'}]+)"
    ),
)


def _bounded_string(value: str) -> str:
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return f"{value[:_MAX_STRING_LENGTH]}...[truncated]"


def safe_text(value: Any) -> str:
    """Return bounded, redacted text without invoking arbitrary object reprs."""
    if isinstance(value, str):
        text = value
    elif value is None:
        return ""
    elif isinstance(value, bool | int | float):
        text = str(value)
    elif isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    else:
        return f"<{type(value).__name__}>"
    text = _SENSITIVE_TEXT_PATTERNS[0].sub("Bearer [REDACTED]", text)
    text = _SENSITIVE_TEXT_PATTERNS[1].sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )
    return _bounded_string(text)


def safe_exception_text(exc: BaseException) -> str:
    """Render an exception message without invoking an overridden ``__str__``."""
    try:
        args = exc.args
    except Exception:  # noqa: BLE001 - custom exceptions may override attributes
        args = ()
    values = args[:4] if isinstance(args, tuple) else ()
    message = ": ".join(filter(None, (safe_text(value) for value in values)))
    return _bounded_string(message or type(exc).__name__)


def _safe_key(value: Any) -> str:
    if isinstance(value, str | bool | int | float) or value is None:
        return safe_text(value)
    return f"<{type(value).__name__}>"


def _is_sensitive_key(value: Any) -> bool:
    key = _safe_key(value).strip().lower()
    return key in _SENSITIVE_KEYS or key.endswith(
        ("_api_key", "_password", "_secret", "_token")
    )


def json_value(value: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if _depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}:max-depth>"
    seen = _seen if _seen is not None else set()
    if id(value) in seen:
        return "<cycle>"
    seen.add(id(value))
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            }
        elif callable(getattr(value, "model_dump", None)):
            value = value.model_dump()
        elif callable(getattr(value, "to_dict", None)):
            value = value.to_dict()
        elif isinstance(value, Mapping | Sequence) and not isinstance(
            value, str | bytes
        ):
            pass
        elif callable(value):
            return {
                "name": safe_text(getattr(value, "__name__", value.__class__.__name__)),
                "description": safe_text(getattr(value, "__doc__", "") or ""),
            }
        elif hasattr(value, "__dict__"):
            value = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        else:
            return f"<{type(value).__name__}>"

        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, item in islice(value.items(), _MAX_ITEMS):
                normalized_key = _safe_key(key)
                normalized[normalized_key] = (
                    _REDACTED
                    if _is_sensitive_key(normalized_key)
                    else json_value(item, _depth=_depth + 1, _seen=seen)
                )
            return normalized
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [
                json_value(item, _depth=_depth + 1, _seen=seen)
                for item in islice(value, _MAX_ITEMS)
            ]
        return value
    except Exception:  # noqa: BLE001 - serializer must tolerate arbitrary vendor values
        return f"<{type(value).__name__}:unserializable>"
    finally:
        seen.discard(id(value))


def json_string(value: Any) -> str:
    encoded = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True)
    if len(encoded) <= _MAX_JSON_LENGTH:
        return encoded
    return json.dumps(
        {
            "preview": encoded[: _MAX_JSON_LENGTH - 200],
            "truncated": True,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
