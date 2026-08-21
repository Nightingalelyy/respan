"""Bounded, privacy-safe serialization for OpenInference attributes."""

from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_ATTRIBUTE_CHARS = 16_000
MAX_LABEL_CHARS = 512
_MAX_DEPTH = 8
_MAX_ITEMS = 1_000
_REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credentials",
    "password",
    "private-key",
    "private_key",
    "refresh-token",
    "refresh_token",
    "secret",
    "token",
    "x-api-key",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "-api-key",
    "_password",
    "-password",
    "_secret",
    "-secret",
    "_access_token",
    "-access-token",
    "_refresh_token",
    "-refresh-token",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")
_BASIC_AUTH_RE = re.compile(r"(?i)\bbasic\s+[a-z0-9+/=]{8,}")
_API_KEY_RE = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b("
    r"api[-_]?key|authorization|password|passwd|secret|"
    r"(?:access|refresh|session|auth)?[-_]?token"
    r")(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def sanitize_text(value: str) -> str:
    """Redact common credential shapes without stringifying arbitrary objects."""
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    value = _BASIC_AUTH_RE.sub("Basic [REDACTED]", value)
    value = _API_KEY_RE.sub(_REDACTED, value)
    return _ASSIGNMENT_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        value,
    )


def bounded_text(value: Any, *, max_chars: int = MAX_LABEL_CHARS) -> str:
    """Return a redacted bounded scalar without invoking user string hooks."""
    if not isinstance(value, str):
        return f"[UNSUPPORTED:{type(value).__name__}]"
    sanitized = sanitize_text(value)
    if len(sanitized) <= max_chars:
        return sanitized
    suffix = "...[TRUNCATED]"
    return f"{sanitized[: max(0, max_chars - len(suffix))]}{suffix}"


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert supported values to JSON data without calling user ``repr``/``str``."""
    if depth > _MAX_DEPTH:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE_FLOAT]"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, bytes):
        return sanitize_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                result["__truncated_items__"] = True
                break
            key = raw_key if isinstance(raw_key, str) else type(raw_key).__name__
            result[key] = (
                _REDACTED
                if _is_sensitive_key(key)
                else to_jsonable(item, depth=depth + 1)
            )
        return result
    if isinstance(value, (list, tuple)):
        result = [to_jsonable(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            result.append("[TRUNCATED_ITEMS]")
        return result
    return f"[UNSUPPORTED:{type(value).__name__}]"


def parse_json(value: Any) -> Any:
    """Parse JSON strings and leave all other supported values unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def bounded_json(value: Any, *, max_chars: int = MAX_ATTRIBUTE_CHARS) -> str:
    """Return redacted valid JSON no longer than ``max_chars``."""
    normalized = to_jsonable(parse_json(value))
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(serialized) <= max_chars:
        return serialized

    low = 0
    high = len(serialized)
    best = json.dumps(
        {"preview": "", "truncated": True},
        separators=(",", ":"),
        sort_keys=True,
    )
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {"preview": serialized[:midpoint], "truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(candidate) <= max_chars:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def content_value(value: Any) -> str:
    """Keep scalar message text readable; JSON-encode structured content."""
    parsed = parse_json(value)
    if isinstance(parsed, str):
        text = sanitize_text(parsed)
        if len(text) <= MAX_ATTRIBUTE_CHARS:
            return text
    return bounded_json(parsed)
