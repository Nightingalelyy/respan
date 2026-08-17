"""Bounded and privacy-safe serialization for Pipecat spans."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_ATTRIBUTE_BYTES = 16_000
MAX_ITEMS = 50
MAX_DEPTH = 8
MAX_TEXT_BYTES = 8_000

_SENSITIVE_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "token",
)
_INLINE_SECRET = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|client[_-]?secret|password|secret|token)[\"']?)"
    r"(\s*[:=]\s*[\"']?\s*)([^\s,;&\"'}]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;&]+")
_MEMORY_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


def safe_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    module = getattr(value_type, "__module__", "")
    qualname = getattr(value_type, "__qualname__", None) or getattr(
        value_type, "__name__", "object"
    )
    candidate = f"{module}.{qualname}" if module and module != "builtins" else qualname
    stable = re.sub(r"[^A-Za-z0-9_.-]+", ".", candidate).strip(".")
    return (stable or "object")[:256]


def truncate_utf8(value: str, max_bytes: int = MAX_TEXT_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = "...[truncated]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def is_sensitive_key(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return any(
        normalized == part or normalized.endswith(f"_{part}") or part in normalized
        for part in _SENSITIVE_PARTS
    )


def sanitize_endpoint(value: str) -> str:
    try:
        candidate = value if "://" in value else f"//{value}"
        parsed = urlsplit(candidate)
        if not parsed.hostname:
            return "<redacted-endpoint>"
        port = f":{parsed.port}" if parsed.port is not None else ""
        scheme = parsed.scheme if "://" in value else ""
        result = urlunsplit((scheme, f"{parsed.hostname}{port}", parsed.path, "", ""))
        return result if scheme else result.removeprefix("//")
    except Exception:  # noqa: BLE001 - malformed endpoints are fully redacted
        return "<redacted-endpoint>"


def safe_text(value: str, *, endpoint: bool = False) -> str:
    if endpoint:
        value = sanitize_endpoint(value)
    value = _BEARER_TOKEN.sub("Bearer <redacted>", value)
    value = _INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value
    )
    value = _MEMORY_ADDRESS.sub("0x<redacted>", value)
    return truncate_utf8(value)


def _stable_key(value: Any) -> str:
    if value is None or isinstance(value, (str, bool, int)):
        return safe_text(str(value))[:256]
    if isinstance(value, float) and math.isfinite(value):
        return str(value)[:256]
    return f"<{safe_type_name(value)}>"


def _summary(value: Any, reason: str | None = None) -> dict[str, str]:
    result = {"type": safe_type_name(value)}
    if reason:
        result["truncated"] = reason
    return result


def jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return _summary(value, "max_depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        try:
            source = list(islice(value.items(), MAX_ITEMS + 1))
        except Exception:  # noqa: BLE001 - hostile mappings become stable summaries
            return _summary(value)
        result: dict[str, Any] = {}
        for key, item in source[:MAX_ITEMS]:
            key_text = _stable_key(key)
            result[key_text] = (
                "<redacted>"
                if is_sensitive_key(key_text)
                else jsonable(item, depth=depth + 1)
            )
        if len(source) > MAX_ITEMS:
            result["__truncated__"] = True
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            source = list(islice(iter(value), MAX_ITEMS + 1))
        except Exception:  # noqa: BLE001 - hostile sequences become stable summaries
            return _summary(value)
        items = [jsonable(item, depth=depth + 1) for item in source[:MAX_ITEMS]]
        if len(source) > MAX_ITEMS:
            return {"items": items, "truncated": True, "count": f">{MAX_ITEMS}"}
        return items
    return _summary(value)


def json_dumps(value: Any) -> str:
    text = json.dumps(
        jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    original_bytes = len(text.encode("utf-8"))
    if original_bytes <= MAX_ATTRIBUTE_BYTES:
        return text
    low, high = 0, min(len(text), MAX_ATTRIBUTE_BYTES)
    result = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {
                "original_bytes": original_bytes,
                "preview": text[:midpoint],
                "truncated": True,
            },
            allow_nan=False,
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
    except (TypeError, json.JSONDecodeError):
        return value


def exception_message(exc: BaseException) -> str:
    try:
        arguments = exc.args
    except Exception:  # noqa: BLE001 - hostile exceptions are summarized
        arguments = ()
    for argument in arguments:
        if isinstance(argument, str):
            return safe_text(argument)
        if argument is None or isinstance(argument, (bool, int)):
            return safe_text(str(argument))
        if isinstance(argument, float) and math.isfinite(argument):
            return safe_text(str(argument))
    return safe_type_name(exc)


def exception_status_code(exc: BaseException) -> int:
    for owner, attribute in ((exc, "status_code"), (exc, "status")):
        try:
            value = getattr(owner, attribute, None)
        except Exception:  # noqa: BLE001 - hostile status properties are ignored
            value = None
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    try:
        response = getattr(exc, "response", None)
    except Exception:  # noqa: BLE001 - hostile response properties are ignored
        response = None
    if response is not None:
        try:
            value = getattr(response, "status_code", None)
        except Exception:  # noqa: BLE001 - hostile status properties are ignored
            value = None
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    return 500
