"""Bounded privacy-safe values for Portkey spans."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_BYTES = 16_000
MAX_TEXT_BYTES = 8_000
MAX_ITEMS = 50
MAX_DEPTH = 8
_SECRET_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
_USAGE_TOKEN_KEYS = frozenset(
    {
        "audio_tokens",
        "cached_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "reasoning_tokens",
        "total_tokens",
    }
)
_INLINE_SECRET = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|token)[\"']?)"
    r"(\s*[:=]\s*[\"']?\s*)([^\s,;&\"'}]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;&]+")
_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


def truncate_utf8(value: str, limit: int = MAX_TEXT_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    budget = max(0, limit - len(suffix.encode()))
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def safe_text(value: str) -> str:
    value = _BEARER.sub("Bearer <redacted>", value)
    value = _INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value
    )
    return truncate_utf8(_ADDRESS.sub("0x<redacted>", value))


def safe_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    module = getattr(value_type, "__module__", "")
    name = getattr(value_type, "__qualname__", None) or getattr(
        value_type, "__name__", "object"
    )
    result = f"{module}.{name}" if module and module != "builtins" else name
    return re.sub(r"[^A-Za-z0-9_.-]+", ".", result).strip(".")[:256] or "object"


def is_sensitive_key(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    if normalized in _USAGE_TOKEN_KEYS:
        return False
    return any(
        normalized == part or normalized.endswith(f"_{part}") or part in normalized
        for part in _SECRET_PARTS
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
    except Exception:  # noqa: BLE001
        return "<redacted-endpoint>"


def _key(value: Any) -> str:
    if value is None or isinstance(value, (str, bool, int)):
        return safe_text(str(value))[:256]
    return f"<{safe_type_name(value)}>"


def jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return {"type": safe_type_name(value), "truncated": "max_depth"}
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
        except Exception:  # noqa: BLE001
            return {"type": safe_type_name(value)}
        result: dict[str, Any] = {}
        for key, item in source[:MAX_ITEMS]:
            key_text = _key(key)
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
        except Exception:  # noqa: BLE001
            return {"type": safe_type_name(value)}
        values = [jsonable(item, depth=depth + 1) for item in source[:MAX_ITEMS]]
        if len(source) > MAX_ITEMS:
            return {"items": values, "truncated": True, "count": f">{MAX_ITEMS}"}
        return values
    module = getattr(type(value), "__module__", "")
    if module.startswith(("portkey_ai", "openai")):
        for method_name in ("model_dump", "to_dict"):
            try:
                method = getattr(value, method_name, None)
                if callable(method):
                    return jsonable(method(), depth=depth + 1)
            except Exception:  # noqa: BLE001, S112
                continue
    return {"type": safe_type_name(value)}


def json_dumps(value: Any) -> str:
    text = json.dumps(
        jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    original_bytes = len(text.encode("utf-8"))
    if original_bytes <= MAX_BYTES:
        return text
    low, high, result = 0, min(len(text), MAX_BYTES), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {
                "original_bytes": original_bytes,
                "preview": text[:middle],
                "truncated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(candidate.encode("utf-8")) <= MAX_BYTES:
            result, low = candidate, middle + 1
        else:
            high = middle - 1
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
        args = exc.args
    except Exception:  # noqa: BLE001
        args = ()
    for value in args:
        if isinstance(value, str):
            return safe_text(value)
    return safe_type_name(exc)


def exception_status(exc: BaseException) -> int:
    for owner, name in ((exc, "status_code"), (exc, "status")):
        try:
            value = getattr(owner, name, None)
        except Exception:  # noqa: BLE001
            value = None
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    try:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    except Exception:  # noqa: BLE001
        value = None
    return value if isinstance(value, int) and 400 <= value <= 599 else 500
