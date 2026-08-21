"""Bounded, privacy-safe serialization for Writer telemetry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

MAX_JSON_BYTES = 16_000
MAX_TEXT_BYTES = 4_000
MAX_ITEMS = 50
MAX_DEPTH = 8

_SENSITIVE_KEY = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|password|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)([a-z0-9_.-]*(?:api[_-]?key|authorization|cookie|password|secret|token))"
    r"\s*[:=]\s*([^\s,;]+)"
)
_QUOTED_SECRET = re.compile(
    r"""(?i)(["'][^"']*(?:api[_-]?key|authorization|cookie|password|secret|token)["']\s*:\s*)(["'])(.*?)\2"""
)
_AUTH_SECRET = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")


def safe_type_name(value: Any) -> str:
    return type(value).__name__[:120]


def redact_text(value: str) -> str:
    value = _AUTH_SECRET.sub(lambda match: f"{match.group(1)} [REDACTED]", value)
    value = _QUOTED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        value,
    )
    return _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)


def safe_text(value: Any, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, str):
        text = redact_text(value)
    elif value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int) or isinstance(value, float) and math.isfinite(value):
        text = str(value)
    else:
        text = f"<{safe_type_name(value)}>"
    if len(text.encode()) <= max_bytes:
        return text
    low, high, best = 0, len(text), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = f"{text[:middle]}…[truncated]"
        if len(candidate.encode()) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or "[truncated]"


def safe_exception_message(exc: BaseException) -> str:
    try:
        args = exc.args
    except BaseException:  # noqa: BLE001
        args = ()
    details = [
        safe_text(item, max_bytes=1_000)
        for item in args[:4]
        if item is None or isinstance(item, (str, bool, int, float))
    ]
    name = safe_type_name(exc)
    return safe_text(f"{name}: {'; '.join(filter(None, details))}" if details else name)


def provider_status_code(exc: BaseException) -> int:
    candidates: list[Any] = []
    for name in ("status_code", "status"):
        try:
            candidates.append(getattr(exc, name, None))
        except BaseException:  # noqa: BLE001, S112
            continue
    try:
        response = getattr(exc, "response", None)
    except BaseException:  # noqa: BLE001
        response = None
    if response is not None:
        for name in ("status_code", "status"):
            try:
                candidates.append(getattr(response, name, None))
            except BaseException:  # noqa: BLE001, S112
                continue
    for value in candidates:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 400 <= value <= 599
        ):
            return value
    return 500


def _safe_key(value: Any) -> str:
    if isinstance(value, str):
        return safe_text(value, max_bytes=256)
    if value is None or isinstance(value, (bool, int)):
        return json.dumps(value)
    return f"<{safe_type_name(value)}>"


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else safe_text(value)
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            length = len(value)
        except BaseException:  # noqa: BLE001
            length = None
        return {"type": "bytes", "length": length}
    if depth >= MAX_DEPTH:
        return {"type": safe_type_name(value), "truncated": True}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            for key, item in islice(iter(value.items()), MAX_ITEMS + 1):
                if len(result) >= MAX_ITEMS:
                    result["_respan_truncated_items"] = True
                    break
                rendered_key = _safe_key(key)
                result[rendered_key] = (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(rendered_key)
                    else to_jsonable(item, depth=depth + 1)
                )
        except BaseException:  # noqa: BLE001
            return {"type": safe_type_name(value), "unavailable": True}
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[Any] = []
        try:
            for item in islice(iter(value), MAX_ITEMS + 1):
                if len(items) >= MAX_ITEMS:
                    items.append({"_respan_truncated_items": True})
                    break
                items.append(to_jsonable(item, depth=depth + 1))
        except BaseException:  # noqa: BLE001
            return {"type": safe_type_name(value), "unavailable": True}
        return items
    for method_name in ("model_dump", "to_dict"):
        try:
            method = getattr(value, method_name, None)
        except BaseException:  # noqa: BLE001
            return {"type": safe_type_name(value), "unavailable": True}
        if not callable(method):
            continue
        try:
            dumped = method(mode="json") if method_name == "model_dump" else method()
        except TypeError:
            try:
                dumped = method()
            except BaseException:  # noqa: BLE001, S112
                continue
        except BaseException:  # noqa: BLE001, S112
            continue
        return to_jsonable(dumped, depth=depth + 1)
    return {"type": safe_type_name(value)}


def json_dumps(value: Any, *, max_bytes: int = MAX_JSON_BYTES) -> str:
    serialized = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(serialized.encode()) <= max_bytes:
        return serialized
    low, high, best = 0, len(serialized), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {"preview": serialized[:middle], "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(candidate.encode()) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or '{"truncated":true}'
