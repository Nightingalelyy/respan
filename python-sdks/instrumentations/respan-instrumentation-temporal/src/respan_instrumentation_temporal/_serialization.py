"""Stable, bounded, privacy-safe Temporal attribute serialization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

_SENSITIVE = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|headers?|password|rpc[_-]?metadata|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_TEXT_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|cookie|password|secret|token)\s*[:=]\s*([^\s,;]+)"
)
_QUOTED_SECRET = re.compile(
    r"""(?i)(["'](?:api[_-]?key|authorization|cookie|password|secret|token)["']\s*:\s*)(["'])(.*?)\2"""
)


def _redact_text(value: str) -> str:
    value = _QUOTED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        value,
    )
    return _TEXT_SECRET.sub(r"\1=[REDACTED]", value)


def _type_name(value: Any) -> str:
    return type(value).__name__[:120]


def _bounded_text(value: str, *, max_bytes: int = 4_000) -> str:
    value = _redact_text(value)
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    low, high, best = 0, len(value), ""
    suffix = "…[truncated]"
    while low <= high:
        middle = (low + high) // 2
        candidate = f"{value[:middle]}{suffix}"
        if len(candidate.encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or "[truncated]"


def safe_baggage_value(key: str, value: Any) -> str:
    if _SENSITIVE.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _bounded_text(value, max_bytes=2_000)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) or isinstance(value, float) and math.isfinite(value):
        return str(value)
    return json_dumps(value, max_bytes=2_000)


def safe_error_message(error: BaseException, *, capture_content: bool) -> str:
    if not capture_content:
        return _type_name(error)
    pieces: list[str] = []
    try:
        for item in islice(iter(error.args), 4):
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, (int, float, bool)):
                pieces.append(json.dumps(item))
    except BaseException:  # noqa: BLE001 - hostile exception args must fail closed
        pieces = []
    return _bounded_text(" ".join(pieces) or _type_name(error))


def to_jsonable(
    value: Any, *, depth: int = 0, max_depth: int = 8, max_items: int = 30
) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _redact_text(value)
    if depth >= max_depth:
        return {"type": _type_name(value), "truncated": True}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            for index, (key, item) in enumerate(
                islice(iter(value.items()), max_items + 1)
            ):
                if index == max_items:
                    result["_respan_truncated_items"] = True
                    break
                safe_key = key if isinstance(key, str) else f"<{_type_name(key)}>"
                safe_key = safe_key[:256]
                result[safe_key] = (
                    "[REDACTED]"
                    if _SENSITIVE.search(safe_key)
                    else to_jsonable(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                    )
                )
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _type_name(value), "unavailable": True}
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[Any] = []
        try:
            for index, item in enumerate(islice(iter(value), max_items + 1)):
                if index == max_items:
                    result.append({"_respan_truncated_items": True})
                    break
                result.append(
                    to_jsonable(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                    )
                )
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _type_name(value), "unavailable": True}
        return result
    try:
        model_dump = getattr(value, "model_dump", None)
    except BaseException:  # noqa: BLE001 - hostile objects must fail closed
        return {"type": _type_name(value), "unavailable": True}
    if callable(model_dump):
        try:
            return to_jsonable(
                model_dump(mode="json"),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
        except BaseException:  # noqa: BLE001 - vendor hooks must not break tracing
            return {"type": _type_name(value), "unavailable": True}
    return {"type": _type_name(value)}


def json_dumps(value: Any, *, max_bytes: int) -> str:
    serialized = json.dumps(
        to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(serialized.encode("utf-8")) <= max_bytes:
        return serialized
    low, high, best = 0, min(len(serialized), max_bytes), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {"preview": serialized[:middle], "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(candidate.encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or '{"truncated":true}'
