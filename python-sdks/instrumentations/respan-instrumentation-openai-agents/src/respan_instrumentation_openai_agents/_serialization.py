"""Bounded, redacting serializers for OpenAI Agents trace payloads."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

MAX_ATTRIBUTE_BYTES = 16_000
_MAX_COLLECTION_ITEMS = 50
_MAX_DEPTH = 6
_MAX_STRING_CHARS = 4_000
_MAX_FLATTENED_METADATA_KEY_CHARS = 128
_REDACTED = "[REDACTED]"

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
    r"(?i)\b(api[_-]?key|authorization|password|secret|session[_-]?token|token)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    redacted = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}", redacted
    )
    if len(redacted) <= _MAX_STRING_CHARS:
        return redacted
    return f"{redacted[:_MAX_STRING_CHARS]}...[truncated]"


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    if normalized in _SENSITIVE_KEYS:
        return True
    return normalized.endswith(
        ("accesstoken", "authtoken", "idtoken", "refreshtoken", "secretkey")
    )


def _safe_key_text(key: Any) -> str:
    if isinstance(key, str):
        return _redact_text(key)
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, float) and math.isfinite(key):
        return str(key)
    return f"<{type(key).__name__}>"


def safe_text(value: Any, *, default: str = "", limit: int = 512) -> str:
    """Return a bounded, redacted scalar without invoking custom string hooks."""
    if isinstance(value, str):
        return _redact_text(value)[:limit]
    if value is None:
        return default
    if isinstance(value, bool):
        return ("true" if value else "false")[:limit]
    if isinstance(value, int):
        return str(value)[:limit]
    if isinstance(value, float) and math.isfinite(value):
        return str(value)[:limit]
    return f"<{type(value).__name__}>"[:limit]


def _public_object_fields(value: Any) -> Mapping[str, Any] | None:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            }
        except Exception:  # noqa: BLE001 - dataclass fields can execute user code.
            return None

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        for kwargs in (
            {"mode": "json", "exclude_none": True},
            {"exclude_none": True},
            {},
        ):
            try:
                dumped = model_dump(**kwargs)
            except (TypeError, ValueError):
                continue
            except Exception:  # noqa: BLE001 - Pydantic hooks can execute user code.
                return None
            if isinstance(dumped, Mapping):
                return dumped

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
        except Exception:  # noqa: BLE001 - SDK conversion hooks are untrusted.
            return None
        if isinstance(dumped, Mapping):
            return dumped

    try:
        fields = vars(value)
    except (TypeError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - custom __dict__ descriptors are untrusted.
        return None
    if isinstance(fields, Mapping):
        return {
            key: item
            for key, item in fields.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    return None


def json_value(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """Convert a value to bounded JSON data without arbitrary ``str``/``repr`` calls."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{type(value).__name__}:{len(value)} bytes>"
    if _depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}:max-depth>"

    seen = _seen if _seen is not None else set()
    object_id = id(value)
    if object_id in seen:
        return "<cycle>"
    seen.add(object_id)
    try:
        mapping: Mapping[Any, Any] | None = None
        sequence: Sequence[Any] | None = None
        if isinstance(value, Mapping):
            mapping = value
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            sequence = value
        else:
            mapping = _public_object_fields(value)

        if mapping is not None:
            result: dict[str, Any] = {}
            for index, (key, item) in enumerate(mapping.items()):
                if index >= _MAX_COLLECTION_ITEMS:
                    result["__truncated_items__"] = True
                    break
                key_text = _safe_key_text(key)
                result[key_text] = (
                    _REDACTED
                    if _is_sensitive_key(key)
                    else json_value(item, _depth=_depth + 1, _seen=seen)
                )
            return result

        if sequence is not None:
            items = list(islice(iter(sequence), _MAX_COLLECTION_ITEMS + 1))
            result = [
                json_value(item, _depth=_depth + 1, _seen=seen)
                for item in items[:_MAX_COLLECTION_ITEMS]
            ]
            if len(items) > _MAX_COLLECTION_ITEMS:
                result.append("<truncated:more items>")
            return result

        return f"<{type(value).__name__}>"
    except Exception:  # noqa: BLE001 - serializer must not break agent execution.
        return f"<{type(value).__name__}:unserializable>"
    finally:
        seen.discard(object_id)


def json_string(value: Any) -> str:
    """Return valid, redacted JSON bounded to the OTel attribute budget."""
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

    preview = encoded[: MAX_ATTRIBUTE_BYTES // 2]
    while (
        len(
            json.dumps(
                {
                    "original_bytes": encoded_bytes,
                    "preview": preview,
                    "truncated": True,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        > MAX_ATTRIBUTE_BYTES
    ):
        preview = preview[: max(0, len(preview) - 256)]
    return json.dumps(
        {
            "original_bytes": encoded_bytes,
            "preview": preview,
            "truncated": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def flatten_metadata_attributes(metadata: Mapping[Any, Any]) -> dict[str, str]:
    """Return bounded, redacted top-level metadata fields for live ingestion.

    The canonical aggregate JSON remains authoritative. These scalar aliases mirror
    the tracing SDK's supported ``respan.metadata.<key>`` wire format so metadata
    stays queryable while backends migrate to merging the aggregate field directly.
    """
    flattened: dict[str, str] = {}
    try:
        for index, (key, value) in enumerate(metadata.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                break
            if value is None:
                continue
            key_text = _safe_key_text(key).strip()
            if not key_text:
                continue
            key_text = re.sub(r"[\x00-\x1f\x7f]", "_", key_text)
            key_text = key_text[:_MAX_FLATTENED_METADATA_KEY_CHARS]
            if not key_text or key_text in flattened:
                continue
            flattened[key_text] = (
                _REDACTED
                if _is_sensitive_key(key)
                else (
                    safe_text(value, limit=_MAX_STRING_CHARS)
                    if isinstance(value, str)
                    else json_string(value)
                )
            )
    except Exception:  # noqa: BLE001 - untrusted mappings must not break tracing.
        return flattened
    return flattened


def parse_json_string(value: Any) -> Any:
    """Decode SDK JSON strings while retaining ordinary text as text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def safe_error_message(error: Any) -> str | None:
    """Render an SDK error without invoking arbitrary exception string methods."""
    if error is None:
        return None
    if isinstance(error, Mapping):
        message = error.get("message") or error.get("error")
        if isinstance(message, str):
            return _redact_text(message)
        return json_string(error)
    if isinstance(error, str):
        return _redact_text(error)
    return type(error).__name__


def error_status_code(error: Any, *, default: int = 500) -> int:
    """Extract a provider/application HTTP status from an Agents SDK error."""
    candidates: list[Any] = [error]
    if isinstance(error, Mapping):
        candidates.extend(
            (error.get("data"), error.get("response"), error.get("cause"))
        )
    else:
        for name in ("data", "response", "cause", "__cause__"):
            try:
                candidates.append(getattr(error, name, None))
            except Exception:  # noqa: BLE001, S112 - inspect next safe candidate.
                continue

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            values = (
                candidate.get("status_code"),
                candidate.get("status"),
                candidate.get("http_status"),
                candidate.get("code"),
            )
        else:
            values = []
            for name in ("status_code", "status", "http_status", "code"):
                try:
                    values.append(getattr(candidate, name, None))
                except Exception:  # noqa: BLE001, S112 - inspect next safe field.
                    continue
        for value in values:
            if isinstance(value, int) and 400 <= value <= 599:
                return value
            if isinstance(value, str) and value.isdigit():
                parsed = int(value)
                if 400 <= parsed <= 599:
                    return parsed

    return default
