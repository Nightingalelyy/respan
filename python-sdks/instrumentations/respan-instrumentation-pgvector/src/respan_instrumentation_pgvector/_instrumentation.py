"""Native pgvector instrumentation for psycopg 3 operations."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import re
import struct
from array import array
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from itertools import islice
from numbers import Integral, Real
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

from respan_instrumentation_pgvector._constants import (
    FUNCTION_PATCH_SPECS,
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_CHARS,
    METHOD_PATCH_SPECS,
    PGVECTOR_INSTRUMENTATION_NAME,
    PGVECTOR_INSTRUMENTATION_VERSION,
    SENSITIVE_KEY_PARTS,
    FunctionPatchSpec,
    MethodPatchSpec,
)

logger = logging.getLogger(__name__)

_SQL_OPERATION = re.compile(
    r"^\s*(?:/\*.*?\*/\s*)*(?:--[^\n]*\n\s*)*([A-Za-z]+)",
    re.DOTALL,
)
_VECTOR_KEY_PARTS = ("embedding", "vector")
_POSTGRES_URI = re.compile(r"(?i)\bpostgres(?:ql)?(?:\+[a-z0-9_.-]+)?://\S+")
_URI_CREDENTIALS = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@\S+")
_LIBPQ_DSN_KEY = re.compile(
    r"(?i)(?:^|\s)(?:host|hostaddr|port|dbname|database|user|password|passfile|"
    r"sslcert|sslkey|service)\s*="
)
_INLINE_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|api[_-]?key|authorization|secret|token)"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;&]+")
_MEMORY_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


def _safe_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    module = getattr(value_type, "__module__", "")
    qualname = getattr(value_type, "__qualname__", None) or getattr(
        value_type,
        "__name__",
        "object",
    )
    candidate = f"{module}.{qualname}" if module and module != "builtins" else qualname
    stable = re.sub(r"[^A-Za-z0-9_.-]+", ".", candidate).strip(".")
    return (stable or "object")[:256]


def _truncate_utf8(value: str, *, max_bytes: int = MAX_STRING_CHARS) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = "...[truncated]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    preview = encoded[:budget].decode("utf-8", errors="ignore")
    return f"{preview}{suffix}"


def _safe_text(value: str) -> str:
    """Return bounded text with credentials, DSNs, and addresses removed."""
    text = value
    if _POSTGRES_URI.search(text) or len(_LIBPQ_DSN_KEY.findall(text)) >= 2:
        return "<redacted-dsn>"
    text = _URI_CREDENTIALS.sub("<redacted-uri>", text)
    text = _BEARER_TOKEN.sub("Bearer <redacted>", text)
    text = _INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text
    )
    text = _MEMORY_ADDRESS.sub("0x<redacted>", text)
    return _truncate_utf8(text)


def _stable_mapping_key(value: Any) -> str:
    if isinstance(value, Enum):
        return _stable_mapping_key(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return _safe_text(str(value))[:256]
    return f"<{_safe_type_name(value)}>"


def _unsupported_summary(value: Any, *, reason: str | None = None) -> dict[str, str]:
    summary = {"type": _safe_type_name(value)}
    if reason:
        summary["truncated"] = reason
    return summary


def _trusted_collection_length(value: Any) -> int | str:
    if type(value) in {list, tuple, dict, range, array}:
        return len(value)
    return f">{MAX_PREVIEW_ITEMS}"


def _bounded_vector_result(items: list[Any], count: int) -> Any:
    if count <= MAX_PREVIEW_ITEMS:
        return items[:count]
    return {
        "count": count,
        "items": items[:MAX_PREVIEW_ITEMS],
        "kind": "vector",
        "truncated": True,
    }


def _trusted_pgvector_preview(value: Any) -> Any | None:
    """Read known pgvector internals without materializing a full vector."""

    value_type = type(value)
    identity = (value_type.__module__, value_type.__name__)
    try:
        if identity == ("pgvector.vector", "Vector"):
            raw = getattr(value, "_value", None)
            if type(raw) is not array:
                return None
            count = len(raw)
            sample = list(islice(iter(raw), MAX_PREVIEW_ITEMS + 1))
            return _bounded_vector_result(sample, count)

        if identity == ("pgvector.halfvec", "HalfVector"):
            raw = getattr(value, "_value", None)
            if type(raw) is not array:
                return None
            count = len(raw)
            sample = [
                struct.unpack("e", struct.pack("H", int(item)))[0]
                for item in islice(iter(raw), MAX_PREVIEW_ITEMS + 1)
            ]
            return _bounded_vector_result(sample, count)

        if identity == ("pgvector.sparsevec", "SparseVector"):
            count = getattr(value, "_dim", None)
            indices = getattr(value, "_indices", None)
            values = getattr(value, "_values", None)
            if (
                not isinstance(count, int)
                or count < 0
                or type(indices) is not list
                or type(values) is not list
            ):
                return None
            preview_length = min(count, MAX_PREVIEW_ITEMS)
            sample: list[int | float] = [0.0] * preview_length
            for index, item in islice(zip(indices, values), MAX_PREVIEW_ITEMS + 1):
                if (
                    isinstance(index, int)
                    and 0 <= index < preview_length
                    and isinstance(item, Real)
                    and not isinstance(item, bool)
                ):
                    number = float(item)
                    sample[index] = number if math.isfinite(number) else 0.0
            return _bounded_vector_result(sample, count)

        if identity == ("pgvector.bit", "Bit"):
            count = getattr(value, "_length", None)
            data = getattr(value, "_data", None)
            if not isinstance(count, int) or count < 0 or type(data) is not bytes:
                return None
            preview_length = min(count, MAX_PREVIEW_ITEMS)
            sample = [
                bool(data[index // 8] & (1 << (7 - index % 8)))
                for index in range(preview_length)
            ]
            return _bounded_vector_result(sample, count)
    except Exception:  # noqa: BLE001 - corrupted vendor objects become summaries
        return None
    return None


def _trusted_numpy_preview(value: Any) -> Any | None:
    value_type = type(value)
    if (value_type.__module__, value_type.__name__) != ("numpy", "ndarray"):
        return None
    try:
        count = int(value.size)
        sample: list[Any] = []
        for item in islice(iter(value.flat), MAX_PREVIEW_ITEMS + 1):
            scalar = item.item()
            if scalar is None or isinstance(scalar, (bool, int, str)):
                sample.append(scalar)
            elif isinstance(scalar, Real):
                number = float(scalar)
                sample.append(number if math.isfinite(number) else None)
            else:
                sample.append(_unsupported_summary(scalar))
        return _bounded_vector_result(sample, count)
    except Exception:  # noqa: BLE001 - malformed arrays become summaries
        return None


@dataclass(frozen=True)
class _AppliedPatch:
    target: Any
    attribute: str
    installed_wrapper: Any


def _is_numeric_vector(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        sample = list(islice(iter(value), MAX_PREVIEW_ITEMS + 1))
    except Exception:  # noqa: BLE001 - hostile SDK sequences must not break calls
        return False
    return bool(sample) and all(
        isinstance(item, Real) and not isinstance(item, bool) for item in sample
    )


def _is_vector_key(value: str) -> bool:
    normalized = value.lower()
    return any(part in normalized for part in _VECTOR_KEY_PARTS)


def _jsonable(
    value: Any,
    *,
    depth: int = 0,
    vector_context: bool = False,
    preserved_vector: list[bool] | None = None,
) -> Any:
    preserved_vector = preserved_vector if preserved_vector is not None else [False]
    if depth > MAX_SERIALIZATION_DEPTH:
        return _unsupported_summary(value, reason="max_depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Enum):
        return _jsonable(
            value.value,
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    if is_dataclass(value) and not isinstance(value, type):
        try:
            field_items = list(islice(fields(value), MAX_PREVIEW_ITEMS + 1))
            dataclass_value = {
                field.name: getattr(value, field.name)
                for field in field_items[:MAX_PREVIEW_ITEMS]
            }
            if len(field_items) > MAX_PREVIEW_ITEMS:
                dataclass_value["__truncated__"] = len(fields(value))
        except Exception:  # noqa: BLE001 - hostile fields become a type summary
            return _unsupported_summary(value)
        return _jsonable(
            dataclass_value,
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            items = list(islice(value.items(), MAX_PREVIEW_ITEMS + 1))
        except Exception:  # noqa: BLE001 - hostile mappings become a type summary
            return _unsupported_summary(value)
        for key, item in items[:MAX_PREVIEW_ITEMS]:
            key_text = _stable_mapping_key(key)
            item_is_vector = vector_context or _is_vector_key(key_text)
            result[key_text] = (
                "<redacted>"
                if any(part in key_text.lower() for part in SENSITIVE_KEY_PARTS)
                else _jsonable(
                    item,
                    depth=depth + 1,
                    vector_context=item_is_vector,
                    preserved_vector=preserved_vector,
                )
            )
        if len(items) > MAX_PREVIEW_ITEMS:
            result["__truncated__"] = _trusted_collection_length(value)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        is_vector = vector_context or _is_numeric_vector(value)
        if is_vector:
            preserved_vector[0] = True
        try:
            source = list(islice(iter(value), MAX_PREVIEW_ITEMS + 1))
        except Exception:  # noqa: BLE001 - hostile sequences become a type summary
            return _unsupported_summary(value)
        items = [
            _jsonable(
                item,
                depth=depth + 1,
                vector_context=is_vector,
                preserved_vector=preserved_vector,
            )
            for item in source[:MAX_PREVIEW_ITEMS]
        ]
        if len(source) > MAX_PREVIEW_ITEMS:
            return {
                "count": _trusted_collection_length(value),
                "items": items,
                "kind": "vector" if is_vector else "sequence",
                "truncated": True,
            }
        return items
    vendor_preview = _trusted_pgvector_preview(value)
    if vendor_preview is None:
        vendor_preview = _trusted_numpy_preview(value)
    if vendor_preview is not None:
        preserved_vector[0] = True
        return _jsonable(
            vendor_preview,
            depth=depth + 1,
            vector_context=True,
            preserved_vector=preserved_vector,
        )
    return _unsupported_summary(value)


def _json_dumps(value: Any) -> str:
    preserved_vector = [False]
    text = json.dumps(
        _jsonable(value, preserved_vector=preserved_vector),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
    )
    encoded_size = len(text.encode("utf-8"))
    if encoded_size <= MAX_ATTRIBUTE_CHARS:
        return text
    low = 0
    high = min(len(text), MAX_ATTRIBUTE_CHARS)
    result = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {
                "original_bytes": encoded_size,
                "preview": text[:midpoint],
                "truncated": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(candidate.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS:
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result


def _call_input(
    operation: str,
    wrapped: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(wrapped).bind_partial(*args, **kwargs)
        arguments = {
            key: value for key, value in bound.arguments.items() if key != "self"
        }
    except (TypeError, ValueError):
        arguments = {"args": args, "kwargs": kwargs}
    return {"operation": operation, **arguments}


def _query_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "query" in kwargs:
        return kwargs["query"]
    if args:
        return args[0]
    return None


def _db_operation(
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    if method not in {"execute", "executemany"}:
        return method
    query = _query_from_call(args, kwargs)
    if not isinstance(query, (str, bytes, bytearray)):
        return method
    query_text = (
        query.decode("utf-8", errors="replace")
        if isinstance(query, bytes)
        else str(query)
    )
    match = _SQL_OPERATION.match(query_text)
    return match.group(1).upper() if match else method


def _result_summary(value: Any) -> Any:
    if value is None:
        return {"status": "ok"}
    summary: dict[str, Any] = {"type": type(value).__name__}
    found = False
    for key in ("rowcount", "statusmessage", "rownumber", "closed"):
        try:
            item = getattr(value, key, None)
        except Exception:  # noqa: BLE001, S112 - omit hostile vendor properties
            continue
        if item is not None:
            summary[key] = item
            found = True
    try:
        description = getattr(value, "description", None)
    except Exception:  # noqa: BLE001 - result properties are vendor-controlled
        description = None
    if description:
        columns: list[str] = []
        try:
            description_items = islice(iter(description), MAX_PREVIEW_ITEMS)
        except Exception:  # noqa: BLE001 - hostile descriptions are omitted
            description_items = ()
        for column in description_items:
            try:
                name = getattr(column, "name", None)
                if name is None and isinstance(column, Sequence) and column:
                    name = column[0]
            except Exception:  # noqa: BLE001 - hostile columns become type names
                name = None
            columns.append(
                _safe_text(name)
                if isinstance(name, str)
                else _safe_type_name(name or column)
            )
        summary["columns"] = columns
        found = True
    return summary if found else _jsonable(value)


class PGVectorInstrumentor:
    """Trace pgvector registration and psycopg sync/async SQL operations."""

    name: ClassVar[str] = PGVECTOR_INSTRUMENTATION_NAME
    _patches_applied: ClassVar[bool] = False
    _activation_count: ClassVar[int] = 0
    _patched_targets: ClassVar[list[_AppliedPatch]] = []
    _capture_content_config: ClassVar[bool | None] = None
    _lifecycle_lock: ClassVar[RLock] = RLock()
    _active_call: ClassVar[ContextVar[bool]] = ContextVar(
        "respan_pgvector_active_call",
        default=False,
    )

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    @staticmethod
    def _tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def _set_start_attributes(
        self,
        span: Any,
        label: str,
        method: str,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        operation = f"{label}.{method}" if method else label
        entity_name = f"pgvector.{operation}"
        span.set_attribute(RESPAN_LOG_TYPE, LOG_TYPE_TASK)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            "" if getattr(span, "parent", None) is None else entity_name,
        )
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, "postgresql")
        span.set_attribute(
            OTelSpanAttributes.DB_OPERATION,
            _db_operation(method, args, kwargs) or label,
        )
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                _json_dumps(_call_input(operation, wrapped, args, kwargs)),
            )

    @staticmethod
    def _exception_message(exc: Exception) -> str:
        try:
            arguments = exc.args
        except Exception:  # noqa: BLE001 - hostile exception attributes are ignored
            arguments = ()
        for argument in arguments:
            if isinstance(argument, str):
                return _safe_text(argument)
            if argument is None or isinstance(argument, (bool, int)):
                return _safe_text(str(argument))
            if isinstance(argument, float) and math.isfinite(argument):
                return _safe_text(str(argument))
        return _safe_type_name(exc)

    def _set_error(self, span: Any, exc: Exception) -> None:
        message = self._exception_message(exc)
        exception_type = _safe_type_name(exc)
        add_event = getattr(span, "add_event", None)
        if callable(add_event):
            add_event(
                "exception",
                {
                    OTelSpanAttributes.EXCEPTION_MESSAGE: message,
                    OTelSpanAttributes.EXCEPTION_TYPE: exception_type,
                },
            )
        span.set_status(Status(StatusCode.ERROR, message))
        span.set_attribute("status_code", 500)
        span.set_attribute(ERROR_MESSAGE_ATTR, message)
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_dumps({"error": exception_type, "message": message}),
            )

    def _trace_sync(
        self,
        label: str,
        method: str,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return wrapped(*args, **kwargs)
        token = active_call.set(True)
        operation = f"{label}.{method}" if method else label
        try:
            tracer = trace.get_tracer(
                PGVECTOR_INSTRUMENTATION_NAME,
                PGVECTOR_INSTRUMENTATION_VERSION,
            )
            with tracer.start_as_current_span(
                f"pgvector.{operation}",
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self._set_start_attributes(span, label, method, wrapped, args, kwargs)
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps(_result_summary(result)),
                    )
                return result
        finally:
            active_call.reset(token)

    async def _trace_async(
        self,
        label: str,
        method: str,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return await wrapped(*args, **kwargs)
        token = active_call.set(True)
        operation = f"{label}.{method}" if method else label
        try:
            tracer = trace.get_tracer(
                PGVECTOR_INSTRUMENTATION_NAME,
                PGVECTOR_INSTRUMENTATION_VERSION,
            )
            with tracer.start_as_current_span(
                f"pgvector.{operation}",
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self._set_start_attributes(span, label, method, wrapped, args, kwargs)
                try:
                    result = await wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps(_result_summary(result)),
                    )
                return result
        finally:
            active_call.reset(token)

    @staticmethod
    def _wrapper_chain_contains(current: Any, expected: Any) -> bool:
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if current is expected:
                return True
            seen.add(id(current))
            current = getattr(current, "__wrapped__", None)
        return False

    @classmethod
    def _unwrap_targets(cls, patches: list[_AppliedPatch]) -> list[_AppliedPatch]:
        failed: list[_AppliedPatch] = []
        for patch in reversed(patches):
            try:
                current = inspect.getattr_static(patch.target, patch.attribute)
                if current is patch.installed_wrapper:
                    unwrap(patch.target, patch.attribute)
                    current = inspect.getattr_static(patch.target, patch.attribute)
                if cls._wrapper_chain_contains(current, patch.installed_wrapper):
                    failed.append(patch)
            except Exception:
                failed.append(patch)
                logger.debug(
                    "Failed to unwrap pgvector %s.%s",
                    getattr(
                        patch.target,
                        "__name__",
                        type(patch.target).__name__,
                    ),
                    patch.attribute,
                    exc_info=True,
                )
        failed.reverse()
        return failed

    def _patch_method_spec(
        self,
        spec: MethodPatchSpec,
        patched: list[_AppliedPatch],
    ) -> None:
        try:
            module = importlib.import_module(spec.module)
            target_class = getattr(module, spec.class_name)
        except (ImportError, AttributeError):
            return
        for method in spec.methods:
            if not callable(getattr(target_class, method, None)):
                continue

            def traced(
                wrapped: Any,
                _instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _label: str = spec.label,
                _method: str = method,
                _is_async: bool = spec.is_async,
            ) -> Any:
                instrumentor_class = type(self)
                if (
                    not instrumentor_class._patches_applied
                    or not instrumentor_class._activation_count
                ):
                    return wrapped(*args, **kwargs)
                if _is_async:
                    return self._trace_async(_label, _method, wrapped, args, kwargs)
                return self._trace_sync(_label, _method, wrapped, args, kwargs)

            wrap_function_wrapper(
                spec.module,
                f"{spec.class_name}.{method}",
                traced,
            )
            patched.append(
                _AppliedPatch(
                    target_class,
                    method,
                    inspect.getattr_static(target_class, method),
                )
            )

    def _patch_function_spec(
        self,
        spec: FunctionPatchSpec,
        patched: list[_AppliedPatch],
    ) -> None:
        try:
            module = importlib.import_module(spec.module)
            original = getattr(module, spec.function_name)
        except (ImportError, AttributeError):
            return
        if not callable(original):
            return

        def traced(
            wrapped: Any,
            _instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            instrumentor_class = type(self)
            if (
                not instrumentor_class._patches_applied
                or not instrumentor_class._activation_count
            ):
                return wrapped(*args, **kwargs)
            if spec.is_async:
                return self._trace_async(spec.label, "", wrapped, args, kwargs)
            return self._trace_sync(spec.label, "", wrapped, args, kwargs)

        wrap_function_wrapper(spec.module, spec.function_name, traced)
        patched.append(
            _AppliedPatch(
                module,
                spec.function_name,
                inspect.getattr_static(module, spec.function_name),
            )
        )

    def activate(self) -> None:
        """Patch pgvector registration and psycopg operations."""
        cls = type(self)
        with cls._lifecycle_lock:
            self._activate_locked()

    def _activate_locked(self) -> None:
        cls = type(self)
        if self._is_instrumented or not self._tracing_enabled():
            return
        if cls._patched_targets and not cls._patches_applied:
            cls._patched_targets = self._unwrap_targets(cls._patched_targets)
            if cls._patched_targets:
                raise RuntimeError(
                    "pgvector instrumentation could not clean up stale wrappers"
                )
            cls._capture_content_config = None
        if cls._patches_applied:
            if cls._capture_content_config != self._capture_content:
                raise ValueError(
                    "all active PGVectorInstrumentor instances must use the same "
                    "capture_content setting"
                )
            cls._activation_count += 1
            self._is_instrumented = True
            return

        patched_targets: list[_AppliedPatch] = []
        try:
            for spec in METHOD_PATCH_SPECS:
                self._patch_method_spec(spec, patched_targets)
            for spec in FUNCTION_PATCH_SPECS:
                self._patch_function_spec(spec, patched_targets)
        except Exception:
            cls._patched_targets = self._unwrap_targets(patched_targets)
            cls._patches_applied = False
            cls._activation_count = 0
            cls._capture_content_config = (
                self._capture_content if cls._patched_targets else None
            )
            self._is_instrumented = False
            logger.exception("Failed to activate pgvector instrumentation")
            raise

        self._is_instrumented = bool(patched_targets)
        cls._patches_applied = self._is_instrumented
        cls._activation_count = int(self._is_instrumented)
        cls._patched_targets = patched_targets
        cls._capture_content_config = (
            self._capture_content if self._is_instrumented else None
        )
        if not self._is_instrumented:
            logger.warning("pgvector instrumentation found no supported methods")

    def deactivate(self) -> None:
        """Remove pgvector patches after the final active instance stops."""
        cls = type(self)
        with cls._lifecycle_lock:
            self._deactivate_locked()

    def _deactivate_locked(self) -> None:
        if not self._is_instrumented:
            return
        self._is_instrumented = False
        cls = type(self)
        cls._activation_count = max(cls._activation_count - 1, 0)
        if cls._activation_count:
            return
        cls._patched_targets = self._unwrap_targets(cls._patched_targets)
        cls._patches_applied = False
        if not cls._patched_targets:
            cls._capture_content_config = None

    def instrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`activate`."""
        self.activate()

    def uninstrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`deactivate`."""
        self.deactivate()
