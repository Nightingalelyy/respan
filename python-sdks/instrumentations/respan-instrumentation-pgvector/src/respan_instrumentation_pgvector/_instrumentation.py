"""Native pgvector instrumentation for psycopg 3 operations."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from enum import Enum
from numbers import Real
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_instrumentation_pgvector._constants import (
    FUNCTION_PATCH_SPECS,
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    METHOD_PATCH_SPECS,
    PGVECTOR_INSTRUMENTATION_NAME,
    SENSITIVE_KEY_PARTS,
    FunctionPatchSpec,
    MethodPatchSpec,
)
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_SQL_OPERATION = re.compile(r"^\s*(?:/\*.*?\*/\s*)*(?:--[^\n]*\n\s*)*([A-Za-z]+)", re.S)


_VECTOR_KEY_PARTS = ("embedding", "vector")


def _is_numeric_vector(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and all(isinstance(item, Real) and not isinstance(item, bool) for item in value)
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
    if depth > 7 and not (vector_context or _is_numeric_vector(value)):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
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
        return _jsonable(
            asdict(value),
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        omitted = 0
        regular_items = 0
        for key, item in value.items():
            key_text = str(key)
            item_is_vector = vector_context or _is_vector_key(key_text)
            if not item_is_vector and regular_items >= MAX_PREVIEW_ITEMS:
                omitted += 1
                continue
            regular_items += int(not item_is_vector)
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
        if omitted:
            result["__truncated__"] = omitted
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        preserve_all = vector_context or _is_numeric_vector(value)
        if preserve_all:
            preserved_vector[0] = True
        source = value if preserve_all else value[:MAX_PREVIEW_ITEMS]
        items = [
            _jsonable(
                item,
                depth=depth + 1,
                vector_context=vector_context,
                preserved_vector=preserved_vector,
            )
            for item in source
        ]
        if preserve_all:
            return items
        if len(value) > MAX_PREVIEW_ITEMS:
            return {"count": len(value), "items": items, "truncated": True}
        return items
    for method_name in (
        "model_dump",
        "to_dict",
        "dict",
        "to_list",
        "tolist",
        "to_numpy",
    ):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            return _jsonable(
                method(),
                depth=depth + 1,
                vector_context=vector_context,
                preserved_vector=preserved_vector,
            )
        except Exception:
            continue
    return repr(value)


def _json_dumps(value: Any) -> str:
    preserved_vector = [False]
    text = json.dumps(
        _jsonable(value, preserved_vector=preserved_vector),
        default=str,
        sort_keys=True,
    )
    if preserved_vector[0] or len(text) <= MAX_ATTRIBUTE_CHARS:
        return text
    return json.dumps(
        {"preview": text[:MAX_ATTRIBUTE_CHARS], "truncated": True},
        sort_keys=True,
    )


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
    except Exception:
        arguments = {"args": list(args), "kwargs": kwargs}
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
    if query is None:
        return method
    match = _SQL_OPERATION.match(str(query))
    return match.group(1).upper() if match else method


def _result_summary(value: Any) -> Any:
    if value is None:
        return {"status": "ok"}
    summary: dict[str, Any] = {"type": type(value).__name__}
    found = False
    for key in ("rowcount", "statusmessage", "rownumber", "closed"):
        item = getattr(value, key, None)
        if item is not None:
            summary[key] = item
            found = True
    description = getattr(value, "description", None)
    if description:
        summary["columns"] = [
            str(getattr(column, "name", column[0] if column else column))
            for column in description
        ]
        found = True
    return summary if found else _jsonable(value)


class PGVectorInstrumentor:
    """Trace pgvector registration and psycopg sync/async SQL operations."""

    name = PGVECTOR_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _patched_targets: list[tuple[Any, str]] = []
    _active_call: ContextVar[bool] = ContextVar(
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
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, "postgresql")
        span.set_attribute(
            OTelSpanAttributes.DB_OPERATION,
            _db_operation(method, args, kwargs),
        )
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                _json_dumps(_call_input(operation, wrapped, args, kwargs)),
            )

    def _set_error(self, span: Any, exc: Exception) -> None:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute("status_code", 500)
        span.set_attribute(ERROR_MESSAGE_ATTR, str(exc))
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_dumps({"error": type(exc).__name__, "message": str(exc)}),
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
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"pgvector.{operation}",
                kind=SpanKind.CLIENT,
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
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"pgvector.{operation}",
                kind=SpanKind.CLIENT,
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

    def _patch_method_spec(self, spec: MethodPatchSpec) -> list[tuple[Any, str]]:
        try:
            module = importlib.import_module(spec.module)
            target_class = getattr(module, spec.class_name)
        except (ImportError, AttributeError):
            return []
        patched: list[tuple[Any, str]] = []
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
                if _is_async:
                    return self._trace_async(_label, _method, wrapped, args, kwargs)
                return self._trace_sync(_label, _method, wrapped, args, kwargs)

            wrap_function_wrapper(
                spec.module,
                f"{spec.class_name}.{method}",
                traced,
            )
            patched.append((target_class, method))
        return patched

    def _patch_function_spec(
        self,
        spec: FunctionPatchSpec,
    ) -> list[tuple[Any, str]]:
        try:
            module = importlib.import_module(spec.module)
            original = getattr(module, spec.function_name)
        except (ImportError, AttributeError):
            return []
        if not callable(original):
            return []

        def traced(
            wrapped: Any,
            _instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            if spec.is_async:
                return self._trace_async(spec.label, "", wrapped, args, kwargs)
            return self._trace_sync(spec.label, "", wrapped, args, kwargs)

        wrap_function_wrapper(spec.module, spec.function_name, traced)
        return [(module, spec.function_name)]

    def activate(self) -> None:
        """Patch pgvector registration and psycopg operations."""
        cls = type(self)
        if self._is_instrumented or not self._tracing_enabled():
            return
        if cls._patches_applied:
            cls._activation_count += 1
            self._is_instrumented = True
            return

        patched_targets: list[tuple[Any, str]] = []
        for spec in METHOD_PATCH_SPECS:
            patched_targets.extend(self._patch_method_spec(spec))
        for spec in FUNCTION_PATCH_SPECS:
            patched_targets.extend(self._patch_function_spec(spec))

        self._is_instrumented = bool(patched_targets)
        cls._patches_applied = self._is_instrumented
        cls._activation_count = int(self._is_instrumented)
        cls._patched_targets = patched_targets
        if not self._is_instrumented:
            logger.warning("pgvector instrumentation found no supported methods")

    def deactivate(self) -> None:
        """Remove pgvector patches after the final active instance stops."""
        if not self._is_instrumented:
            return
        self._is_instrumented = False
        cls = type(self)
        cls._activation_count = max(cls._activation_count - 1, 0)
        if cls._activation_count:
            return
        for target, method in reversed(cls._patched_targets):
            try:
                unwrap(target, method)
            except Exception:
                logger.debug(
                    "Failed to unwrap pgvector %s.%s",
                    getattr(target, "__name__", type(target).__name__),
                    method,
                    exc_info=True,
                )
        cls._patched_targets = []
        cls._patches_applied = False

    def instrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`activate`."""
        self.activate()

    def uninstrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`deactivate`."""
        self.deactivate()
