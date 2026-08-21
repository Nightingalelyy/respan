"""Native Weaviate v4 collection instrumentation for Respan."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib import metadata
from itertools import islice
from numbers import Real
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

from respan_instrumentation_weaviate._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    WEAVIATE_INSTRUMENTATION_NAME,
    WEAVIATE_PATCH_SPECS,
    PatchSpec,
)

logger = logging.getLogger(__name__)
_SCOPE_NAME = "respan-instrumentation-weaviate"
try:
    _SCOPE_VERSION = metadata.version(_SCOPE_NAME)
except metadata.PackageNotFoundError:
    _SCOPE_VERSION = "0.1.0"


_VECTOR_KEY_PARTS = ("embedding", "vector")
_SENSITIVE_KEY = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|password|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)([a-z0-9_.-]*(?:api[_-]?key|authorization|cookie|password|secret|token))"
    r"\s*[:=]\s*([^\s,;]+)"
)


def _safe_text(value: Any, *, max_bytes: int = 4_000) -> str:
    if isinstance(value, str):
        text = _ASSIGNMENT_SECRET.sub(
            lambda match: f"{match.group(1)}=[REDACTED]",
            value,
        )
    elif value is None or isinstance(value, (bool, int)):
        text = json.dumps(value)
    elif isinstance(value, float) and math.isfinite(value):
        text = str(value)
    else:
        text = f"<{type(value).__name__[:120]}>"
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


def _safe_exception_message(exc: BaseException) -> str:
    try:
        args = exc.args
    except BaseException:  # noqa: BLE001
        args = ()
    details = [
        _safe_text(item, max_bytes=1_000)
        for item in args[:4]
        if item is None or isinstance(item, (str, bool, int, float))
    ]
    name = type(exc).__name__[:120]
    return _safe_text(
        f"{name}: {'; '.join(filter(None, details))}" if details else name
    )


def _provider_status_code(exc: BaseException) -> int:
    candidates: list[Any] = []
    for owner in (exc,):
        for name in ("status_code", "status"):
            try:
                candidates.append(getattr(owner, name, None))
            except BaseException:
                logger.debug("Ignored unsafe Weaviate exception status", exc_info=True)
                continue
    try:
        response = getattr(exc, "response", None)
    except BaseException:
        logger.debug("Ignored unsafe Weaviate exception response", exc_info=True)
        response = None
    if response is not None:
        for name in ("status_code", "status"):
            try:
                candidates.append(getattr(response, name, None))
            except BaseException:
                logger.debug("Ignored unsafe Weaviate response status", exc_info=True)
                continue
    for value in candidates:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 400 <= value <= 599
        ):
            return value
    return 500


def _is_numeric_vector(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        return all(
            isinstance(item, Real)
            and not isinstance(item, bool)
            and (not isinstance(item, float) or math.isfinite(item))
            for item in value
        )
    except BaseException:
        logger.debug("Ignored unsafe Weaviate vector iterator", exc_info=True)
        return False


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
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _safe_text(value)
    if isinstance(value, str):
        return _safe_text(value, max_bytes=16_000)
    if depth > 7 and not (vector_context or _is_numeric_vector(value)):
        return {"type": type(value).__name__[:120], "truncated": True}
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
            key_text = _safe_text(key, max_bytes=256)
            item_is_vector = vector_context or _is_vector_key(key_text)
            if not item_is_vector and regular_items >= MAX_PREVIEW_ITEMS:
                omitted += 1
                continue
            regular_items += int(not item_is_vector)
            result[key_text] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(key_text)
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
        source = value if preserve_all else islice(iter(value), MAX_PREVIEW_ITEMS + 1)
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
        if len(items) > MAX_PREVIEW_ITEMS:
            return {
                "items": items[:MAX_PREVIEW_ITEMS],
                "truncated": True,
            }
        return items
    for method_name in ("model_dump", "to_dict", "dict", "to_json", "tolist"):
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
            logger.debug("Ignored unsafe Weaviate model conversion", exc_info=True)
            continue
    public_values = getattr(value, "__dict__", None)
    if isinstance(public_values, dict):
        return _jsonable(
            {
                key: item
                for key, item in public_values.items()
                if not key.startswith("_") and not callable(item)
            },
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    return {"type": type(value).__name__[:120]}


def _json_dumps_impl(value: Any) -> str:
    preserved_vector = [False]
    text = json.dumps(
        _jsonable(value, preserved_vector=preserved_vector),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if preserved_vector[0] or len(text.encode()) <= MAX_ATTRIBUTE_CHARS:
        return text
    low, high, best = 0, len(text), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {"preview": text[:middle], "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(candidate.encode()) <= MAX_ATTRIBUTE_CHARS:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or '{"truncated":true}'


def _json_dumps(value: Any) -> str:
    try:
        return _json_dumps_impl(value)
    except BaseException:
        logger.debug("Failed to serialize Weaviate telemetry", exc_info=True)
        return json.dumps({"type": type(value).__name__[:120], "unavailable": True})


def _instance_identity(instance: Any) -> dict[str, str]:
    identity: dict[str, str] = {}
    for source, target in (
        ("name", "collection_name"),
        ("_name", "collection_name"),
        ("tenant", "tenant"),
        ("_tenant", "tenant"),
    ):
        try:
            value = getattr(instance, source, None)
        except BaseException:
            logger.debug("Ignored unsafe Weaviate identity field", exc_info=True)
            continue
        if value is not None and target not in identity:
            identity[target] = _safe_text(value, max_bytes=1_000)
    return identity


def _call_input(
    label: str,
    operation: str,
    instance: Any,
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
        arguments = {"args": list(args), "kwargs": kwargs}
    return {
        "operation": f"{label}.{operation}",
        **_instance_identity(instance),
        **arguments,
    }


class WeaviateInstrumentor:
    """Trace Weaviate v4 sync and async collection operations."""

    name = WEAVIATE_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _patched_targets: ClassVar[list[tuple[type, str]]] = []
    _installed_targets: ClassVar[dict[tuple[type, str], Any]] = {}
    _lock = RLock()
    _capture_content_config: bool | None = None
    _active_call: ContextVar[bool] = ContextVar(
        "respan_weaviate_active_call",
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
        operation: str,
        instance: Any,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        has_parent: bool,
    ) -> None:
        operation_name = f"{label}.{operation}"
        entity_name = f"weaviate.{operation_name}"
        span.set_attribute(RESPAN_LOG_TYPE, LOG_TYPE_TASK)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            entity_name if has_parent else "",
        )
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, "weaviate")
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, operation_name)
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                _json_dumps(
                    _call_input(
                        label,
                        operation,
                        instance,
                        wrapped,
                        args,
                        kwargs,
                    )
                ),
            )

    def _set_error(self, span: Any, exc: BaseException) -> None:
        message = _safe_exception_message(exc)
        status_code = _provider_status_code(exc)
        span.record_exception(RuntimeError(message))
        span.set_status(Status(StatusCode.ERROR, message))
        span.set_attribute("status_code", status_code)
        span.set_attribute(OTelSpanAttributes.HTTP_STATUS_CODE, status_code)
        span.set_attribute(ERROR_MESSAGE_ATTR, message)
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_dumps({"error": type(exc).__name__, "message": message}),
            )

    def _trace_sync(
        self,
        label: str,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            parent_context = trace.get_current_span().get_span_context()
            has_parent = bool(getattr(parent_context, "is_valid", False))
            tracer = trace.get_tracer(_SCOPE_NAME, _SCOPE_VERSION)
            with tracer.start_as_current_span(
                f"weaviate.{label}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                self._set_start_attributes(
                    span,
                    label,
                    operation,
                    instance,
                    wrapped,
                    args,
                    kwargs,
                    has_parent,
                )
                try:
                    result = wrapped(*args, **kwargs)
                except BaseException as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    async def _trace_async(
        self,
        label: str,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return await wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            parent_context = trace.get_current_span().get_span_context()
            has_parent = bool(getattr(parent_context, "is_valid", False))
            tracer = trace.get_tracer(_SCOPE_NAME, _SCOPE_VERSION)
            with tracer.start_as_current_span(
                f"weaviate.{label}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                self._set_start_attributes(
                    span,
                    label,
                    operation,
                    instance,
                    wrapped,
                    args,
                    kwargs,
                    has_parent,
                )
                try:
                    result = await wrapped(*args, **kwargs)
                except BaseException as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    def _patch_spec(self, spec: PatchSpec) -> list[tuple[type, str]]:
        try:
            module = importlib.import_module(spec.module)
            target_class = getattr(module, spec.class_name)
        except (ImportError, AttributeError):
            return []

        patched: list[tuple[type, str]] = []
        for operation in spec.methods:
            if not callable(getattr(target_class, operation, None)):
                continue

            def traced(
                wrapped: Any,
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _label: str = spec.label,
                _operation: str = operation,
                _is_async: bool = spec.is_async,
            ) -> Any:
                if _is_async:
                    return self._trace_async(
                        _label,
                        _operation,
                        wrapped,
                        instance,
                        args,
                        kwargs,
                    )
                return self._trace_sync(
                    _label,
                    _operation,
                    wrapped,
                    instance,
                    args,
                    kwargs,
                )

            # Use the already imported module object. Importing a dotted private
            # Weaviate module a second time can re-enter the package's broad
            # ``weaviate.__init__`` import graph during startup.
            wrap_function_wrapper(
                module,
                f"{spec.class_name}.{operation}",
                traced,
            )
            patched.append((target_class, operation))
            type(self)._installed_targets[(target_class, operation)] = (
                inspect.getattr_static(target_class, operation)
            )
        return patched

    def activate(self) -> None:
        """Patch supported Weaviate v4 managers."""
        cls = type(self)
        with cls._lock:
            if self._is_instrumented or not self._tracing_enabled():
                return
            if cls._patches_applied:
                if cls._capture_content_config != self._capture_content:
                    raise ValueError(
                        "Weaviate instrumentation is already active with different capture_content"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return

            patched_targets: list[tuple[type, str]] = []
            try:
                for spec in WEAVIATE_PATCH_SPECS:
                    patched_targets.extend(self._patch_spec(spec))
            except BaseException:
                for target_class, operation in reversed(patched_targets):
                    if inspect.getattr_static(
                        target_class,
                        operation,
                        None,
                    ) is cls._installed_targets.get((target_class, operation)):
                        unwrap(target_class, operation)
                    cls._installed_targets.pop((target_class, operation), None)
                raise

            self._is_instrumented = bool(patched_targets)
            cls._patches_applied = self._is_instrumented
            cls._activation_count = int(self._is_instrumented)
            cls._patched_targets = patched_targets
            cls._capture_content_config = (
                self._capture_content if self._is_instrumented else None
            )
            if not self._is_instrumented:
                logger.warning("Weaviate instrumentation found no supported v4 methods")

    def deactivate(self) -> None:
        """Remove Weaviate patches after the final active instance stops."""
        cls = type(self)
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(cls._activation_count - 1, 0)
            if cls._activation_count:
                return
            for target_class, operation in reversed(cls._patched_targets):
                try:
                    if inspect.getattr_static(
                        target_class,
                        operation,
                        None,
                    ) is cls._installed_targets.get((target_class, operation)):
                        unwrap(target_class, operation)
                except Exception:
                    logger.debug(
                        "Failed to unwrap Weaviate %s.%s",
                        target_class.__name__,
                        operation,
                        exc_info=True,
                    )
                finally:
                    cls._installed_targets.pop((target_class, operation), None)
            cls._patched_targets = []
            cls._patches_applied = False
            cls._capture_content_config = None

    def instrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`activate`."""
        self.activate()

    def uninstrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`deactivate`."""
        self.deactivate()
