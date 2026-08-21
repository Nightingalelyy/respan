"""Native Qdrant client instrumentation for Respan."""

from __future__ import annotations

import importlib
import inspect
import logging
from contextvars import ContextVar
from dataclasses import dataclass
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

from respan_instrumentation_qdrant._constants import (
    QDRANT_INSTRUMENTATION_NAME,
    QDRANT_INSTRUMENTATION_VERSION,
    QDRANT_OPERATIONS,
)
from respan_instrumentation_qdrant._serialization import (
    exception_message,
    json_dumps,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AppliedPatch:
    target: Any
    attribute: str
    installed_wrapper: Any


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


def _http_status_code(exc: BaseException) -> int | None:
    candidates: list[Any] = [exc]
    for name in ("response", "__cause__"):
        try:
            candidates.append(getattr(exc, name, None))
        except Exception:  # noqa: BLE001,S112 - hostile properties are ignored.
            continue
    for candidate in candidates:
        for name in ("status_code", "status"):
            try:
                value = getattr(candidate, name, None)
            except Exception:  # noqa: BLE001,S112 - hostile properties are ignored.
                continue
            if isinstance(value, int) and 400 <= value <= 599:
                return value
    return None


class QdrantInstrumentor:
    """Trace synchronous and asynchronous Qdrant operations as task spans."""

    name: ClassVar[str] = QDRANT_INSTRUMENTATION_NAME
    _patches_applied: ClassVar[bool] = False
    _activation_count: ClassVar[int] = 0
    _patched_targets: ClassVar[list[_AppliedPatch]] = []
    _capture_content_config: ClassVar[bool | None] = None
    _lifecycle_lock: ClassVar[RLock] = RLock()
    _active_call: ClassVar[ContextVar[bool]] = ContextVar(
        "respan_qdrant_active_call",
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
        operation: str,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        entity_name = f"qdrant.{operation}"
        span.set_attribute(RESPAN_LOG_TYPE, LOG_TYPE_TASK)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            "" if getattr(span, "parent", None) is None else entity_name,
        )
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, "qdrant")
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, operation)
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json_dumps(_call_input(operation, wrapped, args, kwargs)),
            )

    def _set_error(self, span: Any, exc: Exception) -> None:
        message = exception_message(exc)
        exception_type = type(exc).__name__
        add_event = getattr(span, "add_event", None)
        if callable(add_event):
            add_event(
                "exception",
                {
                    OTelSpanAttributes.EXCEPTION_MESSAGE: message,
                    OTelSpanAttributes.EXCEPTION_TYPE: exception_type,
                },
            )
        status_code = _http_status_code(exc) or 500
        span.set_status(Status(StatusCode.ERROR, message))
        span.set_attribute("status_code", status_code)
        span.set_attribute(ERROR_MESSAGE_ATTR, message)
        if status_code != 500:
            span.set_attribute("http.response.status_code", status_code)
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_dumps(
                    {
                        "error": exception_type,
                        "message": message,
                        "status_code": status_code,
                    }
                ),
            )

    def _trace_sync(
        self,
        operation: str,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            tracer = trace.get_tracer(
                QDRANT_INSTRUMENTATION_NAME,
                QDRANT_INSTRUMENTATION_VERSION,
            )
            with tracer.start_as_current_span(
                f"qdrant.{operation}",
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self._set_start_attributes(span, operation, wrapped, args, kwargs)
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("status_code", 200)
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    async def _trace_async(
        self,
        operation: str,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return await wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            tracer = trace.get_tracer(
                QDRANT_INSTRUMENTATION_NAME,
                QDRANT_INSTRUMENTATION_VERSION,
            )
            with tracer.start_as_current_span(
                f"qdrant.{operation}",
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self._set_start_attributes(span, operation, wrapped, args, kwargs)
                try:
                    result = await wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("status_code", 200)
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        json_dumps(result),
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
        retained: list[_AppliedPatch] = []
        for patch in reversed(patches):
            try:
                current = inspect.getattr_static(patch.target, patch.attribute)
                if current is patch.installed_wrapper:
                    unwrap(patch.target, patch.attribute)
                    current = inspect.getattr_static(patch.target, patch.attribute)
                if cls._wrapper_chain_contains(current, patch.installed_wrapper):
                    retained.append(patch)
            except Exception:
                retained.append(patch)
                logger.debug(
                    "Failed to unwrap Qdrant %s.%s",
                    getattr(patch.target, "__name__", type(patch.target).__name__),
                    patch.attribute,
                    exc_info=True,
                )
        retained.reverse()
        return retained

    def _patch_targets(self) -> list[_AppliedPatch]:
        module = importlib.import_module("qdrant_client")
        patched: list[_AppliedPatch] = []
        try:
            for class_name in ("QdrantClient", "AsyncQdrantClient"):
                target_class = getattr(module, class_name, None)
                if target_class is None:
                    continue
                for operation in QDRANT_OPERATIONS:
                    original = getattr(target_class, operation, None)
                    if not callable(original):
                        continue
                    is_async = inspect.iscoroutinefunction(original)

                    def traced(
                        wrapped: Any,
                        _instance: Any,
                        args: tuple[Any, ...],
                        kwargs: dict[str, Any],
                        *,
                        _operation: str = operation,
                        _is_async: bool = is_async,
                    ) -> Any:
                        cls = type(self)
                        if not cls._patches_applied or not cls._activation_count:
                            return wrapped(*args, **kwargs)
                        if _is_async:
                            return self._trace_async(_operation, wrapped, args, kwargs)
                        return self._trace_sync(_operation, wrapped, args, kwargs)

                    wrap_function_wrapper(
                        "qdrant_client",
                        f"{class_name}.{operation}",
                        traced,
                    )
                    patched.append(
                        _AppliedPatch(
                            target_class,
                            operation,
                            inspect.getattr_static(target_class, operation),
                        )
                    )
        except Exception:
            type(self)._patched_targets = type(self)._unwrap_targets(patched)
            raise
        return patched

    def activate(self) -> None:
        """Patch supported Qdrant client methods transactionally."""
        cls = type(self)
        with cls._lifecycle_lock:
            if self._is_instrumented or not self._tracing_enabled():
                return
            if cls._patches_applied:
                if cls._capture_content_config != self._capture_content:
                    raise ValueError(
                        "all active QdrantInstrumentor instances must use the same "
                        "capture_content setting"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return

            if cls._patched_targets:
                retained = [
                    patch
                    for patch in cls._patched_targets
                    if cls._wrapper_chain_contains(
                        inspect.getattr_static(patch.target, patch.attribute),
                        patch.installed_wrapper,
                    )
                ]
                if len(retained) == len(cls._patched_targets):
                    if cls._capture_content_config != self._capture_content:
                        raise ValueError(
                            "reactivated Qdrant instrumentation must retain its "
                            "capture_content setting while foreign wrappers own the chain"
                        )
                    cls._patches_applied = True
                    cls._activation_count = 1
                    self._is_instrumented = True
                    return
                cls._patched_targets = cls._unwrap_targets(cls._patched_targets)
                if cls._patched_targets:
                    raise RuntimeError(
                        "Qdrant instrumentation could not reconcile stale wrappers"
                    )
                cls._capture_content_config = None

            try:
                patched = self._patch_targets()
            except ImportError as exc:
                logger.warning(
                    "Qdrant instrumentation dependency is unavailable: %s", exc
                )
                return
            except Exception:
                cls._patches_applied = False
                cls._activation_count = 0
                cls._capture_content_config = (
                    self._capture_content if cls._patched_targets else None
                )
                raise

            self._is_instrumented = bool(patched)
            cls._patches_applied = self._is_instrumented
            cls._activation_count = int(self._is_instrumented)
            cls._patched_targets = patched
            cls._capture_content_config = (
                self._capture_content if self._is_instrumented else None
            )
            if not self._is_instrumented:
                logger.warning(
                    "Qdrant instrumentation found no supported client methods"
                )

    def deactivate(self) -> None:
        """Remove owned Qdrant patches after the final active instance stops."""
        cls = type(self)
        with cls._lifecycle_lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(cls._activation_count - 1, 0)
            if cls._activation_count:
                return
            cls._patched_targets = cls._unwrap_targets(cls._patched_targets)
            cls._patches_applied = False
            if not cls._patched_targets:
                cls._capture_content_config = None

    def instrument(self) -> None:
        self.activate()

    def uninstrument(self) -> None:
        self.deactivate()
