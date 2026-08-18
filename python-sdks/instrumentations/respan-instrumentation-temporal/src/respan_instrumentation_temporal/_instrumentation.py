"""Temporal's official tracing interceptor adapted to the Respan contract."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import re
from collections.abc import Mapping
from contextlib import contextmanager
from threading import RLock
from typing import Any

from opentelemetry import baggage, trace
from opentelemetry import context as otel_context
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_TRACE_GROUP_ID
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes

from respan_instrumentation_temporal._constants import (
    MAX_ATTRIBUTE_CHARS,
    TASK_LOG_TYPE,
    TEMPORAL_CAPTURED_INPUT,
    TEMPORAL_CLIENT_CONNECT_TARGET,
    TEMPORAL_CLIENT_MODULE,
    TEMPORAL_INSTRUMENTATION_NAME,
    TEMPORAL_OTEL_MODULE,
    TEMPORAL_RAW_ATTRIBUTE_KEYS,
    WORKFLOW_LOG_TYPE,
    WORKFLOW_OPERATION_PREFIXES,
)
from respan_instrumentation_temporal._serialization import (
    json_dumps,
    safe_baggage_value,
    safe_error_message,
)

logger = logging.getLogger(__name__)

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_SAFE_DETAIL = re.compile(r"[^a-zA-Z0-9_.-]+")
_MISSING = object()
_INTERNAL_WORKFLOW_ID = "__respan_temporal_workflow_id__"
_CLIENT_CONNECT_ATTRIBUTE = TEMPORAL_CLIENT_CONNECT_TARGET.rsplit(".", 1)[-1]
_RESPAN_BAGGAGE_PREFIX = "respan."


def _instrumentation_version() -> str | None:
    try:
        return importlib.metadata.version("respan-instrumentation-temporal")
    except importlib.metadata.PackageNotFoundError:
        return None


def _context_with_respan_baggage(context: Any) -> Any:
    propagated = read_propagated_attributes()
    if not propagated:
        return context
    result = context if context is not None else otel_context.get_current()
    for key, value in propagated.items():
        if isinstance(key, str) and key.startswith(_RESPAN_BAGGAGE_PREFIX):
            result = baggage.set_baggage(
                key,
                safe_baggage_value(key, value),
                context=result,
            )
    return result


def _apply_respan_baggage(attrs: dict[str, Any], context: Any) -> None:
    try:
        values = baggage.get_all(context=context)
    except BaseException:  # noqa: BLE001 - propagation must remain fail-open
        return
    for key, value in values.items():
        if isinstance(key, str) and key.startswith(_RESPAN_BAGGAGE_PREFIX):
            attrs.setdefault(key, safe_baggage_value(key, value))


def _json_dumps(value: Any, *, max_chars: int = MAX_ATTRIBUTE_CHARS) -> str:
    return json_dumps(value, max_bytes=max_chars)


def _snake_case(value: str) -> str:
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", value)
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    return value.replace("-", "_").lower()


def _span_parts(name: str) -> tuple[str, str | None]:
    operation, separator, detail = name.partition(":")
    return operation, detail if separator and detail else None


def _safe_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    cleaned = _SAFE_DETAIL.sub("_", detail).strip("_.-")
    return cleaned[:120] or None


def _extract_temporal_input(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in (
        "args",
        "arg",
        "id",
        "workflow",
        "workflow_type",
        "activity",
        "activity_type",
        "query",
        "signal",
        "update",
        "update_id",
        "task_queue",
    ):
        item = getattr(value, field_name, _MISSING)
        if item is not _MISSING and item is not None and not callable(item):
            payload[field_name] = item
    return payload


def _canonical_attributes(
    name: str,
    attributes: Mapping[str, Any] | None,
    *,
    capture_content: bool,
    max_attribute_chars: int,
) -> dict[str, Any]:
    source = dict(attributes or {})
    captured_input = source.pop(TEMPORAL_CAPTURED_INPUT, None)
    temporal_attributes = {
        key: source.pop(key)
        for key in tuple(source)
        if key in TEMPORAL_RAW_ATTRIBUTE_KEYS
    }
    workflow_id = temporal_attributes.get("temporalWorkflowID")
    operation, detail = _span_parts(name)
    safe_detail = _safe_detail(detail)
    operation_name = _snake_case(operation)
    entity_name = f"temporal.{operation_name}"
    if safe_detail:
        entity_name = f"{entity_name}.{safe_detail}"
    log_type = (
        WORKFLOW_LOG_TYPE if operation in WORKFLOW_OPERATION_PREFIXES else TASK_LOG_TYPE
    )

    input_payload: dict[str, Any] = {
        "operation": operation_name,
        "detail": detail,
        "content_captured": capture_content,
    }
    if capture_content:
        if temporal_attributes:
            input_payload["temporal"] = temporal_attributes
        if captured_input:
            input_payload["input"] = captured_input

    source[RESPAN_LOG_TYPE] = log_type
    if isinstance(workflow_id, str) and workflow_id:
        source[_INTERNAL_WORKFLOW_ID] = workflow_id
    source[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    source[SpanAttributes.TRACELOOP_ENTITY_PATH] = entity_name
    if log_type == WORKFLOW_LOG_TYPE and safe_detail:
        source[RESPAN_TRACE_GROUP_ID] = safe_detail
    source[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_dumps(
        input_payload, max_chars=max_attribute_chars
    )
    source[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_dumps(
        {"status": "completed", "content_captured": capture_content},
        max_chars=max_attribute_chars,
    )
    return source


def _set_success_output(span: Any, *, capture_content: bool, max_chars: int) -> None:
    span.set_attribute("status_code", 200)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        _json_dumps(
            {"status": "completed", "content_captured": capture_content},
            max_chars=max_chars,
        ),
    )


def _set_error_output(
    span: Any,
    exc: BaseException,
    *,
    capture_content: bool,
    max_chars: int,
    record_exception: bool,
) -> None:
    message = safe_error_message(exc, capture_content=capture_content)
    span.set_status(Status(StatusCode.ERROR, message))
    span.set_attribute("status_code", 500)
    span.set_attribute("error.message", message)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        _json_dumps(
            {
                "status": "error",
                "error": type(exc).__name__,
                "message": message,
                "content_captured": capture_content,
            },
            max_chars=max_chars,
        ),
    )


class _CanonicalSpanProxy:
    def __init__(
        self,
        span: Any,
        *,
        capture_content: bool,
        max_attribute_chars: int,
        on_end: Any = None,
    ) -> None:
        self._span = span
        self._capture_content = capture_content
        self._max_attribute_chars = max_attribute_chars
        self._has_error = False
        self._on_end = on_end
        self._ended = False

    def record_exception(self, exception: Exception, *args: Any, **kwargs: Any) -> None:
        self._has_error = True
        _set_error_output(
            self._span,
            exception,
            capture_content=self._capture_content,
            max_chars=self._max_attribute_chars,
            record_exception=False,
        )

    def end(self, *args: Any, **kwargs: Any) -> None:
        if self._ended:
            return
        self._ended = True
        if not self._has_error:
            _set_success_output(
                self._span,
                capture_content=self._capture_content,
                max_chars=self._max_attribute_chars,
            )
        try:
            self._span.end(*args, **kwargs)
        finally:
            if self._on_end is not None:
                self._on_end()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class _CanonicalTracer:
    def __init__(
        self, tracer: Any, *, capture_content: bool, max_attribute_chars: int
    ) -> None:
        self._tracer = tracer
        self._capture_content = capture_content
        self._max_attribute_chars = max_attribute_chars
        self._workflow_groups: dict[int, str] = {}
        self._workflow_contexts: dict[str, Any] = {}

    def _apply_parent_and_group(
        self, name: str, attrs: dict[str, Any], context: Any
    ) -> tuple[Any, str | None]:
        _apply_respan_baggage(attrs, context)
        workflow_id = attrs.pop(_INTERNAL_WORKFLOW_ID, None)
        parent = trace.get_current_span(context)
        parent_context = parent.get_span_context()
        if not parent_context.is_valid and isinstance(workflow_id, str):
            fallback_context = self._workflow_contexts.get(workflow_id)
            if fallback_context is not None:
                context = trace.set_span_in_context(
                    trace.NonRecordingSpan(fallback_context)
                )
                parent_context = fallback_context
        if not parent_context.is_valid:
            attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = ""
        group = attrs.get(RESPAN_TRACE_GROUP_ID)
        if not group and parent_context.is_valid:
            group = self._workflow_groups.get(parent_context.trace_id)
            if group:
                attrs[RESPAN_TRACE_GROUP_ID] = group
        return context, workflow_id if isinstance(workflow_id, str) else None

    @contextmanager
    def start_as_current_span(self, name: str, *args: Any, **kwargs: Any):
        attrs = _canonical_attributes(
            name,
            kwargs.get("attributes"),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
        )
        context, workflow_id = self._apply_parent_and_group(
            name, attrs, kwargs.get("context")
        )
        if context is not None:
            kwargs["context"] = context
        kwargs["attributes"] = attrs
        with self._tracer.start_as_current_span(name, *args, **kwargs) as span:
            span_context = span.get_span_context()
            group = attrs.get(RESPAN_TRACE_GROUP_ID)
            if group and span_context.is_valid:
                self._workflow_groups[span_context.trace_id] = group
            if (
                workflow_id
                and span_context.is_valid
                and name.startswith(("StartWorkflow:", "StartActivity:"))
            ):
                self._workflow_contexts[workflow_id] = span_context
            try:
                yield span
            except BaseException as exc:
                _set_error_output(
                    span,
                    exc,
                    capture_content=self._capture_content,
                    max_chars=self._max_attribute_chars,
                    record_exception=True,
                )
                raise
            else:
                _set_success_output(
                    span,
                    capture_content=self._capture_content,
                    max_chars=self._max_attribute_chars,
                )
            finally:
                if workflow_id and name.startswith("CompleteWorkflow:"):
                    self._workflow_contexts.pop(workflow_id, None)

    def start_span(self, name: str, *args: Any, **kwargs: Any) -> _CanonicalSpanProxy:
        attrs = _canonical_attributes(
            name,
            kwargs.get("attributes"),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
        )
        context, workflow_id = self._apply_parent_and_group(
            name, attrs, kwargs.get("context")
        )
        if context is not None:
            kwargs["context"] = context
        kwargs["attributes"] = attrs
        span = self._tracer.start_span(name, *args, **kwargs)
        span_context = span.get_span_context()
        group = attrs.get(RESPAN_TRACE_GROUP_ID)
        if group and span_context.is_valid:
            self._workflow_groups[span_context.trace_id] = group
        if (
            workflow_id
            and span_context.is_valid
            and name.startswith(("StartWorkflow:", "StartActivity:"))
        ):
            self._workflow_contexts[workflow_id] = span_context
        on_end = None
        if workflow_id and name.startswith("CompleteWorkflow:"):
            on_end = lambda: self._workflow_contexts.pop(workflow_id, None)
        return _CanonicalSpanProxy(
            span,
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
            on_end=on_end,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)


def _build_interceptor(
    base_class: type,
    *,
    tracer: Any,
    capture_content: bool,
    max_attribute_chars: int,
    always_create_workflow_spans: bool,
) -> Any:
    canonical_tracer = _CanonicalTracer(
        tracer,
        capture_content=capture_content,
        max_attribute_chars=max_attribute_chars,
    )

    class RespanTemporalTracingInterceptor(base_class):
        @contextmanager
        def _start_as_current_span(
            self,
            name: str,
            *,
            attributes: Mapping[str, Any] | None,
            input_with_headers: Any = None,
            input_with_ctx: Any = None,
            kind: Any,
            context: Any = None,
        ):
            enriched = dict(attributes or {})
            if capture_content:
                captured: dict[str, Any] = {}
                if input_with_headers is not None:
                    captured.update(_extract_temporal_input(input_with_headers))
                if input_with_ctx is not None:
                    captured.update(_extract_temporal_input(input_with_ctx))
                if captured:
                    enriched[TEMPORAL_CAPTURED_INPUT] = captured
            context = _context_with_respan_baggage(context)
            with super()._start_as_current_span(
                name,
                attributes=enriched,
                input_with_headers=input_with_headers,
                input_with_ctx=input_with_ctx,
                kind=kind,
                context=context,
            ):
                yield None

    RespanTemporalTracingInterceptor.__name__ = "RespanTemporalTracingInterceptor"
    return RespanTemporalTracingInterceptor(
        tracer=canonical_tracer,
        always_create_workflow_spans=always_create_workflow_spans,
    )


class TemporalInstrumentor:
    """Inject a canonicalized official Temporal tracing interceptor."""

    name = TEMPORAL_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _lock = RLock()
    _shared_config: tuple[bool, bool, int] | None = None
    _client_class: type[Any] | None = None
    _original_connect_descriptor_holder: tuple[Any, ...] = ()
    _installed_connect_function: Any = None
    _patch_generation = 0

    def __init__(
        self,
        *,
        capture_content: bool = True,
        always_create_workflow_spans: bool = False,
        max_attribute_chars: int = MAX_ATTRIBUTE_CHARS,
    ) -> None:
        self._capture_content = capture_content
        self._always_create_workflow_spans = always_create_workflow_spans
        self._max_attribute_chars = max(512, int(max_attribute_chars))
        self._is_instrumented = False
        self._interceptor: Any = None
        self._base_interceptor_class: type | None = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _ensure_interceptor(self) -> Any:
        if self._interceptor is not None:
            return self._interceptor
        otel_module = importlib.import_module(TEMPORAL_OTEL_MODULE)
        self._base_interceptor_class = otel_module.TracingInterceptor
        self._interceptor = _build_interceptor(
            self._base_interceptor_class,
            tracer=trace.get_tracer(
                TEMPORAL_INSTRUMENTATION_NAME,
                _instrumentation_version(),
            ),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
            always_create_workflow_spans=self._always_create_workflow_spans,
        )
        return self._interceptor

    @property
    def interceptor(self) -> Any:
        """The interceptor for explicit Temporal client/test-environment wiring."""
        return self._ensure_interceptor()

    async def _connect(
        self,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        interceptor = self._ensure_interceptor()
        connect_kwargs = dict(kwargs)
        interceptors = list(connect_kwargs.get("interceptors") or ())
        base_class = self._base_interceptor_class
        has_temporal_tracing = bool(
            base_class is not None
            and any(isinstance(candidate, base_class) for candidate in interceptors)
        )
        if not has_temporal_tracing:
            interceptors.append(interceptor)
            connect_kwargs["interceptors"] = interceptors
        return await wrapped(*args, **connect_kwargs)

    def activate(self) -> None:
        """Patch `Client.connect` to inject the Respan Temporal interceptor."""
        cls = type(self)
        if not self._is_respan_tracing_enabled():
            logger.info(
                "Temporal instrumentation skipped because Respan tracing is disabled"
            )
            return
        with cls._lock:
            if self._is_instrumented:
                return
            config = (
                self._capture_content,
                self._always_create_workflow_spans,
                self._max_attribute_chars,
            )
            if cls._patches_applied:
                if cls._shared_config != config:
                    raise ValueError(
                        "Temporal instrumentation is already active with different settings"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return
            try:
                client_module = importlib.import_module(TEMPORAL_CLIENT_MODULE)
                client_class = getattr(client_module, "Client", None)
                if client_class is None or not hasattr(client_class, "connect"):
                    logger.warning("Temporal Client.connect is unavailable")
                    return
                self._ensure_interceptor()

                original_descriptor = client_class.__dict__.get(
                    _CLIENT_CONNECT_ATTRIBUTE
                )
                original_connect = getattr(client_class, _CLIENT_CONNECT_ATTRIBUTE)
                cls._patch_generation += 1
                generation = cls._patch_generation

                async def traced_connect(
                    client_cls: type[Any], *args: Any, **kwargs: Any
                ) -> Any:
                    if (
                        type(self)._activation_count == 0
                        or type(self)._patch_generation != generation
                    ):
                        return await original_connect(*args, **kwargs)
                    return await self._connect(original_connect, args, kwargs)

                installed_descriptor = classmethod(traced_connect)
                setattr(
                    client_class,
                    _CLIENT_CONNECT_ATTRIBUTE,
                    installed_descriptor,
                )
                cls._client_class = client_class
                cls._original_connect_descriptor_holder = (original_descriptor,)
                cls._installed_connect_function = traced_connect
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Temporal instrumentation - missing dependency: %s",
                    exc,
                )
                return
            except Exception:
                logger.exception("Failed to activate Temporal instrumentation")
                return
            cls._patches_applied = True
            cls._activation_count = 1
            cls._shared_config = config
            self._is_instrumented = True
            logger.info("Temporal instrumentation activated")

    def deactivate(self) -> None:
        """Restore `Client.connect`; existing clients retain their interceptor."""
        cls = type(self)
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(cls._activation_count - 1, 0)
            if cls._activation_count:
                return
            if (
                cls._client_class is not None
                and getattr(
                    cls._client_class.__dict__.get(_CLIENT_CONNECT_ATTRIBUTE),
                    "__func__",
                    None,
                )
                is cls._installed_connect_function
            ):
                setattr(
                    cls._client_class,
                    _CLIENT_CONNECT_ATTRIBUTE,
                    cls._original_connect_descriptor_holder[0],
                )
            cls._client_class = None
            cls._original_connect_descriptor_holder = ()
            cls._installed_connect_function = None
            cls._patches_applied = False
            cls._shared_config = None
            logger.info("Temporal instrumentation deactivated")
