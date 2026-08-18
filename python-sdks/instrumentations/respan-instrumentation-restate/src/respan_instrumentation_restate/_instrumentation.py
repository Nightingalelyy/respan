"""Instrument Restate handlers through the SDK's invocation context managers."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes
from wrapt import FunctionWrapper

from respan_instrumentation_restate._constants import (
    RESTATE_CONTEXT_MANAGER_MARKER,
    RESTATE_INSTRUMENTATION_NAME,
    RESTATE_REGISTRATION_TARGETS,
)
from respan_instrumentation_restate._serialization import (
    exception_message,
    exception_status,
    json_string,
    json_value,
    safe_text,
    sensitive_key,
)

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_ENABLED = False
_CAPTURE_CONTENT = True


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


_PATCHED_TARGETS: list[_Patch] = []


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - Restate objects are untrusted input.
        return default


def _deserialize_input(context: Any) -> Any:
    invocation = _safe_attr(context, "invocation")
    handler = _safe_attr(context, "handler")
    handler_io = _safe_attr(handler, "handler_io")
    try:
        return handler_io.input_serde.deserialize(invocation.input_buffer)
    except Exception:  # noqa: BLE001 - invalid input falls back to a typed summary.
        return json_value(_safe_attr(invocation, "input_buffer"))


def _invocation_details(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    handler = _safe_attr(context, "handler")
    service_tag = _safe_attr(handler, "service_tag")
    invocation = _safe_attr(context, "invocation")
    server_context = importlib.import_module("restate.server_context")
    replaying_var = getattr(server_context, "restate_context_is_replaying", None)
    is_replaying = bool(replaying_var.get()) if replaying_var is not None else False

    metadata = {
        "service_kind": safe_text(_safe_attr(service_tag, "kind")),
        "service_name": safe_text(_safe_attr(service_tag, "name")),
        "handler_name": safe_text(_safe_attr(handler, "name")),
        "handler_kind": safe_text(_safe_attr(handler, "kind")),
        "invocation_id": safe_text(_safe_attr(invocation, "invocation_id")),
        "replaying": is_replaying,
    }
    for name in ("key", "scope", "limit_key", "idempotency_key"):
        value = _safe_attr(invocation, name)
        if value:
            metadata[name] = safe_text(value)
    service_metadata = _safe_attr(service_tag, "metadata")
    if service_metadata:
        metadata["service_metadata"] = json_value(service_metadata)
    handler_metadata = _safe_attr(handler, "metadata")
    if handler_metadata:
        metadata["handler_metadata"] = json_value(handler_metadata)

    input_payload = dict(metadata)
    if _CAPTURE_CONTENT:
        input_payload["input"] = _deserialize_input(context)
    return metadata, input_payload


def _log_type(context: Any) -> str:
    handler = _safe_attr(context, "handler")
    service_kind = safe_text(_safe_attr(_safe_attr(handler, "service_tag"), "kind"))
    handler_kind = safe_text(_safe_attr(handler, "kind"))
    return (
        "workflow"
        if service_kind == "workflow" and handler_kind == "workflow"
        else "task"
    )


def _span_attributes(
    context: Any,
    *,
    metadata: dict[str, Any],
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    handler = _safe_attr(context, "handler")
    invocation = _safe_attr(context, "invocation")
    service_name = safe_text(_safe_attr(_safe_attr(handler, "service_tag"), "name"))
    handler_name = safe_text(_safe_attr(handler, "name"))
    entity_name = f"{service_name}.{handler_name}"
    current = trace.get_current_span()
    try:
        has_parent = bool(current.get_span_context().is_valid)
    except Exception:  # noqa: BLE001
        has_parent = False
    attrs: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: _log_type(context),
        RESPAN_TRACE_GROUP_ID: safe_text(_safe_attr(invocation, "invocation_id")),
        RESPAN_METADATA: json_string({"restate": metadata}),
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_name if has_parent else "",
    }
    if _CAPTURE_CONTENT:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(input_payload)
    key = _safe_attr(invocation, "key")
    if key:
        attrs[RESPAN_THREADS_ID] = safe_text(key)

    propagated = read_propagated_attributes()
    aggregate: dict[str, Any] = {"restate": metadata}
    for attr_key, value in propagated.items():
        if attr_key == RESPAN_METADATA:
            continue
        if attr_key.startswith(f"{RESPAN_METADATA}."):
            metadata_key = attr_key.removeprefix(f"{RESPAN_METADATA}.")
            safe_value = (
                "[REDACTED]" if sensitive_key(metadata_key) else json_value(value)
            )
            aggregate[metadata_key] = safe_value
            if isinstance(safe_value, str | bool | int | float):
                attrs[attr_key] = safe_value
            else:
                attrs[attr_key] = json_string(safe_value)
        elif attr_key in {RESPAN_TRACE_GROUP_ID, RESPAN_THREADS_ID}:
            attrs[attr_key] = safe_text(value)
    attrs[RESPAN_METADATA] = json_string(aggregate)
    return attrs


@asynccontextmanager
async def _invocation_context():
    """Create one span around a Restate handler invocation attempt."""

    if not _ENABLED:
        yield
        return

    server_context = importlib.import_module("restate.server_context")
    context = server_context.current_context()
    if context is None:
        yield
        return

    metadata, input_payload = _invocation_details(context)
    attrs = _span_attributes(
        context,
        metadata=metadata,
        input_payload=input_payload,
    )
    try:
        version = importlib.metadata.version("respan-instrumentation-restate")
    except importlib.metadata.PackageNotFoundError:
        version = None
    tracer = trace.get_tracer(RESTATE_INSTRUMENTATION_NAME, version)
    span_name = (
        f"restate.{context.handler.service_tag.kind}."
        f"{context.handler.service_tag.name}.{context.handler.name}"
    )
    with tracer.start_as_current_span(
        span_name,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield
        except BaseException as exc:
            message = exception_message(exc)
            span.set_status(Status(StatusCode.ERROR, message))
            span.set_attribute("status_code", exception_status(exc))
            span.set_attribute(ERROR_MESSAGE_ATTR, message)
            if _CAPTURE_CONTENT:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json_string(
                        {
                            "status": "error",
                            "error": type(exc).__name__,
                            "message": message,
                        }
                    ),
                )
            span.add_event(
                "exception",
                {
                    "exception.type": f"{type(exc).__module__}.{type(exc).__name__}",
                    "exception.message": message,
                },
            )
            raise
        else:
            span.set_status(Status(StatusCode.OK))
            span.set_attribute("status_code", 200)
            if _CAPTURE_CONTENT:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json_string({"status": "completed"}),
                )


setattr(_invocation_context, RESTATE_CONTEXT_MANAGER_MARKER, True)


def _ensure_context_manager(instance: Any) -> None:
    managers = list(getattr(instance, "context_managers", None) or ())
    if not any(
        bool(getattr(manager, RESTATE_CONTEXT_MANAGER_MARKER, False))
        for manager in managers
    ):
        managers.append(_invocation_context)
        instance.context_managers = managers


def _registration_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    _ensure_context_manager(instance)
    return wrapped(*args, **kwargs)


def _install_patches() -> None:
    for module_path, target in RESTATE_REGISTRATION_TARGETS:
        module = importlib.import_module(module_path)
        owner: Any = module
        owner_path, name = target.rsplit(".", maxsplit=1)
        for component in owner_path.split("."):
            owner = getattr(owner, component)
        original = getattr(owner, name)
        replacement = FunctionWrapper(original, _registration_wrapper)
        setattr(owner, name, replacement)
        _PATCHED_TARGETS.append(_Patch(owner, name, original, replacement))


def _remove_patches() -> None:
    for patch in reversed(_PATCHED_TARGETS):
        try:
            if getattr(patch.owner, patch.name, None) is patch.replacement:
                setattr(patch.owner, patch.name, patch.original)
        except Exception:  # noqa: BLE001 - foreign-safe best-effort restore.
            logger.debug("Failed to restore Restate target %s", patch.name)
    _PATCHED_TARGETS.clear()


class RestateInstrumentor:
    """Inject canonical Respan spans into Restate handler invocations."""

    name = RESTATE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        """Patch Restate handler registration to add an invocation context."""
        global _ACTIVATION_COUNT, _CAPTURE_CONTENT, _ENABLED

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("restate")
        except ImportError as exc:
            logger.warning("Restate instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if _ACTIVATION_COUNT == 0:
                _CAPTURE_CONTENT = self._capture_content
                try:
                    _install_patches()
                except Exception:
                    _remove_patches()
                    raise
                _ENABLED = True
            elif _CAPTURE_CONTENT != self._capture_content:
                raise ValueError(
                    "all active RestateInstrumentor instances must use the same "
                    "capture_content setting"
                )
            _ACTIVATION_COUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        """Restore Restate registration methods and disable injected contexts."""
        global _ACTIVATION_COUNT, _ENABLED

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            if _ACTIVATION_COUNT:
                return
            _ENABLED = False
            _remove_patches()
