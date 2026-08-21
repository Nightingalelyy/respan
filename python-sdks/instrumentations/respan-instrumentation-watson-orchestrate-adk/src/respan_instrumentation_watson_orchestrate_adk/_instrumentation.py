"""IBM watsonx Orchestrate ADK instrumentation plugin for Respan."""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from opentelemetry import trace
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_watson_orchestrate_adk import _otel_emitter
from respan_instrumentation_watson_orchestrate_adk._constants import (
    AGENT_BUILDER_CLIENT_CLASS,
    AGENT_BUILDER_CLIENT_MODULE,
    ASYNC_RUN_METHODS,
    CHAT_METHODS,
    CHAT_REFINEMENT_METHODS,
    CPE_CLIENT_CLASS,
    CPE_CLIENT_MODULE,
    LLM_CHAT_METHODS,
    PYTHON_TOOL_CLASS,
    PYTHON_TOOL_MODULE,
    RUN_CLIENT_CLASS,
    RUN_CLIENT_MODULE,
    RUN_METHODS,
    TOOL_CALL_METHOD,
    WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME,
    WATSONX_AI_CLIENT_CLASS,
    WATSONX_AI_CLIENT_MODULE,
)
from respan_instrumentation_watson_orchestrate_adk._serialization import (
    provider_status_code,
    safe_exception_message,
    safe_text,
)

logger = logging.getLogger(__name__)

_LOCK = RLock()
_ACTIVATION_COUNT = 0


@dataclass(frozen=True)
class _Patch:
    cls: type[Any]
    method_name: str
    original: Any
    wrapper: Any


_PATCHES: list[_Patch] = []


def _load_class(module_name: str, class_name: str) -> type[Any]:
    module = importlib.import_module(module_name)
    value = getattr(module, class_name, None)
    if value is None:
        raise AttributeError(f"{module_name}.{class_name}")
    return value


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None, None
        return format_trace_id(context.trace_id), format_span_id(context.span_id)
    except BaseException:  # noqa: BLE001
        return None, None


def _safe_attr(instance: Any, name: str) -> Any:
    try:
        return getattr(instance, name, None)
    except BaseException:  # noqa: BLE001
        return None


def _tool_name(instance: Any) -> str:
    for key in ("name", "display_name"):
        value = _safe_attr(instance, key)
        if value:
            return safe_text(value, max_bytes=256)
    spec = _safe_attr(instance, "__tool_spec__")
    value = _safe_attr(spec, "name")
    if value:
        return safe_text(value, max_bytes=256)
    fn = _safe_attr(instance, "fn")
    value = _safe_attr(fn, "__name__")
    if value:
        return safe_text(value, max_bytes=256)
    return safe_text(type(instance).__name__, max_bytes=256)


def _call_kwargs(
    *,
    original: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(original).bind_partial(instance, *args, **kwargs)
    except (TypeError, ValueError):
        result = dict(kwargs)
        if args:
            result["_args"] = list(args[:50])
        return result
    result = {key: value for key, value in bound.arguments.items() if key != "self"}
    nested_kwargs = result.pop("kwargs", None)
    if isinstance(nested_kwargs, dict):
        nested = dict(nested_kwargs)
        nested.update(result)
        result = nested
    positional = result.pop("args", None)
    if isinstance(positional, tuple) and positional:
        result["_args"] = list(positional[:50])
    return result


def _emit_error_kwargs(exc: BaseException) -> dict[str, Any]:
    return {
        "error_message": safe_exception_message(exc),
        "status_code": provider_status_code(exc),
    }


def _wrap_tool_call(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        tool_name = _tool_name(self)
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            _otel_emitter.emit_tool_span(
                tool_name=tool_name,
                args=args,
                kwargs=kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                **_emit_error_kwargs(exc),
            )
            raise
        _otel_emitter.emit_tool_span(
            tool_name=tool_name,
            args=args,
            kwargs=kwargs,
            start_ns=start_ns,
            response=response,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


def _wrap_agent_run(
    method_name: str, original: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        call_kwargs = _call_kwargs(
            original=original, instance=self, args=args, kwargs=kwargs
        )
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            _otel_emitter.emit_agent_run_span(
                method_name=method_name,
                call_kwargs=call_kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                **_emit_error_kwargs(exc),
            )
            raise
        _otel_emitter.emit_agent_run_span(
            method_name=method_name,
            call_kwargs=call_kwargs,
            start_ns=start_ns,
            response=response,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


def _wrap_async_agent_run(
    method_name: str, original: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        call_kwargs = _call_kwargs(
            original=original, instance=self, args=args, kwargs=kwargs
        )
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            _otel_emitter.emit_agent_run_span(
                method_name=method_name,
                call_kwargs=call_kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                **_emit_error_kwargs(exc),
            )
            raise
        _otel_emitter.emit_agent_run_span(
            method_name=method_name,
            call_kwargs=call_kwargs,
            start_ns=start_ns,
            response=response,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


def _wrap_chat_method(
    method_name: str, original: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        call_kwargs = _call_kwargs(
            original=original, instance=self, args=args, kwargs=kwargs
        )
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            _otel_emitter.emit_chat_span(
                method_name=method_name,
                call_kwargs=call_kwargs,
                start_ns=start_ns,
                instance=self,
                trace_id=trace_id,
                parent_id=parent_id,
                **_emit_error_kwargs(exc),
            )
            raise
        _otel_emitter.emit_chat_span(
            method_name=method_name,
            call_kwargs=call_kwargs,
            start_ns=start_ns,
            response=response,
            instance=self,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


def _install(
    installed: list[_Patch],
    cls: type[Any],
    method_name: str,
    factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return
    wrapper = factory(original)
    setattr(cls, method_name, wrapper)
    installed.append(_Patch(cls, method_name, original, wrapper))


def _optional_class(module_name: str, class_name: str) -> type[Any] | None:
    try:
        return _load_class(module_name, class_name)
    except (ImportError, AttributeError):
        return None


def _restore_owned(patches: list[_Patch]) -> None:
    for patch in reversed(patches):
        if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
            setattr(patch.cls, patch.method_name, patch.original)


class WatsonOrchestrateADKInstrumentor:
    """Respan instrumentor for IBM watsonx Orchestrate ADK."""

    name = WATSON_ORCHESTRATE_ADK_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        global _ACTIVATION_COUNT
        with _LOCK:
            if self._is_instrumented:
                return
            if not self._is_respan_tracing_enabled():
                return
            if _ACTIVATION_COUNT:
                _ACTIVATION_COUNT += 1
                self._is_instrumented = True
                return
            installed: list[_Patch] = []
            try:
                tool = _optional_class(PYTHON_TOOL_MODULE, PYTHON_TOOL_CLASS)
                if tool is not None:
                    _install(installed, tool, TOOL_CALL_METHOD, _wrap_tool_call)
                run_client = _optional_class(RUN_CLIENT_MODULE, RUN_CLIENT_CLASS)
                if run_client is not None:
                    for method_name in RUN_METHODS:
                        _install(
                            installed,
                            run_client,
                            method_name,
                            lambda original, name=method_name: _wrap_agent_run(
                                name, original
                            ),
                        )
                    for method_name in ASYNC_RUN_METHODS:
                        _install(
                            installed,
                            run_client,
                            method_name,
                            lambda original, name=method_name: _wrap_async_agent_run(
                                name, original
                            ),
                        )
                for module_name, class_name, methods in (
                    (
                        AGENT_BUILDER_CLIENT_MODULE,
                        AGENT_BUILDER_CLIENT_CLASS,
                        CHAT_METHODS,
                    ),
                    (
                        CPE_CLIENT_MODULE,
                        CPE_CLIENT_CLASS,
                        (*CHAT_METHODS, *CHAT_REFINEMENT_METHODS),
                    ),
                    (
                        WATSONX_AI_CLIENT_MODULE,
                        WATSONX_AI_CLIENT_CLASS,
                        LLM_CHAT_METHODS,
                    ),
                ):
                    cls = _optional_class(module_name, class_name)
                    if cls is None:
                        continue
                    for method_name in methods:
                        _install(
                            installed,
                            cls,
                            method_name,
                            lambda original, name=method_name: _wrap_chat_method(
                                name, original
                            ),
                        )
            except BaseException:
                _restore_owned(installed)
                raise
            if not installed:
                logger.warning(
                    "Watson Orchestrate ADK instrumentation found no supported SDK classes"
                )
                return
            _PATCHES[:] = installed
            _ACTIVATION_COUNT = 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        global _ACTIVATION_COUNT
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            if _ACTIVATION_COUNT:
                return
            _restore_owned(_PATCHES)
            _PATCHES.clear()


def _restore_methods() -> None:
    """Compatibility test helper; restore only wrappers owned by this runtime."""
    global _ACTIVATION_COUNT
    with _LOCK:
        _restore_owned(_PATCHES)
        _PATCHES.clear()
        _ACTIVATION_COUNT = 0
