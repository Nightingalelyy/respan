"""Vertex AI SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from types import TracebackType
from typing import Any, Self

from opentelemetry import trace
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_vertexai._constants import (
    CHAT_SESSION_CLASS_NAME,
    GENERATE_CONTENT_ASYNC_METHOD_NAME,
    GENERATE_CONTENT_METHOD_NAME,
    GENERATIVE_MODEL_CLASS_NAME,
    SEND_MESSAGE_ASYNC_METHOD_NAME,
    SEND_MESSAGE_METHOD_NAME,
    VERTEXAI_CHAT_SPAN_NAME,
    VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
    VERTEXAI_GENERATIVE_MODELS_MODULE,
    VERTEXAI_INSTRUMENTATION_NAME,
)
from respan_instrumentation_vertexai._otel_emitter import emit_generate_content_span
from respan_instrumentation_vertexai._serialization import (
    provider_status_code,
    safe_exception_message,
)
from respan_instrumentation_vertexai._translator import request_payload_from_call

logger = logging.getLogger(__name__)

_LOCK = RLock()
_ACTIVATION_COUNT = 0
_MAX_STREAM_CHUNKS = 50


@dataclass(frozen=True)
class _Patch:
    cls: type[Any]
    method_name: str
    original: Any
    wrapper: Any


_PATCHES: list[_Patch] = []


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name, None)
    if value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return value


def _load_vertexai_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(
            VERTEXAI_GENERATIVE_MODELS_MODULE, GENERATIVE_MODEL_CLASS_NAME
        ),
        _get_module_attr(VERTEXAI_GENERATIVE_MODELS_MODULE, CHAT_SESSION_CLASS_NAME),
    )


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None, None
        return format_trace_id(context.trace_id), format_span_id(context.span_id)
    except BaseException:  # noqa: BLE001
        return None, None


def _emit(
    *,
    request_payload: dict[str, Any],
    start_ns: int,
    span_name: str,
    trace_id: str | None,
    parent_id: str | None,
    response_or_chunks: Any = None,
    error: BaseException | None = None,
) -> None:
    emit_generate_content_span(
        request_payload=request_payload,
        start_ns=start_ns,
        response_or_chunks=response_or_chunks,
        span_name=span_name,
        error_message=safe_exception_message(error) if error is not None else None,
        status_code=provider_status_code(error) if error is not None else 200,
        trace_id=trace_id,
        parent_id=parent_id,
    )


class _StreamState:
    def __init__(
        self,
        *,
        request_payload: dict[str, Any],
        start_ns: int,
        span_name: str,
        trace_id: str | None,
        parent_id: str | None,
    ) -> None:
        self.request_payload = request_payload
        self.start_ns = start_ns
        self.span_name = span_name
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.chunks: list[Any] = []
        self.emitted = False

    def capture(self, chunk: Any) -> None:
        if len(self.chunks) < _MAX_STREAM_CHUNKS:
            self.chunks.append(chunk)
        else:
            self.chunks[-1] = chunk

    def emit(self, error: BaseException | None = None) -> None:
        if self.emitted:
            return
        self.emitted = True
        _emit(
            request_payload=self.request_payload,
            start_ns=self.start_ns,
            span_name=self.span_name,
            trace_id=self.trace_id,
            parent_id=self.parent_id,
            response_or_chunks=self.chunks,
            error=error,
        )


class _SyncStreamProxy:
    def __init__(self, source: Any, state: _StreamState) -> None:
        self._source = source
        self._iterator = iter(source)
        self._state = state

    def __iter__(self) -> _SyncStreamProxy:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._state.emit()
            raise
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.capture(chunk)
        return chunk

    def __enter__(self) -> Self:
        enter = getattr(self._source, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        exit_method = getattr(self._source, "__exit__", None)
        try:
            result = exit_method(exc_type, exc, tb) if callable(exit_method) else None
        except BaseException as close_error:
            self._state.emit(close_error)
            raise
        self._state.emit(exc)
        return result

    def close(self) -> None:
        close = getattr(self._source, "close", None)
        try:
            if callable(close):
                close()
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.emit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class _AsyncStreamProxy:
    def __init__(self, source: Any, state: _StreamState) -> None:
        self._source = source
        self._iterator = source.__aiter__()
        self._state = state

    def __aiter__(self) -> _AsyncStreamProxy:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._state.emit()
            raise
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.capture(chunk)
        return chunk

    async def __aenter__(self) -> Self:
        enter = getattr(self._source, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        exit_method = getattr(self._source, "__aexit__", None)
        try:
            result = (
                await exit_method(exc_type, exc, tb) if callable(exit_method) else None
            )
        except BaseException as close_error:
            self._state.emit(close_error)
            raise
        self._state.emit(exc)
        return result

    async def aclose(self) -> None:
        close = getattr(self._source, "aclose", None)
        if not callable(close):
            close = getattr(self._source, "close", None)
        try:
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.emit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


def _is_sync_stream(value: Any) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray))
        and hasattr(value, "__iter__")
        and not hasattr(value, "candidates")
    )


def _wrap_sync_method(
    original: Callable[..., Any], span_name: str
) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        request_payload = request_payload_from_call(
            instance=self, args=args, kwargs=kwargs
        )
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            _emit(
                request_payload=request_payload,
                start_ns=start_ns,
                span_name=span_name,
                trace_id=trace_id,
                parent_id=parent_id,
                error=exc,
            )
            raise
        if _is_sync_stream(response):
            return _SyncStreamProxy(
                response,
                _StreamState(
                    request_payload=request_payload,
                    start_ns=start_ns,
                    span_name=span_name,
                    trace_id=trace_id,
                    parent_id=parent_id,
                ),
            )
        _emit(
            request_payload=request_payload,
            start_ns=start_ns,
            span_name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _wrap_async_method(
    original: Callable[..., Any], span_name: str
) -> Callable[..., Any]:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        request_payload = request_payload_from_call(
            instance=self, args=args, kwargs=kwargs
        )
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            _emit(
                request_payload=request_payload,
                start_ns=start_ns,
                span_name=span_name,
                trace_id=trace_id,
                parent_id=parent_id,
                error=exc,
            )
            raise
        if hasattr(response, "__aiter__"):
            return _AsyncStreamProxy(
                response,
                _StreamState(
                    request_payload=request_payload,
                    start_ns=start_ns,
                    span_name=span_name,
                    trace_id=trace_id,
                    parent_id=parent_id,
                ),
            )
        _emit(
            request_payload=request_payload,
            start_ns=start_ns,
            span_name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            response_or_chunks=response,
        )
        return response

    return wrapper


class VertexAIInstrumentor:
    """Respan instrumentor for the Google Vertex AI Python SDK."""

    name = VERTEXAI_INSTRUMENTATION_NAME

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
                generative_model, chat_session = _load_vertexai_classes()
                targets = (
                    (
                        generative_model,
                        GENERATE_CONTENT_METHOD_NAME,
                        VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
                        False,
                    ),
                    (
                        generative_model,
                        GENERATE_CONTENT_ASYNC_METHOD_NAME,
                        VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
                        True,
                    ),
                    (
                        chat_session,
                        SEND_MESSAGE_METHOD_NAME,
                        VERTEXAI_CHAT_SPAN_NAME,
                        False,
                    ),
                    (
                        chat_session,
                        SEND_MESSAGE_ASYNC_METHOD_NAME,
                        VERTEXAI_CHAT_SPAN_NAME,
                        True,
                    ),
                )
                for cls, method_name, span_name, is_async in targets:
                    original = getattr(cls, method_name)
                    wrapper = (
                        _wrap_async_method(original, span_name)
                        if is_async
                        else _wrap_sync_method(original, span_name)
                    )
                    setattr(cls, method_name, wrapper)
                    installed.append(_Patch(cls, method_name, original, wrapper))
            except (ImportError, AttributeError) as exc:
                for patch in reversed(installed):
                    if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
                        setattr(patch.cls, patch.method_name, patch.original)
                logger.warning("Vertex AI instrumentation inactive: %s", exc)
                return
            except BaseException:
                for patch in reversed(installed):
                    if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
                        setattr(patch.cls, patch.method_name, patch.original)
                raise
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
            for patch in reversed(_PATCHES):
                if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
                    setattr(patch.cls, patch.method_name, patch.original)
            _PATCHES.clear()


def _reset_runtime_for_tests() -> None:
    global _ACTIVATION_COUNT
    with _LOCK:
        for patch in reversed(_PATCHES):
            if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
                setattr(patch.cls, patch.method_name, patch.original)
        _PATCHES.clear()
        _ACTIVATION_COUNT = 0
