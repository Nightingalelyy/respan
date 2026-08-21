"""Together AI SDK instrumentation plugin for Respan."""

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

from respan_instrumentation_together._constants import (
    ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME,
    ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME,
    ASYNC_IMAGES_RESOURCE_CLASS_NAME,
    ASYNC_RERANK_RESOURCE_CLASS_NAME,
    COMPLETIONS_RESOURCE_CLASS_NAME,
    CREATE_METHOD_NAME,
    EMBEDDINGS_RESOURCE_CLASS_NAME,
    GENERATE_METHOD_NAME,
    IMAGES_RESOURCE_CLASS_NAME,
    RERANK_RESOURCE_CLASS_NAME,
    STREAM_KEY,
    TOGETHER_CHAT_COMPLETIONS_MODULE,
    TOGETHER_EMBEDDINGS_MODULE,
    TOGETHER_IMAGES_MODULE,
    TOGETHER_INSTRUMENTATION_NAME,
    TOGETHER_RERANK_MODULE,
    TOGETHER_TEXT_COMPLETIONS_MODULE,
)
from respan_instrumentation_together._otel_emitter import emit_together_span
from respan_instrumentation_together._serialization import (
    provider_status_code,
    safe_exception_message,
)

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


def _load_class(module_path: str, class_name: str) -> type[Any]:
    return _get_module_attr(module_path, class_name)


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None, None
        return format_trace_id(context.trace_id), format_span_id(context.span_id)
    except BaseException:  # noqa: BLE001
        return None, None


def _emit_span_safely(
    *,
    operation: str,
    kwargs: dict[str, Any],
    start_ns: int,
    trace_id: str | None,
    parent_id: str | None,
    response_or_chunks: Any = None,
    error: BaseException | None = None,
) -> None:
    error_message = safe_exception_message(error) if error is not None else None
    emit_together_span(
        operation=operation,
        request_kwargs=dict(kwargs),
        start_ns=start_ns,
        response_or_chunks=response_or_chunks,
        error_message=error_message,
        status_code=provider_status_code(error) if error is not None else 200,
        trace_id=trace_id,
        parent_id=parent_id,
    )


class _StreamState:
    def __init__(
        self,
        *,
        operation: str,
        kwargs: dict[str, Any],
        start_ns: int,
        trace_id: str | None,
        parent_id: str | None,
    ) -> None:
        self.operation = operation
        self.kwargs = dict(kwargs)
        self.start_ns = start_ns
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
        _emit_span_safely(
            operation=self.operation,
            kwargs=self.kwargs,
            start_ns=self.start_ns,
            trace_id=self.trace_id,
            parent_id=self.parent_id,
            response_or_chunks=self.chunks,
            error=error,
        )


class _InstrumentedStream:
    def __init__(self, *, stream: Any, state: _StreamState) -> None:
        self._stream = stream
        self._state = state

    def __iter__(self) -> _InstrumentedStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._state.emit()
            raise
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.capture(chunk)
        return chunk

    def __enter__(self) -> Self:
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        exit_method = getattr(self._stream, "__exit__", None)
        try:
            result = exit_method(exc_type, exc, tb) if callable(exit_method) else None
        except BaseException as close_error:
            self._state.emit(close_error)
            raise
        self._state.emit(exc)
        return result

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        try:
            if callable(close):
                close()
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.emit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _InstrumentedAsyncStream:
    def __init__(self, *, stream: Any, state: _StreamState) -> None:
        self._stream = stream
        self._state = state

    def __aiter__(self) -> _InstrumentedAsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._state.emit()
            raise
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.capture(chunk)
        return chunk

    async def __aenter__(self) -> Self:
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        exit_method = getattr(self._stream, "__aexit__", None)
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
        close = getattr(self._stream, "aclose", None)
        if not callable(close):
            close = getattr(self._stream, "close", None)
        try:
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except BaseException as exc:
            self._state.emit(exc)
            raise
        self._state.emit()

    async def close(self) -> None:
        await self.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_sync_create(
    original: Callable[..., Any], *, operation: str
) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            _emit_span_safely(
                operation=operation,
                kwargs=kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                error=exc,
            )
            raise
        if kwargs.get(STREAM_KEY) is True:
            return _InstrumentedStream(
                stream=response,
                state=_StreamState(
                    operation=operation,
                    kwargs=kwargs,
                    start_ns=start_ns,
                    trace_id=trace_id,
                    parent_id=parent_id,
                ),
            )
        _emit_span_safely(
            operation=operation,
            kwargs=kwargs,
            start_ns=start_ns,
            trace_id=trace_id,
            parent_id=parent_id,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _wrap_async_create(
    original: Callable[..., Any], *, operation: str
) -> Callable[..., Any]:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = _current_trace_parent_ids()
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            _emit_span_safely(
                operation=operation,
                kwargs=kwargs,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
                error=exc,
            )
            raise
        if kwargs.get(STREAM_KEY) is True:
            return _InstrumentedAsyncStream(
                stream=response,
                state=_StreamState(
                    operation=operation,
                    kwargs=kwargs,
                    start_ns=start_ns,
                    trace_id=trace_id,
                    parent_id=parent_id,
                ),
            )
        _emit_span_safely(
            operation=operation,
            kwargs=kwargs,
            start_ns=start_ns,
            trace_id=trace_id,
            parent_id=parent_id,
            response_or_chunks=response,
        )
        return response

    return wrapper


def _targets() -> list[tuple[type[Any], str, str, bool]]:
    return [
        (
            _load_class(
                TOGETHER_CHAT_COMPLETIONS_MODULE, COMPLETIONS_RESOURCE_CLASS_NAME
            ),
            CREATE_METHOD_NAME,
            "chat",
            False,
        ),
        (
            _load_class(
                TOGETHER_CHAT_COMPLETIONS_MODULE, ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME
            ),
            CREATE_METHOD_NAME,
            "chat",
            True,
        ),
        (
            _load_class(
                TOGETHER_TEXT_COMPLETIONS_MODULE, COMPLETIONS_RESOURCE_CLASS_NAME
            ),
            CREATE_METHOD_NAME,
            "completion",
            False,
        ),
        (
            _load_class(
                TOGETHER_TEXT_COMPLETIONS_MODULE, ASYNC_COMPLETIONS_RESOURCE_CLASS_NAME
            ),
            CREATE_METHOD_NAME,
            "completion",
            True,
        ),
        (
            _load_class(TOGETHER_EMBEDDINGS_MODULE, EMBEDDINGS_RESOURCE_CLASS_NAME),
            CREATE_METHOD_NAME,
            "embedding",
            False,
        ),
        (
            _load_class(
                TOGETHER_EMBEDDINGS_MODULE, ASYNC_EMBEDDINGS_RESOURCE_CLASS_NAME
            ),
            CREATE_METHOD_NAME,
            "embedding",
            True,
        ),
        (
            _load_class(TOGETHER_IMAGES_MODULE, IMAGES_RESOURCE_CLASS_NAME),
            GENERATE_METHOD_NAME,
            "image",
            False,
        ),
        (
            _load_class(TOGETHER_IMAGES_MODULE, ASYNC_IMAGES_RESOURCE_CLASS_NAME),
            GENERATE_METHOD_NAME,
            "image",
            True,
        ),
        (
            _load_class(TOGETHER_RERANK_MODULE, RERANK_RESOURCE_CLASS_NAME),
            CREATE_METHOD_NAME,
            "rerank",
            False,
        ),
        (
            _load_class(TOGETHER_RERANK_MODULE, ASYNC_RERANK_RESOURCE_CLASS_NAME),
            CREATE_METHOD_NAME,
            "rerank",
            True,
        ),
    ]


class TogetherInstrumentor:
    """Respan instrumentor for the Together AI Python SDK."""

    name = TOGETHER_INSTRUMENTATION_NAME

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
                for cls, method_name, operation, is_async in _targets():
                    original = getattr(cls, method_name)
                    wrapper = (
                        _wrap_async_create(original, operation=operation)
                        if is_async
                        else _wrap_sync_create(original, operation=operation)
                    )
                    setattr(cls, method_name, wrapper)
                    installed.append(_Patch(cls, method_name, original, wrapper))
            except (ImportError, AttributeError) as exc:
                for patch in reversed(installed):
                    if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
                        setattr(patch.cls, patch.method_name, patch.original)
                logger.warning("Together instrumentation inactive: %s", exc)
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
    """Restore owned methods and reset shared state for isolated tests."""
    global _ACTIVATION_COUNT
    with _LOCK:
        for patch in reversed(_PATCHES):
            if getattr(patch.cls, patch.method_name, None) is patch.wrapper:
                setattr(patch.cls, patch.method_name, patch.original)
        _PATCHES.clear()
        _ACTIVATION_COUNT = 0
