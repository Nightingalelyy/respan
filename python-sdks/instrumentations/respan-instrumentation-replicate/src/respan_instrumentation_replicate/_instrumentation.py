"""Replicate SDK instrumentation plugin for Respan."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import importlib.metadata
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from opentelemetry import trace
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_replicate._constants import (
    ASYNC_PREFIX,
    INPUT_KEY,
    MAX_STREAM_CHUNKS,
    PREDICTION_RESPAN_MODEL_ATTR,
    REPLICATE_INSTRUMENTATION_NAME,
    REPLICATE_PREDICTION_CREATE_SPAN_NAME,
    REPLICATE_PREDICTION_WAIT_SPAN_NAME,
    REPLICATE_RUN_SPAN_NAME,
    REPLICATE_STREAM_SPAN_NAME,
    RESPAN_PARAMS_KEY,
    RESPAN_PARAMS_MODEL_KEY,
)
from respan_instrumentation_replicate._serialization import (
    exception_message,
    exception_status,
    prediction_summary,
    safe_text,
)
from respan_instrumentation_replicate._translator import (
    build_model_call_span_data,
    build_operation_span_data,
)

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_REFCOUNT = 0
_ENABLED = False


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


_PATCHES: list[_Patch] = []

_SUPPRESSED_SPAN_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "respan_replicate_suppressed_span_depth",
    default=0,
)


def _spans_suppressed() -> bool:
    return _SUPPRESSED_SPAN_DEPTH.get() > 0


@contextlib.contextmanager
def _suppress_nested_spans():
    token = _SUPPRESSED_SPAN_DEPTH.set(_SUPPRESSED_SPAN_DEPTH.get() + 1)
    try:
        yield
    finally:
        _SUPPRESSED_SPAN_DEPTH.reset(token)


def _current_otel_parent() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    try:
        span_context = current_span.get_span_context()
    except Exception:  # noqa: BLE001 - non-recording spans can be hostile proxies.
        return None, None

    trace_id = getattr(span_context, "trace_id", 0)
    span_id = getattr(span_context, "span_id", 0)
    if not isinstance(trace_id, int) or not isinstance(span_id, int):
        return None, None
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _emit_span(
    *,
    span_name: str,
    attributes: dict[str, Any],
    start_time_ns: int,
    end_time_ns: int | None = None,
    error: BaseException | None = None,
    parent_context: tuple[str | None, str | None] | None = None,
) -> None:
    if _spans_suppressed():
        return

    trace_id, parent_id = parent_context or _current_otel_parent()
    attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] = (
        attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH, span_name)
        if parent_id
        else ""
    )
    status_code = exception_status(error) if error is not None else 200
    if error is not None:
        message = exception_message(error)
        attributes["status_code"] = status_code
        attributes[ERROR_MESSAGE_ATTR] = message
    else:
        message = None
        attributes.setdefault("status_code", 200)
    span = build_readable_span(
        name=span_name,
        trace_id=trace_id,
        parent_id=parent_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns or time.time_ns(),
        attributes=attributes,
        status_code=status_code,
        error_message=message,
    )
    try:
        package_version = importlib.metadata.version("respan-instrumentation-replicate")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    span._instrumentation_scope = InstrumentationScope(  # type: ignore[attr-defined]
        REPLICATE_INSTRUMENTATION_NAME,
        package_version,
    )
    inject_span(span=span)


def _pop_respan_params(kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    call_kwargs = dict(kwargs)
    respan_params = call_kwargs.pop(RESPAN_PARAMS_KEY, None)
    return call_kwargs, respan_params


def _reported_model_from_respan_params(respan_params: Any) -> str | None:
    if not isinstance(respan_params, dict):
        return None
    model = respan_params.get(RESPAN_PARAMS_MODEL_KEY)
    return safe_text(model) if model else None


def _set_prediction_reported_model(prediction: Any, respan_params: Any) -> None:
    reported_model = _reported_model_from_respan_params(respan_params)
    if not reported_model:
        return
    try:
        object.__setattr__(prediction, PREDICTION_RESPAN_MODEL_ATTR, reported_model)
    except Exception:  # noqa: BLE001 - resources may reject private attributes.
        try:
            setattr(prediction, PREDICTION_RESPAN_MODEL_ATTR, reported_model)
        except Exception:  # noqa: BLE001 - best-effort metadata only.
            return


def _is_file_output(value: Any) -> bool:
    return value.__class__.__name__ == "FileOutput"


def _is_sync_iterator(value: Any) -> bool:
    return not _is_file_output(value) and isinstance(value, Iterator)


def _is_async_iterator(value: Any) -> bool:
    return isinstance(value, AsyncIterator)


class _SyncIteratorProxy:
    def __init__(
        self,
        *,
        iterator: Iterator[Any],
        emit_once: Callable[[list[Any], BaseException | None], None],
    ) -> None:
        self._iterator = iterator
        self._emit_once = emit_once
        self._chunks: list[Any] = []
        self._emitted = False

    def __iter__(self) -> _SyncIteratorProxy:
        return self

    def __next__(self) -> Any:
        try:
            with _suppress_nested_spans():
                chunk = next(self._iterator)
        except StopIteration:
            self._emit(error=None)
            raise
        except BaseException as exc:
            self._emit(error=exc)
            raise
        if len(self._chunks) < MAX_STREAM_CHUNKS:
            self._chunks.append(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)

    def close(self) -> None:
        error: BaseException | None = None
        try:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._emit(error=error)

    def __enter__(self) -> Self:
        enter = getattr(self._iterator, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        try:
            exit_method = getattr(self._iterator, "__exit__", None)
            if callable(exit_method):
                return exit_method(exc_type, exc, tb)
            self.close()
            return None
        finally:
            self._emit(error=exc)

    def _emit(self, *, error: BaseException | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        self._emit_once(self._chunks, error)


class _AsyncIteratorProxy:
    def __init__(
        self,
        *,
        iterator: AsyncIterator[Any],
        emit_once: Callable[[list[Any], BaseException | None], None],
    ) -> None:
        self._iterator = iterator
        self._chunks: list[Any] = []
        self._emit_once = emit_once
        self._emitted = False

    def __aiter__(self) -> _AsyncIteratorProxy:
        return self

    async def __anext__(self) -> Any:
        try:
            with _suppress_nested_spans():
                chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._emit(error=None)
            raise
        except BaseException as exc:
            self._emit(error=exc)
            raise
        if len(self._chunks) < MAX_STREAM_CHUNKS:
            self._chunks.append(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)

    async def aclose(self) -> None:
        error: BaseException | None = None
        try:
            close = getattr(self._iterator, "aclose", None)
            if callable(close):
                await close()
            else:
                close = getattr(self._iterator, "close", None)
                if callable(close):
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._emit(error=error)

    async def __aenter__(self) -> Self:
        enter = getattr(self._iterator, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        try:
            exit_method = getattr(self._iterator, "__aexit__", None)
            if callable(exit_method):
                return await exit_method(exc_type, exc, tb)
            await self.aclose()
            return None
        finally:
            self._emit(error=exc)

    def _emit(self, *, error: BaseException | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        self._emit_once(self._chunks, error)


def _wrap_sync_run(original: Any, *, span_name: str, stream: bool = False) -> Any:
    @functools.wraps(original)
    def wrapper(
        self: Any, ref: Any, input: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        if not _ENABLED:
            if stream:
                return original(self, ref, *args, input=input, **kwargs)
            return original(self, ref, input, *args, **kwargs)
        call_kwargs, respan_params = _pop_respan_params(kwargs)
        event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                if stream:
                    output = original(self, ref, *args, input=input, **call_kwargs)
                else:
                    output = original(self, ref, input, *args, **call_kwargs)
        except BaseException as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=span_name,
                ref=ref,
                input_value=input,
                kwargs=event_kwargs,
                error=exc,
                stream=stream,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        if _is_sync_iterator(output):

            def emit_once(chunks: list[Any], error: BaseException | None) -> None:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=span_name,
                    ref=ref,
                    input_value=input,
                    kwargs=event_kwargs,
                    output=chunks,
                    error=error,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=error,
                    parent_context=parent_context,
                )

            return _SyncIteratorProxy(iterator=output, emit_once=emit_once)

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=span_name,
            ref=ref,
            input_value=input,
            kwargs=event_kwargs,
            output=output,
            stream=stream,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return output

    return wrapper


def _wrap_async_run(original: Any, *, span_name: str, stream: bool = False) -> Any:
    @functools.wraps(original)
    async def wrapper(
        self: Any, ref: Any, input: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        if not _ENABLED:
            if stream:
                return await original(self, ref, input=input, **kwargs)
            return await original(self, ref, input, *args, **kwargs)
        call_kwargs, respan_params = _pop_respan_params(kwargs)
        event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                if stream:
                    output = await original(self, ref, input=input, **call_kwargs)
                else:
                    output = await original(self, ref, input, *args, **call_kwargs)
        except BaseException as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=span_name,
                ref=ref,
                input_value=input,
                kwargs=event_kwargs,
                error=exc,
                stream=stream,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        if _is_async_iterator(output):

            def emit_once(chunks: list[Any], error: BaseException | None) -> None:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=span_name,
                    ref=ref,
                    input_value=input,
                    kwargs=event_kwargs,
                    output=chunks,
                    error=error,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=error,
                    parent_context=parent_context,
                )

            return _AsyncIteratorProxy(iterator=output, emit_once=emit_once)

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=span_name,
            ref=ref,
            input_value=input,
            kwargs=event_kwargs,
            output=output,
            stream=stream,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return output

    return wrapper


def _wrap_prediction_create(original: Any, *, is_async: bool = False) -> Any:
    if is_async:

        @functools.wraps(original)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return await original(self, *args, **kwargs)
            call_kwargs, respan_params = _pop_respan_params(kwargs)
            event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            try:
                with _suppress_nested_spans():
                    prediction = await original(self, *args, **call_kwargs)
            except BaseException as exc:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
                    ref=args[0] if args else None,
                    input_value=call_kwargs.get(INPUT_KEY),
                    kwargs=event_kwargs,
                    error=exc,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
                ref=args[0] if args else None,
                input_value=call_kwargs.get(INPUT_KEY),
                kwargs=event_kwargs,
                prediction=prediction,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                parent_context=parent_context,
            )
            _set_prediction_reported_model(prediction, respan_params)
            return prediction

        return async_wrapper

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ENABLED:
            return original(self, *args, **kwargs)
        call_kwargs, respan_params = _pop_respan_params(kwargs)
        event_kwargs = {**call_kwargs, RESPAN_PARAMS_KEY: respan_params}
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                prediction = original(self, *args, **call_kwargs)
        except BaseException as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
                ref=args[0] if args else None,
                input_value=call_kwargs.get(INPUT_KEY),
                kwargs=event_kwargs,
                error=exc,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        resolved_span_name, attrs = build_model_call_span_data(
            span_name=REPLICATE_PREDICTION_CREATE_SPAN_NAME,
            ref=args[0] if args else None,
            input_value=call_kwargs.get(INPUT_KEY),
            kwargs=event_kwargs,
            prediction=prediction,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        _set_prediction_reported_model(prediction, respan_params)
        return prediction

    return wrapper


def _wrap_prediction_wait(original: Any, *, is_async: bool = False) -> Any:
    if is_async:

        @functools.wraps(original)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return await original(self, *args, **kwargs)
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            try:
                with _suppress_nested_spans():
                    result = await original(self, *args, **kwargs)
            except BaseException as exc:
                resolved_span_name, attrs = build_operation_span_data(
                    span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
                    input_value=prediction_summary(self),
                    error=exc,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            resolved_span_name, attrs = build_operation_span_data(
                span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
                input_value=prediction_summary(self),
                output=prediction_summary(self),
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                parent_context=parent_context,
            )
            return result

        return async_wrapper

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ENABLED:
            return original(self, *args, **kwargs)
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                result = original(self, *args, **kwargs)
        except BaseException as exc:
            resolved_span_name, attrs = build_operation_span_data(
                span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
                input_value=prediction_summary(self),
                error=exc,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        resolved_span_name, attrs = build_operation_span_data(
            span_name=REPLICATE_PREDICTION_WAIT_SPAN_NAME,
            input_value=prediction_summary(self),
            output=prediction_summary(self),
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return result

    return wrapper


def _wrap_prediction_stream(original: Any, *, is_async: bool = False) -> Any:
    if is_async:

        @functools.wraps(original)
        def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return original(self, *args, **kwargs)
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            try:
                with _suppress_nested_spans():
                    iterator = original(self, *args, **kwargs)
            except BaseException as exc:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_STREAM_SPAN_NAME,
                    prediction=self,
                    error=exc,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            def emit_once(chunks: list[Any], error: BaseException | None) -> None:
                resolved_span_name, attrs = build_model_call_span_data(
                    span_name=REPLICATE_STREAM_SPAN_NAME,
                    prediction=self,
                    output=chunks,
                    error=error,
                    stream=True,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=error,
                    parent_context=parent_context,
                )

            return _AsyncIteratorProxy(iterator=iterator, emit_once=emit_once)

        return async_wrapper

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ENABLED:
            return original(self, *args, **kwargs)
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        try:
            with _suppress_nested_spans():
                iterator = original(self, *args, **kwargs)
        except BaseException as exc:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_STREAM_SPAN_NAME,
                prediction=self,
                error=exc,
                stream=True,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        def emit_once(chunks: list[Any], error: BaseException | None) -> None:
            resolved_span_name, attrs = build_model_call_span_data(
                span_name=REPLICATE_STREAM_SPAN_NAME,
                prediction=self,
                output=chunks,
                error=error,
                stream=True,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=error,
                parent_context=parent_context,
            )

        return _SyncIteratorProxy(iterator=iterator, emit_once=emit_once)

    return wrapper


def _wrap_operation(original: Any, *, span_name: str, is_async: bool = False) -> Any:
    if is_async:

        @functools.wraps(original)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return await original(self, *args, **kwargs)
            parent_context = _current_otel_parent()
            start_ns = time.time_ns()
            input_value = {"args": args, "kwargs": kwargs}
            try:
                with _suppress_nested_spans():
                    result = await original(self, *args, **kwargs)
            except BaseException as exc:
                resolved_span_name, attrs = build_operation_span_data(
                    span_name=span_name,
                    input_value=input_value,
                    error=exc,
                )
                _emit_span(
                    span_name=resolved_span_name,
                    attributes=attrs,
                    start_time_ns=start_ns,
                    error=exc,
                    parent_context=parent_context,
                )
                raise

            resolved_span_name, attrs = build_operation_span_data(
                span_name=span_name,
                input_value=input_value,
                output=result,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                parent_context=parent_context,
            )
            return result

        return async_wrapper

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _ENABLED:
            return original(self, *args, **kwargs)
        parent_context = _current_otel_parent()
        start_ns = time.time_ns()
        input_value = {"args": args, "kwargs": kwargs}
        try:
            with _suppress_nested_spans():
                result = original(self, *args, **kwargs)
        except BaseException as exc:
            resolved_span_name, attrs = build_operation_span_data(
                span_name=span_name,
                input_value=input_value,
                error=exc,
            )
            _emit_span(
                span_name=resolved_span_name,
                attributes=attrs,
                start_time_ns=start_ns,
                error=exc,
                parent_context=parent_context,
            )
            raise

        resolved_span_name, attrs = build_operation_span_data(
            span_name=span_name,
            input_value=input_value,
            output=result,
        )
        _emit_span(
            span_name=resolved_span_name,
            attributes=attrs,
            start_time_ns=start_ns,
            parent_context=parent_context,
        )
        return result

    return wrapper


def _module_run_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    def wrapper(ref: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        client = module.default_client
        method = getattr(client, method_name)
        return method(ref, input, *args, **kwargs)

    return wrapper


def _module_async_run_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    async def wrapper(ref: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        client = module.default_client
        method = getattr(client, method_name)
        return await method(ref, input, *args, **kwargs)

    return wrapper


def _module_stream_wrapper(module: Any, *, method_name: str) -> Callable[..., Any]:
    def wrapper(ref: Any, *, input: Any = None, **kwargs: Any) -> Any:
        client = module.default_client
        method = getattr(client, method_name)
        return method(ref, input=input, **kwargs)

    return wrapper


def _module_async_stream_wrapper(
    module: Any, *, method_name: str
) -> Callable[..., Any]:
    async def wrapper(ref: Any, input: Any = None, **kwargs: Any) -> Any:
        client = module.default_client
        method = getattr(client, method_name)
        return await method(ref, input=input, **kwargs)

    return wrapper


class ReplicateInstrumentor:
    """Respan instrumentor for the Replicate Python SDK."""

    name = REPLICATE_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    @staticmethod
    def _patch_attr(owner: Any, attr_name: str, replacement: Any) -> None:
        original = getattr(owner, attr_name)
        setattr(owner, attr_name, replacement)
        _PATCHES.append(_Patch(owner, attr_name, original, replacement))

    def _activate_once(self) -> None:
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Replicate instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            replicate_module = importlib.import_module("replicate")
            client_module = importlib.import_module("replicate.client")
            prediction_module = importlib.import_module("replicate.prediction")
        except ImportError as exc:
            logger.warning(
                "Failed to activate Replicate instrumentation - missing dependency: %s",
                exc,
            )
            return

        Client = client_module.Client
        Predictions = prediction_module.Predictions
        Prediction = prediction_module.Prediction

        self._patch_attr(
            Client,
            "run",
            _wrap_sync_run(
                Client.run,
                span_name=REPLICATE_RUN_SPAN_NAME,
            ),
        )
        self._patch_attr(
            Client,
            "async_run",
            _wrap_async_run(
                Client.async_run,
                span_name=f"{ASYNC_PREFIX}{REPLICATE_RUN_SPAN_NAME}",
            ),
        )
        self._patch_attr(
            Client,
            "stream",
            _wrap_sync_run(
                Client.stream,
                span_name=REPLICATE_STREAM_SPAN_NAME,
                stream=True,
            ),
        )
        self._patch_attr(
            Client,
            "async_stream",
            _wrap_async_run(
                Client.async_stream,
                span_name=f"{ASYNC_PREFIX}{REPLICATE_STREAM_SPAN_NAME}",
                stream=True,
            ),
        )

        self._patch_attr(
            Predictions,
            "create",
            _wrap_prediction_create(Predictions.create),
        )
        self._patch_attr(
            Predictions,
            "async_create",
            _wrap_prediction_create(Predictions.async_create, is_async=True),
        )
        for method_name in ("list", "get", "cancel"):
            self._patch_attr(
                Predictions,
                method_name,
                _wrap_operation(
                    getattr(Predictions, method_name),
                    span_name=f"replicate.predictions.{method_name}",
                ),
            )
        for method_name in ("async_list", "async_get", "async_cancel"):
            self._patch_attr(
                Predictions,
                method_name,
                _wrap_operation(
                    getattr(Predictions, method_name),
                    span_name=f"replicate.predictions.{method_name}",
                    is_async=True,
                ),
            )

        self._patch_attr(
            Prediction,
            "wait",
            _wrap_prediction_wait(Prediction.wait),
        )
        self._patch_attr(
            Prediction,
            "async_wait",
            _wrap_prediction_wait(Prediction.async_wait, is_async=True),
        )
        self._patch_attr(
            Prediction,
            "stream",
            _wrap_prediction_stream(Prediction.stream),
        )
        self._patch_attr(
            Prediction,
            "async_stream",
            _wrap_prediction_stream(Prediction.async_stream, is_async=True),
        )

        self._patch_attr(
            replicate_module,
            "run",
            _module_run_wrapper(replicate_module, method_name="run"),
        )
        self._patch_attr(
            replicate_module,
            "async_run",
            _module_async_run_wrapper(replicate_module, method_name="async_run"),
        )
        self._patch_attr(
            replicate_module,
            "stream",
            _module_stream_wrapper(replicate_module, method_name="stream"),
        )
        self._patch_attr(
            replicate_module,
            "async_stream",
            _module_async_stream_wrapper(replicate_module, method_name="async_stream"),
        )

        self._is_instrumented = True

    def activate(self) -> None:
        """Monkey-patch the Replicate SDK with shared transactional ownership."""
        global _ENABLED, _REFCOUNT

        if self._is_instrumented:
            return
        with _LOCK:
            if self._is_instrumented:
                return
            if _REFCOUNT:
                _REFCOUNT += 1
                self._is_instrumented = True
                return
            try:
                self._activate_once()
            except Exception:
                for patch in reversed(_PATCHES):
                    if getattr(patch.owner, patch.name, None) is patch.replacement:
                        setattr(patch.owner, patch.name, patch.original)
                _PATCHES.clear()
                raise
            if not self._is_instrumented:
                return
            _ENABLED = True
            _REFCOUNT = 1
        logger.info("Replicate instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched Replicate SDK methods."""
        global _ENABLED, _REFCOUNT

        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            _ENABLED = False
            for patch in reversed(_PATCHES):
                try:
                    if getattr(patch.owner, patch.name, None) is patch.replacement:
                        setattr(patch.owner, patch.name, patch.original)
                except Exception:  # noqa: BLE001 - foreign-safe best-effort restore.
                    logger.debug("Failed to restore Replicate SDK attr %s", patch.name)
            _PATCHES.clear()
        logger.info("Replicate instrumentation deactivated")
