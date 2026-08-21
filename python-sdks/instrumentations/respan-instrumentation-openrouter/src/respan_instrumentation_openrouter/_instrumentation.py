"""OpenRouter instrumentation plugin for Respan."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import wraps
from types import TracebackType
from typing import Any, Self

from opentelemetry import context as otel_context
from opentelemetry import trace
from respan_instrumentation_openai import OpenAIInstrumentor
from respan_instrumentation_openai import _instrumentation as openai_instrumentation
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_openrouter._constants import (
    MAX_ATTRIBUTE_BYTES,
    MAX_COLLECTION_ITEMS,
    MAX_ERROR_BYTES,
    OPENROUTER_INSTRUMENTATION_NAME,
)
from respan_instrumentation_openrouter._processor import (
    OpenRouterSpanProcessor,
    _http_status_code,
    _redact_text,
    openrouter_emission_context,
)

logger = logging.getLogger(__name__)

_BRIDGE_LOCK = threading.RLock()
_ACTIVE_BRIDGE_OWNER: OpenRouterInstrumentor | None = None
_ACTIVE_RUNTIME: _OpenRouterRuntime | None = None
_TRUNCATION_SUFFIX = "...[truncated]"


def _safe_error_message(value: Any) -> str | None:
    """Return useful error text without invoking arbitrary string hooks."""

    if value is None:
        return None
    if isinstance(value, str):
        return _redact_text(value, limit=MAX_ERROR_BYTES)
    if isinstance(value, BaseException):
        for argument in value.args:
            if isinstance(argument, str) and argument:
                return _redact_text(argument, limit=MAX_ERROR_BYTES)
        return type(value).__name__
    return type(value).__name__


def _exception_status_code(exc: BaseException) -> int:
    try:
        status_code = getattr(exc, "status_code", None)
    except Exception:  # noqa: BLE001 - provider exceptions may expose hostile properties
        status_code = None
    if status_code is None:
        try:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        except Exception:  # noqa: BLE001 - response objects are provider-owned
            status_code = None
    if type(status_code) is int and 400 <= status_code <= 599:
        return status_code
    if (
        type(status_code) is str
        and len(status_code) == 3
        and status_code.isascii()
        and status_code.isdigit()
    ):
        resolved = int(status_code)
        if 400 <= resolved <= 599:
            return resolved
    return _http_status_code(_safe_error_message(exc)) or 500


def _append_bounded_text(current: str, fragment: Any) -> str:
    if not isinstance(fragment, str) or not fragment:
        return current
    current_bytes = len(current.encode("utf-8"))
    remaining = MAX_ATTRIBUTE_BYTES - current_bytes
    if remaining <= 0 or current.endswith(_TRUNCATION_SUFFIX):
        return current
    candidate = fragment[:remaining]
    encoded = candidate.encode("utf-8")
    if len(fragment) <= remaining and len(encoded) <= remaining:
        return current + fragment
    suffix = _TRUNCATION_SUFFIX.encode("utf-8")
    if remaining <= len(suffix):
        return current + suffix[:remaining].decode("utf-8", errors="ignore")
    prefix = encoded[: remaining - len(suffix)].decode("utf-8", errors="ignore")
    return current + prefix + _TRUNCATION_SUFFIX


class _StreamAccumulator:
    """Incrementally retain complete bounded stream semantics."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._model: str | None = None
        self._response_id: str | None = None
        self._usage: Any = None
        self._content = ""
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._response: Any = None

    def append(self, item: Any) -> None:
        if self._kind == "response":
            response = getattr(item, "response", None)
            if response is not None:
                self._response = response
            return

        model = getattr(item, "model", None)
        if isinstance(model, str) and model:
            self._model = model
        response_id = getattr(item, "id", None)
        if isinstance(response_id, str) and response_id:
            self._response_id = response_id
        usage = getattr(item, "usage", None)
        if usage is not None:
            self._usage = usage

        choices = getattr(item, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices:
            return
        choice = choices[0]
        if self._kind == "completion":
            self._content = _append_bounded_text(
                self._content,
                getattr(choice, "text", None),
            )
            return
        if self._kind != "chat":
            return

        delta = getattr(choice, "delta", None)
        if delta is None:
            return
        self._content = _append_bounded_text(
            self._content,
            getattr(delta, "content", None),
        )
        tool_call_deltas = getattr(delta, "tool_calls", None)
        if not isinstance(tool_call_deltas, (list, tuple)):
            return
        for tool_call_delta in tool_call_deltas[:MAX_COLLECTION_ITEMS]:
            index = getattr(tool_call_delta, "index", 0)
            if not isinstance(index, int) or index < 0:
                index = 0
            if index not in self._tool_calls:
                if len(self._tool_calls) >= MAX_COLLECTION_ITEMS:
                    continue
                self._tool_calls[index] = {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            slot = self._tool_calls[index]
            tool_call_id = getattr(tool_call_delta, "id", None)
            if isinstance(tool_call_id, str) and tool_call_id:
                slot["id"] = tool_call_id
            function = getattr(tool_call_delta, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                slot["function"]["name"] = name
            slot["function"]["arguments"] = _append_bounded_text(
                slot["function"]["arguments"],
                getattr(function, "arguments", None),
            )

    def response(self) -> Any:
        if self._kind == "response":
            return self._response
        if self._kind == "completion":
            return {
                "model": self._model,
                "id": self._response_id,
                "usage": self._usage,
                "choices": [{"text": self._content}],
            }
        if self._kind == "chat":
            message: dict[str, Any] = {
                "role": "assistant",
                "content": self._content or None,
            }
            if self._tool_calls:
                message["tool_calls"] = [
                    self._tool_calls[index] for index in sorted(self._tool_calls)
                ]
            return {
                "model": self._model,
                "id": self._response_id,
                "usage": self._usage,
                "choices": [{"message": message}],
            }
        return None


class _StreamProxyBase:
    def __init__(
        self,
        stream: Any,
        *,
        kind: str,
        request_kwargs: dict[str, Any],
        start_ns: int,
        emit_fn: Any,
    ) -> None:
        self._stream = stream
        self._request_kwargs = request_kwargs
        self._start_ns = start_ns
        self._emit_fn = emit_fn
        self._accumulator = _StreamAccumulator(kind)
        self._call_context = otel_context.get_current()
        self._emitted = False
        self._exhausted = False

    def _emit_once(self, error: BaseException | None = None) -> None:
        if self._emitted:
            return
        self._emitted = True
        token = otel_context.attach(self._call_context)
        try:
            _emit_safely(
                self._emit_fn,
                request_kwargs=self._request_kwargs,
                start_ns=self._start_ns,
                response=self._accumulator.response(),
                error_message=_safe_error_message(error),
                status_code=_exception_status_code(error) if error is not None else 200,
            )
        finally:
            otel_context.detach(token)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _SyncStreamProxy(_StreamProxyBase):
    def __init__(self, stream: Any, **kwargs: Any) -> None:
        super().__init__(stream, **kwargs)
        self._iterator = (
            stream if callable(getattr(stream, "__next__", None)) else iter(stream)
        )

    def __iter__(self) -> _SyncStreamProxy:
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            self._emit_once()
            raise
        except BaseException as exc:
            self._emit_once(exc)
            self._close_after_error()
            raise
        try:
            self._accumulator.append(item)
        except Exception:
            logger.exception("Failed to aggregate delegated OpenRouter stream item")
        return item

    def _close_after_error(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                logger.debug(
                    "Failed to close delegated OpenRouter sync stream",
                    exc_info=True,
                )

    def close(self) -> Any:
        close = getattr(self._stream, "close", None)
        try:
            result = close() if callable(close) else None
        except BaseException as exc:
            self._emit_once(exc)
            raise
        self._emit_once()
        return result

    def __enter__(self) -> Self:
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            entered = enter()
            if entered is not None and entered is not self._stream:
                self._iterator = (
                    entered
                    if callable(getattr(entered, "__next__", None))
                    else iter(entered)
                )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any:
        exit_method = getattr(self._stream, "__exit__", None)
        try:
            result = (
                exit_method(exc_type, exc, traceback)
                if callable(exit_method)
                else self.close()
            )
        except BaseException as close_error:
            self._emit_once(close_error)
            raise
        self._emit_once()
        return result


class _AsyncStreamProxy(_StreamProxyBase):
    def __init__(self, stream: Any, **kwargs: Any) -> None:
        super().__init__(stream, **kwargs)
        self._iteration_stream = stream
        self._aiterator: Any = None

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate_delegate()

    async def _iterate_delegate(self) -> AsyncIterator[Any]:
        try:
            async for item in self._iteration_stream:
                try:
                    self._accumulator.append(item)
                except Exception:
                    logger.exception(
                        "Failed to aggregate delegated OpenRouter stream item"
                    )
                yield item
        except GeneratorExit:
            self._emit_once()
            await self._close_after_error()
            raise
        except BaseException as exc:
            self._emit_once(exc)
            await self._close_after_error()
            raise
        self._exhausted = True
        self._emit_once()

    async def __anext__(self) -> Any:
        if self._aiterator is None:
            self._aiterator = self._iteration_stream.__aiter__()
        try:
            item = await self._aiterator.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            self._emit_once()
            raise
        except BaseException as exc:
            self._emit_once(exc)
            await self._close_after_error()
            raise
        try:
            self._accumulator.append(item)
        except Exception:
            logger.exception("Failed to aggregate delegated OpenRouter stream item")
        return item

    async def _call_close(self) -> Any:
        close = getattr(self._stream, "close", None)
        if not callable(close):
            close = getattr(self._stream, "aclose", None)
        if not callable(close):
            return None
        result = close()
        return await result if inspect.isawaitable(result) else result

    async def _close_after_error(self) -> None:
        try:
            await self._call_close()
        except BaseException:
            logger.debug(
                "Failed to close delegated OpenRouter async stream",
                exc_info=True,
            )

    async def close(self) -> Any:
        try:
            result = await self._call_close()
        except BaseException as exc:
            self._emit_once(exc)
            raise
        self._emit_once()
        return result

    async def aclose(self) -> Any:
        return await self.close()

    async def __aenter__(self) -> Self:
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            entered = enter()
            entered = await entered if inspect.isawaitable(entered) else entered
            if entered is not None and entered is not self._stream:
                self._iteration_stream = entered
                self._aiterator = None
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any:
        exit_method = getattr(self._stream, "__aexit__", None)
        try:
            if callable(exit_method):
                result = exit_method(exc_type, exc, traceback)
                result = await result if inspect.isawaitable(result) else result
            else:
                result = await self.close()
        except BaseException as close_error:
            self._emit_once(close_error)
            raise
        self._emit_once()
        return result


@dataclass(frozen=True)
class _DelegateMethodPatch:
    target: type[Any]
    attribute: str
    previous: Any
    installed: Any


@dataclass(eq=False)
class _OpenRouterRuntime:
    tracer_provider: Any
    delegate: Any
    processor: OpenRouterSpanProcessor
    owns_delegate_patches: bool
    normalize_all_openai_spans: bool
    capture_content: bool
    members: set[OpenRouterInstrumentor] = field(default_factory=set)
    delegate_kind_entries: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    bridged_emitters: dict[str, Any] = field(default_factory=dict)
    delegate_sync_stream_wrapper: Any = None
    delegate_async_stream_wrapper: Any = None
    bridge_sync_stream_wrapper: Any = None
    bridge_async_stream_wrapper: Any = None
    delegate_method_patches: list[_DelegateMethodPatch] = field(default_factory=list)
    delegate_class: type[Any] | None = None
    delegate_activate: Any = None
    delegate_deactivate: Any = None
    coordinated_activate: Any = None
    coordinated_deactivate: Any = None
    external_delegate_members: set[Any] = field(default_factory=set)
    preexisting_delegate_active: bool = False
    lifecycle_active: bool = False

    @property
    def config(self) -> tuple[bool, bool]:
        return self.normalize_all_openai_spans, self.capture_content


def _active_span_processors(tracer_provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if processors is None:
        return active_span_processor, None
    return active_span_processor, tuple(processors)


def _register_processor_before_exporters(
    tracer_provider: Any,
    processor: OpenRouterSpanProcessor,
) -> None:
    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    if any(existing_processor is processor for existing_processor in processors):
        return
    active_span_processor._span_processors = (processor, *processors)


def _unregister_processor(
    tracer_provider: Any,
    processor: OpenRouterSpanProcessor | None,
) -> None:
    if processor is None:
        return

    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        return
    active_span_processor._span_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )


def _emit_safely(emit_fn: Any, **kwargs: Any) -> None:
    try:
        emit_fn(**kwargs)
    except Exception:
        logger.exception("OpenRouter instrumentation failed to emit a delegated span")


def _delegate_instrumentation_suppressed() -> bool:
    """Read optional delegate suppression state across supported releases."""

    callback = getattr(
        openai_instrumentation,
        "_is_openai_instrumentation_suppressed",
        None,
    )
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception:
        logger.debug("Failed to read OpenAI delegate suppression state", exc_info=True)
        return False


def _make_safe_sync_wrapper(original: Any, *, kind: str) -> Any:
    emit_fn, _aggregate = openai_instrumentation._KINDS[kind]

    @wraps(original)
    def wrapper(resource: Any, *args: Any, **kwargs: Any) -> Any:
        if _delegate_instrumentation_suppressed():
            return original(resource, *args, **kwargs)

        start_ns = time.time_ns()
        try:
            response = original(resource, *args, **kwargs)
        except Exception as exc:
            _emit_safely(
                emit_fn,
                request_kwargs=kwargs,
                start_ns=start_ns,
                error_message=_safe_error_message(exc),
                status_code=_exception_status_code(exc),
            )
            raise
        if kind != "embedding" and openai_instrumentation._is_stream(response):
            return openai_instrumentation._wrap_sync_stream(
                response,
                kind=kind,
                request_kwargs=kwargs,
                start_ns=start_ns,
            )
        _emit_safely(
            emit_fn,
            request_kwargs=kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    wrapper.__respan_openrouter_wrapper__ = True
    return wrapper


def _make_safe_async_wrapper(original: Any, *, kind: str) -> Any:
    emit_fn, _aggregate = openai_instrumentation._KINDS[kind]

    @wraps(original)
    async def wrapper(resource: Any, *args: Any, **kwargs: Any) -> Any:
        if _delegate_instrumentation_suppressed():
            pending = original(resource, *args, **kwargs)
            return pending if hasattr(pending, "__aiter__") else await pending

        start_ns = time.time_ns()
        try:
            pending = original(resource, *args, **kwargs)
            response = pending if hasattr(pending, "__aiter__") else await pending
        except Exception as exc:
            _emit_safely(
                emit_fn,
                request_kwargs=kwargs,
                start_ns=start_ns,
                error_message=_safe_error_message(exc),
                status_code=_exception_status_code(exc),
            )
            raise
        if kind != "embedding" and openai_instrumentation._is_stream(response):
            return openai_instrumentation._wrap_async_stream(
                response,
                kind=kind,
                request_kwargs=kwargs,
                start_ns=start_ns,
            )
        _emit_safely(
            emit_fn,
            request_kwargs=kwargs,
            start_ns=start_ns,
            response=response,
        )
        return response

    wrapper.__respan_openrouter_wrapper__ = True
    return wrapper


def _is_replaceable_delegate_wrapper(current: Any, original: Any) -> bool:
    if getattr(current, "__respan_openrouter_wrapper__", False):
        return True
    if not inspect.isfunction(current):
        return False
    if current.__module__ != openai_instrumentation.__name__:
        return False
    try:
        return inspect.getclosurevars(current).nonlocals.get("original") is original
    except (TypeError, ValueError):
        return False


def _install_safe_delegate_methods(runtime: _OpenRouterRuntime) -> None:
    original_methods = getattr(openai_instrumentation, "_original_methods", {})
    targets = getattr(openai_instrumentation, "_TARGETS", ())
    for module_path, class_name, kind, is_async in targets:
        target = openai_instrumentation._load_class(module_path, class_name)
        if target is None:
            continue
        key = (target, "create")
        original = original_methods.get(key)
        if original is None:
            continue
        current = inspect.getattr_static(target, "create")
        if not _is_replaceable_delegate_wrapper(current, original):
            logger.warning(
                "OpenRouter left a foreign wrapper on %s.%s unchanged",
                target.__name__,
                "create",
            )
            continue
        factory = _make_safe_async_wrapper if is_async else _make_safe_sync_wrapper
        installed = factory(original, kind=kind)
        target.create = installed
        runtime.delegate_method_patches.append(
            _DelegateMethodPatch(target, "create", current, installed)
        )


def _restore_safe_delegate_methods(runtime: _OpenRouterRuntime) -> None:
    for patch in reversed(runtime.delegate_method_patches):
        if inspect.getattr_static(patch.target, patch.attribute) is patch.installed:
            setattr(patch.target, patch.attribute, patch.previous)
    runtime.delegate_method_patches.clear()


def _install_delegate_lifecycle(runtime: _OpenRouterRuntime) -> None:
    delegate_class = type(runtime.delegate)
    original_activate = delegate_class.activate
    original_deactivate = delegate_class.deactivate
    runtime.delegate_class = delegate_class
    runtime.delegate_activate = original_activate
    runtime.delegate_deactivate = original_deactivate
    runtime.preexisting_delegate_active = not runtime.owns_delegate_patches
    runtime.lifecycle_active = True

    def coordinated_activate(delegate_self: Any) -> None:
        with _BRIDGE_LOCK:
            if runtime.lifecycle_active:
                if delegate_self is not runtime.delegate:
                    runtime.external_delegate_members.add(delegate_self)
                delegate_self._is_instrumented = True
                return
        original_activate(delegate_self)

    def coordinated_deactivate(delegate_self: Any) -> None:
        with _BRIDGE_LOCK:
            if runtime.lifecycle_active:
                if delegate_self in runtime.external_delegate_members:
                    runtime.external_delegate_members.discard(delegate_self)
                    delegate_self._is_instrumented = False
                    return
                if delegate_self is not runtime.delegate:
                    runtime.preexisting_delegate_active = False
                    delegate_self._is_instrumented = False
                    return
                return
        original_deactivate(delegate_self)

    runtime.coordinated_activate = coordinated_activate
    runtime.coordinated_deactivate = coordinated_deactivate
    delegate_class.activate = coordinated_activate
    delegate_class.deactivate = coordinated_deactivate


def _restore_delegate_lifecycle(runtime: _OpenRouterRuntime) -> None:
    runtime.lifecycle_active = False
    delegate_class = runtime.delegate_class
    if delegate_class is not None:
        if delegate_class.activate is runtime.coordinated_activate:
            delegate_class.activate = runtime.delegate_activate
        if delegate_class.deactivate is runtime.coordinated_deactivate:
            delegate_class.deactivate = runtime.delegate_deactivate
    runtime.coordinated_activate = None
    runtime.coordinated_deactivate = None


def _has_external_delegate_owner(runtime: _OpenRouterRuntime) -> bool:
    return runtime.preexisting_delegate_active or any(
        bool(getattr(member, "_is_instrumented", False))
        for member in runtime.external_delegate_members
    )


class OpenRouterInstrumentor:
    """Respan instrumentor for OpenRouter's OpenAI-compatible Python usage."""

    name = OPENROUTER_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        normalize_all_openai_spans: bool = True,
        capture_content: bool = True,
    ) -> None:
        self._normalize_all_openai_spans = normalize_all_openai_spans
        self._capture_content = capture_content
        self._delegate = None
        self._processor: OpenRouterSpanProcessor | None = None
        self._is_instrumented = False
        self._runtime: _OpenRouterRuntime | None = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    @staticmethod
    def _install_delegate_bridge(runtime: _OpenRouterRuntime) -> None:
        with _BRIDGE_LOCK:
            kinds = getattr(openai_instrumentation, "_KINDS", None)
            if not isinstance(kinds, dict):
                raise TypeError("OpenAI delegate does not expose its emission registry")

            runtime.delegate_kind_entries = dict(kinds)
            for kind, entry in runtime.delegate_kind_entries.items():
                emit_fn, aggregate = entry

                def bridged_emit(
                    *,
                    _kind: str = kind,
                    _emit_fn: Any = emit_fn,
                    **kwargs: Any,
                ) -> Any:
                    request_kwargs = kwargs.get("request_kwargs") or {}
                    if not isinstance(request_kwargs, dict):
                        request_kwargs = {}
                    error_message = _safe_error_message(kwargs.get("error_message"))
                    status_value = kwargs.get("status_code", 200)
                    if type(status_value) is int:
                        status_code = status_value
                    elif (
                        type(status_value) is str
                        and len(status_value) == 3
                        and status_value.isascii()
                        and status_value.isdigit()
                    ):
                        status_code = int(status_value)
                    else:
                        status_code = 500 if error_message else 200
                    with openrouter_emission_context(
                        kind=_kind,
                        request_kwargs=request_kwargs,
                        error_message=error_message,
                        status_code=status_code,
                    ):
                        safe_kwargs = dict(kwargs)
                        safe_kwargs["request_kwargs"] = request_kwargs
                        safe_kwargs["error_message"] = error_message
                        safe_kwargs["status_code"] = status_code
                        return _emit_fn(**safe_kwargs)

                runtime.bridged_emitters[kind] = bridged_emit
                kinds[kind] = (bridged_emit, aggregate)

            runtime.delegate_sync_stream_wrapper = (
                openai_instrumentation._wrap_sync_stream
            )
            runtime.delegate_async_stream_wrapper = (
                openai_instrumentation._wrap_async_stream
            )

            def sync_stream_wrapper(
                iterator: Any,
                *,
                kind: str,
                request_kwargs: dict[str, Any],
                start_ns: int,
            ) -> _SyncStreamProxy:
                emit_fn, _aggregate = openai_instrumentation._KINDS[kind]
                return _SyncStreamProxy(
                    iterator,
                    kind=kind,
                    request_kwargs=request_kwargs,
                    start_ns=start_ns,
                    emit_fn=emit_fn,
                )

            def async_stream_wrapper(
                aiterator: Any,
                *,
                kind: str,
                request_kwargs: dict[str, Any],
                start_ns: int,
            ) -> _AsyncStreamProxy:
                emit_fn, _aggregate = openai_instrumentation._KINDS[kind]
                return _AsyncStreamProxy(
                    aiterator,
                    kind=kind,
                    request_kwargs=request_kwargs,
                    start_ns=start_ns,
                    emit_fn=emit_fn,
                )

            runtime.bridge_sync_stream_wrapper = sync_stream_wrapper
            runtime.bridge_async_stream_wrapper = async_stream_wrapper
            openai_instrumentation._wrap_sync_stream = sync_stream_wrapper
            openai_instrumentation._wrap_async_stream = async_stream_wrapper

    @staticmethod
    def _uninstall_delegate_bridge(runtime: _OpenRouterRuntime) -> None:
        with _BRIDGE_LOCK:
            kinds = getattr(openai_instrumentation, "_KINDS", {})
            for kind, original_entry in runtime.delegate_kind_entries.items():
                current_entry = kinds.get(kind)
                if (
                    isinstance(current_entry, tuple)
                    and current_entry
                    and current_entry[0] is runtime.bridged_emitters.get(kind)
                ):
                    kinds[kind] = original_entry

            if (
                getattr(openai_instrumentation, "_wrap_sync_stream", None)
                is runtime.bridge_sync_stream_wrapper
            ):
                openai_instrumentation._wrap_sync_stream = (
                    runtime.delegate_sync_stream_wrapper
                )
            if (
                getattr(openai_instrumentation, "_wrap_async_stream", None)
                is runtime.bridge_async_stream_wrapper
            ):
                openai_instrumentation._wrap_async_stream = (
                    runtime.delegate_async_stream_wrapper
                )

            runtime.delegate_kind_entries.clear()
            runtime.bridged_emitters.clear()
            runtime.delegate_sync_stream_wrapper = None
            runtime.delegate_async_stream_wrapper = None
            runtime.bridge_sync_stream_wrapper = None
            runtime.bridge_async_stream_wrapper = None

    def activate(self) -> None:
        """Instrument OpenRouter calls made through the OpenAI Python client."""
        global _ACTIVE_BRIDGE_OWNER, _ACTIVE_RUNTIME

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "OpenRouter instrumentation skipped because Respan tracing is disabled"
            )
            return

        requested_config = (
            self._normalize_all_openai_spans,
            self._capture_content,
        )
        with _BRIDGE_LOCK:
            if _ACTIVE_RUNTIME is not None:
                if _ACTIVE_RUNTIME.config != requested_config:
                    logger.error(
                        "OpenRouter instrumentation config mismatch: active "
                        "normalize_all_openai_spans=%s capture_content=%s; "
                        "requested normalize_all_openai_spans=%s capture_content=%s",
                        *_ACTIVE_RUNTIME.config,
                        *requested_config,
                    )
                    return
                _ACTIVE_RUNTIME.members.add(self)
                self._runtime = _ACTIVE_RUNTIME
                self._delegate = _ACTIVE_RUNTIME.delegate
                self._processor = _ACTIVE_RUNTIME.processor
                self._is_instrumented = True
                if _ACTIVE_BRIDGE_OWNER is None:
                    _ACTIVE_BRIDGE_OWNER = self
                logger.info("OpenRouter instrumentation joined the active runtime")
                return

            tracer_provider = trace.get_tracer_provider()
            delegate = None
            processor = None
            runtime = None
            delegate_methods = getattr(
                openai_instrumentation,
                "_original_methods",
                {},
            )
            delegate_was_active = bool(delegate_methods)
            try:
                delegate = OpenAIInstrumentor()
                processor = OpenRouterSpanProcessor(
                    normalize_all_openai_spans=self._normalize_all_openai_spans,
                    capture_content=self._capture_content,
                )
                runtime = _OpenRouterRuntime(
                    tracer_provider=tracer_provider,
                    delegate=delegate,
                    processor=processor,
                    owns_delegate_patches=not delegate_was_active,
                    normalize_all_openai_spans=self._normalize_all_openai_spans,
                    capture_content=self._capture_content,
                )
                _register_processor_before_exporters(
                    tracer_provider=tracer_provider,
                    processor=processor,
                )
                # The delegate wrapper factories capture their emitter at activation
                # time. Install the bridge first so non-streaming failures retain the
                # safe message and precise provider status.
                self._install_delegate_bridge(runtime)
                delegate.activate()
                if (
                    getattr(delegate, "_is_instrumented", True) is False
                    and not delegate_was_active
                ):
                    self._uninstall_delegate_bridge(runtime)
                    _unregister_processor(tracer_provider, processor)
                    return
                _install_safe_delegate_methods(runtime)
                _install_delegate_lifecycle(runtime)
            except Exception:
                if runtime is not None:
                    _restore_delegate_lifecycle(runtime)
                    _restore_safe_delegate_methods(runtime)
                    self._uninstall_delegate_bridge(runtime)
                _unregister_processor(
                    tracer_provider=tracer_provider,
                    processor=processor,
                )
                if delegate is not None and not delegate_was_active:
                    try:
                        delegate.deactivate()
                    except Exception:
                        logger.exception(
                            "Failed to clean up OpenRouter instrumentation"
                        )
                logger.exception("Failed to activate OpenRouter instrumentation")
                return

            runtime.members.add(self)
            _ACTIVE_RUNTIME = runtime
            _ACTIVE_BRIDGE_OWNER = self
            self._runtime = runtime
            self._delegate = delegate
            self._processor = processor
            self._is_instrumented = True
            logger.info("OpenRouter instrumentation activated")

    def deactivate(self) -> None:
        """Release this instance and tear down the shared runtime when last."""
        global _ACTIVE_BRIDGE_OWNER, _ACTIVE_RUNTIME

        with _BRIDGE_LOCK:
            runtime = self._runtime
            if runtime is None or not self._is_instrumented:
                self._delegate = None
                self._processor = None
                self._runtime = None
                self._is_instrumented = False
                return

            runtime.members.discard(self)
            self._delegate = None
            self._processor = None
            self._runtime = None
            self._is_instrumented = False

            if runtime.members:
                if _ACTIVE_BRIDGE_OWNER is self:
                    _ACTIVE_BRIDGE_OWNER = next(iter(runtime.members))
                logger.info(
                    "OpenRouter instrumentation instance released; shared runtime retained"
                )
                return

            if _ACTIVE_RUNTIME is runtime:
                _ACTIVE_RUNTIME = None
            if _ACTIVE_BRIDGE_OWNER is self or not runtime.members:
                _ACTIVE_BRIDGE_OWNER = None

            external_delegate_owner = _has_external_delegate_owner(runtime)
            _restore_delegate_lifecycle(runtime)
            self._uninstall_delegate_bridge(runtime)
            _unregister_processor(
                tracer_provider=runtime.tracer_provider,
                processor=runtime.processor,
            )
            if external_delegate_owner:
                # Restore the exact delegate wrappers that the independent owner
                # joined; its eventual deactivate call still owns final teardown.
                _restore_safe_delegate_methods(runtime)
            else:
                try:
                    runtime.delegate_deactivate(runtime.delegate)
                except Exception:
                    logger.exception(
                        "Failed to deactivate OpenRouter delegate instrumentation"
                    )
                finally:
                    _restore_safe_delegate_methods(runtime)
            logger.info("OpenRouter instrumentation deactivated")
