"""IBM watsonx.ai SDK instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from threading import Condition, RLock
from typing import Any

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_watsonx._constants import (
    ACHAT_METHOD_NAME,
    ACHAT_STREAM_METHOD_NAME,
    AEMBED_DOCUMENTS_METHOD_NAME,
    AEMBED_QUERY_METHOD_NAME,
    AEMBEDDINGS_GENERATE_METHOD_NAME,
    AGENERATE_METHOD_NAME,
    AGENERATE_STREAM_METHOD_NAME,
    CHAT_METHOD_NAME,
    CHAT_STREAM_METHOD_NAME,
    EMBED_DOCUMENTS_METHOD_NAME,
    EMBED_QUERY_METHOD_NAME,
    EMBEDDINGS_CLASS_NAME,
    EMBEDDINGS_GENERATE_METHOD_NAME,
    EMBEDDINGS_MODULE,
    GENERATE_METHOD_NAME,
    GENERATE_TEXT_METHOD_NAME,
    GENERATE_TEXT_STREAM_METHOD_NAME,
    MODEL_INFERENCE_CLASS_NAME,
    MODEL_INFERENCE_MODULE,
    WATSONX_INSTRUMENTATION_NAME,
)
from respan_instrumentation_watsonx._otel_emitter import (
    current_trace_parent_ids,
    emit_chat_span,
    emit_embedding_span,
    emit_text_span,
)
from respan_instrumentation_watsonx._serialization import (
    provider_status_code,
    safe_exception_message,
)

logger = logging.getLogger(__name__)

_original_methods: dict[tuple[type[Any], str], Any] = {}
_installed_methods: dict[tuple[type[Any], str], Any] = {}
_activation_lock = RLock()
_activation_condition = Condition(_activation_lock)
_activation_count = 0
_activation_installing = False
_MAX_STREAM_CHUNKS = 256

_TEXT_PROMPT_METHODS = {
    GENERATE_METHOD_NAME,
    GENERATE_TEXT_METHOD_NAME,
    GENERATE_TEXT_STREAM_METHOD_NAME,
    AGENERATE_METHOD_NAME,
    AGENERATE_STREAM_METHOD_NAME,
}
_CHAT_MESSAGE_METHODS = {
    CHAT_METHOD_NAME,
    CHAT_STREAM_METHOD_NAME,
    ACHAT_METHOD_NAME,
    ACHAT_STREAM_METHOD_NAME,
}
_EMBEDDING_INPUT_METHODS = {
    EMBEDDINGS_GENERATE_METHOD_NAME,
    AEMBEDDINGS_GENERATE_METHOD_NAME,
}
_EMBEDDING_TEXTS_METHODS = {
    EMBED_DOCUMENTS_METHOD_NAME,
    AEMBED_DOCUMENTS_METHOD_NAME,
}
_EMBEDDING_TEXT_METHODS = {
    EMBED_QUERY_METHOD_NAME,
    AEMBED_QUERY_METHOD_NAME,
}


def _get_module_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    attr_value = getattr(module, attr_name, None)
    if attr_value is None:
        raise AttributeError(f"{module_path}.{attr_name}")
    return attr_value


def _load_watsonx_classes() -> tuple[type[Any], type[Any]]:
    return (
        _get_module_attr(MODEL_INFERENCE_MODULE, MODEL_INFERENCE_CLASS_NAME),
        _get_module_attr(EMBEDDINGS_MODULE, EMBEDDINGS_CLASS_NAME),
    )


def _request_kwargs_from_call(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    positional_arg_name: str | None = None,
) -> dict[str, Any]:
    result = dict(kwargs)
    if not args:
        return result
    if positional_arg_name is not None:
        result.setdefault(positional_arg_name, args[0])
        return result
    if method_name in _TEXT_PROMPT_METHODS:
        result.setdefault("prompt", args[0])
    elif method_name in _CHAT_MESSAGE_METHODS:
        result.setdefault("messages", args[0])
    elif method_name in _EMBEDDING_INPUT_METHODS:
        result.setdefault("inputs", args[0])
    elif method_name in _EMBEDDING_TEXTS_METHODS:
        result.setdefault("texts", args[0])
    elif method_name in _EMBEDDING_TEXT_METHODS:
        result.setdefault("text", args[0])
    return result


def _is_iterator(value: Any) -> bool:
    if isinstance(value, str | bytes | bytearray | dict | list | tuple):
        return False
    return hasattr(value, "__iter__")


def _append_stream_chunk(chunks: list[Any], chunk: Any) -> None:
    """Keep bounded raw stream state while retaining the terminal chunk."""
    if len(chunks) < _MAX_STREAM_CHUNKS:
        chunks.append(chunk)
    else:
        chunks[-1] = chunk


def _wrap_sync_iterator(
    *,
    iterator: Iterator[Any],
    emit: Callable[..., None],
    instance: Any,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_kwarg_name: str,
    trace_id: str | None,
    parent_id: str | None,
) -> Iterator[Any]:
    chunks: list[Any] = []
    emitted = False
    try:
        for chunk in iterator:
            _append_stream_chunk(chunks, chunk)
            yield chunk
    except GeneratorExit:
        emit(
            instance=instance,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: chunks},
            is_streaming=True,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        emitted = True
        raise
    except BaseException as exc:
        emit(
            instance=instance,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: chunks},
            error_message=safe_exception_message(exc),
            status_code=provider_status_code(exc),
            is_streaming=True,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        emitted = True
        raise
    else:
        emit(
            instance=instance,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: chunks},
            is_streaming=True,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        emitted = True
    finally:
        if not emitted:
            emit(
                instance=instance,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                **{response_kwarg_name: chunks},
                is_streaming=True,
                trace_id=trace_id,
                parent_id=parent_id,
            )
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                logger.debug("Failed to close Watsonx stream", exc_info=True)


async def _wrap_async_iterator(
    *,
    async_iterator: AsyncIterator[Any],
    emit: Callable[..., None],
    instance: Any,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_kwarg_name: str,
    trace_id: str | None,
    parent_id: str | None,
) -> AsyncIterator[Any]:
    chunks: list[Any] = []
    emitted = False
    try:
        async for chunk in async_iterator:
            _append_stream_chunk(chunks, chunk)
            yield chunk
    except (GeneratorExit, StopAsyncIteration):
        emit(
            instance=instance,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: chunks},
            is_streaming=True,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        emitted = True
        raise
    except BaseException as exc:
        emit(
            instance=instance,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: chunks},
            error_message=safe_exception_message(exc),
            status_code=provider_status_code(exc),
            is_streaming=True,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        emitted = True
        raise
    else:
        emit(
            instance=instance,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: chunks},
            is_streaming=True,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        emitted = True
    finally:
        if not emitted:
            emit(
                instance=instance,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                **{response_kwarg_name: chunks},
                is_streaming=True,
                trace_id=trace_id,
                parent_id=parent_id,
            )
        close = getattr(async_iterator, "aclose", None)
        if callable(close):
            try:
                await close()
            except BaseException:
                logger.debug("Failed to close Watsonx async stream", exc_info=True)


def _wrap_sync_method(
    *,
    original: Any,
    method_name: str,
    emit: Callable[..., None],
    response_kwarg_name: str,
    positional_arg_name: str | None = None,
    stream: bool = False,
) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = current_trace_parent_ids()
        request_kwargs = _request_kwargs_from_call(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            positional_arg_name=positional_arg_name,
        )
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            emit(
                instance=self,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=safe_exception_message(exc),
                status_code=provider_status_code(exc),
                trace_id=trace_id,
                parent_id=parent_id,
            )
            raise

        if stream or _is_iterator(response):
            return _wrap_sync_iterator(
                iterator=response,
                emit=emit,
                instance=self,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                response_kwarg_name=response_kwarg_name,
                trace_id=trace_id,
                parent_id=parent_id,
            )

        emit(
            instance=self,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: response},
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


def _wrap_async_method(
    *,
    original: Any,
    method_name: str,
    emit: Callable[..., None],
    response_kwarg_name: str,
    positional_arg_name: str | None = None,
    stream: bool = False,
) -> Any:
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        trace_id, parent_id = current_trace_parent_ids()
        request_kwargs = _request_kwargs_from_call(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            positional_arg_name=positional_arg_name,
        )
        try:
            pending_response = original(self, *args, **kwargs)
            if hasattr(pending_response, "__aiter__"):
                response = pending_response
            else:
                response = await pending_response
        except BaseException as exc:
            emit(
                instance=self,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                error_message=safe_exception_message(exc),
                status_code=provider_status_code(exc),
                trace_id=trace_id,
                parent_id=parent_id,
            )
            raise

        if stream or hasattr(response, "__aiter__"):
            return _wrap_async_iterator(
                async_iterator=response,
                emit=emit,
                instance=self,
                request_kwargs=request_kwargs,
                start_ns=start_ns,
                response_kwarg_name=response_kwarg_name,
                trace_id=trace_id,
                parent_id=parent_id,
            )

        emit(
            instance=self,
            request_kwargs=request_kwargs,
            start_ns=start_ns,
            **{response_kwarg_name: response},
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


def _patch_method(target_class: type[Any], method_name: str, replacement: Any) -> None:
    original = getattr(target_class, method_name, None)
    if original is None:
        return
    key = (target_class, method_name)
    if key not in _original_methods:
        _original_methods[key] = original
    wrapped = replacement(original=_original_methods[key])
    setattr(target_class, method_name, wrapped)
    _installed_methods[key] = wrapped


def _complete_installation(*, success: bool) -> None:
    global _activation_count, _activation_installing
    with _activation_condition:
        if success:
            _activation_count = 1
        _activation_installing = False
        _activation_condition.notify_all()


class WatsonxInstrumentor:
    """Respan instrumentor for the IBM watsonx.ai Python SDK."""

    name = WATSONX_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch Watsonx model and embedding methods."""
        global _activation_count, _activation_installing
        with _activation_condition:
            if self._is_instrumented:
                return
            if not self._is_respan_tracing_enabled():
                logger.info(
                    "Watsonx instrumentation skipped because Respan tracing is disabled"
                )
                return
            while _activation_installing:
                _activation_condition.wait()
            if _activation_count:
                _activation_count += 1
                self._is_instrumented = True
                return
            _activation_installing = True
            self._is_instrumented = True

        try:
            ModelInference, Embeddings = _load_watsonx_classes()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Watsonx instrumentation - missing dependency: %s",
                exc,
            )
            self.deactivate()
            _complete_installation(success=False)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to activate Watsonx instrumentation: %s", exc)
            self.deactivate()
            _complete_installation(success=False)
            return

        try:
            for method_name in (GENERATE_METHOD_NAME, GENERATE_TEXT_METHOD_NAME):
                _patch_method(
                    ModelInference,
                    method_name,
                    lambda original, method_name=method_name: _wrap_sync_method(
                        original=original,
                        method_name=method_name,
                        emit=emit_text_span,
                        response_kwarg_name="response_or_chunks",
                        positional_arg_name="prompt",
                    ),
                )
            _patch_method(
                ModelInference,
                GENERATE_TEXT_STREAM_METHOD_NAME,
                lambda original: _wrap_sync_method(
                    original=original,
                    method_name=GENERATE_TEXT_STREAM_METHOD_NAME,
                    emit=emit_text_span,
                    response_kwarg_name="response_or_chunks",
                    positional_arg_name="prompt",
                    stream=True,
                ),
            )
            for method_name in (AGENERATE_METHOD_NAME,):
                _patch_method(
                    ModelInference,
                    method_name,
                    lambda original, method_name=method_name: _wrap_async_method(
                        original=original,
                        method_name=method_name,
                        emit=emit_text_span,
                        response_kwarg_name="response_or_chunks",
                        positional_arg_name="prompt",
                    ),
                )
            _patch_method(
                ModelInference,
                AGENERATE_STREAM_METHOD_NAME,
                lambda original: _wrap_async_method(
                    original=original,
                    method_name=AGENERATE_STREAM_METHOD_NAME,
                    emit=emit_text_span,
                    response_kwarg_name="response_or_chunks",
                    positional_arg_name="prompt",
                    stream=True,
                ),
            )
            _patch_method(
                ModelInference,
                CHAT_METHOD_NAME,
                lambda original: _wrap_sync_method(
                    original=original,
                    method_name=CHAT_METHOD_NAME,
                    emit=emit_chat_span,
                    response_kwarg_name="response_or_chunks",
                    positional_arg_name="messages",
                ),
            )
            _patch_method(
                ModelInference,
                CHAT_STREAM_METHOD_NAME,
                lambda original: _wrap_sync_method(
                    original=original,
                    method_name=CHAT_STREAM_METHOD_NAME,
                    emit=emit_chat_span,
                    response_kwarg_name="response_or_chunks",
                    positional_arg_name="messages",
                    stream=True,
                ),
            )
            _patch_method(
                ModelInference,
                ACHAT_METHOD_NAME,
                lambda original: _wrap_async_method(
                    original=original,
                    method_name=ACHAT_METHOD_NAME,
                    emit=emit_chat_span,
                    response_kwarg_name="response_or_chunks",
                    positional_arg_name="messages",
                ),
            )
            _patch_method(
                ModelInference,
                ACHAT_STREAM_METHOD_NAME,
                lambda original: _wrap_async_method(
                    original=original,
                    method_name=ACHAT_STREAM_METHOD_NAME,
                    emit=emit_chat_span,
                    response_kwarg_name="response_or_chunks",
                    positional_arg_name="messages",
                    stream=True,
                ),
            )

            for method_name in (
                EMBEDDINGS_GENERATE_METHOD_NAME,
                EMBED_DOCUMENTS_METHOD_NAME,
                EMBED_QUERY_METHOD_NAME,
            ):
                positional_arg_name = "inputs"
                if method_name == EMBED_DOCUMENTS_METHOD_NAME:
                    positional_arg_name = "texts"
                elif method_name == EMBED_QUERY_METHOD_NAME:
                    positional_arg_name = "text"
                _patch_method(
                    Embeddings,
                    method_name,
                    lambda original, method_name=method_name, positional_arg_name=positional_arg_name: (
                        _wrap_sync_method(
                            original=original,
                            method_name=method_name,
                            emit=emit_embedding_span,
                            response_kwarg_name="response",
                            positional_arg_name=positional_arg_name,
                        )
                    ),
                )
            for method_name in (
                AEMBEDDINGS_GENERATE_METHOD_NAME,
                AEMBED_DOCUMENTS_METHOD_NAME,
                AEMBED_QUERY_METHOD_NAME,
            ):
                positional_arg_name = "inputs"
                if method_name == AEMBED_DOCUMENTS_METHOD_NAME:
                    positional_arg_name = "texts"
                elif method_name == AEMBED_QUERY_METHOD_NAME:
                    positional_arg_name = "text"
                _patch_method(
                    Embeddings,
                    method_name,
                    lambda original, method_name=method_name, positional_arg_name=positional_arg_name: (
                        _wrap_async_method(
                            original=original,
                            method_name=method_name,
                            emit=emit_embedding_span,
                            response_kwarg_name="response",
                            positional_arg_name=positional_arg_name,
                        )
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to activate Watsonx instrumentation: %s", exc)
            self.deactivate()
            _complete_installation(success=False)
            return

        _complete_installation(success=True)
        logger.info("Watsonx instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Watsonx SDK methods."""
        global _activation_count
        with _activation_lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _activation_count = max(_activation_count - 1, 0)
            if _activation_count:
                return
            for (target_class, method_name), original in list(
                _original_methods.items()
            ):
                try:
                    current = getattr(target_class, method_name, None)
                    installed = _installed_methods.get((target_class, method_name))
                    if current is installed:
                        setattr(target_class, method_name, original)
                except Exception:
                    logger.debug(
                        "Failed to restore Watsonx method %s.%s",
                        target_class,
                        method_name,
                        exc_info=True,
                    )
                finally:
                    _original_methods.pop((target_class, method_name), None)
                    _installed_methods.pop((target_class, method_name), None)
        logger.info("Watsonx instrumentation deactivated")
