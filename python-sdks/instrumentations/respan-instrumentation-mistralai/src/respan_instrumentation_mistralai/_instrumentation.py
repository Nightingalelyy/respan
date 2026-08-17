"""Mistral AI instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from threading import RLock
from types import TracebackType
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.constants.tracing import SAMPLE_RATE_ATTR
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

MISTRALAI_INSTRUMENTATION_NAME = "mistralai"
OPENINFERENCE_MISTRALAI_MODULE = "openinference.instrumentation.mistralai"
MISTRALAI_SDK_TRACER_NAME = "mistralai_sdk_tracer"
_OFF_CONTRACT_ALIAS_KEYS = (
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    TLSpanAttributes.TRACELOOP_SPAN_KIND,
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
)
_GEN_AI_MESSAGE_PREFIXES = (
    f"{TLSpanAttributes.LLM_PROMPTS}.",
    f"{TLSpanAttributes.LLM_COMPLETIONS}.",
)
_TOOL_CALLS_SUFFIX = ".tool_calls"
_STATUS_CODE_PATTERN = re.compile(r"\bStatus\s+(?P<status>[1-5]\d{2})\b", re.IGNORECASE)
_SECRET_KEY_PATTERN = re.compile(
    r"(?:authorization|api[-_ ]?key|access[-_ ]?token|password|secret)",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|password|secret)"
    r"\s*[:=]\s*[^\s,;]+"
)
_MAX_JSON_LENGTH = 16_384
_MAX_SOURCE_JSON_LENGTH = 65_536
_MAX_STRING_LENGTH = 4_096
_MAX_COLLECTION_ITEMS = 64
_MAX_JSON_DEPTH = 8
_REDACTED = "[REDACTED]"
_TRUNCATED = "[TRUNCATED]"
_UNSUPPORTED = "[UNSUPPORTED]"
_LIFECYCLE_LOCK = RLock()
_SHARED_DELEGATE: OpenInferenceInstrumentor | None = None
_SHARED_CLEANUP_PROCESSOR: _MistralAIOffContractAliasProcessor | None = None
_SHARED_STREAM_GUARD_PATCH: tuple[Any, type, type] | None = None
_SHARED_INSTRUMENTOR_KWARGS: dict[str, Any] | None = None
_SHARED_REFCOUNT = 0


def _load_openinference_mistralai_class() -> type:
    mistralai_module = importlib.import_module(OPENINFERENCE_MISTRALAI_MODULE)
    return mistralai_module.MistralAIInstrumentor


def _is_mistralai_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return scope_name == OPENINFERENCE_MISTRALAI_MODULE


def _is_mistralai_sdk_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return scope_name == MISTRALAI_SDK_TRACER_NAME


def _is_gen_ai_tool_calls_attr(key: str) -> bool:
    return key.endswith(_TOOL_CALLS_SUFFIX) and key.startswith(_GEN_AI_MESSAGE_PREFIXES)


def _bounded_redacted_text(value: str) -> str:
    bounded = value[:_MAX_STRING_LENGTH]
    if len(value) > _MAX_STRING_LENGTH:
        bounded += _TRUNCATED
    bounded = _BEARER_PATTERN.sub(f"Bearer {_REDACTED}", bounded)
    return _INLINE_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}={_REDACTED}",
        bounded,
    )


def _safe_scalar_text(value: Any) -> str | None:
    if isinstance(value, str):
        return _bounded_redacted_text(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return None


def _safe_json_value(
    value: Any,
    *,
    depth: int = 0,
    key: str | None = None,
) -> Any:
    if key is not None and _SECRET_KEY_PATTERN.search(key):
        return _REDACTED
    if depth >= _MAX_JSON_DEPTH:
        return _TRUNCATED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _bounded_redacted_text(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        truncated = False
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                truncated = True
                break
            safe_key = _safe_scalar_text(raw_key)
            if safe_key is None:
                continue
            result[safe_key] = _safe_json_value(
                item,
                depth=depth + 1,
                key=safe_key,
            )
        if truncated:
            result[_TRUNCATED] = True
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        result = []
        truncated = False
        for index, item in enumerate(value):
            if index >= _MAX_COLLECTION_ITEMS:
                truncated = True
                break
            result.append(_safe_json_value(item, depth=depth + 1))
        if truncated:
            result.append(_TRUNCATED)
        return result
    return _UNSUPPORTED


def _safe_json_str(value: Any) -> str:
    candidate = value
    if isinstance(value, str) and len(value) <= _MAX_SOURCE_JSON_LENGTH:
        try:
            candidate = json.loads(value)
        except (TypeError, ValueError):
            candidate = value

    sanitized = _safe_json_value(candidate)
    encoded = json.dumps(
        sanitized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(encoded) <= _MAX_JSON_LENGTH:
        return encoded

    preview = encoded[: _MAX_JSON_LENGTH // 2]
    return json.dumps(
        {"preview": preview, "truncated": True},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value or len(value) > _MAX_SOURCE_JSON_LENGTH:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_tool_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    normalized: dict[str, Any] = {}
    tool_call_id = value.get("id")
    if tool_call_id not in (None, "") and (safe_id := _safe_scalar_text(tool_call_id)):
        normalized["id"] = safe_id

    tool_type = value.get("type")
    if tool_type not in (None, "") and (safe_type := _safe_scalar_text(tool_type)):
        normalized["type"] = safe_type

    function = value.get("function")
    if isinstance(function, dict):
        normalized_function: dict[str, Any] = {}
        function_name = function.get("name")
        if function_name not in (None, "") and (
            safe_name := _safe_scalar_text(function_name)
        ):
            normalized_function["name"] = safe_name
        arguments = function.get("arguments")
        if arguments is not None:
            normalized_function["arguments"] = _safe_json_str(arguments)
        if normalized_function:
            normalized["function"] = normalized_function

    if "function" in normalized and "type" not in normalized:
        normalized["type"] = "function"
    return normalized or None


def _current_turn_tool_calls(
    output_payload: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    response_payload = output_payload.get("data", output_payload)
    if not isinstance(response_payload, dict):
        return {}
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return {}

    result: dict[int, list[dict[str, Any]]] = {}
    for fallback_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        choice_index = choice.get("index", fallback_index)
        if not isinstance(choice_index, int):
            choice_index = fallback_index
        message = choice.get("message") or choice.get("delta")
        if not isinstance(message, dict):
            continue
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        calls = [
            normalized
            for raw_call in raw_calls
            if (normalized := _normalize_tool_call(raw_call)) is not None
        ]
        if calls:
            result[choice_index] = calls
    return result


def _exception_details(span: ReadableSpan) -> tuple[str | None, str | None]:
    for event in reversed(tuple(getattr(span, "events", ()) or ())):
        if getattr(event, "name", None) != "exception":
            continue
        attrs = dict(getattr(event, "attributes", {}) or {})
        error_type = attrs.get("exception.type")
        error_message = attrs.get("exception.message")
        return (
            safe_error_type.rsplit(".", 1)[-1]
            if (safe_error_type := _safe_scalar_text(error_type))
            else None,
            _safe_scalar_text(error_message),
        )
    return None, None


def _error_status_code(attrs: dict[str, Any], error_message: str | None) -> int | None:
    for key in ("http.response.status_code", "http.status_code", "status_code"):
        value = attrs.get(key)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
        if isinstance(value, str) and value.isdigit() and 100 <= int(value) <= 599:
            return int(value)
    if error_message and (match := _STATUS_CODE_PATTERN.search(error_message)):
        return int(match.group("status"))
    return None


class _MistralAIOffContractAliasProcessor(SpanProcessor):
    """Normalize Mistral spans before the Respan exporter sees them."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if _is_mistralai_sdk_span(span):
            original_attrs = getattr(span, "_attributes", None)
            if original_attrs is not None:
                attrs = dict(original_attrs)
                attrs[SAMPLE_RATE_ATTR] = 0
                span._attributes = attrs
            return

        if not _is_mistralai_span(span):
            return

        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        for key in _OFF_CONTRACT_ALIAS_KEYS:
            attrs.pop(key, None)

        raw_input = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_INPUT)
        request_payload = _json_object(raw_input)
        if raw_input is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json_str(raw_input)
        if request_payload is not None:
            stream = request_payload.get("stream")
            if isinstance(stream, bool):
                attrs[TLSpanAttributes.LLM_IS_STREAMING] = stream

            tools = request_payload.get("tools")
            if isinstance(tools, list) and tools:
                attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = _safe_json_str(tools)

        raw_output = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT)
        output_payload = _json_object(raw_output)
        if raw_output is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json_str(raw_output)
        if output_payload is not None:
            for choice_index, tool_calls in _current_turn_tool_calls(
                output_payload
            ).items():
                attrs[
                    f"{TLSpanAttributes.LLM_COMPLETIONS}.{choice_index}.tool_calls"
                ] = _safe_json_str(tool_calls)

        for key, value in list(attrs.items()):
            if _is_gen_ai_tool_calls_attr(key):
                attrs[key] = _safe_json_str(value)
        request_functions = attrs.get(TLSpanAttributes.LLM_REQUEST_FUNCTIONS)
        if request_functions is not None:
            attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = _safe_json_str(
                request_functions
            )

        status = getattr(span, "status", None)
        status_code = getattr(status, "status_code", None)
        if status_code is trace.StatusCode.ERROR:
            error_type, error_message = _exception_details(span)
            provider_status = _error_status_code(attrs, error_message)
            effective_status = provider_status or 500
            attrs["status_code"] = effective_status
            bounded_error = {
                "error": error_type or "MistralAIError",
                "message": (
                    f"Mistral request failed with status {provider_status}"
                    if provider_status is not None
                    else "Mistral request failed"
                ),
                "status": "error",
                "status_code": effective_status,
            }
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _safe_json_str(
                bounded_error
            )
            attrs["error.message"] = bounded_error["message"]

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _active_span_processors() -> tuple[Any, Any]:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return active_span_processor, processors


def _stream_status(exception: BaseException | None) -> trace.Status:
    if exception is None or isinstance(exception, GeneratorExit):
        return trace.Status(status_code=trace.StatusCode.OK)
    return trace.Status(
        status_code=trace.StatusCode.ERROR,
        description=f"{type(exception).__name__}: Mistral stream interrupted",
    )


def _finish_guarded_stream(stream: Any, exception: BaseException | None = None) -> None:
    if getattr(stream, "_self_is_finished", False):
        return
    if exception is not None and not isinstance(exception, GeneratorExit):
        with_span = getattr(stream, "_self_with_span", None)
        if with_span is not None:
            with_span.record_exception(exception)
    try:
        stream._finish_tracing(status=_stream_status(exception))
    except Exception:
        logger.exception("Failed to finalize guarded Mistral stream")


async def _close_async_stream_source(
    source: Any,
    exception: BaseException | None,
) -> None:
    try:
        exit_method = getattr(source, "__aexit__", None)
        if callable(exit_method):
            await exit_method(
                type(exception) if exception is not None else None,
                exception,
                exception.__traceback__ if exception is not None else None,
            )
            return
        close_method = getattr(source, "aclose", None)
        if callable(close_method):
            await close_method()
    except BaseException:
        logger.debug(
            "Failed to close guarded Mistral async stream",
            exc_info=True,
        )


def _install_stream_guards() -> tuple[Any, type, type]:
    chat_wrapper = importlib.import_module(
        f"{OPENINFERENCE_MISTRALAI_MODULE}._chat_wrapper"
    )
    original_sync_stream = chat_wrapper._Stream
    original_async_stream = chat_wrapper._AsyncStream

    class _RespanMistralSyncStream(original_sync_stream):
        def __next__(self) -> Any:
            try:
                return super().__next__()
            except GeneratorExit as exception:
                _finish_guarded_stream(self, exception)
                raise
            except BaseException as exception:
                _finish_guarded_stream(self, exception)
                raise

        def __enter__(self) -> Any:
            enter_method = getattr(self.__wrapped__, "__enter__", None)
            if callable(enter_method):
                enter_method()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> Any:
            raised: BaseException | None = None
            try:
                exit_method = getattr(self.__wrapped__, "__exit__", None)
                if callable(exit_method):
                    return exit_method(exc_type, exc_value, traceback)
                return None
            except BaseException as exception:
                raised = exception
                raise
            finally:
                _finish_guarded_stream(self, raised or exc_value)

        def close(self) -> None:
            raised: BaseException | None = None
            try:
                exit_method = getattr(self.__wrapped__, "__exit__", None)
                if callable(exit_method):
                    exit_method(None, None, None)
                    return
                close_method = getattr(self.__wrapped__, "close", None)
                if callable(close_method):
                    close_method()
            except BaseException as exception:
                raised = exception
                raise
            finally:
                _finish_guarded_stream(self, raised)

        def __del__(self) -> None:
            if getattr(self, "_self_is_finished", True):
                return
            try:
                self.close()
            except BaseException:
                logger.debug(
                    "Failed to close abandoned Mistral stream",
                    exc_info=True,
                )

        def throw(self, *args: Any) -> Any:
            target = self.__wrapped__
            throw_method = getattr(target, "throw", None)
            if not callable(throw_method):
                throw_method = getattr(
                    getattr(target, "generator", None), "throw", None
                )
            if not callable(throw_method):
                raise TypeError("Mistral stream does not support throw")
            try:
                result = throw_method(*args)
            except GeneratorExit as exception:
                _finish_guarded_stream(self, exception)
                raise
            except BaseException as exception:
                _finish_guarded_stream(self, exception)
                raise
            self._process_chunk(result)
            return result

    class _RespanMistralAsyncStream(original_async_stream):
        async def stream_async_with_accumulator(self) -> Any:
            async def generator() -> Any:
                source = None
                interruption: BaseException | None = None
                try:
                    source = await self.stream
                    async for event in source:
                        self._process_chunk(event)
                        choices = getattr(getattr(event, "data", None), "choices", ())
                        if choices and choices[0].finish_reason is not None:
                            self._finish_tracing()
                        yield event
                except GeneratorExit as exception:
                    interruption = exception
                    raise
                except BaseException as exception:
                    interruption = exception
                    _finish_guarded_stream(self, exception)
                    raise
                finally:
                    _finish_guarded_stream(self, interruption)
                    if source is not None:
                        await _close_async_stream_source(source, interruption)

            return generator()

    chat_wrapper._Stream = _RespanMistralSyncStream
    chat_wrapper._AsyncStream = _RespanMistralAsyncStream
    return chat_wrapper, original_sync_stream, original_async_stream


def _restore_stream_guards(patch: tuple[Any, type, type] | None) -> None:
    if patch is None:
        return
    chat_wrapper, original_sync_stream, original_async_stream = patch
    chat_wrapper._Stream = original_sync_stream
    chat_wrapper._AsyncStream = original_async_stream


def _instrumentor_kwargs_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any] | None,
) -> bool:
    if right is None or left.keys() != right.keys():
        return False
    for key, left_value in left.items():
        right_value = right[key]
        if left_value is right_value:
            continue
        try:
            matches = left_value == right_value
        except (TypeError, ValueError):
            return False
        if not isinstance(matches, bool) or not matches:
            return False
    return True


class MistralAIInstrumentor:
    """Respan instrumentor for the official Mistral AI Python SDK."""

    name = MISTRALAI_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._delegate = None
        self._cleanup_processor = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Mistral AI via OpenInference and Respan's translator."""
        global _SHARED_CLEANUP_PROCESSOR
        global _SHARED_DELEGATE
        global _SHARED_INSTRUMENTOR_KWARGS
        global _SHARED_REFCOUNT
        global _SHARED_STREAM_GUARD_PATCH

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Mistral AI instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            mistralai_instrumentor_class = _load_openinference_mistralai_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Mistral AI instrumentation - missing dependency: %s",
                exc,
            )
            return

        with _LIFECYCLE_LOCK:
            if self._is_instrumented:
                return
            if _SHARED_REFCOUNT:
                if not _instrumentor_kwargs_match(
                    self._instrumentor_kwargs,
                    _SHARED_INSTRUMENTOR_KWARGS,
                ):
                    logger.warning(
                        "Mistral AI instrumentation activation rejected because "
                        "an active instance uses different configuration"
                    )
                    return
                self._delegate = _SHARED_DELEGATE
                self._cleanup_processor = _SHARED_CLEANUP_PROCESSOR
                self._is_instrumented = True
                _SHARED_REFCOUNT += 1
                logger.info(
                    "Mistral AI instrumentation activation shared (%d owners)",
                    _SHARED_REFCOUNT,
                )
                return

            delegate = None
            cleanup_processor = None
            stream_guard_patch = None
            try:
                delegate = OpenInferenceInstrumentor(
                    mistralai_instrumentor_class,
                    **self._instrumentor_kwargs,
                )
                delegate.activate()
                stream_guard_patch = _install_stream_guards()
                cleanup_processor = self._register_cleanup_processor()
                if cleanup_processor is None:
                    raise RuntimeError(
                        "Mistral AI cleanup processor could not be registered"
                    )
                _SHARED_DELEGATE = delegate
                _SHARED_CLEANUP_PROCESSOR = cleanup_processor
                _SHARED_STREAM_GUARD_PATCH = stream_guard_patch
                _SHARED_INSTRUMENTOR_KWARGS = dict(self._instrumentor_kwargs)
                _SHARED_REFCOUNT = 1
                self._delegate = delegate
                self._cleanup_processor = cleanup_processor
                self._is_instrumented = True
                logger.info("Mistral AI instrumentation activated")
            except Exception:
                if cleanup_processor is not None:
                    self._unregister_cleanup_processor(cleanup_processor)
                _restore_stream_guards(stream_guard_patch)
                if delegate is not None:
                    try:
                        delegate.deactivate()
                    except Exception:
                        logger.exception(
                            "Failed to clean up Mistral AI instrumentation"
                        )
                self._delegate = None
                self._cleanup_processor = None
                self._is_instrumented = False
                logger.exception("Failed to activate Mistral AI instrumentation")

    def _register_cleanup_processor(
        self,
    ) -> _MistralAIOffContractAliasProcessor | None:
        translator_getter = getattr(OpenInferenceInstrumentor, "_get_translator", None)
        if translator_getter is None:
            return None

        translator = translator_getter()
        active_span_processor, processors = _active_span_processors()
        if active_span_processor is None or processors is None:
            return None

        cleanup_processor = _MistralAIOffContractAliasProcessor()
        rebuilt_processors = []
        inserted = False

        for processor in processors:
            if isinstance(processor, _MistralAIOffContractAliasProcessor):
                continue
            rebuilt_processors.append(processor)
            if processor is translator:
                rebuilt_processors.append(cleanup_processor)
                inserted = True

        if inserted:
            active_span_processor._span_processors = tuple(rebuilt_processors)
            return cleanup_processor
        return None

    @staticmethod
    def _unregister_cleanup_processor(
        cleanup_processor: _MistralAIOffContractAliasProcessor | None,
    ) -> None:
        if cleanup_processor is None:
            return

        active_span_processor, processors = _active_span_processors()
        if active_span_processor is not None and processors is not None:
            active_span_processor._span_processors = tuple(
                processor
                for processor in processors
                if processor is not cleanup_processor
            )

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        global _SHARED_CLEANUP_PROCESSOR
        global _SHARED_DELEGATE
        global _SHARED_INSTRUMENTOR_KWARGS
        global _SHARED_REFCOUNT
        global _SHARED_STREAM_GUARD_PATCH

        with _LIFECYCLE_LOCK:
            if not self._is_instrumented:
                return

            self._is_instrumented = False
            _SHARED_REFCOUNT = max(0, _SHARED_REFCOUNT - 1)
            if _SHARED_REFCOUNT:
                self._delegate = None
                self._cleanup_processor = None
                logger.info(
                    "Mistral AI instrumentation remains active (%d owners)",
                    _SHARED_REFCOUNT,
                )
                return

            cleanup_processor = _SHARED_CLEANUP_PROCESSOR
            delegate = _SHARED_DELEGATE
            self._unregister_cleanup_processor(cleanup_processor)
            _restore_stream_guards(_SHARED_STREAM_GUARD_PATCH)
            if delegate is not None:
                try:
                    delegate.deactivate()
                except Exception:
                    logger.exception("Failed to deactivate Mistral AI instrumentation")

            _SHARED_CLEANUP_PROCESSOR = None
            _SHARED_DELEGATE = None
            _SHARED_STREAM_GUARD_PATCH = None
            _SHARED_INSTRUMENTOR_KWARGS = None
            self._delegate = None
            self._cleanup_processor = None
            logger.info("Mistral AI instrumentation deactivated")
