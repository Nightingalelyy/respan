"""Instrumentation for Mirascope 2.x model and toolkit execution surfaces."""

from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
import json
import logging
import threading
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_mirascope._serialization import (
    json_string,
    json_value,
    safe_exception_text,
    safe_text,
)

logger = logging.getLogger(__name__)

MIRASCOPE_INSTRUMENTATION_NAME = "mirascope"
_LOCK = threading.RLock()
_REFCOUNT = 0
_CAPTURE_CONTENT = True
_PATCHES: list[_Patch] = []

_MODERN_INPUT_USAGE = getattr(
    SpanAttributes, "LLM_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"
)
_MODERN_OUTPUT_USAGE = getattr(
    SpanAttributes, "LLM_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"
)
_CACHE_READ_USAGE = getattr(
    SpanAttributes,
    "LLM_USAGE_CACHE_READ_INPUT_TOKENS",
    "llm.usage.cache_read_input_tokens",
)
_CACHE_WRITE_USAGE = getattr(
    SpanAttributes,
    "LLM_USAGE_CACHE_WRITE_INPUT_TOKENS",
    "llm.usage.cache_write_input_tokens",
)


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _model_identity(model: Any, response: Any = None) -> tuple[str, str]:
    raw_model = safe_text(
        getattr(response, "model_id", None)
        or getattr(model, "model_id", None)
        or "unknown"
    )
    if "/" in raw_model:
        inferred_provider, model_name = raw_model.split("/", 1)
    else:
        inferred_provider, model_name = "unknown", raw_model
    provider = safe_text(getattr(response, "provider_id", None) or inferred_provider)
    return provider, model_name


def _chat_span_name(model: Any) -> str:
    return safe_text(f"mirascope.{_model_identity(model)[1]}.chat")


def _message_parts(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, str):
        return [("user", value)]
    values = value if isinstance(value, (list, tuple)) else [value]
    messages: list[tuple[str, Any]] = []
    for item in values:
        if isinstance(item, str):
            messages.append(("user", item))
            continue
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        if isinstance(item, dict):
            role = item.get("role", role)
            content = item.get("content", content)
        messages.append(
            (safe_text(role or "user"), content if content is not None else item)
        )
    return messages


def _set_messages(span: Any, *, prefix: str, value: Any) -> None:
    for index, (role, content) in enumerate(_message_parts(value)):
        normalized_content = json_value(content)
        span.set_attribute(f"{prefix}.{index}.role", role)
        span.set_attribute(
            f"{prefix}.{index}.content",
            normalized_content
            if isinstance(normalized_content, str)
            else json_string(normalized_content),
        )


def _tool_definitions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    toolkit_tools = getattr(value, "tools", None)
    if toolkit_tools is not None:
        value = toolkit_tools
    tools = (
        value
        if isinstance(value, Sequence) and not isinstance(value, str | bytes)
        else [value]
    )
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": safe_text(
                getattr(tool, "name", None)
                or getattr(tool, "__name__", tool.__class__.__name__)
            ),
        }
        description = getattr(tool, "description", None) or getattr(
            tool, "__doc__", None
        )
        if description:
            function["description"] = safe_text(description)
        parameters = getattr(tool, "parameters", None) or getattr(tool, "schema", None)
        if parameters is None and callable(tool):
            properties: dict[str, Any] = {}
            required: list[str] = []
            try:
                for parameter in inspect.signature(tool).parameters.values():
                    if parameter.name in {"self", "cls"}:
                        continue
                    annotation = parameter.annotation
                    type_name = (
                        getattr(annotation, "__name__", None)
                        if annotation is not inspect.Parameter.empty
                        else None
                    )
                    json_type = {
                        "bool": "boolean",
                        "dict": "object",
                        "float": "number",
                        "int": "integer",
                        "list": "array",
                        "str": "string",
                    }.get(type_name, "string")
                    properties[parameter.name] = {"type": json_type}
                    if parameter.default is inspect.Parameter.empty:
                        required.append(parameter.name)
                parameters = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            except (TypeError, ValueError):
                parameters = {}
        if callable(getattr(parameters, "model_dump", None)):
            parameters = parameters.model_dump(by_alias=True, exclude_none=True)
        normalized_parameters = json_value(parameters or {})
        if (
            isinstance(normalized_parameters, dict)
            and "properties" in normalized_parameters
        ):
            normalized_parameters.setdefault("type", "object")
        function["parameters"] = normalized_parameters
        definitions.append(
            {
                "type": "function",
                "function": function,
            }
        )
    return definitions


def _tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return json_value(value)
    try:
        return json_value(json.loads(value))
    except (TypeError, ValueError):
        return safe_text(value)


def _tool_calls(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    calls = (
        value
        if isinstance(value, Sequence) and not isinstance(value, str | bytes)
        else [value]
    )
    normalized: list[dict[str, Any]] = []
    for call in calls:
        function = getattr(call, "function", None)
        name = safe_text(
            getattr(call, "name", None)
            or getattr(function, "name", None)
            or "mirascope.tool"
        )
        arguments = getattr(call, "args", None)
        if arguments is None:
            arguments = getattr(call, "arguments", None)
        parsed_arguments = _tool_arguments(arguments if arguments is not None else {})
        normalized_call: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    parsed_arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        call_id = getattr(call, "id", None) or getattr(call, "call_id", None)
        if call_id:
            normalized_call["id"] = safe_text(call_id)
        normalized.append(normalized_call)
    return normalized


def _prepare_chat_span(
    span: Any,
    *,
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    stream: bool = False,
) -> None:
    provider, model_name = _model_identity(model)
    entity_name = safe_text(f"mirascope.{model_name}")
    span.set_attribute(RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    span.set_attribute(RESPAN_LOG_TYPE, "chat")
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
    span.set_attribute(SpanAttributes.LLM_SYSTEM, provider)
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_name)
    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)
    if stream:
        span.set_attribute(SpanAttributes.LLM_IS_STREAMING, True)
    content = args[1] if len(args) > 1 else kwargs.get("content")
    if _CAPTURE_CONTENT:
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json_string({"content": content}),
        )
        _set_messages(span, prefix=SpanAttributes.LLM_PROMPTS, value=content)
        definitions = _tool_definitions(kwargs.get("tools"))
        if definitions:
            span.set_attribute(
                SpanAttributes.LLM_REQUEST_FUNCTIONS, json_string(definitions)
            )


def _set_usage(span: Any, usage: Any) -> None:
    if usage is None:
        return
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_read = getattr(usage, "cache_read_tokens", None)
    cache_write = getattr(usage, "cache_write_tokens", None)
    if input_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
        span.set_attribute(_MODERN_INPUT_USAGE, input_tokens)
    if output_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)
        span.set_attribute(_MODERN_OUTPUT_USAGE, output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            int(input_tokens) + int(output_tokens),
        )
    if cache_read is not None:
        span.set_attribute(_CACHE_READ_USAGE, cache_read)
    if cache_write is not None:
        span.set_attribute(_CACHE_WRITE_USAGE, cache_write)


def _status_code(value: Any, *, default: int) -> int:
    pending = [value]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        for name in ("status_code", "status"):
            try:
                code = getattr(candidate, name, None)
                if isinstance(code, int) and not isinstance(code, bool):
                    return code
            except Exception:  # noqa: BLE001, S112 - vendor properties may raise
                continue
        for name in (
            "response",
            "raw_response",
            "original_exception",
            "tool_exception",
            "__cause__",
        ):
            try:
                nested = getattr(candidate, name, None)
            except Exception:  # noqa: BLE001 - vendor exception properties may raise
                nested = None
            if nested is not None:
                pending.append(nested)
    return default


def _finish_chat_span(span: Any, *, model: Any, response: Any) -> None:
    span.set_attribute("status_code", _status_code(response, default=200))
    _, model_name = _model_identity(model, response)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model_name)
    _set_usage(span, getattr(response, "usage", None))
    if not _CAPTURE_CONTENT or response is None:
        return
    content = getattr(response, "content", None)
    text_value = getattr(response, "text", None)
    if callable(text_value):
        try:
            text_value = text_value()
        except Exception:  # noqa: BLE001 - response helpers are vendor-controlled
            text_value = None
    output = text_value if text_value is not None else content
    tool_calls = _tool_calls(getattr(response, "tool_calls", None))
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        json_string({"content": output, "tool_calls": tool_calls}),
    )
    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
    if output is not None:
        normalized_output = json_value(output)
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            normalized_output
            if isinstance(normalized_output, str)
            else json_string(normalized_output),
        )
    if tool_calls:
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls",
            json_string(tool_calls),
        )


def _record_error(span: Any, exc: BaseException) -> None:
    status_code = _status_code(exc, default=500)
    if status_code < 400:
        status_code = 500
    message = safe_exception_text(exc)
    span.set_attribute("status_code", status_code)
    span.set_attribute(ERROR_MESSAGE_ATTR, message)
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
        attributes={
            OTelSpanAttributes.EXCEPTION_TYPE: type(exc).__name__,
            OTelSpanAttributes.EXCEPTION_MESSAGE: message,
            OTelSpanAttributes.EXCEPTION_ESCAPED: True,
        },
    )
    span.set_status(Status(StatusCode.ERROR, message))


class _StreamSpanState:
    def __init__(self, span: Any, model: Any) -> None:
        self.span = span
        self.model = model
        self._lock = threading.Lock()
        self._ended = False
        self.finalizer: weakref.finalize | None = None

    def finish(self, response: Any = None) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
            if response is not None:
                _finish_chat_span(self.span, model=self.model, response=response)
            self.span.end()
            if self.finalizer is not None and self.finalizer.alive:
                self.finalizer.detach()

    def fail(self, exc: BaseException) -> None:
        _record_error(self.span, exc)
        self.finish()


def _wrap_sync_iterator(source: Any, response: Any, state: _StreamSpanState) -> Any:
    def iterator() -> Any:
        try:
            while True:
                try:
                    with trace.use_span(
                        state.span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        chunk = next(source)
                except StopIteration:
                    break
                yield chunk
        except GeneratorExit:
            close = getattr(source, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    state.fail(exc)
                    raise
            state.finish(response)
            raise
        except BaseException as exc:
            state.fail(exc)
            raise
        else:
            state.finish(response)

    return iterator()


def _wrap_async_iterator(source: Any, response: Any, state: _StreamSpanState) -> Any:
    async def iterator() -> Any:
        try:
            while True:
                try:
                    with trace.use_span(
                        state.span,
                        end_on_exit=False,
                        record_exception=False,
                        set_status_on_exception=False,
                    ):
                        chunk = await source.__anext__()
                except StopAsyncIteration:
                    break
                yield chunk
        except GeneratorExit:
            aclose = getattr(source, "aclose", None)
            close = getattr(source, "close", None)
            try:
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    close()
            except BaseException as exc:
                state.fail(exc)
                raise
            state.finish(response)
            raise
        except BaseException as exc:
            state.fail(exc)
            raise
        else:
            state.finish(response)

    return iterator()


def _finalize_abandoned_sync_stream(source: Any, state: _StreamSpanState) -> None:
    close = getattr(source, "close", None)
    try:
        if callable(close):
            close()
    except Exception as exc:  # noqa: BLE001 - provider close hooks are untrusted
        state.fail(exc)
    else:
        state.finish()


def _finalize_abandoned_async_stream(source: Any, state: _StreamSpanState) -> None:
    async def close_source() -> None:
        aclose = getattr(source, "aclose", None)
        close = getattr(source, "close", None)
        try:
            if callable(aclose):
                result = aclose()
                if inspect.isawaitable(result):
                    await result
            elif callable(close):
                close()
        except Exception as exc:  # noqa: BLE001 - provider close hooks are untrusted
            state.fail(exc)
        else:
            state.finish()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(close_source())
        except Exception as exc:  # noqa: BLE001 - provider close hooks are untrusted
            state.fail(exc)
    else:
        loop.create_task(close_source())


def _call_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            _chat_span_name(model)
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs)
        try:
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                response = original(*args, **kwargs)
            _finish_chat_span(span, model=model, response=response)
            return response
        except BaseException as exc:
            _record_error(span, exc)
            raise
        finally:
            span.end()

    return wrapper


def _async_call_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            _chat_span_name(model)
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs)
        try:
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                response = await original(*args, **kwargs)
            _finish_chat_span(span, model=model, response=response)
            return response
        except BaseException as exc:
            _record_error(span, exc)
            raise
        finally:
            span.end()

    return wrapper


def _stream_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            _chat_span_name(model)
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs, stream=True)
        try:
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                response = original(*args, **kwargs)
        except BaseException as exc:
            _record_error(span, exc)
            span.end()
            raise
        state = _StreamSpanState(span, model)
        source = getattr(response, "_chunk_iterator", None)
        if source is None:
            state.finish(response)
            return response
        response._chunk_iterator = _wrap_sync_iterator(source, response, state)
        try:
            state.finalizer = weakref.finalize(
                response,
                _finalize_abandoned_sync_stream,
                source,
                state,
            )
        except TypeError:
            pass
        return response

    return wrapper


def _async_stream_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0]
        span = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME).start_span(
            _chat_span_name(model)
        )
        _prepare_chat_span(span, model=model, args=args, kwargs=kwargs, stream=True)
        try:
            with trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                response = await original(*args, **kwargs)
        except BaseException as exc:
            _record_error(span, exc)
            span.end()
            raise
        state = _StreamSpanState(span, model)
        source = getattr(response, "_chunk_iterator", None)
        if source is None:
            state.finish(response)
            return response
        response._chunk_iterator = _wrap_async_iterator(source, response, state)
        try:
            state.finalizer = weakref.finalize(
                response,
                _finalize_abandoned_async_stream,
                source,
                state,
            )
        except TypeError:
            pass
        return response

    return wrapper


def _tool_call(args: tuple[Any, ...]) -> Any:
    return args[-1] if len(args) > 1 else None


def _tool_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    return safe_text(
        getattr(tool_call, "name", None)
        or getattr(function, "name", None)
        or "mirascope.tool"
    )


def _prepare_tool_span(span: Any, tool_call: Any) -> None:
    name = _tool_name(tool_call)
    span.set_attribute(RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    span.set_attribute(RESPAN_LOG_TYPE, "tool")
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, name)
    if _CAPTURE_CONTENT:
        value = getattr(tool_call, "args", None)
        if value is None:
            value = getattr(tool_call, "arguments", tool_call)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json_string({"name": name, "arguments": _tool_arguments(value)}),
        )


def _finish_tool_span(span: Any, result: Any) -> None:
    error = getattr(result, "error", None)
    output = getattr(result, "result", result)
    if error is not None:
        status_code = _status_code(error, default=500)
        if status_code < 400:
            status_code = 500
        message = (
            safe_exception_text(error)
            if isinstance(error, BaseException)
            else safe_text(error)
        )
        span.set_attribute("status_code", status_code)
        span.set_attribute(ERROR_MESSAGE_ATTR, message)
        if isinstance(error, BaseException):
            span.add_event(
                "exception",
                attributes={
                    OTelSpanAttributes.EXCEPTION_TYPE: type(error).__name__,
                    OTelSpanAttributes.EXCEPTION_MESSAGE: message,
                    OTelSpanAttributes.EXCEPTION_ESCAPED: False,
                },
            )
        span.set_status(Status(StatusCode.ERROR, message))
    else:
        span.set_attribute("status_code", _status_code(result, default=200))
    if _CAPTURE_CONTENT:
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(output))


def _tool_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_call = kwargs.get("tool_call") or _tool_call(args)
        name = _tool_name(tool_call)
        tracer = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME)
        with tracer.start_as_current_span(
            safe_text(f"{name}.tool"),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            _prepare_tool_span(span, tool_call)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                _record_error(span, exc)
                raise
            _finish_tool_span(span, result)
            return result

    return wrapper


def _async_tool_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_call = kwargs.get("tool_call") or _tool_call(args)
        name = _tool_name(tool_call)
        tracer = trace.get_tracer(MIRASCOPE_INSTRUMENTATION_NAME)
        with tracer.start_as_current_span(
            safe_text(f"{name}.tool"),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            _prepare_tool_span(span, tool_call)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                _record_error(span, exc)
                raise
            _finish_tool_span(span, result)
            return result

    return wrapper


def _patch(owner: Any, name: str, factory: Callable[[Any], Any]) -> None:
    original = getattr(owner, name, None)
    if original is None or getattr(original, "__respan_mirascope_wrapper__", False):
        return
    replacement = factory(original)
    replacement.__respan_mirascope_wrapper__ = True
    setattr(owner, name, replacement)
    _PATCHES.append(_Patch(owner, name, original, replacement))


def _install_patches() -> None:
    first_new_patch = len(_PATCHES)
    try:
        models = importlib.import_module("mirascope.llm.models.models")
        model = models.Model
        for name in ("call", "context_call"):
            _patch(model, name, _call_wrapper)
        for name in ("call_async", "context_call_async"):
            _patch(model, name, _async_call_wrapper)
        for name in ("stream", "context_stream"):
            _patch(model, name, _stream_wrapper)
        for name in ("stream_async", "context_stream_async"):
            _patch(model, name, _async_stream_wrapper)

        toolkit_module = importlib.import_module("mirascope.llm.tools.toolkit")
        for class_name in ("Toolkit", "ContextToolkit"):
            owner = getattr(toolkit_module, class_name, None)
            if owner is not None:
                _patch(owner, "execute", _tool_wrapper)
        for class_name in ("AsyncToolkit", "AsyncContextToolkit"):
            owner = getattr(toolkit_module, class_name, None)
            if owner is not None:
                _patch(owner, "execute", _async_tool_wrapper)
    except Exception:
        for patch in reversed(_PATCHES[first_new_patch:]):
            if getattr(patch.owner, patch.name, None) is patch.replacement:
                setattr(patch.owner, patch.name, patch.original)
        del _PATCHES[first_new_patch:]
        raise


def _remove_patches() -> None:
    for patch in reversed(_PATCHES):
        if getattr(patch.owner, patch.name, None) is patch.replacement:
            setattr(patch.owner, patch.name, patch.original)
    _PATCHES.clear()


class MirascopeInstrumentor:
    """Instrument Mirascope Model and Toolkit execution surfaces."""

    name = MIRASCOPE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        global _CAPTURE_CONTENT, _REFCOUNT

        if not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("mirascope")
        except ImportError as exc:
            logger.warning("Mirascope instrumentation unavailable: %s", exc)
            return
        with _LOCK:
            if self._is_instrumented:
                return
            if _REFCOUNT == 0:
                previous_capture_content = _CAPTURE_CONTENT
                _CAPTURE_CONTENT = self._capture_content
                try:
                    _install_patches()
                except Exception as exc:  # noqa: BLE001 - activation is best-effort
                    _CAPTURE_CONTENT = previous_capture_content
                    logger.warning("Mirascope instrumentation unavailable: %s", exc)
                    return
            elif _CAPTURE_CONTENT != self._capture_content:
                logger.warning(
                    "Mirascope is already instrumented; the first capture_content setting wins"
                )
            _REFCOUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        global _REFCOUNT

        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT == 0:
                _remove_patches()
