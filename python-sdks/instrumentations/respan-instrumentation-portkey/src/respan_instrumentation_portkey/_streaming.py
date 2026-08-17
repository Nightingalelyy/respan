"""Completion stream support missing from the upstream Portkey instrumentor."""

from __future__ import annotations

import inspect
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR

from respan_instrumentation_portkey._constants import OPENINFERENCE_PORTKEY_MODULE
from respan_instrumentation_portkey._serialization import (
    exception_message,
    exception_status,
    json_dumps,
    jsonable,
    safe_text,
)

_REQUEST_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "respan_portkey_request", default=None
)


def current_request() -> dict[str, Any] | None:
    """Return the request active while an upstream Portkey span is ending."""

    request = _REQUEST_CONTEXT.get()
    return dict(request) if request is not None else None


def _attach_request(kwargs: dict[str, Any]) -> Token[dict[str, Any] | None]:
    return _REQUEST_CONTEXT.set(dict(kwargs))


def _getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - provider-owned descriptors are untrusted
        return default


def _bound(method: Any, instance: Any) -> Any:
    descriptor = _getattr(method, "__get__")
    return descriptor(instance, type(instance)) if callable(descriptor) else method


def _request_attrs(kwargs: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "openinference.span.kind": "LLM",
        "input.value": json_dumps(kwargs),
        "input.mime_type": "application/json",
        "llm.invocation_parameters": json_dumps(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"messages", "tools"}
            }
        ),
    }
    model = kwargs.get("model")
    if isinstance(model, str):
        attrs["llm.model_name"] = safe_text(model)
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages[:50]):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str):
                attrs[f"llm.input_messages.{index}.message.role"] = safe_text(role)
            if isinstance(content, str):
                attrs[f"llm.input_messages.{index}.message.content"] = safe_text(
                    content
                )
    tools = kwargs.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools[:50]):
            attrs[f"llm.tools.{index}.tool.json_schema"] = json_dumps(tool)
    return attrs


class StreamAccumulator:
    def __init__(self) -> None:
        self.content = ""
        self.model: str | None = None
        self.response_id: str | None = None
        self.usage: Any = None
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def add(self, item: Any) -> None:
        model = _getattr(item, "model")
        if isinstance(model, str):
            self.model = safe_text(model)
        response_id = _getattr(item, "id")
        if isinstance(response_id, str):
            self.response_id = safe_text(response_id)
        usage = _getattr(item, "usage")
        if usage is not None:
            self.usage = usage
        choices = _getattr(item, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            return
        delta = _getattr(choices[0], "delta")
        content = _getattr(delta, "content")
        if isinstance(content, str):
            self.content = safe_text(self.content + content)
        calls = _getattr(delta, "tool_calls")
        if not isinstance(calls, (list, tuple)):
            return
        for call in calls[:50]:
            index = _getattr(call, "index", 0)
            if not isinstance(index, int) or index < 0:
                index = 0
            slot = self.tool_calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            call_id = _getattr(call, "id")
            if isinstance(call_id, str):
                slot["id"] = safe_text(call_id)
            function = _getattr(call, "function")
            name = _getattr(function, "name")
            arguments = _getattr(function, "arguments")
            if isinstance(name, str):
                slot["function"]["name"] = safe_text(name)
            if isinstance(arguments, str):
                slot["function"]["arguments"] = safe_text(
                    slot["function"]["arguments"] + arguments
                )

    def response(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                self.tool_calls[index] for index in sorted(self.tool_calls)
            ]
        return {
            "id": self.response_id,
            "model": self.model,
            "choices": [{"index": 0, "message": message}],
            "usage": jsonable(self.usage),
        }

    def finish(self, span: Any, error: BaseException | None = None) -> None:
        if not span.is_recording():
            return
        response = self.response()
        if self.model:
            span.set_attribute("llm.model_name", self.model)
        span.set_attribute("output.value", json_dumps(response))
        span.set_attribute("output.mime_type", "application/json")
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        if self.content:
            span.set_attribute("llm.output_messages.0.message.content", self.content)
        if self.tool_calls:
            for index, call in enumerate(
                response["choices"][0]["message"]["tool_calls"]
            ):
                prefix = f"llm.output_messages.0.message.tool_calls.{index}.tool_call"
                if call.get("id"):
                    span.set_attribute(f"{prefix}.id", call["id"])
                span.set_attribute(f"{prefix}.function.name", call["function"]["name"])
                span.set_attribute(
                    f"{prefix}.function.arguments", call["function"]["arguments"]
                )
        usage = jsonable(self.usage)
        if isinstance(usage, dict):
            for source, target in (
                ("prompt_tokens", "llm.token_count.prompt"),
                ("completion_tokens", "llm.token_count.completion"),
                ("total_tokens", "llm.token_count.total"),
            ):
                value = usage.get(source)
                if isinstance(value, int):
                    span.set_attribute(target, value)
        if error is not None:
            message = exception_message(error)
            code = exception_status(error)
            span.set_attribute(ERROR_MESSAGE_ATTR, message)
            span.set_attribute("http.response.status_code", code)
            span.set_attribute("status_code", code)
            span.set_status(Status(StatusCode.ERROR, message))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()


class SyncStreamProxy:
    def __init__(self, stream: Any, span: Any) -> None:
        self._stream = stream
        self._span = span
        self._accumulator = StreamAccumulator()
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._stream)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._accumulator.add(item)
        return item

    def _finish(self, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        self._accumulator.finish(self._span, error)

    def close(self) -> None:
        try:
            close = _getattr(self._stream, "close")
            if callable(close):
                close()
        finally:
            self._finish()

    def __enter__(self):
        enter = _getattr(self._stream, "__enter__")
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            exit_method = _getattr(self._stream, "__exit__")
            if callable(exit_method):
                return exit_method(exc_type, exc, traceback)
            return False
        finally:
            self._finish(exc)


class AsyncStreamProxy:
    def __init__(self, stream: Any, span: Any) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._span = span
        self._accumulator = StreamAccumulator()
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._accumulator.add(item)
        return item

    def _finish(self, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        self._accumulator.finish(self._span, error)

    async def aclose(self) -> None:
        try:
            close = _getattr(self._stream, "aclose") or _getattr(self._stream, "close")
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            self._finish()

    async def __aenter__(self):
        enter = _getattr(self._stream, "__aenter__")
        if callable(enter):
            result = enter()
            if inspect.isawaitable(result):
                await result
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            exit_method = _getattr(self._stream, "__aexit__")
            if callable(exit_method):
                result = exit_method(exc_type, exc, traceback)
                return await result if inspect.isawaitable(result) else result
            return False
        finally:
            self._finish(exc)


@dataclass
class StreamHooks:
    sync_class: type
    async_class: type
    sync_original: Any
    async_original: Any
    sync_wrapper: Any
    async_wrapper: Any


def install_stream_hooks(provider: Any) -> StreamHooks:
    module = __import__("portkey_ai.api_resources.apis.chat_complete", fromlist=["x"])
    sync_class = module.Completions
    async_class = module.AsyncCompletions
    sync_original = sync_class.create
    async_original = async_class.create
    tracer = provider.get_tracer(OPENINFERENCE_PORTKEY_MODULE, "0.1.0")

    def sync_wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is not True:
            request_token = _attach_request(kwargs)
            try:
                return _bound(sync_original, instance)(*args, **kwargs)
            finally:
                _REQUEST_CONTEXT.reset(request_token)
        span = tracer.start_span("Completions", attributes=_request_attrs(dict(kwargs)))
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            stream = _bound(sync_original, instance)(*args, **kwargs)
        except BaseException as exc:
            StreamAccumulator().finish(span, exc)
            raise
        finally:
            context_api.detach(token)
        return SyncStreamProxy(stream, span)

    async def async_wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is not True:
            request_token = _attach_request(kwargs)
            try:
                return await _bound(async_original, instance)(*args, **kwargs)
            finally:
                _REQUEST_CONTEXT.reset(request_token)
        span = tracer.start_span(
            "AsyncCompletions", attributes=_request_attrs(dict(kwargs))
        )
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            stream = await _bound(async_original, instance)(*args, **kwargs)
        except BaseException as exc:
            StreamAccumulator().finish(span, exc)
            raise
        finally:
            context_api.detach(token)
        return AsyncStreamProxy(stream, span)

    sync_class.create = sync_wrapper
    async_class.create = async_wrapper
    return StreamHooks(
        sync_class,
        async_class,
        sync_original,
        async_original,
        sync_wrapper,
        async_wrapper,
    )


def remove_stream_hooks(hooks: StreamHooks | None) -> None:
    if hooks is None:
        return
    if hooks.sync_class.create is hooks.sync_wrapper:
        hooks.sync_class.create = hooks.sync_original
    if hooks.async_class.create is hooks.async_wrapper:
        hooks.async_class.create = hooks.async_original
