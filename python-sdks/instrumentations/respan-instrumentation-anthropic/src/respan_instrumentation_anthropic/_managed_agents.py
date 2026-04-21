"""Anthropic managed-agent stream instrumentation helpers."""

from __future__ import annotations

import time
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_anthropic._constants import (
    AGENT_MCP_TOOL_USE_EVENT,
    AGENT_MESSAGE_EVENT,
    AGENT_TOOL_USE_EVENTS,
    ARGUMENTS_KEY,
    ANTHROPIC_MANAGED_AGENT_SPAN_NAME,
    CLOSE_METHOD_NAME,
    CONTENT_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    INPUT_KEY,
    INPUT_TOKENS_KEY,
    MANAGED_AGENT_SESSION_ID_ATTR,
    MANAGED_AGENT_STOP_REASON_ATTR,
    MCP_SERVER_KEY,
    MCP_SERVER_NAME_KEY,
    MODEL_REQUEST_END_EVENT,
    NAME_KEY,
    MODEL_USAGE_KEY,
    OUTPUT_TOKENS_KEY,
    ROLE_KEY,
    SESSION_ERROR_EVENT,
    SESSION_STATUS_IDLE_EVENT,
    SESSION_STATUS_RUNNING_EVENT,
    STOP_REASON_KEY,
    TOOL_CALLS_OVERRIDE,
    TYPE_KEY,
    USER_ROLE,
    USER_MESSAGE_EVENT,
    CACHE_CREATION_INPUT_TOKENS_KEY,
    CACHE_READ_INPUT_TOKENS_KEY,
)
from respan_instrumentation_anthropic._messages import (
    _build_base_chat_attrs,
    _emit_span,
    _normalize_content_block,
    _safe_json,
)
from respan_sdk.constants.span_attributes import (
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_SPAN_TOOL_CALLS,
)


class _ManagedAgentTurnTracker:
    """Accumulate streamed managed-agent events for one agent turn."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._reset()

    def process(self, event: Any) -> None:
        event_type = getattr(event, TYPE_KEY, None)

        if event_type == USER_MESSAGE_EVENT:
            for block in getattr(event, CONTENT_KEY, []):
                normalized_text = _normalize_content_block(block=block)
                if normalized_text:
                    self.input_messages.append(normalized_text)
            return

        if event_type == AGENT_MESSAGE_EVENT:
            for block in getattr(event, CONTENT_KEY, []):
                normalized_text = _normalize_content_block(block=block)
                if normalized_text:
                    self.output_messages.append(normalized_text)
            return

        if event_type in AGENT_TOOL_USE_EVENTS:
            tool_call = {
                ID_KEY: getattr(event, ID_KEY, ""),
                TYPE_KEY: FUNCTION_TOOL_TYPE,
                FUNCTION_KEY: {
                    NAME_KEY: getattr(event, NAME_KEY, ""),
                    ARGUMENTS_KEY: _safe_json(
                        value=getattr(event, INPUT_KEY, {}),
                    ),
                },
            }
            if event_type == AGENT_MCP_TOOL_USE_EVENT:
                tool_call[MCP_SERVER_KEY] = getattr(event, MCP_SERVER_NAME_KEY, "")
            self.tool_calls.append(tool_call)
            return

        if event_type == MODEL_REQUEST_END_EVENT:
            model_usage = getattr(event, MODEL_USAGE_KEY, None)
            if model_usage:
                self.total_input_tokens += getattr(model_usage, INPUT_TOKENS_KEY, 0)
                self.total_output_tokens += getattr(model_usage, OUTPUT_TOKENS_KEY, 0)
                self.total_cache_read += getattr(
                    model_usage,
                    CACHE_READ_INPUT_TOKENS_KEY,
                    0,
                )
                self.total_cache_creation += getattr(
                    model_usage,
                    CACHE_CREATION_INPUT_TOKENS_KEY,
                    0,
                )
            return

        if event_type == SESSION_STATUS_IDLE_EVENT:
            stop_reason = getattr(event, STOP_REASON_KEY, None)
            self._emit(
                stop_reason=getattr(stop_reason, TYPE_KEY, None)
                if stop_reason
                else None,
            )
            self._reset()
            return

        if event_type == SESSION_ERROR_EVENT:
            self.error = str(event)
            self._emit(stop_reason=None)
            self._reset()
            return

        if event_type == SESSION_STATUS_RUNNING_EVENT and not self.input_messages:
            self.start_ns = time.time_ns()

    def _emit(self, stop_reason: str | None) -> None:
        attrs = _build_base_chat_attrs(span_name=ANTHROPIC_MANAGED_AGENT_SPAN_NAME)
        attrs[MANAGED_AGENT_SESSION_ID_ATTR] = self.session_id

        if self.input_messages:
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _safe_json(
                value=[
                    {
                        ROLE_KEY: USER_ROLE,
                        CONTENT_KEY: "\n".join(self.input_messages),
                    }
                ]
            )

        if self.output_messages:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = "\n".join(
                self.output_messages
            )

        if self.total_input_tokens or self.total_output_tokens:
            attrs[LLM_USAGE_PROMPT_TOKENS] = self.total_input_tokens
            attrs[LLM_USAGE_COMPLETION_TOKENS] = self.total_output_tokens

        if self.total_cache_read:
            attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = (
                self.total_cache_read
            )
        if self.total_cache_creation:
            attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
                self.total_cache_creation
            )

        if self.tool_calls:
            attrs[RESPAN_SPAN_TOOL_CALLS] = _safe_json(value=self.tool_calls)
            attrs[TOOL_CALLS_OVERRIDE] = self.tool_calls

        if stop_reason:
            attrs[MANAGED_AGENT_STOP_REASON_ATTR] = stop_reason

        _emit_span(
            attrs=attrs,
            start_ns=self.start_ns,
            error_message=self.error,
            name=ANTHROPIC_MANAGED_AGENT_SPAN_NAME,
        )

    def _reset(self) -> None:
        self.start_ns = time.time_ns()
        self.error: str | None = None
        self.input_messages: list[str] = []
        self.output_messages: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read = 0
        self.total_cache_creation = 0


class _InstrumentedManagedStream:
    """Proxy for a sync stream that intercepts managed-agent events."""

    def __init__(self, stream: Any, session_id: str) -> None:
        self._stream = stream
        self._tracker = _ManagedAgentTurnTracker(session_id=session_id)

    def __iter__(self) -> Any:
        for event in self._stream:
            self._tracker.process(event=event)
            yield event

    def __enter__(self) -> Any:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(exc_type, exc_val, exc_tb)
        return None

    def close(self) -> None:
        if hasattr(self._stream, CLOSE_METHOD_NAME):
            getattr(self._stream, CLOSE_METHOD_NAME)()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _InstrumentedAsyncManagedStream:
    """Proxy for an async stream that intercepts managed-agent events."""

    def __init__(self, stream: Any, session_id: str) -> None:
        self._stream = stream
        self._tracker = _ManagedAgentTurnTracker(session_id=session_id)

    async def __aiter__(self) -> Any:
        async for event in self._stream:
            self._tracker.process(event=event)
            yield event

    async def __aenter__(self) -> Any:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        return None

    async def close(self) -> None:
        if hasattr(self._stream, CLOSE_METHOD_NAME):
            close_result = getattr(self._stream, CLOSE_METHOD_NAME)()
            if hasattr(close_result, "__await__"):
                await close_result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_sync_events_stream(original: Any) -> Any:
    """Wrap ``Events.stream()`` to intercept managed-agent events."""

    def wrapper(self: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        stream = original(self, session_id, *args, **kwargs)
        return _InstrumentedManagedStream(stream=stream, session_id=session_id)

    return wrapper


def _wrap_async_events_stream(original: Any) -> Any:
    """Wrap ``AsyncEvents.stream()`` to intercept managed-agent events."""

    async def wrapper(self: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        stream = await original(self, session_id, *args, **kwargs)
        return _InstrumentedAsyncManagedStream(stream=stream, session_id=session_id)

    return wrapper
