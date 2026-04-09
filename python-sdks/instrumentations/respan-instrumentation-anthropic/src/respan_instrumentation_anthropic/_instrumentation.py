"""Anthropic SDK instrumentation plugin for Respan.

Monkey-patches ``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` to
trace ``messages.create()`` and ``messages.stream()`` calls.  Also patches
``beta.sessions.events.stream()`` to trace Managed Agents sessions — one
span per agent turn with accumulated token usage, tool calls, and messages.

Spans carry GenAI semantic conventions and pass ``is_processable_span()``
natively.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _serialize_content_value(value: Any) -> Any:
    """Serialize Anthropic content while preserving structured blocks."""
    if isinstance(value, list):
        parts = [_serialize_content_block(block) for block in value]
        parts = [part for part in parts if part not in ("", None, [], {})]

        if not parts:
            return ""
        if all(isinstance(part, str) for part in parts):
            return "\n".join(parts)
        return parts

    if isinstance(value, dict):
        if "type" in value:
            return _serialize_content_block(value)
        return {key: _serialize_content_value(val) for key, val in value.items()}

    return value


def _serialize_content_block(block: Any) -> Any:
    """Serialize a single Anthropic content block."""
    if isinstance(block, str):
        return block

    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "text":
            return block.get("text", "")

        result: Dict[str, Any] = {}
        for key in (
            "type",
            "id",
            "name",
            "input",
            "tool_use_id",
            "text",
            "content",
            "source",
            "citations",
            "cache_control",
        ):
            if key in block:
                result[key] = _serialize_content_value(block[key])
        return result or str(block)

    block_type = getattr(block, "type", None)
    if block_type == "text":
        return getattr(block, "text", "")

    if block_type:
        result: Dict[str, Any] = {"type": block_type}
        for attr in (
            "id",
            "name",
            "input",
            "tool_use_id",
            "text",
            "content",
            "source",
            "citations",
            "cache_control",
        ):
            if hasattr(block, attr):
                result[attr] = _serialize_content_value(getattr(block, attr))
        return result

    if hasattr(block, "text"):
        return block.text

    return str(block)


def _normalize_content_block(block: Any) -> str:
    """Normalize a content block into human-readable text for turn tracking."""
    value = _serialize_content_block(block)
    if value in ("", None, [], {}):
        return ""
    if isinstance(value, str):
        return value
    return _safe_json(value)


def _format_input_messages(
    messages: List[Any],
    system: Any = None,
) -> str:
    """Normalize Anthropic messages to [{role, content}] JSON."""
    result: List[Dict[str, Any]] = []

    # System prompt (string or list of content blocks)
    if system is not None:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        else:
            result.append({"role": "system", "content": _serialize_content_value(system)})

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            role = msg.role
            content = msg.content
        else:
            role = "user"
            content = str(msg)

        content = _serialize_content_value(content)

        result.append({"role": role, "content": content})

    return _safe_json(result)


def _format_output(message: Any) -> str:
    """Extract assistant response text from a Message object."""
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return _safe_json(_serialize_content_value(content))


def _extract_tool_calls(message: Any) -> Optional[str]:
    """Extract tool_use blocks as normalized tool_calls JSON."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return None

    tool_calls: List[Dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if isinstance(block, dict):
            block_type = block.get("type")

        if block_type != "tool_use":
            continue

        if hasattr(block, "id"):
            tc = {
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": _safe_json(block.input),
                },
            }
        elif isinstance(block, dict):
            tc = {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": _safe_json(block.get("input", {})),
                },
            }
        else:
            continue
        tool_calls.append(tc)

    return _safe_json(tool_calls) if tool_calls else None


def _format_tools(tools: Any) -> Optional[str]:
    """Normalize tool definitions to JSON."""
    if not tools:
        return None

    result: List[Dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("input_schema", {})
        elif hasattr(tool, "name"):
            name = tool.name
            desc = getattr(tool, "description", "")
            params = getattr(tool, "input_schema", {})
        else:
            continue

        entry: Dict[str, Any] = {"type": "function", "function": {"name": name}}
        if desc:
            entry["function"]["description"] = desc
        if params:
            entry["function"]["parameters"] = params
        result.append(entry)

    return _safe_json(result) if result else None


def _build_span_attrs(kwargs: Dict[str, Any], message: Any) -> Dict[str, Any]:
    """Build the full span attribute dict from request kwargs and response Message."""
    attrs: Dict[str, Any] = {
        "gen_ai.system": "anthropic",
        "llm.request.type": "chat",
        "traceloop.entity.name": "anthropic.chat",
        "traceloop.entity.path": "anthropic.chat",
        "respan.entity.log_type": "chat",
        SpanAttributes.TRACELOOP_SPAN_KIND: LLMRequestTypeValues.CHAT.value,
    }

    # Model
    model = getattr(message, "model", None) or kwargs.get("model")
    if model:
        attrs["gen_ai.request.model"] = model

    # Input messages
    messages = kwargs.get("messages", [])
    system = kwargs.get("system")
    attrs["traceloop.entity.input"] = _format_input_messages(messages, system)

    # Output
    attrs["traceloop.entity.output"] = _format_output(message)

    # Token usage
    usage = getattr(message, "usage", None)
    if usage is not None:
        attrs["gen_ai.usage.prompt_tokens"] = getattr(usage, "input_tokens", 0)
        attrs["gen_ai.usage.completion_tokens"] = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
        if cache_read:
            attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read
        if cache_creation:
            attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
                cache_creation
            )

    # Tool calls
    tool_calls_json = _extract_tool_calls(message)
    if tool_calls_json:
        attrs["respan.span.tool_calls"] = tool_calls_json

    # Tool definitions
    tools_json = _format_tools(kwargs.get("tools"))
    if tools_json:
        attrs["respan.span.tools"] = tools_json

    return attrs


def _build_error_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build minimal span attributes for a failed request."""
    attrs: Dict[str, Any] = {
        "gen_ai.system": "anthropic",
        "llm.request.type": "chat",
        "traceloop.entity.name": "anthropic.chat",
        "traceloop.entity.path": "anthropic.chat",
        "respan.entity.log_type": "chat",
        SpanAttributes.TRACELOOP_SPAN_KIND: LLMRequestTypeValues.CHAT.value,
    }
    model = kwargs.get("model")
    if model:
        attrs["gen_ai.request.model"] = model

    messages = kwargs.get("messages", [])
    system = kwargs.get("system")
    attrs["traceloop.entity.input"] = _format_input_messages(messages, system)
    return attrs


# ---------------------------------------------------------------------------
# Monkey-patch wrappers
# ---------------------------------------------------------------------------

_original_sync_create = None
_original_async_create = None
_original_sync_stream = None
_original_async_stream = None


def _current_trace_parent_ids() -> tuple[Optional[str], Optional[str]]:
    """Return the active OTEL trace/span IDs for parenting injected spans."""
    try:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0) or 0
    span_id = getattr(span_context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None

    return format(trace_id, "032x"), format(span_id, "016x")


def _emit_span(
    attrs: Dict[str, Any],
    start_ns: int,
    error_message: Optional[str] = None,
    *,
    name: str = "anthropic.chat",
) -> None:
    """Build a ReadableSpan and inject into the OTEL pipeline."""
    try:
        from respan_tracing.utils.span_factory import build_readable_span, inject_span

        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            error_message=error_message,
        )
        inject_span(span)
    except Exception:
        logger.debug("Failed to emit Anthropic span", exc_info=True)


def _wrap_sync_create(original):
    """Wrap Messages.create() for sync Anthropic client."""

    def wrapper(self, *args, **kwargs):
        start_ns = time.time_ns()
        try:
            message = original(self, *args, **kwargs)
        except Exception as exc:
            attrs = _build_error_attrs(kwargs)
            _emit_span(attrs, start_ns, error_message=str(exc))
            raise

        try:
            attrs = _build_span_attrs(kwargs, message)
            _emit_span(attrs, start_ns)
        except Exception:
            logger.debug("Failed to build Anthropic span attrs", exc_info=True)

        return message

    return wrapper


def _wrap_async_create(original):
    """Wrap AsyncMessages.create() for async Anthropic client."""

    async def wrapper(self, *args, **kwargs):
        start_ns = time.time_ns()
        try:
            message = await original(self, *args, **kwargs)
        except Exception as exc:
            attrs = _build_error_attrs(kwargs)
            _emit_span(attrs, start_ns, error_message=str(exc))
            raise

        try:
            attrs = _build_span_attrs(kwargs, message)
            _emit_span(attrs, start_ns)
        except Exception:
            logger.debug("Failed to build Anthropic span attrs", exc_info=True)

        return message

    return wrapper


def _wrap_sync_stream(original):
    """Wrap Messages.stream() for sync Anthropic client.

    Returns a context manager whose __exit__ emits the span once the
    final accumulated message is available.
    """

    def wrapper(self, *args, **kwargs):
        start_ns = time.time_ns()
        stream_cm = original(self, *args, **kwargs)

        class _InstrumentedStream:
            """Proxy that delegates to the real MessageStream context manager."""

            def __init__(self, cm):
                self._cm = cm
                self._stream = None

            def __enter__(self):
                self._stream = self._cm.__enter__()
                return self._stream

            def __exit__(self, exc_type, exc_val, exc_tb):
                result = self._cm.__exit__(exc_type, exc_val, exc_tb)
                try:
                    final = getattr(self._stream, "get_final_message", None)
                    if callable(final):
                        message = final()
                        attrs = _build_span_attrs(kwargs, message)
                        _emit_span(attrs, start_ns)
                    elif exc_val is not None:
                        attrs = _build_error_attrs(kwargs)
                        _emit_span(attrs, start_ns, error_message=str(exc_val))
                except Exception:
                    logger.debug("Failed to emit stream span", exc_info=True)
                return result

            # Support direct iteration (non-context-manager usage)
            def __iter__(self):
                return iter(self._cm)

        return _InstrumentedStream(stream_cm)

    return wrapper


def _wrap_async_stream(original):
    """Wrap AsyncMessages.stream() for async Anthropic client."""

    def wrapper(self, *args, **kwargs):
        start_ns = time.time_ns()
        stream_cm = original(self, *args, **kwargs)

        class _InstrumentedAsyncStream:
            """Proxy that delegates to the real AsyncMessageStream."""

            def __init__(self, cm):
                self._cm = cm
                self._stream = None

            async def __aenter__(self):
                self._stream = await self._cm.__aenter__()
                return self._stream

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                result = await self._cm.__aexit__(exc_type, exc_val, exc_tb)
                try:
                    final = getattr(self._stream, "get_final_message", None)
                    if callable(final):
                        message = final()
                        attrs = _build_span_attrs(kwargs, message)
                        _emit_span(attrs, start_ns)
                    elif exc_val is not None:
                        attrs = _build_error_attrs(kwargs)
                        _emit_span(attrs, start_ns, error_message=str(exc_val))
                except Exception:
                    logger.debug("Failed to emit async stream span", exc_info=True)
                return result

            # Support direct async iteration
            def __aiter__(self):
                return self._cm.__aiter__()

        return _InstrumentedAsyncStream(stream_cm)

    return wrapper


# ---------------------------------------------------------------------------
# Managed Agents — event stream instrumentation
# ---------------------------------------------------------------------------

_original_sync_events_stream = None
_original_async_events_stream = None


class _ManagedAgentTurnTracker:
    """Accumulates streamed events for one agent turn (user.message → idle).

    Each ``session.status_idle`` or ``session.error`` flushes a span that
    captures the full turn: user input, agent output, aggregated token
    usage across all model requests, and every tool call.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._reset()

    # -- event dispatch -------------------------------------------------------

    def process(self, event: Any) -> None:
        event_type = getattr(event, "type", None)

        if event_type == "user.message":
            for block in getattr(event, "content", []):
                text = _normalize_content_block(block)
                if text:
                    self.input_messages.append(text)

        elif event_type == "agent.message":
            for block in getattr(event, "content", []):
                text = _normalize_content_block(block)
                if text:
                    self.output_messages.append(text)

        elif event_type in (
            "agent.tool_use",
            "agent.mcp_tool_use",
            "agent.custom_tool_use",
        ):
            tc: Dict[str, Any] = {
                "id": getattr(event, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(event, "name", ""),
                    "arguments": _safe_json(getattr(event, "input", {})),
                },
            }
            if event_type == "agent.mcp_tool_use":
                tc["mcp_server"] = getattr(event, "mcp_server_name", "")
            self.tool_calls.append(tc)

        elif event_type == "span.model_request_end":
            usage = getattr(event, "model_usage", None)
            if usage:
                self.total_input_tokens += getattr(usage, "input_tokens", 0)
                self.total_output_tokens += getattr(usage, "output_tokens", 0)
                self.total_cache_read += getattr(usage, "cache_read_input_tokens", 0)
                self.total_cache_creation += getattr(
                    usage, "cache_creation_input_tokens", 0
                )

        elif event_type == "session.status_idle":
            stop_reason = getattr(event, "stop_reason", None)
            self._emit(
                stop_reason=getattr(stop_reason, "type", None) if stop_reason else None
            )
            self._reset()

        elif event_type == "session.error":
            self.error = str(event)
            self._emit()
            self._reset()

        elif event_type == "session.status_running":
            # Mark the actual start of agent work (may lag behind stream open).
            if not self.input_messages:
                self.start_ns = time.time_ns()

    # -- span emission --------------------------------------------------------

    def _emit(self, stop_reason: Optional[str] = None) -> None:
        attrs: Dict[str, Any] = {
            "gen_ai.system": "anthropic",
            "llm.request.type": "chat",
            "traceloop.entity.name": "anthropic.managed_agent",
            "traceloop.entity.path": "anthropic.managed_agent",
            "respan.entity.log_type": "chat",
            "respan.managed_agent.session_id": self.session_id,
            SpanAttributes.TRACELOOP_SPAN_KIND: LLMRequestTypeValues.CHAT.value,
        }

        if self.input_messages:
            attrs["traceloop.entity.input"] = _safe_json(
                [{"role": "user", "content": "\n".join(self.input_messages)}]
            )

        if self.output_messages:
            attrs["traceloop.entity.output"] = "\n".join(self.output_messages)

        if self.total_input_tokens or self.total_output_tokens:
            attrs["gen_ai.usage.prompt_tokens"] = self.total_input_tokens
            attrs["gen_ai.usage.completion_tokens"] = self.total_output_tokens

        if self.total_cache_read:
            attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = (
                self.total_cache_read
            )
        if self.total_cache_creation:
            attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
                self.total_cache_creation
            )

        if self.tool_calls:
            attrs["respan.span.tool_calls"] = _safe_json(self.tool_calls)

        if stop_reason:
            attrs["respan.managed_agent.stop_reason"] = stop_reason

        _emit_span(
            attrs,
            self.start_ns,
            error_message=self.error,
            name="anthropic.managed_agent",
        )

    def _reset(self) -> None:
        self.start_ns = time.time_ns()
        self.input_messages: List[str] = []
        self.output_messages: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read = 0
        self.total_cache_creation = 0
        self.error: Optional[str] = None


class _InstrumentedManagedStream:
    """Proxy for a sync ``Stream`` that intercepts events to emit spans."""

    def __init__(self, stream: Any, session_id: str) -> None:
        self._stream = stream
        self._tracker = _ManagedAgentTurnTracker(session_id)

    def __iter__(self):
        for event in self._stream:
            self._tracker.process(event)
            yield event

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(exc_type, exc_val, exc_tb)
        return None

    def close(self):
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


class _InstrumentedAsyncManagedStream:
    """Proxy for an async ``AsyncStream`` that intercepts events to emit spans."""

    def __init__(self, stream: Any, session_id: str) -> None:
        self._stream = stream
        self._tracker = _ManagedAgentTurnTracker(session_id)

    async def __aiter__(self):
        async for event in self._stream:
            self._tracker.process(event)
            yield event

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        return None

    async def close(self):
        if hasattr(self._stream, "close"):
            result = self._stream.close()
            if hasattr(result, "__await__"):
                await result

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _wrap_sync_events_stream(original):
    """Wrap ``Events.stream()`` to intercept managed agent events."""

    def wrapper(self, session_id, *args, **kwargs):
        stream = original(self, session_id, *args, **kwargs)
        return _InstrumentedManagedStream(stream, session_id)

    return wrapper


def _wrap_async_events_stream(original):
    """Wrap ``AsyncEvents.stream()`` to intercept managed agent events."""

    async def wrapper(self, session_id, *args, **kwargs):
        stream = await original(self, session_id, *args, **kwargs)
        return _InstrumentedAsyncManagedStream(stream, session_id)

    return wrapper


# ---------------------------------------------------------------------------
# Instrumentor
# ---------------------------------------------------------------------------


class AnthropicInstrumentor:
    """Respan instrumentor for the Anthropic SDK.

    Patches ``messages.create()`` and ``messages.stream()`` on both the
    sync and async clients to emit OTEL spans with GenAI attributes.

    Also patches ``beta.sessions.events.stream()`` to trace Managed Agents
    sessions.  Each agent turn (user message → ``session.status_idle``)
    produces one span with accumulated token usage, tool calls, and the
    full agent response.

    Usage::

        from respan import Respan
        from respan_instrumentation_anthropic import AnthropicInstrumentor

        respan = Respan(instrumentations=[AnthropicInstrumentor()])
    """

    name = "anthropic"

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        """Monkey-patch the Anthropic SDK."""
        global _original_sync_create, _original_async_create
        global _original_sync_stream, _original_async_stream
        global _original_sync_events_stream, _original_async_events_stream

        try:
            import anthropic
        except ImportError as exc:
            logger.warning(
                "Failed to activate Anthropic instrumentation — missing dependency: %s",
                exc,
            )
            return

        try:
            from anthropic.resources import Messages, AsyncMessages

            # Patch sync messages.create
            if _original_sync_create is None:
                _original_sync_create = Messages.create
            Messages.create = _wrap_sync_create(_original_sync_create)

            # Patch async messages.create
            if _original_async_create is None:
                _original_async_create = AsyncMessages.create
            AsyncMessages.create = _wrap_async_create(_original_async_create)

            # Patch sync messages.stream
            if hasattr(Messages, "stream"):
                if _original_sync_stream is None:
                    _original_sync_stream = Messages.stream
                Messages.stream = _wrap_sync_stream(_original_sync_stream)

            # Patch async messages.stream
            if hasattr(AsyncMessages, "stream"):
                if _original_async_stream is None:
                    _original_async_stream = AsyncMessages.stream
                AsyncMessages.stream = _wrap_async_stream(_original_async_stream)

        except Exception as exc:
            logger.warning("Failed to activate Anthropic instrumentation: %s", exc)

        # ------------------------------------------------------------------
        # Managed Agents (beta) — gracefully skip if SDK is too old
        # ------------------------------------------------------------------
        try:
            from anthropic.resources.beta.sessions import Events, AsyncEvents

            if _original_sync_events_stream is None:
                _original_sync_events_stream = Events.stream
            Events.stream = _wrap_sync_events_stream(_original_sync_events_stream)

            if _original_async_events_stream is None:
                _original_async_events_stream = AsyncEvents.stream
            AsyncEvents.stream = _wrap_async_events_stream(
                _original_async_events_stream
            )

            logger.info("Anthropic Managed Agents instrumentation activated")

        except (ImportError, AttributeError):
            # SDK version doesn't have beta.sessions — that's fine.
            logger.debug(
                "Managed Agents beta not available in installed anthropic SDK; skipping"
            )
        except Exception as exc:
            logger.warning(
                "Failed to activate Managed Agents instrumentation: %s", exc
            )

        self._is_instrumented = True
        logger.info("Anthropic SDK instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Anthropic SDK methods."""
        global _original_sync_create, _original_async_create
        global _original_sync_stream, _original_async_stream
        global _original_sync_events_stream, _original_async_events_stream

        if not self._is_instrumented:
            return

        try:
            from anthropic.resources import Messages, AsyncMessages

            if _original_sync_create is not None:
                Messages.create = _original_sync_create
                _original_sync_create = None

            if _original_async_create is not None:
                AsyncMessages.create = _original_async_create
                _original_async_create = None

            if _original_sync_stream is not None:
                Messages.stream = _original_sync_stream
                _original_sync_stream = None

            if _original_async_stream is not None:
                AsyncMessages.stream = _original_async_stream
                _original_async_stream = None

        except Exception:
            pass

        # Managed Agents
        try:
            from anthropic.resources.beta.sessions import Events, AsyncEvents

            if _original_sync_events_stream is not None:
                Events.stream = _original_sync_events_stream
                _original_sync_events_stream = None

            if _original_async_events_stream is not None:
                AsyncEvents.stream = _original_async_events_stream
                _original_async_events_stream = None

        except Exception:
            pass

        self._is_instrumented = False
        logger.info("Anthropic SDK instrumentation deactivated")
