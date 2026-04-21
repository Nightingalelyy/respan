import json
from types import SimpleNamespace

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_anthropic import _instrumentation as instrumentation
from respan_instrumentation_anthropic import _managed_agents as managed_agent_helpers
from respan_instrumentation_anthropic import _messages as message_helpers
from respan_sdk.constants.span_attributes import (
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_SESSION_ID,
)


TRACE_ID = "0123456789abcdef0123456789abcdef"
SPAN_ID = "fedcba9876543210"
TOOL_CALL_ID = "toolu_123"
WEATHER_TOOL_NAME = "get_weather"
WEATHER_TOOL_INPUT = {"city": "Tokyo"}
WEATHER_TOOL_ARGUMENTS = '{"city": "Tokyo"}'
WEATHER_TOOL_RESULT = {"city": "Tokyo", "temperature": "22C"}


def _capture_build(monkeypatch):
    captured = []

    def _fake_build_readable_span(name, **kwargs):
        span = {"name": name, **kwargs}
        captured.append(span)
        return span

    monkeypatch.setattr(
        message_helpers,
        "build_readable_span",
        _fake_build_readable_span,
    )
    monkeypatch.setattr(
        message_helpers,
        "inject_span",
        lambda span: True,
    )
    return captured


def test_emit_span_uses_active_parent_context(monkeypatch):
    captured = _capture_build(monkeypatch)

    class _FakeSpan:
        def get_span_context(self):
            return SimpleNamespace(
                trace_id=int(TRACE_ID, 16),
                span_id=int(SPAN_ID, 16),
            )

    monkeypatch.setattr(message_helpers.trace, "get_current_span", lambda: _FakeSpan())

    message_helpers._emit_span({"respan.entity.log_type": "chat"}, start_ns=123)

    assert captured[0]["trace_id"] == TRACE_ID
    assert captured[0]["parent_id"] == SPAN_ID


def test_emit_span_without_active_parent_creates_root(monkeypatch):
    captured = _capture_build(monkeypatch)

    class _InvalidSpan:
        def get_span_context(self):
            return SimpleNamespace(trace_id=0, span_id=0)

    monkeypatch.setattr(message_helpers.trace, "get_current_span", lambda: _InvalidSpan())

    message_helpers._emit_span({"respan.entity.log_type": "chat"}, start_ns=123)

    assert captured[0]["trace_id"] is None
    assert captured[0]["parent_id"] is None


def test_build_span_attrs_sets_chat_span_kind():
    message = SimpleNamespace(
        model="claude-sonnet-4-20250514",
        content=[SimpleNamespace(text="hello")],
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=4,
        ),
    )

    attrs = message_helpers._build_span_attrs(
        kwargs={"messages": [{"role": "user", "content": "hi"}]},
        message=message,
    )

    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 4


def test_build_error_attrs_sets_chat_span_kind():
    attrs = message_helpers._build_error_attrs(
        kwargs={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == LLMRequestTypeValues.CHAT.value


def test_format_input_messages_preserves_tool_blocks():
    formatted = message_helpers._format_input_messages(
        [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll look that up."},
                    {
                        "type": "tool_use",
                        "id": TOOL_CALL_ID,
                        "name": WEATHER_TOOL_NAME,
                        "input": WEATHER_TOOL_INPUT,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": TOOL_CALL_ID,
                        "content": json.dumps(
                            WEATHER_TOOL_RESULT,
                            separators=(",", ":"),
                        ),
                    }
                ],
            },
        ]
    )

    payload = json.loads(formatted)
    assert payload[1] == {
        "role": "assistant",
        "content": "I'll look that up.",
        "tool_calls": [
            {
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL_NAME,
                    "arguments": WEATHER_TOOL_ARGUMENTS,
                },
            }
        ],
    }
    assert payload[2] == {
        "role": "tool",
        "tool_call_id": TOOL_CALL_ID,
        "content": WEATHER_TOOL_RESULT,
    }


def test_format_output_preserves_tool_use_blocks():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="I'll get the weather."),
            SimpleNamespace(
                type="tool_use",
                id=TOOL_CALL_ID,
                name=WEATHER_TOOL_NAME,
                input=WEATHER_TOOL_INPUT,
            ),
        ]
    )

    assert message_helpers._format_output(message) == "I'll get the weather."


def test_format_output_returns_empty_for_tool_only_response():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=TOOL_CALL_ID,
                name=WEATHER_TOOL_NAME,
                input=WEATHER_TOOL_INPUT,
            )
        ]
    )

    assert message_helpers._format_output(message) == ""


def test_build_span_attrs_sets_tool_metadata_overrides():
    message = SimpleNamespace(
        model="claude-sonnet-4-20250514",
        content=[
            SimpleNamespace(
                type="tool_use",
                id=TOOL_CALL_ID,
                name=WEATHER_TOOL_NAME,
                input=WEATHER_TOOL_INPUT,
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=5),
    )

    attrs = message_helpers._build_span_attrs(
        kwargs={
            "messages": [{"role": "user", "content": "Use the weather tool."}],
            "tools": [
                {
                    "name": WEATHER_TOOL_NAME,
                    "description": "Get the current weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        },
        message=message,
    )

    assert attrs["tool_calls"] == [
        {
            "id": TOOL_CALL_ID,
            "type": "function",
            "function": {
                "name": WEATHER_TOOL_NAME,
                "arguments": WEATHER_TOOL_ARGUMENTS,
            },
        }
    ]
    assert attrs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": WEATHER_TOOL_NAME,
                "description": "Get the current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    assert attrs["gen_ai.completion.0.tool_calls"] == attrs["tool_calls"]
    assert json.loads(attrs["respan.span.tool_calls"]) == attrs["tool_calls"]
    assert json.loads(attrs["respan.span.tools"]) == attrs["tools"]


def test_emit_message_spans_emits_resolved_child_tool_span(monkeypatch):
    captured = _capture_build(monkeypatch)
    with message_helpers._PENDING_TOOL_CALLS_LOCK:
        message_helpers._PENDING_TOOL_CALLS.clear()

    class _FakeSpan:
        def get_span_context(self):
            return SimpleNamespace(
                trace_id=int(TRACE_ID, 16),
                span_id=int(SPAN_ID, 16),
            )

    monkeypatch.setattr(message_helpers.trace, "get_current_span", lambda: _FakeSpan())
    timestamps = iter([150, 200, 500])
    monkeypatch.setattr(message_helpers.time, "time_ns", lambda: next(timestamps))

    message_helpers._emit_message_spans(
        kwargs={
            "messages": [{"role": "user", "content": "Use the weather tool."}],
            "tools": [{"name": WEATHER_TOOL_NAME, "input_schema": {"type": "object"}}],
        },
        message=SimpleNamespace(
            model="claude-sonnet-4-20250514",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id=TOOL_CALL_ID,
                    name=WEATHER_TOOL_NAME,
                    input=WEATHER_TOOL_INPUT,
                )
            ],
        ),
        start_ns=123,
    )

    message_helpers._emit_message_spans(
        kwargs={
            "messages": [
                {"role": "user", "content": "Use the weather tool."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": TOOL_CALL_ID,
                            "name": WEATHER_TOOL_NAME,
                            "input": WEATHER_TOOL_INPUT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": TOOL_CALL_ID,
                            "content": json.dumps(
                                WEATHER_TOOL_RESULT,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                },
            ],
            "tools": [{"name": WEATHER_TOOL_NAME, "input_schema": {"type": "object"}}],
        },
        message=SimpleNamespace(
            model="claude-sonnet-4-20250514",
            content=[SimpleNamespace(type="text", text="It is sunny in Tokyo.")],
        ),
        start_ns=456,
    )

    assert [span["name"] for span in captured] == [
        "anthropic.chat",
        "anthropic.chat",
        "get_weather",
    ]

    first_chat_span, second_chat_span, tool_span = captured
    assert first_chat_span["trace_id"] == TRACE_ID
    assert first_chat_span["parent_id"] == SPAN_ID
    assert second_chat_span["trace_id"] == TRACE_ID
    assert second_chat_span["parent_id"] == SPAN_ID
    assert tool_span["trace_id"] == TRACE_ID
    assert tool_span["parent_id"] == first_chat_span["span_id"]
    assert tool_span["attributes"]["respan.entity.log_type"] == "tool"
    assert tool_span["attributes"]["gen_ai.tool.name"] == WEATHER_TOOL_NAME
    assert tool_span["attributes"]["traceloop.entity.input"] == WEATHER_TOOL_ARGUMENTS
    assert (
        tool_span["attributes"]["traceloop.entity.output"]
        == json.dumps(WEATHER_TOOL_RESULT)
    )
    assert "gen_ai.system" not in tool_span["attributes"]
    assert tool_span["attributes"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": WEATHER_TOOL_NAME,
                "parameters": {"type": "object"},
            },
        }
    ]
    assert tool_span["end_time_ns"] == 456


def test_managed_agent_tracker_emits_stop_reason_and_tool_calls(monkeypatch):
    captured = []

    monkeypatch.setattr(
        managed_agent_helpers,
        "_emit_span",
        lambda **kwargs: captured.append(kwargs),
    )

    tracker = managed_agent_helpers._ManagedAgentTurnTracker(session_id="sess_123")
    tracker.process(
        SimpleNamespace(
            type="user.message",
            content=[{"type": "text", "text": "Find the weather in Tokyo."}],
        )
    )
    tracker.process(
        SimpleNamespace(
            type="agent.tool_use",
            id=TOOL_CALL_ID,
            name=WEATHER_TOOL_NAME,
            input=WEATHER_TOOL_INPUT,
        )
    )
    tracker.process(
        SimpleNamespace(
            type="agent.message",
            content=[{"type": "text", "text": "Done."}],
        )
    )
    tracker.process(
        SimpleNamespace(
            type="span.model_request_end",
            model_usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )
    )
    tracker.process(
        SimpleNamespace(
            type="session.status_idle",
            stop_reason=SimpleNamespace(type="end_turn"),
        )
    )

    emitted_span = captured[0]
    attrs = emitted_span["attrs"]
    assert attrs[RESPAN_SESSION_ID] == "sess_123"
    assert attrs["respan.managed_agent.stop_reason"] == "end_turn"
    assert json.loads(attrs["traceloop.entity.input"]) == [
        {"role": "user", "content": "Find the weather in Tokyo."}
    ]
    assert attrs["traceloop.entity.output"] == "Done."
    assert attrs[LLM_USAGE_PROMPT_TOKENS] == 3
    assert attrs[LLM_USAGE_COMPLETION_TOKENS] == 4
    assert json.loads(attrs["respan.span.tool_calls"]) == [
        {
            "id": TOOL_CALL_ID,
            "type": "function",
            "function": {
                "name": WEATHER_TOOL_NAME,
                "arguments": WEATHER_TOOL_ARGUMENTS,
            },
        }
    ]


def test_emit_message_spans_marks_failed_tool_span(monkeypatch):
    captured = _capture_build(monkeypatch)
    with message_helpers._PENDING_TOOL_CALLS_LOCK:
        message_helpers._PENDING_TOOL_CALLS.clear()

    class _FakeSpan:
        def get_span_context(self):
            return SimpleNamespace(
                trace_id=int(TRACE_ID, 16),
                span_id=int(SPAN_ID, 16),
            )

    monkeypatch.setattr(message_helpers.trace, "get_current_span", lambda: _FakeSpan())
    timestamps = iter([150, 200, 500])
    monkeypatch.setattr(message_helpers.time, "time_ns", lambda: next(timestamps))

    message_helpers._emit_message_spans(
        kwargs={
            "messages": [{"role": "user", "content": "Look up the customer."}],
            "tools": [{"name": "lookup_customer", "input_schema": {"type": "object"}}],
        },
        message=SimpleNamespace(
            model="claude-sonnet-4-20250514",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id=TOOL_CALL_ID,
                    name="lookup_customer",
                    input={"customer_id": "cust_404"},
                )
            ],
        ),
        start_ns=123,
    )

    message_helpers._emit_message_spans(
        kwargs={
            "messages": [
                {"role": "user", "content": "Look up the customer."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": TOOL_CALL_ID,
                            "name": "lookup_customer",
                            "input": {"customer_id": "cust_404"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": TOOL_CALL_ID,
                            "content": "Customer not found.",
                            "is_error": True,
                        }
                    ],
                },
            ],
            "tools": [{"name": "lookup_customer", "input_schema": {"type": "object"}}],
        },
        message=SimpleNamespace(
            model="claude-sonnet-4-20250514",
            content=[SimpleNamespace(type="text", text="Customer lookup failed.")],
        ),
        start_ns=456,
    )

    tool_span = captured[-1]
    assert tool_span["status_code"] == 500
    assert tool_span["error_message"] == "Customer not found."
    assert tool_span["attributes"]["error.message"] == "Customer not found."
    assert tool_span["attributes"]["status_code"] == 500



def test_register_pending_tool_calls_prunes_expired_entries(monkeypatch):
    with message_helpers._PENDING_TOOL_CALLS_LOCK:
        message_helpers._PENDING_TOOL_CALLS.clear()
        message_helpers._PENDING_TOOL_CALLS[(TRACE_ID, "expired_tool")] = {
            "tool_call": {"id": "expired_tool"},
            "tool_definition": {"type": "function", "function": {"name": "expired"}},
            "parent_id": SPAN_ID,
            "start_ns": 1,
            "expires_at_ns": 1,
        }

    monkeypatch.setattr(message_helpers.time, "time_ns", lambda: 5)
    message_helpers._register_pending_tool_calls(
        trace_id=TRACE_ID,
        parent_id=SPAN_ID,
        tool_calls=[
            {
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL_NAME,
                    "arguments": WEATHER_TOOL_ARGUMENTS,
                },
            }
        ],
        tools=[{"type": "function", "function": {"name": WEATHER_TOOL_NAME}}],
    )

    with message_helpers._PENDING_TOOL_CALLS_LOCK:
        assert (TRACE_ID, "expired_tool") not in message_helpers._PENDING_TOOL_CALLS
        pending_entry = message_helpers._PENDING_TOOL_CALLS[(TRACE_ID, TOOL_CALL_ID)]
    assert pending_entry["expires_at_ns"] == 5 + message_helpers._PENDING_TOOL_CALL_TTL_NS



def test_activate_rolls_back_partial_patch(monkeypatch):
    original_sync_create = object()
    original_async_create = object()

    class _Messages:
        create = original_sync_create

    class _AsyncMessages:
        create = original_async_create

    instrumentor = instrumentation.AnthropicInstrumentor()

    monkeypatch.setattr(
        instrumentation,
        "_load_messages_classes",
        lambda: (_Messages, _AsyncMessages),
    )
    monkeypatch.setattr(
        instrumentation,
        "_wrap_async_create",
        lambda original: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    instrumentor.activate()

    assert _Messages.create is original_sync_create
    assert _AsyncMessages.create is original_async_create
    assert instrumentor._is_instrumented is False
