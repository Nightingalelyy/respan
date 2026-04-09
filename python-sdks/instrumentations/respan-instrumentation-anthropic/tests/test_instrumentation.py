import json
from types import SimpleNamespace

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_anthropic import _instrumentation as instrumentation


TRACE_ID = "0123456789abcdef0123456789abcdef"
SPAN_ID = "fedcba9876543210"


def _capture_build(monkeypatch):
    captured = {}

    def _fake_build_readable_span(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"name": name, **kwargs}

    monkeypatch.setattr(
        "respan_tracing.utils.span_factory.build_readable_span",
        _fake_build_readable_span,
    )
    monkeypatch.setattr(
        "respan_tracing.utils.span_factory.inject_span",
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

    monkeypatch.setattr(instrumentation.trace, "get_current_span", lambda: _FakeSpan())

    instrumentation._emit_span({"respan.entity.log_type": "chat"}, start_ns=123)

    assert captured["trace_id"] == TRACE_ID
    assert captured["parent_id"] == SPAN_ID


def test_emit_span_without_active_parent_creates_root(monkeypatch):
    captured = _capture_build(monkeypatch)

    class _InvalidSpan:
        def get_span_context(self):
            return SimpleNamespace(trace_id=0, span_id=0)

    monkeypatch.setattr(
        instrumentation.trace, "get_current_span", lambda: _InvalidSpan()
    )

    instrumentation._emit_span({"respan.entity.log_type": "chat"}, start_ns=123)

    assert captured["trace_id"] is None
    assert captured["parent_id"] is None


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

    attrs = instrumentation._build_span_attrs(
        kwargs={"messages": [{"role": "user", "content": "hi"}]},
        message=message,
    )

    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 4


def test_build_error_attrs_sets_chat_span_kind():
    attrs = instrumentation._build_error_attrs(
        kwargs={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == LLMRequestTypeValues.CHAT.value


def test_format_input_messages_preserves_tool_blocks():
    formatted = instrumentation._format_input_messages(
        [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll look that up."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": '{"city":"Tokyo","temperature":"22C"}',
                    }
                ],
            },
        ]
    )

    payload = json.loads(formatted)
    assert payload[1]["content"][0] == "I'll look that up."
    assert payload[1]["content"][1] == {
        "type": "tool_use",
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"city": "Tokyo"},
    }
    assert payload[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": '{"city":"Tokyo","temperature":"22C"}',
        }
    ]


def test_format_output_preserves_tool_use_blocks():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="I'll get the weather."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"city": "Tokyo"},
            ),
        ]
    )

    payload = json.loads(instrumentation._format_output(message))
    assert payload[0] == "I'll get the weather."
    assert payload[1] == {
        "type": "tool_use",
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"city": "Tokyo"},
    }
