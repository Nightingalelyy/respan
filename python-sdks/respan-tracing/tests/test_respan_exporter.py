from types import SimpleNamespace
from unittest.mock import Mock

from opentelemetry.semconv_ai import SpanAttributes

from respan_tracing.exporters.respan import _prepare_spans_for_export


def _make_span(
    *,
    name: str,
    span_id: int,
    trace_id: int = 1,
    parent: SimpleNamespace | None = None,
    attributes: dict | None = None,
    scope_name: str = "test-scope",
) -> Mock:
    span = Mock()
    span.name = name
    span.parent = parent
    span._parent = parent
    span.attributes = attributes or {}
    span.instrumentation_scope = SimpleNamespace(name=scope_name, version="1.0.0")
    span.get_span_context.return_value = SimpleNamespace(
        trace_id=trace_id,
        span_id=span_id,
    )
    return span


def test_prepare_spans_passes_all_spans_through():
    """All spans are passed through without filtering."""

    agent_span = _make_span(
        name="invoke_agent agent",
        span_id=1001,
        attributes={"respan.entity.log_type": "agent"},
        scope_name="pydantic-ai",
    )
    agent_context = agent_span.get_span_context.return_value

    wrapper_span = _make_span(
        name="chat gpt-4o",
        span_id=1002,
        parent=agent_context,
        attributes={
            "respan.entity.log_type": "chat",
            "model": "gpt-4o",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_request_tokens": 18,
            "traceloop.entity.input": '[{"role": "user", "content": "compute 1 + 2"}]',
            "traceloop.entity.output": '[{"role": "assistant", "content": "3"}]',
        },
        scope_name="pydantic-ai",
    )

    prepared = _prepare_spans_for_export(
        spans=[agent_span, wrapper_span]
    )

    assert [s.name for s in prepared] == [
        "invoke_agent agent",
        "chat gpt-4o",
    ]

    kept_chat = prepared[1]
    assert kept_chat.attributes["respan.entity.log_type"] == "chat"
    assert kept_chat.attributes["model"] == "gpt-4o"
    assert kept_chat.attributes["prompt_tokens"] == 11
    assert kept_chat.attributes["traceloop.entity.input"] is not None
    assert kept_chat.attributes["traceloop.entity.output"] is not None


def test_prepare_spans_preserves_parent_relationships():
    """Parent-child relationships are preserved."""

    wrapper_span = _make_span(
        name="chat gpt-4o",
        span_id=2001,
        attributes={
            "respan.entity.log_type": "chat",
            "traceloop.entity.input": '[{"role": "user", "content": "hi"}]',
            "traceloop.entity.output": '[{"role": "assistant", "content": "hello"}]',
        },
        scope_name="pydantic-ai",
    )
    wrapper_context = wrapper_span.get_span_context.return_value

    http_child = _make_span(
        name="http.request",
        span_id=2003,
        parent=wrapper_context,
        attributes={"http.method": "POST"},
        scope_name="opentelemetry.instrumentation.requests",
    )

    prepared = _prepare_spans_for_export(
        spans=[wrapper_span, http_child]
    )

    assert [s.name for s in prepared] == ["chat gpt-4o", "http.request"]
    assert prepared[1].parent.span_id == wrapper_context.span_id


def test_prepare_spans_keeps_all_provider_spans():
    """Spans from any provider (OpenAI, Anthropic, etc.) are kept."""

    wrapper_span = _make_span(
        name="chat anthropic",
        span_id=3001,
        attributes={
            "respan.entity.log_type": "chat",
            "traceloop.entity.input": '[{"role": "user", "content": "hi"}]',
            "traceloop.entity.output": '[{"role": "assistant", "content": "hello"}]',
        },
        scope_name="pydantic-ai",
    )

    prepared = _prepare_spans_for_export(spans=[wrapper_span])

    assert [s.name for s in prepared] == ["chat anthropic"]
