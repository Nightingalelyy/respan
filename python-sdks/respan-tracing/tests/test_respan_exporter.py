import json
from types import SimpleNamespace
from unittest.mock import Mock

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.otlp_constants import (
    OTLP_ARRAY_VALUE,
    OTLP_ARRAY_VALUES_KEY,
    OTLP_ATTR_KEY,
    OTLP_ATTR_VALUE,
    OTLP_ATTRIBUTES_KEY,
    OTLP_KVLIST_VALUE,
    OTLP_RESOURCE_SPANS_KEY,
    OTLP_SCOPE_SPANS_KEY,
    OTLP_SPANS_KEY,
    OTLP_STRING_VALUE,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_tracing.exporters.respan import (
    RespanSpanExporter,
    _get_enrichment_attrs,
    _prepare_spans_for_export,
    _span_to_otlp_json,
)


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
    span.kind = None
    span.start_time = None
    span.end_time = None
    span.status = None
    span.events = []
    span.links = ()
    span.resource = SimpleNamespace(attributes={})
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


def test_exporter_normalizes_base_endpoint_to_v2_traces():
    exporter = RespanSpanExporter(endpoint="https://api.respan.ai/api", api_key="test-key")

    assert exporter._traces_url == "https://api.respan.ai/api/v2/traces"


def test_exporter_accepts_full_v2_traces_endpoint_without_duplication():
    exporter = RespanSpanExporter(
        endpoint="https://api.respan.ai/api/v2/traces",
        api_key="test-key",
    )

    assert exporter._traces_url == "https://api.respan.ai/api/v2/traces"


def test_prepare_spans_adds_claude_agent_final_chat_child_for_tool_turn():
    """Claude Agent tool turns should emit a synthetic final child chat span."""

    wrapper_span = _make_span(
        name="ClaudeAgentSDK.query",
        span_id=3002,
        attributes={
            "respan.entity.log_type": "agent",
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-5",
            "traceloop.entity.input": "Use the weather tool.",
            "traceloop.entity.output": "Tokyo is sunny and 22C.",
            RESPAN_SPAN_TOOL_CALLS: json.dumps([
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Tokyo"}',
                    },
                }
            ]),
        },
        scope_name="openinference.instrumentation.claude_agent_sdk",
    )
    wrapper_context = wrapper_span.get_span_context.return_value

    prepared = _prepare_spans_for_export(spans=[wrapper_span])

    assert [s.name for s in prepared] == [
        "ClaudeAgentSDK.query",
        "assistant_message",
    ]
    synthetic_child = prepared[1]
    assert synthetic_child.parent.span_id == wrapper_context.span_id
    assert synthetic_child.attributes["respan.entity.log_type"] == "chat"
    assert synthetic_child.attributes["gen_ai.completion.0.role"] == "assistant"
    assert (
        synthetic_child.attributes["gen_ai.completion.0.content"]
        == "Tokyo is sunny and 22C."
    )
    assert synthetic_child.attributes["traceloop.entity.input"] == "Use the weather tool."


def test_prepare_spans_remaps_tool_call_helpers_and_strips_helper_attrs():
    """Exporter remaps helper attrs to completion message fields before OTLP serialization."""

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"NYC"}',
            },
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    chat_span = _make_span(
        name="openai.chat",
        span_id=4001,
        attributes={
            "gen_ai.system": "openai",
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
            RESPAN_SPAN_TOOLS: json.dumps(tools),
        },
        scope_name="openai-agents",
    )

    prepared = _prepare_spans_for_export(spans=[chat_span])
    prepared_attrs = prepared[0].attributes

    assert prepared_attrs["gen_ai.completion.0.tool_calls"] == tool_calls
    assert prepared_attrs["gen_ai.completion.0.role"] == "assistant"
    assert prepared_attrs["gen_ai.completion.0.content"] == ""

    otlp_span = _span_to_otlp_json(prepared[0])
    otlp_attrs = {
        item[OTLP_ATTR_KEY]: item[OTLP_ATTR_VALUE]
        for item in otlp_span[OTLP_ATTRIBUTES_KEY]
    }

    assert RESPAN_SPAN_TOOL_CALLS not in otlp_attrs
    assert RESPAN_SPAN_TOOLS not in otlp_attrs
    assert "gen_ai.completion.0.tool_calls" in otlp_attrs
    tool_calls_value = otlp_attrs["gen_ai.completion.0.tool_calls"][OTLP_ARRAY_VALUE][
        OTLP_ARRAY_VALUES_KEY
    ]
    assert len(tool_calls_value) == 1
    first_tool_call = {
        item[OTLP_ATTR_KEY]: item[OTLP_ATTR_VALUE]
        for item in tool_calls_value[0][OTLP_KVLIST_VALUE][OTLP_ARRAY_VALUES_KEY]
    }
    assert first_tool_call["id"][OTLP_STRING_VALUE] == "call_1"


def test_prepare_spans_backfills_completion_content_from_output_when_needed():
    """Tool-call OTLP spans should surface the final assistant text when available."""

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]
    final_text = "Tokyo is sunny and 22C."
    chat_span = _make_span(
        name="ClaudeAgentSDK.query",
        span_id=4002,
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "",
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json.dumps(final_text),
        },
        scope_name="openinference.instrumentation.claude_agent_sdk",
    )

    prepared = _prepare_spans_for_export(spans=[chat_span])
    prepared_attrs = prepared[0].attributes

    assert prepared_attrs["gen_ai.completion.0.tool_calls"] == tool_calls
    assert prepared_attrs["gen_ai.completion.0.content"] == final_text
    assert prepared_attrs["gen_ai.completion.0.role"] == "assistant"


def test_span_to_otlp_json_strips_redundant_override_attrs():
    chat_span = _make_span(
        name="invoke_agent claude-agent-sdk-complex-edge-cases",
        span_id=4004,
        attributes={
            "gen_ai.system": "anthropic",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_calls": [{"id": "call_1"}],
            "span_tools": ["get_weather"],
            "span_workflow_name": "claude-agent-sdk-complex-edge-cases",
            "input": '[{"role":"user","content":"hi"}]',
            "output": '{"role":"assistant","content":"hello"}',
            "traceloop.entity.name": "claude-agent-sdk-complex-edge-cases",
        },
        scope_name="opentelemetry.instrumentation.claude_agent_sdk",
    )

    otlp_span = _span_to_otlp_json(chat_span)
    otlp_attrs = {
        item[OTLP_ATTR_KEY]: item[OTLP_ATTR_VALUE]
        for item in otlp_span[OTLP_ATTRIBUTES_KEY]
    }

    assert "tools" in otlp_attrs
    assert "tool_calls" not in otlp_attrs
    assert "span_tools" not in otlp_attrs
    assert "span_workflow_name" not in otlp_attrs
    assert "input" not in otlp_attrs
    assert "output" not in otlp_attrs
    assert "traceloop.entity.name" not in otlp_attrs


def test_exporter_omits_internal_scope_metadata_for_claude_spans():
    chat_span = _make_span(
        name="invoke_agent claude-agent-sdk-complex-edge-cases",
        span_id=4005,
        attributes={"gen_ai.system": "anthropic"},
        scope_name="opentelemetry.instrumentation.claude_agent_sdk",
    )

    exporter = RespanSpanExporter(endpoint="https://example.com/api", api_key="test-key")
    exporter._session = Mock()
    exporter._session.post.return_value = SimpleNamespace(status_code=200, text="ok")

    result = exporter.export([chat_span])

    assert result == SpanExportResult.SUCCESS

    otlp_call = exporter._session.post.call_args.kwargs
    otlp_payload = json.loads(otlp_call["data"])
    scope_spans = otlp_payload[OTLP_RESOURCE_SPANS_KEY][0][OTLP_SCOPE_SPANS_KEY][0]

    assert "scope" not in scope_spans


def test_get_enrichment_attrs_remaps_cache_usage_to_override_fields():
    span = _make_span(
        name="anthropic.chat",
        span_id=4003,
        attributes={
            "gen_ai.system": "anthropic",
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS: 1422,
            SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS: 71,
        },
    )

    enriched = _get_enrichment_attrs(span)

    assert enriched["prompt_cache_hit_tokens"] == 1422
    assert enriched["prompt_cache_creation_tokens"] == 71


def test_export_keeps_tool_helper_spans_in_single_otlp_pipeline():
    """Tool helper spans should stay in the OTLP export path."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    tool_span = _make_span(
        name="anthropic.chat",
        span_id=5004,
        attributes={
            "respan.entity.log_type": "generation",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: '[{"role":"user","content":"weather?"}]',
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: '{"role":"assistant","content":""}',
            RESPAN_SPAN_TOOLS: json.dumps(tools),
        },
    )
    tool_context = tool_span.get_span_context.return_value
    plain_span = _make_span(
        name="http.request",
        span_id=5005,
        parent=tool_context,
        attributes={"http.method": "POST"},
    )

    exporter = RespanSpanExporter(endpoint="https://example.com/api", api_key="test-key")
    exporter._session = Mock()
    exporter._session.post.return_value = SimpleNamespace(status_code=200, text="ok")

    result = exporter.export([tool_span, plain_span])

    assert result == SpanExportResult.SUCCESS
    assert exporter._session.post.call_count == 1

    otlp_call = exporter._session.post.call_args.kwargs
    assert otlp_call["url"] == "https://example.com/api/v2/traces"
    otlp_payload = json.loads(otlp_call["data"])
    otlp_spans = otlp_payload[OTLP_RESOURCE_SPANS_KEY][0][OTLP_SCOPE_SPANS_KEY][0][
        OTLP_SPANS_KEY
    ]
    assert len(otlp_spans) == 2
    assert [span["name"] for span in otlp_spans] == ["anthropic.chat", "http.request"]
