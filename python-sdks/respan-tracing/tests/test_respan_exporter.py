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
    _partition_spans_for_export,
    _prepare_spans_for_export,
    _span_to_direct_log,
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


def test_partition_spans_routes_tool_helper_spans_to_direct_ingest():
    """Tool helper spans are exported via direct ingest, not OTLP."""

    tool_span = _make_span(
        name="anthropic.chat",
        span_id=5001,
        attributes={
            RESPAN_SPAN_TOOLS: json.dumps([]),
            "respan.entity.log_type": "generation",
        },
    )
    plain_span = _make_span(
        name="http.request",
        span_id=5002,
        attributes={"http.method": "POST"},
    )

    direct_spans, otlp_spans = _partition_spans_for_export(
        _prepare_spans_for_export([tool_span, plain_span])
    )

    assert [span.name for span in direct_spans] == ["anthropic.chat"]
    assert [span.name for span in otlp_spans] == ["http.request"]


def test_span_to_direct_log_promotes_tools_and_tool_calls():
    """Direct-ingest conversion preserves structured tools/tool_calls fields."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Paris"}',
            },
        }
    ]
    span = _make_span(
        name="anthropic.chat",
        span_id=5003,
        attributes={
            "respan.entity.log_type": "generation",
            "gen_ai.request.model": "claude-3-7-sonnet",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: '[{"role":"user","content":"weather?"}]',
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: '{"role":"assistant","content":"","tool_calls":[{"id":"call_1"}]}',
            RESPAN_SPAN_TOOLS: json.dumps(tools),
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
        },
    )

    direct_log = _span_to_direct_log(span)

    assert direct_log["log_type"] == "generation"
    assert direct_log["model"] == "claude-3-7-sonnet"
    assert direct_log["tools"] == tools
    assert direct_log["tool_calls"] == tool_calls
    parsed_input = json.loads(direct_log["input"])
    parsed_output = (
        json.loads(direct_log["output"])
        if isinstance(direct_log["output"], str)
        else direct_log["output"]
    )
    assert parsed_input[0]["role"] == "user"
    assert parsed_output["role"] == "assistant"


def test_span_to_direct_log_normalizes_tool_choice_chat_payloads():
    """Tool-bearing chat spans prefer normalized messages over raw provider payloads."""

    raw_request = {
        "model": "claude-haiku-4-5",
        "max_tokens": 200,
        "tool_choice": {"type": "auto"},
        "tools": [{"name": "lookup_weather"}],
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    raw_response = {
        "id": "msg_123",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "lookup_weather",
                "input": {"city": "Tokyo"},
            }
        ],
        "stop_reason": "tool_use",
    }
    tools = [{"name": "lookup_weather"}]
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]
    span = _make_span(
        name="messages.create",
        span_id=5006,
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-haiku-4-5-20251001",
            SpanAttributes.TRACELOOP_WORKFLOW_NAME: "anthropic_complex_edge_cases_v2",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: json.dumps(raw_request),
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json.dumps(raw_response),
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "What's the weather in Tokyo?",
            RESPAN_SPAN_TOOLS: json.dumps(tools),
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
        },
    )

    prepared_span = _prepare_spans_for_export([span])[0]
    direct_log = _span_to_direct_log(prepared_span)

    assert direct_log["log_type"] == "chat"
    assert direct_log["span_workflow_name"] == "anthropic_complex_edge_cases_v2"
    assert direct_log["max_tokens"] == 200
    assert direct_log["tool_choice"] == {"type": "auto"}
    assert direct_log["full_request"] == raw_request
    assert direct_log["full_response"] == raw_response

    parsed_input = json.loads(direct_log["input"])
    parsed_output = (
        json.loads(direct_log["output"])
        if isinstance(direct_log["output"], str)
        else direct_log["output"]
    )

    assert parsed_input == raw_request["messages"]
    assert parsed_output == {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    assert direct_log["prompt_messages"] == raw_request["messages"]
    assert direct_log["prompt_message_count"] == 1
    assert direct_log["prompt_text"] == "What's the weather in Tokyo?"
    assert direct_log["completion_message"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    assert direct_log["completion_message_count"] == 1
    assert direct_log["completion_text"] == ""


def test_span_to_direct_log_prefers_final_raw_output_over_tool_only_completion():
    """Tool-only completion messages should not hide the final assistant result."""

    tools = [{"name": "lookup_weather"}]
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Kyoto"}',
            },
        }
    ]
    final_result = "Kyoto is 18C and sunny."
    span = _make_span(
        name="ClaudeAgentSDK.query",
        span_id=5007,
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-5",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: json.dumps(
                [{"role": "user", "content": "Check Kyoto weather."}]
            ),
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json.dumps(final_result),
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Check Kyoto weather.",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "",
            "gen_ai.completion.0.tool_calls": tool_calls,
            RESPAN_SPAN_TOOLS: json.dumps(tools),
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
        },
    )

    direct_log = _span_to_direct_log(span)

    parsed_output = (
        json.loads(direct_log["output"])
        if isinstance(direct_log["output"], str)
        else direct_log["output"]
    )
    assert parsed_output == {
        "role": "assistant",
        "content": final_result,
    }
    assert direct_log["completion_message"] == {
        "role": "assistant",
        "content": final_result,
    }
    assert direct_log["completion_message_count"] == 1
    assert direct_log["completion_text"] == final_result
    assert direct_log["full_response"] == final_result


def test_span_to_direct_log_prefers_last_meaningful_completion_message():
    """Multi-step agent spans should expose the final assistant completion as output."""

    tools = [{"name": "lookup_weather"}]
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Kyoto"}',
            },
        }
    ]
    final_message = {
        "role": "assistant",
        "content": "Kyoto is 18C and sunny, and Fushimi Inari is a great stop today.",
    }
    first_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    span = _make_span(
        name="ClaudeAgentSDK.query",
        span_id=5008,
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-5",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: json.dumps(
                [{"role": "user", "content": "Check Kyoto weather."}]
            ),
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json.dumps(final_message["content"]),
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Check Kyoto weather.",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "",
            "gen_ai.completion.0.tool_calls": tool_calls,
            "gen_ai.completion.1.role": "assistant",
            "gen_ai.completion.1.content": final_message["content"],
            RESPAN_SPAN_TOOLS: json.dumps(tools),
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
        },
    )

    direct_log = _span_to_direct_log(span)

    parsed_output = (
        json.loads(direct_log["output"])
        if isinstance(direct_log["output"], str)
        else direct_log["output"]
    )
    assert parsed_output == final_message
    assert direct_log["completion_message"] == final_message
    assert direct_log["completion_messages"] == [
        first_message,
        final_message,
    ]
    assert direct_log["completion_message_count"] == 2
    assert direct_log["completion_text"] == final_message["content"]


def test_export_uses_direct_ingest_for_tool_helper_spans_only():
    """Tool helper spans go to direct ingest while other spans stay on OTLP."""

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
    exporter._session.post.side_effect = [
        SimpleNamespace(status_code=200, text="ok"),
        SimpleNamespace(status_code=200, text="ok"),
    ]

    result = exporter.export([tool_span, plain_span])

    assert result == SpanExportResult.SUCCESS
    assert exporter._session.post.call_count == 2

    direct_call = exporter._session.post.call_args_list[0].kwargs
    assert direct_call["url"] == "https://example.com/api/v1/traces/ingest"
    direct_payload = json.loads(direct_call["data"])
    assert direct_payload["data"][0]["tools"] == tools
    assert direct_payload["data"][0]["span_name"] == "anthropic.chat"

    otlp_call = exporter._session.post.call_args_list[1].kwargs
    assert otlp_call["url"] == "https://example.com/api/v2/traces"
    otlp_payload = json.loads(otlp_call["data"])
    otlp_spans = otlp_payload[OTLP_RESOURCE_SPANS_KEY][0][OTLP_SCOPE_SPANS_KEY][0][
        OTLP_SPANS_KEY
    ]
    assert len(otlp_spans) == 1
    assert otlp_spans[0]["name"] == "http.request"
