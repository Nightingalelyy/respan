from types import SimpleNamespace

from opentelemetry.trace import Status, StatusCode

from respan_instrumentation_agentops._processor import translate_agentops_span
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA
from opentelemetry.semconv_ai import SpanAttributes


class FakeSpan:
    def __init__(self, name: str, attributes: dict, *, error: str | None = None):
        self.name = name
        self._attributes = dict(attributes)
        self.instrumentation_scope = SimpleNamespace(name="agentops")
        self.status = (
            Status(StatusCode.ERROR, error) if error else Status(StatusCode.OK)
        )


def test_agentops_task_maps_native_content_and_strips_vendor_fields() -> None:
    span = FakeSpan(
        "prepare.task",
        {
            "agentops.span.kind": "task",
            "operation.name": "prepare",
            "agentops.task.input": '{"args":["hello"],"kwargs":{}}',
            "agentops.task.output": '"HELLO"',
            "agentops.tags": ["example"],
        },
    )

    assert translate_agentops_span(span, capture_content=True)
    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "prepare"
    assert (
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] == '{"args":["hello"],"kwargs":{}}'
    )
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '"HELLO"'
    assert '"kind": "task"' in attrs[RESPAN_METADATA]
    assert not any(key.startswith("agentops.") for key in attrs)
    assert "operation.name" not in attrs


def test_agentops_tool_uses_tool_identity_and_error_contract() -> None:
    span = FakeSpan(
        "lookup.tool",
        {
            "agentops.span.kind": "tool",
            "operation.name": "lookup",
            "tool.name": "weather_lookup",
            "agentops.tool.input": '{"city":"Paris"}',
        },
        error="lookup failed",
    )

    assert translate_agentops_span(span, capture_content=True)
    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather_lookup"
    assert attrs["status_code"] == 500
    assert attrs["error.message"] == "lookup failed"
    assert "tool_calls" not in attrs
    assert "respan.span.tool_calls" not in attrs


def test_agentops_llm_promotes_only_fields_agentops_emits() -> None:
    span = FakeSpan(
        "chat.llm",
        {
            "agentops.span.kind": "llm",
            "operation.name": "chat",
            "gen_ai.request.type": "chat",
            "gen_ai.request.functions": '[{"name":"lookup"}]',
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS: 8,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS: 3,
            "gen_ai.usage.total_tokens": 11,
        },
    )

    assert translate_agentops_span(span, capture_content=True)
    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] == '[{"name":"lookup"}]'
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 11
    assert "gen_ai.request.type" not in attrs
    assert "gen_ai.request.functions" not in attrs
