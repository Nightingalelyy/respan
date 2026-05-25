import json

from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_braintrust import BraintrustInstrumentor
from respan_instrumentation_braintrust import _instrumentation as braintrust_module
from respan_instrumentation_braintrust._instrumentation import _build_span_from_record
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.data_processing.id_processing import format_span_id


def _llm_record():
    return {
        "id": "00000000-0000-0000-0000-000000000099",
        "root_span_id": "00000000-0000-0000-0000-000000000001",
        "span_id": "00000000-0000-0000-0000-000000000002",
        "span_parents": [],
        "span_attributes": {"type": "llm", "name": "answer_question"},
        "input": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is Respan?"},
        ],
        "output": {"role": "assistant", "content": "Respan captures LLM traces."},
        "metadata": {"request_id": "req-1", "model": "gpt-4o-mini"},
        "tags": ["demo"],
        "scores": {"quality": 0.92},
        "metrics": {
            "start": 1_700_000_000.0,
            "end": 1_700_000_001.25,
            "prompt_tokens": 12,
            "completion_tokens": 8,
        },
    }


def test_llm_record_maps_to_canonical_chat_span():
    span = _build_span_from_record(_llm_record())
    attrs = span.attributes

    assert span.name == "answer_question"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[GEN_AI_SYSTEM] == "braintrust"
    assert attrs[LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[LLM_USAGE_COMPLETION_TOKENS] == 8
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert attrs["llm.usage.total_tokens"] == 20
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == "Answer briefly."
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.1.content"] == "What is Respan?"
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "Respan captures LLM traces."
    )

    metadata = json.loads(attrs[RESPAN_METADATA])
    assert metadata["request_id"] == "req-1"
    assert metadata["braintrust_tags"] == ["demo"]
    assert metadata["braintrust_scores"] == {"quality": 0.92}
    assert metadata["braintrust_log_id"] == "00000000000000000000000000000099"

    forbidden_aliases = {
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "tools",
        "tool_calls",
        "span_tools",
        "has_tool_calls",
        "respan.span.tools",
        "respan.span.tool_calls",
    }
    assert forbidden_aliases.isdisjoint(attrs)


def test_child_tool_record_preserves_parent_and_avoids_llm_fields():
    parent_id = "00000000-0000-0000-0000-00000000aaaa"
    record = {
        "root_span_id": "00000000-0000-0000-0000-000000000001",
        "span_id": "00000000-0000-0000-0000-00000000bbbb",
        "span_parents": [parent_id],
        "span_attributes": {"type": "function", "name": "lookup_order"},
        "input": {"order_id": "ord-1"},
        "output": {"status": "shipped"},
    }

    span = _build_span_from_record(record)
    attrs = span.attributes

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_order"
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] == "braintrust"
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] == '{"order_id": "ord-1"}'
    assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '{"status": "shipped"}'
    assert LLM_REQUEST_TYPE not in attrs
    assert span.parent is not None
    assert format_span_id(span.parent.span_id) == "000000000000aaaa"


def test_flush_injects_buffered_braintrust_records(monkeypatch):
    exported = []

    class LazyRecord:
        def get(self):
            return _llm_record()

    monkeypatch.setattr(
        braintrust_module,
        "inject_span",
        lambda span: exported.append(span) or True,
    )
    monkeypatch.setattr(
        braintrust_module,
        "read_propagated_attributes",
        lambda: {RESPAN_TRACE_GROUP_ID: "Braintrust Basic Workflow"},
    )

    instrumentor = BraintrustInstrumentor()
    instrumentor.log(LazyRecord())
    instrumentor.flush()

    assert [span.name for span in exported] == ["answer_question"]
    assert exported[0].attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert exported[0].attributes[RESPAN_TRACE_GROUP_ID] == "Braintrust Basic Workflow"
    assert (
        exported[0].attributes[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME]
        == "Braintrust Basic Workflow"
    )
