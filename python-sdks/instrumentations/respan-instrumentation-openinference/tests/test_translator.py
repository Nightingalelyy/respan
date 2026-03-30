"""Unit tests for OpenInferenceTranslator.

Uses a lightweight mock span (just needs ``_attributes`` and ``name``)
so we don't depend on any OpenInference packages at test time.
"""

import json
from types import SimpleNamespace

import pytest
from opentelemetry.attributes import BoundedAttributes

from respan_instrumentation_openinference._translator import OpenInferenceTranslator
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


def _make_span(attrs: dict, name: str = "test-span"):
    """Return a minimal mock ReadableSpan with a mutable _attributes dict."""
    span = SimpleNamespace()
    span._attributes = dict(attrs)
    span.name = name
    return span


@pytest.fixture
def translator():
    return OpenInferenceTranslator()


# ------------------------------------------------------------------
# 1. Non-OI span is ignored
# ------------------------------------------------------------------

def test_non_oi_span_ignored(translator):
    """Span without openinference.span.kind is not modified."""
    span = _make_span({"some.attr": "value"})
    original = dict(span._attributes)
    translator.on_end(span)
    assert span._attributes == original


# ------------------------------------------------------------------
# 2. CHAIN → workflow
# ------------------------------------------------------------------

def test_chain_span_maps_to_workflow(translator):
    span = _make_span({"openinference.span.kind": "CHAIN"})
    translator.on_end(span)
    assert span._attributes["traceloop.span.kind"] == "workflow"
    assert span._attributes["respan.entity.log_type"] == "workflow"


# ------------------------------------------------------------------
# 3. LLM → task + llm.request.type=chat
# ------------------------------------------------------------------

def test_llm_span_maps_to_task_with_chat(translator):
    span = _make_span({"openinference.span.kind": "LLM"})
    translator.on_end(span)
    assert span._attributes["traceloop.span.kind"] == "task"
    assert span._attributes["llm.request.type"] == "chat"
    assert span._attributes["respan.entity.log_type"] == "chat"


# ------------------------------------------------------------------
# 4. TOOL → tool
# ------------------------------------------------------------------

def test_tool_span_maps_to_tool(translator):
    span = _make_span({"openinference.span.kind": "TOOL"})
    translator.on_end(span)
    assert span._attributes["traceloop.span.kind"] == "tool"
    assert span._attributes["respan.entity.log_type"] == "tool"


# ------------------------------------------------------------------
# 5. AGENT → agent
# ------------------------------------------------------------------

def test_agent_span_maps_to_agent(translator):
    span = _make_span({"openinference.span.kind": "AGENT"})
    translator.on_end(span)
    assert span._attributes["traceloop.span.kind"] == "agent"
    assert span._attributes["respan.entity.log_type"] == "agent"


# ------------------------------------------------------------------
# 6. input/output mapped
# ------------------------------------------------------------------

def test_input_output_mapped(translator):
    span = _make_span({
        "openinference.span.kind": "CHAIN",
        "input.value": "hello world",
        "output.value": "goodbye world",
    })
    translator.on_end(span)
    assert span._attributes["traceloop.entity.input"] == "hello world"
    assert span._attributes["traceloop.entity.output"] == "goodbye world"


# ------------------------------------------------------------------
# 7. model name mapped
# ------------------------------------------------------------------

def test_model_name_mapped(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4o",
    })
    translator.on_end(span)
    assert span._attributes["gen_ai.request.model"] == "gpt-4o"


# ------------------------------------------------------------------
# 8. token counts mapped
# ------------------------------------------------------------------

def test_token_counts_mapped(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.token_count.prompt": 100,
        "llm.token_count.completion": 50,
        "llm.token_count.total": 150,
        "llm.token_count.prompt_details.cache_read": 20,
    })
    translator.on_end(span)
    assert span._attributes["gen_ai.usage.prompt_tokens"] == 100
    assert span._attributes["gen_ai.usage.input_tokens"] == 100
    assert span._attributes["gen_ai.usage.completion_tokens"] == 50
    assert span._attributes["gen_ai.usage.output_tokens"] == 50
    assert span._attributes["llm.usage.total_tokens"] == 150
    assert span._attributes["llm.usage.cache_read_input_tokens"] == 20


# ------------------------------------------------------------------
# 9. system / provider mapped
# ------------------------------------------------------------------

def test_system_provider_mapped(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.system": "OpenAI",
        "llm.provider": "Azure",
    })
    translator.on_end(span)
    assert span._attributes["gen_ai.system"] == "openai"
    assert span._attributes["gen_ai.provider.name"] == "azure"


# ------------------------------------------------------------------
# 10. setdefault does not overwrite existing attributes
# ------------------------------------------------------------------

def test_setdefault_does_not_overwrite(translator):
    span = _make_span({
        "openinference.span.kind": "CHAIN",
        "traceloop.span.kind": "agent",  # pre-existing
        "traceloop.entity.input": "keep me",  # pre-existing
        "input.value": "should not overwrite",
    })
    translator.on_end(span)
    # Pre-existing values must be preserved
    assert span._attributes["traceloop.span.kind"] == "agent"
    assert span._attributes["traceloop.entity.input"] == "keep me"


# ------------------------------------------------------------------
# 11. messages translated
# ------------------------------------------------------------------

def test_messages_translated(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "Hello!",
        "llm.input_messages.1.message.role": "assistant",
        "llm.input_messages.1.message.content": "Hi there!",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "Goodbye!",
    })
    translator.on_end(span)
    assert span._attributes["gen_ai.prompt.0.role"] == "user"
    assert span._attributes["gen_ai.prompt.0.content"] == "Hello!"
    assert span._attributes["gen_ai.prompt.1.role"] == "assistant"
    assert span._attributes["gen_ai.prompt.1.content"] == "Hi there!"
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.completion.0.content"] == "Goodbye!"


def test_indexed_message_content_translated(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content.0": "First line",
        "llm.output_messages.0.message.content.1": "Second line",
        "llm.output_messages.1.message.role": "assistant",
        "llm.output_messages.1.message.content.0": "Final answer",
    })

    translator.on_end(span)

    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.completion.0.content"] == "First line\nSecond line"
    assert span._attributes["gen_ai.completion.1.role"] == "assistant"
    assert span._attributes["gen_ai.completion.1.content"] == "Final answer"


def test_indexed_message_content_preserves_empty_blocks(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content.0": "",
        "llm.output_messages.0.message.content.1": "tool result here",
    })

    translator.on_end(span)

    assert span._attributes["gen_ai.completion.0.content"] == "\ntool result here"


# ------------------------------------------------------------------
# 12. invocation parameters extracted
# ------------------------------------------------------------------

def test_invocation_params_extracted(translator):
    params = {
        "model": "claude-3-opus",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
    }
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.invocation_parameters": json.dumps(params),
    })
    translator.on_end(span)
    assert span._attributes["gen_ai.request.model"] == "claude-3-opus"
    assert span._attributes["gen_ai.request.temperature"] == 0.7
    assert span._attributes["gen_ai.request.top_p"] == 0.9
    assert span._attributes["gen_ai.request.max_tokens"] == 1024


def test_redundant_oi_attrs_removed_after_translation(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "input.value": "hello world",
        "output.value": "goodbye world",
        "llm.model_name": "claude-3-opus",
        "llm.provider": "Anthropic",
        "llm.system": "Anthropic",
        "llm.invocation_parameters": json.dumps({"temperature": 0.7}),
        "llm.token_count.prompt": 100,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "Goodbye!",
    })

    translator.on_end(span)

    assert span._attributes["traceloop.entity.input"] == "hello world"
    assert span._attributes["traceloop.entity.output"] == "goodbye world"
    assert span._attributes["gen_ai.request.model"] == "claude-3-opus"
    assert span._attributes["gen_ai.request.temperature"] == 0.7
    assert span._attributes["gen_ai.usage.prompt_tokens"] == 100
    assert span._attributes["gen_ai.completion.0.content"] == "Goodbye!"

    assert "openinference.span.kind" not in span._attributes
    assert "input.value" not in span._attributes
    assert "output.value" not in span._attributes
    assert "llm.model_name" not in span._attributes
    assert "llm.provider" not in span._attributes
    assert "llm.system" not in span._attributes
    assert "llm.invocation_parameters" not in span._attributes
    assert "llm.token_count.prompt" not in span._attributes
    assert "llm.output_messages.0.message.role" not in span._attributes
    assert "llm.output_messages.0.message.content" not in span._attributes


def test_tools_and_tool_calls_promoted(translator):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.tools": json.dumps(tools),
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city":"NYC"}',
    })

    translator.on_end(span)

    expected_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
            },
        }
    ]
    assert json.loads(span._attributes[RESPAN_SPAN_TOOLS]) == tools
    assert json.loads(span._attributes[RESPAN_SPAN_TOOL_CALLS]) == expected_tool_calls
    assert span._attributes["gen_ai.completion.0.tool_calls"] == expected_tool_calls
    assert span._attributes["llm.request.functions"] == json.dumps(tools)
    assert (
        span._attributes["gen_ai.completion.0.tool_calls.0.function.name"]
        == "get_weather"
    )
    assert "tools" not in span._attributes
    assert "tool_calls" not in span._attributes


def test_indexed_anthropic_tools_promoted_from_bounded_attrs(translator):
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    span = _make_span({})
    span._attributes = BoundedAttributes(
        maxlen=64,
        attributes={
            "openinference.span.kind": "LLM",
            "llm.tools.0.tool.json_schema": json.dumps(tools[0]),
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city":"Tokyo"}',
        },
        immutable=False,
    )

    translator.on_end(span)

    assert isinstance(span._attributes, dict)
    expected_tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]
    assert json.loads(span._attributes[RESPAN_SPAN_TOOLS]) == tools
    assert json.loads(span._attributes[RESPAN_SPAN_TOOL_CALLS]) == expected_tool_calls
    assert span._attributes["gen_ai.completion.0.tool_calls"] == expected_tool_calls
    assert span._attributes["llm.request.functions"] == json.dumps(tools)
    assert "llm.tools.0.tool.json_schema" not in span._attributes
    assert (
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
        not in span._attributes
    )
    assert (
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
        not in span._attributes
    )


def test_legacy_function_call_fields_promoted_as_tool_calls(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.function_call_name": "Glob",
        "llm.output_messages.0.message.function_call_arguments_json": '{"pattern":"*.py"}',
    })

    translator.on_end(span)

    expected_tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "Glob",
                "arguments": '{"pattern":"*.py"}',
            },
        }
    ]
    assert json.loads(span._attributes[RESPAN_SPAN_TOOL_CALLS]) == expected_tool_calls
    assert span._attributes["gen_ai.completion.0.tool_calls"] == expected_tool_calls
    assert "tool_calls" not in span._attributes


def test_input_history_tool_calls_do_not_become_top_level_tool_calls(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "assistant",
        "llm.input_messages.0.message.tool_calls.0.tool_call.id": "call_history",
        "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "lookup_weather",
        "llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city":"Tokyo"}',
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "No tool call in this turn",
    })

    translator.on_end(span)

    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes
    assert span._attributes["gen_ai.prompt.0.tool_calls"] == [
        {
            "id": "call_history",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
            "type": "function",
        }
    ]
