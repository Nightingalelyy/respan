"""Unit tests for OpenInferenceTranslator.

Uses a lightweight mock span (just needs ``_attributes`` and ``name``)
so we don't depend on any OpenInference packages at test time.
"""

import json
from types import SimpleNamespace

import pytest

from respan_instrumentation_openinference._translator import OpenInferenceTranslator


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


# ------------------------------------------------------------------
# 13. noisy raw OI attrs are removed after translation
# ------------------------------------------------------------------

def test_redundant_oi_attrs_removed_after_translation(translator):
    span = _make_span({
        "openinference.span.kind": "LLM",
        "input.value": "hello world",
        "input.mime_type": "text/plain",
        "output.value": "goodbye world",
        "output.mime_type": "text/plain",
        "llm.model_name": "gpt-4o",
        "llm.system": "OpenAI",
        "llm.provider": "OpenAI",
        "llm.invocation_parameters": json.dumps({"temperature": 0.3}),
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "Hello!",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "Goodbye!",
        "llm.token_count.prompt": 10,
        "llm.token_count.completion": 4,
        "llm.token_count.total": 14,
    })

    translator.on_end(span)

    assert span._attributes["traceloop.entity.input"] == "hello world"
    assert span._attributes["traceloop.entity.output"] == "goodbye world"
    assert span._attributes["gen_ai.request.model"] == "gpt-4o"
    assert span._attributes["gen_ai.system"] == "openai"
    assert span._attributes["gen_ai.provider.name"] == "openai"
    assert span._attributes["gen_ai.prompt.0.content"] == "Hello!"
    assert span._attributes["gen_ai.completion.0.content"] == "Goodbye!"
    assert span._attributes["gen_ai.usage.prompt_tokens"] == 10
    assert span._attributes["gen_ai.usage.completion_tokens"] == 4

    assert "input.value" not in span._attributes
    assert "input.mime_type" not in span._attributes
    assert "output.value" not in span._attributes
    assert "output.mime_type" not in span._attributes
    assert "openinference.span.kind" not in span._attributes
    assert "llm.model_name" not in span._attributes
    assert "llm.system" not in span._attributes
    assert "llm.provider" not in span._attributes
    assert "llm.invocation_parameters" not in span._attributes
    assert "llm.input_messages.0.message.role" not in span._attributes
    assert "llm.input_messages.0.message.content" not in span._attributes
    assert "llm.output_messages.0.message.role" not in span._attributes
    assert "llm.output_messages.0.message.content" not in span._attributes
    assert "llm.token_count.prompt" not in span._attributes
    assert "llm.token_count.completion" not in span._attributes
    assert "llm.token_count.total" not in span._attributes
