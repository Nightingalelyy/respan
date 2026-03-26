"""Unit tests for OpenAI Agents OTEL emitter tool attributes."""

import json
from types import SimpleNamespace

from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_openai_agents import _otel_emitter


def _make_span_item() -> SimpleNamespace:
    return SimpleNamespace(
        trace_id="trace_123",
        span_id="span_456",
        parent_id=None,
        started_at=None,
        ended_at=None,
        error=None,
    )


def _capture_attrs(monkeypatch):
    captured = {}

    def _fake_build_readable_span(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(_otel_emitter, "build_readable_span", _fake_build_readable_span)
    monkeypatch.setattr(_otel_emitter, "inject_span", lambda span: None)
    return captured


def test_emit_response_serializes_namespaced_tool_attrs(monkeypatch):
    captured = _capture_attrs(monkeypatch)
    response = SimpleNamespace(
        model="gpt-4o",
        output=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_weather",
                "arguments": '{"city":"NYC"}',
            }
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Look up the weather.",
                "parameters": {"type": "object"},
            }
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4),
    )
    span_data = SimpleNamespace(response=response, input="What is the weather in NYC?")

    _otel_emitter.emit_response(_make_span_item(), span_data)

    attrs = captured["attributes"]
    assert attrs[RESPAN_SPAN_TOOL_CALLS] == json.dumps(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "arguments": '{"city":"NYC"}',
                },
            }
        ],
        default=str,
    )
    assert attrs[RESPAN_SPAN_TOOLS] == json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up the weather.",
                    "parameters": {"type": "object"},
                },
            }
        ],
        default=str,
    )
    assert "tool_calls" not in attrs
    assert "tools" not in attrs


def test_emit_generation_extracts_tool_calls(monkeypatch):
    captured = _capture_attrs(monkeypatch)
    span_data = SimpleNamespace(
        input=[{"role": "user", "content": "Use the tool"}],
        output=[
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "search_docs",
                "arguments": '{"query":"otel"}',
            }
        ],
        model="gpt-4o",
        usage={"prompt_tokens": 8, "completion_tokens": 2},
    )

    _otel_emitter.emit_generation(_make_span_item(), span_data)

    attrs = captured["attributes"]
    assert attrs[RESPAN_SPAN_TOOL_CALLS] == json.dumps(
        [
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "arguments": '{"query":"otel"}',
                },
            }
        ],
        default=str,
    )
    assert "tool_calls" not in attrs
