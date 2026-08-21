import json

from opentelemetry.semconv_ai import SpanAttributes
from respan_tracing.decorators.base import (
    _handle_span_input,
    _handle_span_output,
    _should_send_prompts,
)


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class ChromaLikeResult:
    def model_dump(self):
        return {
            "ids": ["doc-1", "doc-2"],
            "embeddings": ArrayLike(),
        }


class ArrayLike:
    def tolist(self):
        return [[0.1, 0.2], [0.3, 0.4]]


def test_span_output_serializes_nested_sdk_and_array_values(monkeypatch):
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    span = RecordingSpan()

    _handle_span_output(
        span,
        {
            "count": 2,
            "peek": ChromaLikeResult(),
        },
    )

    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "count": 2,
        "peek": {
            "ids": ["doc-1", "doc-2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        },
    }


def test_trace_content_environment_disables_decorator_output(monkeypatch):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    span = RecordingSpan()

    assert _should_send_prompts() is False
    _handle_span_output(span, {"secret": "must not be recorded"})

    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_span_input_bounds_runtime_clients_without_serializing_private_state(
    monkeypatch,
):
    class Client:
        def __init__(self):
            self.api_key = "must-not-leak"

    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    span = RecordingSpan()

    _handle_span_input(span, (Client(), "invoice extraction"), {})

    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "args": ["<Client>", "invoice extraction"],
        "kwargs": {},
    }
    assert "must-not-leak" not in span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
