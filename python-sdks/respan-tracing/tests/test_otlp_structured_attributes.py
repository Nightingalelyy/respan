from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind

from respan_sdk.constants.otlp_constants import (
    OTLP_ARRAY_VALUE,
    OTLP_ARRAY_VALUES_KEY,
    OTLP_ATTR_KEY,
    OTLP_ATTR_VALUE,
    OTLP_ATTRIBUTES_KEY,
    OTLP_KVLIST_VALUE,
    OTLP_STRING_VALUE,
)
from respan_tracing.exporters.respan import _span_to_otlp_json
from respan_tracing.testing import InMemorySpanExporter


def _kvlist_to_dict(value: dict) -> dict:
    return {
        item[OTLP_ATTR_KEY]: item[OTLP_ATTR_VALUE]
        for item in value[OTLP_KVLIST_VALUE][OTLP_ARRAY_VALUES_KEY]
    }


def test_otlp_json_serializes_nested_attribute_structures():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("structured-otlp-test")

    with tracer.start_as_current_span("chat request", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.system", "openai")

    readable_span = exporter.get_finished_spans()[0]
    readable_span._attributes = {
        **dict(readable_span.attributes),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "StructuredAnswer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
        },
    }

    otlp_span = _span_to_otlp_json(readable_span)
    attributes = {
        item[OTLP_ATTR_KEY]: item[OTLP_ATTR_VALUE]
        for item in otlp_span[OTLP_ATTRIBUTES_KEY]
    }

    tools_value = attributes["tools"][OTLP_ARRAY_VALUE][OTLP_ARRAY_VALUES_KEY][0]
    tool_entries = _kvlist_to_dict(value=tools_value)
    assert tool_entries["type"][OTLP_STRING_VALUE] == "function"

    function_entries = _kvlist_to_dict(value=tool_entries["function"])
    assert function_entries["name"][OTLP_STRING_VALUE] == "lookup_weather"

    parameters_entries = _kvlist_to_dict(value=function_entries["parameters"])
    assert parameters_entries["type"][OTLP_STRING_VALUE] == "object"

    response_format_entries = _kvlist_to_dict(value=attributes["response_format"])
    assert response_format_entries["type"][OTLP_STRING_VALUE] == "json_schema"

    json_schema_entries = _kvlist_to_dict(value=response_format_entries["json_schema"])
    assert json_schema_entries["name"][OTLP_STRING_VALUE] == "StructuredAnswer"

    schema_entries = _kvlist_to_dict(value=json_schema_entries["schema"])
    assert schema_entries["type"][OTLP_STRING_VALUE] == "object"
