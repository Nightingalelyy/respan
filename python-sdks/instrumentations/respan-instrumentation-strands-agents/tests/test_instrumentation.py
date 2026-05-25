import json
import logging
import os
import sys
from types import ModuleType, SimpleNamespace

from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_strands_agents import StrandsAgentsInstrumentor
from respan_instrumentation_strands_agents import _instrumentation
from respan_instrumentation_strands_agents._constants import (
    STRANDS_EVENT_CHOICE,
    STRANDS_EVENT_TOOL_MESSAGE,
    STRANDS_EVENT_USER_MESSAGE,
    STRANDS_OPERATION_CHAT,
    STRANDS_OPERATION_EXECUTE_TOOL,
    STRANDS_OPERATION_INVOKE_AGENT,
    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN,
    STRANDS_SYSTEM_NAME,
    STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
    STRANDS_TOOL_CALL_ID_ATTR,
    STRANDS_TOOL_DEFINITIONS_ATTR,
    STRANDS_TOOL_DESCRIPTION_ATTR,
    STRANDS_TOOL_JSON_SCHEMA_ATTR,
    STRANDS_USAGE_INPUT_TOKENS_ATTR,
    STRANDS_USAGE_OUTPUT_TOKENS_ATTR,
)
from respan_instrumentation_strands_agents._processor import (
    StrandsAgentsSpanProcessor,
    enrich_strands_agents_span,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


def _make_fake_tracer_provider():
    tracer_provider = SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=()),
        added_processors=[],
        get_tracer=lambda service_name: f"tracer:{service_name}",
    )

    def add_span_processor(processor):
        tracer_provider.added_processors.append(processor)

    tracer_provider.add_span_processor = add_span_processor
    return tracer_provider


def _install_fake_strands_modules(monkeypatch):
    tracer_instance = SimpleNamespace(
        service_name="strands.telemetry.tracer",
        tracer_provider=None,
        tracer=None,
        _include_tool_definitions=False,
    )

    def _parse_semconv_opt_in():
        opt_in = os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "")
        return {value.strip() for value in opt_in.split(",") if value.strip()}

    tracer_instance._parse_semconv_opt_in = _parse_semconv_opt_in

    strands_module = ModuleType("strands")
    strands_module.__path__ = []
    telemetry_module = ModuleType("strands.telemetry")
    telemetry_module.__path__ = []
    tracer_module = ModuleType("strands.telemetry.tracer")
    tracer_module._tracer_instance = tracer_instance

    monkeypatch.setitem(sys.modules, "strands", strands_module)
    monkeypatch.setitem(sys.modules, "strands.telemetry", telemetry_module)
    monkeypatch.setitem(sys.modules, "strands.telemetry.tracer", tracer_module)
    return tracer_instance


def _make_span(name, attributes=None, events=None):
    attrs = dict(attributes or {})
    return SimpleNamespace(
        name=name,
        _attributes=attrs,
        attributes=attrs,
        events=tuple(events or ()),
    )


def _event(name, attributes):
    return SimpleNamespace(name=name, attributes=attributes)


def _assert_no_off_contract_aliases(attrs):
    for key in (
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
        *STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
    ):
        assert key not in attrs


def test_package_exports_instrumentor():
    assert StrandsAgentsInstrumentor is _instrumentation.StrandsAgentsInstrumentor
    assert StrandsAgentsInstrumentor.name == "strands-agents"


def test_activate_registers_processor_first_and_refreshes_existing_tracer(
    monkeypatch,
):
    tracer_provider = _make_fake_tracer_provider()
    existing_processor = object()
    tracer_provider._active_span_processor._span_processors = (existing_processor,)
    tracer_instance = _install_fake_strands_modules(monkeypatch)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)

    instrumentor = StrandsAgentsInstrumentor()
    instrumentor.activate()

    processors = tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], StrandsAgentsSpanProcessor)
    assert processors[1] is existing_processor
    assert tracer_instance.tracer_provider is tracer_provider
    assert tracer_instance.tracer == "tracer:strands.telemetry.tracer"
    assert tracer_instance._include_tool_definitions is True
    assert (
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"]
        == STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN
    )

    instrumentor.deactivate()

    assert tracer_provider._active_span_processor._span_processors == (
        existing_processor,
    )
    assert "OTEL_SEMCONV_STABILITY_OPT_IN" not in os.environ


def test_activate_logs_warning_when_dependency_missing(monkeypatch, caplog):
    def raise_import_error(module_name):
        if module_name == "strands.telemetry.tracer":
            raise ImportError(module_name)
        return __import__(module_name)

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        raise_import_error,
    )
    instrumentor = StrandsAgentsInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Strands Agents instrumentation" in caplog.text


def test_enrich_agent_span_maps_common_fields_and_tool_definitions():
    span = _make_span(
        name=f"{STRANDS_OPERATION_INVOKE_AGENT} WeatherAgent",
        attributes={
            GEN_AI_SYSTEM: STRANDS_SYSTEM_NAME,
            GEN_AI_OPERATION_NAME: STRANDS_OPERATION_INVOKE_AGENT,
            GEN_AI_AGENT_NAME: "WeatherAgent",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
            STRANDS_TOOL_DEFINITIONS_ATTR: json.dumps(
                [
                    {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "inputSchema": {"type": "object"},
                    }
                ]
            ),
            STRANDS_USAGE_INPUT_TOKENS_ATTR: 30,
            STRANDS_USAGE_OUTPUT_TOKENS_ATTR: 6,
        },
        events=[
            _event(
                STRANDS_EVENT_USER_MESSAGE,
                {"content": json.dumps([{"text": "weather in Seattle"}])},
            ),
            _event(STRANDS_EVENT_CHOICE, {"message": "It is sunny."}),
        ],
    )

    enrich_strands_agents_span(span)

    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert attrs[RESPAN_LOG_METHOD] == LogMethodChoices.TRACING_INTEGRATION.value
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "WeatherAgent"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "WeatherAgent"
    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "WeatherAgent"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "weather in Seattle"}
    ]
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        {"role": "assistant", "content": "It is sunny."}
    ]
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert SpanAttributes.LLM_REQUEST_MODEL not in attrs
    assert GEN_AI_SYSTEM not in attrs
    assert SpanAttributes.LLM_REQUEST_TYPE not in attrs
    assert STRANDS_USAGE_INPUT_TOKENS_ATTR not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert GEN_AI_AGENT_NAME not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_enrich_chat_span_maps_messages_usage_and_tool_calls():
    span = _make_span(
        name=STRANDS_OPERATION_CHAT,
        attributes={
            GEN_AI_SYSTEM: STRANDS_SYSTEM_NAME,
            GEN_AI_OPERATION_NAME: STRANDS_OPERATION_CHAT,
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
            STRANDS_USAGE_INPUT_TOKENS_ATTR: 12,
            STRANDS_USAGE_OUTPUT_TOKENS_ATTR: 4,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS: 16,
        },
        events=[
            _event(
                STRANDS_EVENT_USER_MESSAGE,
                {"content": json.dumps([{"text": "use the tool"}])},
            ),
            _event(
                STRANDS_EVENT_CHOICE,
                {
                    "message": json.dumps(
                        [
                            {
                                "toolUse": {
                                    "toolUseId": "tool_1",
                                    "name": "lookup",
                                    "input": {"query": "tracing"},
                                }
                            }
                        ]
                    )
                },
            ),
        ],
    )

    enrich_strands_agents_span(span)

    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 16
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "use the tool"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "id": "tool_1",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"query": "tracing"}',
            },
        }
    ]
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert GEN_AI_OPERATION_NAME not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_enrich_tool_span_maps_input_and_output():
    span = _make_span(
        name=f"{STRANDS_OPERATION_EXECUTE_TOOL} get_weather",
        attributes={
            GEN_AI_SYSTEM: STRANDS_SYSTEM_NAME,
            GEN_AI_OPERATION_NAME: STRANDS_OPERATION_EXECUTE_TOOL,
            GEN_AI_TOOL_NAME: "get_weather",
            STRANDS_TOOL_CALL_ID_ATTR: "tool_1",
            STRANDS_TOOL_DESCRIPTION_ATTR: "Get weather.",
            STRANDS_TOOL_JSON_SCHEMA_ATTR: json.dumps({"type": "object"}),
        },
        events=[
            _event(STRANDS_EVENT_TOOL_MESSAGE, {"content": '{"city":"Seattle"}'}),
            _event(
                STRANDS_EVENT_CHOICE,
                {"message": json.dumps([{"text": "Sunny and 72F."}])},
            ),
        ],
    )

    enrich_strands_agents_span(span)

    attrs = span._attributes
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "get_weather"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "get_weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "get_weather",
        "id": "tool_1",
        "arguments": {"city": "Seattle"},
    }
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "Sunny and 72F."
    assert GEN_AI_TOOL_NAME not in attrs
    assert GEN_AI_SYSTEM not in attrs
    assert SpanAttributes.LLM_REQUEST_TYPE not in attrs
    assert STRANDS_TOOL_CALL_ID_ATTR not in attrs
    assert STRANDS_TOOL_DESCRIPTION_ATTR not in attrs
    assert STRANDS_TOOL_JSON_SCHEMA_ATTR not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    _assert_no_off_contract_aliases(attrs)


def test_non_strands_span_is_unchanged():
    span = _make_span(
        name="other",
        attributes={
            GEN_AI_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
        },
    )

    enrich_strands_agents_span(span)

    assert span._attributes == {
        GEN_AI_SYSTEM: "openai",
        SpanAttributes.LLM_REQUEST_MODEL: "gpt-4o-mini",
    }
