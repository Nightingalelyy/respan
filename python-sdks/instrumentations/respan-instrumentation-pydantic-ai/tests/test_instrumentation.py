import builtins
import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_pydantic_ai import (
    PydanticAIInstrumentor,
    _instrumentation,
    _processor,
)
from respan_instrumentation_pydantic_ai._processor import (
    PydanticAISpanProcessor,
    enrich_pydantic_ai_span,
)
from respan_instrumentation_pydantic_ai._serialization import (
    MAX_ATTRIBUTE_BYTES,
    json_string,
)

_BANNED_ALIASES = {
    SpanAttributes.TRACELOOP_SPAN_KIND,
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "span_workflow_name",
    "input",
    "output",
    "has_tool_calls",
    "parallel_tool_calls",
}


@pytest.fixture(autouse=True)
def _reset_shared_lifecycle():
    PydanticAIInstrumentor._shared_processor = None
    PydanticAIInstrumentor._shared_provider = None
    PydanticAIInstrumentor._processor_refcount = 0
    PydanticAIInstrumentor._global_refcount = 0
    PydanticAIInstrumentor._global_config = None
    PydanticAIInstrumentor._global_agent_class = None
    PydanticAIInstrumentor._global_previous = _instrumentation._UNSET
    PydanticAIInstrumentor._specific_agents = {}
    yield
    PydanticAIInstrumentor._shared_processor = None
    PydanticAIInstrumentor._shared_provider = None
    PydanticAIInstrumentor._processor_refcount = 0
    PydanticAIInstrumentor._global_refcount = 0
    PydanticAIInstrumentor._global_config = None
    PydanticAIInstrumentor._global_agent_class = None
    PydanticAIInstrumentor._global_previous = _instrumentation._UNSET
    PydanticAIInstrumentor._specific_agents = {}


def _assert_otel_safe_attrs(attrs):
    for key, value in attrs.items():
        assert value is not None, key
        if isinstance(value, (list, tuple)):
            assert all(
                isinstance(item, (str, bool, int, float, bytes)) for item in value
            ), key
            continue
        assert isinstance(value, (str, bool, int, float, bytes)), key


def _assert_no_banned_aliases(attrs):
    assert _BANNED_ALIASES.isdisjoint(attrs)
    _assert_otel_safe_attrs(attrs)


def _make_fake_tracer_provider():
    return SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=()),
        add_span_processor=lambda processor: None,
    )


def _install_fake_modules(monkeypatch):
    class FakeInstrumentationSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        _instrument_default = "existing-default"
        instrument_all_calls: ClassVar[list[object]] = []

        @classmethod
        def instrument_all(cls, instrument):
            cls.instrument_all_calls.append(instrument)
            cls._instrument_default = instrument

    pydantic_ai_module = ModuleType("pydantic_ai")
    pydantic_ai_agent_module = ModuleType("pydantic_ai.agent")
    pydantic_ai_agent_module.Agent = FakeAgent
    pydantic_ai_models_module = ModuleType("pydantic_ai.models")
    pydantic_ai_models_instrumented_module = ModuleType(
        "pydantic_ai.models.instrumented"
    )
    pydantic_ai_models_instrumented_module.InstrumentationSettings = (
        FakeInstrumentationSettings
    )

    monkeypatch.setitem(sys.modules, "pydantic_ai", pydantic_ai_module)
    monkeypatch.setitem(sys.modules, "pydantic_ai.agent", pydantic_ai_agent_module)
    monkeypatch.setitem(sys.modules, "pydantic_ai.models", pydantic_ai_models_module)
    monkeypatch.setitem(
        sys.modules,
        "pydantic_ai.models.instrumented",
        pydantic_ai_models_instrumented_module,
    )

    return SimpleNamespace(
        agent_class=FakeAgent,
        instrumentation_settings_class=FakeInstrumentationSettings,
    )


def test_activate_instruments_all_agents_and_restores_previous_global(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_modules(monkeypatch)
    previous_default = fake.agent_class._instrument_default

    instrumentor = PydanticAIInstrumentor(
        include_content=False,
        include_binary_content=False,
    )
    instrumentor.activate()

    assert len(fake.agent_class.instrument_all_calls) == 1
    settings = fake.agent_class.instrument_all_calls[0]
    assert isinstance(settings, fake.instrumentation_settings_class)
    assert settings.kwargs["tracer_provider"] is tracer_provider
    assert settings.kwargs["include_content"] is False
    assert settings.kwargs["include_binary_content"] is False
    assert settings.kwargs["version"] == 5

    active_processors = getattr(
        tracer_provider._active_span_processor, "_span_processors", ()
    )
    assert any(
        isinstance(processor, PydanticAISpanProcessor)
        for processor in active_processors
    )

    instrumentor.deactivate()

    assert fake.agent_class.instrument_all_calls[-1] == previous_default
    assert tracer_provider._active_span_processor._span_processors == ()


def test_activate_specific_agent_restores_existing_agent_instrument(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_modules(monkeypatch)
    agent = SimpleNamespace(instrument="existing-agent-setting")

    instrumentor = PydanticAIInstrumentor(agent=agent, version=4)
    instrumentor.activate()

    assert agent.instrument != "existing-agent-setting"
    assert isinstance(agent.instrument, fake.instrumentation_settings_class)
    assert fake.agent_class.instrument_all_calls == []

    instrumentor.deactivate()

    assert agent.instrument == "existing-agent-setting"
    assert tracer_provider._active_span_processor._span_processors == ()


def test_enrich_pydantic_ai_tool_span_maps_tool_fields():
    span = SimpleNamespace(
        name="execute_tool add",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.tool.name": "add",
            "gen_ai.tool.call.arguments": '{"a":1,"b":2}',
            "gen_ai.tool.call.result": "3",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 7,
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "tool"
    assert span._attributes["respan.entity.log_method"] == "tracing_integration"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "add"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "add",
        "arguments": {"a": 1, "b": 2},
    }
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == 3
    assert SpanAttributes.LLM_REQUEST_MODEL not in span._attributes
    assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in span._attributes
    assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in span._attributes
    assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in span._attributes
    assert "gen_ai.usage.input_tokens" not in span._attributes
    assert "gen_ai.usage.output_tokens" not in span._attributes
    assert "gen_ai.tool.name" not in span._attributes
    _assert_no_banned_aliases(span._attributes)


def test_enrich_pydantic_ai_agent_span_stays_common_only():
    span = SimpleNamespace(
        name="invoke_agent weather",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.agent.name": "weather",
            "model_request_parameters": json.dumps(
                {
                    "output_mode": "native",
                    "output_object": {
                        "name": "WeatherAnswer",
                        "json_schema": {"type": "object"},
                    },
                    "function_tools": [
                        {
                            "name": "lookup_weather",
                            "description": "Look up the weather.",
                            "parameters_json_schema": {"type": "object"},
                        }
                    ],
                }
            ),
            "gen_ai.request.model": "gpt-4o-mini",
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "agent"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert span._attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "weather"
    assert SpanAttributes.LLM_REQUEST_MODEL not in span._attributes
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in span._attributes
    assert "response_format" not in span._attributes
    assert "gen_ai.agent.name" not in span._attributes
    _assert_no_banned_aliases(span._attributes)


def test_enrich_pydantic_ai_chat_span_maps_messages():
    span = SimpleNamespace(
        name="chat completion",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "chat"
    assert span._attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "hi"}
    ]
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "role": "assistant",
        "content": "hello",
    }
    assert span._attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert span._attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "hi"
    assert span._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert span._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "hello"
    _assert_no_banned_aliases(span._attributes)


def test_enrich_pydantic_ai_chat_span_flattens_structured_message_parts():
    span = SimpleNamespace(
        name="chat gemini/gemini-2.5-flash",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "chat",
            "model_request_parameters": json.dumps({"output_mode": "text"}),
            "gen_ai.input.messages": json.dumps(
                [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "content": "You are a helpful assistant.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "content": "What is the capital of France?",
                            }
                        ],
                    },
                ]
            ),
            "gen_ai.output.messages": json.dumps(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "content": "The capital of France is Paris.",
                        }
                    ],
                }
            ),
        },
    )

    enrich_pydantic_ai_span(span)

    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "role": "assistant",
        "content": "The capital of France is Paris.",
    }
    assert span._attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert span._attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == (
        "You are a helpful assistant."
    )
    assert span._attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert span._attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == (
        "What is the capital of France?"
    )
    assert span._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == ("assistant")
    assert span._attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "The capital of France is Paris."
    )
    assert json.loads(span._attributes["response_format"]) == {"type": "text"}
    _assert_no_banned_aliases(span._attributes)


def test_enrich_pydantic_ai_response_operation_maps_as_chat_span():
    span = SimpleNamespace(
        name="openai.responses.create",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "response",
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "chat"
    assert span._attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "hi"}
    ]
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "role": "assistant",
        "content": "hello",
    }
    _assert_no_banned_aliases(span._attributes)


def test_enrich_pydantic_ai_running_tools_span_maps_task_fields():
    span = SimpleNamespace(
        name="running tools",
        _attributes={
            "gen_ai.system": "openai",
            "tools": '["add","multiply"]',
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "task"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "running_tools"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    _assert_no_banned_aliases(span._attributes)


def test_extract_usage_ignores_non_usage_agent_name():
    assert _processor._extract_usage(
        {
            "gen_ai.agent.name": 999,
            "gen_ai.usage.input_tokens": 2,
            "gen_ai.usage.output_tokens": 3,
        }
    ) == (2, 3, 5)


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"pydantic_ai.agent", "pydantic_ai.models.instrumented"}:
            raise ImportError("missing pydantic ai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    instrumentor = PydanticAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate PydanticAI instrumentation" in caplog.text


def test_global_lifecycle_is_shared_and_config_mismatch_is_rejected(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    tracer_provider._active_span_processor._span_processors = ("exporter",)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_modules(monkeypatch)
    first = PydanticAIInstrumentor()
    second = PydanticAIInstrumentor()

    first.activate()
    second.activate()
    assert len(fake.agent_class.instrument_all_calls) == 1
    processors = tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], PydanticAISpanProcessor)
    assert processors[1:] == ("exporter",)

    mismatch = PydanticAIInstrumentor(include_content=False)
    with pytest.raises(ValueError, match="same content and version"):
        mismatch.activate()

    first.deactivate()
    assert (
        fake.agent_class._instrument_default is fake.agent_class.instrument_all_calls[0]
    )
    second.deactivate()
    assert fake.agent_class._instrument_default == "existing-default"
    assert tracer_provider._active_span_processor._span_processors == ("exporter",)


def test_serializer_is_bounded_redacting_and_never_calls_hostile_string_hooks():
    class Hostile:
        def __str__(self):
            raise AssertionError("hostile __str__ called")

        def __repr__(self):
            raise AssertionError("hostile __repr__ called")

    encoded = json_string(
        {
            "api_key": "plain-secret",
            "nested": {"client_secret": "also-secret"},
            "hostile": Hostile(),
            "unicode": "😀" * 10_000,
        }
    )
    assert len(encoded.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES
    assert "plain-secret" not in encoded
    assert "also-secret" not in encoded
    assert json.loads(encoded)


def test_real_pydantic_ai_tool_round_exports_canonical_connected_spans(monkeypatch):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    instrumentor = PydanticAIInstrumentor()
    instrumentor.activate()

    agent = Agent(TestModel(), name="calculator", instructions="Use the add tool.")

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    result = agent.run_sync("Use add for 15 plus 27.")
    assert result.output == '{"add":0}'
    assert tracer_provider.force_flush()

    spans = list(exporter.get_finished_spans())
    assert len(spans) == 4
    assert len({span.context.span_id for span in spans}) == 4
    agent_span = next(span for span in spans if span.name == "invoke_agent calculator")
    chats = [span for span in spans if span.name == "chat test"]
    tool_span = next(span for span in spans if span.name == "execute_tool add")
    assert len(chats) == 2
    assert all(span.parent.span_id == agent_span.context.span_id for span in chats)
    assert tool_span.parent.span_id == agent_span.context.span_id

    agent_attrs = agent_span.attributes
    assert agent_attrs["respan.entity.log_type"] == "agent"
    assert agent_attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert json.loads(agent_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])[0] == {
        "content": "Use add for 15 plus 27.",
        "role": "user",
    }
    assert json.loads(agent_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {"add": 0}
    assert SpanAttributes.LLM_REQUEST_MODEL not in agent_attrs
    assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in agent_attrs

    first_chat, second_chat = sorted(chats, key=lambda span: span.start_time)
    first_attrs = first_chat.attributes
    first_calls = json.loads(
        first_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert first_calls == [
        {
            "function": {
                "arguments": '{"a":0,"b":0}',
                "name": "add",
            },
            "id": "pyd_ai_tool_call_id__add",
            "type": "function",
        }
    ]
    assert (
        json.loads(first_attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
            "name"
        ]
        == "add"
    )
    assert first_attrs["gen_ai.usage.input_tokens"] > 0
    assert first_attrs["gen_ai.usage.output_tokens"] > 0
    assert (
        first_attrs["gen_ai.usage.input_tokens"]
        == first_attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
    )
    assert (
        first_attrs["gen_ai.usage.output_tokens"]
        == first_attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
    )

    second_attrs = second_chat.attributes
    assert second_attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
    assert (
        json.loads(second_attrs[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls"])
        == first_calls
    )
    assert second_attrs[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "tool"
    assert json.loads(second_attrs[f"{SpanAttributes.LLM_PROMPTS}.2.content"]) == {
        "name": "add",
        "result": 0,
        "tool_call_id": "pyd_ai_tool_call_id__add",
    }
    assert json.loads(tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"a": 0, "b": 0},
        "name": "add",
    }
    _assert_no_banned_aliases(agent_attrs)
    for span in (*chats, tool_span):
        _assert_no_banned_aliases(span.attributes)

    instrumentor.deactivate()
    tracer_provider.shutdown()
