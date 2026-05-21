import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_pipecat import PipecatInstrumentor
from respan_instrumentation_pipecat import _instrumentation
from respan_instrumentation_pipecat._instrumentation import (
    OPENINFERENCE_PIPECAT_MODULE,
    OPENINFERENCE_PIPECAT_OBSERVER_MODULE,
)
from respan_instrumentation_pipecat._translator import PipecatOpenInferenceTranslator
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TOOL,
    LOG_TYPE_TRANSCRIPTION,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer

_OFF_CONTRACT_ALIAS_ATTRS = {
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
}


class FakeTracerProvider:
    def __init__(self):
        self._active_span_processor = SimpleNamespace(_span_processors=("exporter",))
        self.added_processors = []

    def add_span_processor(self, processor):
        self.added_processors.append(processor)


def _install_fake_modules(monkeypatch, *, activate_raises=False):
    class FakeObserverContext:
        pass

    class FakePipecatInstrumentor:
        created = []

        def __init__(self):
            self.instrument_kwargs = None
            self.uninstrument_called = False
            self.__class__.created.append(self)

        def instrument(self, **kwargs):
            self.instrument_kwargs = kwargs
            if activate_raises:
                raise RuntimeError("boom")

        def uninstrument(self):
            self.uninstrument_called = True

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_pipecat_module = ModuleType(OPENINFERENCE_PIPECAT_MODULE)
    openinference_pipecat_module.PipecatInstrumentor = FakePipecatInstrumentor
    openinference_pipecat_observer_module = ModuleType(
        OPENINFERENCE_PIPECAT_OBSERVER_MODULE
    )
    openinference_pipecat_observer_module.Context = FakeObserverContext
    openinference_instrumentation_module.pipecat = openinference_pipecat_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_PIPECAT_MODULE,
        openinference_pipecat_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_PIPECAT_OBSERVER_MODULE,
        openinference_pipecat_observer_module,
    )

    tracer_provider = FakeTracerProvider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    return SimpleNamespace(
        pipecat_instrumentor_class=FakePipecatInstrumentor,
        observer_context_class=FakeObserverContext,
        observer_module=openinference_pipecat_observer_module,
        tracer_provider=tracer_provider,
    )


def _make_span(attrs: dict, name: str = "pipecat.llm"):
    span = SimpleNamespace()
    span._attributes = dict(attrs)
    span.name = name
    return span


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_pipecat_defaults(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = PipecatInstrumentor()
    instrumentor.activate()

    upstream = fake.pipecat_instrumentor_class.created[0]
    assert upstream.instrument_kwargs == {
        "tracer_provider": fake.tracer_provider,
    }
    processors = fake.tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], PipecatOpenInferenceTranslator)
    assert processors[1:] == ("exporter",)
    assert fake.observer_module.Context is _instrumentation.context_api.get_current
    context = _instrumentation.context_api.set_value("respan-test-key", "active")
    token = _instrumentation.context_api.attach(context)
    try:
        assert (
            _instrumentation.context_api.get_value(
                "respan-test-key",
                context=fake.observer_module.Context(),
            )
            == "active"
        )
    finally:
        _instrumentation.context_api.detach(token)
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert upstream.uninstrument_called is True
    assert fake.observer_module.Context is fake.observer_context_class
    assert fake.tracer_provider._active_span_processor._span_processors == ("exporter",)
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = PipecatInstrumentor(
        debug_log_filename="pipecat-frames.log",
        config=object(),
    )
    instrumentor.activate()

    upstream = fake.pipecat_instrumentor_class.created[0]
    assert upstream.instrument_kwargs == {
        "tracer_provider": fake.tracer_provider,
        "debug_log_filename": "pipecat-frames.log",
        "config": instrumentor._instrumentor_kwargs["config"],
    }


def test_activate_cleans_up_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch, activate_raises=True)

    instrumentor = PipecatInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    upstream = fake.pipecat_instrumentor_class.created[0]
    assert upstream.uninstrument_called is True
    assert fake.observer_module.Context is fake.observer_context_class
    assert fake.tracer_provider._active_span_processor._span_processors == ("exporter",)
    assert instrumentor._instrumentor is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate Pipecat instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = PipecatInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.pipecat_instrumentor_class.created == []
    assert fake.tracer_provider._active_span_processor._span_processors == ("exporter",)
    assert instrumentor._is_instrumented is False
    assert (
        "Pipecat instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_PIPECAT_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = PipecatInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Pipecat instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_translator_maps_pipecat_turn_span():
    translator = PipecatOpenInferenceTranslator()
    span = _make_span(
        {
            "openinference.span.kind": "CHAIN",
            "session.id": "session-1",
            "input.value": "hello",
            "output.value": "world",
        },
        name="pipecat.conversation.turn",
    )

    translator.on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_WORKFLOW
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in span._attributes
    assert (
        span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_NAME]
        == "pipecat.conversation.turn"
    )
    assert span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] == "hello"
    assert span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "world"
    assert span._attributes[RESPAN_SESSION_ID] == "session-1"
    assert "openinference.span.kind" not in span._attributes


def test_translator_maps_pipecat_llm_span():
    translator = PipecatOpenInferenceTranslator()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "service.type": "llm",
            "llm.model_name": "unknown",
            "gen_ai.request.model": "unknown",
            "llm.provider": "openai",
            "llm.token_count.prompt": 7,
            "llm.token_count.completion": 3,
            "llm.token_count.total": 10,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Trace this.",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Done.",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call-1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
            "tools.definitions": json.dumps(tools),
            "metadata": json.dumps({"model": "gpt-4.1-nano"}),
        }
    )

    translator.on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in span._attributes
    assert span._attributes[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span._attributes[TLSpanAttributes.LLM_REQUEST_MODEL] == "gpt-4.1-nano"
    assert span._attributes[TLSpanAttributes.LLM_SYSTEM] == "openai"
    assert span._attributes[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert span._attributes[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
    assert span._attributes[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 10
    assert (
        span._attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == "Trace this."
    )
    assert span._attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == "Done."
    assert json.loads(
        span._attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls"]
    ) == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    assert json.loads(span._attributes[TLSpanAttributes.LLM_REQUEST_FUNCTIONS]) == tools
    for attr_name in _OFF_CONTRACT_ALIAS_ATTRS:
        assert attr_name not in span._attributes
    assert "llm.output_messages.0.message.content" not in span._attributes
    assert "tools.definitions" not in span._attributes


def test_translator_maps_stt_and_tts_service_types():
    translator = PipecatOpenInferenceTranslator()
    stt_span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "service.type": "stt",
            "input.value": "spoken text",
        }
    )
    tts_span = _make_span(
        {
            "openinference.span.kind": "LLM",
            "service.type": "tts",
            "output.value": "spoken response",
        }
    )

    translator.on_end(stt_span)
    translator.on_end(tts_span)

    assert stt_span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TRANSCRIPTION
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in stt_span._attributes
    assert (
        stt_span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] == "spoken text"
    )
    assert TLSpanAttributes.LLM_REQUEST_TYPE not in stt_span._attributes
    assert tts_span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_SPEECH
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in tts_span._attributes
    assert (
        tts_span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        == "spoken response"
    )
    assert TLSpanAttributes.LLM_REQUEST_TYPE not in tts_span._attributes


def test_translator_maps_tool_span():
    translator = PipecatOpenInferenceTranslator()
    span = _make_span(
        {
            "openinference.span.kind": "TOOL",
            "tool.name": "lookup",
            "tool.parameters": '{"city":"Tokyo"}',
            "tool.result": '{"weather":"sunny"}',
        },
        name="pipecat.tool.lookup",
    )

    translator.on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in span._attributes
    assert (
        span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_NAME]
        == "pipecat.tool.lookup"
    )
    assert (
        span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] == '{"city":"Tokyo"}'
    )
    assert (
        span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        == '{"weather":"sunny"}'
    )
    for attr_name in _OFF_CONTRACT_ALIAS_ATTRS:
        assert attr_name not in span._attributes
    assert "tool.name" not in span._attributes
    assert "tool.parameters" not in span._attributes
    assert "tool.result" not in span._attributes
