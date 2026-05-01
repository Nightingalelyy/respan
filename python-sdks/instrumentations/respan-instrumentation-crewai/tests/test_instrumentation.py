import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from openinference.semconv import trace

from respan_instrumentation_crewai import CrewAIInstrumentor, CrewAITranslator
from respan_instrumentation_crewai import _instrumentation
from respan_instrumentation_crewai._instrumentation import (
    CREATE_LLM_SPANS_KWARG,
    OPENINFERENCE_CREWAI_MODULE,
    RESPAN_OPENINFERENCE_MODULE,
    USE_EVENT_LISTENER_KWARG,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_MODEL,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer


def _install_fake_openinference_modules(monkeypatch):
    class FakeCrewAIInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created = []

        def __init__(self, instrumentor_class, **kwargs):
            self.instrumentor_class = instrumentor_class
            self.kwargs = kwargs
            self.is_activated = False
            self.is_deactivated = False
            self.__class__.created.append(self)

        def activate(self):
            self.is_activated = True

        def deactivate(self):
            self.is_deactivated = True

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_crewai_module = ModuleType(OPENINFERENCE_CREWAI_MODULE)
    openinference_crewai_module.CrewAIInstrumentor = FakeCrewAIInstrumentor
    openinference_instrumentation_module.crewai = openinference_crewai_module

    respan_openinference_module = ModuleType(RESPAN_OPENINFERENCE_MODULE)
    respan_openinference_module.OpenInferenceInstrumentor = FakeOpenInferenceInstrumentor

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_CREWAI_MODULE,
        openinference_crewai_module,
    )
    monkeypatch.setitem(
        sys.modules,
        RESPAN_OPENINFERENCE_MODULE,
        respan_openinference_module,
    )

    return SimpleNamespace(
        crewai_instrumentor_class=FakeCrewAIInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_crewai_defaults(monkeypatch):
    fake = _install_fake_openinference_modules(monkeypatch)

    instrumentor = CrewAIInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.crewai_instrumentor_class
    assert delegate.kwargs == {
        USE_EVENT_LISTENER_KWARG: True,
        CREATE_LLM_SPANS_KWARG: True,
    }
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_openinference_modules(monkeypatch)

    instrumentor = CrewAIInstrumentor(
        use_event_listener=False,
        create_llm_spans=False,
        trace_content=False,
    )
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {
        USE_EVENT_LISTENER_KWARG: False,
        CREATE_LLM_SPANS_KWARG: False,
        "trace_content": False,
    }


def test_activate_cleans_up_delegate_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_openinference_modules(monkeypatch)

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = CrewAIInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate CrewAI instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_openinference_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = CrewAIInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "CrewAI instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    original_import_module = _instrumentation.importlib.import_module

    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_CREWAI_MODULE:
            raise ImportError(module_name)
        return original_import_module(module_name)

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = CrewAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate CrewAI instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_crewai_translator_maps_openinference_span_to_respan_shape():
    span = SimpleNamespace(
        name="Crew.kickoff",
        _attributes={
            trace.SpanAttributes.OPENINFERENCE_SPAN_KIND: "AGENT",
            trace.SpanAttributes.AGENT_NAME: "Research crew",
            trace.SpanAttributes.INPUT_VALUE: {"topic": "weather"},
            trace.SpanAttributes.OUTPUT_VALUE: {"result": "done"},
        },
    )

    CrewAITranslator().on_end(span)

    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "agent"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "Research crew"
    assert (
        span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
        == '{"topic": "weather"}'
    )
    assert (
        span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        == '{"result": "done"}'
    )
    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert trace.SpanAttributes.OPENINFERENCE_SPAN_KIND not in span._attributes


def test_crewai_translator_maps_llm_model_and_usage():
    span = SimpleNamespace(
        name="CrewAI LLM",
        _attributes={
            trace.SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            trace.SpanAttributes.LLM_MODEL_NAME: "gpt-4o-mini",
            trace.SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 11,
            trace.SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 7,
            trace.SpanAttributes.LLM_INVOCATION_PARAMETERS: '{"temperature": 0.2}',
        },
    )

    CrewAITranslator().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert span._attributes[LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert span._attributes[LLM_USAGE_PROMPT_TOKENS] == 11
    assert span._attributes[LLM_USAGE_COMPLETION_TOKENS] == 7
    assert span._attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.2
