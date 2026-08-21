import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_instrumentation_smolagents import SmolagentsInstrumentor, _instrumentation
from respan_instrumentation_smolagents._constants import (
    GEN_AI_COMPLETION_CONTENT_ATTR,
    GEN_AI_COMPLETION_ROLE_ATTR,
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
    LLM_REQUEST_FUNCTIONS_ATTR,
    OPENINFERENCE_INPUT_MESSAGES_ATTR,
    OPENINFERENCE_INSTRUMENTATION_MODULE,
    OPENINFERENCE_MESSAGE_CONTENT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
    OPENINFERENCE_MESSAGE_CONTENTS_ATTR,
    OPENINFERENCE_MESSAGE_ROLE_ATTR,
    OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
    OPENINFERENCE_SMOLAGENTS_MODULE,
    OTEL_SCOPE_NAME,
    SMOLAGENTS_FINAL_ANSWER_ARGUMENT,
    SMOLAGENTS_FINAL_ANSWER_TOOL_NAME,
    SMOLAGENTS_TOOL_NAME_HINT,
    SPAN_ALIAS_COMPLETION_TOKENS,
    SPAN_ALIAS_MODEL,
    SPAN_ALIAS_PROMPT_TOKENS,
    SPAN_ALIAS_TOOL_CALLS,
    SPAN_ALIAS_TOOLS,
    SPAN_ALIAS_TOTAL_REQUEST_TOKENS,
    TOOL_CALL_FUNCTION_ARGUMENTS_FIELD,
    TOOL_CALL_FUNCTION_FIELD,
    TOOL_CALL_FUNCTION_NAME_FIELD,
)
from respan_instrumentation_smolagents._processor import (
    SmolagentsSpanContentProcessor,
    SmolagentsSpanContractProcessor,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer

NON_SMOLAGENTS_SCOPE_NAME = f"{OPENINFERENCE_INSTRUMENTATION_MODULE}.crewai"


def _install_fake_modules(monkeypatch):
    class FakeSmolagentsInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created: ClassVar[list["FakeOpenInferenceInstrumentor"]] = []

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
    openinference_instrumentation_module = ModuleType(
        OPENINFERENCE_INSTRUMENTATION_MODULE
    )
    openinference_smolagents_module = ModuleType(OPENINFERENCE_SMOLAGENTS_MODULE)
    openinference_smolagents_module.SmolagentsInstrumentor = FakeSmolagentsInstrumentor
    openinference_instrumentation_module.smolagents = openinference_smolagents_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_INSTRUMENTATION_MODULE,
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_SMOLAGENTS_MODULE,
        openinference_smolagents_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        smolagents_instrumentor_class=FakeSmolagentsInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


def _make_fake_tracer_provider(processors=()):
    return SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=processors),
        add_span_processor=lambda processor: None,
    )


def _oi_message_attr(prefix: str, index: int, attr: str) -> str:
    return f"{prefix}.{index}.{attr}"


def _oi_message_content_attr(
    prefix: str,
    message_index: int,
    content_index: int,
    attr: str,
) -> str:
    return (
        f"{prefix}.{message_index}.{OPENINFERENCE_MESSAGE_CONTENTS_ATTR}."
        f"{content_index}.{attr}"
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    _instrumentation._RUNTIME_COUNT = 0
    _instrumentation._RUNTIME_CONFIG = None
    _instrumentation._RUNTIME_DELEGATE = None
    _instrumentation._RUNTIME_CONTENT_PROCESSOR = None
    _instrumentation._RUNTIME_CONTRACT_PROCESSOR = None
    RespanTracer.reset_instance()
    yield
    _instrumentation._RUNTIME_COUNT = 0
    _instrumentation._RUNTIME_CONFIG = None
    _instrumentation._RUNTIME_DELEGATE = None
    _instrumentation._RUNTIME_CONTENT_PROCESSOR = None
    _instrumentation._RUNTIME_CONTRACT_PROCESSOR = None
    RespanTracer.reset_instance()


def test_activate_uses_openinference_smolagents(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.smolagents_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor(trace_content=False)
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {"trace_content": False}


def test_activate_is_idempotent(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert len(fake.openinference_instrumentor_class.created) == 1


def test_multiple_instances_share_delegate_and_processors_until_final_release(
    monkeypatch,
):
    fake = _install_fake_modules(monkeypatch)

    class FakeOpenInferenceTranslator:
        pass

    translator = FakeOpenInferenceTranslator()
    tracer_provider = _make_fake_tracer_provider((translator, "exporter"))
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceTranslator",
        FakeOpenInferenceTranslator,
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )
    first = SmolagentsInstrumentor()
    second = SmolagentsInstrumentor()

    first.activate()
    second.activate()
    assert len(fake.openinference_instrumentor_class.created) == 1
    assert (
        sum(
            isinstance(item, SmolagentsSpanContractProcessor)
            for item in tracer_provider._active_span_processor._span_processors
        )
        == 1
    )

    first.deactivate()
    assert fake.openinference_instrumentor_class.created[0].is_deactivated is False
    second.deactivate()
    assert fake.openinference_instrumentor_class.created[0].is_deactivated is True
    assert tracer_provider._active_span_processor._span_processors == (
        translator,
        "exporter",
    )


def test_activate_cleans_up_delegate_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = SmolagentsInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate smolagents instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = SmolagentsInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "smolagents instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_SMOLAGENTS_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = SmolagentsInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate smolagents instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_activate_registers_content_and_contract_processors_around_translator(
    monkeypatch,
):
    fake = _install_fake_modules(monkeypatch)

    class FakeOpenInferenceTranslator:
        pass

    translator = FakeOpenInferenceTranslator()
    tracer_provider = _make_fake_tracer_provider(
        processors=(translator, "exporter"),
    )
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceTranslator",
        FakeOpenInferenceTranslator,
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = SmolagentsInstrumentor()
    instrumentor.activate()

    processors = tracer_provider._active_span_processor._span_processors
    assert isinstance(processors[0], SmolagentsSpanContentProcessor)
    assert processors[1] is translator
    assert isinstance(processors[2], SmolagentsSpanContractProcessor)
    assert processors[3] == "exporter"
    assert fake.openinference_instrumentor_class.created[0].is_activated is True

    instrumentor.deactivate()

    processors = tracer_provider._active_span_processor._span_processors
    assert processors == (translator, "exporter")


def test_content_processor_flattens_openinference_message_content():
    processor = SmolagentsSpanContentProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            _oi_message_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_ROLE_ATTR,
            ): "user",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
            ): "text",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
            ): "hello",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                1,
                OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
            ): "text",
            _oi_message_content_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                1,
                OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
            ): "world",
            _oi_message_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_ROLE_ATTR,
            ): "assistant",
            _oi_message_content_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TYPE_ATTR,
            ): "text",
            _oi_message_content_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
            ): "done",
        }
    )

    processor.on_end(span)

    assert (
        span._attributes[
            _oi_message_attr(
                OPENINFERENCE_INPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_ATTR,
            )
        ]
        == "hello\nworld"
    )
    assert (
        span._attributes[
            _oi_message_attr(
                OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
                0,
                OPENINFERENCE_MESSAGE_CONTENT_ATTR,
            )
        ]
        == "done"
    )


def test_contract_processor_normalizes_and_removes_aliases_from_smolagents_spans():
    processor = SmolagentsSpanContractProcessor()
    nested_tool_call_attr = f"{GEN_AI_COMPLETION_TOOL_CALLS_ATTR}.0.id"
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            SPAN_ALIAS_MODEL: "openai/gpt-4o-mini",
            SPAN_ALIAS_PROMPT_TOKENS: 10,
            SPAN_ALIAS_COMPLETION_TOKENS: 5,
            SPAN_ALIAS_TOTAL_REQUEST_TOKENS: 15,
            SPAN_ALIAS_TOOLS: [{"type": "function"}],
            SPAN_ALIAS_TOOL_CALLS: [{"id": "call_1"}],
            RESPAN_SPAN_TOOLS: '[{"type":"function"}]',
            RESPAN_SPAN_TOOL_CALLS: '[{"id":"call_1"}]',
            TLSpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            LLM_REQUEST_FUNCTIONS_ATTR: [{"type": "function"}],
            GEN_AI_COMPLETION_TOOL_CALLS_ATTR: [{"id": "call_1"}],
            nested_tool_call_attr: "call_1",
        }
    )

    processor.on_end(span)

    assert SPAN_ALIAS_MODEL not in span._attributes
    assert SPAN_ALIAS_PROMPT_TOKENS not in span._attributes
    assert SPAN_ALIAS_COMPLETION_TOKENS not in span._attributes
    assert SPAN_ALIAS_TOTAL_REQUEST_TOKENS not in span._attributes
    assert SPAN_ALIAS_TOOLS not in span._attributes
    assert SPAN_ALIAS_TOOL_CALLS not in span._attributes
    assert RESPAN_SPAN_TOOLS not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes
    assert span._attributes[TLSpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-4o-mini"
    assert json.loads(span._attributes[LLM_REQUEST_FUNCTIONS_ATTR]) == [
        {"type": "function"}
    ]
    assert json.loads(span._attributes[GEN_AI_COMPLETION_TOOL_CALLS_ATTR]) == [
        {"id": "call_1"}
    ]
    assert nested_tool_call_attr not in span._attributes
    assert span._attributes[GEN_AI_COMPLETION_ROLE_ATTR] == "assistant"
    assert span._attributes[GEN_AI_COMPLETION_CONTENT_ATTR] == ""


def test_contract_processor_backfills_canonical_tool_fields_from_helpers():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            RESPAN_SPAN_TOOLS: '[{"type":"function"}]',
            RESPAN_SPAN_TOOL_CALLS: '[{"id":"call_1"}]',
        }
    )

    processor.on_end(span)

    assert json.loads(span._attributes[LLM_REQUEST_FUNCTIONS_ATTR]) == [
        {"type": "function"}
    ]
    assert json.loads(span._attributes[GEN_AI_COMPLETION_TOOL_CALLS_ATTR]) == [
        {"id": "call_1"}
    ]
    assert RESPAN_SPAN_TOOLS not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes


def test_contract_processor_promotes_final_answer_tool_call_to_content():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            RESPAN_SPAN_TOOL_CALLS: json.dumps(
                [
                    {
                        "id": "call_final",
                        "type": "function",
                        TOOL_CALL_FUNCTION_FIELD: {
                            TOOL_CALL_FUNCTION_NAME_FIELD: (
                                SMOLAGENTS_FINAL_ANSWER_TOOL_NAME
                            ),
                            TOOL_CALL_FUNCTION_ARGUMENTS_FIELD: json.dumps(
                                {
                                    SMOLAGENTS_FINAL_ANSWER_ARGUMENT: (
                                        "The final total is $63."
                                    )
                                }
                            ),
                        },
                    }
                ]
            ),
        }
    )

    processor.on_end(span)

    assert span._attributes[GEN_AI_COMPLETION_CONTENT_ATTR] == (
        "The final total is $63."
    )
    assert span._attributes[GEN_AI_COMPLETION_ROLE_ATTR] == "assistant"
    assert GEN_AI_COMPLETION_TOOL_CALLS_ATTR not in span._attributes
    assert RESPAN_SPAN_TOOL_CALLS not in span._attributes


def test_contract_processor_ignores_non_smolagents_spans():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        _attributes={
            OTEL_SCOPE_NAME: NON_SMOLAGENTS_SCOPE_NAME,
            SPAN_ALIAS_MODEL: "openai/gpt-4o-mini",
            SPAN_ALIAS_TOOLS: [{"type": "function"}],
        }
    )

    processor.on_end(span)

    assert span._attributes[SPAN_ALIAS_MODEL] == "openai/gpt-4o-mini"
    assert span._attributes[SPAN_ALIAS_TOOLS] == [{"type": "function"}]


def test_content_and_contract_processors_use_real_tool_name_and_canonical_io():
    content = SmolagentsSpanContentProcessor()
    contract = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        name="SimpleTool",
        context=SimpleNamespace(trace_id=101),
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_SMOLAGENTS_MODULE),
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "openinference.span.kind": "TOOL",
            "tool.name": "calculate_invoice_total",
            TLSpanAttributes.TRACELOOP_ENTITY_INPUT: '{"unit_price_usd":9,"quantity":7}',
            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT: "7 items cost $63",
            "respan.entity.log_type": "tool",
            "tool_calls": "bad-alias",
        },
    )

    content.on_end(span)
    contract.on_end(span)

    assert span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "calculate_invoice_total"
    )
    assert json.loads(span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"quantity": 7, "unit_price_usd": 9},
        "name": "calculate_invoice_total",
    }
    assert "tool_calls" not in span._attributes
    assert "smolagents.respan.tool_name" not in span._attributes


def test_step_is_stable_task_without_runtime_repr():
    processor = SmolagentsSpanContractProcessor()
    span = SimpleNamespace(
        name="Step 2",
        context=SimpleNamespace(trace_id=102),
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_SMOLAGENTS_MODULE),
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "respan.entity.log_type": "workflow",
            TLSpanAttributes.TRACELOOP_ENTITY_INPUT: (
                "ActionStep(step_number=2, timing=Timing(start_time=123.4))"
            ),
        },
    )

    processor.on_end(span)

    assert span._attributes["respan.entity.log_type"] == "task"
    assert span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "smolagents.step"
    )
    assert json.loads(span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "step_number": 2
    }
    assert "123.4" not in json.dumps(span._attributes)


def test_agent_does_not_duplicate_child_model_or_usage():
    processor = SmolagentsSpanContractProcessor()
    child = SimpleNamespace(
        name="LiteLLMModel.generate",
        context=SimpleNamespace(trace_id=103),
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_SMOLAGENTS_MODULE),
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "respan.entity.log_type": "chat",
            TLSpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
        },
    )
    agent = SimpleNamespace(
        name="CodeAgent.run",
        context=SimpleNamespace(trace_id=103),
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_SMOLAGENTS_MODULE),
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "respan.entity.log_type": "agent",
            TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS: 100,
            TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS: 20,
            TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS: 120,
        },
    )

    processor.on_end(child)
    processor.on_end(agent)

    assert TLSpanAttributes.LLM_REQUEST_MODEL not in agent._attributes
    assert TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS not in agent._attributes
    assert TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in agent._attributes
    assert TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS not in agent._attributes


def test_streaming_agent_output_is_semantic_and_common_only():
    processor = SmolagentsSpanContractProcessor()
    agent = SimpleNamespace(
        name="ToolCallingAgent.run",
        context=SimpleNamespace(trace_id=104),
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_SMOLAGENTS_MODULE),
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "respan.entity.log_type": "agent",
            TLSpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT: (
                "ToolCall(name='final_answer')"
                "ActionOutput(output='streamed result', is_final_answer=True)"
            ),
        },
    )
    processor.on_end(agent)

    assert TLSpanAttributes.LLM_REQUEST_MODEL not in agent._attributes
    assert (
        json.loads(agent._attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        == "streamed result"
    )


def test_hostile_and_oversized_values_fail_open_with_valid_bounded_json():
    class HostileMapping(dict):
        def items(self):
            raise RuntimeError("must not escape")

    span = SimpleNamespace(
        name="weather_tool",
        context=SimpleNamespace(trace_id=105),
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_SMOLAGENTS_MODULE),
        _attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "respan.entity.log_type": "tool",
            SMOLAGENTS_TOOL_NAME_HINT: "weather_tool",
            TLSpanAttributes.TRACELOOP_ENTITY_INPUT: HostileMapping(
                {"api_key": "plain-input-secret"}
            ),
            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT: {
                "client_secret": "plain-secret",
                "content": "😀" * 10_000,
            },
        },
    )

    SmolagentsSpanContractProcessor().on_end(span)
    input_value = json.loads(span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT])
    output_json = span._attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert input_value["arguments"] == {
        "serialization_error": True,
        "type": "HostileMapping",
    }
    assert "plain-secret" not in output_json
    assert len(output_json.encode("utf-8")) <= 16_000
    json.loads(output_json)


def test_real_readable_span_reaches_canonical_tool_contract():
    span = ReadableSpan(
        name="SimpleTool",
        attributes={
            OTEL_SCOPE_NAME: OPENINFERENCE_SMOLAGENTS_MODULE,
            "openinference.span.kind": "TOOL",
            "tool.name": "calculate_invoice_total",
            TLSpanAttributes.TRACELOOP_ENTITY_INPUT: '{"quantity":7,"price":9}',
            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT: "63",
            "respan.entity.log_type": "tool",
            "tool_calls": "legacy-alias",
        },
        instrumentation_scope=InstrumentationScope(OPENINFERENCE_SMOLAGENTS_MODULE),
    )

    SmolagentsSpanContentProcessor().on_end(span)
    SmolagentsSpanContractProcessor().on_end(span)
    assert span.attributes[TLSpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "calculate_invoice_total"
    )
    assert json.loads(span.attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"price": 9, "quantity": 7},
        "name": "calculate_invoice_total",
    }
    assert "tool_calls" not in span.attributes
