import json
import logging
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest

from respan_instrumentation_beeai import BeeAIInstrumentor
from respan_instrumentation_beeai import _instrumentation
from respan_instrumentation_beeai._instrumentation import OPENINFERENCE_BEEAI_MODULE
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer


def _install_fake_modules(monkeypatch):
    class FakeBeeAIInstrumentor:
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
    openinference_beeai_module = ModuleType(OPENINFERENCE_BEEAI_MODULE)
    openinference_beeai_module.BeeAIInstrumentor = FakeBeeAIInstrumentor
    openinference_instrumentation_module.beeai = openinference_beeai_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_BEEAI_MODULE,
        openinference_beeai_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )
    monkeypatch.setattr(_instrumentation, "_patch_beeai_processors", lambda: None)
    monkeypatch.setattr(_instrumentation, "_unpatch_beeai_processors", lambda: None)

    return SimpleNamespace(
        beeai_instrumentor_class=FakeBeeAIInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_beeai(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = BeeAIInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.beeai_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = BeeAIInstrumentor(trace_content=False, custom_option="value")
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {
        "trace_content": False,
        "custom_option": "value",
    }


def test_activate_cleans_up_delegate_when_activation_fails(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = BeeAIInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate BeeAI instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = BeeAIInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert "BeeAI instrumentation skipped because Respan tracing is disabled" in caplog.text


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_BEEAI_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = BeeAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate BeeAI instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_cleanup_processor_removes_beeai_off_contract_aliases():
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "final_answer", "arguments": "{}"},
        }
    ]
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_BEEAI_MODULE),
        _attributes={
            RESPAN_SPAN_TOOLS: "[]",
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
            "tools": [],
            "tool_calls": tool_calls,
            "model": "gpt-4.1-nano",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_request_tokens": 15,
            "llm.request.functions": "[]",
            "gen_ai.completion.0.tool_calls": tool_calls,
        },
    )

    _instrumentation._BeeAIOffContractAliasProcessor().on_end(span)

    for key in (
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
    ):
        assert key not in span._attributes
    assert span._attributes["llm.request.functions"] == "[]"
    assert json.loads(span._attributes["gen_ai.completion.0.tool_calls"]) == tool_calls


def test_cleanup_processor_ignores_non_beeai_spans():
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="openinference.instrumentation.other"),
        _attributes={
            "tools": [],
            "gen_ai.completion.0.tool_calls": [],
        },
    )

    _instrumentation._BeeAIOffContractAliasProcessor().on_end(span)

    assert span._attributes["tools"] == []
    assert span._attributes["gen_ai.completion.0.tool_calls"] == []


def test_activate_places_cleanup_after_openinference_translator(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    translator = object()
    exporter = object()

    class FakeOpenInferenceInstrumentor(fake.openinference_instrumentor_class):
        created = []

        @classmethod
        def _get_translator(cls):
            return translator

    active_span_processor = SimpleNamespace(
        _span_processors=(
            translator,
            exporter,
        )
    )
    tracer_provider = SimpleNamespace(_active_span_processor=active_span_processor)
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: tracer_provider,
    )

    instrumentor = BeeAIInstrumentor()
    instrumentor.activate()

    processors = active_span_processor._span_processors
    assert processors[0] is translator
    assert isinstance(processors[1], _instrumentation._BeeAIOffContractAliasProcessor)
    assert processors[2] is exporter

    instrumentor.deactivate()

    assert active_span_processor._span_processors == (translator, exporter)


def test_beeai_error_patch_preserves_status_and_drops_duplicate_child() -> None:
    from openinference.instrumentation.beeai._span import SpanWrapper
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry.trace import StatusCode

    class ProviderError(RuntimeError):
        status_code = 404

    provider_error = ProviderError("requested model was not found")
    error = RuntimeError("Chat Model error")
    error.__cause__ = provider_error
    span = SpanWrapper(name="ChatModel", kind=OpenInferenceSpanKindValues.LLM)
    meta = SimpleNamespace(name="error", created_at=datetime.now(timezone.utc))

    _instrumentation._patch_beeai_processors()
    try:
        child = span.child(
            "error",
            event=(SimpleNamespace(error=error), meta),
        )
    finally:
        _instrumentation._unpatch_beeai_processors()

    assert child is span
    assert span.children == []
    assert span.status == StatusCode.ERROR
    assert span.attributes["status_code"] == 404
    assert span.attributes["error.message"] == "Chat Model error"
    assert SpanAttributes.OUTPUT_VALUE not in span.attributes
    assert [event.name for event in span.events] == ["error"]


def test_beeai_finish_error_cannot_be_reset_to_success() -> None:
    import asyncio

    from openinference.instrumentation.beeai._span import SpanWrapper
    from openinference.instrumentation.beeai.processors.base import Processor
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry.trace import StatusCode

    error = RuntimeError("deterministic failure")
    processor = object.__new__(Processor)
    processor.span = SpanWrapper(
        name="ChatModel",
        kind=OpenInferenceSpanKindValues.LLM,
    )
    event = SimpleNamespace(error=error, output={"partial": "ignored"})
    meta = SimpleNamespace(created_at=datetime.now(timezone.utc))

    _instrumentation._patch_beeai_processors()
    try:
        asyncio.run(processor.end(event, meta))
    finally:
        _instrumentation._unpatch_beeai_processors()

    assert processor.span.status == StatusCode.ERROR
    assert processor.span.attributes["status_code"] == 500
    assert processor.span.attributes["error.message"] == "deterministic failure"
    assert SpanAttributes.OUTPUT_VALUE not in processor.span.attributes


def test_beeai_error_marks_active_parent_span(monkeypatch) -> None:
    from openinference.instrumentation.beeai._span import SpanWrapper
    from openinference.semconv.trace import OpenInferenceSpanKindValues
    from opentelemetry.trace import StatusCode

    class ActiveSpan:
        def __init__(self) -> None:
            self.attributes = {}
            self.exceptions = []
            self.status = SimpleNamespace(status_code=StatusCode.UNSET)

        def is_recording(self) -> bool:
            return True

        def record_exception(self, error) -> None:
            self.exceptions.append(error)

        def set_status(self, status) -> None:
            self.status = status

        def set_attribute(self, key, value) -> None:
            self.attributes[key] = value

    class ProviderError(RuntimeError):
        status_code = 404

    active_span = ActiveSpan()
    monkeypatch.setattr(_instrumentation.trace, "get_current_span", lambda: active_span)
    wrapped_span = SpanWrapper(name="ChatModel", kind=OpenInferenceSpanKindValues.LLM)
    error = RuntimeError("Chat Model error")
    error.__cause__ = ProviderError("requested model was not found")

    _instrumentation._patch_beeai_processors()
    try:
        wrapped_span.record_exception(error)
    finally:
        _instrumentation._unpatch_beeai_processors()

    assert active_span.status.status_code == StatusCode.ERROR
    assert active_span.attributes["status_code"] == 404
    assert active_span.attributes["error.message"] == "Chat Model error"
    assert active_span.exceptions == [error]


def test_beeai_finish_preserves_direct_chat_text_content() -> None:
    import asyncio

    from openinference.instrumentation.beeai._span import SpanWrapper
    from openinference.instrumentation.beeai.processors.base import Processor
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

    processor = object.__new__(Processor)
    processor.span = SpanWrapper(
        name="ChatModel",
        kind=OpenInferenceSpanKindValues.LLM,
    )
    output = SimpleNamespace(get_text_content=lambda: "A complete assistant answer.")
    event = SimpleNamespace(error=None, output=output)
    meta = SimpleNamespace(created_at=datetime.now(timezone.utc))

    _instrumentation._patch_beeai_processors()
    try:
        asyncio.run(processor.end(event, meta))
    finally:
        _instrumentation._unpatch_beeai_processors()

    assert (
        processor.span.attributes[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.content"
        ]
        == "A complete assistant answer."
    )
