import logging
import sys
from types import ModuleType, SimpleNamespace

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
import pytest

from respan_instrumentation_guardrails import GuardrailsInstrumentor
from respan_instrumentation_guardrails import _instrumentation
from respan_instrumentation_guardrails._instrumentation import (
    GUARDRAILS_RUNTIME_MODULE,
    GuardrailsSpanProcessor,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_GUARDRAIL
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exception):
        self.exceptions.append(exception)

    def set_status(self, status):
        self.status = status


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = FakeSpan(name=name)
        self.spans.append(span)
        return span


class FakeReadableSpan:
    def __init__(self, name, attributes):
        self.name = name
        self._attributes = dict(attributes)


def _install_fake_guardrails(monkeypatch):
    class FakeGuard:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                validation_passed=True,
                validated_output={"mode": "call"},
                raw_llm_output='{"mode": "call"}',
            )

        def parse(self, *args, **kwargs):
            return SimpleNamespace(
                validation_passed=True,
                validated_output={"mode": "parse"},
                raw_llm_output=kwargs.get("llm_output", ""),
            )

        def validate(self, llm_output, *args, **kwargs):
            return SimpleNamespace(
                validation_passed=True,
                validated_output=llm_output,
                raw_llm_output=llm_output,
            )

    guardrails_module = ModuleType(GUARDRAILS_RUNTIME_MODULE)
    guardrails_module.Guard = FakeGuard
    monkeypatch.setitem(sys.modules, GUARDRAILS_RUNTIME_MODULE, guardrails_module)
    return FakeGuard


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_parse_emits_guardrail_span(monkeypatch):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: fake_tracer)

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()

    result = fake_guard_class().parse(
        llm_output='{"issue": "late shipment"}',
        num_reasks=0,
    )

    span = fake_tracer.spans[0]
    assert result.validation_passed is True
    assert span.name == "guardrails.parse"
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_GUARDRAIL
    assert span.attributes["traceloop.entity.name"] == "guardrails.parse"
    assert '"method": "parse"' in span.attributes["traceloop.entity.input"]
    assert '"validation_passed": true' in span.attributes["traceloop.entity.output"]


def test_call_and_validate_are_wrapped(monkeypatch):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: fake_tracer)

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()

    fake_guard = fake_guard_class()
    fake_guard(model="gpt-4o-mini", messages=[])
    fake_guard.validate("known output")

    assert [span.name for span in fake_tracer.spans] == [
        "guardrails.call",
        "guardrails.validate",
    ]


def test_exception_records_span_error(monkeypatch):
    class FailingGuard:
        def parse(self, *args, **kwargs):
            raise ValueError("invalid output")

    guardrails_module = ModuleType(GUARDRAILS_RUNTIME_MODULE)
    guardrails_module.Guard = FailingGuard
    monkeypatch.setitem(sys.modules, GUARDRAILS_RUNTIME_MODULE, guardrails_module)

    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: fake_tracer)

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()

    with pytest.raises(ValueError, match="invalid output"):
        FailingGuard().parse(llm_output="bad")

    span = fake_tracer.spans[0]
    assert isinstance(span.exceptions[0], ValueError)
    assert '"error_type": "ValueError"' in span.attributes["traceloop.entity.output"]


def test_deactivate_restores_original_methods(monkeypatch):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    original_parse = fake_guard_class.parse

    instrumentor = GuardrailsInstrumentor()
    instrumentor.activate()
    assert fake_guard_class.parse is not original_parse

    instrumentor.deactivate()
    assert fake_guard_class.parse is original_parse
    assert instrumentor._is_instrumented is False


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake_guard_class = _install_fake_guardrails(monkeypatch)
    original_parse = fake_guard_class.parse
    RespanTracer(is_enabled=False)

    instrumentor = GuardrailsInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake_guard_class.parse is original_parse
    assert instrumentor._is_instrumented is False
    assert (
        "Guardrails instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_guardrails_runtime_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == GUARDRAILS_RUNTIME_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = GuardrailsInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Guardrails instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_span_processor_translates_guardrails_llm_call_span():
    span = FakeReadableSpan(
        name="call",
        attributes={
            "type": "guardrails/guard/step/call",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Return JSON",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": '{"ok": true}',
            "llm.invocation_parameters": "{'model': 'gpt-4o', 'temperature': 0}",
            "llm.token_count.prompt": "27",
            "llm.token_count.completion": "105",
            "llm.token_count.total": "132",
            "input.value": '{"messages": []}',
            "output.value": '{"output": "{\\"ok\\": true}"}',
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == LOG_TYPE_CHAT
    assert span._attributes[LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert span._attributes[LLM_REQUEST_MODEL] == "gpt-4o"
    assert span._attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0
    assert span._attributes[LLM_USAGE_PROMPT_TOKENS] == 27
    assert span._attributes[LLM_USAGE_COMPLETION_TOKENS] == 105
    assert span._attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 132
    assert span._attributes["gen_ai.prompt.0.role"] == "user"
    assert span._attributes["gen_ai.prompt.0.content"] == "Return JSON"
    assert span._attributes["gen_ai.completion.0.role"] == "assistant"
    assert span._attributes["gen_ai.completion.0.content"] == '{"ok": true}'
    assert span._attributes["traceloop.entity.name"] == "guardrails.call"
    assert span._attributes["traceloop.entity.input"] == '{"messages": []}'
    assert span._attributes["traceloop.entity.output"] == '{"output": "{\\"ok\\": true}"}'


def test_span_processor_marks_guardrails_non_llm_span_as_guardrail():
    span = FakeReadableSpan(
        name="guard",
        attributes={
            "type": "guardrails/guard",
            "input.value": "input",
            "output.value": "output",
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_GUARDRAIL
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == LOG_TYPE_GUARDRAIL
    assert span._attributes["traceloop.entity.name"] == "guardrails.guard"
    assert span._attributes["traceloop.entity.input"] == "input"
    assert span._attributes["traceloop.entity.output"] == "output"


def test_span_processor_keeps_local_validation_step_call_as_guardrail():
    span = FakeReadableSpan(
        name="call",
        attributes={
            "type": "guardrails/guard/step/call",
            "input.value": '{"args": []}',
            "output.value": '{"output": "validated"}',
        },
    )

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_GUARDRAIL
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == LOG_TYPE_GUARDRAIL
    assert LLM_REQUEST_TYPE not in span._attributes
    assert LLM_USAGE_PROMPT_TOKENS not in span._attributes
    assert span._attributes["traceloop.entity.name"] == "guardrails.call"


def test_span_processor_ignores_unrelated_span():
    span = FakeReadableSpan(name="http", attributes={"http.method": "POST"})

    GuardrailsSpanProcessor().on_end(span)

    assert span._attributes == {"http.method": "POST"}
