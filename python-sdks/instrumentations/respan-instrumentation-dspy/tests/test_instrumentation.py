import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

from opentelemetry.semconv_ai import SpanAttributes

import respan_instrumentation_dspy._callback
from respan_instrumentation_dspy import DSPyInstrumentor
from respan_instrumentation_dspy._callback import DSPyInstrumentationCallback
from respan_instrumentation_dspy._utils import (
    add_lm_usage_attributes,
    extract_provider_name,
    normalize_messages,
)


def _install_fake_dspy(monkeypatch, callbacks=None):
    configured_callbacks = list(callbacks or [])

    class FakeSettings:
        def get(self, key, default=None):
            if key == "callbacks":
                return configured_callbacks
            return default

    def configure(**kwargs):
        if "callbacks" in kwargs:
            configured_callbacks.clear()
            configured_callbacks.extend(kwargs["callbacks"])

    dspy_module = ModuleType("dspy")
    dspy_module.settings = FakeSettings()
    dspy_module.configure = configure
    monkeypatch.setitem(dic=sys.modules, name="dspy", value=dspy_module)

    return SimpleNamespace(
        dspy_module=dspy_module,
        configured_callbacks=configured_callbacks,
    )


def _capture_spans(monkeypatch):
    captured_spans = []

    def fake_build_readable_span(name, **kwargs):
        span = {"name": name, **kwargs}
        captured_spans.append(span)
        return span

    monkeypatch.setattr(
        target=respan_instrumentation_dspy._callback,
        name="build_readable_span",
        value=fake_build_readable_span,
    )
    monkeypatch.setattr(
        target=respan_instrumentation_dspy._callback,
        name="inject_span",
        value=lambda span: True,
    )
    monkeypatch.setattr(
        target=respan_instrumentation_dspy._callback,
        name="_get_current_otel_parent",
        value=lambda: (None, None),
    )
    monkeypatch.setattr(
        target=respan_instrumentation_dspy._callback,
        name="_get_active_dspy_call_id",
        value=lambda: None,
    )
    return captured_spans


def test_activate_instruments_global_callbacks_and_deactivates(monkeypatch):
    fake_dspy = _install_fake_dspy(monkeypatch=monkeypatch, callbacks=["existing"])
    instrumentor = DSPyInstrumentor()

    instrumentor.activate()

    assert fake_dspy.configured_callbacks[0] == "existing"
    assert any(
        isinstance(callback, DSPyInstrumentationCallback)
        for callback in fake_dspy.configured_callbacks
    )

    instrumentor.deactivate()

    assert fake_dspy.configured_callbacks == ["existing"]


def test_activate_target_callbacks_and_deactivates(monkeypatch):
    _install_fake_dspy(monkeypatch=monkeypatch)
    target = SimpleNamespace(callbacks=["target-existing"])
    instrumentor = DSPyInstrumentor(target=target)

    instrumentor.activate()

    assert target.callbacks[0] == "target-existing"
    assert any(
        isinstance(callback, DSPyInstrumentationCallback)
        for callback in target.callbacks
    )

    instrumentor.deactivate()

    assert target.callbacks == ["target-existing"]


def test_activate_logs_warning_when_dspy_missing(monkeypatch, caplog):
    def fake_import_module(name):
        if name == "dspy":
            raise ImportError("No module named dspy")
        raise AssertionError(name)

    monkeypatch.setattr(
        target=importlib,
        name="import_module",
        value=fake_import_module,
    )
    instrumentor = DSPyInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate DSPy instrumentation" in caplog.text


def test_lm_callback_emits_canonical_chat_span(monkeypatch):
    captured_spans = _capture_spans(monkeypatch=monkeypatch)
    callback = DSPyInstrumentationCallback()
    language_model = SimpleNamespace(
        model="openai/gpt-4o-mini",
        model_type="chat",
        kwargs={"temperature": 0.2, "max_tokens": 64},
        history=[],
    )

    callback.on_lm_start(
        call_id="language-model-call",
        instance=language_model,
        inputs={
            "prompt": None,
            "messages": [{"role": "user", "content": "hello"}],
            "kwargs": {},
        },
    )
    language_model.history.append(
        {
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            }
        }
    )
    callback.on_lm_end(
        call_id="language-model-call",
        outputs=["hi there"],
        exception=None,
    )

    attributes = captured_spans[0]["attributes"]
    assert captured_spans[0]["name"] == "dspy.lm"
    assert attributes["respan.entity.log_type"] == "chat"
    assert attributes["respan.entity.log_method"] == "tracing_integration"
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["gen_ai.request.model"] == "openai/gpt-4o-mini"
    assert attributes["llm.request.type"] == "chat"
    assert attributes["gen_ai.prompt.0.role"] == "user"
    assert attributes["gen_ai.prompt.0.content"] == "hello"
    assert attributes["gen_ai.completion.0.role"] == "assistant"
    assert attributes["gen_ai.completion.0.content"] == "hi there"
    assert attributes["gen_ai.usage.input_tokens"] == 11
    assert attributes["gen_ai.usage.output_tokens"] == 7
    assert attributes["gen_ai.usage.prompt_tokens"] == 11
    assert attributes["gen_ai.usage.completion_tokens"] == 7
    assert attributes["llm.usage.total_tokens"] == 18
    assert attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "dspy.lm"
    assert attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "dspy.lm"
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attributes
    for off_contract_key in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        "respan.span.tools",
        "respan.span.tool_calls",
    ):
        assert off_contract_key not in attributes


def test_module_and_nested_lm_callbacks_share_parent_trace(monkeypatch):
    captured_spans = _capture_spans(monkeypatch=monkeypatch)
    active_call_identifiers = iter([None, "module-call"])
    monkeypatch.setattr(
        target=respan_instrumentation_dspy._callback,
        name="_get_active_dspy_call_id",
        value=lambda: next(active_call_identifiers),
    )
    callback = DSPyInstrumentationCallback()

    class Predict:
        stage = "question-answer-stage"

    module = Predict()
    language_model = SimpleNamespace(
        model="openai/gpt-4o-mini",
        model_type="chat",
        kwargs={},
        history=[],
    )

    callback.on_module_start(
        call_id="module-call",
        instance=module,
        inputs={"question": "What is tracing?"},
    )
    callback.on_lm_start(
        call_id="language-model-call",
        instance=language_model,
        inputs={"prompt": "What is tracing?", "messages": None, "kwargs": {}},
    )
    callback.on_lm_end(
        call_id="language-model-call",
        outputs=["Tracing is context."],
        exception=None,
    )
    callback.on_module_end(
        call_id="module-call",
        outputs={"answer": "Tracing is context."},
        exception=None,
    )

    lm_span = captured_spans[0]
    module_span = captured_spans[1]
    assert lm_span["trace_id"] == module_span["trace_id"]
    assert lm_span["parent_id"] == module_span["span_id"]
    assert module_span["attributes"]["respan.entity.log_type"] == "task"
    assert lm_span["attributes"]["respan.entity.log_type"] == "chat"


def test_tool_callback_emits_tool_span(monkeypatch):
    captured_spans = _capture_spans(monkeypatch=monkeypatch)
    callback = DSPyInstrumentationCallback()
    tool = SimpleNamespace(name="lookup_order")

    callback.on_tool_start(
        call_id="tool-call",
        instance=tool,
        inputs={"order_id": "ord_123"},
    )
    callback.on_tool_end(call_id="tool-call", outputs={"status": "shipped"})

    attributes = captured_spans[0]["attributes"]
    assert captured_spans[0]["name"] == "dspy.tool"
    assert attributes["respan.entity.log_type"] == "tool"
    assert attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_order"
    assert (
        attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
        == '{"order_id": "ord_123"}'
    )
    assert (
        attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        == '{"status": "shipped"}'
    )
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attributes


def test_lm_callback_records_error_status(monkeypatch):
    captured_spans = _capture_spans(monkeypatch=monkeypatch)
    callback = DSPyInstrumentationCallback()
    language_model = SimpleNamespace(
        model="openai/gpt-4o-mini",
        model_type="chat",
        kwargs={},
        history=[],
    )
    exception = RuntimeError("provider failed")

    callback.on_lm_start(
        call_id="language-model-call",
        instance=language_model,
        inputs={"prompt": "hello", "messages": None, "kwargs": {}},
    )
    callback.on_lm_end(
        call_id="language-model-call",
        outputs=None,
        exception=exception,
    )

    span = captured_spans[0]
    assert span["status_code"] == 500
    assert span["error_message"] == "provider failed"
    assert span["attributes"]["gen_ai.completion.0.content"] == "provider failed"


def test_normalize_messages_maps_prompt_and_message_inputs():
    assert normalize_messages(prompt="hello", messages=None) == [
        {"role": "user", "content": "hello"}
    ]
    assert normalize_messages(
        prompt=None,
        messages=[{"role": "system", "content": {"mode": "brief"}}, "hi"],
    ) == [
        {"role": "system", "content": {"mode": "brief"}},
        {"role": "user", "content": "hi"},
    ]


def test_extract_provider_name_maps_common_litellm_prefixes():
    assert extract_provider_name(model_name="openai/gpt-4o-mini") == "openai"
    assert extract_provider_name(model_name="anthropic/claude-sonnet-4") == "anthropic"
    assert extract_provider_name(model_name="gemini/gemini-2.5-pro") == "google"
    assert extract_provider_name(model_name="custom-provider/model") == "dspy"


def test_add_lm_usage_attributes_sets_modern_and_legacy_token_fields():
    attributes = {}

    add_lm_usage_attributes(
        attributes=attributes,
        usage={
            "input_tokens": 3,
            "output_tokens": 4,
        },
    )

    assert attributes["gen_ai.usage.input_tokens"] == 3
    assert attributes["gen_ai.usage.output_tokens"] == 4
    assert attributes["gen_ai.usage.prompt_tokens"] == 3
    assert attributes["gen_ai.usage.completion_tokens"] == 4
    assert attributes["llm.usage.total_tokens"] == 7
