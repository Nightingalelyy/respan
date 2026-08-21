import json
import logging
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry import trace
from respan_instrumentation_mistralai import MistralAIInstrumentor, _instrumentation
from respan_instrumentation_mistralai._instrumentation import (
    MISTRALAI_SDK_TRACER_NAME,
    OPENINFERENCE_MISTRALAI_MODULE,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.constants.tracing import SAMPLE_RATE_ATTR
from respan_tracing.core.tracer import RespanTracer


def _install_fake_modules(monkeypatch):
    translator = object()

    class FakeMistralAIInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created: ClassVar[list] = []

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

        @classmethod
        def _get_translator(cls):
            return translator

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_mistralai_module = ModuleType(OPENINFERENCE_MISTRALAI_MODULE)
    openinference_mistralai_module.MistralAIInstrumentor = FakeMistralAIInstrumentor
    openinference_instrumentation_module.mistralai = openinference_mistralai_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_MISTRALAI_MODULE,
        openinference_mistralai_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )
    active_span_processor = SimpleNamespace(_span_processors=(translator,))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: SimpleNamespace(_active_span_processor=active_span_processor),
    )
    monkeypatch.setattr(
        _instrumentation,
        "_install_stream_guards",
        lambda: (SimpleNamespace(), object, object),
    )
    monkeypatch.setattr(_instrumentation, "_restore_stream_guards", lambda _patch: None)

    return SimpleNamespace(
        mistralai_instrumentor_class=FakeMistralAIInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
        translator=translator,
        active_span_processor=active_span_processor,
    )


@pytest.fixture(autouse=True)
def reset_tracer(monkeypatch):
    RespanTracer.reset_instance()
    monkeypatch.setattr(_instrumentation, "_SHARED_DELEGATE", None)
    monkeypatch.setattr(_instrumentation, "_SHARED_CLEANUP_PROCESSOR", None)
    monkeypatch.setattr(_instrumentation, "_SHARED_STREAM_GUARD_PATCH", None)
    monkeypatch.setattr(_instrumentation, "_SHARED_INSTRUMENTOR_KWARGS", None)
    monkeypatch.setattr(_instrumentation, "_SHARED_REFCOUNT", 0)
    yield
    RespanTracer.reset_instance()


def test_activate_uses_openinference_mistralai(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = MistralAIInstrumentor()
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.mistralai_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_activate_passes_custom_openinference_kwargs(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = MistralAIInstrumentor(trace_content=False, custom_option="value")
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

    instrumentor = MistralAIInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert "Failed to activate Mistral AI instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = MistralAIInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "Mistral AI instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_MISTRALAI_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = MistralAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Mistral AI instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_cleanup_processor_removes_mistralai_off_contract_aliases():
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_MISTRALAI_MODULE),
        _attributes={
            RESPAN_SPAN_TOOLS: "[]",
            RESPAN_SPAN_TOOL_CALLS: json.dumps(tool_calls),
            "tools": [],
            "tool_calls": tool_calls,
            "model": "mistral-large-latest",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_request_tokens": 15,
            "llm.request.functions": "[]",
            "traceloop.span.kind": "chat",
            "traceloop.entity.input": json.dumps(
                {
                    "api_key": "raw-input-secret",
                    "messages": [
                        {
                            "content": "Bearer raw-input-token" + ("x" * 20_000),
                            "role": "user",
                        }
                    ],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}},
                        }
                    ],
                }
            ),
            "traceloop.entity.output": json.dumps(
                {
                    "authorization": {"nested": "raw-output-secret"},
                    "choices": [
                        {
                            "index": 0,
                            "message": {"tool_calls": tool_calls},
                        }
                    ],
                }
            ),
            "gen_ai.completion.0.tool_calls": tool_calls,
        },
        status=trace.Status(trace.StatusCode.OK),
        events=(),
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    for key in (
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "traceloop.span.kind",
    ):
        assert key not in span._attributes
    assert json.loads(span._attributes["llm.request.functions"]) == [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {}},
        }
    ]
    assert span._attributes["llm.is_streaming"] is False
    assert json.loads(span._attributes["gen_ai.completion.0.tool_calls"]) == tool_calls
    safe_input = span._attributes["traceloop.entity.input"]
    safe_output = span._attributes["traceloop.entity.output"]
    assert len(safe_input) <= _instrumentation._MAX_JSON_LENGTH
    assert len(safe_output) <= _instrumentation._MAX_JSON_LENGTH
    assert json.loads(safe_input)["api_key"] == "[REDACTED]"
    assert json.loads(safe_output)["authorization"] == "[REDACTED]"
    assert "raw-input-secret" not in safe_input
    assert "raw-input-token" not in safe_input
    assert "raw-output-secret" not in safe_output


def test_cleanup_processor_ignores_non_mistralai_spans():
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(
            name="openinference.instrumentation.other"
        ),
        _attributes={
            "tools": [],
            "gen_ai.completion.0.tool_calls": [],
        },
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    assert span._attributes["tools"] == []
    assert span._attributes["gen_ai.completion.0.tool_calls"] == []


def test_cleanup_processor_drops_native_mistral_sdk_spans_from_respan_export():
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=MISTRALAI_SDK_TRACER_NAME),
        _attributes={
            "gen_ai.system": "mistralai",
            "gen_ai.request.model": "mistral-large-latest",
        },
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    assert span._attributes[SAMPLE_RATE_ATTR] == 0
    assert span._attributes["gen_ai.system"] == "mistralai"


def test_cleanup_processor_does_not_mutate_unrelated_sdk_span():
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="unrelated_sdk_tracer"),
        _attributes={"gen_ai.system": "other"},
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    assert span._attributes == {"gen_ai.system": "other"}


def test_cleanup_processor_preserves_precise_native_error_status():
    unsafe_value = "api_key=should-not-survive-" + ("x" * 20_000)
    span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=OPENINFERENCE_MISTRALAI_MODULE),
        _attributes={
            "traceloop.span.kind": "chat",
            "traceloop.entity.input": json.dumps(
                {"model": "mistral-small-latest", "stream": False}
            ),
            "traceloop.entity.output": unsafe_value,
            "error.message": unsafe_value,
        },
        status=trace.Status(
            trace.StatusCode.ERROR,
            'SDKError: Status 401. Body: {"request_id":"do-not-copy"}',
        ),
        events=(
            SimpleNamespace(
                name="exception",
                attributes={
                    "exception.type": "mistralai.client.errors.sdkerror.SDKError",
                    "exception.message": (
                        "API error occurred: Status 401. Body: "
                        '{"request_id":"do-not-copy"}'
                    ),
                },
            ),
        ),
    )

    _instrumentation._MistralAIOffContractAliasProcessor().on_end(span)

    assert span._attributes["status_code"] == 401
    assert span._attributes["error.message"] == "Mistral request failed with status 401"
    assert json.loads(span._attributes["traceloop.entity.output"]) == {
        "error": "SDKError",
        "message": "Mistral request failed with status 401",
        "status": "error",
        "status_code": 401,
    }
    assert "request_id" not in span._attributes["traceloop.entity.output"]
    assert "should-not-survive" not in span._attributes["traceloop.entity.output"]
    assert "should-not-survive" not in span._attributes["error.message"]
    assert "traceloop.span.kind" not in span._attributes


def test_safe_json_is_bounded_redacted_and_never_uses_arbitrary_repr():
    class UnsafeRepresentation:
        def __str__(self):
            raise AssertionError("arbitrary __str__ must not be called")

    encoded = _instrumentation._safe_json_str(
        {
            "api_key": "secret-value",
            "authorization": "Bearer top-secret",
            "content": "Bearer another-secret " + ("z" * 40_000),
            "parameters": {"properties": {"api_key": {"type": "string"}}},
            "unsupported": UnsafeRepresentation(),
        }
    )
    payload = json.loads(encoded)

    assert len(encoded) <= _instrumentation._MAX_JSON_LENGTH
    assert "secret-value" not in encoded
    assert "top-secret" not in encoded
    assert "another-secret" not in encoded
    assert payload["api_key"] == "[REDACTED]"
    assert payload["authorization"] == "[REDACTED]"
    assert payload["parameters"]["properties"]["api_key"] == "[REDACTED]"
    assert payload["unsupported"] == "[UNSUPPORTED]"


def test_concurrent_activation_of_same_instance_has_one_shared_owner(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    activation_entered = threading.Event()
    second_reached_lock = threading.Event()
    release_activation = threading.Event()
    load_lock = threading.Lock()
    load_count = 0

    def tracked_load():
        nonlocal load_count
        with load_lock:
            load_count += 1
            if load_count == 2:
                second_reached_lock.set()
        return fake.mistralai_instrumentor_class

    monkeypatch.setattr(
        _instrumentation,
        "_load_openinference_mistralai_class",
        tracked_load,
    )

    def slow_activate(delegate):
        delegate.is_activated = True
        activation_entered.set()
        assert release_activation.wait(timeout=5)

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        slow_activate,
    )
    instrumentor = MistralAIInstrumentor()

    first = threading.Thread(target=instrumentor.activate)
    second = threading.Thread(target=instrumentor.activate)
    first.start()
    assert activation_entered.wait(timeout=5)
    second.start()
    assert second_reached_lock.wait(timeout=5)
    release_activation.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(fake.openinference_instrumentor_class.created) == 1
    assert instrumentor._is_instrumented is True
    assert _instrumentation._SHARED_REFCOUNT == 1

    instrumentor.deactivate()
    assert _instrumentation._SHARED_REFCOUNT == 0


def test_two_instances_share_cleanup_and_delegate_lifecycle(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    translator = object()
    exporter = object()

    class FakeOpenInferenceInstrumentor(fake.openinference_instrumentor_class):
        created: ClassVar[list] = []

        @classmethod
        def _get_translator(cls):
            return translator

    active_span_processor = SimpleNamespace(_span_processors=(translator, exporter))
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

    first = MistralAIInstrumentor()
    second = MistralAIInstrumentor()
    first.activate()
    first.activate()
    second.activate()

    assert len(FakeOpenInferenceInstrumentor.created) == 1
    assert _instrumentation._SHARED_REFCOUNT == 2
    assert (
        sum(
            isinstance(
                processor,
                _instrumentation._MistralAIOffContractAliasProcessor,
            )
            for processor in active_span_processor._span_processors
        )
        == 1
    )

    delegate = FakeOpenInferenceInstrumentor.created[0]
    first.deactivate()
    first.deactivate()
    assert delegate.is_deactivated is False
    assert _instrumentation._SHARED_REFCOUNT == 1

    second.deactivate()
    assert delegate.is_deactivated is True
    assert active_span_processor._span_processors == (translator, exporter)
    assert _instrumentation._SHARED_REFCOUNT == 0


def test_second_instance_with_different_kwargs_is_rejected(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    first = MistralAIInstrumentor(trace_content=False)
    second = MistralAIInstrumentor(trace_content=True)

    first.activate()
    with caplog.at_level(logging.WARNING):
        second.activate()

    assert len(fake.openinference_instrumentor_class.created) == 1
    assert first._is_instrumented is True
    assert second._is_instrumented is False
    assert _instrumentation._SHARED_REFCOUNT == 1
    assert "active instance uses different configuration" in caplog.text

    second.deactivate()
    first.deactivate()


def test_activation_rolls_back_when_cleanup_registration_fails(
    monkeypatch,
    caplog,
):
    fake = _install_fake_modules(monkeypatch)
    restored = []
    monkeypatch.setattr(
        MistralAIInstrumentor,
        "_register_cleanup_processor",
        lambda _self: None,
    )
    monkeypatch.setattr(
        _instrumentation,
        "_restore_stream_guards",
        lambda patch: restored.append(patch),
    )
    instrumentor = MistralAIInstrumentor()

    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False
    assert _instrumentation._SHARED_REFCOUNT == 0
    assert len(restored) == 1
    assert "cleanup processor could not be registered" in caplog.text


def test_activate_places_cleanup_after_openinference_translator(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    translator = object()
    exporter = object()

    class FakeOpenInferenceInstrumentor(fake.openinference_instrumentor_class):
        created: ClassVar[list] = []

        @classmethod
        def _get_translator(cls):
            return translator

    active_span_processor = SimpleNamespace(_span_processors=(translator, exporter))
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

    instrumentor = MistralAIInstrumentor()
    instrumentor.activate()

    processors = active_span_processor._span_processors
    assert processors[0] is translator
    assert isinstance(
        processors[1],
        _instrumentation._MistralAIOffContractAliasProcessor,
    )
    assert processors[2] is exporter

    instrumentor.deactivate()

    assert active_span_processor._span_processors == (translator, exporter)
