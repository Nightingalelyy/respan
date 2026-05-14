import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from respan_instrumentation_autogen import AutoGenInstrumentor
from respan_instrumentation_autogen import _instrumentation
from respan_instrumentation_autogen._instrumentation import (
    OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES,
    OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE,
    TRACER_PROVIDER_KWARG,
    _load_openinference_autogen_agentchat_class,
)
from respan_instrumentation_autogen._native_processor import AutoGenNativeSpanProcessor
from respan_tracing.core.tracer import RespanTracer


def _install_fake_modules(monkeypatch, *, class_name: str | None = None):
    selected_class_name = class_name or OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES[0]

    class FakeAutogenAgentChatInstrumentor:
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
    openinference_autogen_module = ModuleType(OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE)
    setattr(
        openinference_autogen_module,
        selected_class_name,
        FakeAutogenAgentChatInstrumentor,
    )
    openinference_instrumentation_module.autogen_agentchat = openinference_autogen_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE,
        openinference_autogen_module,
    )

    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        autogen_instrumentor_class=FakeAutogenAgentChatInstrumentor,
        openinference_instrumentor_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def fake_tracer_provider(monkeypatch):
    provider = SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=("exporter",))
    )
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: provider)
    return provider


def test_load_openinference_autogen_agentchat_class_uses_preferred_name(monkeypatch):
    fake = _install_fake_modules(monkeypatch)

    assert _load_openinference_autogen_agentchat_class() is (
        fake.autogen_instrumentor_class
    )


def test_load_openinference_autogen_agentchat_class_supports_fallback_name(
    monkeypatch,
):
    fallback_class_name = OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES[-1]
    fake = _install_fake_modules(monkeypatch, class_name=fallback_class_name)

    assert _load_openinference_autogen_agentchat_class() is (
        fake.autogen_instrumentor_class
    )


def test_activate_uses_openinference_autogen_defaults(
    monkeypatch,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = AutoGenInstrumentor()
    instrumentor.activate()

    assert isinstance(
        fake_tracer_provider._active_span_processor._span_processors[0],
        AutoGenNativeSpanProcessor,
    )
    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.instrumentor_class is fake.autogen_instrumentor_class
    assert delegate.kwargs == {}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == ("exporter",)


def test_activate_passes_custom_openinference_kwargs(monkeypatch, fake_tracer_provider):
    fake = _install_fake_modules(monkeypatch)

    instrumentor = AutoGenInstrumentor(
        trace_content=False,
        custom_option="value",
        tracer_provider="ignored",
    )
    instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.kwargs == {
        "trace_content": False,
        "custom_option": "value",
    }
    assert TRACER_PROVIDER_KWARG not in delegate.kwargs
    instrumentor.deactivate()


def test_activate_cleans_up_delegate_when_activation_fails(
    monkeypatch,
    caplog,
    fake_tracer_provider,
):
    fake = _install_fake_modules(monkeypatch)

    def activate_raises(self):
        self.is_activated = True
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake.openinference_instrumentor_class,
        "activate",
        activate_raises,
    )

    instrumentor = AutoGenInstrumentor()
    with caplog.at_level(logging.ERROR):
        instrumentor.activate()

    delegate = fake.openinference_instrumentor_class.created[0]
    assert delegate.is_deactivated is True
    assert instrumentor._delegate is None
    assert instrumentor._is_instrumented is False
    assert fake_tracer_provider._active_span_processor._span_processors == ("exporter",)
    assert "Failed to activate AutoGen instrumentation" in caplog.text


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = AutoGenInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.openinference_instrumentor_class.created == []
    assert instrumentor._is_instrumented is False
    assert (
        "AutoGen instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependencies_are_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE:
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = AutoGenInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate AutoGen instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
