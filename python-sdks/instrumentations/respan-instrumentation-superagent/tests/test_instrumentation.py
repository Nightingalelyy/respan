import asyncio
import logging
import sys
from types import ModuleType

import pytest

from respan_instrumentation_superagent import _instrumentation
from respan_instrumentation_superagent import SuperagentInstrumentor
from respan_tracing.core.tracer import RespanTracer


def _install_fake_safety_agent(monkeypatch):
    class FakeSafetyClient:
        async def guard(self, input=None, **kwargs):
            return {"classification": "pass", "input": input, "kwargs": kwargs}

        async def redact(self, input=None, **kwargs):
            return {"redacted": input, "kwargs": kwargs}

        async def scan(self, repo=None, **kwargs):
            return {"result": repo, "kwargs": kwargs}

    package_module = ModuleType("safety_agent")
    client_module = ModuleType("safety_agent.client")
    client_module.SafetyClient = FakeSafetyClient
    package_module.client = client_module

    monkeypatch.setitem(sys.modules, "safety_agent", package_module)
    monkeypatch.setitem(sys.modules, "safety_agent.client", client_module)
    return FakeSafetyClient


@pytest.fixture(autouse=True)
def reset_instrumentation_state():
    RespanTracer.reset_instance()
    _instrumentation._ACTIVE_INSTANCES = 0
    _instrumentation._restore_safety_client()
    yield
    _instrumentation._ACTIVE_INSTANCES = 0
    _instrumentation._restore_safety_client()
    RespanTracer.reset_instance()


def test_activate_patches_and_deactivate_restores_methods(monkeypatch):
    FakeSafetyClient = _install_fake_safety_agent(monkeypatch)
    original_guard = FakeSafetyClient.guard
    emitted = []

    monkeypatch.setattr(
        _instrumentation,
        "emit_superagent_span",
        lambda **kwargs: emitted.append(kwargs) or True,
    )

    instrumentor = SuperagentInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert FakeSafetyClient.guard is not original_guard

    client = FakeSafetyClient()
    result = asyncio.run(client.guard(input="hello", model="superagent/guard-1.7b"))

    assert result["classification"] == "pass"
    assert emitted[0]["method_name"] == "guard"
    assert emitted[0]["kwargs"] == {
        "input": "hello",
        "model": "superagent/guard-1.7b",
    }

    instrumentor.deactivate()

    assert FakeSafetyClient.guard is original_guard
    assert instrumentor._is_instrumented is False


def test_multiple_instrumentors_restore_after_last_deactivate(monkeypatch):
    FakeSafetyClient = _install_fake_safety_agent(monkeypatch)
    original_guard = FakeSafetyClient.guard

    first = SuperagentInstrumentor(methods=("guard",))
    second = SuperagentInstrumentor(methods=("guard",))

    first.activate()
    second.activate()

    assert FakeSafetyClient.guard is not original_guard

    first.deactivate()
    assert FakeSafetyClient.guard is not original_guard

    second.deactivate()
    assert FakeSafetyClient.guard is original_guard


def test_wrapped_method_emits_error_span(monkeypatch):
    class FakeSafetyClient:
        async def guard(self, input=None, **kwargs):
            raise RuntimeError("blocked")

    package_module = ModuleType("safety_agent")
    client_module = ModuleType("safety_agent.client")
    client_module.SafetyClient = FakeSafetyClient
    package_module.client = client_module
    monkeypatch.setitem(sys.modules, "safety_agent", package_module)
    monkeypatch.setitem(sys.modules, "safety_agent.client", client_module)

    emitted = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_superagent_span",
        lambda **kwargs: emitted.append(kwargs) or True,
    )

    instrumentor = SuperagentInstrumentor(methods=("guard",))
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="blocked"):
        asyncio.run(FakeSafetyClient().guard(input="hello"))

    assert emitted[0]["method_name"] == "guard"
    assert isinstance(emitted[0]["error"], RuntimeError)


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    FakeSafetyClient = _install_fake_safety_agent(monkeypatch)
    original_guard = FakeSafetyClient.guard
    RespanTracer(is_enabled=False)

    instrumentor = SuperagentInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert FakeSafetyClient.guard is original_guard
    assert instrumentor._is_instrumented is False
    assert "Superagent instrumentation skipped because Respan tracing is disabled" in caplog.text


def test_activate_logs_warning_when_dependency_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        if module_name == "safety_agent.client":
            raise ImportError(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )

    instrumentor = SuperagentInstrumentor()
    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate Superagent instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False
