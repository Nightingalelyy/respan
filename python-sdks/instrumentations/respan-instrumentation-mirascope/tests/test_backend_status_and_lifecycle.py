from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import respan_instrumentation_mirascope._instrumentation as instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_mirascope import MirascopeInstrumentor


def _exporter(monkeypatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.mirascope.status"),
    )
    return exporter


def test_backend_error_attributes_preserve_upstream_status(monkeypatch) -> None:
    exporter = _exporter(monkeypatch)
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    class ProviderError(RuntimeError):
        status_code = 503

    def call(self, content):
        raise ProviderError("provider unavailable")

    with pytest.raises(ProviderError):
        instrumentation._call_wrapper(call)(model, "hello")

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["status_code"] == 503
    assert attrs["error.message"] == "provider unavailable"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "ProviderError",
        "message": "provider unavailable",
        "status": "error",
    }


def test_lifecycle_is_reference_counted_and_idempotent(monkeypatch) -> None:
    installed = 0
    removed = 0

    def install() -> None:
        nonlocal installed
        installed += 1

    def remove() -> None:
        nonlocal removed
        removed += 1

    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(instrumentation, "_install_patches", install)
    monkeypatch.setattr(instrumentation, "_remove_patches", remove)
    monkeypatch.setattr(
        instrumentation.importlib, "import_module", lambda name: object()
    )

    first = MirascopeInstrumentor()
    second = MirascopeInstrumentor()
    first.activate()
    first.activate()
    second.activate()
    assert installed == 1

    first.deactivate()
    first.deactivate()
    assert removed == 0
    second.deactivate()
    assert removed == 1


def test_same_instance_concurrent_lifecycle_changes_refcount_once(monkeypatch) -> None:
    installed = 0
    removed = 0

    class BarrierLock:
        def __init__(self) -> None:
            self._barrier = threading.Barrier(2)
            self._lock = threading.Lock()

        def __enter__(self):
            self._barrier.wait(timeout=5)
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self._lock.release()

    def install() -> None:
        nonlocal installed
        installed += 1

    def remove() -> None:
        nonlocal removed
        removed += 1

    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(instrumentation, "_LOCK", BarrierLock())
    monkeypatch.setattr(instrumentation, "_install_patches", install)
    monkeypatch.setattr(instrumentation, "_remove_patches", remove)
    monkeypatch.setattr(instrumentation, "_is_respan_tracing_enabled", lambda: True)
    monkeypatch.setattr(
        instrumentation.importlib, "import_module", lambda name: object()
    )
    instrumentor = MirascopeInstrumentor()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(instrumentor.activate) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)

    assert installed == 1
    assert instrumentation._REFCOUNT == 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(instrumentor.deactivate) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)

    assert removed == 1
    assert instrumentation._REFCOUNT == 0


def test_partial_first_install_rolls_back_every_new_patch(monkeypatch) -> None:
    class Model:
        def call(self):
            return None

        context_call = call
        call_async = call
        context_call_async = call
        stream = call
        context_stream = call
        stream_async = call
        context_stream_async = call

    original_methods = {
        name: getattr(Model, name)
        for name in (
            "call",
            "context_call",
            "call_async",
            "context_call_async",
            "stream",
            "context_stream",
            "stream_async",
            "context_stream_async",
        )
    }

    def import_module(name: str):
        if name == "mirascope":
            return object()
        if name == "mirascope.llm.models.models":
            return SimpleNamespace(Model=Model)
        if name == "mirascope.llm.tools.toolkit":
            raise RuntimeError("deterministic partial install failure")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHES", [])
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", True)
    monkeypatch.setattr(instrumentation, "_is_respan_tracing_enabled", lambda: True)
    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    instrumentor = MirascopeInstrumentor(capture_content=False)

    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert instrumentation._REFCOUNT == 0
    assert instrumentation._PATCHES == []
    assert instrumentation._CAPTURE_CONTENT is True
    assert {name: getattr(Model, name) for name in original_methods} == original_methods
