from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_watson_orchestrate_adk import (
    WatsonOrchestrateADKInstrumentor,
    _instrumentation,
)
from respan_instrumentation_watson_orchestrate_adk._constants import (
    RUN_CLIENT_CLASS,
    RUN_CLIENT_MODULE,
)
from respan_instrumentation_watson_orchestrate_adk._serialization import json_dumps


@pytest.fixture(autouse=True)
def _clean_runtime() -> Any:
    _instrumentation._restore_methods()
    yield
    _instrumentation._restore_methods()


def _current_run_client() -> type[Any]:
    module = pytest.importorskip(RUN_CLIENT_MODULE)
    return getattr(module, RUN_CLIENT_CLASS)


def test_real_current_watson_run_client_exports_connected_readable_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_client = _current_run_client()
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_watson_orchestrate_adk._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    def fake_create_run(
        self: Any,
        message: str,
        agent_id: str | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        del self, kwargs
        return {
            "run_id": "run-current-1",
            "thread_id": thread_id,
            "agent_id": agent_id,
            "status": "queued",
            "message": message,
        }

    monkeypatch.setattr(run_client, "create_run", fake_create_run)
    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()
    provider = TracerProvider()
    with provider.get_tracer(__name__).start_as_current_span("parent") as parent:
        response = run_client.create_run(
            SimpleNamespace(),
            "Trace current Watson client.",
            agent_id="agent-current",
            thread_id="thread-current",
        )
        parent_context = parent.get_span_context()
    instrumentor.deactivate()

    assert response["run_id"] == "run-current-1"
    assert len(emitted) == 1
    span = emitted[0]
    attrs = span.attributes
    assert span.parent.span_id == parent_context.span_id
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "agent-current"
    assert (
        json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])["request"]["message"]
        == "Trace current Watson client."
    )
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs


def test_real_current_watson_lifecycle_is_shared_and_foreign_safe() -> None:
    run_client = _current_run_client()
    original = run_client.create_run
    first = WatsonOrchestrateADKInstrumentor()
    second = WatsonOrchestrateADKInstrumentor()
    first.activate()
    installed = run_client.create_run
    second.activate()
    first.deactivate()
    assert run_client.create_run is installed
    second.deactivate()
    assert run_client.create_run is original

    first.activate()
    installed = run_client.create_run

    def foreign(self: Any, *args: Any, **kwargs: Any) -> Any:
        return installed(self, *args, **kwargs)

    run_client.create_run = foreign
    first.deactivate()
    assert run_client.create_run is foreign
    run_client.create_run = original


def test_watson_provider_error_keeps_status_and_redacts_without_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_client = _current_run_client()
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_watson_orchestrate_adk._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    class ProviderError(Exception):
        status_code = 401

        def __str__(self) -> str:
            raise AssertionError("instrumentation called hostile __str__")

    error = ProviderError("authorization=Basic abcdef")

    def fail(self: Any, message: str, **kwargs: Any) -> Any:
        del self, message, kwargs
        raise error

    monkeypatch.setattr(run_client, "create_run", fail)
    instrumentor = WatsonOrchestrateADKInstrumentor()
    instrumentor.activate()
    with pytest.raises(ProviderError) as caught:
        run_client.create_run(SimpleNamespace(), "hello")
    instrumentor.deactivate()
    assert caught.value is error
    assert len(emitted) == 1
    span = emitted[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["http.response.status_code"] == 401
    assert "abcdef" not in json.dumps(dict(span.attributes), sort_keys=True)


def test_watson_json_capture_is_utf8_bounded_redacted_and_hostile_safe() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError

        __repr__ = __str__

    value = json_dumps(
        {
            "password": "plain-secret",
            "nested": {"client_secret": "secret-value", "object": Hostile()},
            "unicode": "😀" * 20_000,
        }
    )
    assert len(value.encode("utf-8")) <= 16_000
    assert "plain-secret" not in value
    assert "secret-value" not in value
    assert json.loads(value)
