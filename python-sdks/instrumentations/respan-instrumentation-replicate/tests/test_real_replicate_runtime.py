from __future__ import annotations

import json
from collections import Counter

import httpx
import pytest
import replicate
from opentelemetry.semconv_ai import SpanAttributes
from replicate.exceptions import ReplicateError
from respan_instrumentation_replicate import ReplicateInstrumentor
from respan_instrumentation_replicate import _instrumentation as instrumentation
from respan_instrumentation_replicate._serialization import json_string
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer


def _prediction(identifier: str = "pred-1", *, prompt: str = "hello") -> dict:
    return {
        "id": identifier,
        "model": "owner/model",
        "version": "owner/model:version-1",
        "status": "succeeded",
        "input": {"prompt": prompt},
        "output": ["bounded ", "answer"],
        "error": None,
        "logs": "completed",
        "metrics": {"predict_time": 0.01},
        "urls": {
            "get": f"https://mock.replicate.local/v1/predictions/{identifier}",
            "cancel": f"https://mock.replicate.local/v1/predictions/{identifier}/cancel",
        },
        "created_at": "2026-08-18T00:00:00Z",
        "started_at": "2026-08-18T00:00:00Z",
        "completed_at": "2026-08-18T00:00:01Z",
    }


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    RespanTracer.reset_instance()
    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHES", [])
    monkeypatch.setattr(instrumentation, "_ENABLED", False)
    yield
    RespanTracer.reset_instance()


def test_real_current_sdk_exports_run_wait_and_management_without_duplicates(
    monkeypatch,
) -> None:
    emitted = []
    monkeypatch.setattr(
        instrumentation,
        "inject_span",
        lambda span: emitted.append(span) or True,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/predictions"):
            body = json.loads(request.content)
            return httpx.Response(
                201,
                json=_prediction(prompt=(body.get("input") or {}).get("prompt", "")),
            )
        if request.method == "GET" and path == "/v1/predictions":
            return httpx.Response(
                200,
                json={"next": None, "previous": None, "results": [_prediction()]},
            )
        if request.method == "GET" and path == "/v1/predictions/pred-1":
            return httpx.Response(200, json=_prediction())
        return httpx.Response(
            404, json={"detail": f"unhandled {request.method} {path}"}
        )

    adapter = ReplicateInstrumentor()
    adapter.activate()
    client = replicate.Client(
        api_token="fixture-token",
        base_url="https://mock.replicate.local",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.run("owner/model", input={"prompt": "hello"}) == [
            "bounded ",
            "answer",
        ]
        prediction = client.predictions.create(
            version="owner/model:version-1",
            input={"prompt": "lifecycle"},
        )
        prediction.wait()
        assert client.predictions.get("pred-1").id == "pred-1"
        assert len(client.predictions.list().results) == 1
    finally:
        adapter.deactivate()

    names = Counter(span.name for span in emitted)
    assert names == Counter(
        {
            "replicate.run": 1,
            "replicate.predictions.create": 1,
            "replicate.prediction.wait": 1,
            "replicate.predictions.get": 1,
            "replicate.predictions.list": 1,
        }
    )
    assert len({span.context.span_id for span in emitted}) == 5
    run = next(span for span in emitted if span.name == "replicate.run")
    wait = next(span for span in emitted if span.name == "replicate.prediction.wait")
    listed = next(span for span in emitted if span.name == "replicate.predictions.list")
    assert run.attributes[RESPAN_LOG_TYPE] == "text"
    assert run.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert wait.attributes[RESPAN_LOG_TYPE] == "task"
    assert SpanAttributes.LLM_REQUEST_MODEL not in wait.attributes
    assert listed.attributes[RESPAN_LOG_TYPE] == "task"
    listed_output = listed.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert "__orig_class__" not in listed_output
    assert "0x" not in listed_output
    assert all("traceloop.span.kind" not in span.attributes for span in emitted)
    assert all(span.instrumentation_scope.name == "replicate" for span in emitted)


def test_real_current_sdk_preserves_provider_status_and_safe_error(monkeypatch) -> None:
    emitted = []
    monkeypatch.setattr(
        instrumentation,
        "inject_span",
        lambda span: emitted.append(span) or True,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"detail": 'api_key="plain-secret" provider limited'},
        )

    adapter = ReplicateInstrumentor()
    adapter.activate()
    client = replicate.Client(
        api_token="fixture-token",
        base_url="https://mock.replicate.local",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ReplicateError):
            client.run("owner/model", input={"prompt": "fail"})
    finally:
        adapter.deactivate()

    assert len(emitted) == 1
    span = emitted[0]
    assert span.attributes["status_code"] == 429
    assert span.status.status_code.name == "ERROR"
    assert "plain-secret" not in span.attributes["error.message"]


def test_replicate_serialization_is_valid_bounded_and_private() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError("must not stringify")

        def __repr__(self) -> str:
            raise AssertionError("must not repr")

    encoded = json_string(
        {
            "auth_token": "plain-secret",
            "content": "😀" * 5_000,
            "hostile": Hostile(),
            "nonfinite": float("nan"),
        }
    )
    assert len(encoded.encode("utf-8")) <= 16_000
    assert "plain-secret" not in encoded
    assert json.loads(encoded)["nonfinite"] is None
