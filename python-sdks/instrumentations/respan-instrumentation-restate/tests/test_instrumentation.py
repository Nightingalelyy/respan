import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import respan_instrumentation_restate._instrumentation as instrumentation
import restate
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_restate import RestateInstrumentor
from respan_instrumentation_restate._serialization import json_string
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)


class FakeSpan:
    def __init__(self, attributes: dict):
        self.attributes = dict(attributes)
        self.status = None
        self.exceptions = []
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exception):
        self.exceptions.append(exception)

    def add_event(self, name, attributes):
        self.events.append((name, attributes))


class FakeTracer:
    def __init__(self):
        self.span = None

    @contextmanager
    def start_as_current_span(self, name, *, attributes, **kwargs):
        del name, kwargs
        self.span = FakeSpan(attributes)
        yield self.span


class JsonSerde:
    def deserialize(self, value: bytes):
        return {"name": value.decode()}


def _fake_context():
    return SimpleNamespace(
        handler=SimpleNamespace(
            service_tag=SimpleNamespace(
                kind="workflow",
                name="OrderWorkflow",
                metadata={"team": "payments"},
            ),
            name="run",
            kind="workflow",
            metadata={"purpose": "checkout"},
            handler_io=SimpleNamespace(input_serde=JsonSerde()),
        ),
        invocation=SimpleNamespace(
            invocation_id="inv-123",
            input_buffer=b"Ada",
            key="order-42",
            scope="tenant",
            limit_key="tenant/ada",
            idempotency_key="checkout-1",
        ),
    )


def test_context_manager_maps_restate_invocation_fields(monkeypatch) -> None:
    fake_tracer = FakeTracer()
    context = _fake_context()
    server_context = SimpleNamespace(
        current_context=lambda: context,
        restate_context_is_replaying=SimpleNamespace(get=lambda: True),
    )
    real_import = instrumentation.importlib.import_module

    def import_module(name: str):
        if name == "restate.server_context":
            return server_context
        return real_import(name)

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    monkeypatch.setattr(
        instrumentation.trace, "get_tracer", lambda *args, **kwargs: fake_tracer
    )
    monkeypatch.setattr(instrumentation, "_ENABLED", True)
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", True)

    async def run():
        async with instrumentation._invocation_context():
            pass

    asyncio.run(run())
    attrs = fake_tracer.span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "workflow"
    assert attrs[RESPAN_TRACE_GROUP_ID] == "inv-123"
    assert attrs[RESPAN_THREADS_ID] == "order-42"
    assert '"input":{"name":"Ada"}' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert '"replaying":true' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert attrs["status_code"] == 200


def test_context_manager_records_handler_failure(monkeypatch) -> None:
    fake_tracer = FakeTracer()
    server_context = SimpleNamespace(
        current_context=_fake_context,
        restate_context_is_replaying=SimpleNamespace(get=lambda: False),
    )
    real_import = instrumentation.importlib.import_module

    def import_module(name: str):
        if name == "restate.server_context":
            return server_context
        return real_import(name)

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)
    monkeypatch.setattr(
        instrumentation.trace, "get_tracer", lambda *args, **kwargs: fake_tracer
    )
    monkeypatch.setattr(instrumentation, "_ENABLED", True)

    async def run():
        try:
            async with instrumentation._invocation_context():
                raise RuntimeError("deterministic Restate failure")
        except RuntimeError:
            pass

    asyncio.run(run())
    attrs = fake_tracer.span.attributes
    assert attrs["status_code"] == 500
    assert attrs["error.message"] == "deterministic Restate failure"
    assert fake_tracer.span.events[0][0] == "exception"


def test_registration_injects_context_only_once() -> None:
    instance = SimpleNamespace(context_managers=None)
    instrumentation._ensure_context_manager(instance)
    instrumentation._ensure_context_manager(instance)
    assert instance.context_managers == [instrumentation._invocation_context]


def test_activate_and_deactivate_patch_all_restate_registration_paths(
    monkeypatch,
) -> None:
    installed = 0
    removed = 0
    real_import = instrumentation.importlib.import_module

    def import_module(name: str):
        if name == "restate":
            return SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(instrumentation.importlib, "import_module", import_module)

    def install() -> None:
        nonlocal installed
        installed += 1

    def remove() -> None:
        nonlocal removed
        removed += 1

    monkeypatch.setattr(instrumentation, "_install_patches", install)
    monkeypatch.setattr(instrumentation, "_remove_patches", remove)
    monkeypatch.setattr(instrumentation, "_ACTIVATION_COUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHED_TARGETS", [])
    monkeypatch.setattr(instrumentation, "_ENABLED", False)

    adapter = RestateInstrumentor()
    adapter.activate()
    assert installed == 1
    adapter.deactivate()
    assert removed == 1


def test_real_current_restate_registration_exports_connected_readable_span(
    monkeypatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("restate", "0.1.0"),
    )
    monkeypatch.setattr(instrumentation, "_ACTIVATION_COUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHED_TARGETS", [])
    monkeypatch.setattr(instrumentation, "_ENABLED", False)

    adapter = RestateInstrumentor()
    adapter.activate()
    workflow = restate.Workflow("CheckoutWorkflow", metadata={"team": "payments"})

    @workflow.main(name="run")
    async def run(_ctx, request: dict) -> dict:
        return {"accepted": request["order_id"]}

    handler = workflow.handlers["run"]
    assert instrumentation._invocation_context in handler.context_managers

    context = SimpleNamespace(
        handler=handler,
        invocation=SimpleNamespace(
            invocation_id="inv-real-123",
            input_buffer=b'{"order_id":"order-42"}',
            key="order-42",
            scope=None,
            limit_key=None,
            idempotency_key=None,
        ),
    )
    server_context = SimpleNamespace(
        current_context=lambda: context,
        restate_context_is_replaying=SimpleNamespace(get=lambda: False),
    )
    real_import = instrumentation.importlib.import_module
    monkeypatch.setattr(
        instrumentation.importlib,
        "import_module",
        lambda name: (
            server_context if name == "restate.server_context" else real_import(name)
        ),
    )

    async def invoke() -> None:
        with provider.get_tracer("test").start_as_current_span("outer"):
            async with instrumentation._invocation_context():
                pass

    asyncio.run(invoke())
    adapter.deactivate()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "restate.workflow.CheckoutWorkflow.run",
        "outer",
    ]
    child, root = spans
    assert child.parent.span_id == root.context.span_id
    assert child.status.status_code is StatusCode.OK
    assert child.instrumentation_scope.name == "restate"
    attrs = child.attributes
    assert attrs[RESPAN_LOG_TYPE] == "workflow"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "CheckoutWorkflow.run"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])["input"] == {
        "order_id": "order-42"
    }
    assert "traceloop.span.kind" not in attrs


def test_serialization_is_bounded_redacted_and_hostile_safe() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError("must not stringify")

        def __repr__(self) -> str:
            raise AssertionError("must not repr")

    payload = {
        "client_secret": "plain-secret",
        "url": "https://user:password@example.com/restate?token=abc",
        "emoji": "😀" * 5_000,
        "hostile": Hostile(),
        "nonfinite": float("nan"),
    }
    encoded = json_string(payload)
    assert len(encoded.encode("utf-8")) <= 16_000
    assert "plain-secret" not in encoded
    assert "password" not in encoded
    assert "token=abc" not in encoded
    assert json.loads(encoded)["nonfinite"] is None


def test_lifecycle_rejects_mismatched_capture_config(monkeypatch) -> None:
    monkeypatch.setattr(instrumentation, "_ACTIVATION_COUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHED_TARGETS", [])
    monkeypatch.setattr(instrumentation, "_install_patches", lambda: None)
    monkeypatch.setattr(instrumentation, "_remove_patches", lambda: None)

    first = RestateInstrumentor(capture_content=True)
    second = RestateInstrumentor(capture_content=False)
    first.activate()
    with pytest.raises(ValueError, match="capture_content"):
        second.activate()
    first.deactivate()
