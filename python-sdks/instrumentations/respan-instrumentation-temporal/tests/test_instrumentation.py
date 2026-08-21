from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace

import pytest
from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode
from respan_instrumentation_temporal import TemporalInstrumentor, _instrumentation
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.utils.span_factory import propagate_attributes
from temporalio.contrib.opentelemetry import TracingInterceptor


class _FakeSpan:
    def __init__(self, name: str, attributes=None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.status = None
        self.exceptions = []
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc, *args, **kwargs):
        self.exceptions.append(exc)

    def end(self, *args, **kwargs):
        self.ended = True

    def get_span_context(self):
        return _instrumentation.trace.INVALID_SPAN_CONTEXT

    def is_recording(self):
        return True


class _FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, *args, **kwargs):
        span = _FakeSpan(name, kwargs.get("attributes"))
        self.spans.append(span)
        yield span

    def start_span(self, name, *args, **kwargs):
        span = _FakeSpan(name, kwargs.get("attributes"))
        self.spans.append(span)
        return span


@pytest.fixture(autouse=True)
def reset_instrumentor():
    TemporalInstrumentor._patches_applied = False
    TemporalInstrumentor._activation_count = 0
    TemporalInstrumentor._client_class = None
    TemporalInstrumentor._original_connect_descriptor_holder = ()
    TemporalInstrumentor._installed_connect_function = None
    TemporalInstrumentor._shared_config = None
    yield
    TemporalInstrumentor._patches_applied = False
    TemporalInstrumentor._activation_count = 0
    TemporalInstrumentor._client_class = None
    TemporalInstrumentor._original_connect_descriptor_holder = ()
    TemporalInstrumentor._installed_connect_function = None
    TemporalInstrumentor._shared_config = None


def _make_interceptor(*, capture_content=True):
    tracer = _FakeTracer()
    interceptor = _instrumentation._build_interceptor(
        TracingInterceptor,
        tracer=tracer,
        capture_content=capture_content,
        max_attribute_chars=16_000,
        always_create_workflow_spans=False,
    )
    return tracer, interceptor


def _assert_contract(attrs, log_type):
    assert attrs[RESPAN_LOG_TYPE] == log_type
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert SpanAttributes.TRACELOOP_ENTITY_PATH in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in attrs
    for banned_alias in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "span_tools",
        "has_tool_calls",
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    ):
        assert banned_alias not in attrs


def test_workflow_start_maps_to_canonical_workflow_span():
    tracer, interceptor = _make_interceptor()
    temporal_input = SimpleNamespace(
        args=("Ada",),
        id="greeting-workflow-1",
        workflow="GreetingWorkflow",
        headers={},
    )

    with interceptor._start_as_current_span(
        "StartWorkflow:GreetingWorkflow",
        attributes={"temporalWorkflowID": "greeting-workflow-1"},
        input_with_headers=temporal_input,
        kind=SpanKind.CLIENT,
    ):
        pass

    span = tracer.spans[-1]
    _assert_contract(span.attributes, "workflow")
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "temporal.start_workflow.GreetingWorkflow"
    )
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert span.attributes["respan.trace.trace_group_identifier"] == "GreetingWorkflow"
    assert (
        "greeting-workflow-1" in span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert "temporalWorkflowID" not in span.attributes
    assert span.attributes["status_code"] == 200


def test_activity_error_records_status_and_backend_error_fields():
    tracer, interceptor = _make_interceptor()

    with (
        pytest.raises(RuntimeError, match="activity exploded"),
        interceptor._start_as_current_span(
            "RunActivity:compose_greeting",
            attributes={"temporalActivityID": "activity-1"},
            kind=SpanKind.SERVER,
        ),
    ):
        raise RuntimeError("activity exploded")

    span = tracer.spans[-1]
    _assert_contract(span.attributes, "task")
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "activity exploded"
    assert (
        "activity exploded" in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )


def test_completed_workflow_span_proxy_handles_success_and_error():
    _tracer, interceptor = _make_interceptor()
    successful = interceptor.tracer.start_span(
        "CompleteWorkflow:GreetingWorkflow",
        attributes={"temporalRunID": "run-1"},
    )
    successful.end()
    assert successful.attributes[RESPAN_LOG_TYPE] == "workflow"
    assert successful.attributes["status_code"] == 200

    failed = interceptor.tracer.start_span(
        "CompleteWorkflow:BrokenWorkflow",
        attributes={"temporalRunID": "run-2"},
    )
    failed.record_exception(ValueError("workflow failed"))
    failed.end()
    assert failed.attributes["status_code"] == 500
    assert failed.attributes["error.message"] == "workflow failed"
    assert failed.ended


def test_capture_content_false_omits_args_ids_and_error_text():
    tracer, interceptor = _make_interceptor(capture_content=False)
    temporal_input = SimpleNamespace(
        args=("top-secret",),
        id="customer-workflow-id",
        workflow="SecretWorkflow",
        headers={},
    )
    with interceptor._start_as_current_span(
        "StartWorkflow:SecretWorkflow",
        attributes={"temporalWorkflowID": "customer-workflow-id"},
        input_with_headers=temporal_input,
        kind=SpanKind.CLIENT,
    ):
        pass

    attrs = tracer.spans[-1].attributes
    assert "top-secret" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert "customer-workflow-id" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert '"content_captured":false' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]


def test_temporal_serialization_is_bounded_and_redacts_sensitive_values():
    attrs = _instrumentation._canonical_attributes(
        "StartWorkflow:SecretWorkflow",
        {
            _instrumentation.TEMPORAL_CAPTURED_INPUT: {
                "api_key": "plain-secret",
                "nested": {"auth_token": "token"},
                "text": "😀" * 10_000,
            }
        },
        capture_content=True,
        max_attribute_chars=16_000,
    )
    encoded = attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT].encode("utf-8")
    assert len(encoded) <= 16_000
    assert b"plain-secret" not in encoded
    assert b"[REDACTED]" in encoded


def test_temporal_error_redaction_handles_quoted_json_credentials():
    message = _instrumentation.safe_error_message(
        RuntimeError('{"api_key":"plain-secret"}' + "😀" * 10_000),
        capture_content=True,
    )
    assert "plain-secret" not in message
    assert "[REDACTED]" in message
    assert len(message.encode("utf-8")) <= 4_000


def test_interceptor_uses_public_package_instrumentation_scope(monkeypatch):
    requested = []
    tracer = _FakeTracer()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda name, version=None: requested.append((name, version)) or tracer,
    )

    instrumentor = TemporalInstrumentor()
    assert instrumentor.interceptor is not None
    assert requested == [("temporal", "0.1.0")]


def test_respan_metadata_crosses_temporal_baggage_privately():
    extracted_parent = otel_context.set_value(
        "temporal-parent-sentinel",
        "preserved",
        otel_context.Context(),
    )
    with propagate_attributes(
        metadata={
            "example_run_id": "exact-marker",
            "api_key": "plain-secret",
        }
    ):
        outbound = _instrumentation._context_with_respan_baggage(extracted_parent)

    assert (
        otel_context.get_value("temporal-parent-sentinel", context=outbound)
        == "preserved"
    )
    assert (
        baggage.get_baggage("respan.metadata.example_run_id", context=outbound)
        == "exact-marker"
    )
    assert baggage.get_baggage("respan.metadata.api_key", context=outbound) == (
        "[REDACTED]"
    )

    tracer = _FakeTracer()
    canonical = _instrumentation._CanonicalTracer(
        tracer,
        capture_content=True,
        max_attribute_chars=16_000,
    )
    span = canonical.start_span(
        "RunWorkflow:GreetingWorkflow",
        context=outbound,
        attributes={"temporalWorkflowID": "workflow-1"},
    )
    span.end()
    attrs = tracer.spans[-1].attributes
    assert attrs["respan.metadata.example_run_id"] == "exact-marker"
    assert attrs["respan.metadata.api_key"] == "[REDACTED]"


def test_real_otel_provider_connects_activity_fallback_and_cleans_cache():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    canonical = _instrumentation._CanonicalTracer(
        provider.get_tracer("temporal-contract-test"),
        capture_content=True,
        max_attribute_chars=16_000,
    )
    workflow_id = "workflow-1"

    start = canonical.start_span(
        "StartWorkflow:GreetingWorkflow",
        attributes={"temporalWorkflowID": workflow_id},
    )
    start.end()
    activity = canonical.start_span(
        "StartActivity:compose_greeting",
        attributes={"temporalWorkflowID": workflow_id},
    )
    activity.end()
    with canonical.start_as_current_span(
        "RunActivity:compose_greeting",
        attributes={"temporalWorkflowID": workflow_id},
    ):
        pass
    complete = canonical.start_span(
        "CompleteWorkflow:GreetingWorkflow",
        attributes={"temporalWorkflowID": workflow_id},
    )
    complete.end()

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert len(spans) == 4
    assert by_name["RunActivity:compose_greeting"].parent.span_id == (
        by_name["StartActivity:compose_greeting"].context.span_id
    )
    assert by_name["StartActivity:compose_greeting"].parent.span_id == (
        by_name["StartWorkflow:GreetingWorkflow"].context.span_id
    )
    assert canonical._workflow_contexts == {}


@pytest.mark.asyncio
async def test_connect_injects_once_and_respects_existing_tracing_interceptor():
    instrumentor = TemporalInstrumentor()
    instrumentor._base_interceptor_class = TracingInterceptor
    instrumentor._interceptor = object()

    async def connect(*args, **kwargs):
        return kwargs

    result = await instrumentor._connect(connect, ("localhost:7233",), {})
    assert result["interceptors"] == [instrumentor._interceptor]

    existing = TracingInterceptor()
    result = await instrumentor._connect(
        connect,
        ("localhost:7233",),
        {"interceptors": [existing]},
    )
    assert result["interceptors"] == [existing]


@pytest.mark.asyncio
async def test_real_client_connect_injects_interceptor_without_network():
    from temporalio.client import Client

    original_descriptor = Client.__dict__["connect"]
    instrumentor = TemporalInstrumentor()
    instrumentor.activate()
    try:
        client = await Client.connect("localhost:7233", lazy=True)
        interceptors = client.config()["interceptors"]
        assert len(interceptors) == 1
        assert type(interceptors[0]).__name__ == "RespanTemporalTracingInterceptor"
    finally:
        instrumentor.deactivate()

    assert Client.__dict__["connect"] is original_descriptor


def test_activate_and_deactivate_are_idempotent(monkeypatch):
    class FakeClient:
        @classmethod
        async def connect(cls, *args, **kwargs):
            return args, kwargs

    client_module = SimpleNamespace(Client=FakeClient)
    real_import = import_module

    def fake_import(module_name):
        if module_name == _instrumentation.TEMPORAL_CLIENT_MODULE:
            return client_module
        return real_import(module_name)

    monkeypatch.setattr(_instrumentation.importlib, "import_module", fake_import)
    original_descriptor = FakeClient.__dict__["connect"]

    instrumentor = TemporalInstrumentor()
    monkeypatch.setattr(instrumentor, "_ensure_interceptor", lambda: object())
    instrumentor.activate()
    instrumentor.activate()
    installed_descriptor = FakeClient.__dict__["connect"]
    assert installed_descriptor is not original_descriptor
    assert instrumentor._is_instrumented

    observer = TemporalInstrumentor()
    monkeypatch.setattr(observer, "_ensure_interceptor", lambda: object())
    observer.activate()
    assert observer._is_instrumented
    assert TemporalInstrumentor._activation_count == 2

    instrumentor.deactivate()
    assert TemporalInstrumentor._patches_applied
    assert FakeClient.__dict__["connect"] is installed_descriptor
    assert not instrumentor._is_instrumented

    observer.deactivate()
    observer.deactivate()
    assert FakeClient.__dict__["connect"] is original_descriptor
    assert not observer._is_instrumented
    assert TemporalInstrumentor._activation_count == 0


def test_activation_rejects_mismatched_shared_configuration(monkeypatch):
    class FakeClient:
        @classmethod
        async def connect(cls, *args, **kwargs):
            return args, kwargs

    client_module = SimpleNamespace(Client=FakeClient)
    real_import = import_module
    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        lambda name: (
            client_module
            if name == _instrumentation.TEMPORAL_CLIENT_MODULE
            else real_import(name)
        ),
    )
    owner = TemporalInstrumentor(capture_content=True)
    monkeypatch.setattr(owner, "_ensure_interceptor", lambda: object())
    owner.activate()
    try:
        with pytest.raises(ValueError, match="different settings"):
            TemporalInstrumentor(capture_content=False).activate()
    finally:
        owner.deactivate()


def test_deactivate_preserves_later_foreign_connect_wrapper(monkeypatch):
    class FakeClient:
        @classmethod
        async def connect(cls, *args, **kwargs):
            return args, kwargs

    client_module = SimpleNamespace(Client=FakeClient)
    real_import = import_module
    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        lambda name: (
            client_module
            if name == _instrumentation.TEMPORAL_CLIENT_MODULE
            else real_import(name)
        ),
    )
    instrumentor = TemporalInstrumentor()
    monkeypatch.setattr(instrumentor, "_ensure_interceptor", lambda: object())
    instrumentor.activate()

    async def foreign_connect(cls, *args, **kwargs):
        return args, kwargs

    foreign_descriptor = classmethod(foreign_connect)
    FakeClient.connect = foreign_descriptor
    instrumentor.deactivate()
    assert FakeClient.__dict__["connect"] is foreign_descriptor
