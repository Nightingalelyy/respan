import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_pytest._runtime import PytestRuntimePlugin
from respan_instrumentation_pytest._serialization import json_dumps
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.utils.span_factory import read_propagated_attributes


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.status = None
        self.exceptions = []
        self.events = []
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        span = FakeSpan(name)
        self.spans.append(span)
        try:
            yield span
        finally:
            span.ended = True


class Outcome:
    def __init__(self, report):
        self.report = report

    def get_result(self):
        return self.report


def make_session():
    config = SimpleNamespace(rootpath=Path("/tmp/demo-suite"), args=["tests"])
    return SimpleNamespace(config=config, items=[])


def make_item():
    markers = [SimpleNamespace(name="asyncio"), SimpleNamespace(name="integration")]
    return SimpleNamespace(
        nodeid="tests/test_checkout.py::test_total[usd]",
        name="test_total[usd]",
        callspec=SimpleNamespace(params={"currency": "usd", "amount": 42}),
        fixturenames=["database", "currency"],
        iter_markers=lambda: iter(markers),
    )


def add_report(plugin, item, *, when, outcome, exc=None):
    report = SimpleNamespace(
        when=when,
        outcome=outcome,
        failed=outcome == "failed",
        skipped=outcome == "skipped",
        passed=outcome == "passed",
        duration=0.01,
    )
    call = SimpleNamespace(
        excinfo=SimpleNamespace(value=exc) if exc is not None else None
    )
    hook = plugin.pytest_runtest_makereport(item, call)
    next(hook)
    try:
        hook.send(Outcome(report))
    except StopIteration:
        pass


def finish_protocol(protocol):
    try:
        protocol.send(None)
    except StopIteration:
        pass


def assert_contract(attrs, log_type):
    assert attrs[RESPAN_LOG_TYPE] == log_type
    assert SpanAttributes.TRACELOOP_ENTITY_NAME in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_PATH in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in attrs
    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME]
    assert attrs[RESPAN_TRACE_GROUP_ID]
    assert attrs[f"{RESPAN_METADATA}.integration"] == "pytest"
    assert attrs[f"{RESPAN_METADATA}.example_run_id"]
    for banned in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "span_tools",
        "has_tool_calls",
        "respan.span.tools",
        "respan.span.tool_calls",
        "respan.span.handoffs",
    ):
        assert banned not in attrs


def test_session_and_passing_test_emit_contract_spans():
    tracer = FakeTracer()
    plugin = PytestRuntimePlugin(
        tracer=tracer, workflow_name="checkout_pytest_workflow"
    )
    plugin.activate()
    session = make_session()
    plugin.pytest_sessionstart(session)
    propagated = read_propagated_attributes()
    assert propagated[f"{RESPAN_METADATA}.example_run_id"]
    assert (
        propagated[f"{RESPAN_METADATA}.run_id"]
        == propagated[f"{RESPAN_METADATA}.example_run_id"]
    )
    assert propagated[f"{RESPAN_METADATA}.workflow_name"] == (
        "checkout_pytest_workflow"
    )
    item = make_item()
    protocol = plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    for phase in ("setup", "call", "teardown"):
        add_report(plugin, item, when=phase, outcome="passed")
    finish_protocol(protocol)
    plugin.pytest_sessionfinish(session, 0)

    assert [span.name for span in tracer.spans] == ["pytest.session", "pytest.test"]
    session_span, test_span = tracer.spans
    assert_contract(session_span.attributes, "workflow")
    assert_contract(test_span.attributes, "task")
    assert test_span.attributes["status_code"] == 200
    assert (
        json.loads(test_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
            "outcome"
        ]
        == "passed"
    )
    assert (
        json.loads(test_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
            "parameters"
        ]["currency"]
        == "usd"
    )
    assert session_span.ended and test_span.ended


def test_failure_records_status_error_and_phase():
    tracer = FakeTracer()
    plugin = PytestRuntimePlugin(tracer=tracer)
    plugin.activate()
    session = make_session()
    plugin.pytest_sessionstart(session)
    item = make_item()
    protocol = plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    add_report(plugin, item, when="setup", outcome="passed")
    add_report(
        plugin,
        item,
        when="call",
        outcome="failed",
        exc=AssertionError("expected 42, got 41"),
    )
    add_report(plugin, item, when="teardown", outcome="passed")
    finish_protocol(protocol)

    span = tracer.spans[-1]
    assert_contract(span.attributes, "task")
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "expected 42, got 41"
    assert (
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["error"][
            "phase"
        ]
        == "call"
    )
    assert span.events[0][0] == "exception"
    plugin.pytest_sessionfinish(session, 1)
    assert read_propagated_attributes() == {}
    assert tracer.spans[0].status.status_code is StatusCode.ERROR


def test_capture_disabled_omits_params_and_failure_text():
    tracer = FakeTracer()
    plugin = PytestRuntimePlugin(tracer=tracer, capture_content=False)
    plugin.activate()
    session = make_session()
    plugin.pytest_sessionstart(session)
    item = make_item()
    protocol = plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    add_report(
        plugin,
        item,
        when="call",
        outcome="failed",
        exc=ValueError("secret customer value"),
    )
    finish_protocol(protocol)

    attrs = tracer.spans[-1].attributes
    assert "secret customer value" not in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert '"parameters"' not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert "usd" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert "usd" not in attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert "usd" not in attrs[SpanAttributes.TRACELOOP_ENTITY_PATH]
    assert attrs["error.message"] == "ValueError"
    assert not tracer.spans[-1].exceptions
    plugin.pytest_sessionfinish(session, 1)


def test_lifecycle_is_idempotent():
    plugin = PytestRuntimePlugin(tracer=FakeTracer())
    plugin.activate()
    plugin.activate()
    assert plugin._is_instrumented
    plugin.deactivate()
    plugin.deactivate()
    assert not plugin._is_instrumented


def test_serializer_is_bounded_redacting_and_hostile_safe():
    class Hostile:
        def __str__(self):
            raise AssertionError("hostile __str__ called")

        def __repr__(self):
            raise AssertionError("hostile __repr__ called")

    encoded = json_dumps(
        {
            "api_key": "plain-secret",
            "nested": {"session_token": "nested-secret"},
            "hostile": Hostile(),
            "unicode": "😀" * 10_000,
        }
    )
    assert len(encoded.encode("utf-8")) <= 16_000
    assert "plain-secret" not in encoded
    assert "nested-secret" not in encoded
    assert json.loads(encoded)


def test_real_otel_export_preserves_metadata_grouping_parentage_and_failure():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = PytestRuntimePlugin(
        tracer=provider.get_tracer("pytest-contract"),
        workflow_name="checkout_pytest_workflow",
    )
    plugin.activate()
    session = make_session()
    plugin.pytest_sessionstart(session)
    item = make_item()
    protocol = plugin.pytest_runtest_protocol(item, None)
    next(protocol)
    add_report(plugin, item, when="setup", outcome="passed")
    add_report(
        plugin,
        item,
        when="call",
        outcome="failed",
        exc=AssertionError("expected 42, got 41"),
    )
    add_report(plugin, item, when="teardown", outcome="passed")
    finish_protocol(protocol)
    plugin.pytest_sessionfinish(session, 1)
    assert provider.force_flush()

    spans = list(exporter.get_finished_spans())
    assert [span.name for span in spans] == ["pytest.test", "pytest.session"]
    test_span, session_span = spans
    assert test_span.parent.span_id == session_span.context.span_id
    assert test_span.context.trace_id == session_span.context.trace_id
    assert session_span.parent is None
    assert (
        json.loads(session_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
            "rootpath"
        ]
        == "demo-suite"
    )
    assert len({span.context.span_id for span in spans}) == 2
    for span in spans:
        attrs = span.attributes
        assert attrs[f"{RESPAN_METADATA}.integration"] == "pytest"
        assert attrs[f"{RESPAN_METADATA}.workflow_name"] == ("checkout_pytest_workflow")
        assert attrs[f"{RESPAN_METADATA}.example_run_id"]
        assert (
            attrs[f"{RESPAN_METADATA}.run_id"]
            == attrs[f"{RESPAN_METADATA}.example_run_id"]
        )
        assert attrs[RESPAN_TRACE_GROUP_ID] == "checkout_pytest_workflow"
        assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == (
            "checkout_pytest_workflow"
        )
        assert span.status.status_code is StatusCode.ERROR
    assert test_span.events[0].name == "exception"
    assert test_span.events[0].attributes["exception.type"] == "AssertionError"
    provider.shutdown()
