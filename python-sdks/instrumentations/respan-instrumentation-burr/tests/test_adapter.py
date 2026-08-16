import json
from contextvars import copy_context
from types import SimpleNamespace

from respan_instrumentation_burr._adapter import (
    _ACTIVE_SPANS,
    _BURR_METADATA_ATTRIBUTE,
    BurrLifecycleAdapter,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from opentelemetry.semconv_ai import SpanAttributes


class FakeSpan:
    def __init__(self, name: str, attributes: dict):
        self.name = name
        self.attributes = dict(attributes)
        self.events = []
        self.exceptions = []
        self.ended = False

    def get_span_context(self):
        return SimpleNamespace(is_valid=True, trace_id=1, span_id=2)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exception):
        self.exceptions.append(exception)

    def add_event(self, name, attributes):
        self.events.append((name, attributes))

    def end(self):
        self.ended = True


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, *, attributes):
        span = FakeSpan(name, attributes)
        self.spans.append(span)
        return span


class FakeState:
    def __init__(self, values):
        self.values = values

    def serialize(self):
        return dict(self.values)


def _action():
    return SimpleNamespace(
        name="increment",
        reads=["count"],
        writes=["count"],
        tags=["counter"],
        streaming=False,
        inputs=[],
    )


def test_application_and_action_hooks_capture_burr_state_machine_data() -> None:
    _ACTIVE_SPANS.set(())
    tracer = FakeTracer()
    adapter = BurrLifecycleAdapter(tracer=tracer)
    method = SimpleNamespace(value="run")

    adapter.pre_run_execute_call(
        app_id="app-123",
        partition_key="customer-42",
        state=FakeState({"count": 0}),
        method=method,
    )
    adapter.pre_run_step(
        app_id="app-123",
        partition_key="customer-42",
        sequence_id=1,
        state=FakeState({"count": 0}),
        action=_action(),
        inputs={},
    )
    adapter.post_run_step(
        state=FakeState({"count": 1}),
        result={"count": 1},
        exception=None,
    )
    adapter.post_run_execute_call(
        state=FakeState({"count": 1}),
        exception=None,
    )

    workflow, task = tracer.spans
    assert workflow.attributes[RESPAN_LOG_TYPE] == "workflow"
    assert task.attributes[RESPAN_LOG_TYPE] == "task"
    assert task.attributes[RESPAN_TRACE_GROUP_ID] == "app-123"
    assert task.attributes[RESPAN_THREADS_ID] == "customer-42"
    assert (
        '"reads": ["count"]' in task.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert (
        '"result": {"count": 1}'
        in task.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    assert workflow.ended and task.ended


def test_action_failure_uses_backend_error_contract() -> None:
    _ACTIVE_SPANS.set(())
    tracer = FakeTracer()
    adapter = BurrLifecycleAdapter(tracer=tracer)
    adapter.pre_run_step(
        app_id="app-failure",
        partition_key=None,
        sequence_id=7,
        state=FakeState({"count": 0}),
        action=_action(),
        inputs={},
    )
    error = RuntimeError("deterministic Burr failure")
    adapter.post_run_step(
        state=FakeState({"count": 0}),
        result=None,
        exception=error,
    )

    span = tracer.spans[0]
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "deterministic Burr failure"
    assert span.exceptions == [error]


def test_action_failure_fails_application_when_burr_drops_root_exception() -> None:
    _ACTIVE_SPANS.set(())
    tracer = FakeTracer()
    adapter = BurrLifecycleAdapter(tracer=tracer)
    method = SimpleNamespace(value="run")
    state = FakeState({"count": 0})

    adapter.pre_run_execute_call(
        app_id="app-failure",
        partition_key="customer-42",
        state=state,
        method=method,
    )
    adapter.pre_run_step(
        app_id="app-failure",
        partition_key="customer-42",
        sequence_id=1,
        state=state,
        action=_action(),
        inputs={},
    )
    error = RuntimeError("deterministic Burr failure")
    adapter.post_run_step(
        state=state,
        result=None,
        exception=error,
    )
    adapter.post_run_execute_call(state=state, exception=None)

    workflow, task = tracer.spans
    assert workflow.attributes["status_code"] == 500
    assert workflow.attributes["error.message"] == "deterministic Burr failure"
    assert '"status": "error"' in workflow.attributes[
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT
    ]
    assert task.attributes["status_code"] == 500
    assert workflow.exceptions == [error]
    assert _ACTIVE_SPANS.get() == ()


def test_nested_application_failure_does_not_leak_to_outer_application() -> None:
    _ACTIVE_SPANS.set(())
    tracer = FakeTracer()
    adapter = BurrLifecycleAdapter(tracer=tracer)
    method = SimpleNamespace(value="run")
    state = FakeState({"count": 0})

    adapter.pre_run_execute_call(
        app_id="outer",
        partition_key="outer-thread",
        state=state,
        method=method,
    )
    adapter.pre_run_execute_call(
        app_id="inner",
        partition_key="inner-thread",
        state=state,
        method=method,
    )
    adapter.pre_run_step(
        app_id="inner",
        partition_key="inner-thread",
        sequence_id=1,
        state=state,
        action=_action(),
        inputs={},
    )
    error = RuntimeError("inner failure")
    adapter.post_run_step(state=state, result=None, exception=error)
    adapter.post_run_execute_call(state=state, exception=None)
    adapter.post_run_execute_call(state=state, exception=None)

    outer, inner, task = tracer.spans
    assert inner.attributes["status_code"] == 500
    assert task.attributes["status_code"] == 500
    assert outer.attributes["status_code"] == 200
    assert _ACTIVE_SPANS.get() == ()


def test_concurrent_contexts_keep_application_failures_isolated() -> None:
    _ACTIVE_SPANS.set(())
    tracer = FakeTracer()
    adapter = BurrLifecycleAdapter(tracer=tracer)
    method = SimpleNamespace(value="arun")
    state = FakeState({"count": 0})
    failed_context = copy_context()
    successful_context = copy_context()

    def start(app_id: str) -> None:
        adapter.pre_run_execute_call(
            app_id=app_id,
            partition_key=app_id,
            state=state,
            method=method,
        )

    failed_context.run(start, "failed-app")
    successful_context.run(start, "successful-app")

    def fail() -> None:
        adapter.pre_run_step(
            app_id="failed-app",
            partition_key="failed-app",
            sequence_id=1,
            state=state,
            action=_action(),
            inputs={},
        )
        adapter.post_run_step(
            state=state,
            result=None,
            exception=RuntimeError("context failure"),
        )
        adapter.post_run_execute_call(state=state, exception=None)

    failed_context.run(fail)
    successful_context.run(
        adapter.post_run_execute_call,
        state=state,
        exception=None,
    )

    failed_workflow, successful_workflow, failed_task = tracer.spans
    assert failed_workflow.attributes["status_code"] == 500
    assert failed_task.attributes["status_code"] == 500
    assert successful_workflow.attributes["status_code"] == 200
    assert failed_context.run(_ACTIVE_SPANS.get) == ()
    assert successful_context.run(_ACTIVE_SPANS.get) == ()


def test_custom_span_attributes_and_stream_events_follow_burr_hooks() -> None:
    _ACTIVE_SPANS.set(())
    tracer = FakeTracer()
    adapter = BurrLifecycleAdapter(tracer=tracer)
    adapter.pre_start_span(
        action="respond",
        action_sequence_id=3,
        span=SimpleNamespace(name="retrieve_context"),
        span_dependencies=["prompt"],
        app_id="app-stream",
        partition_key="thread-1",
    )
    adapter.do_log_attributes(
        attributes={"documents": 2},
        tags={"phase": "retrieval"},
    )
    adapter.pre_start_stream(
        action="respond",
        sequence_id=3,
        app_id="app-stream",
        partition_key="thread-1",
    )
    adapter.post_stream_item(
        item={"text": "hello"},
        item_index=0,
        action="respond",
        sequence_id=3,
    )
    adapter.post_end_stream(action="respond", sequence_id=3)
    adapter.post_end_span()

    span = tracer.spans[0]
    assert [event[0] for event in span.events] == [
        "burr.stream.start",
        "burr.stream.item",
        "burr.stream.end",
    ]
    assert '"logged_attributes": {"documents": 2}' in span.attributes["respan.metadata"]
    assert json.loads(span.attributes[_BURR_METADATA_ATTRIBUTE])[
        "logged_attributes"
    ] == {"documents": 2}
    stream = json.loads(span.attributes[_BURR_METADATA_ATTRIBUTE])["stream"]
    assert stream["started"] is True
    assert stream["completed"] is True
    assert stream["item_count"] == 1
    assert stream["items"] == [{"index": 0, "value": {"text": "hello"}}]
