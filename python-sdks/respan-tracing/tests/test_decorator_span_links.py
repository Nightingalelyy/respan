import pytest
from respan_tracing import (
    RespanTelemetry,
    SpanLink,
    attach_span_links,
    task,
    workflow,
    agent,
    tool,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter


@pytest.fixture(scope="module")
def telemetry_env():
    RespanTracer.reset_instance()

    exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="decorator-link-tests",
        is_enabled=True,
        is_batching_enabled=False,
    )
    telemetry.add_processor(exporter=exporter, is_batching_enabled=False)

    yield telemetry, exporter

    exporter.clear()
    RespanTracer.reset_instance()


@pytest.fixture
def clean_exporter(telemetry_env):
    _, exporter = telemetry_env
    exporter.clear()
    yield telemetry_env
    exporter.clear()


def _make_link(suffix: str = "a") -> SpanLink:
    return SpanLink(
        trace_id=suffix * 32,
        span_id=suffix * 16,
        attributes={"link.type": f"test-{suffix}"},
    )


# --- Static links on decorators ---


def test_task_static_links(clean_exporter):
    """Static SpanLink list passed to @task decorator."""
    telemetry, exporter = clean_exporter
    link = _make_link("a")

    @task(name="static_link_task", links=[link])
    def my_task():
        return "ok"

    my_task()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1
    assert format(spans[0].links[0].context.trace_id, "032x") == "a" * 32
    assert spans[0].links[0].attributes == {"link.type": "test-a"}


def test_workflow_static_links(clean_exporter):
    """Static SpanLink list passed to @workflow decorator."""
    telemetry, exporter = clean_exporter
    link = _make_link("b")

    @workflow(name="static_link_workflow", links=[link])
    def my_workflow():
        return "ok"

    my_workflow()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1
    assert format(spans[0].links[0].context.span_id, "016x") == "b" * 16


# --- Dynamic links (callable) ---


def test_task_dynamic_links(clean_exporter):
    """Callable links resolved at call time."""
    telemetry, exporter = clean_exporter
    call_count = 0

    def get_links():
        nonlocal call_count
        call_count += 1
        return [_make_link("c")]

    @task(name="dynamic_link_task", links=get_links)
    def my_task():
        return "ok"

    my_task()
    my_task()
    telemetry.flush()

    assert call_count == 2
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert len(span.links) == 1
        assert format(span.links[0].context.trace_id, "032x") == "c" * 32


# --- Context-based attach_span_links() ---


def test_attach_span_links_context(clean_exporter):
    """Links attached via attach_span_links() are picked up by decorated span."""
    telemetry, exporter = clean_exporter

    @workflow(name="context_link_workflow")
    def my_workflow():
        return "ok"

    attach_span_links([_make_link("d")])
    my_workflow()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1
    assert format(spans[0].links[0].context.trace_id, "032x") == "d" * 32
    assert spans[0].links[0].attributes == {"link.type": "test-d"}


def test_attach_span_links_consumed_once(clean_exporter):
    """Context links are consumed on first decorated call, not repeated."""
    telemetry, exporter = clean_exporter

    @task(name="consume_once_task")
    def my_task():
        return "ok"

    attach_span_links([_make_link("e")])
    my_task()  # Should consume the link
    my_task()  # Should have no links
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert len(spans[0].links) == 1  # First call got the link
    assert len(spans[1].links) == 0  # Second call: no links


# --- Merging explicit + context links ---


def test_links_merge_explicit_and_context(clean_exporter):
    """Explicit decorator links and context-attached links are merged."""
    telemetry, exporter = clean_exporter
    explicit_link = _make_link("f")

    @task(name="merge_link_task", links=[explicit_link])
    def my_task():
        return "ok"

    context_link = _make_link("a")
    attach_span_links([context_link])
    my_task()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 2
    trace_ids = {format(link.context.trace_id, "032x") for link in spans[0].links}
    assert trace_ids == {"f" * 32, "a" * 32}


# --- No links (regression) ---


def test_no_links_regression(clean_exporter):
    """Decorators without links still work (no regression)."""
    telemetry, exporter = clean_exporter

    @task(name="no_link_task")
    def my_task():
        return 42

    result = my_task()
    telemetry.flush()

    assert result == 42
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 0


# --- Empty links list ---


def test_empty_links_list(clean_exporter):
    """Empty links list produces no links on span."""
    telemetry, exporter = clean_exporter

    @task(name="empty_link_task", links=[])
    def my_task():
        return "ok"

    my_task()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 0


def test_attach_empty_links_noop(clean_exporter):
    """attach_span_links([]) is a no-op."""
    telemetry, exporter = clean_exporter

    @task(name="noop_attach_task")
    def my_task():
        return "ok"

    attach_span_links([])
    my_task()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 0


# --- Async decorator support ---


@pytest.mark.asyncio
async def test_async_task_with_links(clean_exporter):
    """Links work with async decorated functions."""
    telemetry, exporter = clean_exporter
    link = _make_link("a")

    @task(name="async_link_task", links=[link])
    async def my_async_task():
        return "async ok"

    result = await my_async_task()
    telemetry.flush()

    assert result == "async ok"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1
    assert format(spans[0].links[0].context.trace_id, "032x") == "a" * 32


@pytest.mark.asyncio
async def test_async_attach_span_links(clean_exporter):
    """attach_span_links() works with async decorated functions."""
    telemetry, exporter = clean_exporter

    @workflow(name="async_context_workflow")
    async def my_async_workflow():
        return "ok"

    attach_span_links([_make_link("b")])
    await my_async_workflow()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1
    assert format(spans[0].links[0].context.span_id, "016x") == "b" * 16


# --- All 4 decorator types ---


def test_agent_decorator_links(clean_exporter):
    """@agent decorator supports links."""
    telemetry, exporter = clean_exporter

    @agent(name="linked_agent", links=[_make_link("a")])
    def my_agent():
        return "ok"

    my_agent()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1


def test_tool_decorator_links(clean_exporter):
    """@tool decorator supports links."""
    telemetry, exporter = clean_exporter

    @tool(name="linked_tool", links=[_make_link("a")])
    def my_tool():
        return "ok"

    my_tool()
    telemetry.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].links) == 1
