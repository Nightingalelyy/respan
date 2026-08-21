"""Lifecycle and streaming tests for the OpenAI Agents instrumentor."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from agents.tracing import (
    agent_span,
    generation_span,
    set_trace_processors,
    task_span,
    trace,
    turn_span,
)
from agents.tracing.processor_interface import TracingProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from respan_sdk.constants.span_attributes import RESPAN_METADATA

from respan_instrumentation_openai_agents import _instrumentation
from respan_instrumentation_openai_agents._instrumentation import (
    OpenAIAgentsInstrumentor,
    _RespanTracingProcessor,
)


class _SentinelProcessor(TracingProcessor):
    def on_trace_start(self, trace: Any) -> None:
        pass

    def on_trace_end(self, trace: Any) -> None:
        pass

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _restore_processors():
    previous = _instrumentation._current_processors()
    yield
    for _ in range(100):
        if not _instrumentation._ACTIVATION_COUNT:
            break
        active = OpenAIAgentsInstrumentor()
        active._is_active = True
        active.deactivate()
    set_trace_processors(list(previous))
    _instrumentation._remove_stream_patches()


def test_repeated_activation_is_ref_counted_and_restores_previous_processor():
    sentinel = _SentinelProcessor()
    set_trace_processors([sentinel])
    first = OpenAIAgentsInstrumentor()
    second = OpenAIAgentsInstrumentor()

    first.activate()
    first.activate()
    second.activate()

    current = _instrumentation._current_processors()
    assert len(current) == 1
    assert current[0] is first._processor is second._processor
    assert _instrumentation._ACTIVATION_COUNT == 2

    first.deactivate()
    assert _instrumentation._current_processors() == current
    second.deactivate()
    assert _instrumentation._current_processors() == (sentinel,)
    assert _instrumentation._ACTIVATION_COUNT == 0


def test_concurrent_instances_share_one_processor_and_restore_cleanly():
    sentinel = _SentinelProcessor()
    set_trace_processors([sentinel])
    instances = [OpenAIAgentsInstrumentor() for _ in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: item.activate(), instances))

    current = _instrumentation._current_processors()
    assert len(current) == 1
    assert len({id(item._processor) for item in instances}) == 1
    assert _instrumentation._ACTIVATION_COUNT == len(instances)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: item.deactivate(), instances))

    assert _instrumentation._current_processors() == (sentinel,)
    assert _instrumentation._ACTIVATION_COUNT == 0


def test_deactivation_merges_preserved_and_late_processors():
    original = _SentinelProcessor()
    late = _SentinelProcessor()
    set_trace_processors([original])
    instrumentor = OpenAIAgentsInstrumentor()

    instrumentor.activate()
    shared = instrumentor._processor
    set_trace_processors([shared, late])
    instrumentor.deactivate()

    assert _instrumentation._current_processors() == (original, late)


def test_real_otel_provider_receives_complete_agents_trace(monkeypatch):
    """Exercise the actual SDK processor -> OTEL provider delivery boundary."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        _instrumentation.otel_trace,
        "get_tracer_provider",
        lambda: provider,
    )
    processor = _RespanTracingProcessor(
        metadata={"example_run_id": "otel-delivery-marker"}
    )
    set_trace_processors([processor])

    try:
        with (
            trace("delivery"),
            task_span("delivery task"),
            agent_span("Delivery Agent"),
            turn_span(1, "Delivery Agent"),
            generation_span(
                input=[{"role": "user", "content": "Hello"}],
                output=[{"role": "assistant", "content": "Hi"}],
                model="gpt-4o-mini",
                usage={"input_tokens": 1, "output_tokens": 1},
            ),
        ):
            pass
        processor.force_flush()
        spans = exporter.get_finished_spans()
    finally:
        processor.shutdown()
        provider.shutdown()

    assert {span.name for span in spans} == {
        "Delivery Agent.agent",
        "delivery task.task",
        "delivery.workflow",
        "openai.chat",
        "turn-1.task",
    }
    assert len({span.get_span_context().trace_id for span in spans}) == 1
    assert all(
        json.loads(span.attributes[RESPAN_METADATA])["example_run_id"]
        == "otel-delivery-marker"
        for span in spans
    )
    assert all(
        span.attributes[f"{RESPAN_METADATA}.example_run_id"] == "otel-delivery-marker"
        for span in spans
    )


@pytest.mark.asyncio
async def test_stream_wrapper_marks_only_the_wrapped_async_iterator():
    observations: list[bool] = []
    source_closed = False

    async def original():
        nonlocal source_closed
        observations.append(_instrumentation._STREAMING.get())
        try:
            yield "chunk"
            observations.append(_instrumentation._STREAMING.get())
        finally:
            source_closed = True

    wrapped = _instrumentation._wrap_stream_method(original)
    assert [chunk async for chunk in wrapped()] == ["chunk"]
    assert observations == [True, True]
    assert source_closed is True
    assert _instrumentation._STREAMING.get() is False


@pytest.mark.asyncio
async def test_stream_wrapper_closes_source_on_early_close():
    source_closed = False

    async def original():
        nonlocal source_closed
        try:
            yield "first"
            yield "second"
        finally:
            source_closed = True

    iterator = _instrumentation._wrap_stream_method(original)()
    assert await anext(iterator) == "first"
    await iterator.aclose()

    assert source_closed is True
    assert _instrumentation._STREAMING.get() is False


@pytest.mark.asyncio
async def test_stream_wrapper_surfaces_source_close_error_without_context_leak():
    async def original():
        try:
            yield "first"
        finally:
            raise RuntimeError("source close failed")

    iterator = _instrumentation._wrap_stream_method(original)()
    assert await anext(iterator) == "first"
    with pytest.raises(RuntimeError, match="source close failed"):
        await iterator.aclose()

    assert _instrumentation._STREAMING.get() is False
