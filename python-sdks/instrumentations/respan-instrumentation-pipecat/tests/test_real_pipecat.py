from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import StatusCode
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService, LLMSettings
from pipecat.workers.runner import WorkerRunner
from respan_instrumentation_pipecat import PipecatInstrumentor, _instrumentation
from respan_instrumentation_pipecat._serialization import json_dumps
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class OfflineLLMService(LLMService):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(
            name="OfflineLLMService",
            settings=LLMSettings(
                model="offline-pipecat",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            ),
        )
        self._fail = fail

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        if self._fail:
            error = ProviderAuthError("provider rejected request", status_code=401)
            await self.push_frame(
                ErrorFrame(
                    error="provider rejected request",
                    exception=error,
                    processor=self,
                )
            )
            return
        for chunk in ("Pi", "pecat tracing ", "is connected."):
            await self.push_frame(LLMTextFrame(chunk))
        await self.start_llm_usage_metrics(
            LLMTokenUsage(prompt_tokens=7, completion_tokens=5, total_tokens=12)
        )
        await self.push_frame(LLMFullResponseEndFrame())


class ProviderAuthError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class Collector(FrameProcessor):
    def __init__(self) -> None:
        super().__init__(name="collector", enable_direct_mode=True)
        self.done = asyncio.Event()
        self.text: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.text.append(frame.text)
        if isinstance(frame, (LLMFullResponseEndFrame, ErrorFrame)):
            self.done.set()
        await self.push_frame(frame, direction)


async def _run_pipeline(*, fail: bool) -> None:
    collector = Collector()
    worker = PipelineWorker(
        Pipeline([OfflineLLMService(fail=fail), collector]),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
        conversation_id="pipecat-real-test",
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    async def drive() -> None:
        await asyncio.sleep(0.05)
        await worker.queue_frame(
            LLMContextFrame(
                LLMContext(
                    messages=[{"role": "user", "content": "Trace a real Pipecat turn."}]
                )
            )
        )
        await asyncio.wait_for(collector.done.wait(), timeout=10)
        await worker.queue_frame(EndFrame())

    await asyncio.wait_for(asyncio.gather(runner.run(), drive()), timeout=15)


@pytest.mark.asyncio
async def test_real_current_pipecat_exports_success_and_provider_failure(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: provider)
    instrumentor = PipecatInstrumentor()
    instrumentor.activate()
    tracer = provider.get_tracer("respan.pipecat.test")
    try:
        with tracer.start_as_current_span("success.root"):
            await _run_pipeline(fail=False)
        with tracer.start_as_current_span("failure.root"):
            await _run_pipeline(fail=True)
    finally:
        instrumentor.deactivate()
        provider.force_flush()

    spans = list(exporter.get_finished_spans())
    names = Counter(span.name for span in spans)
    assert names == Counter(
        {
            "success.root": 1,
            "failure.root": 1,
            "pipecat.conversation.turn": 2,
            "pipecat.llm": 2,
        }
    )
    assert len({span.context.span_id for span in spans}) == len(spans)
    by_name: dict[str, list[Any]] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)

    roots = {span.name: span for span in spans if span.parent is None}
    assert set(roots) == {"success.root", "failure.root"}
    all_ids = {span.context.span_id for span in spans}
    for span in spans:
        if span.parent is not None:
            assert span.parent.span_id in all_ids

    turns = by_name["pipecat.conversation.turn"]
    success_turn = next(
        span for span in turns if span.status.status_code is not StatusCode.ERROR
    )
    failure_turn = next(
        span for span in turns if span.status.status_code is StatusCode.ERROR
    )
    assert success_turn.attributes[RESPAN_LOG_TYPE] == "workflow"
    assert (
        json.loads(success_turn.attributes[TLSpanAttributes.TRACELOOP_ENTITY_INPUT])
        == "Trace a real Pipecat turn."
    )
    assert (
        json.loads(success_turn.attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        == "Pipecat tracing is connected."
    )
    assert success_turn.attributes["conversation.end_reason"] == "completed"
    assert success_turn.attributes["conversation.was_interrupted"] is False

    success_llm = next(
        span
        for span in by_name["pipecat.llm"]
        if span.status.status_code is not StatusCode.ERROR
    )
    assert success_llm.attributes.get(TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7, (
        success_llm.attributes
    )
    assert success_llm.attributes[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 5

    assert failure_turn.attributes["http.response.status_code"] == 401
    assert failure_turn.attributes["status_code"] == 401
    assert failure_turn.attributes[ERROR_MESSAGE_ATTR] == "provider rejected request"
    failure_llm = next(
        span
        for span in by_name["pipecat.llm"]
        if span.status.status_code is StatusCode.ERROR
    )
    assert failure_llm.attributes["http.response.status_code"] == 401
    assert failure_llm.attributes[RESPAN_LOG_TYPE] == "chat"
    assert (
        json.loads(failure_llm.attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
            "message"
        ]
        == "provider rejected request"
    )
    assert all(
        TLSpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes for span in turns
    )

    provider.shutdown()


def test_serializer_is_bounded_redacted_and_never_calls_hostile_repr():
    class Hostile:
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

        def __str__(self) -> str:
            raise AssertionError("str must not run")

    value = {
        "client_secret": "plain-secret",
        "nested": {"auth_token": "token-value"},
        "hostile": Hostile(),
        "unicode": "😀" * 20_000,
    }
    encoded = json_dumps(value)
    parsed = json.loads(encoded)
    assert len(encoded.encode("utf-8")) <= 16_000
    assert "plain-secret" not in encoded
    assert "token-value" not in encoded
    assert parsed
