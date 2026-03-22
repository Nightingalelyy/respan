"""OpenAI Agents SDK instrumentation plugin for Respan."""

import logging
from typing import Any, Optional

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

from ._otel_emitter import emit_sdk_item

logger = logging.getLogger(__name__)


class _RespanTracingProcessor(TracingProcessor):
    """OpenAI Agents SDK TracingProcessor that emits spans into the OTEL pipeline."""

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        emit_sdk_item(trace)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        emit_sdk_item(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class OpenAIAgentsInstrumentor:
    """Respan instrumentor for the OpenAI Agents SDK.

    Registers a ``TracingProcessor`` that converts OpenAI Agents SDK
    traces/spans to OTEL ``ReadableSpan`` objects and injects them into
    the single OTEL pipeline (``TracerProvider`` → ``RespanSpanExporter``
    → ``/v2/traces``).

    Usage::

        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

        respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])
    """

    name = "openai-agents"

    def __init__(self) -> None:
        self._processor: Optional[_RespanTracingProcessor] = None

    def activate(self) -> None:
        """Register the tracing processor with the OpenAI Agents SDK.

        Replaces the default OpenAI backend processor so traces are only
        sent to Respan, not to OpenAI's tracing endpoint.
        """
        from agents.tracing import set_trace_processors

        self._processor = _RespanTracingProcessor()
        set_trace_processors([self._processor])
        logger.info("OpenAI Agents SDK instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        self._processor = None
        logger.info("OpenAI Agents SDK instrumentation deactivated")


