"""Claude Agent SDK OTEL instrumentation plugin for Respan."""

import logging
from typing import Any

from opentelemetry import trace

from respan_instrumentation_claude_agent_sdk._processor import (
    ClaudeAgentSDKSpanProcessor,
)

logger = logging.getLogger(__name__)


class ClaudeAgentSDKInstrumentor:
    """Respan instrumentor for the Claude Agent SDK."""

    name = "claude-agent-sdk"

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        capture_content: bool = False,
    ) -> None:
        self._agent_name = agent_name
        self._capture_content = capture_content
        self._otel_instrumentor = None
        self._processor = None
        self._is_instrumented = False

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: ClaudeAgentSDKSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if processors is not None:
            active_span_processor._span_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            active_span_processor._span_processors = (
                *active_span_processor._span_processors,
                processor,
            )
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: ClaudeAgentSDKSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    def activate(self) -> None:
        if self._is_instrumented:
            return

        try:
            from opentelemetry.instrumentation.claude_agent_sdk import (
                ClaudeAgentSdkInstrumentor,
            )
        except ImportError as exc:
            logger.warning(
                "Failed to activate Claude Agent SDK instrumentation — missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()
        if self._processor is None:
            self._processor = ClaudeAgentSDKSpanProcessor()
        self._register_processor(tracer_provider=tracer_provider, processor=self._processor)

        self._otel_instrumentor = ClaudeAgentSdkInstrumentor()
        try:
            self._otel_instrumentor.instrument(
                tracer_provider=tracer_provider,
                agent_name=self._agent_name,
                capture_content=self._capture_content,
            )
        except Exception as exc:
            self._unregister_processor(
                tracer_provider=tracer_provider,
                processor=self._processor,
            )
            self._otel_instrumentor = None
            logger.warning(
                "Failed to activate Claude Agent SDK instrumentation: %s",
                exc,
            )
            return

        self._is_instrumented = True
        logger.info("Claude Agent SDK instrumentation activated")

    def deactivate(self) -> None:
        if not self._is_instrumented:
            return

        try:
            if self._otel_instrumentor is not None:
                self._otel_instrumentor.uninstrument()
        finally:
            self._otel_instrumentor = None
            if self._processor is not None:
                self._unregister_processor(
                    tracer_provider=trace.get_tracer_provider(),
                    processor=self._processor,
                )

        self._is_instrumented = False
        logger.info("Claude Agent SDK instrumentation deactivated")
