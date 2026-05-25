"""Strands Agents OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from opentelemetry import trace

from respan_instrumentation_strands_agents._constants import (
    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN,
)
from respan_instrumentation_strands_agents._processor import (
    StrandsAgentsSpanProcessor,
)

logger = logging.getLogger(__name__)


class StrandsAgentsInstrumentor:
    """Respan instrumentor for Strands Agents."""

    name = "strands-agents"

    def __init__(self, *, include_tool_definitions: bool = True) -> None:
        self._include_tool_definitions = include_tool_definitions
        self._processor: StrandsAgentsSpanProcessor | None = None
        self._is_instrumented = False
        self._previous_semconv_opt_in: str | None = None

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: StrandsAgentsSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if active_span_processor is not None and processors is not None:
            remaining_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            active_span_processor._span_processors = (processor, *remaining_processors)
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: StrandsAgentsSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if active_span_processor is None or processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    def _enable_semconv_opt_ins(self) -> None:
        if not self._include_tool_definitions:
            return

        self._previous_semconv_opt_in = os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN")
        opt_in_values = {
            value.strip()
            for value in (self._previous_semconv_opt_in or "").split(",")
            if value.strip()
        }
        opt_in_values.add(STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN)
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = ",".join(sorted(opt_in_values))

    def _restore_semconv_opt_ins(self) -> None:
        if not self._include_tool_definitions:
            return
        if self._previous_semconv_opt_in is None:
            os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
        else:
            os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = self._previous_semconv_opt_in
        self._previous_semconv_opt_in = None

    @staticmethod
    def _refresh_strands_tracer(tracer_module: Any, tracer_provider: Any) -> None:
        tracer_instance = getattr(tracer_module, "_tracer_instance", None)
        if tracer_instance is None:
            return

        tracer_instance.tracer_provider = tracer_provider
        service_name = getattr(tracer_instance, "service_name", tracer_module.__name__)
        tracer_instance.tracer = tracer_provider.get_tracer(service_name)
        parse_opt_in = getattr(tracer_instance, "_parse_semconv_opt_in", None)
        if callable(parse_opt_in):
            opt_in_values = parse_opt_in()
            if hasattr(tracer_instance, "_include_tool_definitions"):
                tracer_instance._include_tool_definitions = (
                    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN in opt_in_values
                )

    def activate(self) -> None:
        if self._is_instrumented:
            return

        try:
            tracer_module = importlib.import_module("strands.telemetry.tracer")
        except ImportError as exc:
            logger.warning(
                "Failed to activate Strands Agents instrumentation - "
                "missing dependency: %s",
                exc,
            )
            return

        self._enable_semconv_opt_ins()
        if self._processor is None:
            self._processor = StrandsAgentsSpanProcessor()

        tracer_provider = trace.get_tracer_provider()
        self._register_processor(
            tracer_provider=tracer_provider,
            processor=self._processor,
        )
        self._refresh_strands_tracer(
            tracer_module=tracer_module,
            tracer_provider=tracer_provider,
        )
        self._is_instrumented = True
        logger.info("Strands Agents instrumentation activated")

    def deactivate(self) -> None:
        if not self._is_instrumented:
            return

        if self._processor is not None:
            self._unregister_processor(
                tracer_provider=trace.get_tracer_provider(),
                processor=self._processor,
            )
        self._restore_semconv_opt_ins()
        self._is_instrumented = False
        logger.info("Strands Agents instrumentation deactivated")
