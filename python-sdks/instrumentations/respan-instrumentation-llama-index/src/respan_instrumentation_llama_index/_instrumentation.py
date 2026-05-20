"""Respan plugin for native LlamaIndex instrumentation."""

from __future__ import annotations

import importlib
import logging

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_llama_index._constants import (
    LLAMA_INDEX_INSTRUMENTATION_NAME,
    LLAMA_INDEX_ROOT_MODULE,
)
from respan_instrumentation_llama_index._handlers import (
    RespanLlamaIndexEventHandler,
    RespanLlamaIndexSpanHandler,
)

logger = logging.getLogger(__name__)


class LlamaIndexInstrumentor:
    """Respan instrumentor for LlamaIndex.

    Registers native LlamaIndex span and event handlers with
    ``llama_index.core.instrumentation.root_dispatcher`` and sends
    Respan-compatible OTEL spans through the existing Respan tracing runtime.
    """

    name = LLAMA_INDEX_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._span_handler = RespanLlamaIndexSpanHandler(
            capture_content=capture_content
        )
        self._event_handler = RespanLlamaIndexEventHandler(
            capture_content=capture_content
        )
        self._root_dispatcher = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Register native LlamaIndex handlers."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "LlamaIndex instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            llama_index_instrumentation = importlib.import_module(
                LLAMA_INDEX_ROOT_MODULE
            )
        except ImportError as exc:
            logger.warning(
                "Failed to activate LlamaIndex instrumentation — missing dependency: %s",
                exc,
            )
            return

        self._root_dispatcher = llama_index_instrumentation.root_dispatcher
        self._register_handlers()
        self._is_instrumented = True
        logger.info("LlamaIndex instrumentation activated")

    def deactivate(self) -> None:
        """Remove native LlamaIndex handlers."""
        if self._root_dispatcher is not None:
            self._root_dispatcher.span_handlers = [
                handler
                for handler in self._root_dispatcher.span_handlers
                if handler is not self._span_handler
            ]
            self._root_dispatcher.event_handlers = [
                handler
                for handler in self._root_dispatcher.event_handlers
                if handler is not self._event_handler
            ]
        self._root_dispatcher = None
        self._is_instrumented = False
        logger.info("LlamaIndex instrumentation deactivated")

    def _register_handlers(self) -> None:
        if self._span_handler not in self._root_dispatcher.span_handlers:
            self._root_dispatcher.add_span_handler(self._span_handler)
        if self._event_handler not in self._root_dispatcher.event_handlers:
            self._root_dispatcher.add_event_handler(self._event_handler)
