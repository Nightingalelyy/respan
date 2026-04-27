"""Haystack instrumentation plugin for Respan.

Thin wrapper around ``opentelemetry-instrumentation-haystack`` (Traceloop/OpenLLMetry).
Spans carry GenAI semantic conventions and pass ``is_processable_span()`` natively.
"""

import logging

logger = logging.getLogger(__name__)


class HaystackInstrumentor:
    """Respan instrumentor for Haystack.

    Activates OTEL auto-instrumentation for the ``haystack-ai`` package so
    that pipeline runs, component executions, and LLM calls are traced automatically.

    Usage::

        from respan import Respan
        from respan_instrumentation_haystack import HaystackInstrumentor

        respan = Respan(instrumentations=[HaystackInstrumentor()])
    """

    name = "haystack"

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        """Instrument Haystack via the Traceloop OTEL instrumentor."""
        try:
            from opentelemetry.instrumentation.haystack import HaystackInstrumentor as OTELHaystack

            instrumentor = OTELHaystack()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            self._is_instrumented = True
            logger.info("Haystack instrumentation activated")
        except ImportError as exc:
            logger.warning(
                "Failed to activate Haystack instrumentation — missing dependency: %s", exc
            )

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented:
            try:
                from opentelemetry.instrumentation.haystack import HaystackInstrumentor as OTELHaystack
                OTELHaystack().uninstrument()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("Haystack instrumentation deactivated")
