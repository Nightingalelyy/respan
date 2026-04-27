"""CrewAI instrumentation plugin for Respan.

Thin wrapper around ``opentelemetry-instrumentation-crewai`` (Traceloop/OpenLLMetry).
Spans carry GenAI semantic conventions and pass ``is_processable_span()`` natively.
"""

import logging

logger = logging.getLogger(__name__)


class CrewAIInstrumentor:
    """Respan instrumentor for CrewAI.

    Activates OTEL auto-instrumentation for the ``crewai`` package so
    that agent runs, task executions, and tool calls are traced automatically.

    Usage::

        from respan import Respan
        from respan_instrumentation_crewai import CrewAIInstrumentor

        respan = Respan(instrumentations=[CrewAIInstrumentor()])
    """

    name = "crewai"

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        """Instrument CrewAI via the Traceloop OTEL instrumentor."""
        try:
            from opentelemetry.instrumentation.crewai import CrewAIInstrumentor as OTELCrewAI

            instrumentor = OTELCrewAI()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            self._is_instrumented = True
            logger.info("CrewAI instrumentation activated")
        except ImportError as exc:
            logger.warning(
                "Failed to activate CrewAI instrumentation — missing dependency: %s", exc
            )

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented:
            try:
                from opentelemetry.instrumentation.crewai import CrewAIInstrumentor as OTELCrewAI
                OTELCrewAI().uninstrument()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("CrewAI instrumentation deactivated")
