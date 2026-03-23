"""Generic wrapper that adapts any OTEL-style instrumentor to the Respan protocol."""

import logging

from opentelemetry import trace

logger = logging.getLogger(__name__)


class OTELInstrumentor:
    """Wraps any OTEL instrumentor for use with ``Respan(instrumentations=[...])``.

    Bridges the OTEL instrumentor interface (``.instrument()`` / ``.uninstrument()``)
    to the Respan ``Instrumentation`` protocol (``.activate()`` / ``.deactivate()``).

    Works with any instrumentor that follows the OTEL pattern, including:
    - Traceloop/OpenLLMetry packages (``opentelemetry-instrumentation-*``)
    - Any custom OTEL instrumentor

    Usage::

        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        respan = Respan(instrumentations=[
            OTELInstrumentor(AnthropicInstrumentor)
        ])
    """

    def __init__(self, instrumentor_class: type) -> None:
        self._instrumentor_class = instrumentor_class
        self._instrumentor = None
        self._is_instrumented = False
        self.name = f"otel-{instrumentor_class.__name__}"

    def activate(self) -> None:
        self._instrumentor = self._instrumentor_class()
        tp = trace.get_tracer_provider()
        self._instrumentor.instrument(tracer_provider=tp)
        self._is_instrumented = True
        logger.info("%s instrumentation activated (via OTEL)", self.name)

    def deactivate(self) -> None:
        if self._is_instrumented and self._instrumentor is not None:
            try:
                self._instrumentor.uninstrument()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("%s instrumentation deactivated", self.name)
