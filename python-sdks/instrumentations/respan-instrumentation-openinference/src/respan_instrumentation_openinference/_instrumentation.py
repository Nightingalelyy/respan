import logging

from opentelemetry import trace

logger = logging.getLogger(__name__)


class OpenInferenceInstrumentor:
    """Generic wrapper that adapts any OpenInference instrumentor to the
    Respan ``Instrumentation`` protocol.

    Usage::

        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        respan = Respan(instrumentations=[
            OpenInferenceInstrumentor(GoogleADKInstrumentor)
        ])

    Handles both standard OI instrumentors (``.instrument()``) and
    SpanProcessor-based ones (e.g. pydantic-ai, strands-agents) automatically.
    """

    def __init__(self, instrumentor_class: type) -> None:
        self._instrumentor_class = instrumentor_class
        self._instrumentor = None
        self._is_instrumented = False
        self._is_span_processor = False
        self.name = f"openinference-{instrumentor_class.__name__}"

    def activate(self) -> None:
        self._instrumentor = self._instrumentor_class()
        tp = trace.get_tracer_provider()

        # Standard OI instrumentors expose .instrument()
        if hasattr(self._instrumentor, "instrument"):
            self._instrumentor.instrument(tracer_provider=tp)
        # SpanProcessor-based OI packages (pydantic-ai, strands-agents)
        elif hasattr(tp, "add_span_processor"):
            tp.add_span_processor(self._instrumentor)
            self._is_span_processor = True

        self._is_instrumented = True
        logger.info("%s instrumentation activated (via OpenInference)", self.name)

    def deactivate(self) -> None:
        if self._is_instrumented and self._instrumentor is not None:
            try:
                if self._is_span_processor:
                    self._instrumentor.shutdown()
                else:
                    self._instrumentor.uninstrument()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("%s instrumentation deactivated", self.name)
