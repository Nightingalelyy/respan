import logging

from opentelemetry import trace

from respan_instrumentation_openinference._translator import OpenInferenceTranslator

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

    Automatically registers an ``OpenInferenceTranslator`` SpanProcessor that
    converts OI span attributes to OpenLLMetry/Traceloop format before export.
    """

    # Class-level flag: only register the translator once across all instances
    _translator_registered = False

    def __init__(self, instrumentor_class: type, **kwargs) -> None:
        self._instrumentor_class = instrumentor_class
        # tracer_provider is always set by activate(); drop to avoid collision
        kwargs.pop("tracer_provider", None)
        self._instrumentor_kwargs = kwargs
        self._instrumentor = None
        self._is_instrumented = False
        self._is_span_processor = False
        self.name = f"openinference-{instrumentor_class.__name__}"

    def activate(self) -> None:
        self._instrumentor = self._instrumentor_class()
        tp = trace.get_tracer_provider()

        # Register the OI → OpenLLMetry translator once so it runs
        # BEFORE the export processors on every OI span.  The translator
        # enriches spans with traceloop.* attributes that is_processable_span()
        # needs to see.  Insert at position 0 of the processor list so it
        # runs before BufferingSpanProcessor → FilteringSpanProcessor → exporter.
        if not OpenInferenceInstrumentor._translator_registered:
            translator = OpenInferenceTranslator()
            asp = getattr(tp, "_active_span_processor", None)
            processors = getattr(asp, "_span_processors", None)
            if processors is not None:
                # _span_processors may be a tuple; rebuild with translator first
                asp._span_processors = (translator, *processors)
            elif hasattr(tp, "add_span_processor"):
                tp.add_span_processor(translator)
            OpenInferenceInstrumentor._translator_registered = True
            logger.info("Registered OpenInference → OpenLLMetry translator")

        # Standard OI instrumentors expose .instrument()
        if hasattr(self._instrumentor, "instrument"):
            self._instrumentor.instrument(tracer_provider=tp, **self._instrumentor_kwargs)
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
