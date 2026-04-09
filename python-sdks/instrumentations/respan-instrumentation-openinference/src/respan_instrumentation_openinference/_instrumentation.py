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
    _translator = None
    _active_span_processors = []

    def __init__(self, instrumentor_class: type, **kwargs) -> None:
        self._instrumentor_class = instrumentor_class
        # tracer_provider is always set by activate(); drop to avoid collision
        kwargs.pop("tracer_provider", None)
        self._instrumentor_kwargs = kwargs
        self._instrumentor = None
        self._is_instrumented = False
        self._is_span_processor = False
        self.name = f"openinference-{instrumentor_class.__name__}"

    @classmethod
    def _get_translator(cls) -> OpenInferenceTranslator:
        if cls._translator is None:
            cls._translator = OpenInferenceTranslator()
        return cls._translator

    @staticmethod
    def _get_active_span_processor(tp):
        return getattr(tp, "_active_span_processor", None)

    @classmethod
    def _rebuild_processor_chain(cls, tp) -> None:
        asp = cls._get_active_span_processor(tp)
        translator = cls._get_translator()

        if asp is None:
            return

        processors = getattr(asp, "_span_processors", None)
        if processors is None:
            return

        active_oi_processors = tuple(cls._active_span_processors)
        other_processors = tuple(
            processor
            for processor in processors
            if processor is not translator and processor not in active_oi_processors
        )

        asp._span_processors = (
            *active_oi_processors,
            translator,
            *other_processors,
        )
        cls._translator_registered = True

    @classmethod
    def _remove_processor_instance(cls, tp, processor_to_remove) -> None:
        asp = cls._get_active_span_processor(tp)
        if asp is None:
            return
        processors = getattr(asp, "_span_processors", None)
        if processors is None:
            return
        asp._span_processors = tuple(
            processor
            for processor in processors
            if processor is not processor_to_remove
        )

    @classmethod
    def _ensure_translator_registered(cls, tp) -> None:
        translator = cls._get_translator()
        asp = cls._get_active_span_processor(tp)
        processors = getattr(asp, "_span_processors", None) if asp is not None else None

        if processors is not None:
            cls._rebuild_processor_chain(tp)
            logger.info("Registered OpenInference → OpenLLMetry translator")
            return

        if hasattr(tp, "add_span_processor"):
            tp.add_span_processor(translator)
            cls._translator_registered = True
            logger.info("Registered OpenInference → OpenLLMetry translator")

    def activate(self) -> None:
        self._instrumentor = self._instrumentor_class()
        tp = trace.get_tracer_provider()

        # Register the OI → OpenLLMetry translator once so it runs
        # BEFORE the export processors on every OI span.  The translator
        # enriches spans with traceloop.* attributes that is_processable_span()
        # needs to see.  Insert at position 0 of the processor list so it
        # runs before BufferingSpanProcessor → FilteringSpanProcessor → exporter.
        if not OpenInferenceInstrumentor._translator_registered:
            OpenInferenceInstrumentor._ensure_translator_registered(tp)

        # Standard OI instrumentors expose .instrument()
        if hasattr(self._instrumentor, "instrument"):
            self._instrumentor.instrument(tracer_provider=tp, **self._instrumentor_kwargs)
        # SpanProcessor-based OI packages (pydantic-ai, strands-agents)
        elif hasattr(tp, "add_span_processor"):
            OpenInferenceInstrumentor._active_span_processors = [
                *OpenInferenceInstrumentor._active_span_processors,
                self._instrumentor,
            ]
            asp = self._get_active_span_processor(tp)
            processors = getattr(asp, "_span_processors", None) if asp is not None else None
            if processors is not None:
                OpenInferenceInstrumentor._rebuild_processor_chain(tp)
            else:
                tp.add_span_processor(self._instrumentor)
                OpenInferenceInstrumentor._ensure_translator_registered(tp)
            self._is_span_processor = True

        self._is_instrumented = True
        logger.info("%s instrumentation activated (via OpenInference)", self.name)

    def deactivate(self) -> None:
        if self._is_instrumented and self._instrumentor is not None:
            try:
                if self._is_span_processor:
                    self._instrumentor.shutdown()
                    OpenInferenceInstrumentor._active_span_processors = [
                        processor
                        for processor in OpenInferenceInstrumentor._active_span_processors
                        if processor is not self._instrumentor
                    ]
                    OpenInferenceInstrumentor._remove_processor_instance(
                        trace.get_tracer_provider(),
                        self._instrumentor,
                    )
                    OpenInferenceInstrumentor._rebuild_processor_chain(
                        trace.get_tracer_provider()
                    )
                else:
                    self._instrumentor.uninstrument()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("%s instrumentation deactivated", self.name)
