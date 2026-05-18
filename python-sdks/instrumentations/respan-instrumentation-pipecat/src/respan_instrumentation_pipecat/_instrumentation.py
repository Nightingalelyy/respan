"""Pipecat instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace

from respan_instrumentation_pipecat._translator import PipecatOpenInferenceTranslator
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

PIPECAT_INSTRUMENTATION_NAME = "pipecat"
OPENINFERENCE_PIPECAT_MODULE = "openinference.instrumentation.pipecat"
OPENINFERENCE_PIPECAT_OBSERVER_MODULE = "openinference.instrumentation.pipecat._observer"
_ORIGINAL_OBSERVER_CONTEXT_ATTR = "_respan_original_context"


def _load_openinference_pipecat_class() -> type:
    pipecat_module = importlib.import_module(OPENINFERENCE_PIPECAT_MODULE)
    return pipecat_module.PipecatInstrumentor


def _load_openinference_pipecat_observer_module() -> Any:
    return importlib.import_module(OPENINFERENCE_PIPECAT_OBSERVER_MODULE)


class PipecatInstrumentor:
    """Respan instrumentor for Pipecat.

    Activates the upstream OpenInference Pipecat instrumentor and registers a
    Pipecat-local translator so pipeline, LLM, STT, TTS, and tool spans reach
    the Respan OTEL pipeline with canonical tracing attributes.
    """

    name = PIPECAT_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._instrumentor = None
        self._processor = None
        self._is_instrumented = False
        self._patched_observer_module = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    @staticmethod
    def _register_processor(
        tracer_provider,
        processor: PipecatOpenInferenceTranslator,
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
                processor,
                *active_span_processor._span_processors,
            )
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider,
        processor: PipecatOpenInferenceTranslator,
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

    def _patch_observer_context(self) -> None:
        observer_module = _load_openinference_pipecat_observer_module()
        if not hasattr(observer_module, _ORIGINAL_OBSERVER_CONTEXT_ATTR):
            setattr(
                observer_module,
                _ORIGINAL_OBSERVER_CONTEXT_ATTR,
                observer_module.Context,
            )
        observer_module.Context = context_api.get_current
        self._patched_observer_module = observer_module

    def _restore_observer_context(self) -> None:
        observer_module = self._patched_observer_module
        if observer_module is None:
            return
        original_context = getattr(observer_module, _ORIGINAL_OBSERVER_CONTEXT_ATTR, None)
        if original_context is not None:
            observer_module.Context = original_context
            delattr(observer_module, _ORIGINAL_OBSERVER_CONTEXT_ATTR)
        self._patched_observer_module = None

    def activate(self) -> None:
        """Instrument Pipecat via OpenInference and Pipecat's local translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Pipecat instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            pipecat_instrumentor_class = _load_openinference_pipecat_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Pipecat instrumentation — missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()
        if self._processor is None:
            self._processor = PipecatOpenInferenceTranslator()
        self._register_processor(tracer_provider, self._processor)

        try:
            self._patch_observer_context()
            self._instrumentor = pipecat_instrumentor_class()
            self._instrumentor.instrument(
                tracer_provider=tracer_provider,
                **self._instrumentor_kwargs,
            )
            self._is_instrumented = True
            logger.info("Pipecat instrumentation activated")
        except Exception:
            if self._instrumentor is not None:
                try:
                    self._instrumentor.uninstrument()
                except Exception:
                    logger.exception("Failed to clean up Pipecat instrumentation")
            self._unregister_processor(tracer_provider, self._processor)
            self._restore_observer_context()
            self._instrumentor = None
            self._is_instrumented = False
            logger.exception("Failed to activate Pipecat instrumentation")

    def deactivate(self) -> None:
        """Deactivate Pipecat instrumentation."""
        if self._is_instrumented and self._instrumentor is not None:
            try:
                self._instrumentor.uninstrument()
            except Exception:
                logger.exception("Failed to deactivate Pipecat instrumentation")
        if self._processor is not None:
            self._unregister_processor(trace.get_tracer_provider(), self._processor)
        self._restore_observer_context()
        self._instrumentor = None
        self._is_instrumented = False
        logger.info("Pipecat instrumentation deactivated")
