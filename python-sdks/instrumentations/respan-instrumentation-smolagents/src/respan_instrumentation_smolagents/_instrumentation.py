"""smolagents instrumentation plugin for Respan."""

import importlib
import logging
import threading
from typing import Any

from opentelemetry import trace
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_instrumentation_openinference._translator import OpenInferenceTranslator
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_smolagents._constants import (
    OPENINFERENCE_SMOLAGENTS_MODULE,
    SMOLAGENTS_INSTRUMENTATION_NAME,
)
from respan_instrumentation_smolagents._processor import (
    SmolagentsSpanContentProcessor,
    SmolagentsSpanContractProcessor,
)

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_COUNT = 0
_RUNTIME_CONFIG: dict[str, Any] | None = None
_RUNTIME_DELEGATE: Any = None
_RUNTIME_CONTENT_PROCESSOR: SmolagentsSpanContentProcessor | None = None
_RUNTIME_CONTRACT_PROCESSOR: SmolagentsSpanContractProcessor | None = None


def _load_openinference_smolagents_class() -> type:
    smolagents_module = importlib.import_module(OPENINFERENCE_SMOLAGENTS_MODULE)
    return smolagents_module.SmolagentsInstrumentor


def _get_active_span_processors(tracer_provider) -> tuple[Any, ...] | None:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if processors is None:
        return None
    return tuple(processors)


def _set_active_span_processors(tracer_provider, processors: tuple[Any, ...]) -> bool:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    if active_span_processor is None:
        return False
    if getattr(active_span_processor, "_span_processors", None) is None:
        return False
    active_span_processor._span_processors = processors
    return True


def _register_processor_after_translator(tracer_provider, processor) -> None:
    processors = _get_active_span_processors(tracer_provider)
    if processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )

    for index, existing_processor in enumerate(processors):
        if isinstance(existing_processor, OpenInferenceTranslator):
            _set_active_span_processors(
                tracer_provider=tracer_provider,
                processors=(
                    *processors[: index + 1],
                    processor,
                    *processors[index + 1 :],
                ),
            )
            return

    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=(*processors, processor),
    )


def _register_processor_before_translator(tracer_provider, processor) -> None:
    processors = _get_active_span_processors(tracer_provider)
    if processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )

    for index, existing_processor in enumerate(processors):
        if isinstance(existing_processor, OpenInferenceTranslator):
            _set_active_span_processors(
                tracer_provider=tracer_provider,
                processors=(
                    *processors[:index],
                    processor,
                    *processors[index:],
                ),
            )
            return

    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=(processor, *processors),
    )


def _unregister_processor(tracer_provider, processor) -> None:
    processors = _get_active_span_processors(tracer_provider)
    if processors is None:
        return
    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        ),
    )


class SmolagentsInstrumentor:
    """Respan instrumentor for smolagents.

    Activates the OpenInference smolagents instrumentor and registers Respan's
    OpenInference translator so smolagents spans reach the Respan OTLP pipeline
    with the canonical tracing fields.
    """

    name = SMOLAGENTS_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._content_processor = SmolagentsSpanContentProcessor()
        self._contract_processor = SmolagentsSpanContractProcessor()
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument smolagents through OpenInference and Respan's translator."""
        global _RUNTIME_CONFIG, _RUNTIME_CONTRACT_PROCESSOR
        global _RUNTIME_CONTENT_PROCESSOR, _RUNTIME_COUNT, _RUNTIME_DELEGATE

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "smolagents instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            smolagents_instrumentor_class = _load_openinference_smolagents_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate smolagents instrumentation - missing dependency: %s",
                exc,
            )
            return

        with _RUNTIME_LOCK:
            if self._is_instrumented:
                return
            if _RUNTIME_COUNT:
                if _RUNTIME_CONFIG != self._instrumentor_kwargs:
                    raise ValueError(
                        "smolagents instrumentation is already active with different options"
                    )
                _RUNTIME_COUNT += 1
                self._delegate = _RUNTIME_DELEGATE
                self._content_processor = _RUNTIME_CONTENT_PROCESSOR
                self._contract_processor = _RUNTIME_CONTRACT_PROCESSOR
                self._is_instrumented = True
                return

            tracer_provider = trace.get_tracer_provider()
            content_registered = False
            contract_registered = False
            try:
                self._delegate = OpenInferenceInstrumentor(
                    smolagents_instrumentor_class,
                    **self._instrumentor_kwargs,
                )
                self._delegate.activate()
                _register_processor_before_translator(
                    tracer_provider=tracer_provider,
                    processor=self._content_processor,
                )
                content_registered = True
                _register_processor_after_translator(
                    tracer_provider=tracer_provider,
                    processor=self._contract_processor,
                )
                contract_registered = True
                self._is_instrumented = True
                _RUNTIME_CONFIG = dict(self._instrumentor_kwargs)
                _RUNTIME_COUNT = 1
                _RUNTIME_DELEGATE = self._delegate
                _RUNTIME_CONTENT_PROCESSOR = self._content_processor
                _RUNTIME_CONTRACT_PROCESSOR = self._contract_processor
                logger.info("smolagents instrumentation activated")
            except Exception:
                if contract_registered:
                    _unregister_processor(tracer_provider, self._contract_processor)
                if content_registered:
                    _unregister_processor(tracer_provider, self._content_processor)
                if self._delegate is not None:
                    try:
                        self._delegate.deactivate()
                    except Exception:
                        logger.exception(
                            "Failed to clean up smolagents instrumentation"
                        )
                self._delegate = None
                self._is_instrumented = False
                logger.exception("Failed to activate smolagents instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        global _RUNTIME_CONFIG, _RUNTIME_CONTRACT_PROCESSOR
        global _RUNTIME_CONTENT_PROCESSOR, _RUNTIME_COUNT, _RUNTIME_DELEGATE

        with _RUNTIME_LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _RUNTIME_COUNT = max(_RUNTIME_COUNT - 1, 0)
            if _RUNTIME_COUNT:
                return

            tracer_provider = trace.get_tracer_provider()
            if _RUNTIME_CONTRACT_PROCESSOR is not None:
                _unregister_processor(tracer_provider, _RUNTIME_CONTRACT_PROCESSOR)
            if _RUNTIME_CONTENT_PROCESSOR is not None:
                _unregister_processor(tracer_provider, _RUNTIME_CONTENT_PROCESSOR)
            if _RUNTIME_DELEGATE is not None:
                try:
                    _RUNTIME_DELEGATE.deactivate()
                except Exception:
                    logger.exception("Failed to deactivate smolagents instrumentation")
            _RUNTIME_DELEGATE = None
            _RUNTIME_CONTENT_PROCESSOR = None
            _RUNTIME_CONTRACT_PROCESSOR = None
            _RUNTIME_CONFIG = None
            self._delegate = None
            logger.info("smolagents instrumentation deactivated")
