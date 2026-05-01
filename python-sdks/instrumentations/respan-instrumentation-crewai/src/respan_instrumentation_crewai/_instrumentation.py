"""CrewAI instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

CREWAI_INSTRUMENTATION_NAME = "crewai"
OPENINFERENCE_CREWAI_MODULE = "openinference.instrumentation.crewai"
RESPAN_OPENINFERENCE_MODULE = "respan_instrumentation_openinference"
USE_EVENT_LISTENER_KWARG = "use_event_listener"
CREATE_LLM_SPANS_KWARG = "create_llm_spans"


def _load_openinference_crewai() -> tuple[type, type]:
    crewai_module = importlib.import_module(OPENINFERENCE_CREWAI_MODULE)
    openinference_module = importlib.import_module(RESPAN_OPENINFERENCE_MODULE)
    return (
        openinference_module.OpenInferenceInstrumentor,
        crewai_module.CrewAIInstrumentor,
    )


class CrewAIInstrumentor:
    """Respan instrumentor for CrewAI.

    Activates the OpenInference CrewAI instrumentor and registers Respan's
    OpenInference translator so CrewAI spans reach the Respan OTLP pipeline
    with the expected ``traceloop.*``, ``gen_ai.*``, and ``respan.*`` fields.

    """

    name = CREWAI_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        use_event_listener: bool = True,
        create_llm_spans: bool = True,
        **instrumentor_kwargs: Any,
    ) -> None:
        self._instrumentor_kwargs = {
            USE_EVENT_LISTENER_KWARG: use_event_listener,
            CREATE_LLM_SPANS_KWARG: create_llm_spans,
            **instrumentor_kwargs,
        }
        self._delegate = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument CrewAI via OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "CrewAI instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            openinference_instrumentor, crewai_instrumentor = (
                _load_openinference_crewai()
            )
        except ImportError as exc:
            logger.warning(
                "Failed to activate CrewAI instrumentation — missing dependency: %s",
                exc,
            )
            return

        try:
            self._delegate = openinference_instrumentor(
                crewai_instrumentor,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            self._is_instrumented = True
            logger.info("CrewAI instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up CrewAI instrumentation")
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate CrewAI instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate CrewAI instrumentation")
        self._delegate = None
        self._is_instrumented = False
        logger.info("CrewAI instrumentation deactivated")
