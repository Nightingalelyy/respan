"""AutoGen AgentChat instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from opentelemetry import trace

from respan_instrumentation_autogen._native_processor import (
    AutoGenNativeSpanProcessor,
)
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

AUTOGEN_INSTRUMENTATION_NAME = "autogen"
OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE = (
    "openinference.instrumentation.autogen_agentchat"
)
OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES = (
    "AutogenAgentChatInstrumentor",
    "AutoGenAgentChatInstrumentor",
    "AutoGenInstrumentor",
)
TRACER_PROVIDER_KWARG = "tracer_provider"


def _load_openinference_autogen_agentchat_class() -> type:
    autogen_module = importlib.import_module(OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE)
    for class_name in OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES:
        instrumentor_class = getattr(autogen_module, class_name, None)
        if instrumentor_class is not None:
            return instrumentor_class
    expected = ", ".join(OPENINFERENCE_AUTOGEN_AGENTCHAT_CLASS_NAMES)
    raise ImportError(
        f"{OPENINFERENCE_AUTOGEN_AGENTCHAT_MODULE} does not expose any of: {expected}"
    )


class AutoGenInstrumentor:
    """Respan instrumentor for AutoGen AgentChat.

    Activates OpenInference's AutoGen AgentChat instrumentor and registers
    Respan's OpenInference translator so AutoGen spans reach the Respan OTLP
    pipeline with the expected ``traceloop.*``, ``gen_ai.*``, and
    ``respan.*`` fields.
    """

    name = AUTOGEN_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        instrumentor_kwargs.pop(TRACER_PROVIDER_KWARG, None)
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._native_processor = AutoGenNativeSpanProcessor()
        self._is_instrumented = False

    @staticmethod
    def _register_native_processor(tracer_provider, processor) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if processors is not None:
            remaining_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            active_span_processor._span_processors = (
                processor,
                *remaining_processors,
            )
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_native_processor(tracer_provider, processor) -> None:
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

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument AutoGen AgentChat via OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "AutoGen instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            autogen_instrumentor_class = _load_openinference_autogen_agentchat_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate AutoGen instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            self._register_native_processor(
                trace.get_tracer_provider(),
                self._native_processor,
            )
            self._delegate = OpenInferenceInstrumentor(
                autogen_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            self._is_instrumented = True
            logger.info("AutoGen instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up AutoGen instrumentation")
            self._unregister_native_processor(
                trace.get_tracer_provider(),
                self._native_processor,
            )
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate AutoGen instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate AutoGen instrumentation")
        self._unregister_native_processor(
            trace.get_tracer_provider(),
            self._native_processor,
        )
        self._delegate = None
        self._is_instrumented = False
        logger.info("AutoGen instrumentation deactivated")
