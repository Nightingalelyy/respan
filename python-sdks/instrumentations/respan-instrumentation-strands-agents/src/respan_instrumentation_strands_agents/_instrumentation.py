"""Strands Agents OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import os
from threading import RLock
from typing import Any, ClassVar

from opentelemetry import trace

from respan_instrumentation_strands_agents._constants import (
    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN,
    STRANDS_TOOL_DEFINITIONS_ATTR,
)
from respan_instrumentation_strands_agents._processor import (
    StrandsAgentsSpanProcessor,
)
from respan_instrumentation_strands_agents._serialization import json_dumps

logger = logging.getLogger(__name__)


class StrandsAgentsInstrumentor:
    """Respan instrumentor for Strands Agents."""

    name = "strands-agents"
    _lock = RLock()
    _activation_count = 0
    _shared_processor: StrandsAgentsSpanProcessor | None = None
    _shared_provider: Any = None
    _shared_include_tool_definitions: bool | None = None
    _shared_previous_semconv_opt_in: str | None = None
    _shared_installed_semconv_opt_in: str | None = None
    _shared_tracer_module: Any = None
    _shared_tracer_instance: Any = None
    _shared_previous_tracer_provider: Any = None
    _shared_previous_tracer: Any = None
    _shared_previous_include_tool_definitions: bool | None = None
    _shared_installed_tracer_provider: Any = None
    _shared_installed_tracer: Any = None
    _shared_installed_include_tool_definitions: bool | None = None
    _tracer_class: type[Any] | None = None
    _original_tracer_methods: ClassVar[dict[str, Any]] = {}
    _installed_tracer_methods: ClassVar[dict[str, Any]] = {}
    _tool_definitions_by_trace: ClassVar[dict[int, str]] = {}

    def __init__(self, *, include_tool_definitions: bool = True) -> None:
        self._include_tool_definitions = include_tool_definitions
        self._is_instrumented = False

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: StrandsAgentsSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if active_span_processor is not None and processors is not None:
            remaining_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            active_span_processor._span_processors = (processor, *remaining_processors)
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: StrandsAgentsSpanProcessor,
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if active_span_processor is None or processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    @classmethod
    def _enable_semconv_opt_ins(cls) -> None:
        if not cls._shared_include_tool_definitions:
            return

        cls._shared_previous_semconv_opt_in = os.environ.get(
            "OTEL_SEMCONV_STABILITY_OPT_IN"
        )
        opt_in_values = {
            value.strip()
            for value in (cls._shared_previous_semconv_opt_in or "").split(",")
            if value.strip()
        }
        opt_in_values.add(STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN)
        installed = ",".join(sorted(opt_in_values))
        os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = installed
        cls._shared_installed_semconv_opt_in = installed

    @classmethod
    def _restore_semconv_opt_ins(cls) -> None:
        if not cls._shared_include_tool_definitions:
            return
        if (
            os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN")
            == cls._shared_installed_semconv_opt_in
        ):
            if cls._shared_previous_semconv_opt_in is None:
                os.environ.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
            else:
                os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = (
                    cls._shared_previous_semconv_opt_in
                )
        cls._shared_previous_semconv_opt_in = None
        cls._shared_installed_semconv_opt_in = None

    @staticmethod
    def _refresh_strands_tracer(tracer_module: Any, tracer_provider: Any) -> None:
        tracer_instance = getattr(tracer_module, "_tracer_instance", None)
        if tracer_instance is None:
            return

        tracer_instance.tracer_provider = tracer_provider
        service_name = getattr(tracer_instance, "service_name", tracer_module.__name__)
        tracer_instance.tracer = tracer_provider.get_tracer(service_name)
        parse_opt_in = getattr(tracer_instance, "_parse_semconv_opt_in", None)
        if callable(parse_opt_in):
            opt_in_values = parse_opt_in()
            if hasattr(tracer_instance, "_include_tool_definitions"):
                tracer_instance._include_tool_definitions = (
                    STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN in opt_in_values
                )

    @classmethod
    def _patch_tracer_tool_propagation(cls, tracer_module: Any) -> None:
        tracer_class = getattr(tracer_module, "Tracer", None)
        if tracer_class is None or cls._tracer_class is not None:
            return
        originals = {
            name: getattr(tracer_class, name)
            for name in (
                "start_agent_span",
                "start_model_invoke_span",
                "end_agent_span",
            )
        }

        def start_agent_span(instance: Any, *args: Any, **kwargs: Any) -> Any:
            span = originals["start_agent_span"](instance, *args, **kwargs)
            if cls._activation_count == 0 or span is None:
                return span
            tools_config = kwargs.get("tools_config")
            if tools_config is None and len(args) > 5:
                tools_config = args[5]
            construct = getattr(instance, "_construct_tool_definitions", None)
            try:
                definitions = (
                    construct(tools_config)
                    if tools_config and callable(construct)
                    else None
                )
                context = span.get_span_context()
                if definitions and context.is_valid:
                    cls._tool_definitions_by_trace[context.trace_id] = json_dumps(
                        definitions
                    )
            except Exception:
                logger.debug(
                    "Failed to capture Strands tool definitions", exc_info=True
                )
            return span

        def start_model_invoke_span(instance: Any, *args: Any, **kwargs: Any) -> Any:
            span = originals["start_model_invoke_span"](instance, *args, **kwargs)
            if cls._activation_count and span is not None:
                context = span.get_span_context()
                definitions = cls._tool_definitions_by_trace.get(context.trace_id)
                if definitions:
                    span.set_attribute(STRANDS_TOOL_DEFINITIONS_ATTR, definitions)
            return span

        def end_agent_span(instance: Any, span: Any, *args: Any, **kwargs: Any) -> Any:
            context = span.get_span_context() if span is not None else None
            try:
                return originals["end_agent_span"](instance, span, *args, **kwargs)
            finally:
                if context is not None and context.is_valid:
                    cls._tool_definitions_by_trace.pop(context.trace_id, None)

        installed = {
            "start_agent_span": start_agent_span,
            "start_model_invoke_span": start_model_invoke_span,
            "end_agent_span": end_agent_span,
        }
        for name, wrapper in installed.items():
            setattr(tracer_class, name, wrapper)
        cls._tracer_class = tracer_class
        cls._original_tracer_methods = originals
        cls._installed_tracer_methods = installed

    @classmethod
    def _restore_tracer_tool_propagation(cls) -> None:
        if cls._tracer_class is not None:
            for name, original in cls._original_tracer_methods.items():
                if getattr(
                    cls._tracer_class, name, None
                ) is cls._installed_tracer_methods.get(name):
                    setattr(cls._tracer_class, name, original)
        cls._tracer_class = None
        cls._original_tracer_methods.clear()
        cls._installed_tracer_methods.clear()
        cls._tool_definitions_by_trace.clear()

    @classmethod
    def _remember_refreshed_tracer(cls, tracer_module: Any) -> None:
        cls._shared_tracer_module = tracer_module
        tracer_instance = getattr(tracer_module, "_tracer_instance", None)
        if tracer_instance is None:
            return
        cls._shared_tracer_instance = tracer_instance
        cls._shared_previous_tracer_provider = getattr(
            tracer_instance, "tracer_provider", None
        )
        cls._shared_previous_tracer = getattr(tracer_instance, "tracer", None)
        cls._shared_previous_include_tool_definitions = getattr(
            tracer_instance, "_include_tool_definitions", None
        )

    @classmethod
    def _record_installed_tracer_state(cls) -> None:
        tracer_instance = cls._shared_tracer_instance
        if tracer_instance is None:
            return
        cls._shared_installed_tracer_provider = getattr(
            tracer_instance, "tracer_provider", None
        )
        cls._shared_installed_tracer = getattr(tracer_instance, "tracer", None)
        cls._shared_installed_include_tool_definitions = getattr(
            tracer_instance, "_include_tool_definitions", None
        )

    @classmethod
    def _restore_refreshed_tracer(cls) -> None:
        tracer_instance = cls._shared_tracer_instance
        if tracer_instance is None and cls._shared_tracer_module is not None:
            tracer_instance = getattr(
                cls._shared_tracer_module, "_tracer_instance", None
            )
        if tracer_instance is not None:
            if (
                getattr(tracer_instance, "tracer_provider", None)
                is cls._shared_installed_tracer_provider
            ):
                tracer_instance.tracer_provider = cls._shared_previous_tracer_provider
            if getattr(tracer_instance, "tracer", None) is cls._shared_installed_tracer:
                tracer_instance.tracer = cls._shared_previous_tracer
            if (
                getattr(tracer_instance, "_include_tool_definitions", None)
                == cls._shared_installed_include_tool_definitions
            ):
                tracer_instance._include_tool_definitions = (
                    cls._shared_previous_include_tool_definitions
                )
            elif cls._shared_tracer_instance is None:
                parse_opt_in = getattr(tracer_instance, "_parse_semconv_opt_in", None)
                if callable(parse_opt_in):
                    tracer_instance._include_tool_definitions = (
                        STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN in parse_opt_in()
                    )
        cls._shared_tracer_module = None
        cls._shared_tracer_instance = None
        cls._shared_previous_tracer_provider = None
        cls._shared_previous_tracer = None
        cls._shared_previous_include_tool_definitions = None
        cls._shared_installed_tracer_provider = None
        cls._shared_installed_tracer = None
        cls._shared_installed_include_tool_definitions = None

    def activate(self) -> None:
        cls = type(self)
        with cls._lock:
            if self._is_instrumented:
                return
            if cls._activation_count:
                if (
                    cls._shared_include_tool_definitions
                    != self._include_tool_definitions
                ):
                    raise ValueError(
                        "Strands Agents instrumentation is already active with a "
                        "different include_tool_definitions setting"
                    )
                cls._activation_count += 1
                self._is_instrumented = True
                return

            try:
                tracer_module = importlib.import_module("strands.telemetry.tracer")
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Strands Agents instrumentation - "
                    "missing dependency: %s",
                    exc,
                )
                return

            tracer_provider = trace.get_tracer_provider()
            processor = StrandsAgentsSpanProcessor()
            cls._shared_include_tool_definitions = self._include_tool_definitions
            cls._shared_processor = processor
            cls._shared_provider = tracer_provider
            try:
                cls._remember_refreshed_tracer(tracer_module)
                cls._enable_semconv_opt_ins()
                cls._activation_count = 1
                self._patch_tracer_tool_propagation(tracer_module)
                self._register_processor(tracer_provider, processor)
                self._refresh_strands_tracer(tracer_module, tracer_provider)
                cls._record_installed_tracer_state()
            except BaseException:
                self._unregister_processor(tracer_provider, processor)
                cls._restore_semconv_opt_ins()
                cls._activation_count = 0
                cls._restore_tracer_tool_propagation()
                cls._restore_refreshed_tracer()
                cls._shared_processor = None
                cls._shared_provider = None
                cls._shared_include_tool_definitions = None
                raise
            cls._activation_count = 1
            self._is_instrumented = True
            logger.info("Strands Agents instrumentation activated")

    def deactivate(self) -> None:
        cls = type(self)
        with cls._lock:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            cls._activation_count = max(0, cls._activation_count - 1)
            if cls._activation_count:
                return
            if cls._shared_processor is not None and cls._shared_provider is not None:
                self._unregister_processor(
                    cls._shared_provider,
                    cls._shared_processor,
                )
                cls._shared_processor.shutdown()
            cls._restore_semconv_opt_ins()
            cls._restore_tracer_tool_propagation()
            cls._restore_refreshed_tracer()
            cls._shared_processor = None
            cls._shared_provider = None
            cls._shared_include_tool_definitions = None
            logger.info("Strands Agents instrumentation deactivated")
