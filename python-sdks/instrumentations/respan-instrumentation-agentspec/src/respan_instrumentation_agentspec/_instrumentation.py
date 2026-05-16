"""AgentSpec instrumentation plugin for Respan."""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.sdk.trace.export import SpanProcessor
from respan_instrumentation_openinference import OpenInferenceTranslator
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

AGENTSPEC_INSTRUMENTATION_NAME = "agentspec"
_ORIGINAL_ON_LLM_END_ATTR = "_respan_original_on_llm_end"
_USAGE_PATCHED_ATTR = "_respan_usage_patched"
TRACELOOP_WORKFLOW_NAME = TLSpanAttributes.TRACELOOP_WORKFLOW_NAME


def _coerce_token_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        token_count = int(value)
    except (TypeError, ValueError):
        return None
    return token_count if token_count >= 0 else None


def _first_token_count(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in mapping:
            continue
        token_count = _coerce_token_count(mapping[key])
        if token_count is not None:
            return token_count
    return None


def _usage_from_mapping(mapping: Any) -> tuple[int | None, int | None]:
    if not isinstance(mapping, dict):
        return None, None

    input_tokens = _first_token_count(
        mapping,
        "input_tokens",
        "prompt_tokens",
        "prompt",
    )
    output_tokens = _first_token_count(
        mapping,
        "output_tokens",
        "completion_tokens",
        "completion",
    )
    return input_tokens, output_tokens


def _extract_langchain_usage(response: Any) -> tuple[int | None, int | None]:
    """Extract LangChain token usage from response/message metadata."""
    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        message = getattr(generations[0][0], "message", None)
        if message is not None:
            for mapping in (
                getattr(message, "usage_metadata", None),
                getattr(message, "response_metadata", None),
            ):
                input_tokens, output_tokens = _usage_from_mapping(mapping)
                if input_tokens is not None or output_tokens is not None:
                    return input_tokens, output_tokens

            response_metadata = getattr(message, "response_metadata", None) or {}
            for key in ("token_usage", "usage"):
                input_tokens, output_tokens = _usage_from_mapping(
                    response_metadata.get(key)
                )
                if input_tokens is not None or output_tokens is not None:
                    return input_tokens, output_tokens

    llm_output = getattr(response, "llm_output", None) or {}
    for mapping in (llm_output, llm_output.get("token_usage"), llm_output.get("usage")):
        input_tokens, output_tokens = _usage_from_mapping(mapping)
        if input_tokens is not None or output_tokens is not None:
            return input_tokens, output_tokens

    return None, None


def _patch_agentspec_langgraph_usage() -> None:
    try:
        import pyagentspec.adapters.langgraph.tracing as agentspec_tracing
    except ImportError:
        return

    handler_class = getattr(agentspec_tracing, "AgentSpecCallbackHandler", None)
    if handler_class is None or getattr(handler_class, _USAGE_PATCHED_ATTR, False):
        return

    original_on_llm_end = handler_class.on_llm_end

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        run_id_str = str(run_id)
        span = self.agentspec_spans_registry.get(run_id_str)
        if not isinstance(span, agentspec_tracing.AgentSpecLlmGenerationSpan):
            raise RuntimeError("LLM span not started; on_chat_model_start must run first")

        message_id, content, tool_calls = (
            agentspec_tracing._extract_message_content_and_tool_calls(response)
        )
        input_tokens, output_tokens = _extract_langchain_usage(response)
        event = agentspec_tracing.AgentSpecLlmGenerationResponse(
            llm_config=self.llm_config,
            request_id=run_id_str,
            completion_id=message_id,
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._add_event(run_id_str, span, event)
        self._end_span(run_id_str, span)
        self.agentspec_spans_registry.pop(run_id_str, None)
        self.messages_in_process.pop(run_id_str, None)

    setattr(handler_class, _ORIGINAL_ON_LLM_END_ATTR, original_on_llm_end)
    handler_class.on_llm_end = on_llm_end
    setattr(handler_class, _USAGE_PATCHED_ATTR, True)


def _restore_agentspec_langgraph_usage_patch() -> None:
    try:
        import pyagentspec.adapters.langgraph.tracing as agentspec_tracing
    except ImportError:
        return

    handler_class = getattr(agentspec_tracing, "AgentSpecCallbackHandler", None)
    if handler_class is None or not getattr(handler_class, _USAGE_PATCHED_ATTR, False):
        return

    original_on_llm_end = getattr(handler_class, _ORIGINAL_ON_LLM_END_ATTR, None)
    if original_on_llm_end is not None:
        handler_class.on_llm_end = original_on_llm_end
    delattr(handler_class, _ORIGINAL_ON_LLM_END_ATTR)
    setattr(handler_class, _USAGE_PATCHED_ATTR, False)


class _TranslatedProcessorChain(SpanProcessor):
    """Run Respan's OI translator before the active export processors."""

    def __init__(
        self,
        *,
        translator: OpenInferenceTranslator,
        processors: tuple[SpanProcessor, ...],
        workflow_name: str | None = None,
    ) -> None:
        self._translator = translator
        self._processors = processors
        self._workflow_name = workflow_name

    def _set_workflow_name(self, span: Any) -> None:
        if not self._workflow_name:
            return

        attributes = getattr(span, "_attributes", None)
        if attributes is not None:
            if attributes.get(TRACELOOP_WORKFLOW_NAME) is None:
                attributes[TRACELOOP_WORKFLOW_NAME] = self._workflow_name
            return

        set_attribute = getattr(span, "set_attribute", None)
        if callable(set_attribute):
            set_attribute(TRACELOOP_WORKFLOW_NAME, self._workflow_name)

    def on_start(self, span, parent_context=None) -> None:
        self._set_workflow_name(span)
        for processor in self._processors:
            processor.on_start(span=span, parent_context=parent_context)

    def on_end(self, span) -> None:
        self._set_workflow_name(span)
        self._translator.on_end(span)
        for processor in self._processors:
            processor.on_end(span=span)

    def shutdown(self) -> None:
        # This chain borrows processors owned by the active tracer provider.
        # AgentSpec deactivation must not shut down Respan's global pipeline.
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        results = []
        for processor in self._processors:
            force_flush = getattr(processor, "force_flush", None)
            if force_flush is not None:
                results.append(force_flush(timeout_millis=timeout_millis))
        return all(results) if results else True


def _active_span_processors(tracer_provider) -> tuple[SpanProcessor, ...]:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return tuple(processors or ())


class AgentSpecInstrumentor:
    """Respan instrumentor for AgentSpec.

    Activates AgentSpec tracing with the upstream OpenInference AgentSpec span
    processor, then runs Respan's OpenInference translator before the active
    Respan export processors. AgentSpec's upstream instrumentor fans out to
    each existing OTel processor individually, so this package builds one local
    processor chain to keep translation and export on the same span instance.
    """

    name = AGENTSPEC_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        mask_sensitive_information: bool = False,
        workflow_name: str | None = None,
    ) -> None:
        self._mask_sensitive_information = mask_sensitive_information
        self._workflow_name = workflow_name
        self._trace = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    @staticmethod
    def _workflow_name_from_resource(resource: Any) -> str | None:
        attributes = getattr(resource, "attributes", None)
        if not attributes:
            return None
        workflow_name = attributes.get(ResourceAttributes.SERVICE_NAME)
        return workflow_name if isinstance(workflow_name, str) and workflow_name else None

    def activate(self) -> None:
        """Instrument AgentSpec via OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "AgentSpec instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            from openinference.instrumentation.agentspec import OpenInferenceSpanProcessor
            from pyagentspec.tracing.trace import Trace, get_trace
            from pyagentspec.tracing.spans import RootSpan
        except ImportError as exc:
            logger.warning(
                "Failed to activate AgentSpec instrumentation - missing dependency: %s",
                exc,
            )
            return

        if get_trace() is not None:
            logger.warning(
                "AgentSpec instrumentation skipped because an AgentSpec Trace is already active"
            )
            return

        try:
            _patch_agentspec_langgraph_usage()
            tracer_provider = trace.get_tracer_provider()
            resource = getattr(tracer_provider, "resource", None)
            workflow_name = (
                self._workflow_name or self._workflow_name_from_resource(resource)
            )
            translator = OpenInferenceTranslator()
            processors = tuple(
                processor
                for processor in _active_span_processors(tracer_provider)
                if not isinstance(processor, OpenInferenceTranslator)
            )
            processor_chain = _TranslatedProcessorChain(
                translator=translator,
                processors=processors,
                workflow_name=workflow_name,
            )
            agentspec_processor = OpenInferenceSpanProcessor(
                otel_span_processor=processor_chain,
                resource=resource,
                mask_sensitive_information=self._mask_sensitive_information,
            )
            self._trace = Trace(
                name=workflow_name,
                root_span=RootSpan(name=workflow_name) if workflow_name else None,
                span_processors=[agentspec_processor],
                shutdown_on_exit=False,
            )
            self._trace._start()
            self._is_instrumented = True
            logger.info("AgentSpec instrumentation activated")
        except Exception:
            self._trace = None
            self._is_instrumented = False
            logger.exception("Failed to activate AgentSpec instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._is_instrumented and self._trace is not None:
            try:
                self._trace._end()
            except Exception:
                logger.exception("Failed to deactivate AgentSpec instrumentation")
        _restore_agentspec_langgraph_usage_patch()
        self._trace = None
        self._is_instrumented = False
        logger.info("AgentSpec instrumentation deactivated")
