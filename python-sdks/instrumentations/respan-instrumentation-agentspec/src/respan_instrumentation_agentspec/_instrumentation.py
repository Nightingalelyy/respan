"""AgentSpec instrumentation plugin for Respan."""

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from opentelemetry import trace
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import SpanContext
from openinference.semconv.trace import SpanAttributes as OISpanAttributes
from respan_instrumentation_openinference import OpenInferenceTranslator
from respan_sdk.constants.span_attributes import (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_CUSTOM_ID,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

AGENTSPEC_INSTRUMENTATION_NAME = "agentspec"
_ORIGINAL_ON_LLM_END_ATTR = "_respan_original_on_llm_end"
_ORIGINAL_ON_LLM_END_ASYNC_ATTR = "_respan_original_on_llm_end_async"
_USAGE_PATCHED_ATTR = "_respan_usage_patched"
TRACELOOP_WORKFLOW_NAME = TLSpanAttributes.TRACELOOP_WORKFLOW_NAME
TRACELOOP_ENTITY_INPUT = TLSpanAttributes.TRACELOOP_ENTITY_INPUT
TRACELOOP_ENTITY_OUTPUT = TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT
OI_SESSION_ID = OISpanAttributes.SESSION_ID
PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."

_PROPAGATED_ATTRS = (
    RESPAN_CUSTOMER_PARAMS_ID,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
    RESPAN_SPAN_CUSTOM_ID,
)
_OFF_CONTRACT_ALIASES = {
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
}


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

    handler_class = getattr(
        agentspec_tracing,
        "AgentSpecLlmCallbackHandler",
        None,
    ) or getattr(agentspec_tracing, "AgentSpecCallbackHandler", None)
    if handler_class is None or getattr(handler_class, _USAGE_PATCHED_ATTR, False):
        return

    original_on_llm_end = getattr(handler_class, "on_llm_end", None)
    original_on_llm_end_async = getattr(handler_class, "on_llm_end_async", None)

    if original_on_llm_end is None and original_on_llm_end_async is None:
        return

    def build_response_event(self, response, run_id_str):
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
        return span, event

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        del parent_run_id, kwargs
        run_id_str = str(run_id)
        span, event = build_response_event(self, response, run_id_str)
        self._add_event(run_id_str, span, event)
        self._end_span(run_id_str, span)
        self.agentspec_spans_registry.pop(run_id_str, None)
        self.messages_in_process.pop(run_id_str, None)

    async def on_llm_end_async(
        self,
        response,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        del parent_run_id, kwargs
        run_id_str = str(run_id)
        span, event = build_response_event(self, response, run_id_str)
        await self._add_event_async(run_id_str, span, event)
        await self._end_span_async(run_id_str, span)
        self.agentspec_spans_registry.pop(run_id_str, None)
        self.messages_in_process.pop(run_id_str, None)

    if original_on_llm_end is not None:
        setattr(handler_class, _ORIGINAL_ON_LLM_END_ATTR, original_on_llm_end)
        handler_class.on_llm_end = on_llm_end
    if original_on_llm_end_async is not None:
        setattr(
            handler_class,
            _ORIGINAL_ON_LLM_END_ASYNC_ATTR,
            original_on_llm_end_async,
        )
        handler_class.on_llm_end_async = on_llm_end_async
    setattr(handler_class, _USAGE_PATCHED_ATTR, True)


def _restore_agentspec_langgraph_usage_patch() -> None:
    try:
        import pyagentspec.adapters.langgraph.tracing as agentspec_tracing
    except ImportError:
        return

    handler_class = getattr(
        agentspec_tracing,
        "AgentSpecLlmCallbackHandler",
        None,
    ) or getattr(agentspec_tracing, "AgentSpecCallbackHandler", None)
    if handler_class is None or not getattr(handler_class, _USAGE_PATCHED_ATTR, False):
        return

    original_on_llm_end = getattr(handler_class, _ORIGINAL_ON_LLM_END_ATTR, None)
    if original_on_llm_end is not None:
        handler_class.on_llm_end = original_on_llm_end
        delattr(handler_class, _ORIGINAL_ON_LLM_END_ATTR)
    original_on_llm_end_async = getattr(
        handler_class,
        _ORIGINAL_ON_LLM_END_ASYNC_ATTR,
        None,
    )
    if original_on_llm_end_async is not None:
        handler_class.on_llm_end_async = original_on_llm_end_async
        delattr(handler_class, _ORIGINAL_ON_LLM_END_ASYNC_ATTR)
    setattr(handler_class, _USAGE_PATCHED_ATTR, False)


def _trace_id_from_session_id(session_id: Any) -> int | None:
    if not session_id:
        return None
    try:
        trace_id = UUID(str(session_id)).int
    except (TypeError, ValueError, AttributeError):
        trace_id = int.from_bytes(
            hashlib.blake2b(str(session_id).encode(), digest_size=16).digest(),
            "big",
        )
    return trace_id or 1


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "{}", "[]", "null"}
    return True


def _parse_structured_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _indexed_messages(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    index = 0
    while True:
        role_key = f"{prefix}{index}.role"
        content_key = f"{prefix}{index}.content"
        tool_calls_key = f"{prefix}{index}.tool_calls"
        tool_call_id_key = f"{prefix}{index}.tool_call_id"
        if not any(
            key in attrs
            for key in (role_key, content_key, tool_calls_key, tool_call_id_key)
        ):
            break

        message: dict[str, Any] = {}
        if role_key in attrs:
            message["role"] = attrs[role_key]
        if content_key in attrs:
            message["content"] = _parse_structured_value(attrs[content_key])
        if tool_calls_key in attrs:
            message["tool_calls"] = _parse_structured_value(attrs[tool_calls_key])
        if tool_call_id_key in attrs:
            message["tool_call_id"] = attrs[tool_call_id_key]
        messages.append(message)
        index += 1
    return messages


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
        self._trace_state: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _replace_trace_id(context: Any, trace_id: int) -> SpanContext:
        return SpanContext(
            trace_id=trace_id,
            span_id=context.span_id,
            is_remote=context.is_remote,
            trace_flags=context.trace_flags,
            trace_state=context.trace_state,
        )

    def _normalize_trace_id(self, span: Any) -> int | None:
        attributes = getattr(span, "_attributes", None)
        get_span_context = getattr(span, "get_span_context", None)
        if attributes is None or not callable(get_span_context):
            return None

        trace_id = _trace_id_from_session_id(attributes.get(OI_SESSION_ID))
        if trace_id is None:
            return None

        context = get_span_context()
        if context.trace_id != trace_id:
            span._context = self._replace_trace_id(context, trace_id)

        parent = getattr(span, "parent", None)
        if parent is not None and parent.trace_id != trace_id:
            span._parent = self._replace_trace_id(parent, trace_id)
        return trace_id

    def _enrich_span_contract(self, span: Any, trace_id: int | None) -> bool:
        attributes = getattr(span, "_attributes", None)
        if attributes is None or trace_id is None:
            return False

        attrs = dict(attributes)
        state = self._trace_state.setdefault(trace_id, {})
        log_type = attrs.get(RESPAN_LOG_TYPE)
        is_root = getattr(span, "parent", None) is None

        input_value = attrs.get(TRACELOOP_ENTITY_INPUT)
        output_value = attrs.get(TRACELOOP_ENTITY_OUTPUT)
        if log_type == "chat":
            if not _has_content(input_value):
                prompt_messages = _indexed_messages(attrs, PROMPT_PREFIX)
                if prompt_messages:
                    input_value = json.dumps(prompt_messages, separators=(",", ":"))
                    attrs[TRACELOOP_ENTITY_INPUT] = input_value
            if not _has_content(output_value):
                completion_messages = _indexed_messages(attrs, COMPLETION_PREFIX)
                if completion_messages:
                    output_payload: Any = (
                        completion_messages[0]
                        if len(completion_messages) == 1
                        else completion_messages
                    )
                    output_value = json.dumps(output_payload, separators=(",", ":"))
                    attrs[TRACELOOP_ENTITY_OUTPUT] = output_value
            if _has_content(input_value) and "input" not in state:
                state["input"] = input_value
            if _has_content(output_value):
                state["output"] = output_value

        propagation = state.setdefault("propagation", {})
        for key in _PROPAGATED_ATTRS:
            if key in attrs:
                propagation[key] = attrs[key]
        for key, value in attrs.items():
            if key.startswith(f"{RESPAN_METADATA}."):
                propagation[key] = value

        if log_type == "agent" or is_root:
            if not _has_content(input_value) and "input" in state:
                attrs[TRACELOOP_ENTITY_INPUT] = state["input"]
            if not _has_content(output_value) and "output" in state:
                attrs[TRACELOOP_ENTITY_OUTPUT] = state["output"]
        if is_root:
            for key, value in propagation.items():
                attrs.setdefault(key, value)

        for key in _OFF_CONTRACT_ALIASES:
            attrs.pop(key, None)

        span._attributes = attrs
        return is_root

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
        self._normalize_trace_id(span)
        self._set_workflow_name(span)
        for processor in self._processors:
            processor.on_start(span=span, parent_context=parent_context)

    def on_end(self, span) -> None:
        trace_id = self._normalize_trace_id(span)
        self._set_workflow_name(span)
        self._translator.on_end(span)
        is_root = self._enrich_span_contract(span, trace_id)
        for processor in self._processors:
            processor.on_end(span=span)
        if is_root and trace_id is not None:
            self._trace_state.pop(trace_id, None)

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
