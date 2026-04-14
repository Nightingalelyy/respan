"""Claude Agent SDK OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace

from respan_instrumentation_claude_agent_sdk._processor import (  # type: ignore[reportMissingImports]
    ClaudeAgentSDKSpanProcessor,
    _safe_json_loads,
)

logger = logging.getLogger(__name__)

def _get_span_attr_value(span: Any, key: str) -> Any:
    attributes = getattr(span, "attributes", None)
    if attributes is None:
        attributes = getattr(span, "_attributes", None)
    if not isinstance(attributes, Mapping):
        return None
    return attributes.get(key)


def _load_upstream_instrumentor_class() -> type[Any]:
    upstream_module = importlib.import_module(
        "opentelemetry.instrumentation.claude_agent_sdk"
    )
    instrumentor_class = getattr(upstream_module, "ClaudeAgentSdkInstrumentor", None)
    if instrumentor_class is None:
        raise AttributeError(
            "opentelemetry.instrumentation.claude_agent_sdk.ClaudeAgentSdkInstrumentor"
        )
    return instrumentor_class


class ClaudeAgentSDKInstrumentor:
    """Respan instrumentor for the Claude Agent SDK."""

    name = "claude-agent-sdk"

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        capture_content: bool = False,
    ) -> None:
        self._agent_name = agent_name
        self._capture_content = capture_content
        self._otel_instrumentor = None
        self._processor = None
        self._is_instrumented = False
        self._patched_modules: list[tuple[Any, str, Any]] = []

    @staticmethod
    def _register_processor(
        tracer_provider: Any,
        processor: ClaudeAgentSDKSpanProcessor,
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
            # Normalize Claude spans before exporters or other processors read them.
            active_span_processor._span_processors = (processor, *remaining_processors)
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider: Any,
        processor: ClaudeAgentSDKSpanProcessor,
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

    def _patch_upstream_helpers(self) -> bool:
        try:
            query_module = importlib.import_module("claude_agent_sdk._internal.query")
            Query = getattr(query_module, "Query")
            constants_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._constants"
            )
            context_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._context"
            )
            instrumentor_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._instrumentor"
            )
            spans_module = importlib.import_module(
                "opentelemetry.instrumentation.claude_agent_sdk._spans"
            )
            instrumentor_class = _load_upstream_instrumentor_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to patch Claude Agent SDK helpers — missing dependency: %s",
                exc,
            )
            return False

        if self._patched_modules:
            return True

        output_messages_attr = constants_module.GEN_AI_OUTPUT_MESSAGES
        usage_input_tokens_attr = constants_module.GEN_AI_USAGE_INPUT_TOKENS
        usage_cache_creation_tokens_attr = (
            constants_module.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        )
        usage_cache_read_tokens_attr = (
            constants_module.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
        )
        serialize_value = getattr(spans_module, "_to_serializable", lambda value: value)

        original_set_response_content = spans_module.set_response_content
        original_set_result_attributes = spans_module.set_result_attributes
        original_wrap_client_query = instrumentor_class._wrap_client_query
        original_instrumented_receive_response = (
            instrumentor_class._instrumented_receive_response
        )
        original_handle_control_request = Query._handle_control_request

        def patched_set_response_content(span: Any, content: Any) -> None:
            if content is None:
                return

            existing_messages = _safe_json_loads(
                _get_span_attr_value(span=span, key=output_messages_attr)
            )
            if not isinstance(existing_messages, list):
                existing_messages = []

            appended_messages = [
                *existing_messages,
                {
                    "role": "assistant",
                    "content": serialize_value(content),
                },
            ]
            try:
                span.set_attribute(
                    output_messages_attr,
                    json.dumps(serialize_value(appended_messages), default=str),
                )
            except (TypeError, ValueError):
                span.set_attribute(output_messages_attr, str(appended_messages))

        def patched_set_result_attributes(span: Any, result_message: Any) -> None:
            original_set_result_attributes(span, result_message)

            usage = getattr(result_message, "usage", None)
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0) or 0
                cache_creation_tokens = usage.get("cache_creation_input_tokens", 0) or 0
                cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0

                span.set_attribute(usage_input_tokens_attr, int(input_tokens))
                if cache_creation_tokens > 0:
                    span.set_attribute(
                        usage_cache_creation_tokens_attr,
                        int(cache_creation_tokens),
                    )
                if cache_read_tokens > 0:
                    span.set_attribute(
                        usage_cache_read_tokens_attr,
                        int(cache_read_tokens),
                    )

            total_cost = getattr(result_message, "total_cost_usd", None)
            if isinstance(total_cost, int | float):
                span.set_attribute("cost", float(total_cost))

        def patched_wrap_client_query(
            instrumentor: Any,
            wrapped: Any,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            result = original_wrap_client_query(
                instrumentor,
                wrapped,
                instance,
                args,
                kwargs,
            )
            invocation_ctx = getattr(instance, "_otel_invocation_ctx", None)
            query = getattr(instance, "_query", None)
            if invocation_ctx is not None and query is not None:
                query._otel_invocation_ctx = invocation_ctx
            return result

        async def patched_instrumented_receive_response(
            instrumentor: Any,
            wrapped: Any,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            previous_invocation_ctx = context_module.get_invocation_context()
            invocation_ctx = getattr(instance, "_otel_invocation_ctx", None)
            query = getattr(instance, "_query", None)
            if invocation_ctx is not None:
                context_module.set_invocation_context(invocation_ctx)
                if query is not None:
                    query._otel_invocation_ctx = invocation_ctx

            try:
                async for message in original_instrumented_receive_response(
                    instrumentor,
                    wrapped,
                    instance,
                    args,
                    kwargs,
                ):
                    yield message
            finally:
                if query is not None:
                    query._otel_invocation_ctx = None
                context_module.set_invocation_context(previous_invocation_ctx)

        async def patched_handle_control_request(query: Any, request: Any) -> Any:
            previous_invocation_ctx = context_module.get_invocation_context()
            query_invocation_ctx = getattr(query, "_otel_invocation_ctx", None)
            if query_invocation_ctx is not None:
                context_module.set_invocation_context(query_invocation_ctx)
            try:
                return await original_handle_control_request(query, request)
            finally:
                context_module.set_invocation_context(previous_invocation_ctx)

        for module in (spans_module, instrumentor_module):
            self._patched_modules.append(
                (module, "set_response_content", getattr(module, "set_response_content"))
            )
            self._patched_modules.append(
                (module, "set_result_attributes", getattr(module, "set_result_attributes"))
            )
            module.set_response_content = patched_set_response_content
            module.set_result_attributes = patched_set_result_attributes

        self._patched_modules.append(
            (
                instrumentor_class,
                "_wrap_client_query",
                getattr(instrumentor_class, "_wrap_client_query"),
            )
        )
        self._patched_modules.append(
            (
                instrumentor_class,
                "_instrumented_receive_response",
                getattr(instrumentor_class, "_instrumented_receive_response"),
            )
        )
        instrumentor_class._wrap_client_query = patched_wrap_client_query
        instrumentor_class._instrumented_receive_response = (
            patched_instrumented_receive_response
        )

        self._patched_modules.append(
            (Query, "_handle_control_request", getattr(Query, "_handle_control_request"))
        )
        Query._handle_control_request = patched_handle_control_request

        return True

    def _restore_upstream_helpers(self) -> None:
        while self._patched_modules:
            module, attr_name, original = self._patched_modules.pop()
            setattr(module, attr_name, original)

    def activate(self) -> None:
        if self._is_instrumented:
            return

        try:
            upstream_instrumentor_class = _load_upstream_instrumentor_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Claude Agent SDK instrumentation — missing dependency: %s",
                exc,
            )
            return

        if not self._patch_upstream_helpers():
            return

        tracer_provider = trace.get_tracer_provider()
        if self._processor is None:
            self._processor = ClaudeAgentSDKSpanProcessor()
        self._register_processor(tracer_provider=tracer_provider, processor=self._processor)

        self._otel_instrumentor = upstream_instrumentor_class()
        try:
            self._otel_instrumentor.instrument(
                tracer_provider=tracer_provider,
                agent_name=self._agent_name,
                capture_content=self._capture_content,
            )
        except Exception as exc:
            self._unregister_processor(
                tracer_provider=tracer_provider,
                processor=self._processor,
            )
            self._otel_instrumentor = None
            self._restore_upstream_helpers()
            logger.warning(
                "Failed to activate Claude Agent SDK instrumentation: %s",
                exc,
            )
            return

        self._is_instrumented = True
        logger.info("Claude Agent SDK instrumentation activated")

    def deactivate(self) -> None:
        if not self._is_instrumented:
            return

        try:
            if self._otel_instrumentor is not None:
                self._otel_instrumentor.uninstrument()
        finally:
            self._otel_instrumentor = None
            self._restore_upstream_helpers()
            if self._processor is not None:
                self._unregister_processor(
                    tracer_provider=trace.get_tracer_provider(),
                    processor=self._processor,
                )

        self._is_instrumented = False
        logger.info("Claude Agent SDK instrumentation deactivated")
