"""Portkey-specific contract normalization after OpenInference translation."""

from __future__ import annotations

import re
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_portkey._constants import OPENINFERENCE_PORTKEY_MODULE
from respan_instrumentation_portkey._serialization import (
    json_dumps,
    parse_json,
    safe_text,
    sanitize_endpoint,
)
from respan_instrumentation_portkey._streaming import current_request

OTEL_SCOPE_NAME = "otel.scope.name"
GEN_AI_COMPLETION_TOOL_CALLS_ATTR = f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
LLM_REQUEST_FUNCTIONS_ATTR = TLSpanAttributes.LLM_REQUEST_FUNCTIONS
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

_OFF_CONTRACT_ALIAS_ATTRS = frozenset(
    {
        "completion_tokens",
        "has_tool_calls",
        "model",
        "parallel_tool_calls",
        "prompt_tokens",
        "span_tools",
        "tool_calls",
        "tools",
        "total_request_tokens",
        RESPAN_SPAN_HANDOFFS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_TOOLS,
    }
)
_OPENAI_OMIT_VALUE_PREFIX = "<openai.Omit object"
_JSON_ATTRS = {
    TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
    TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
    LLM_REQUEST_FUNCTIONS_ATTR,
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
}


def _is_portkey_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    if attrs.get(OTEL_SCOPE_NAME) == OPENINFERENCE_PORTKEY_MODULE:
        return True
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None) == OPENINFERENCE_PORTKEY_MODULE


def _has_parent(span: ReadableSpan) -> bool:
    parent = getattr(span, "parent", None)
    return bool(parent and getattr(parent, "span_id", 0))


def _dict(value: Any) -> dict[str, Any] | None:
    parsed = parse_json(value)
    return parsed if isinstance(parsed, dict) else None


def _response_tool_calls(output: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if output is None:
        return None
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return None
    return [call for call in calls[:50] if isinstance(call, dict)] or None


def _status_message(span: ReadableSpan, attrs: dict[str, Any]) -> str:
    message = attrs.get(ERROR_MESSAGE_ATTR)
    if isinstance(message, str) and message:
        return safe_text(message)
    description = getattr(getattr(span, "status", None), "description", None)
    if isinstance(description, str) and description:
        return safe_text(description)
    for event in getattr(span, "events", ()) or ():
        event_attrs = getattr(event, "attributes", None) or {}
        message = event_attrs.get("exception.message")
        if isinstance(message, str) and message:
            return safe_text(message)
    return "Portkey operation failed"


def _normalize_status(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    is_error = (
        getattr(getattr(span, "status", None), "status_code", None) is StatusCode.ERROR
    )
    explicit = None
    for key in ("http.response.status_code", "http.status_code", "status_code"):
        value = attrs.get(key)
        if isinstance(value, int) and 400 <= value <= 599:
            explicit = value
            break
    if not is_error and explicit is None:
        return
    message = _status_message(span, attrs)
    if explicit is None:
        match = re.search(
            r"(?i)\b(?:error|status)(?:\s+code)?\s*[:=]?\s*([45]\d\d)\b",
            message,
        )
        explicit = int(match.group(1)) if match else None
    code = explicit or 500
    attrs[ERROR_MESSAGE_ATTR] = message
    attrs["status_code"] = code
    attrs["http.response.status_code"] = code
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
        {"status": "error", "type": "PortkeyError", "message": message}
    )
    try:
        span._status = Status(StatusCode.ERROR, message)
        span._events = ()
    except Exception:  # noqa: BLE001 - fake/immutable spans keep sanitized attrs
        return


class PortkeySpanContractProcessor(SpanProcessor):
    """Normalize final Portkey spans into the current Respan contract."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original = getattr(span, "_attributes", None)
        if original is None:
            return
        attrs = dict(original)
        if not _is_portkey_span(span, attrs):
            return

        for key, value in list(attrs.items()):
            if isinstance(value, str) and value.startswith(_OPENAI_OMIT_VALUE_PREFIX):
                attrs.pop(key, None)

        entity_name = safe_text(
            str(attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_NAME) or "portkey.chat")
        )
        attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_CHAT
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = (
            entity_name if _has_parent(span) else ""
        )
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        attrs[TLSpanAttributes.LLM_SYSTEM] = "portkey"
        attrs[GEN_AI_PROVIDER_NAME] = "portkey"
        attrs.pop(TLSpanAttributes.TRACELOOP_SPAN_KIND, None)

        input_payload = _dict(attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_INPUT))
        active_request = current_request()
        if input_payload is None and active_request is not None:
            input_payload = active_request
        elif input_payload is not None and active_request is not None:
            input_payload = {**input_payload, **active_request}
        output_payload = _dict(attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT))
        if input_payload is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(input_payload)
        if output_payload is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(output_payload)

        tools = input_payload.get("tools") if input_payload is not None else None
        if not isinstance(tools, list):
            tools = parse_json(attrs.get(LLM_REQUEST_FUNCTIONS_ATTR))
        if not isinstance(tools, list):
            tools = parse_json(attrs.get(RESPAN_SPAN_TOOLS))
        if isinstance(tools, list) and tools:
            attrs[LLM_REQUEST_FUNCTIONS_ATTR] = json_dumps(tools)

        calls = _response_tool_calls(output_payload)
        if calls is None:
            parsed_calls = parse_json(attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR))
            calls = parsed_calls if isinstance(parsed_calls, list) else None
        if calls is None:
            parsed_calls = parse_json(attrs.get(RESPAN_SPAN_TOOL_CALLS))
            calls = parsed_calls if isinstance(parsed_calls, list) else None
        if calls:
            attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = json_dumps(calls)

        if input_payload is not None and input_payload.get("stream") is True:
            attrs[TLSpanAttributes.LLM_IS_STREAMING] = True

        _normalize_status(span, attrs)
        for alias in _OFF_CONTRACT_ALIAS_ATTRS:
            attrs.pop(alias, None)
        attrs.pop(OTEL_SCOPE_NAME, None)

        for key, value in list(attrs.items()):
            if key in _JSON_ATTRS and value is not None:
                attrs[key] = json_dumps(parse_json(value))
            elif isinstance(value, str):
                attrs[key] = (
                    sanitize_endpoint(value)
                    if "url" in key.lower() or "endpoint" in key.lower()
                    else safe_text(value)
                )
        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
