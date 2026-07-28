"""Translate AgentOps decorator spans into the Respan span contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_agentops._constants import (
    AGENT_NAME,
    AGENTOPS_DECORATOR_INPUT_TEMPLATE,
    AGENTOPS_DECORATOR_OUTPUT_TEMPLATE,
    AGENTOPS_ENTITY_INPUT,
    AGENTOPS_ENTITY_NAME,
    AGENTOPS_ENTITY_OUTPUT,
    AGENTOPS_INSTRUMENTATION_NAME,
    AGENTOPS_KIND_LOG_TYPES,
    AGENTOPS_REQUEST_FUNCTIONS,
    AGENTOPS_REQUEST_TYPE,
    AGENTOPS_SCOPE_PREFIX,
    AGENTOPS_SESSION_END_STATE,
    AGENTOPS_SPAN_KIND,
    AGENTOPS_TAGS,
    AGENTOPS_USAGE_TOTAL_TOKENS,
    OFF_CONTRACT_ALIASES,
    OPERATION_NAME,
    OPERATION_VERSION,
    TOOL_NAME,
)
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)


def _scope_name(span: ReadableSpan) -> str:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    return str(getattr(scope, "name", "") or "")


def _is_agentops_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    scope_name = _scope_name(span)
    return bool(
        AGENTOPS_SPAN_KIND in attrs
        or scope_name == AGENTOPS_SCOPE_PREFIX
        or scope_name.startswith(f"{AGENTOPS_SCOPE_PREFIX}.")
    )


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return json.dumps(value, default=str, ensure_ascii=False)
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _metadata_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(decoded, Mapping):
            return dict(decoded)
    return {}


def _content(
    attrs: Mapping[str, Any],
    *,
    kind: str,
    direction: str,
) -> Any:
    key = (
        AGENTOPS_DECORATOR_INPUT_TEMPLATE
        if direction == "input"
        else AGENTOPS_DECORATOR_OUTPUT_TEMPLATE
    ).format(kind=kind)
    generic_key = (
        AGENTOPS_ENTITY_INPUT if direction == "input" else AGENTOPS_ENTITY_OUTPUT
    )
    return attrs.get(key, attrs.get(generic_key))


def _entity_name(span: ReadableSpan, attrs: Mapping[str, Any], kind: str) -> str:
    candidates: tuple[Any, ...]
    if kind == "agent":
        candidates = (attrs.get(AGENT_NAME), attrs.get(AGENTOPS_ENTITY_NAME))
    elif kind == "tool":
        candidates = (attrs.get(TOOL_NAME), attrs.get(AGENTOPS_ENTITY_NAME))
    else:
        candidates = (attrs.get(AGENTOPS_ENTITY_NAME), attrs.get(OPERATION_NAME))
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return str(getattr(span, "name", None) or AGENTOPS_INSTRUMENTATION_NAME)


def _set_llm_fields(attrs: dict[str, Any], kind: str) -> None:
    if kind not in {"llm", "text"}:
        return
    request_type = attrs.pop(AGENTOPS_REQUEST_TYPE, None)
    if request_type is None:
        request_type = (
            LLMRequestTypeValues.CHAT.value
            if kind == "llm"
            else LLMRequestTypeValues.COMPLETION.value
        )
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = str(request_type)

    functions = attrs.pop(AGENTOPS_REQUEST_FUNCTIONS, None)
    if functions is not None:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = _json_string(functions)

    prompt_tokens = attrs.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
    completion_tokens = attrs.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS)
    if prompt_tokens is not None:
        modern_input = getattr(
            SpanAttributes,
            "LLM_USAGE_INPUT_TOKENS",
            "gen_ai.usage.input_tokens",
        )
        attrs.setdefault(modern_input, prompt_tokens)
    if completion_tokens is not None:
        modern_output = getattr(
            SpanAttributes,
            "LLM_USAGE_OUTPUT_TOKENS",
            "gen_ai.usage.output_tokens",
        )
        attrs.setdefault(modern_output, completion_tokens)
    total_tokens = attrs.pop(AGENTOPS_USAGE_TOTAL_TOKENS, None)
    if total_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def _set_status(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    status = getattr(span, "status", None)
    status_code = getattr(status, "status_code", StatusCode.UNSET)
    description = getattr(status, "description", None)
    end_state = attrs.get(AGENTOPS_SESSION_END_STATE)
    normalized_state = str(end_state or "").lower()
    failed_state = any(
        marker in normalized_state
        for marker in ("error", "fail", "indeterminate", "unknown")
    )
    if status_code == StatusCode.ERROR or failed_state:
        message = str(
            description or f"AgentOps trace ended with {end_state or 'error'}"
        )
        attrs["status_code"] = 500
        attrs[ERROR_MESSAGE_ATTR] = message
    else:
        attrs.setdefault("status_code", 200)


def _strip_agentops_fields(attrs: dict[str, Any]) -> None:
    for key in tuple(attrs):
        if key.startswith("agentops.") or key in {OPERATION_NAME, OPERATION_VERSION}:
            attrs.pop(key, None)


def translate_agentops_span(
    span: ReadableSpan,
    *,
    capture_content: bool,
) -> bool:
    """Translate one completed AgentOps span in place."""

    original = getattr(span, "_attributes", None)
    if original is None:
        original = getattr(span, "attributes", None)
    attrs = dict(original or {})
    if not _is_agentops_span(span, attrs):
        return False

    kind = str(attrs.get(AGENTOPS_SPAN_KIND) or "task").lower()
    log_type = AGENTOPS_KIND_LOG_TYPES.get(kind, "task")
    entity_name = _entity_name(span, attrs, kind)
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[RESPAN_LOG_TYPE] = log_type
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = entity_name

    input_value = _content(attrs, kind=kind, direction="input")
    output_value = _content(attrs, kind=kind, direction="output")
    if capture_content:
        if input_value is not None:
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(input_value)
        if output_value is not None:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(output_value)
    else:
        attrs.pop(SpanAttributes.TRACELOOP_ENTITY_INPUT, None)
        attrs.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, None)

    metadata = _metadata_value(attrs.get(RESPAN_METADATA))
    agentops_metadata: dict[str, Any] = {"kind": kind}
    if attrs.get(OPERATION_VERSION) is not None:
        agentops_metadata["operation_version"] = attrs[OPERATION_VERSION]
    if attrs.get(AGENTOPS_TAGS) is not None:
        agentops_metadata["tags"] = attrs[AGENTOPS_TAGS]
    if attrs.get(AGENTOPS_SESSION_END_STATE) is not None:
        agentops_metadata["end_state"] = str(attrs[AGENTOPS_SESSION_END_STATE])
    metadata["agentops"] = agentops_metadata
    attrs[RESPAN_METADATA] = _json_string(metadata)

    _set_llm_fields(attrs, kind)
    _set_status(span, attrs)
    for key in OFF_CONTRACT_ALIASES:
        attrs.pop(key, None)
    _strip_agentops_fields(attrs)
    span._attributes = attrs
    return True


class AgentOpsSpanProcessor(SpanProcessor):
    """Normalize AgentOps-owned spans before downstream Respan exporters."""

    def __init__(self, *, capture_content: bool = True) -> None:
        self.capture_content = capture_content

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        del span, parent_context

    def on_end(self, span: ReadableSpan) -> None:
        translate_agentops_span(span, capture_content=self.capture_content)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True
