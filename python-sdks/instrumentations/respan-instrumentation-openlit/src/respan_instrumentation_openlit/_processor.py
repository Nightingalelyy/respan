"""Normalize native OpenLIT spans into the Respan span contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from respan_instrumentation_openlit._constants import (
    OFF_CONTRACT_ALIASES,
    OPENLIT_INSTRUMENTATION_NAME,
    OPENLIT_OPERATION_LOG_TYPES,
    OPENLIT_PROVIDER_USAGE,
    OPENLIT_REQUEST_PROVIDER,
    OPENLIT_RESPONSE_TOOL_CALLS,
    OPENLIT_SCOPE_PREFIX,
    OPENLIT_TOOL_ARGS,
    OPENLIT_TOOL_INPUT,
    OPENLIT_TOOL_OUTPUT,
    OPENLIT_USAGE_TOTAL_TOKENS,
    OPENLIT_WORKFLOW_INPUT,
    OPENLIT_WORKFLOW_OUTPUT,
    STANDARD_DB_ATTRIBUTES,
    STANDARD_GEN_AI_ATTRIBUTES,
)
from respan_instrumentation_openlit._serialization import (
    MAX_ATTRIBUTE_BYTES,
    MAX_COLLECTION_ITEMS,
    MAX_STRING_CHARS,
    safe_text,
    safe_url,
)
from respan_instrumentation_openlit._serialization import (
    json_string as _bounded_json_string,
)
from respan_instrumentation_openlit._serialization import (
    json_value as _bounded_json_value,
)

try:
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
except ImportError:  # pragma: no cover - supported OTel releases provide this module
    gen_ai_attributes = None


def _otel_key(name: str, fallback: str) -> str:
    if gen_ai_attributes is None:
        return fallback
    value = getattr(gen_ai_attributes, name, fallback)
    return str(getattr(value, "value", value))


GEN_AI_OPERATION_NAME = _otel_key("GEN_AI_OPERATION_NAME", "gen_ai.operation.name")
GEN_AI_PROVIDER_NAME = _otel_key("GEN_AI_PROVIDER_NAME", "gen_ai.provider.name")
GEN_AI_INPUT_MESSAGES = _otel_key("GEN_AI_INPUT_MESSAGES", "gen_ai.input.messages")
GEN_AI_OUTPUT_MESSAGES = _otel_key("GEN_AI_OUTPUT_MESSAGES", "gen_ai.output.messages")
GEN_AI_SYSTEM_INSTRUCTIONS = _otel_key(
    "GEN_AI_SYSTEM_INSTRUCTIONS", "gen_ai.system_instructions"
)
GEN_AI_TOOL_DEFINITIONS = _otel_key(
    "GEN_AI_TOOL_DEFINITIONS", "gen_ai.tool.definitions"
)
GEN_AI_TOOL_NAME = _otel_key("GEN_AI_TOOL_NAME", "gen_ai.tool.name")
GEN_AI_TOOL_CALL_ARGUMENTS = _otel_key(
    "GEN_AI_TOOL_CALL_ARGUMENTS", "gen_ai.tool.call.arguments"
)
GEN_AI_TOOL_CALL_RESULT = _otel_key(
    "GEN_AI_TOOL_CALL_RESULT", "gen_ai.tool.call.result"
)
GEN_AI_AGENT_NAME = _otel_key("GEN_AI_AGENT_NAME", "gen_ai.agent.name")
GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"
GEN_AI_USAGE_INPUT_TOKENS = _otel_key(
    "GEN_AI_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"
)
GEN_AI_USAGE_OUTPUT_TOKENS = _otel_key(
    "GEN_AI_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"
)
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = _otel_key(
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
    "gen_ai.usage.cache_read.input_tokens",
)
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = _otel_key(
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS",
    "gen_ai.usage.cache_creation.input_tokens",
)
DB_SYSTEM_NAME = "db.system.name"
DB_OPERATION_NAME = "db.operation.name"
DB_QUERY_TEXT = "db.query.text"

_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
_MODERN_INPUT_USAGE = getattr(
    TLSpanAttributes, "LLM_USAGE_INPUT_TOKENS", GEN_AI_USAGE_INPUT_TOKENS
)
_MODERN_OUTPUT_USAGE = getattr(
    TLSpanAttributes, "LLM_USAGE_OUTPUT_TOKENS", GEN_AI_USAGE_OUTPUT_TOKENS
)
_CACHE_READ_USAGE = getattr(
    TLSpanAttributes,
    "LLM_USAGE_CACHE_READ_INPUT_TOKENS",
    "llm.usage.cache_read_input_tokens",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES:
            try:
                return _bounded_json_value(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                pass
        return safe_text(value, limit=MAX_STRING_CHARS)
    return _bounded_json_value(value)


def _json_string(value: Any) -> str:
    return _bounded_json_string(_json_value(value))


def _sequence(value: Any) -> list[Any]:
    parsed = _json_value(value)
    if isinstance(parsed, list):
        return parsed[:MAX_COLLECTION_ITEMS]
    if parsed is None:
        return []
    return [parsed]


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return False


def _part_payload(part: Any) -> Any:
    if not isinstance(part, Mapping):
        return part
    for key in ("content", "text", "value"):
        if part.get(key) is not None:
            return part[key]
    return dict(part)


def _message_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if content is not None:
        return content
    parts = message.get("parts")
    if isinstance(parts, Sequence) and not isinstance(parts, str | bytes):
        values = [
            _part_payload(part)
            for part in parts
            if not (isinstance(part, Mapping) and part.get("type") == "tool_call")
        ]
        if len(values) == 1:
            return values[0]
        if values:
            return values
    return None


def _message_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = _json_value(message.get("tool_calls"))
    if isinstance(direct, list):
        return [
            dict(item)
            for item in direct[:MAX_COLLECTION_ITEMS]
            if isinstance(item, Mapping)
        ]

    calls: list[dict[str, Any]] = []
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
        return calls
    for part in parts[:MAX_COLLECTION_ITEMS]:
        if not isinstance(part, Mapping) or part.get("type") not in {
            "tool_call",
            "function_call",
        }:
            continue
        calls.append(
            {
                "id": part.get("id") or part.get("tool_call_id"),
                "type": "function",
                "function": {
                    "name": part.get("name"),
                    "arguments": part.get("arguments") or part.get("args") or "{}",
                },
            }
        )
    return calls


def _set_message_attributes(
    attrs: dict[str, Any],
    *,
    messages: list[Any],
    target_prefix: str,
) -> None:
    for index, raw_message in enumerate(messages[:MAX_COLLECTION_ITEMS]):
        if isinstance(raw_message, str):
            message: Mapping[str, Any] = {"role": "user", "content": raw_message}
        elif isinstance(raw_message, Mapping):
            message = raw_message
        else:
            message = {"role": "user", "content": raw_message}

        role = message.get("role") or (
            "assistant" if target_prefix == _COMPLETION_PREFIX else "user"
        )
        attrs[f"{target_prefix}{index}.role"] = safe_text(
            role, default="user", limit=64
        )
        content = _message_content(message)
        if content is not None:
            attrs[f"{target_prefix}{index}.content"] = (
                safe_text(content, limit=MAX_STRING_CHARS)
                if isinstance(content, str)
                else _json_string(content)
            )
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            attrs[f"{target_prefix}{index}.tool_calls"] = _json_string(tool_calls)


def _scope_name(span: ReadableSpan) -> str:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    return safe_text(getattr(scope, "name", ""), limit=256)


def _has_parent(span: ReadableSpan) -> bool:
    parent = getattr(span, "parent", None)
    if parent is None:
        return False
    is_valid = getattr(parent, "is_valid", None)
    if is_valid is not None:
        return bool(is_valid)
    span_id = getattr(parent, "span_id", None)
    return bool(span_id) if span_id is not None else True


def _is_openlit_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    scope_name = _scope_name(span)
    if scope_name == OPENLIT_SCOPE_PREFIX or scope_name.startswith(
        f"{OPENLIT_SCOPE_PREFIX}."
    ):
        return True
    return bool(
        attrs.get("gen_ai.sdk.version")
        and (attrs.get(GEN_AI_OPERATION_NAME) or attrs.get(DB_SYSTEM_NAME))
    )


def _operation_log_type(attrs: Mapping[str, Any]) -> str:
    operation = safe_text(attrs.get(GEN_AI_OPERATION_NAME), limit=128).lower()
    if attrs.get(DB_SYSTEM_NAME):
        return "task"
    return OPENLIT_OPERATION_LOG_TYPES.get(operation, "task")


def _entity_name(span: ReadableSpan, attrs: Mapping[str, Any], log_type: str) -> str:
    candidates: tuple[Any, ...]
    if log_type == "tool":
        candidates = (attrs.get(GEN_AI_TOOL_NAME), attrs.get(DB_OPERATION_NAME))
    elif log_type == "agent":
        candidates = (attrs.get(GEN_AI_AGENT_NAME),)
    elif log_type == "workflow":
        candidates = (attrs.get(GEN_AI_WORKFLOW_NAME),)
    elif log_type == "task" and attrs.get(DB_SYSTEM_NAME):
        candidates = (
            ".".join(
                filter(
                    None,
                    [
                        safe_text(attrs.get(DB_SYSTEM_NAME), limit=128),
                        safe_text(attrs.get(DB_OPERATION_NAME), limit=128),
                    ],
                )
            ),
        )
    else:
        candidates = ()
    for candidate in candidates:
        if candidate:
            return safe_text(candidate, default=OPENLIT_INSTRUMENTATION_NAME, limit=256)
    return safe_text(
        getattr(span, "name", None),
        default=OPENLIT_INSTRUMENTATION_NAME,
        limit=256,
    )


def _set_usage(attrs: dict[str, Any]) -> None:
    input_tokens = attrs.get(GEN_AI_USAGE_INPUT_TOKENS)
    output_tokens = attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS)
    total_tokens = attrs.pop(OPENLIT_USAGE_TOTAL_TOKENS, None)
    cache_read_tokens = attrs.get(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)

    if input_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
        attrs[_MODERN_INPUT_USAGE] = input_tokens
    if output_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
        attrs[_MODERN_OUTPUT_USAGE] = output_tokens
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        try:
            total_tokens = int(input_tokens) + int(output_tokens)
        except (TypeError, ValueError):
            total_tokens = None
    if total_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens
    if cache_read_tokens is not None:
        attrs[_CACHE_READ_USAGE] = cache_read_tokens


def _clear_synthetic_usage(attrs: dict[str, Any]) -> None:
    for key in (
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        "gen_ai.usage.cache_creation_input_tokens",
        "gen_ai.usage.cache_read_input_tokens",
        OPENLIT_USAGE_TOTAL_TOKENS,
        TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        _MODERN_INPUT_USAGE,
        _MODERN_OUTPUT_USAGE,
        _CACHE_READ_USAGE,
    ):
        attrs.pop(key, None)


def _strip_content(attrs: dict[str, Any]) -> None:
    for key in (
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_OUTPUT_MESSAGES,
        GEN_AI_SYSTEM_INSTRUCTIONS,
        GEN_AI_TOOL_DEFINITIONS,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.retrieval.query.text",
        "gen_ai.retrieval.documents",
        OPENLIT_RESPONSE_TOOL_CALLS,
        OPENLIT_TOOL_ARGS,
        OPENLIT_TOOL_INPUT,
        OPENLIT_TOOL_OUTPUT,
        OPENLIT_WORKFLOW_INPUT,
        OPENLIT_WORKFLOW_OUTPUT,
        DB_QUERY_TEXT,
        "db.query.parameter",
        TLSpanAttributes.LLM_REQUEST_FUNCTIONS,
        TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
        TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
    ):
        attrs.pop(key, None)
    for key in list(attrs):
        if key.startswith(
            (
                _PROMPT_PREFIX,
                _COMPLETION_PREFIX,
                "gen_ai.retrieval.",
                "db.query.parameter.",
            )
        ):
            attrs.pop(key, None)


def _strip_openlit_vendor_attributes(attrs: dict[str, Any]) -> None:
    """Drop OpenLIT-only fields while retaining standard span semantics."""

    for key in list(attrs):
        if (
            key.startswith("openlit.")
            or (
                key.startswith("gen_ai.")
                and key not in STANDARD_GEN_AI_ATTRIBUTES
                and not key.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            )
            or key.startswith("db.")
            and key not in STANDARD_DB_ATTRIBUTES
        ):
            attrs.pop(key, None)


def _sanitize_url_attributes(attrs: dict[str, Any]) -> None:
    for key in list(attrs):
        normalized = key.lower()
        if not (
            normalized in {"http.url", "url.full", "url.original"}
            or normalized.endswith((".url", "_url", ".base_url", "_base_url"))
        ):
            continue
        sanitized = safe_url(attrs[key])
        if sanitized:
            attrs[key] = sanitized
        else:
            attrs.pop(key, None)


def _backend_status_code(attrs: Mapping[str, Any], *, is_error: bool) -> int:
    for key in (
        "status_code",
        "http.response.status_code",
        "http.status_code",
        "gen_ai.response.status_code",
    ):
        value = attrs.get(key)
        try:
            if value is not None:
                code = int(value)
                return code if not is_error or code >= 400 else 500
        except (TypeError, ValueError):
            continue
    return 500 if is_error else 200


def _set_backend_status(
    span: ReadableSpan, attrs: dict[str, Any], *, capture_content: bool
) -> None:
    status = getattr(span, "status", None)
    otel_error = getattr(status, "status_code", None) is StatusCode.ERROR
    code = _backend_status_code(attrs, is_error=otel_error)
    is_error = otel_error or code >= 400
    if is_error and code < 400:
        code = 500
    attrs["status_code"] = code
    if not is_error:
        return

    message = attrs.get(ERROR_MESSAGE_ATTR) or getattr(status, "description", None)
    if not message:
        for event in getattr(span, "events", ()) or ():
            event_attrs = getattr(event, "attributes", None) or {}
            message = event_attrs.get("exception.message")
            if message:
                break
    message = (
        safe_text(message, default="OpenLIT operation failed", limit=1_024)
        if capture_content
        else "OpenLIT operation failed"
    )
    attrs[ERROR_MESSAGE_ATTR] = message
    sanitized_status = Status(StatusCode.ERROR, message)
    if hasattr(span, "_status"):
        span._status = sanitized_status
    else:
        try:
            span.status = sanitized_status
        except AttributeError:
            pass
    if capture_content:
        attrs.setdefault(
            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_string(
                {
                    "status": "error",
                    "error": safe_text(
                        attrs.get("error.type"), default="OpenLITError", limit=256
                    ),
                    "message": message,
                }
            ),
        )


def _remove_openlit_events(span: ReadableSpan) -> None:
    events = getattr(span, "_events", None)
    if events is None:
        return
    span._events = ()


def translate_openlit_span(span: ReadableSpan, *, capture_content: bool) -> bool:
    """Translate a completed native OpenLIT span in place.

    The processor edits the mutable SDK ReadableSpan before downstream
    exporters observe it. It returns False for spans not owned by OpenLIT.
    """

    original = getattr(span, "_attributes", None)
    if original is None:
        original = getattr(span, "attributes", None)
    attrs = dict(original or {})
    if not _is_openlit_span(span, attrs):
        return False

    log_type = _operation_log_type(attrs)
    entity_name = _entity_name(span, attrs, log_type)
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[RESPAN_LOG_TYPE] = log_type
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = (
        entity_name if _has_parent(span) else ""
    )

    provider = attrs.get(GEN_AI_PROVIDER_NAME) or attrs.pop(
        OPENLIT_REQUEST_PROVIDER, None
    )
    if provider is not None:
        attrs[TLSpanAttributes.LLM_SYSTEM] = safe_text(provider, limit=128)

    if _is_truthy(attrs.get("gen_ai.request.stream")):
        attrs[TLSpanAttributes.LLM_IS_STREAMING] = True

    operation = safe_text(attrs.get(GEN_AI_OPERATION_NAME), limit=128).lower()
    if log_type in {"chat", "text"}:
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    elif operation in {"embeddings", "embedding"}:
        attrs[TLSpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.EMBEDDING.value

    provider_usage = attrs.pop(OPENLIT_PROVIDER_USAGE, None)
    if provider_usage is False:
        _clear_synthetic_usage(attrs)
    else:
        _set_usage(attrs)

    if capture_content:
        input_messages = _sequence(attrs.get(GEN_AI_INPUT_MESSAGES))
        output_messages = _sequence(attrs.get(GEN_AI_OUTPUT_MESSAGES))
        if input_messages:
            if log_type == "embedding":
                if TLSpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs:
                    embedding_input = [
                        _message_content(message)
                        if isinstance(message, Mapping)
                        else message
                        for message in input_messages
                    ]
                    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                        embedding_input
                    )
                attrs.pop(GEN_AI_INPUT_MESSAGES, None)
                for key in list(attrs):
                    if key.startswith(_PROMPT_PREFIX):
                        attrs.pop(key, None)
            else:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                    input_messages
                )
                _set_message_attributes(
                    attrs, messages=input_messages, target_prefix=_PROMPT_PREFIX
                )
        if output_messages:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
                output_messages
            )
            _set_message_attributes(
                attrs, messages=output_messages, target_prefix=_COMPLETION_PREFIX
            )

        tool_definitions = _json_value(attrs.get(GEN_AI_TOOL_DEFINITIONS))
        if tool_definitions:
            attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = _json_string(
                tool_definitions
            )
        response_tool_calls = _json_value(attrs.pop(OPENLIT_RESPONSE_TOOL_CALLS, None))
        if response_tool_calls:
            attrs[f"{_COMPLETION_PREFIX}0.tool_calls"] = _json_string(
                response_tool_calls
            )

        if log_type == "tool":
            tool_input = attrs.get(GEN_AI_TOOL_CALL_ARGUMENTS)
            if tool_input is None:
                tool_input = attrs.get(OPENLIT_TOOL_ARGS, attrs.get(OPENLIT_TOOL_INPUT))
            tool_output = attrs.get(
                GEN_AI_TOOL_CALL_RESULT, attrs.get(OPENLIT_TOOL_OUTPUT)
            )
            if tool_input is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                    _json_value(tool_input)
                )
            if tool_output is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
                    _json_value(tool_output)
                )
            for key in list(attrs):
                if key.startswith("gen_ai.tool."):
                    attrs.pop(key, None)
        elif log_type == "workflow":
            workflow_input = attrs.get(OPENLIT_WORKFLOW_INPUT)
            workflow_output = attrs.get(OPENLIT_WORKFLOW_OUTPUT)
            if workflow_input is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                    _json_value(workflow_input)
                )
            if workflow_output is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
                    _json_value(workflow_output)
                )
        elif attrs.get(DB_QUERY_TEXT) is not None:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_string(
                {"query": attrs.get(DB_QUERY_TEXT)}
            )
    else:
        _strip_content(attrs)

    for key in list(attrs):
        if key.startswith("gen_ai.tool."):
            attrs.pop(key, None)

    _set_backend_status(span, attrs, capture_content=capture_content)
    _remove_openlit_events(span)
    _sanitize_url_attributes(attrs)
    for key in OFF_CONTRACT_ALIASES:
        attrs.pop(key, None)
    _strip_openlit_vendor_attributes(attrs)
    span._attributes = attrs
    return True


class OpenLITSpanProcessor(SpanProcessor):
    """First-in-pipeline processor that normalizes OpenLIT native spans."""

    def __init__(self, *, capture_content: bool = True) -> None:
        self.capture_content = capture_content

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        del span, parent_context

    def on_end(self, span: ReadableSpan) -> None:
        translate_openlit_span(span, capture_content=self.capture_content)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True
