"""Normalize Claude Agent SDK OTEL spans for the Respan OTLP pipeline."""

from __future__ import annotations

import ast
import json
import logging
import threading
from typing import Any, Mapping

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

from respan_instrumentation_claude_agent_sdk._constants import (
    CLAUDE_AGENT_SDK_CONVERSATION_ID_ATTR,
    CLAUDE_AGENT_SDK_INPUT_MESSAGES_ATTR,
    CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR,
    CLAUDE_AGENT_SDK_RESPONSE_MODEL_ATTR,
    CLAUDE_AGENT_SDK_STRIP_ATTRS,
    CLAUDE_AGENT_SDK_SYSTEM_INSTRUCTIONS_ATTR,
    CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR,
    CLAUDE_AGENT_SDK_TOOL_DEFINITIONS_ATTR,
    CLAUDE_AGENT_SDK_USAGE_CACHE_CREATION_INPUT_TOKENS_ATTR,
    CLAUDE_AGENT_SDK_USAGE_CACHE_READ_INPUT_TOKENS_ATTR,
    CLAUDE_AGENT_SDK_USAGE_INPUT_TOKENS_ATTR,
    CLAUDE_AGENT_SDK_USAGE_OUTPUT_TOKENS_ATTR,
    INPUT_VALUE_ATTR,
    OUTPUT_VALUE_ATTR,
    RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR,
    RESPAN_OVERRIDE_INPUT_ATTR,
    RESPAN_OVERRIDE_MODEL_ATTR,
    RESPAN_OVERRIDE_OUTPUT_ATTR,
    RESPAN_OVERRIDE_TOOL_CALLS_ATTR,
    RESPAN_OVERRIDE_TOOLS_ATTR,
    RESPAN_OVERRIDE_PROMPT_CACHE_CREATION_TOKENS_ATTR,
    RESPAN_OVERRIDE_PROMPT_CACHE_HIT_TOKENS_ATTR,
    RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR,
    RESPAN_OVERRIDE_SPAN_TOOLS_ATTR,
    RESPAN_OVERRIDE_SPAN_WORKFLOW_NAME_ATTR,
    RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

logger = logging.getLogger(__name__)

_CLAUDE_AGENT_OPERATION_NAME = "invoke_agent"
_CLAUDE_TOOL_OPERATION_NAME = "execute_tool"


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        stripped = value.strip()
        if stripped[:1] not in {"{", "[", "("}:
            return None
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed_value = _safe_json_loads(value)
        if parsed_value is not None and value.strip()[:1] in {"{", "[", "("}:
            return json.dumps(parsed_value, default=str)
        return value
    return json.dumps(value, default=str)


def _set_if_missing(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if attrs.get(key) in (None, "", (), []):
        attrs[key] = value


def _extract_first(attrs: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, "", (), []):
            return value
    return None


def _extract_agent_name(span: ReadableSpan, attrs: Mapping[str, Any]) -> str:
    raw_agent_name = attrs.get(GEN_AI_AGENT_NAME)
    if isinstance(raw_agent_name, str) and raw_agent_name:
        return raw_agent_name

    span_name = span.name.strip()
    if span_name.startswith(f"{_CLAUDE_AGENT_OPERATION_NAME} "):
        parsed_agent_name = span_name[len(_CLAUDE_AGENT_OPERATION_NAME) + 1 :].strip()
        if parsed_agent_name:
            return parsed_agent_name
    return _CLAUDE_AGENT_OPERATION_NAME


def _extract_model(attrs: Mapping[str, Any]) -> str | None:
    raw_model = _extract_first(
        attrs,
        (
            CLAUDE_AGENT_SDK_RESPONSE_MODEL_ATTR,
            LLM_REQUEST_MODEL,
        ),
    )
    if isinstance(raw_model, str) and raw_model:
        return raw_model
    return None


def _extract_usage(
    attrs: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, int | None]:
    prompt_tokens = attrs.get(CLAUDE_AGENT_SDK_USAGE_INPUT_TOKENS_ATTR)
    completion_tokens = attrs.get(CLAUDE_AGENT_SDK_USAGE_OUTPUT_TOKENS_ATTR)
    cache_hit_tokens = attrs.get(CLAUDE_AGENT_SDK_USAGE_CACHE_READ_INPUT_TOKENS_ATTR)
    cache_creation_tokens = attrs.get(
        CLAUDE_AGENT_SDK_USAGE_CACHE_CREATION_INPUT_TOKENS_ATTR
    )
    return (
        prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens if isinstance(completion_tokens, int) else None,
        cache_hit_tokens if isinstance(cache_hit_tokens, int) else None,
        cache_creation_tokens if isinstance(cache_creation_tokens, int) else None,
    )


def _extract_messages(attrs: Mapping[str, Any], attr_name: str) -> list[Any] | None:
    messages = _safe_json_loads(attrs.get(attr_name))
    if isinstance(messages, list):
        return messages
    return None


def _extract_input_output(attrs: Mapping[str, Any]) -> tuple[str | None, str | None]:
    input_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_INPUT_MESSAGES_ATTR)
    if input_messages is not None:
        system_instructions = attrs.get(CLAUDE_AGENT_SDK_SYSTEM_INSTRUCTIONS_ATTR)
        normalized_input_messages = []
        if isinstance(system_instructions, str) and system_instructions:
            normalized_input_messages.append(
                {"role": "system", "content": system_instructions}
            )
        normalized_input_messages.extend(input_messages)
        input_value = json.dumps(normalized_input_messages, default=str)
    else:
        input_value = _json_string(
            _extract_first(
                attrs,
                (
                    INPUT_VALUE_ATTR,
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                ),
            )
        )

    output_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR)
    if output_messages is not None:
        output_value = json.dumps(output_messages, default=str)
    else:
        output_value = _json_string(
            _extract_first(
                attrs,
                (
                    OUTPUT_VALUE_ATTR,
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                ),
            )
        )

    return input_value, output_value


def _normalize_tool_definition(tool_definition: Any) -> dict[str, Any] | None:
    if isinstance(tool_definition, str) and tool_definition:
        return {
            "type": "function",
            "function": {"name": tool_definition},
        }

    if not isinstance(tool_definition, Mapping):
        return None

    function_payload = tool_definition.get("function")
    if isinstance(function_payload, Mapping):
        normalized = {
            "type": tool_definition.get("type", "function"),
            "function": {"name": function_payload.get("name")},
        }
        for key in ("description", "parameters", "strict"):
            value = function_payload.get(key)
            if value is not None:
                normalized["function"][key] = value
        if normalized["function"].get("name"):
            return normalized
        return None

    tool_name = tool_definition.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    normalized_function: dict[str, Any] = {"name": tool_name}
    description = tool_definition.get("description")
    if description is not None:
        normalized_function["description"] = description
    parameters = tool_definition.get("input_schema") or tool_definition.get("parameters")
    if parameters is not None:
        normalized_function["parameters"] = parameters

    return {
        "type": tool_definition.get("type", "function"),
        "function": normalized_function,
    }


def _extract_tools(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw_tool_definitions = _safe_json_loads(
        attrs.get(CLAUDE_AGENT_SDK_TOOL_DEFINITIONS_ATTR)
    )
    if isinstance(raw_tool_definitions, Mapping):
        raw_tool_definitions = [raw_tool_definitions]
    if not isinstance(raw_tool_definitions, list):
        return None

    normalized_tools = []
    for tool_definition in raw_tool_definitions:
        normalized_tool = _normalize_tool_definition(tool_definition)
        if normalized_tool is not None:
            normalized_tools.append(normalized_tool)
    return normalized_tools or None


def _extract_tool_calls(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    output_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR)
    if not output_messages:
        return None

    tool_calls = []
    for message in output_messages:
        if not isinstance(message, Mapping):
            continue
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            continue

        for block in content_blocks:
            if not isinstance(block, Mapping):
                continue

            block_type = block.get("type")
            if block_type not in {None, "tool_use"}:
                continue

            tool_name = block.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            tool_arguments = _json_string(block.get("input", {}))
            tool_call_id = (
                block.get("id")
                or block.get("tool_use_id")
                or attrs.get(CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR)
                or ""
            )

            tool_calls.append(
                {
                    "id": str(tool_call_id),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_arguments or "",
                    },
                }
            )

    return tool_calls or None


def _extract_existing_tool_calls(
    attrs: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    raw_tool_calls = attrs.get(RESPAN_OVERRIDE_TOOL_CALLS_ATTR)
    if isinstance(raw_tool_calls, list):
        return [tool_call for tool_call in raw_tool_calls if isinstance(tool_call, Mapping)]

    parsed_tool_calls = _safe_json_loads(attrs.get(RESPAN_SPAN_TOOL_CALLS))
    if isinstance(parsed_tool_calls, list):
        return [
            tool_call
            for tool_call in parsed_tool_calls
            if isinstance(tool_call, Mapping)
        ]
    return None


def _build_tool_call_from_tool_span_attrs(
    attrs: Mapping[str, Any],
) -> dict[str, Any] | None:
    tool_name = _extract_first(
        attrs,
        (
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            GEN_AI_TOOL_NAME,
            "tool",
        ),
    )
    if not isinstance(tool_name, str) or not tool_name:
        return None

    tool_arguments = _json_string(
        _extract_first(
            attrs,
            (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                GEN_AI_TOOL_CALL_ARGUMENTS,
                RESPAN_OVERRIDE_INPUT_ATTR,
            ),
        )
    )
    tool_call_id = attrs.get(CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR) or ""

    return {
        "id": str(tool_call_id),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": tool_arguments or "",
        },
    }


def _tool_call_signature(tool_call: Mapping[str, Any]) -> tuple[str, str, str]:
    function_payload = tool_call.get("function")
    if not isinstance(function_payload, Mapping):
        return ("", "", "")

    tool_call_id = tool_call.get("id")
    tool_name = function_payload.get("name")
    tool_arguments = function_payload.get("arguments")
    return (
        str(tool_call_id or ""),
        str(tool_name or ""),
        str(tool_arguments or ""),
    )


def _merge_tool_calls(
    *tool_call_lists: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    merged_tool_calls = []
    seen_signatures = set()

    for tool_call_list in tool_call_lists:
        if not isinstance(tool_call_list, list):
            continue
        for tool_call in tool_call_list:
            if not isinstance(tool_call, Mapping):
                continue
            signature = _tool_call_signature(tool_call)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged_tool_calls.append(dict(tool_call))

    return merged_tool_calls or None


def _get_span_key(span: ReadableSpan) -> tuple[int, int] | None:
    span_context = span.get_span_context()
    if span_context is None:
        return None
    return span_context.trace_id, span_context.span_id


def _get_parent_span_key(span: ReadableSpan) -> tuple[int, int] | None:
    span_context = span.get_span_context()
    if span_context is None:
        return None

    parent_context = getattr(span, "parent", None)
    parent_span_id = getattr(parent_context, "span_id", None)
    if parent_span_id is None:
        return None

    return span_context.trace_id, parent_span_id


def is_claude_agent_sdk_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    operation_name = attrs.get(GEN_AI_OPERATION_NAME)
    return (
        operation_name in {_CLAUDE_AGENT_OPERATION_NAME, _CLAUDE_TOOL_OPERATION_NAME}
        or bool(attrs.get(GEN_AI_AGENT_NAME))
        or bool(attrs.get(GEN_AI_TOOL_NAME))
        or span.name.startswith(_CLAUDE_AGENT_OPERATION_NAME)
        or span.name.startswith(_CLAUDE_TOOL_OPERATION_NAME)
    )


def enrich_claude_agent_sdk_span(span: ReadableSpan) -> None:
    original_attrs = getattr(span, "_attributes", None)
    if original_attrs is None:
        return

    attrs = dict(original_attrs)
    if not is_claude_agent_sdk_span(span, attrs):
        return

    _set_if_missing(
        attrs,
        RESPAN_LOG_METHOD,
        LogMethodChoices.TRACING_INTEGRATION.value,
    )

    operation_name = attrs.get(GEN_AI_OPERATION_NAME)
    model = _extract_model(attrs)
    prompt_tokens, completion_tokens, cache_hit_tokens, cache_creation_tokens = (
        _extract_usage(attrs)
    )
    session_id = attrs.get(CLAUDE_AGENT_SDK_CONVERSATION_ID_ATTR)
    if isinstance(session_id, str) and session_id:
        _set_if_missing(attrs, RESPAN_SESSION_ID, session_id)

    if model is not None:
        _set_if_missing(attrs, LLM_REQUEST_MODEL, model)
        _set_if_missing(attrs, RESPAN_OVERRIDE_MODEL_ATTR, model)

    if prompt_tokens is not None:
        _set_if_missing(attrs, LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        _set_if_missing(attrs, RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR, prompt_tokens)
    if completion_tokens is not None:
        _set_if_missing(attrs, LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR,
            completion_tokens,
        )
    if prompt_tokens is not None or completion_tokens is not None:
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR,
            (prompt_tokens or 0) + (completion_tokens or 0),
        )
    if cache_hit_tokens is not None:
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_PROMPT_CACHE_HIT_TOKENS_ATTR,
            cache_hit_tokens,
        )
    if cache_creation_tokens is not None:
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_PROMPT_CACHE_CREATION_TOKENS_ATTR,
            cache_creation_tokens,
        )

    tools = _extract_tools(attrs)
    if tools is not None:
        _set_if_missing(attrs, RESPAN_SPAN_TOOLS, json.dumps(tools, default=str))
        _set_if_missing(attrs, RESPAN_OVERRIDE_TOOLS_ATTR, tools)
        tool_names = [
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, Mapping)
        ]
        tool_names = [name for name in tool_names if isinstance(name, str) and name]
        if tool_names:
            _set_if_missing(attrs, RESPAN_OVERRIDE_SPAN_TOOLS_ATTR, tool_names)

    if operation_name == _CLAUDE_TOOL_OPERATION_NAME or attrs.get(GEN_AI_TOOL_NAME):
        tool_name = attrs.get(GEN_AI_TOOL_NAME)
        tool_name = tool_name if isinstance(tool_name, str) and tool_name else "tool"
        tool_input = _json_string(attrs.get(GEN_AI_TOOL_CALL_ARGUMENTS))
        tool_output = _json_string(attrs.get(GEN_AI_TOOL_CALL_RESULT))

        _set_if_missing(attrs, RESPAN_LOG_TYPE, LOG_TYPE_TOOL)
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TOOL.value,
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, tool_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, tool_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, tool_input)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, tool_output)
        _set_if_missing(attrs, RESPAN_OVERRIDE_SPAN_TOOLS_ATTR, [tool_name])
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_TOOLS_ATTR,
            [{"type": "function", "function": {"name": tool_name}}],
        )
        _set_if_missing(attrs, RESPAN_OVERRIDE_INPUT_ATTR, tool_input)
        _set_if_missing(attrs, RESPAN_OVERRIDE_OUTPUT_ATTR, tool_output)
        attrs.pop(LLM_REQUEST_TYPE, None)
    else:
        agent_name = _extract_agent_name(span, attrs)
        input_value, output_value = _extract_input_output(attrs)
        tool_calls = _extract_tool_calls(attrs)

        _set_if_missing(attrs, RESPAN_LOG_TYPE, LOG_TYPE_AGENT)
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.AGENT.value,
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, agent_name)
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_SPAN_WORKFLOW_NAME_ATTR,
            agent_name,
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, input_value)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_value)
        _set_if_missing(attrs, RESPAN_OVERRIDE_INPUT_ATTR, input_value)
        _set_if_missing(attrs, RESPAN_OVERRIDE_OUTPUT_ATTR, output_value)
        if tool_calls is not None:
            _set_if_missing(
                attrs,
                RESPAN_SPAN_TOOL_CALLS,
                json.dumps(tool_calls, default=str),
            )
            _set_if_missing(attrs, RESPAN_OVERRIDE_TOOL_CALLS_ATTR, tool_calls)

    span._attributes = {
        key: value
        for key, value in attrs.items()
        if key not in CLAUDE_AGENT_SDK_STRIP_ATTRS
    }


class ClaudeAgentSDKSpanProcessor(SpanProcessor):
    """Normalize native Claude Agent SDK spans into Respan OTLP attributes."""

    def __init__(self) -> None:
        self._pending_tool_calls_by_parent: dict[
            tuple[int, int],
            list[tuple[int, dict[str, Any]]],
        ] = {}
        self._pending_tool_calls_lock = threading.Lock()

    def _store_pending_tool_call(self, span: ReadableSpan) -> None:
        parent_span_key = _get_parent_span_key(span)
        if parent_span_key is None:
            return

        attrs = getattr(span, "_attributes", None)
        if not isinstance(attrs, Mapping):
            return

        tool_call = _build_tool_call_from_tool_span_attrs(attrs)
        if tool_call is None:
            return

        tool_start_time = getattr(span, "start_time", None)
        sort_key = tool_start_time if isinstance(tool_start_time, int) else 0

        with self._pending_tool_calls_lock:
            pending_tool_calls = self._pending_tool_calls_by_parent.setdefault(
                parent_span_key,
                [],
            )
            pending_tool_calls.append((sort_key, tool_call))

    def _consume_pending_tool_calls(
        self,
        span: ReadableSpan,
    ) -> list[dict[str, Any]] | None:
        span_key = _get_span_key(span)
        if span_key is None:
            return None

        with self._pending_tool_calls_lock:
            pending_tool_calls = self._pending_tool_calls_by_parent.pop(span_key, None)

        if not pending_tool_calls:
            return None

        pending_tool_calls.sort(key=lambda entry: entry[0])
        return [tool_call for _, tool_call in pending_tool_calls]

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        try:
            enrich_claude_agent_sdk_span(span)

            attrs = getattr(span, "_attributes", None)
            if not isinstance(attrs, Mapping):
                return

            if attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_TOOL:
                self._store_pending_tool_call(span)
                return

            if attrs.get(RESPAN_LOG_TYPE) != LOG_TYPE_AGENT:
                return

            pending_tool_calls = self._consume_pending_tool_calls(span)
            existing_tool_calls = _extract_existing_tool_calls(attrs)
            merged_tool_calls = _merge_tool_calls(existing_tool_calls, pending_tool_calls)
            if merged_tool_calls is None:
                return

            updated_attrs = dict(attrs)
            updated_attrs[RESPAN_SPAN_TOOL_CALLS] = json.dumps(
                merged_tool_calls,
                default=str,
            )
            updated_attrs[RESPAN_OVERRIDE_TOOL_CALLS_ATTR] = merged_tool_calls
            span._attributes = updated_attrs
        except Exception:
            logger.exception("Failed to enrich Claude Agent SDK span")

    def shutdown(self) -> None:
        with self._pending_tool_calls_lock:
            self._pending_tool_calls_by_parent.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
