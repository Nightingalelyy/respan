"""Direct PydanticAI span normalization for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
    TraceloopSpanKindValues,
)

from respan_instrumentation_pydantic_ai._constants import (
    FINAL_RESULT_ATTR,
    MODEL_NAME_ATTR,
    PYDANTIC_AI_INPUT_MESSAGES_ATTR,
    PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
    PYDANTIC_AI_OUTPUT_MESSAGES_ATTR,
    PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
    PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME,
    PYDANTIC_AI_STRIP_ATTRS,
    PYDANTIC_AI_SYSTEM_ATTR,
    PYDANTIC_AI_TOOL_DEFINITIONS_ATTR,
    PYDANTIC_AI_TOOLS_ATTR,
    PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
    PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
    RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR,
    RESPAN_OVERRIDE_INPUT_ATTR,
    RESPAN_OVERRIDE_MODEL_ATTR,
    RESPAN_OVERRIDE_OUTPUT_ATTR,
    RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR,
    RESPAN_OVERRIDE_SPAN_TOOLS_ATTR,
    RESPAN_OVERRIDE_SPAN_WORKFLOW_NAME_ATTR,
    RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR,
    RESPAN_RESPONSE_FORMAT_ATTR,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_RESPONSE,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_TRANSCRIPTION,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_MODEL,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOLS,
)

logger = logging.getLogger(__name__)

_PYDANTIC_AI_OPERATION_TO_LOG_TYPE = {
    "chat": LOG_TYPE_CHAT,
    "embedding": LOG_TYPE_EMBEDDING,
    "response": LOG_TYPE_RESPONSE,
    "speech": LOG_TYPE_SPEECH,
    "transcription": LOG_TYPE_TRANSCRIPTION,
}


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _extract_request_parameters(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    request_parameters = _safe_json_loads(attrs.get(PYDANTIC_AI_REQUEST_PARAMETERS_ATTR))
    if isinstance(request_parameters, dict):
        return request_parameters
    return None


def _extract_messages(attrs: Mapping[str, Any], attr_name: str) -> list[Any] | None:
    value = _safe_json_loads(attrs.get(attr_name))
    if isinstance(value, list):
        return value
    return None


def _extract_tool_names(attrs: Mapping[str, Any]) -> list[str] | None:
    raw_tools = _safe_json_loads(attrs.get(PYDANTIC_AI_TOOLS_ATTR))
    if not isinstance(raw_tools, list):
        return None
    tool_names = [tool_name for tool_name in raw_tools if isinstance(tool_name, str)]
    return tool_names or None


def _normalize_tool_definition(tool_definition: Mapping[str, Any]) -> dict[str, Any] | None:
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
    parameters = tool_definition.get("parameters") or tool_definition.get(
        "parameters_json_schema"
    )
    if parameters is not None:
        normalized_function["parameters"] = parameters
    strict = tool_definition.get("strict")
    if strict is not None:
        normalized_function["strict"] = strict

    return {
        "type": tool_definition.get("type", "function"),
        "function": normalized_function,
    }


def _extract_tools(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    tool_definitions = _safe_json_loads(attrs.get(PYDANTIC_AI_TOOL_DEFINITIONS_ATTR))
    if not isinstance(tool_definitions, list):
        request_parameters = _extract_request_parameters(attrs)
        if request_parameters is None:
            return None
        tool_definitions = [
            *(request_parameters.get("function_tools") or []),
            *(request_parameters.get("output_tools") or []),
        ]

    normalized_tools = []
    for tool_definition in tool_definitions:
        if not isinstance(tool_definition, Mapping):
            continue
        normalized_tool = _normalize_tool_definition(tool_definition)
        if normalized_tool is not None:
            normalized_tools.append(normalized_tool)
    return normalized_tools or None


def _extract_response_format(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    existing = attrs.get(RESPAN_RESPONSE_FORMAT_ATTR)
    if isinstance(existing, dict):
        return existing
    parsed_existing = _safe_json_loads(existing)
    if isinstance(parsed_existing, dict):
        return parsed_existing

    request_parameters = _extract_request_parameters(attrs)
    if request_parameters is None:
        return None

    output_mode = request_parameters.get("output_mode")
    if output_mode == "text":
        return {"type": "text"}
    if output_mode == "image":
        return {"type": "image"}
    if output_mode not in {"native", "prompted"}:
        return None

    output_object = request_parameters.get("output_object")
    if not isinstance(output_object, dict):
        return {"type": "json_schema"}

    json_schema_payload: dict[str, Any] = {
        "schema": output_object.get("json_schema") or {}
    }
    for key in ("name", "description", "strict"):
        value = output_object.get(key)
        if value is not None:
            json_schema_payload[key] = value

    return {"type": "json_schema", "json_schema": json_schema_payload}


def _extract_model(attrs: Mapping[str, Any]) -> str | None:
    for key in (LLM_REQUEST_MODEL, MODEL_NAME_ATTR, RESPAN_OVERRIDE_MODEL_ATTR):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_usage(attrs: Mapping[str, Any]) -> tuple[int | None, int | None]:
    prompt_tokens = attrs.get(PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR)
    completion_tokens = attrs.get(PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR)
    return (
        prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens if isinstance(completion_tokens, int) else None,
    )


def _extract_log_type(span: ReadableSpan, attrs: Mapping[str, Any]) -> str | None:
    if isinstance(attrs.get(GEN_AI_TOOL_NAME), str):
        return LOG_TYPE_TOOL
    if isinstance(attrs.get(GEN_AI_AGENT_NAME), str) or isinstance(
        attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR), str
    ):
        return LOG_TYPE_AGENT

    operation_name = attrs.get(GEN_AI_OPERATION_NAME)
    if isinstance(operation_name, str):
        return _PYDANTIC_AI_OPERATION_TO_LOG_TYPE.get(operation_name)

    if span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME and _extract_tool_names(attrs):
        return LOG_TYPE_TASK
    return None


def is_pydantic_ai_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    return (
        bool(attrs.get(PYDANTIC_AI_SYSTEM_ATTR))
        or PYDANTIC_AI_REQUEST_PARAMETERS_ATTR in attrs
        or PYDANTIC_AI_TOOL_DEFINITIONS_ATTR in attrs
        or bool(attrs.get(GEN_AI_TOOL_NAME))
        or bool(attrs.get(GEN_AI_AGENT_NAME))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR))
        or bool(attrs.get(GEN_AI_TOOL_CALL_ARGUMENTS))
        or bool(attrs.get(GEN_AI_TOOL_CALL_RESULT))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR))
        or span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME
        or FINAL_RESULT_ATTR in attrs
    )


def _set_if_missing(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    existing = attrs.get(key)
    if existing in (None, "", (), []):
        attrs[key] = value


def enrich_pydantic_ai_span(span: ReadableSpan) -> None:
    original_attrs = getattr(span, "_attributes", None)
    if original_attrs is None:
        return

    attrs = dict(original_attrs)
    if not is_pydantic_ai_span(span, attrs):
        return

    log_type = _extract_log_type(span, attrs)
    if log_type is None:
        return

    _set_if_missing(attrs, RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    _set_if_missing(attrs, RESPAN_LOG_TYPE, log_type)

    model = _extract_model(attrs)
    if model is not None:
        _set_if_missing(attrs, LLM_REQUEST_MODEL, model)
        _set_if_missing(attrs, RESPAN_OVERRIDE_MODEL_ATTR, model)

    prompt_tokens, completion_tokens = _extract_usage(attrs)
    if prompt_tokens is not None:
        _set_if_missing(attrs, LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        _set_if_missing(attrs, RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR, prompt_tokens)
    if completion_tokens is not None:
        _set_if_missing(attrs, LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        _set_if_missing(
            attrs, RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR, completion_tokens
        )
    if prompt_tokens is not None or completion_tokens is not None:
        _set_if_missing(
            attrs,
            RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR,
            (prompt_tokens or 0) + (completion_tokens or 0),
        )

    response_format = _extract_response_format(attrs)
    if response_format is not None:
        _set_if_missing(attrs, RESPAN_RESPONSE_FORMAT_ATTR, response_format)

    tools = _extract_tools(attrs)
    if tools is not None:
        _set_if_missing(attrs, RESPAN_SPAN_TOOLS, json.dumps(tools, default=str))
        tool_names = [
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, Mapping)
        ]
        tool_names = [name for name in tool_names if isinstance(name, str) and name]
        if tool_names:
            _set_if_missing(attrs, RESPAN_OVERRIDE_SPAN_TOOLS_ATTR, tool_names)

    tool_name = attrs.get(GEN_AI_TOOL_NAME)
    tool_name = tool_name if isinstance(tool_name, str) else None
    agent_name = attrs.get(GEN_AI_AGENT_NAME)
    if not isinstance(agent_name, str):
        legacy_agent_name = attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR)
        agent_name = legacy_agent_name if isinstance(legacy_agent_name, str) else None

    tool_input = _json_string(
        attrs.get(
            GEN_AI_TOOL_CALL_ARGUMENTS, attrs.get(PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR)
        )
    )
    tool_output = _json_string(
        attrs.get(
            GEN_AI_TOOL_CALL_RESULT, attrs.get(PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR)
        )
    )

    if log_type == LOG_TYPE_TOOL and tool_name is not None:
        _set_if_missing(
            attrs, SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, tool_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, tool_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, tool_input)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, tool_output)
        _set_if_missing(attrs, RESPAN_OVERRIDE_SPAN_TOOLS_ATTR, [tool_name])
        _set_if_missing(attrs, RESPAN_OVERRIDE_INPUT_ATTR, tool_input)
        _set_if_missing(attrs, RESPAN_OVERRIDE_OUTPUT_ATTR, tool_output)

    if log_type == LOG_TYPE_CHAT:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_SPAN_KIND, LLMRequestTypeValues.CHAT.value)
        input_messages = _extract_messages(attrs, PYDANTIC_AI_INPUT_MESSAGES_ATTR)
        output_messages = _extract_messages(attrs, PYDANTIC_AI_OUTPUT_MESSAGES_ATTR)
        if input_messages is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json.dumps(input_messages, default=str),
            )
        if output_messages is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(output_messages, default=str),
            )

    if log_type == LOG_TYPE_AGENT and agent_name is not None:
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.AGENT.value,
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, agent_name)
        _set_if_missing(attrs, RESPAN_OVERRIDE_SPAN_WORKFLOW_NAME_ATTR, agent_name)

    if log_type == LOG_TYPE_TASK and span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME:
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, "running_tools")
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, "running_tools")
        tool_names = _extract_tool_names(attrs)
        if tool_names:
            _set_if_missing(attrs, RESPAN_OVERRIDE_SPAN_TOOLS_ATTR, tool_names)

    span._attributes = {
        key: value for key, value in attrs.items() if key not in PYDANTIC_AI_STRIP_ATTRS
    }


class PydanticAISpanProcessor(SpanProcessor):
    """Normalize raw PydanticAI spans into Respan's OTLP conventions."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        try:
            enrich_pydantic_ai_span(span)
        except Exception:
            logger.exception("Failed to enrich PydanticAI span")

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
