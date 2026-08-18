"""smolagents-specific span cleanup for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import re
from ast import literal_eval
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_smolagents._constants import (
    ASSISTANT_ROLE,
    GEN_AI_COMPLETION_CONTENT_ATTR,
    GEN_AI_COMPLETION_ROLE_ATTR,
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
    LLM_REQUEST_FUNCTIONS_ATTR,
    OPENINFERENCE_INPUT_MESSAGES_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENTS_ATTR,
    OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
    OPENINFERENCE_SMOLAGENTS_MODULE,
    OPENINFERENCE_SPAN_KIND_ATTR,
    OPENINFERENCE_TOOL_NAME_ATTR,
    OTEL_SCOPE_NAME,
    SMOLAGENTS_FINAL_ANSWER_ARGUMENT,
    SMOLAGENTS_FINAL_ANSWER_TOOL_NAME,
    SMOLAGENTS_TOOL_NAME_HINT,
    SPAN_ALIAS_COMPLETION_TOKENS,
    SPAN_ALIAS_MODEL,
    SPAN_ALIAS_PROMPT_TOKENS,
    SPAN_ALIAS_TOOL_CALLS,
    SPAN_ALIAS_TOOLS,
    SPAN_ALIAS_TOTAL_REQUEST_TOKENS,
    TOOL_CALL_FUNCTION_ARGUMENTS_FIELD,
    TOOL_CALL_FUNCTION_FIELD,
    TOOL_CALL_FUNCTION_NAME_FIELD,
)
from respan_instrumentation_smolagents._serialization import json_string, redact_text

_TOOL_CALLS_ATTR_RE = re.compile(
    rf"^{re.escape(TLSpanAttributes.LLM_COMPLETIONS)}\.\d+\.tool_calls$"
    rf"|^{re.escape(TLSpanAttributes.LLM_PROMPTS)}\.\d+\.tool_calls$"
)

_OFF_CONTRACT_ALIAS_ATTRS = frozenset(
    {
        SPAN_ALIAS_MODEL,
        SPAN_ALIAS_PROMPT_TOKENS,
        SPAN_ALIAS_COMPLETION_TOKENS,
        SPAN_ALIAS_TOTAL_REQUEST_TOKENS,
        SPAN_ALIAS_TOOLS,
        SPAN_ALIAS_TOOL_CALLS,
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
)
_STEP_NAME = re.compile(r"^Step\s+(\d+)$", re.IGNORECASE)


def _is_smolagents_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    if attrs.get(OTEL_SCOPE_NAME) == OPENINFERENCE_SMOLAGENTS_MODULE:
        return True

    instrumentation_scope = getattr(span, "instrumentation_scope", None)
    return (
        getattr(instrumentation_scope, "name", None) == OPENINFERENCE_SMOLAGENTS_MODULE
    )


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return json_string(value)


def _bounded_entity_value(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = redact_text(value)
    return json_string(value)


def _json_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value[:51])
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _json_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json_string(value) or ""


def _final_stream_output(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(
        r"ActionOutput\(output=(?P<literal>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")"
        r",\s*is_final_answer=True\)",
        value,
    )
    if match is None:
        return None
    try:
        output = literal_eval(match.group("literal"))
    except (SyntaxError, ValueError):
        return None
    return output if isinstance(output, str) else None


def _extract_final_answer_content(tool_calls: list[Any]) -> str | None:
    if len(tool_calls) != 1:
        return None

    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get(TOOL_CALL_FUNCTION_FIELD)
    if not isinstance(function, dict):
        return None
    if function.get(TOOL_CALL_FUNCTION_NAME_FIELD) != SMOLAGENTS_FINAL_ANSWER_TOOL_NAME:
        return None

    arguments = _json_dict(function.get(TOOL_CALL_FUNCTION_ARGUMENTS_FIELD))
    if arguments is None or SMOLAGENTS_FINAL_ANSWER_ARGUMENT not in arguments:
        return None

    return _stringify_content(arguments[SMOLAGENTS_FINAL_ANSWER_ARGUMENT])


def _normalize_structured_contract_attrs(attrs: dict[str, Any]) -> None:
    if attrs.get(LLM_REQUEST_FUNCTIONS_ATTR) is None:
        helper_tools = _json_string(attrs.get(RESPAN_SPAN_TOOLS))
        if helper_tools:
            attrs[LLM_REQUEST_FUNCTIONS_ATTR] = helper_tools
    else:
        tools = _json_string(attrs[LLM_REQUEST_FUNCTIONS_ATTR])
        if tools:
            attrs[LLM_REQUEST_FUNCTIONS_ATTR] = tools

    if attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR) is None:
        helper_tool_calls = _json_list(attrs.get(RESPAN_SPAN_TOOL_CALLS))
        if helper_tool_calls:
            attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = helper_tool_calls

    tool_call_roots: set[str] = set()
    for key, value in list(attrs.items()):
        if _TOOL_CALLS_ATTR_RE.match(key):
            normalized = _json_list(value)
            if normalized:
                attrs[key] = normalized
                tool_call_roots.add(key)

    for root in tool_call_roots:
        for key in list(attrs):
            if key.startswith(f"{root}."):
                attrs.pop(key, None)

    completion_tool_calls = _json_list(attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR))
    if completion_tool_calls:
        final_answer_content = _extract_final_answer_content(completion_tool_calls)
        if final_answer_content not in {None, ""}:
            attrs[GEN_AI_COMPLETION_CONTENT_ATTR] = final_answer_content
            attrs.pop(GEN_AI_COMPLETION_TOOL_CALLS_ATTR, None)
        else:
            attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = json_string(
                completion_tool_calls
            )

    if attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR):
        attrs.setdefault(GEN_AI_COMPLETION_ROLE_ATTR, ASSISTANT_ROLE)
        attrs.setdefault(GEN_AI_COMPLETION_CONTENT_ATTR, "")
    elif attrs.get(GEN_AI_COMPLETION_CONTENT_ATTR) not in {None, ""}:
        attrs.setdefault(GEN_AI_COMPLETION_ROLE_ATTR, ASSISTANT_ROLE)

    for key, value in list(attrs.items()):
        if key.endswith(".content") and isinstance(value, str):
            attrs[key] = redact_text(value)


def _flatten_openinference_message_content(attrs: dict[str, Any]) -> None:
    for prefix in (
        OPENINFERENCE_INPUT_MESSAGES_ATTR,
        OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
    ):
        idx_to_texts: dict[int, list[str]] = {}
        for key, value in attrs.items():
            attr_prefix = f"{prefix}."
            if not key.startswith(attr_prefix):
                continue

            rest = key[len(attr_prefix) :]
            idx_part, _, field = rest.partition(".")
            if not idx_part.isdigit():
                continue
            if not field.endswith(f".{OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR}"):
                continue
            if f"{OPENINFERENCE_MESSAGE_CONTENTS_ATTR}." not in field:
                continue
            if value in {None, ""}:
                continue
            if isinstance(value, str):
                idx_to_texts.setdefault(int(idx_part), []).append(redact_text(value))

        for idx, texts in idx_to_texts.items():
            content_key = f"{prefix}.{idx}.{OPENINFERENCE_MESSAGE_CONTENT_ATTR}"
            attrs.setdefault(content_key, "\n".join(texts))


class SmolagentsSpanContentProcessor(SpanProcessor):
    """Flatten smolagents structured OI message content before translation."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not _is_smolagents_span(span=span, attrs=attrs):
            return

        _flatten_openinference_message_content(attrs)
        if attrs.get(OPENINFERENCE_SPAN_KIND_ATTR) == "TOOL":
            tool_name = attrs.get(OPENINFERENCE_TOOL_NAME_ATTR)
            if isinstance(tool_name, str) and tool_name:
                attrs[SMOLAGENTS_TOOL_NAME_HINT] = tool_name
        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class SmolagentsSpanContractProcessor(SpanProcessor):
    """Normalize smolagents spans after OpenInference translation.

    The shared OpenInference translator still emits legacy convenience aliases
    and in-process structured values for existing integrations. smolagents keeps
    the cleanup local so it exports backend-compatible canonical fields.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not _is_smolagents_span(span=span, attrs=attrs):
            return

        _normalize_structured_contract_attrs(attrs)

        self._normalize_smolagents_entities(span=span, attrs=attrs)

        for key in (
            TLSpanAttributes.TRACELOOP_ENTITY_INPUT,
            TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        ):
            if key in attrs:
                bounded = _bounded_entity_value(attrs[key])
                if bounded is not None:
                    attrs[key] = bounded

        for alias_attr in _OFF_CONTRACT_ALIAS_ATTRS:
            attrs.pop(alias_attr, None)

        attrs.pop(TLSpanAttributes.TRACELOOP_SPAN_KIND, None)
        attrs.pop(SMOLAGENTS_TOOL_NAME_HINT, None)

        span._attributes = attrs

    def _normalize_smolagents_entities(
        self,
        *,
        span: ReadableSpan,
        attrs: dict[str, Any],
    ) -> None:
        log_type = attrs.get(RESPAN_LOG_TYPE)

        span_name = getattr(span, "name", "")
        step_match = (
            _STEP_NAME.fullmatch(span_name) if isinstance(span_name, str) else None
        )
        if step_match:
            step_number = int(step_match.group(1))
            attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TASK
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = "smolagents.step"
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = "smolagents.step"
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
                {"step_number": step_number}
            )

        tool_name = attrs.get(SMOLAGENTS_TOOL_NAME_HINT)
        if isinstance(tool_name, str) and tool_name:
            attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TOOL
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_NAME] = tool_name
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_PATH] = tool_name
            raw_input = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_INPUT)
            arguments = _json_dict(raw_input) or {}
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(
                {"name": tool_name, "arguments": arguments}
            )
            raw_output = attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT)
            if raw_output is not None:
                parsed_output = _json_dict(raw_output)
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
                    parsed_output if parsed_output is not None else raw_output
                )

        if log_type == LOG_TYPE_AGENT:
            attrs.pop(TLSpanAttributes.LLM_REQUEST_MODEL, None)
            final_stream_output = _final_stream_output(
                attrs.get(TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT)
            )
            if final_stream_output is not None:
                attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_string(
                    final_stream_output
                )
            for key in (
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                GenAIAttributes.GEN_AI_USAGE_PROMPT_TOKENS,
                GenAIAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            ):
                attrs.pop(key, None)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
