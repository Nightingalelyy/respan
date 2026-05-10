"""Emit Agno runs as Respan-compatible OTEL spans."""

import json
import logging
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues

from respan_instrumentation_agno._constants import (
    AGENT_ID_KEY,
    AGENT_NAME_KEY,
    AGNO_AGENT_ID_ATTR,
    AGNO_AGENT_NAME_ATTR,
    AGNO_EVENT_NAME_ATTR,
    AGNO_EVENT_SPAN_NAME,
    AGNO_INSTRUMENTATION_NAME,
    AGNO_MODEL_REQUEST_SPAN_NAME,
    AGNO_RUN_ID_ATTR,
    AGNO_SESSION_ID_ATTR,
    AGNO_STATUS_ATTR,
    AGNO_TARGET_AGENT,
    AGNO_TEAM_ID_ATTR,
    AGNO_TEAM_NAME_ATTR,
    AGNO_TOOL_CALL_ID_ATTR,
    AGNO_TOOL_SPAN_NAME,
    AGNO_TOOL_NAME_ATTR,
    AGNO_USER_ID_ATTR,
    ARGUMENTS_KEY,
    ASSISTANT_ROLE,
    CACHE_READ_TOKENS_KEY,
    CANCELLED_STATUS,
    CHAT_SPAN_SEED_PART,
    COMPLETED_EVENT_SUFFIX,
    COMPLETED_STATUS,
    CONTENT_KEY,
    CONTENT_EVENT_SUFFIX,
    DEFAULT_AGNO_EVENT_NAME,
    DEFAULT_AGENT_NAME,
    DEFAULT_EVENT_NAME,
    DEFAULT_TEAM_NAME,
    DEFAULT_TOOL_NAME,
    DESCRIPTION_KEY,
    ERROR_STATUS,
    EVENT_KEY,
    EVENT_SPAN_SEED_PART,
    FUNCTION_KEY,
    FUNCTION_TOOL_SCHEMA_KEYS,
    FUNCTION_TYPE,
    GEN_AI_COMPLETION_PREFIX,
    GEN_AI_PROMPT_PREFIX,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    ID_KEY,
    INPUT_KEY,
    INPUT_TOKENS_KEY,
    LLM_REQUEST_FUNCTIONS,
    LLM_USAGE_CACHE_READ_INPUT_TOKENS,
    LLM_USAGE_TOTAL_TOKENS,
    MESSAGES_KEY,
    METADATA_KEY,
    METRICS_KEY,
    MODEL_KEY,
    MODEL_PROVIDER_KEY,
    MODEL_REQUEST_ENTITY_NAME,
    NAME_KEY,
    OUTPUT_TOKENS_KEY,
    PARAMETERS_JSON_SCHEMA_KEY,
    PARAMETERS_KEY,
    PROVIDER_KEY,
    RESULT_KEY,
    ROLE_KEY,
    ROOT_OUTPUT_PAYLOAD_KEYS,
    ROOT_SPAN_SEED_PART,
    RUN_COMPLETED_EVENT_SUFFIX,
    RUN_ID_KEY,
    SESSION_ID_KEY,
    STATUS_KEY,
    TEAM_ID_KEY,
    TEAM_NAME_KEY,
    TOOL_ARGS_KEY,
    TOOL_CALL_ERROR_KEY,
    TOOL_CALL_ID_KEY,
    TOOL_CALLS_KEY,
    TOOL_NAME_KEY,
    TOOL_SPAN_SEED_PART,
    TOOLS_KEY,
    TOTAL_TOKENS_KEY,
    TRACELOOP_ENTITY_INPUT,
    TRACELOOP_ENTITY_NAME,
    TRACELOOP_ENTITY_OUTPUT,
    TRACELOOP_ENTITY_PATH,
    TRACELOOP_WORKFLOW_NAME,
    TYPE_KEY,
    USER_ID_KEY,
    USER_ROLE,
    VALUE_KEY,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_sdk.utils.data_processing.id_processing import (
    ensure_span_id,
    ensure_trace_id,
    format_span_id,
    format_trace_id,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        converted = model_dump()
        if isinstance(converted, Mapping):
            return dict(converted)

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return dict(value_dict)

    return {VALUE_KEY: value}


def _json_string(value: Any) -> str:
    try:
        serialized_value = serialize_value(value=value)
        return json.dumps(obj=serialized_value, default=str, separators=(",", ":"))
    except Exception:
        return json.dumps(obj=str(value), separators=(",", ":"))


def _attribute_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _json_string(value=value)


def _set_if_present(attributes: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        attributes[key] = value


def _status_value(output: Any) -> str | None:
    status = _object_value(value=output, key=STATUS_KEY)
    if status is None:
        return None
    return str(getattr(status, VALUE_KEY, status))


def _is_error_output(output: Any) -> bool:
    status_value = _status_value(output=output)
    if status_value is None:
        return False
    normalized_status = status_value.lower()
    return ERROR_STATUS in normalized_status or CANCELLED_STATUS in normalized_status


def _current_parent_ids() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()
    if not getattr(span_context, "is_valid", False):
        return None, None
    return (
        format_trace_id(span_context.trace_id),
        format_span_id(span_context.span_id),
    )


def _normalize_provider(provider: Any) -> str | None:
    if provider is None:
        return None
    provider_name = str(provider).strip().lower()
    if not provider_name:
        return None
    return provider_name.replace(" ", "_")


def _target_name(target: Any, output: Any, target_kind: str) -> str:
    if target_kind == AGNO_TARGET_AGENT:
        for key in (AGENT_NAME_KEY, NAME_KEY):
            value = _object_value(value=output, key=key) or _object_value(
                value=target,
                key=key,
            )
            if value:
                return str(value)
        return DEFAULT_AGENT_NAME

    for key in (TEAM_NAME_KEY, NAME_KEY):
        value = _object_value(value=output, key=key) or _object_value(
            value=target,
            key=key,
        )
        if value:
            return str(value)
    return DEFAULT_TEAM_NAME


def _target_id(target: Any, output: Any, target_kind: str) -> str | None:
    if target_kind == AGNO_TARGET_AGENT:
        return _object_value(value=output, key=AGENT_ID_KEY) or _object_value(
            value=target,
            key=ID_KEY,
        )
    return _object_value(value=output, key=TEAM_ID_KEY) or _object_value(
        value=target,
        key=ID_KEY,
    )


def _input_payload(input_value: Any, output: Any) -> Any:
    output_input = _object_value(value=output, key=INPUT_KEY)
    if output_input is not None:
        output_input_dict = _object_to_dict(value=output_input)
        if output_input_dict:
            return output_input_dict
    return input_value


def _output_payload(output: Any) -> Any:
    content = _object_value(value=output, key=CONTENT_KEY)
    if content is not None:
        return content
    output_dict = _object_to_dict(value=output)
    return {
        key: value
        for key, value in output_dict.items()
        if key in ROOT_OUTPUT_PAYLOAD_KEYS
    }


def _normalize_message(message: Any) -> dict[str, Any] | None:
    message_dict = _object_to_dict(value=message)
    role = message_dict.get(ROLE_KEY)
    content = message_dict.get(CONTENT_KEY)
    tool_calls = message_dict.get(TOOL_CALLS_KEY)
    tool_call_id = message_dict.get(TOOL_CALL_ID_KEY)

    if role is None and content is None and tool_calls is None:
        return None

    normalized: dict[str, Any] = {}
    if role is not None:
        normalized[ROLE_KEY] = str(role)
    if content is not None:
        normalized[CONTENT_KEY] = content
    if tool_calls is not None:
        normalized[TOOL_CALLS_KEY] = tool_calls
    if tool_call_id is not None:
        normalized[TOOL_CALL_ID_KEY] = str(tool_call_id)
    return normalized


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    messages: list[dict[str, Any]] = []
    for item in value:
        message = _normalize_message(message=item)
        if message is not None:
            messages.append(message)
    return messages


def _extract_prompt_messages(input_value: Any, output: Any) -> list[dict[str, Any]]:
    output_messages = _normalize_messages(
        value=_object_value(value=output, key=MESSAGES_KEY)
    )
    if output_messages:
        if output_messages[-1].get(ROLE_KEY) == ASSISTANT_ROLE:
            return output_messages[:-1] or output_messages
        return output_messages

    input_messages = _normalize_messages(value=input_value)
    if input_messages:
        return input_messages

    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: input_value}]


def _extract_completion_message(output: Any) -> dict[str, Any]:
    content = _object_value(value=output, key=CONTENT_KEY)
    output_messages = _normalize_messages(
        value=_object_value(value=output, key=MESSAGES_KEY)
    )
    if content is None and output_messages:
        last_message = output_messages[-1]
        if last_message.get(ROLE_KEY) == ASSISTANT_ROLE:
            content = last_message.get(CONTENT_KEY)
    return {ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: content or ""}


def _set_message_attributes(
    attributes: dict[str, Any],
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for message_index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        tool_calls = message.get(TOOL_CALLS_KEY)
        tool_call_id = message.get(TOOL_CALL_ID_KEY)

        if role is not None:
            attributes[f"{prefix}{message_index}.role"] = str(role)
        if content is not None:
            attributes[f"{prefix}{message_index}.content"] = _attribute_string(
                value=content,
            )
        if tool_calls is not None:
            attributes[f"{prefix}{message_index}.{TOOL_CALLS_KEY}"] = _json_string(
                value=tool_calls,
            )
        if tool_call_id is not None:
            attributes[f"{prefix}{message_index}.{TOOL_CALL_ID_KEY}"] = str(
                tool_call_id
            )


def _normalize_tool_definition(tool: Any) -> dict[str, Any] | None:
    tool_dict = _object_to_dict(value=tool)
    if tool_dict.get(TYPE_KEY) == FUNCTION_TYPE and isinstance(
        tool_dict.get(FUNCTION_KEY),
        Mapping,
    ):
        function_payload = dict(tool_dict[FUNCTION_KEY])
        name = function_payload.get(NAME_KEY)
        if name:
            return {
                TYPE_KEY: FUNCTION_TYPE,
                FUNCTION_KEY: {
                    key: value
                    for key, value in function_payload.items()
                    if key in FUNCTION_TOOL_SCHEMA_KEYS and value is not None
                },
            }

    name = tool_dict.get(NAME_KEY) or getattr(tool, "__name__", None)
    if not name:
        return None

    function_payload = {NAME_KEY: str(name)}
    description = tool_dict.get(DESCRIPTION_KEY) or getattr(tool, "__doc__", None)
    if description:
        function_payload[DESCRIPTION_KEY] = str(description).strip()
    parameters = tool_dict.get(PARAMETERS_KEY) or tool_dict.get(
        PARAMETERS_JSON_SCHEMA_KEY
    )
    if parameters is not None:
        function_payload[PARAMETERS_KEY] = parameters
    return {TYPE_KEY: FUNCTION_TYPE, FUNCTION_KEY: function_payload}


def _extract_tool_definitions(target: Any) -> list[dict[str, Any]]:
    tools = _object_value(value=target, key=TOOLS_KEY)
    if not isinstance(tools, list):
        return []

    tool_definitions: list[dict[str, Any]] = []
    for tool in tools:
        tool_definition = _normalize_tool_definition(tool=tool)
        if tool_definition is not None:
            tool_definitions.append(tool_definition)
    return tool_definitions


def _extract_tool_executions(output: Any) -> list[Any]:
    tools = _object_value(value=output, key=TOOLS_KEY)
    if isinstance(tools, list):
        return tools
    return []


def _tool_call_payload(tool_execution: Any) -> dict[str, Any] | None:
    tool_name = _object_value(value=tool_execution, key=TOOL_NAME_KEY)
    tool_arguments = _object_value(value=tool_execution, key=TOOL_ARGS_KEY)
    tool_call_id = _object_value(value=tool_execution, key=TOOL_CALL_ID_KEY)
    if not tool_name:
        return None
    return {
        ID_KEY: str(tool_call_id or tool_name),
        TYPE_KEY: FUNCTION_TYPE,
        FUNCTION_KEY: {
            NAME_KEY: str(tool_name),
            ARGUMENTS_KEY: _json_string(value=tool_arguments or {}),
        },
    }


def _extract_tool_calls(tool_executions: list[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for tool_execution in tool_executions:
        tool_call = _tool_call_payload(tool_execution=tool_execution)
        if tool_call is not None:
            tool_calls.append(tool_call)
    return tool_calls


def _extract_usage(
    output: Any,
    events: list[Any],
) -> tuple[int | None, int | None, int | None, int | None]:
    metrics = _object_value(value=output, key=METRICS_KEY)
    prompt_tokens = _object_value(value=metrics, key=INPUT_TOKENS_KEY)
    completion_tokens = _object_value(value=metrics, key=OUTPUT_TOKENS_KEY)
    total_tokens = _object_value(value=metrics, key=TOTAL_TOKENS_KEY)
    cache_read_tokens = _object_value(value=metrics, key=CACHE_READ_TOKENS_KEY)

    for event in reversed(events):
        if prompt_tokens is None:
            prompt_tokens = _object_value(value=event, key=INPUT_TOKENS_KEY)
        if completion_tokens is None:
            completion_tokens = _object_value(value=event, key=OUTPUT_TOKENS_KEY)
        if total_tokens is None:
            total_tokens = _object_value(value=event, key=TOTAL_TOKENS_KEY)
        if cache_read_tokens is None:
            cache_read_tokens = _object_value(value=event, key=CACHE_READ_TOKENS_KEY)

    if total_tokens in (None, 0) and (
        isinstance(prompt_tokens, int) or isinstance(completion_tokens, int)
    ):
        total_tokens = (prompt_tokens if isinstance(prompt_tokens, int) else 0) + (
            completion_tokens if isinstance(completion_tokens, int) else 0
        )

    return (
        prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens if isinstance(completion_tokens, int) else None,
        total_tokens if isinstance(total_tokens, int) else None,
        cache_read_tokens if isinstance(cache_read_tokens, int) else None,
    )


def _completed_event(events: list[Any]) -> Any | None:
    for event in reversed(events):
        event_name = str(_object_value(value=event, key=EVENT_KEY, default=""))
        if event_name.endswith(COMPLETED_EVENT_SUFFIX) or event_name.endswith(
            RUN_COMPLETED_EVENT_SUFFIX
        ):
            return event
    return None


def _span_ids(
    output: Any,
    target_kind: str,
    started_at_ns: int,
) -> tuple[str, str, str | None]:
    parent_trace_id, parent_span_id = _current_parent_ids()
    run_id = _object_value(value=output, key=RUN_ID_KEY)
    trace_seed = str(
        run_id or f"{AGNO_INSTRUMENTATION_NAME}:{target_kind}:{started_at_ns}"
    )
    root_seed = f"{trace_seed}:{ROOT_SPAN_SEED_PART}"
    trace_id = parent_trace_id or format_trace_id(ensure_trace_id(val=trace_seed))
    root_span_id = format_span_id(ensure_span_id(val=root_seed))
    return trace_id, root_span_id, parent_span_id


def _root_attributes(
    target: Any,
    target_kind: str,
    input_value: Any,
    output: Any,
) -> dict[str, Any]:
    entity_name = _target_name(target=target, output=output, target_kind=target_kind)
    target_id = _target_id(target=target, output=output, target_kind=target_kind)
    log_type = LOG_TYPE_AGENT if target_kind == AGNO_TARGET_AGENT else LOG_TYPE_WORKFLOW

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        TRACELOOP_ENTITY_NAME: entity_name,
        TRACELOOP_ENTITY_PATH: "",
        TRACELOOP_WORKFLOW_NAME: entity_name,
        TRACELOOP_ENTITY_INPUT: _attribute_string(
            value=_input_payload(input_value=input_value, output=output),
        ),
        TRACELOOP_ENTITY_OUTPUT: _attribute_string(value=_output_payload(output=output)),
    }

    run_id = _object_value(value=output, key=RUN_ID_KEY)
    session_id = _object_value(value=output, key=SESSION_ID_KEY)
    user_id = _object_value(value=output, key=USER_ID_KEY)
    metadata = _object_value(value=output, key=METADATA_KEY)
    status = _status_value(output=output)

    _set_if_present(attributes=attributes, key=AGNO_RUN_ID_ATTR, value=run_id)
    _set_if_present(attributes=attributes, key=AGNO_SESSION_ID_ATTR, value=session_id)
    _set_if_present(attributes=attributes, key=AGNO_USER_ID_ATTR, value=user_id)
    _set_if_present(attributes=attributes, key=AGNO_STATUS_ATTR, value=status)
    if target_kind == AGNO_TARGET_AGENT:
        _set_if_present(attributes=attributes, key=AGNO_AGENT_ID_ATTR, value=target_id)
        attributes[AGNO_AGENT_NAME_ATTR] = entity_name
    else:
        _set_if_present(attributes=attributes, key=AGNO_TEAM_ID_ATTR, value=target_id)
        attributes[AGNO_TEAM_NAME_ATTR] = entity_name
    if metadata:
        attributes[RESPAN_METADATA] = _json_string(value=metadata)
    return attributes


def _chat_attributes(
    target: Any,
    input_value: Any,
    output: Any,
    events: list[Any],
) -> dict[str, Any]:
    entity_name = MODEL_REQUEST_ENTITY_NAME
    target_model = _object_value(value=target, key=MODEL_KEY)
    prompt_messages = _extract_prompt_messages(input_value=input_value, output=output)
    completion_message = _extract_completion_message(output=output)
    provider = _normalize_provider(
        provider=_object_value(value=output, key=MODEL_PROVIDER_KEY)
        or _object_value(value=target_model, key=PROVIDER_KEY),
    )
    model_name = _object_value(value=output, key=MODEL_KEY) or _object_value(
        value=target_model,
        key=ID_KEY,
    )
    prompt_tokens, completion_tokens, total_tokens, cache_read_tokens = _extract_usage(
        output=output,
        events=events,
    )

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        TRACELOOP_ENTITY_NAME: entity_name,
        TRACELOOP_ENTITY_PATH: entity_name,
        TRACELOOP_ENTITY_INPUT: _json_string(
            value=prompt_messages,
        ),
        TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=completion_message.get(CONTENT_KEY),
        )
        or "",
        LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
    }

    _set_if_present(attributes=attributes, key=GEN_AI_SYSTEM, value=provider)
    _set_if_present(attributes=attributes, key=LLM_REQUEST_MODEL, value=model_name)
    _set_if_present(
        attributes=attributes,
        key=GEN_AI_USAGE_INPUT_TOKENS,
        value=prompt_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=GEN_AI_USAGE_OUTPUT_TOKENS,
        value=completion_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=LLM_USAGE_PROMPT_TOKENS,
        value=prompt_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=LLM_USAGE_COMPLETION_TOKENS,
        value=completion_tokens,
    )
    _set_if_present(attributes=attributes, key=LLM_USAGE_TOTAL_TOKENS, value=total_tokens)
    _set_if_present(
        attributes=attributes,
        key=LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        value=cache_read_tokens,
    )

    _set_message_attributes(
        attributes=attributes,
        prefix=GEN_AI_PROMPT_PREFIX,
        messages=prompt_messages,
    )
    _set_message_attributes(
        attributes=attributes,
        prefix=GEN_AI_COMPLETION_PREFIX,
        messages=[completion_message],
    )

    tool_definitions = _extract_tool_definitions(target=target)
    if tool_definitions:
        attributes[LLM_REQUEST_FUNCTIONS] = _json_string(value=tool_definitions)

    tool_calls = _extract_tool_calls(
        tool_executions=_extract_tool_executions(output=output),
    )
    if tool_calls:
        attributes[f"{GEN_AI_COMPLETION_PREFIX}0.{TOOL_CALLS_KEY}"] = _json_string(
            value=tool_calls,
        )

    return attributes


def _tool_attributes(tool_execution: Any) -> dict[str, Any]:
    tool_name = (
        _object_value(value=tool_execution, key=TOOL_NAME_KEY) or DEFAULT_TOOL_NAME
    )
    tool_arguments = _object_value(value=tool_execution, key=TOOL_ARGS_KEY) or {}
    tool_result = _object_value(value=tool_execution, key=RESULT_KEY)
    tool_call_id = _object_value(value=tool_execution, key=TOOL_CALL_ID_KEY)

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
        TRACELOOP_ENTITY_NAME: str(tool_name),
        TRACELOOP_ENTITY_PATH: f"{DEFAULT_TOOL_NAME}.{tool_name}",
        TRACELOOP_ENTITY_INPUT: _json_string(
            value={NAME_KEY: tool_name, ARGUMENTS_KEY: tool_arguments},
        ),
        TRACELOOP_ENTITY_OUTPUT: _attribute_string(value=tool_result) or "",
        AGNO_TOOL_NAME_ATTR: str(tool_name),
    }
    _set_if_present(
        attributes=attributes,
        key=AGNO_TOOL_CALL_ID_ATTR,
        value=tool_call_id,
    )
    return attributes


def _event_attributes(event: Any) -> dict[str, Any]:
    event_name = _object_value(value=event, key=EVENT_KEY)
    entity_name = str(event_name or DEFAULT_AGNO_EVENT_NAME)
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_WORKFLOW,
        TRACELOOP_ENTITY_NAME: entity_name,
        TRACELOOP_ENTITY_PATH: f"{DEFAULT_EVENT_NAME}.{entity_name}",
        TRACELOOP_ENTITY_INPUT: _json_string(value=_object_to_dict(value=event)),
    }
    _set_if_present(attributes=attributes, key=AGNO_EVENT_NAME_ATTR, value=event_name)
    return attributes


def emit_agno_run(
    *,
    target: Any,
    target_kind: str,
    input_value: Any,
    output: Any | None,
    events: list[Any] | None,
    started_at_ns: int,
    ended_at_ns: int,
) -> None:
    """Emit root, chat, and tool spans for a completed Agno run."""
    safe_events = list(events or [])
    selected_output = output or _completed_event(events=safe_events)
    if selected_output is None:
        selected_output = {CONTENT_KEY: "", STATUS_KEY: COMPLETED_STATUS}

    trace_id, root_span_id, parent_span_id = _span_ids(
        output=selected_output,
        target_kind=target_kind,
        started_at_ns=started_at_ns,
    )
    root_status_code = 500 if _is_error_output(output=selected_output) else 200

    root_span = build_readable_span(
        name=f"{AGNO_INSTRUMENTATION_NAME}.{target_kind}",
        trace_id=trace_id,
        span_id=root_span_id,
        parent_id=parent_span_id,
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        attributes=_root_attributes(
            target=target,
            target_kind=target_kind,
            input_value=input_value,
            output=selected_output,
        ),
        status_code=root_status_code,
    )
    inject_span(span=root_span)

    chat_span_id = format_span_id(
        ensure_span_id(val=f"{root_span_id}:{CHAT_SPAN_SEED_PART}")
    )
    chat_span = build_readable_span(
        name=AGNO_MODEL_REQUEST_SPAN_NAME,
        trace_id=trace_id,
        span_id=chat_span_id,
        parent_id=root_span_id,
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        attributes=_chat_attributes(
            target=target,
            input_value=input_value,
            output=selected_output,
            events=safe_events,
        ),
        status_code=root_status_code,
    )
    inject_span(span=chat_span)

    for tool_index, tool_execution in enumerate(
        _extract_tool_executions(output=selected_output)
    ):
        tool_span_id = format_span_id(
            ensure_span_id(val=f"{root_span_id}:{TOOL_SPAN_SEED_PART}:{tool_index}"),
        )
        tool_status_code = (
            500 if _object_value(value=tool_execution, key=TOOL_CALL_ERROR_KEY) else 200
        )
        tool_span = build_readable_span(
            name=AGNO_TOOL_SPAN_NAME,
            trace_id=trace_id,
            span_id=tool_span_id,
            parent_id=root_span_id,
            start_time_ns=started_at_ns,
            end_time_ns=ended_at_ns,
            attributes=_tool_attributes(tool_execution=tool_execution),
            status_code=tool_status_code,
        )
        inject_span(span=tool_span)

    if output is None and safe_events:
        for event_index, event in enumerate(safe_events):
            event_name = str(
                _object_value(value=event, key=EVENT_KEY, default=DEFAULT_EVENT_NAME)
            )
            if event_name.endswith(COMPLETED_EVENT_SUFFIX) or event_name.endswith(
                CONTENT_EVENT_SUFFIX
            ):
                continue
            event_span = build_readable_span(
                name=AGNO_EVENT_SPAN_NAME,
                trace_id=trace_id,
                span_id=format_span_id(
                    ensure_span_id(
                        val=f"{root_span_id}:{EVENT_SPAN_SEED_PART}:{event_index}"
                    ),
                ),
                parent_id=root_span_id,
                start_time_ns=started_at_ns,
                end_time_ns=ended_at_ns,
                attributes=_event_attributes(event=event),
            )
            inject_span(span=event_span)


def emit_agno_error(
    *,
    target: Any,
    target_kind: str,
    input_value: Any,
    exception: Exception,
    started_at_ns: int,
    ended_at_ns: int,
) -> None:
    """Emit a failed Agno root span when the wrapped run raises."""
    output = {
        CONTENT_KEY: str(exception),
        STATUS_KEY: ERROR_STATUS,
        RUN_ID_KEY: None,
    }
    trace_id, root_span_id, parent_span_id = _span_ids(
        output=output,
        target_kind=target_kind,
        started_at_ns=started_at_ns,
    )
    attributes = _root_attributes(
        target=target,
        target_kind=target_kind,
        input_value=input_value,
        output=output,
    )
    error_span = build_readable_span(
        name=f"{AGNO_INSTRUMENTATION_NAME}.{target_kind}",
        trace_id=trace_id,
        span_id=root_span_id,
        parent_id=parent_span_id,
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        attributes=attributes,
        status_code=500,
        error_message=str(exception),
    )
    inject_span(span=error_span)
