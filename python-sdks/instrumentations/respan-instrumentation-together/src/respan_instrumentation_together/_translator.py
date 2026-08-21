"""Translate Together SDK requests and responses into Respan span fields."""

from __future__ import annotations

import json
import math
from itertools import islice
from typing import Any

from respan_instrumentation_together._constants import (
    ARGUMENTS_KEY,
    B64_JSON_KEY,
    CHOICES_KEY,
    CONTENT_KEY,
    DATA_KEY,
    DELTA_KEY,
    DESCRIPTION_KEY,
    DOCUMENT_KEY,
    DOCUMENTS_KEY,
    EMBEDDING_KEY,
    FINISH_REASON_KEY,
    FUNCTION_CALL_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    INDEX_KEY,
    INPUT_KEY,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_KEY,
    NAME_KEY,
    PARAMETERS_KEY,
    PROMPT_KEY,
    QUERY_KEY,
    RELEVANCE_SCORE_KEY,
    RESULTS_KEY,
    ROLE_KEY,
    TEXT_KEY,
    TOOL_CALL_ID_KEY,
    TOOL_CALLS_KEY,
    TYPE_KEY,
    URL_KEY,
    USAGE_KEY,
)
from respan_instrumentation_together._serialization import (
    json_dumps,
    safe_text,
    to_jsonable,
)


def safe_json(value: Any) -> str:
    """JSON-encode a value with the instrumentation capture policy."""
    return json_dumps(value)


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return safe_text(value)
    return safe_json(value=value)


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    try:
        return getattr(value, name, default)
    except BaseException:  # noqa: BLE001
        return default


def _is_omitted(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__.startswith("together.") and value_type.__name__ in {
        "Omit",
        "NotGiven",
    }


def dump_value(value: Any) -> Any:
    if _is_omitted(value):
        return None
    return to_jsonable(value)


def sequence_value(value: Any) -> list[Any]:
    if value is None or _is_omitted(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, bytes, dict)):
        return [value]
    try:
        return list(islice(iter(value), 50))
    except (TypeError, RuntimeError):
        return [value]


def normalize_message(message: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    role = field(message, ROLE_KEY)
    content = field(message, CONTENT_KEY)
    if role is not None:
        normalized[ROLE_KEY] = safe_text(role, max_bytes=256)
    if content is not None:
        normalized[CONTENT_KEY] = dump_value(content)

    tool_calls = normalize_tool_calls(field(message, TOOL_CALLS_KEY))
    if tool_calls:
        normalized[TOOL_CALLS_KEY] = tool_calls

    function_call = normalize_function_call(field(message, FUNCTION_CALL_KEY))
    if function_call:
        normalized[FUNCTION_CALL_KEY] = function_call

    tool_call_id = field(message, TOOL_CALL_ID_KEY)
    if tool_call_id:
        normalized[TOOL_CALL_ID_KEY] = safe_text(tool_call_id, max_bytes=256)

    return normalized


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    return [normalize_message(message) for message in sequence_value(messages)]


def normalize_function_call(function_call: Any) -> dict[str, Any]:
    if function_call is None or _is_omitted(function_call):
        return {}
    name = field(function_call, NAME_KEY)
    arguments = field(function_call, ARGUMENTS_KEY)
    result: dict[str, Any] = {}
    if name:
        result[NAME_KEY] = safe_text(name, max_bytes=256)
    if arguments is not None:
        result[ARGUMENTS_KEY] = (
            arguments if isinstance(arguments, str) else safe_json(arguments)
        )
    return result


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    function = field(tool_call, FUNCTION_KEY, {}) or {}
    normalized_function = normalize_function_call(function)
    if not normalized_function.get(NAME_KEY):
        return {}

    normalized: dict[str, Any] = {
        TYPE_KEY: field(tool_call, TYPE_KEY, FUNCTION_TOOL_TYPE) or FUNCTION_TOOL_TYPE,
        FUNCTION_KEY: normalized_function,
    }
    call_id = field(tool_call, ID_KEY)
    if call_id:
        normalized[ID_KEY] = safe_text(call_id, max_bytes=256)
    return normalized


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool_call in sequence_value(tool_calls):
        normalized = _normalize_tool_call(tool_call)
        if not normalized:
            continue
        signature = safe_json(normalized)
        if signature in seen:
            continue
        seen.add(signature)
        normalized_calls.append(normalized)
    return normalized_calls


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    normalized_tools: list[dict[str, Any]] = []
    for tool in sequence_value(tools):
        tool_type = field(tool, TYPE_KEY, FUNCTION_TOOL_TYPE) or FUNCTION_TOOL_TYPE
        function = field(tool, FUNCTION_KEY)
        if function is None:
            function = tool
        function_name = field(function, NAME_KEY)
        if not function_name:
            continue
        normalized_function: dict[str, Any] = {
            NAME_KEY: safe_text(function_name, max_bytes=256)
        }
        description = field(function, DESCRIPTION_KEY)
        if description is not None:
            normalized_function[DESCRIPTION_KEY] = safe_text(description)
        parameters = field(function, PARAMETERS_KEY)
        if parameters is not None:
            normalized_function[PARAMETERS_KEY] = dump_value(parameters)
        normalized_tools.append(
            {
                TYPE_KEY: safe_text(tool_type, max_bytes=64),
                FUNCTION_KEY: normalized_function,
            }
        )
    return normalized_tools


def chat_input_messages(request_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    return normalize_messages(request_kwargs.get(MESSAGES_KEY))


def chat_input(request_kwargs: dict[str, Any]) -> str:
    return safe_json(chat_input_messages(request_kwargs))


def text_completion_input(request_kwargs: dict[str, Any]) -> str:
    prompt = request_kwargs.get(PROMPT_KEY)
    return prompt if isinstance(prompt, str) else to_json_attr(prompt)


def _first_choice(response: Any) -> Any:
    choices = field(response, CHOICES_KEY, []) or []
    return choices[0] if choices else None


def _choice_message(choice: Any) -> Any:
    if choice is None:
        return None
    return field(choice, MESSAGE_KEY)


def _choice_text(choice: Any) -> str:
    text = field(choice, TEXT_KEY)
    if isinstance(text, str):
        return text
    message = _choice_message(choice)
    content = field(message, CONTENT_KEY)
    if isinstance(content, str):
        return content
    return ""


def chat_output(response_or_chunks: Any) -> str:
    if isinstance(response_or_chunks, list):
        parts: list[str] = []
        for chunk in response_or_chunks:
            for choice in field(chunk, CHOICES_KEY, []) or []:
                delta = field(choice, DELTA_KEY)
                content = field(delta, CONTENT_KEY)
                if isinstance(content, str):
                    parts.append(content)
        return "".join(parts)
    return _choice_text(_first_choice(response_or_chunks))


def text_completion_output(response_or_chunks: Any) -> str:
    if isinstance(response_or_chunks, list):
        parts: list[str] = []
        for chunk in response_or_chunks:
            for choice in field(chunk, CHOICES_KEY, []) or []:
                text = field(choice, TEXT_KEY)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                delta = field(choice, DELTA_KEY)
                content = field(delta, CONTENT_KEY)
                if isinstance(content, str):
                    parts.append(content)
            token = field(chunk, "token")
            token_text = field(token, TEXT_KEY)
            if isinstance(token_text, str) and not parts:
                parts.append(token_text)
        return "".join(parts)
    return _choice_text(_first_choice(response_or_chunks))


def extract_tool_calls(response_or_chunks: Any) -> list[dict[str, Any]]:
    raw_calls: list[Any] = []
    chunks = (
        response_or_chunks
        if isinstance(response_or_chunks, list)
        else [response_or_chunks]
    )
    for response in chunks:
        choice = _first_choice(response)
        message = _choice_message(choice)
        raw_calls.extend(sequence_value(field(message, TOOL_CALLS_KEY)))
        function_call = field(message, FUNCTION_CALL_KEY)
        if function_call is not None:
            raw_calls.append(
                {
                    TYPE_KEY: FUNCTION_TOOL_TYPE,
                    FUNCTION_KEY: function_call,
                }
            )
        for chunk_choice in field(response, CHOICES_KEY, []) or []:
            delta = field(chunk_choice, DELTA_KEY)
            raw_calls.extend(sequence_value(field(delta, TOOL_CALLS_KEY)))
            delta_function_call = field(delta, FUNCTION_CALL_KEY)
            if delta_function_call is not None:
                raw_calls.append(
                    {
                        TYPE_KEY: FUNCTION_TOOL_TYPE,
                        FUNCTION_KEY: delta_function_call,
                    }
                )
    return normalize_tool_calls(raw_calls)


def finish_reason(response_or_chunks: Any) -> str | None:
    chunks = (
        response_or_chunks
        if isinstance(response_or_chunks, list)
        else [response_or_chunks]
    )
    for response in reversed(chunks):
        for choice in reversed(field(response, CHOICES_KEY, []) or []):
            reason = field(choice, FINISH_REASON_KEY)
            if isinstance(reason, str):
                return reason
    return None


def extract_usage(response_or_chunks: Any) -> dict[str, int]:
    chunks = (
        response_or_chunks
        if isinstance(response_or_chunks, list)
        else [response_or_chunks]
    )
    for response in reversed(chunks):
        usage = field(response, USAGE_KEY)
        if usage is None:
            continue
        result: dict[str, int] = {}
        for source_name, target_name in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            token_count = field(usage, source_name)
            if isinstance(token_count, int):
                result[target_name] = token_count
        if result:
            return result
    return {}


def embedding_input(request_kwargs: dict[str, Any]) -> str:
    return to_json_attr(request_kwargs.get(INPUT_KEY))


def embedding_output(response: Any) -> str:
    """Retain complete numeric vectors as required by the span contract."""
    entries: list[dict[str, Any]] = []
    data = field(response, DATA_KEY, [])
    try:
        iterator = iter(data)
    except TypeError:
        iterator = iter(())
    for item in iterator:
        raw_embedding = field(item, EMBEDDING_KEY)
        if not isinstance(raw_embedding, (list, tuple)):
            continue
        vector: list[int | float] = []
        valid = True
        for value in raw_embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                valid = False
                break
            if isinstance(value, float) and not math.isfinite(value):
                valid = False
                break
            vector.append(value)
        if not valid:
            continue
        entry: dict[str, Any] = {EMBEDDING_KEY: vector}
        index = field(item, INDEX_KEY)
        if isinstance(index, int) and not isinstance(index, bool):
            entry[INDEX_KEY] = index
        entries.append(entry)
    return json.dumps(
        {DATA_KEY: entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def rerank_input(request_kwargs: dict[str, Any]) -> str:
    return safe_json(
        {
            QUERY_KEY: dump_value(request_kwargs.get(QUERY_KEY)),
            DOCUMENTS_KEY: dump_value(request_kwargs.get(DOCUMENTS_KEY)),
        }
    )


def rerank_output(response: Any) -> str:
    results: list[dict[str, Any]] = []
    for result in field(response, RESULTS_KEY, []) or []:
        normalized: dict[str, Any] = {}
        index = field(result, INDEX_KEY)
        if index is not None:
            normalized[INDEX_KEY] = index
        score = field(result, RELEVANCE_SCORE_KEY)
        if score is not None:
            normalized[RELEVANCE_SCORE_KEY] = score
        document = field(result, DOCUMENT_KEY)
        document_text = field(document, TEXT_KEY)
        if document_text is not None:
            normalized[DOCUMENT_KEY] = {TEXT_KEY: document_text}
        results.append(normalized)
    return safe_json(results)


def image_input(request_kwargs: dict[str, Any]) -> str:
    return safe_json(
        {
            PROMPT_KEY: dump_value(request_kwargs.get(PROMPT_KEY)),
            "image_url": dump_value(request_kwargs.get("image_url")),
            "reference_images": dump_value(request_kwargs.get("reference_images")),
        }
    )


def image_output(response: Any) -> str:
    images: list[dict[str, Any]] = []
    for item in field(response, DATA_KEY, []) or []:
        image_type = field(item, TYPE_KEY)
        normalized: dict[str, Any] = {
            INDEX_KEY: field(item, INDEX_KEY),
            TYPE_KEY: image_type,
        }
        if image_type == URL_KEY:
            normalized[URL_KEY] = field(item, URL_KEY)
        elif image_type == B64_JSON_KEY:
            value = field(item, B64_JSON_KEY)
            normalized[B64_JSON_KEY] = {
                "present": bool(value),
                "length": len(value) if isinstance(value, str) else 0,
            }
        images.append(
            {key: value for key, value in normalized.items() if value is not None}
        )
    return safe_json({"image_count": len(images), "images": images})


def request_model(request_kwargs: dict[str, Any], response: Any = None) -> str | None:
    model = request_kwargs.get(MODEL_KEY)
    if isinstance(model, str) and model:
        return model
    response_model = field(response, MODEL_KEY)
    return safe_text(response_model, max_bytes=512) if response_model else None
