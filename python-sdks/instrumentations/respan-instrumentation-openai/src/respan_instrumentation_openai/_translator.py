"""Translate current OpenAI SDK values into canonical span primitives."""

from __future__ import annotations

import json
from typing import Any

from respan_instrumentation_openai._constants import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    ROLE_KEY,
    USER_ROLE,
)
from respan_instrumentation_openai._serialization import (
    json_string,
    json_value,
    redact_text,
)


def safe_json(value: Any) -> str:
    return json_string(value)


def to_attr_value(value: Any) -> str:
    return redact_text(value) if isinstance(value, str) else safe_json(value)


def _dump(value: Any) -> Any:
    return json_value(value)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return getattr(obj, key, default)
    except Exception:  # noqa: BLE001 - defensive vendor object access
        return default


def _string(value: Any) -> str:
    if isinstance(value, str):
        return redact_text(value)
    if value is None:
        return ""
    return safe_json(value)


def request_model(request_kwargs: dict[str, Any]) -> str | None:
    model = request_kwargs.get("model")
    return model if isinstance(model, str) and model else None


def response_model(response: Any) -> str | None:
    model = _get(response, "model")
    return model if isinstance(model, str) and model else None


def response_id(response: Any) -> str | None:
    response_id_value = _get(response, "id")
    return (
        response_id_value
        if isinstance(response_id_value, str) and response_id_value
        else None
    )


def extract_usage(response: Any) -> dict[str, int]:
    usage = _get(response, "usage")
    if usage is None:
        return {}
    prompt = _get(usage, "prompt_tokens")
    completion = _get(usage, "completion_tokens")
    if prompt is None:
        prompt = _get(usage, "input_tokens")
    if completion is None:
        completion = _get(usage, "output_tokens")
    total = _get(usage, "total_tokens")
    result: dict[str, int] = {}
    for key, value in (
        ("prompt", prompt),
        ("completion", completion),
        ("total", total),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    return result


def _normalize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            return safe_json(json.loads(arguments))
        except (TypeError, ValueError):
            return safe_json(arguments)
    return safe_json(arguments if arguments is not None else {})


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    function = _get(tool_call, "function")
    name = _get(function, "name") or _get(tool_call, "name")
    if not isinstance(name, str) or not name:
        return None
    arguments = _get(function, "arguments")
    if arguments is None:
        arguments = _get(tool_call, "arguments")
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": _normalize_arguments(arguments)},
    }
    call_id = _get(tool_call, "call_id") or _get(tool_call, "id")
    if isinstance(call_id, str) and call_id:
        normalized["id"] = call_id
    return normalized


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    if not isinstance(tool_calls, list | tuple):
        tool_calls = [tool_calls]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool_call in tool_calls[:50]:
        normalized = _normalize_tool_call(tool_call)
        if normalized is None:
            continue
        signature = safe_json(normalized)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(normalized)
    return result


def normalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list | tuple):
        messages = [messages]
    normalized: list[dict[str, Any]] = []
    for message in messages[:50]:
        role = _get(message, ROLE_KEY) or USER_ROLE
        entry: dict[str, Any] = {
            ROLE_KEY: role if isinstance(role, str) else USER_ROLE,
            CONTENT_KEY: _dump(_get(message, CONTENT_KEY)),
        }
        tool_calls = _normalize_tool_calls(_get(message, "tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
        tool_call_id = _get(message, "tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            entry["tool_call_id"] = tool_call_id
        name = _get(message, "name")
        if isinstance(name, str) and name:
            entry["name"] = name
        normalized.append(entry)
    return normalized


def managed_prompt_input(extra_body: Any) -> dict[str, Any] | None:
    """Return a visible input for Respan managed-prompt-shaped requests."""
    prompt = _get(extra_body, "prompt")
    if prompt is None:
        return None
    return {"prompt": _dump(prompt)}


def normalize_responses_input(
    value: Any, instructions: Any = None
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions:
        normalized.append({ROLE_KEY: "system", CONTENT_KEY: instructions})
    if value is None:
        return normalized
    if isinstance(value, str):
        return [*normalized, {ROLE_KEY: USER_ROLE, CONTENT_KEY: value}]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list | tuple):
        value = [value]
    for item in value[:50]:
        item_type = _get(item, "type")
        if item_type == "function_call":
            call = _normalize_tool_call(item)
            if call:
                normalized.append(
                    {ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: None, "tool_calls": [call]}
                )
            continue
        if item_type == "function_call_output":
            normalized.append(
                {
                    ROLE_KEY: "tool",
                    CONTENT_KEY: _dump(_get(item, "output")),
                    "tool_call_id": _get(item, "call_id"),
                }
            )
            continue
        normalized.append(
            {
                ROLE_KEY: _get(item, ROLE_KEY) or USER_ROLE,
                CONTENT_KEY: _dump(_get(item, CONTENT_KEY)),
            }
        )
    return normalized


def format_input_messages(messages: list[dict[str, Any]]) -> str:
    return safe_json(messages)


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices") or []
    return choices[0] if isinstance(choices, list | tuple) and choices else None


def chat_output_message(response: Any) -> dict[str, Any]:
    choice = _first_choice(response)
    message = _get(choice, "message") if choice is not None else None
    result: dict[str, Any] = {
        ROLE_KEY: _get(message, ROLE_KEY) or ASSISTANT_ROLE,
        CONTENT_KEY: _dump(_get(message, CONTENT_KEY)),
    }
    tool_calls = _normalize_tool_calls(_get(message, "tool_calls"))
    if tool_calls:
        result["tool_calls"] = tool_calls
    parsed = _get(message, "parsed")
    if parsed is not None:
        result["parsed"] = _dump(parsed)
    return result


def format_chat_output(response: Any) -> str:
    return _string(chat_output_message(response).get(CONTENT_KEY))


def extract_chat_tool_calls(response: Any) -> list[dict[str, Any]]:
    return chat_output_message(response).get("tool_calls", [])


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []
    if not isinstance(tools, list | tuple):
        tools = [tools]
    normalized: list[dict[str, Any]] = []
    for tool in tools[:50]:
        dumped = _dump(tool)
        if not isinstance(dumped, dict):
            continue
        function = dumped.get("function")
        if isinstance(function, dict) and function.get("name"):
            normalized.append(
                {"type": dumped.get("type", "function"), "function": function}
            )
            continue
        name = dumped.get("name")
        if name:
            normalized_function = {"name": name}
            for key in ("description", "parameters", "strict"):
                if key in dumped:
                    normalized_function[key] = dumped[key]
            normalized.append({"type": "function", "function": normalized_function})
    return normalized


def normalize_text_prompts(prompt: Any) -> list[dict[str, Any]]:
    if prompt is None:
        return []
    if isinstance(prompt, list | tuple):
        return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _dump(item)} for item in prompt[:50]]
    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _dump(prompt)}]


def format_completion_output(response: Any) -> str:
    choice = _first_choice(response)
    return _string(_get(choice, "text") if choice is not None else None)


def _responses_output_items(response: Any) -> list[Any]:
    output = _get(response, "output") or []
    return list(output[:50]) if isinstance(output, list | tuple) else []


def format_responses_output(response: Any) -> str:
    output_text = _get(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    parts: list[str] = []
    for item in _responses_output_items(response):
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content") or []:
            text = _get(content, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def responses_output_payload(response: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"output_text": format_responses_output(response)}
    output_parsed = _get(response, "output_parsed")
    if output_parsed is not None:
        payload["output_parsed"] = _dump(output_parsed)
    tool_calls = extract_responses_tool_calls(response)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def extract_responses_tool_calls(response: Any) -> list[dict[str, Any]]:
    return _normalize_tool_calls(
        [
            item
            for item in _responses_output_items(response)
            if _get(item, "type") == "function_call"
        ]
    )


def normalize_embedding_inputs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [_dump(item) for item in value[:50]]
    return [_dump(value)]


def embedding_payload(response: Any) -> list[Any]:
    data = _get(response, "data") or []
    if not isinstance(data, list | tuple):
        return []
    return [_dump(_get(item, "embedding")) for item in data[:50]]


def embedding_summary(response: Any) -> dict[str, Any]:
    vectors = embedding_payload(response)
    return {
        "vector_count": len(vectors),
        "dimension": len(vectors[0]) if vectors and isinstance(vectors[0], list) else 0,
    }
