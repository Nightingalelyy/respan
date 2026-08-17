"""Translate Ollama SDK payloads into Respan span fields."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from respan_instrumentation_ollama._constants import (
    ARGUMENTS_KEY,
    CONTENT_KEY,
    DESCRIPTION_KEY,
    EMBEDDING_KEY,
    EMBEDDINGS_KEY,
    EVAL_COUNT_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    INPUT_KEY,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_KEY,
    NAME_KEY,
    PARAMETERS_KEY,
    PROMPT_EVAL_COUNT_KEY,
    RESPONSE_KEY,
    ROLE_KEY,
    SYSTEM_KEY,
    TOOL_CALLS_KEY,
    TOOL_NAME_KEY,
    TOOLS_KEY,
    TYPE_KEY,
    USER_ROLE,
)

_MAX_DEPTH = 6
_MAX_ITEMS = 50
_MAX_STRING_LENGTH = 8_000
_MAX_JSON_LENGTH = 16_000
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "set-cookie",
    "token",
}
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"(?i)\b(authorization|api[-_ ]?key|password|secret|token)\b"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+|\bbearer\s+[^\s,;]+"
)


def bounded_text(value: str, *, limit: int = _MAX_STRING_LENGTH) -> str:
    """Bound direct string attributes and redact obvious credential fragments."""
    redacted = _SENSITIVE_TEXT_PATTERN.sub(_REDACTED, value)
    if len(redacted) <= limit:
        return redacted
    marker = "...[truncated]"
    return f"{redacted[: max(limit - len(marker), 0)]}{marker}"


def _is_sensitive_key(value: Any) -> bool:
    key = value.strip().lower() if isinstance(value, str) else ""
    return bool(key) and (
        key in _SENSITIVE_KEYS
        or key.endswith(("_api_key", "_password", "_secret", "_token"))
    )


def _safe_key(value: Any) -> str:
    if isinstance(value, str):
        return bounded_text(value, limit=256)
    if isinstance(value, bool | int | float):
        return str(value)
    return f"<{type(value).__name__}>"


def _json_value(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Return a bounded JSON value without falling back to arbitrary reprs."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return bounded_text(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}:max-depth>"

    active = seen if seen is not None else set()
    object_id = id(value)
    if object_id in active:
        return "<cycle>"
    active.add(object_id)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        elif callable(getattr(value, "model_dump", None)):
            try:
                value = value.model_dump(exclude_none=True, by_alias=False)
            except TypeError:
                value = value.model_dump()
        elif callable(getattr(value, "to_dict", None)):
            value = value.to_dict()
        elif isinstance(value, Mapping):
            value = dict(value)
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            or isinstance(value, set | frozenset)
        ):
            value = list(value)
        elif callable(value):
            return {
                NAME_KEY: bounded_text(
                    getattr(value, "__name__", value.__class__.__name__),
                    limit=256,
                )
            }
        elif hasattr(value, "__dict__"):
            value = {
                key: item
                for key, item in vars(value).items()
                if isinstance(key, str) and not key.startswith("_")
            }
        else:
            return f"<{type(value).__name__}>"

        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in list(value.items())[:_MAX_ITEMS]:
                safe_key = _safe_key(key)
                result[safe_key] = (
                    _REDACTED
                    if _is_sensitive_key(key)
                    else _json_value(item, depth=depth + 1, seen=active)
                )
            if len(value) > _MAX_ITEMS:
                result["_respan_truncated_items"] = len(value) - _MAX_ITEMS
            return result
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            result = [
                _json_value(item, depth=depth + 1, seen=active)
                for item in list(value)[:_MAX_ITEMS]
            ]
            if len(value) > _MAX_ITEMS:
                result.append({"_respan_truncated_items": len(value) - _MAX_ITEMS})
            return result
        return value
    except Exception:  # noqa: BLE001 - vendor values must not break app calls
        return f"<{type(value).__name__}:unserializable>"
    finally:
        active.discard(object_id)


def safe_json(value: Any) -> str:
    """JSON-encode a bounded, redacted value without arbitrary repr fallback."""
    encoded = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
    if len(encoded) <= _MAX_JSON_LENGTH:
        return encoded

    # JSON escaping can expand a preview (quotes, backslashes, and control
    # characters), so choose the largest prefix whose *encoded* wrapper still
    # respects the configured limit.
    low = 0
    high = min(len(encoded), _MAX_JSON_LENGTH)
    bounded = json.dumps(
        {"preview": "", "truncated": True},
        ensure_ascii=False,
        sort_keys=True,
    )
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {"preview": encoded[:midpoint], "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(candidate) <= _MAX_JSON_LENGTH:
            bounded = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return bounded


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return bounded_text(value)
    return safe_json(value=value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except Exception:  # noqa: BLE001 - fall back to attribute access
            return getattr(value, name, default)
    return getattr(value, name, default)


class StreamResponseAccumulator:
    """Retain only bounded output plus terminal fields needed for span emission."""

    def __init__(self) -> None:
        self._message_seen = False
        self._message_role = "assistant"
        self._message_content = ""
        self._message_content_truncated = False
        self._response_seen = False
        self._response_content = ""
        self._response_content_truncated = False
        self._tool_calls: list[dict[str, Any]] = []
        self._tool_call_signatures: set[str] = set()
        self._model: Any = None
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None

    @staticmethod
    def _append_content(
        current: str, value: Any, *, truncated: bool
    ) -> tuple[str, bool]:
        if truncated or value is None or (isinstance(value, str) and not value):
            return current, truncated
        next_part = _stringify_content(value)
        combined = current + next_part
        if len(combined) <= _MAX_STRING_LENGTH:
            return combined, False
        marker = "...[truncated]"
        return f"{combined[: _MAX_STRING_LENGTH - len(marker)]}{marker}", True

    def append(self, chunk: Any) -> None:
        model = _field(chunk, MODEL_KEY)
        if model is not None:
            self._model = _json_value(model)

        prompt_tokens = _field(chunk, PROMPT_EVAL_COUNT_KEY)
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            self._prompt_tokens = prompt_tokens
        completion_tokens = _field(chunk, EVAL_COUNT_KEY)
        if isinstance(completion_tokens, int) and not isinstance(
            completion_tokens, bool
        ):
            self._completion_tokens = completion_tokens

        message = _field(chunk, MESSAGE_KEY)
        if message is not None:
            self._message_seen = True
            role = _field(message, ROLE_KEY)
            if role:
                self._message_role = to_json_attr(role)
            self._message_content, self._message_content_truncated = (
                self._append_content(
                    self._message_content,
                    _field(message, CONTENT_KEY),
                    truncated=self._message_content_truncated,
                )
            )
            for tool_call in normalize_tool_calls(_field(message, TOOL_CALLS_KEY)):
                if len(self._tool_calls) >= _MAX_ITEMS:
                    break
                signature = safe_json(tool_call)
                if signature not in self._tool_call_signatures:
                    self._tool_call_signatures.add(signature)
                    self._tool_calls.append(tool_call)

        response = _field(chunk, RESPONSE_KEY)
        if response is not None:
            self._response_seen = True
            self._response_content, self._response_content_truncated = (
                self._append_content(
                    self._response_content,
                    response,
                    truncated=self._response_content_truncated,
                )
            )

    def response(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self._model is not None:
            result[MODEL_KEY] = _json_value(self._model)
        if self._message_seen:
            message: dict[str, Any] = {
                ROLE_KEY: self._message_role,
                CONTENT_KEY: self._message_content,
            }
            if self._tool_calls:
                message[TOOL_CALLS_KEY] = list(self._tool_calls)
            result[MESSAGE_KEY] = message
        if self._response_seen:
            result[RESPONSE_KEY] = self._response_content
        if self._prompt_tokens is not None:
            result[PROMPT_EVAL_COUNT_KEY] = self._prompt_tokens
        if self._completion_tokens is not None:
            result[EVAL_COUNT_KEY] = self._completion_tokens
        return result


def _dump_value(value: Any) -> Any:
    return _json_value(value)


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return bounded_text(value)
    return safe_json(value=value)


def normalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if not isinstance(messages, (list, tuple)):
        messages = [messages]

    normalized_messages: list[dict[str, Any]] = []
    for message in messages[:_MAX_ITEMS]:
        role = _field(message, ROLE_KEY, USER_ROLE) or USER_ROLE
        normalized: dict[str, Any] = {ROLE_KEY: to_json_attr(role)}

        content = _field(message, CONTENT_KEY)
        if content is not None:
            normalized[CONTENT_KEY] = _stringify_content(content)

        tool_name = _field(message, TOOL_NAME_KEY)
        if tool_name is not None:
            normalized[TOOL_NAME_KEY] = to_json_attr(tool_name)

        tool_calls = normalize_tool_calls(_field(message, TOOL_CALLS_KEY))
        if tool_calls:
            normalized[TOOL_CALLS_KEY] = tool_calls

        normalized_messages.append(normalized)
    return normalized_messages


def normalize_generate_messages(
    *, prompt: Any, system: Any = None
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system not in {None, ""}:
        messages.append({ROLE_KEY: SYSTEM_KEY, CONTENT_KEY: _stringify_content(system)})
    if prompt not in {None, ""}:
        messages.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: _stringify_content(prompt)})
    return messages


def format_chat_input(*, messages: Any, tools: Any = None) -> str:
    payload: dict[str, Any] = {MESSAGES_KEY: normalize_chat_messages(messages)}
    normalized_tools = normalize_tools(tools)
    if normalized_tools:
        payload[TOOLS_KEY] = normalized_tools
    return safe_json(payload)


def format_generate_input(*, prompt: Any, system: Any = None) -> str:
    return safe_json(normalize_generate_messages(prompt=prompt, system=system))


def format_embedding_input(*, prompt: Any = None, input_value: Any = None) -> str:
    value = input_value if input_value is not None else prompt
    return safe_json({INPUT_KEY: _dump_value(value)})


def _iter_responses(response_or_chunks: Any) -> Iterable[Any]:
    if response_or_chunks is None:
        return ()
    if isinstance(response_or_chunks, list):
        return (chunk for chunk in response_or_chunks if chunk is not None)
    return (response_or_chunks,)


def _last_response(response_or_chunks: Any) -> Any:
    if isinstance(response_or_chunks, list):
        for chunk in reversed(response_or_chunks):
            if chunk is not None:
                return chunk
        return None
    return response_or_chunks


def format_chat_output(response_or_chunks: Any) -> str:
    parts: list[str] = []
    for response in _iter_responses(response_or_chunks):
        message = _field(response, MESSAGE_KEY)
        content = _field(message, CONTENT_KEY)
        if content:
            parts.append(_stringify_content(content))
    return bounded_text("".join(parts))


def format_generate_output(response_or_chunks: Any) -> str:
    parts: list[str] = []
    for response in _iter_responses(response_or_chunks):
        content = _field(response, RESPONSE_KEY)
        if content:
            parts.append(_stringify_content(content))
    return bounded_text("".join(parts))


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    if not isinstance(tool_calls, (list, tuple)):
        tool_calls = [tool_calls]

    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls[:_MAX_ITEMS]:
        function = _field(tool_call, FUNCTION_KEY)
        name = _field(function, NAME_KEY) or _field(tool_call, NAME_KEY)
        if not name:
            continue

        normalized_function: dict[str, Any] = {NAME_KEY: to_json_attr(name)}
        arguments = _field(function, ARGUMENTS_KEY)
        if arguments is None:
            arguments = _field(tool_call, ARGUMENTS_KEY)
        if arguments is not None:
            normalized_function[ARGUMENTS_KEY] = to_json_attr(_dump_value(arguments))

        normalized: dict[str, Any] = {
            TYPE_KEY: FUNCTION_TOOL_TYPE,
            FUNCTION_KEY: normalized_function,
        }
        call_id = _field(tool_call, ID_KEY)
        if call_id:
            normalized[ID_KEY] = to_json_attr(call_id)
        normalized_calls.append(normalized)
    return normalized_calls


def extract_chat_tool_calls(response_or_chunks: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in _iter_responses(response_or_chunks):
        message = _field(response, MESSAGE_KEY)
        for tool_call in normalize_tool_calls(_field(message, TOOL_CALLS_KEY)):
            signature = safe_json(tool_call)
            if signature in seen:
                continue
            seen.add(signature)
            tool_calls.append(tool_call)
    return tool_calls


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    if annotation in (str, inspect.Signature.empty):
        return {TYPE_KEY: "string"}
    if annotation is int:
        return {TYPE_KEY: "integer"}
    if annotation is float:
        return {TYPE_KEY: "number"}
    if annotation is bool:
        return {TYPE_KEY: "boolean"}
    if annotation is list:
        return {TYPE_KEY: "array"}
    if annotation is dict:
        return {TYPE_KEY: "object"}
    return {TYPE_KEY: "string"}


def _callable_tool_definition(tool: Any) -> dict[str, Any]:
    function: dict[str, Any] = {
        NAME_KEY: bounded_text(
            getattr(tool, "__name__", tool.__class__.__name__),
            limit=256,
        ),
    }
    doc = inspect.getdoc(tool)
    if doc:
        function[DESCRIPTION_KEY] = bounded_text(doc)

    try:
        signature = inspect.signature(tool)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, parameter in list(signature.parameters.items())[:_MAX_ITEMS]:
            if param_name in {"self", "cls"}:
                continue
            properties[param_name] = _annotation_to_json_schema(parameter.annotation)
            if parameter.default is inspect.Signature.empty:
                required.append(param_name)
        if properties:
            parameters: dict[str, Any] = {TYPE_KEY: "object", "properties": properties}
            if required:
                parameters["required"] = required
            function[PARAMETERS_KEY] = parameters

    return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function}


def _mapping_tool_definition(tool: Any) -> dict[str, Any] | None:
    dumped = _dump_value(tool)
    if not isinstance(dumped, dict):
        return None

    function = dumped.get(FUNCTION_KEY)
    if isinstance(function, dict) and function.get(NAME_KEY):
        normalized_function: dict[str, Any] = {NAME_KEY: function[NAME_KEY]}
        if function.get(DESCRIPTION_KEY):
            normalized_function[DESCRIPTION_KEY] = function[DESCRIPTION_KEY]
        if function.get(PARAMETERS_KEY) is not None:
            normalized_function[PARAMETERS_KEY] = function[PARAMETERS_KEY]
        return {
            TYPE_KEY: dumped.get(TYPE_KEY, FUNCTION_TOOL_TYPE),
            FUNCTION_KEY: normalized_function,
        }

    name = dumped.get(NAME_KEY)
    if name:
        normalized_function = {NAME_KEY: name}
        if dumped.get(DESCRIPTION_KEY):
            normalized_function[DESCRIPTION_KEY] = dumped[DESCRIPTION_KEY]
        if dumped.get(PARAMETERS_KEY) is not None:
            normalized_function[PARAMETERS_KEY] = dumped[PARAMETERS_KEY]
        return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: normalized_function}

    return None


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []
    if not isinstance(tools, (list, tuple)):
        tools = [tools]

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools[:_MAX_ITEMS]:
        if callable(tool):
            normalized_tools.append(_callable_tool_definition(tool))
            continue
        normalized = _mapping_tool_definition(tool)
        if normalized is not None:
            normalized_tools.append(normalized)
    return normalized_tools


def extract_usage(response_or_chunks: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    responses = list(_iter_responses(response_or_chunks))
    for response in reversed(responses):
        if PROMPT_EVAL_COUNT_KEY not in result:
            prompt_tokens = _field(response, PROMPT_EVAL_COUNT_KEY)
            if isinstance(prompt_tokens, int):
                result[PROMPT_EVAL_COUNT_KEY] = prompt_tokens
        if EVAL_COUNT_KEY not in result:
            completion_tokens = _field(response, EVAL_COUNT_KEY)
            if isinstance(completion_tokens, int):
                result[EVAL_COUNT_KEY] = completion_tokens
        if PROMPT_EVAL_COUNT_KEY in result and EVAL_COUNT_KEY in result:
            break
    return result


def extract_model(
    *, request_kwargs: dict[str, Any], response_or_chunks: Any = None
) -> str | None:
    model = request_kwargs.get(MODEL_KEY)
    if model:
        return to_json_attr(model)
    response = _last_response(response_or_chunks)
    response_model = _field(response, MODEL_KEY)
    if response_model:
        return to_json_attr(response_model)
    return None


def has_embedding_payload(response: Any) -> bool:
    return (
        _field(response, EMBEDDING_KEY) is not None
        or _field(response, EMBEDDINGS_KEY) is not None
    )
