"""Serialization and extraction helpers for LlamaIndex payloads."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from respan_instrumentation_llama_index._constants import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_USER,
)

_REACT_OBSERVATION_PREFIX = "Observation:"
_CONTEXT_PROMPT_PREFIX = "Context information is below."
_QUERY_MARKER = "\nQuery:"
_ANSWER_MARKER = "\nAnswer:"


def enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def safe_json(value: Any) -> str:
    return json.dumps(obj=to_jsonable(value), default=str)


def to_jsonable(value: Any) -> Any:
    value = enum_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        jsonable_dict = {
            str(enum_value(key)): to_jsonable(item_value)
            for key, item_value in value.items()
        }
        return normalize_message_dict(jsonable_dict)
    if isinstance(value, (list, tuple, set)):
        return normalize_message_sequence([to_jsonable(item) for item in value])
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        public_items = {
            key: item_value
            for key, item_value in vars(value).items()
            if not key.startswith("_")
        }
        if public_items:
            return to_jsonable(public_items)
    return str(value)


def get_model_name(model_dict: dict[str, Any] | None) -> str | None:
    if not model_dict:
        return None
    for key in ("model_name", "model", "model_id", "deployment_name", "name"):
        value = model_dict.get(key)
        if value:
            return str(value)
    return None


def get_model_system(model_dict: dict[str, Any] | None) -> str | None:
    if not model_dict:
        return None
    candidates = (
        model_dict.get("class_name"),
        model_dict.get("provider"),
        model_dict.get("model_provider"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        normalized = str(candidate).lower()
        if "openai" in normalized:
            return "openai"
        if "anthropic" in normalized or "claude" in normalized:
            return "anthropic"
        if "gemini" in normalized or "google" in normalized:
            return "google"
        if "bedrock" in normalized:
            return "bedrock"
        return normalized.replace(" ", "_")
    return None


def message_to_dict(message: Any) -> dict[str, Any]:
    role = enum_value(getattr(message, "role", None)) or MESSAGE_ROLE_USER
    content = getattr(message, "content", None)
    if content is None and hasattr(message, "blocks"):
        content = [to_jsonable(block) for block in getattr(message, "blocks", [])]

    result: dict[str, Any] = {
        "role": str(role),
        "content": to_jsonable(content),
    }
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        result["additional_kwargs"] = to_jsonable(additional_kwargs)
    return normalize_message_dict(result)


def chat_messages_to_dicts(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    return normalize_message_sequence(
        [message_to_dict(message) for message in messages]
    )


def normalize_message_sequence(messages: list[Any]) -> list[Any]:
    if not all(isinstance(message, dict) for message in messages):
        return messages

    result: list[dict[str, Any]] = []
    for message in messages:
        for candidate in split_generated_context_message(
            message=normalize_message_dict(message)
        ):
            result.append(
                normalize_react_observation_message(
                    message=candidate,
                    previous_messages=result,
                )
            )
    return result


def normalize_message_dict(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize LlamaIndex-generated messages before export."""
    role = message.get("role")
    if role != MESSAGE_ROLE_USER:
        return message

    text = _message_text(message.get("content"))
    if text is None:
        text = _message_text(message.get("blocks"))
    if text is None:
        return message

    if _is_llama_index_context_prompt(text=text):
        normalized = dict(message)
        normalized["role"] = MESSAGE_ROLE_SYSTEM
        return normalized
    return message


def normalize_react_observation_message(
    *,
    message: dict[str, Any],
    previous_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    if not _is_react_observation_message(message=message):
        return message
    if not _previous_message_is_react_action(previous_messages=previous_messages):
        return message

    normalized = dict(message)
    normalized["role"] = MESSAGE_ROLE_SYSTEM
    return normalized


def split_generated_context_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Separate LlamaIndex-generated retrieval context from the user query."""
    text = _message_text(message.get("content"))
    if text is None or not _is_llama_index_context_prompt(text=text):
        return [message]

    query = _extract_generated_query(text=text)
    if not query:
        return [message]

    context_message = dict(message)
    context_message["role"] = MESSAGE_ROLE_SYSTEM
    context_message["content"] = text.split(_QUERY_MARKER, maxsplit=1)[0].rstrip()
    user_message = {
        "role": MESSAGE_ROLE_USER,
        "content": query,
    }
    return [context_message, user_message]


def chat_response_to_message_dict(response: Any) -> dict[str, Any]:
    message = getattr(response, "message", None)
    if message is not None:
        return message_to_dict(message)
    content = getattr(response, "text", None)
    if content is None:
        content = getattr(response, "response", None)
    if content is None:
        content = str(response) if response is not None else ""
    return {"role": MESSAGE_ROLE_ASSISTANT, "content": to_jsonable(content)}


def completion_response_to_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    response_text = getattr(response, "response", None)
    if response_text is not None:
        return str(response_text)
    return str(response) if response is not None else ""


def extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage_candidates = [
        getattr(response, "raw", None),
        getattr(response, "additional_kwargs", None),
        response,
    ]
    for candidate in usage_candidates:
        usage = _find_usage_dict(candidate)
        if usage:
            prompt_tokens = _get_int(
                usage,
                "prompt_tokens",
                "input_tokens",
                "total_prompt_tokens",
            )
            completion_tokens = _get_int(
                usage,
                "completion_tokens",
                "output_tokens",
                "total_completion_tokens",
            )
            total_tokens = _get_int(usage, "total_tokens")
            if total_tokens is None and (
                prompt_tokens is not None or completion_tokens is not None
            ):
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            return prompt_tokens, completion_tokens, total_tokens
    return None, None, None


def _find_usage_dict(value: Any) -> dict[str, Any] | None:
    value = to_jsonable(value)
    if not isinstance(value, dict):
        return None
    direct_keys = {
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
    }
    if any(key in value for key in direct_keys):
        return value
    for key in ("usage", "token_usage", "usage_metadata"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return nested
    return None


def _message_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        segments: list[str] = []
        for item in value:
            if isinstance(item, str):
                segments.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    segments.append(text)
        return "\n".join(segments) if segments else None
    return None


def _is_react_observation_message(*, message: dict[str, Any]) -> bool:
    if message.get("role") != MESSAGE_ROLE_USER:
        return False

    text = _message_text(message.get("content"))
    if text is None:
        text = _message_text(message.get("blocks"))
    return bool(text and text.lstrip().startswith(_REACT_OBSERVATION_PREFIX))


def _previous_message_is_react_action(
    *,
    previous_messages: list[dict[str, Any]],
) -> bool:
    for previous_message in reversed(previous_messages):
        role = previous_message.get("role")
        if role == MESSAGE_ROLE_SYSTEM:
            continue
        if role != MESSAGE_ROLE_ASSISTANT:
            return False

        text = _message_text(previous_message.get("content"))
        if text is None:
            text = _message_text(previous_message.get("blocks"))
        return bool(text and "Action:" in text)
    return False


def _is_llama_index_context_prompt(*, text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(_CONTEXT_PROMPT_PREFIX) and _QUERY_MARKER in stripped


def _extract_generated_query(*, text: str) -> str | None:
    _, query_part = text.split(_QUERY_MARKER, maxsplit=1)
    if _ANSWER_MARKER in query_part:
        query_part = query_part.split(_ANSWER_MARKER, maxsplit=1)[0]
    query = query_part.strip()
    return query or None


def _get_int(value: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, float) and candidate.is_integer():
            return int(candidate)
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None
