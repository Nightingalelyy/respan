"""Serialization and extraction helpers for LlamaIndex payloads."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


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
        return {
            str(enum_value(key)): to_jsonable(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
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
    role = enum_value(getattr(message, "role", None)) or "user"
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
    return result


def chat_messages_to_dicts(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    return [message_to_dict(message) for message in messages]


def chat_response_to_message_dict(response: Any) -> dict[str, Any]:
    message = getattr(response, "message", None)
    if message is not None:
        return message_to_dict(message)
    content = getattr(response, "text", None)
    if content is None:
        content = getattr(response, "response", None)
    if content is None:
        content = str(response) if response is not None else ""
    return {"role": "assistant", "content": to_jsonable(content)}


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
