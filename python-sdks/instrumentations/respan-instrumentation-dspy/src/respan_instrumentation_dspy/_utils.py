"""Serialization and attribute helpers for DSPy instrumentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_dspy._constants import (
    ANTHROPIC_PROVIDER_PREFIX,
    ASSISTANT_ROLE,
    AZURE_PROVIDER_PREFIX,
    BEDROCK_PROVIDER_PREFIX,
    CHAT_MODEL_TYPE,
    COMPLETION_TOKENS_KEY,
    DSPY_PROVIDER_NAME,
    GEMINI_PROVIDER_PREFIX,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GOOGLE_PROVIDER_PREFIX,
    INPUT_TOKENS_KEY,
    OLLAMA_PROVIDER_PREFIX,
    OPENAI_PROVIDER_PREFIX,
    OUTPUT_TOKENS_KEY,
    PROMPT_TOKENS_KEY,
    RESPONSES_MODEL_TYPE,
    TEXT_MODEL_TYPE,
    TOTAL_TOKENS_KEY,
    USER_ROLE,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
)
from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """Serialize arbitrary DSPy payloads into OTEL-safe JSON strings."""
    try:
        return json.dumps(serialize_value(value=value), default=str)
    except Exception:
        return str(value)


def content_to_string(value: Any) -> str:
    """Convert a prompt/completion content value into a readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def output_to_plain_value(value: Any) -> Any:
    """Normalize DSPy outputs before serializing them on entity spans."""
    if value is None:
        return None

    for method_name in ("toDict", "to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                continue

    return serialize_value(value=value)


def output_to_json(value: Any) -> str:
    """Serialize DSPy outputs into the canonical entity output attribute."""
    return safe_json(value=output_to_plain_value(value=value))


def normalize_messages(prompt: Any, messages: Any) -> list[dict[str, Any]]:
    """Normalize DSPy LM prompt inputs into chat-style message dictionaries."""
    if isinstance(messages, list):
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, Mapping):
                role = message.get("role") or USER_ROLE
                content = message.get("content")
                normalized_messages.append(
                    {
                        "role": str(role),
                        "content": output_to_plain_value(value=content),
                    }
                )
            else:
                normalized_messages.append(
                    {
                        "role": USER_ROLE,
                        "content": output_to_plain_value(value=message),
                    }
                )
        return normalized_messages

    if prompt is not None:
        return [{"role": USER_ROLE, "content": output_to_plain_value(value=prompt)}]

    return []


def extract_first_completion(outputs: Any) -> str:
    """Extract the first assistant completion text from DSPy LM outputs."""
    if isinstance(outputs, list) and outputs:
        first_output = outputs[0]
    else:
        first_output = outputs

    if isinstance(first_output, Mapping):
        for key in ("content", "text", "answer", "output"):
            value = first_output.get(key)
            if value is not None:
                return content_to_string(value=value)
        return safe_json(value=first_output)

    return content_to_string(value=first_output)


def extract_provider_name(model_name: Any) -> str:
    """Infer the GenAI provider name from a DSPy/LiteLLM model string."""
    if not isinstance(model_name, str) or not model_name:
        return DSPY_PROVIDER_NAME

    provider_prefix = model_name.split("/", maxsplit=1)[0].lower()
    if provider_prefix in {
        OPENAI_PROVIDER_PREFIX,
        ANTHROPIC_PROVIDER_PREFIX,
        GOOGLE_PROVIDER_PREFIX,
        GEMINI_PROVIDER_PREFIX,
        BEDROCK_PROVIDER_PREFIX,
        AZURE_PROVIDER_PREFIX,
        OLLAMA_PROVIDER_PREFIX,
    }:
        if provider_prefix == GEMINI_PROVIDER_PREFIX:
            return GOOGLE_PROVIDER_PREFIX
        return provider_prefix

    return DSPY_PROVIDER_NAME


def request_type_from_model_type(model_type: Any) -> str:
    """Map DSPy model types to the Respan LLM request type value."""
    if model_type == TEXT_MODEL_TYPE:
        return LLMRequestTypeValues.COMPLETION.value
    if model_type in {CHAT_MODEL_TYPE, RESPONSES_MODEL_TYPE}:
        return LLMRequestTypeValues.CHAT.value
    return LLMRequestTypeValues.CHAT.value


def extract_int(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    """Return the first integer-like usage value for the provided keys."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def add_lm_usage_attributes(
    attributes: dict[str, Any],
    usage: Mapping[str, Any],
) -> None:
    """Populate canonical and legacy token usage attributes from LiteLLM usage."""
    prompt_tokens = extract_int(
        mapping=usage,
        keys=(INPUT_TOKENS_KEY, PROMPT_TOKENS_KEY),
    )
    completion_tokens = extract_int(
        mapping=usage,
        keys=(OUTPUT_TOKENS_KEY, COMPLETION_TOKENS_KEY),
    )
    total_tokens = extract_int(mapping=usage, keys=(TOTAL_TOKENS_KEY,))

    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    if prompt_tokens is not None:
        attributes[GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attributes[LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attributes[LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def add_lm_request_attributes(
    attributes: dict[str, Any],
    *,
    instance: Any,
    inputs: Mapping[str, Any],
) -> None:
    """Populate canonical LLM request attributes for a DSPy LM call."""
    model_name = getattr(instance, "model", None)
    model_type = getattr(instance, "model_type", None)
    instance_kwargs = getattr(instance, "kwargs", None)
    request_kwargs = inputs.get("kwargs")

    attributes[GEN_AI_SYSTEM] = extract_provider_name(model_name=model_name)
    if isinstance(model_name, str) and model_name:
        attributes[LLM_REQUEST_MODEL] = model_name
    attributes[LLM_REQUEST_TYPE] = request_type_from_model_type(model_type=model_type)

    merged_kwargs: dict[str, Any] = {}
    if isinstance(instance_kwargs, Mapping):
        merged_kwargs.update(instance_kwargs)
    if isinstance(request_kwargs, Mapping):
        merged_kwargs.update(request_kwargs)

    temperature = merged_kwargs.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] = temperature

    max_tokens = merged_kwargs.get("max_tokens") or merged_kwargs.get(
        "max_completion_tokens"
    )
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
        attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = max_tokens
