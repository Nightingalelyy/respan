"""Serialization and attribute helpers for DSPy instrumentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_dspy._constants import (
    ANTHROPIC_PROVIDER_PREFIX,
    AZURE_PROVIDER_PREFIX,
    BEDROCK_PROVIDER_PREFIX,
    CHAT_MODEL_TYPE,
    COMPLETION_TOKENS_KEY,
    DSPY_PROVIDER_NAME,
    DSPY_USAGE_INPUT_TOKENS_ATTR,
    DSPY_USAGE_OUTPUT_TOKENS_ATTR,
    GEMINI_PROVIDER_PREFIX,
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
from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """Serialize arbitrary DSPy payloads into OTEL-safe JSON strings."""
    try:
        return json.dumps(_dspy_safe_value(value=value), default=str)
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

    signature_value = _dspy_signature_value(value=value)
    if signature_value is not None:
        return signature_value

    for method_name in ("toDict", "to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _dspy_safe_value(value=method())
            except Exception:
                continue

    return _dspy_safe_value(value=value)


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
        attributes[DSPY_USAGE_INPUT_TOKENS_ATTR] = prompt_tokens
        attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attributes[DSPY_USAGE_OUTPUT_TOKENS_ATTR] = completion_tokens
        attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
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

    attributes[SpanAttributes.LLM_SYSTEM] = extract_provider_name(
        model_name=model_name
    )
    if isinstance(model_name, str) and model_name:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name
    attributes[SpanAttributes.LLM_REQUEST_TYPE] = request_type_from_model_type(
        model_type=model_type
    )

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


def _dspy_safe_value(value: Any) -> Any:
    module_value = _dspy_module_value(value=value)
    if module_value is not None:
        return module_value

    signature_value = _dspy_signature_value(value=value)
    if signature_value is not None:
        return signature_value

    field_value = _dspy_field_value(value=value)
    if field_value is not None:
        return field_value

    example_value = _dspy_example_value(value=value)
    if example_value is not None:
        return example_value

    method_value = _dspy_method_value(value=value)
    if method_value is not None:
        return method_value

    if isinstance(value, Mapping):
        return {
            str(key): _dspy_safe_value(value=nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_dspy_safe_value(value=nested_value) for nested_value in value]

    return serialize_value(value=value)


def _dspy_module_value(value: Any) -> dict[str, Any] | None:
    named_predictors = getattr(value, "named_predictors", None)
    if not callable(named_predictors):
        return None

    result: dict[str, Any] = {
        "name": type(value).__name__,
    }
    stage = getattr(value, "stage", None)
    if isinstance(stage, str) and stage:
        result["stage"] = stage

    signature_value = _dspy_signature_value(value=getattr(value, "signature", None))
    if signature_value is not None:
        result["signature"] = signature_value

    try:
        predictors = [str(name) for name, _ in named_predictors()]
    except Exception:
        predictors = []
    if predictors:
        result["predictors"] = predictors

    return result


def _dspy_signature_value(value: Any) -> dict[str, Any] | None:
    input_fields = getattr(value, "input_fields", None)
    output_fields = getattr(value, "output_fields", None)
    instructions = getattr(value, "instructions", None)
    if not isinstance(input_fields, Mapping) or not isinstance(output_fields, Mapping):
        return None

    name = getattr(value, "__name__", None) or type(value).__name__
    return {
        "name": str(name),
        "instructions": instructions,
        "input_fields": _field_names(fields=input_fields),
        "output_fields": _field_names(fields=output_fields),
    }


def _dspy_field_value(value: Any) -> dict[str, Any] | None:
    json_schema_extra = getattr(value, "json_schema_extra", None)
    annotation = getattr(value, "annotation", None)
    if not isinstance(json_schema_extra, Mapping) or annotation is None:
        return None

    field_type = json_schema_extra.get("__dspy_field_type")
    description = json_schema_extra.get("desc")
    annotation_name = getattr(annotation, "__name__", None) or str(annotation)
    return {
        "field_type": field_type,
        "annotation": annotation_name,
        "description": description,
    }


def _dspy_example_value(value: Any) -> dict[str, Any] | None:
    to_dict = getattr(value, "toDict", None)
    inputs = getattr(value, "inputs", None)
    labels = getattr(value, "labels", None)
    if not callable(to_dict) or not callable(inputs) or not callable(labels):
        return None

    try:
        return {
            "values": _dspy_safe_value(value=to_dict()),
            "inputs": _dspy_safe_value(value=inputs().toDict()),
            "labels": _dspy_safe_value(value=labels().toDict()),
        }
    except Exception:
        return None


def _dspy_method_value(value: Any) -> Any | None:
    module_name = type(value).__module__
    if not module_name.startswith("dspy."):
        return None

    for method_name in ("toDict", "to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _dspy_safe_value(value=method())
            except Exception:
                continue
    return None


def _field_names(*, fields: Mapping[str, Any]) -> list[str]:
    return [str(field_name) for field_name in fields]
