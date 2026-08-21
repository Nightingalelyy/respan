"""Translate OpenInference spans into Respan's canonical span contract."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from openinference.semconv.trace import (
    EmbeddingAttributes,
)
from openinference.semconv.trace import (
    SpanAttributes as OISpanAttributes,
)
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
)
from opentelemetry.semconv_ai import (
    SpanAttributes as TLSpanAttributes,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_GUARDRAIL,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

from respan_instrumentation_openinference._serialization import (
    bounded_json,
    bounded_text,
    content_value,
    parse_json,
    to_jsonable,
)

logger = logging.getLogger(__name__)

# Traceloop/GenAI attributes come from the upstream semantic-conventions package.
TRACELOOP_ENTITY_NAME = TLSpanAttributes.TRACELOOP_ENTITY_NAME
TRACELOOP_ENTITY_INPUT = TLSpanAttributes.TRACELOOP_ENTITY_INPUT
TRACELOOP_ENTITY_OUTPUT = TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT
TRACELOOP_ENTITY_PATH = TLSpanAttributes.TRACELOOP_ENTITY_PATH
TRACELOOP_SPAN_KIND = TLSpanAttributes.TRACELOOP_SPAN_KIND
GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
GEN_AI_SYSTEM = TLSpanAttributes.LLM_SYSTEM
LLM_REQUEST_MODEL = TLSpanAttributes.LLM_REQUEST_MODEL
LLM_REQUEST_TYPE = TLSpanAttributes.LLM_REQUEST_TYPE
LLM_USAGE_PROMPT_TOKENS = TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS
LLM_USAGE_COMPLETION_TOKENS = TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS
LLM_USAGE_TOTAL_TOKENS = TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS
LLM_REQUEST_FUNCTIONS = TLSpanAttributes.LLM_REQUEST_FUNCTIONS
LLM_REQUEST_TEMPERATURE = TLSpanAttributes.LLM_REQUEST_TEMPERATURE
LLM_REQUEST_TOP_P = TLSpanAttributes.LLM_REQUEST_TOP_P
LLM_REQUEST_MAX_TOKENS = TLSpanAttributes.LLM_REQUEST_MAX_TOKENS
LLM_REQUEST_REPETITION_PENALTY = TLSpanAttributes.LLM_REQUEST_REPETITION_PENALTY
LLM_TOP_K = TLSpanAttributes.LLM_TOP_K
LLM_CHAT_STOP_SEQUENCES = TLSpanAttributes.LLM_CHAT_STOP_SEQUENCES
LLM_FREQUENCY_PENALTY = TLSpanAttributes.LLM_FREQUENCY_PENALTY
LLM_PRESENCE_PENALTY = TLSpanAttributes.LLM_PRESENCE_PENALTY

# OpenInference attributes come from its upstream semantic-conventions package.
OI_SPAN_KIND = OISpanAttributes.OPENINFERENCE_SPAN_KIND
OI_INPUT_VALUE = OISpanAttributes.INPUT_VALUE
OI_INPUT_MIME_TYPE = OISpanAttributes.INPUT_MIME_TYPE
OI_OUTPUT_VALUE = OISpanAttributes.OUTPUT_VALUE
OI_OUTPUT_MIME_TYPE = OISpanAttributes.OUTPUT_MIME_TYPE
OI_LLM_MODEL_NAME = OISpanAttributes.LLM_MODEL_NAME
OI_LLM_PROVIDER = OISpanAttributes.LLM_PROVIDER
OI_LLM_SYSTEM = OISpanAttributes.LLM_SYSTEM
OI_LLM_INVOCATION_PARAMETERS = OISpanAttributes.LLM_INVOCATION_PARAMETERS
OI_LLM_TOKEN_COUNT_PROMPT = OISpanAttributes.LLM_TOKEN_COUNT_PROMPT
OI_LLM_TOKEN_COUNT_COMPLETION = OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION
OI_LLM_TOKEN_COUNT_TOTAL = OISpanAttributes.LLM_TOKEN_COUNT_TOTAL
OI_LLM_TOKEN_COUNT_CACHE_READ = (
    OISpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
)
OI_LLM_TOOLS = OISpanAttributes.LLM_TOOLS
OI_AGENT_NAME = OISpanAttributes.AGENT_NAME
OI_EMBEDDING_MODEL_NAME = OISpanAttributes.EMBEDDING_MODEL_NAME
OI_EMBEDDING_INVOCATION_PARAMETERS = OISpanAttributes.EMBEDDING_INVOCATION_PARAMETERS
OI_EMBEDDINGS = OISpanAttributes.EMBEDDING_EMBEDDINGS
OI_TOOL_NAME = OISpanAttributes.TOOL_NAME

_LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens"
_OI_INPUT_MESSAGES_PREFIX = "llm.input_messages."
_OI_OUTPUT_MESSAGES_PREFIX = "llm.output_messages."
_OI_TOKEN_COUNT_PREFIX = "llm.token_count."
_OI_TOOLS_PREFIX = "llm.tools."
_OI_EMBEDDINGS_PREFIX = f"{OI_EMBEDDINGS}."
_OI_MESSAGE_ROLE = "message.role"
_OI_MESSAGE_CONTENT = "message.content"
_OI_MESSAGE_CONTENT_PREFIX = "message.content."
_OI_MESSAGE_TOOL_CALLS_PREFIX = "message.tool_calls."
_OI_MESSAGE_FUNCTION_CALL_NAME = "message.function_call_name"
_OI_MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "message.function_call_arguments_json"
_OI_MESSAGE_FINISH_REASON = "message.finish_reason"
_OI_TOOL_PREFIX = "tool."
_OI_TOOL_JSON_SCHEMA = "tool.json_schema"
_OI_TOOL_CALL_PREFIX = "tool_call."
_OI_EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
_OI_EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR

_OI_KIND_TO_LOG_TYPE = {
    "CHAIN": LOG_TYPE_WORKFLOW,
    "LLM": LOG_TYPE_CHAT,
    "TOOL": LOG_TYPE_TOOL,
    "AGENT": LOG_TYPE_AGENT,
    "RETRIEVER": LOG_TYPE_TASK,
    "EMBEDDING": LOG_TYPE_EMBEDDING,
    "RERANKER": LOG_TYPE_TASK,
    "GUARDRAIL": LOG_TYPE_GUARDRAIL,
    "EVALUATOR": LOG_TYPE_TASK,
    "PROMPT": LOG_TYPE_TASK,
    "UNKNOWN": LOG_TYPE_TASK,
}
_LLM_KINDS = {"LLM", "EMBEDDING"}
_INVOCATION_PARAM_MAP = {
    "model": LLM_REQUEST_MODEL,
    "temperature": LLM_REQUEST_TEMPERATURE,
    "top_p": LLM_REQUEST_TOP_P,
    "max_tokens": LLM_REQUEST_MAX_TOKENS,
    "max_output_tokens": LLM_REQUEST_MAX_TOKENS,
    "top_k": LLM_TOP_K,
    "stop_sequences": LLM_CHAT_STOP_SEQUENCES,
    "stop": LLM_CHAT_STOP_SEQUENCES,
    "repetition_penalty": LLM_REQUEST_REPETITION_PENALTY,
    "frequency_penalty": LLM_FREQUENCY_PENALTY,
    "presence_penalty": LLM_PRESENCE_PENALTY,
    "stream": TLSpanAttributes.LLM_IS_STREAMING,
}
_OFF_CONTRACT_ALIAS_KEYS = {
    TRACELOOP_SPAN_KIND,
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
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


def _collect_buckets(attrs: dict[str, Any], prefix: str) -> dict[int, dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = defaultdict(dict)
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        index, separator, field = rest.partition(".")
        if separator and index.isdigit():
            buckets[int(index)][field] = value
    return buckets


def _set_nested(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _signature(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    call_id = tool_call.get("id")
    if isinstance(call_id, (str, int)):
        normalized["id"] = bounded_text(str(call_id))

    function = tool_call.get("function")
    normalized_function: dict[str, Any] = {}
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            normalized_function["name"] = bounded_text(name)
        if "arguments" in function:
            normalized_function["arguments"] = bounded_json(function["arguments"])

    tool_type = tool_call.get("type")
    if isinstance(tool_type, str) and tool_type:
        normalized["type"] = bounded_text(tool_type)
    elif normalized_function:
        normalized["type"] = "function"
    if normalized_function:
        normalized["function"] = normalized_function
    return normalized


def _extract_tool_calls_from_buckets(
    buckets: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    modern_function_signatures: set[str] = set()
    for index in sorted(buckets):
        raw = buckets[index]
        tool_call_buckets: dict[int, dict[str, Any]] = defaultdict(dict)
        for field, value in raw.items():
            if not field.startswith(_OI_MESSAGE_TOOL_CALLS_PREFIX):
                continue
            rest = field[len(_OI_MESSAGE_TOOL_CALLS_PREFIX) :]
            tool_index, separator, tool_field = rest.partition(".")
            if not separator or not tool_index.isdigit():
                continue
            tool_field = tool_field.removeprefix(_OI_TOOL_CALL_PREFIX)
            tool_call_buckets[int(tool_index)][tool_field] = value

        for tool_index in sorted(tool_call_buckets):
            reconstructed: dict[str, Any] = {}
            for field, value in tool_call_buckets[tool_index].items():
                _set_nested(reconstructed, field, value)
            tool_call = _normalize_tool_call(reconstructed)
            signature = _signature(tool_call)
            if tool_call and signature not in seen:
                seen.add(signature)
                result.append(tool_call)
                modern_function_signatures.add(
                    _signature(tool_call.get("function", {}))
                )

        legacy_name = raw.get(_OI_MESSAGE_FUNCTION_CALL_NAME)
        legacy_arguments = raw.get(_OI_MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON)
        if legacy_name is None and legacy_arguments is None:
            continue
        legacy = _normalize_tool_call(
            {
                "type": "function",
                "function": {
                    "name": legacy_name,
                    "arguments": legacy_arguments,
                },
            }
        )
        signature = _signature(legacy)
        function_signature = _signature(legacy.get("function", {}))
        if (
            legacy
            and signature not in seen
            and function_signature not in modern_function_signatures
        ):
            seen.add(signature)
            result.append(legacy)
    return result


def _extract_message_content(raw: dict[str, Any]) -> Any:
    if _OI_MESSAGE_CONTENT in raw:
        return raw[_OI_MESSAGE_CONTENT]
    indexed = []
    for field, value in raw.items():
        if not field.startswith(_OI_MESSAGE_CONTENT_PREFIX):
            continue
        index = field[len(_OI_MESSAGE_CONTENT_PREFIX) :]
        if index.isdigit():
            indexed.append((int(index), value))
    values = [value for _, value in sorted(indexed)]
    if len(values) == 1:
        return values[0]
    if values and all(isinstance(value, str) for value in values):
        return "\n".join(values)
    return values or None


def _message_payloads(buckets: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in sorted(buckets):
        raw = buckets[index]
        message: dict[str, Any] = {}
        role = raw.get(_OI_MESSAGE_ROLE)
        if isinstance(role, str):
            message["role"] = bounded_text(role)
        content = _extract_message_content(raw)
        if content is not None:
            message["content"] = to_jsonable(parse_json(content))
        tool_calls = _extract_tool_calls_from_buckets({index: raw})
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = raw.get(_OI_MESSAGE_FINISH_REASON)
        if isinstance(finish_reason, str):
            message["finish_reason"] = bounded_text(finish_reason)
        if message:
            messages.append(message)
    return messages


def _messages_to_canonical(
    attrs: dict[str, Any],
    buckets: dict[int, dict[str, Any]],
    target_prefix: str,
) -> None:
    for index in sorted(buckets):
        raw = buckets[index]
        target = f"{target_prefix}{index}"
        role = raw.get(_OI_MESSAGE_ROLE)
        if isinstance(role, str):
            attrs[f"{target}.role"] = bounded_text(role)
        content = _extract_message_content(raw)
        if content is not None:
            attrs[f"{target}.content"] = content_value(content)
        tool_calls = _extract_tool_calls_from_buckets({index: raw})
        if tool_calls:
            attrs[f"{target}.tool_calls"] = bounded_json(tool_calls)
        finish_reason = raw.get(_OI_MESSAGE_FINISH_REASON)
        if isinstance(finish_reason, str):
            attrs[f"{target}.finish_reason"] = bounded_text(finish_reason)


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json(value)
    candidates = parsed if isinstance(parsed, list) else [parsed]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        normalized = to_jsonable(candidate)
        if not isinstance(normalized, dict):
            continue
        signature = _signature(normalized)
        if signature not in seen:
            seen.add(signature)
            result.append(normalized)
    return result


def _indexed_tools(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = _collect_buckets(attrs, _OI_TOOLS_PREFIX)
    tools: list[dict[str, Any]] = []
    for index in sorted(buckets):
        raw = buckets[index]
        reconstructed: dict[str, Any] = {}
        schema = raw.get(_OI_TOOL_JSON_SCHEMA)
        if schema is not None:
            parsed_schema = parse_json(schema)
            if isinstance(parsed_schema, dict):
                reconstructed.update(parsed_schema)
            else:
                reconstructed["json_schema"] = parsed_schema
        for field, value in raw.items():
            if field == _OI_TOOL_JSON_SCHEMA:
                continue
            normalized_field = field.removeprefix(_OI_TOOL_PREFIX)
            _set_nested(reconstructed, normalized_field, parse_json(value))
        if reconstructed:
            tools.extend(_normalize_tools(reconstructed))
    return _normalize_tools(tools)


def _attribute_value(value: Any) -> Any:
    normalized = to_jsonable(value)
    if isinstance(normalized, str):
        return bounded_text(normalized)
    if normalized is None or isinstance(normalized, (bool, int, float, str)):
        return normalized
    if (
        isinstance(normalized, list)
        and all(isinstance(item, (bool, int, float, str)) for item in normalized)
        and len({type(item) for item in normalized}) <= 1
    ):
        return tuple(normalized)
    return bounded_json(normalized)


def _lower_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return bounded_text(value.strip().lower())


class OpenInferenceTranslator(SpanProcessor):
    """Normalize ended OpenInference spans before Respan export."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        del span, parent_context

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return
        attrs = dict(original_attrs)
        oi_kind = attrs.get(OI_SPAN_KIND)
        if not isinstance(oi_kind, str) or not oi_kind:
            return

        kind = oi_kind.upper()
        logger.debug("[OI->Respan] Translating %s span: %s", kind, span.name)
        attrs.setdefault(RESPAN_LOG_TYPE, _OI_KIND_TO_LOG_TYPE.get(kind, LOG_TYPE_TASK))

        entity_name = attrs.get(OI_TOOL_NAME) if kind == "TOOL" else None
        entity_name = entity_name or attrs.get(OI_AGENT_NAME) or span.name
        if isinstance(entity_name, str):
            attrs.setdefault(TRACELOOP_ENTITY_NAME, bounded_text(entity_name))
        canonical_name = attrs.get(TRACELOOP_ENTITY_NAME)
        default_path = (
            ""
            if getattr(span, "parent", None) is None
            else bounded_text(
                canonical_name if isinstance(canonical_name, str) else span.name
            )
        )
        attrs.setdefault(TRACELOOP_ENTITY_PATH, default_path)

        input_value = attrs.get(OI_INPUT_VALUE)
        if input_value is None:
            input_value = attrs.get(TRACELOOP_ENTITY_INPUT)
        output_value = attrs.get(OI_OUTPUT_VALUE)
        if output_value is None:
            output_value = attrs.get(TRACELOOP_ENTITY_OUTPUT)
        if kind == "TOOL":
            arguments = parse_json(input_value) if input_value is not None else {}
            if isinstance(arguments, dict) and set(arguments) == {"name", "arguments"}:
                tool_input = arguments
            else:
                tool_input = {"name": entity_name, "arguments": arguments}
            attrs[TRACELOOP_ENTITY_INPUT] = bounded_json(tool_input)
        elif input_value is not None:
            attrs[TRACELOOP_ENTITY_INPUT] = bounded_json(input_value)
        if output_value is not None:
            attrs[TRACELOOP_ENTITY_OUTPUT] = bounded_json(output_value)

        model = attrs.get(OI_LLM_MODEL_NAME) or attrs.get(OI_EMBEDDING_MODEL_NAME)
        if isinstance(model, str) and model:
            attrs.setdefault(LLM_REQUEST_MODEL, bounded_text(model))

        system = _lower_label(attrs.get(OI_LLM_SYSTEM))
        provider = _lower_label(attrs.get(OI_LLM_PROVIDER))
        canonical_system = _lower_label(attrs.get(GEN_AI_SYSTEM))
        canonical_provider = _lower_label(attrs.get(GEN_AI_PROVIDER_NAME))
        if canonical_system or system or provider or canonical_provider:
            attrs[GEN_AI_SYSTEM] = (
                canonical_system or system or provider or canonical_provider
            )
        if canonical_provider or provider or system or canonical_system:
            attrs[GEN_AI_PROVIDER_NAME] = (
                canonical_provider or provider or system or canonical_system
            )

        prompt_tokens = attrs.get(OI_LLM_TOKEN_COUNT_PROMPT)
        completion_tokens = attrs.get(OI_LLM_TOKEN_COUNT_COMPLETION)
        total_tokens = attrs.get(OI_LLM_TOKEN_COUNT_TOTAL)
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            attrs.setdefault(LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
            attrs.setdefault(GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        if isinstance(completion_tokens, int) and not isinstance(
            completion_tokens, bool
        ):
            attrs.setdefault(LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
            attrs.setdefault(GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            attrs.setdefault(LLM_USAGE_TOTAL_TOKENS, total_tokens)
        elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            attrs.setdefault(LLM_USAGE_TOTAL_TOKENS, prompt_tokens + completion_tokens)
        cache_read = attrs.get(OI_LLM_TOKEN_COUNT_CACHE_READ)
        if isinstance(cache_read, int) and not isinstance(cache_read, bool):
            attrs.setdefault(_LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read)

        if kind in _LLM_KINDS:
            self._translate_llm(attrs, kind)
        if kind == "EMBEDDING":
            self._translate_embedding(attrs)

        self._normalize_canonical_content(attrs)
        self._remove_raw_and_alias_attrs(attrs)
        span._attributes = attrs

    def _translate_llm(self, attrs: dict[str, Any], kind: str) -> None:
        attrs.setdefault(
            LLM_REQUEST_TYPE,
            LLMRequestTypeValues.EMBEDDING.value
            if kind == "EMBEDDING"
            else LLMRequestTypeValues.CHAT.value,
        )
        input_buckets = _collect_buckets(attrs, _OI_INPUT_MESSAGES_PREFIX)
        output_buckets = _collect_buckets(attrs, _OI_OUTPUT_MESSAGES_PREFIX)
        _messages_to_canonical(attrs, input_buckets, GEN_AI_PROMPT_PREFIX)
        _messages_to_canonical(attrs, output_buckets, GEN_AI_COMPLETION_PREFIX)
        if TRACELOOP_ENTITY_INPUT not in attrs and input_buckets:
            attrs[TRACELOOP_ENTITY_INPUT] = bounded_json(
                {"messages": _message_payloads(input_buckets)}
            )
        if TRACELOOP_ENTITY_OUTPUT not in attrs and output_buckets:
            attrs[TRACELOOP_ENTITY_OUTPUT] = bounded_json(
                {"messages": _message_payloads(output_buckets)}
            )

        invocation = attrs.get(OI_LLM_INVOCATION_PARAMETERS)
        if kind == "EMBEDDING" and invocation is None:
            invocation = attrs.get(OI_EMBEDDING_INVOCATION_PARAMETERS)
        parameters = parse_json(invocation)
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                target = _INVOCATION_PARAM_MAP.get(key)
                if target:
                    attrs.setdefault(target, _attribute_value(value))

        tools = _normalize_tools(attrs.get(OI_LLM_TOOLS))
        if not tools:
            tools = _indexed_tools(attrs)
        if tools and kind == "LLM":
            attrs.setdefault(LLM_REQUEST_FUNCTIONS, bounded_json(tools))

    @staticmethod
    def _translate_embedding(attrs: dict[str, Any]) -> None:
        buckets = _collect_buckets(attrs, _OI_EMBEDDINGS_PREFIX)
        texts: list[Any] = []
        vectors: list[Any] = []
        for index in sorted(buckets):
            raw = buckets[index]
            if _OI_EMBEDDING_TEXT in raw:
                texts.append(raw[_OI_EMBEDDING_TEXT])
            if _OI_EMBEDDING_VECTOR in raw:
                vectors.append(raw[_OI_EMBEDDING_VECTOR])
        if texts and TRACELOOP_ENTITY_INPUT not in attrs:
            attrs[TRACELOOP_ENTITY_INPUT] = bounded_json(
                texts[0] if len(texts) == 1 else texts
            )
        if vectors and TRACELOOP_ENTITY_OUTPUT not in attrs:
            attrs[TRACELOOP_ENTITY_OUTPUT] = bounded_json(
                vectors[0] if len(vectors) == 1 else vectors
            )

    @staticmethod
    def _normalize_canonical_content(attrs: dict[str, Any]) -> None:
        label_keys = {
            TRACELOOP_ENTITY_NAME,
            TRACELOOP_ENTITY_PATH,
            GEN_AI_SYSTEM,
            GEN_AI_PROVIDER_NAME,
            LLM_REQUEST_MODEL,
            LLM_REQUEST_TYPE,
        }
        for key, value in tuple(attrs.items()):
            if (
                key in {TRACELOOP_ENTITY_INPUT, TRACELOOP_ENTITY_OUTPUT}
                or key == LLM_REQUEST_FUNCTIONS
                or (
                    key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
                    and key.endswith(".tool_calls")
                )
            ):
                attrs[key] = bounded_json(value)
            elif key.startswith(
                (GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX)
            ) and key.endswith(".content"):
                attrs[key] = content_value(value)
            elif key in label_keys or (
                key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
                and key.endswith((".role", ".finish_reason"))
            ):
                attrs[key] = bounded_text(value)

    @staticmethod
    def _remove_raw_and_alias_attrs(attrs: dict[str, Any]) -> None:
        exact_raw_keys = {
            OI_SPAN_KIND,
            OI_INPUT_VALUE,
            OI_INPUT_MIME_TYPE,
            OI_OUTPUT_VALUE,
            OI_OUTPUT_MIME_TYPE,
            OI_LLM_MODEL_NAME,
            OI_LLM_PROVIDER,
            OI_LLM_SYSTEM,
            OI_LLM_INVOCATION_PARAMETERS,
            OI_LLM_TOKEN_COUNT_PROMPT,
            OI_LLM_TOKEN_COUNT_COMPLETION,
            OI_LLM_TOKEN_COUNT_TOTAL,
            OI_LLM_TOKEN_COUNT_CACHE_READ,
            OI_LLM_TOOLS,
            OI_AGENT_NAME,
            OI_EMBEDDING_MODEL_NAME,
            OI_EMBEDDING_INVOCATION_PARAMETERS,
            OI_TOOL_NAME,
            *_OFF_CONTRACT_ALIAS_KEYS,
        }
        raw_prefixes = (
            _OI_INPUT_MESSAGES_PREFIX,
            _OI_OUTPUT_MESSAGES_PREFIX,
            _OI_TOKEN_COUNT_PREFIX,
            _OI_TOOLS_PREFIX,
            _OI_EMBEDDINGS_PREFIX,
            "openinference.",
            "llm.cost.",
            "llm.choices",
            "llm.function_call",
            "llm.prompt",
            "tool.",
        )
        for key in exact_raw_keys:
            attrs.pop(key, None)
        for key in tuple(attrs):
            if key.startswith(raw_prefixes):
                attrs.pop(key, None)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True
