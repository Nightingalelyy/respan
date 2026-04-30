"""LangChain callback handler that emits Respan-compatible spans."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from typing import Any
from uuid import UUID, uuid4

from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_COMPLETION,
    LOG_TYPE_CUSTOM,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.otlp_constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import (
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span
from respan_sdk.utils.data_processing.id_processing import format_span_id, format_trace_id

from respan_instrumentation_langchain._constants import (
    GEN_AI_COMPLETION_PREFIX,
    GEN_AI_PROMPT_PREFIX,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    LANGCHAIN_FRAMEWORK_ATTR,
    LANGCHAIN_METADATA_ATTR,
    LANGCHAIN_PARENT_RUN_ID_ATTR,
    LANGCHAIN_RUN_ID_ATTR,
    LANGCHAIN_SERIALIZED_ATTR,
    LANGCHAIN_TAGS_ATTR,
    RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR,
    RESPAN_OVERRIDE_INPUT_ATTR,
    RESPAN_OVERRIDE_MODEL_ATTR,
    RESPAN_OVERRIDE_OUTPUT_ATTR,
    RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR,
    RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR,
)

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised in missing-dependency envs
    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback so importing the package does not require LangChain eagerly."""

        raise_error = False
        run_inline = True


try:
    from langgraph.callbacks import GraphCallbackHandler as _GraphCallbackHandler
except Exception:  # pragma: no cover - optional dependency
    _GraphCallbackHandler = None


if _GraphCallbackHandler is None:
    _CallbackBase = BaseCallbackHandler
elif issubclass(_GraphCallbackHandler, BaseCallbackHandler):
    _CallbackBase = _GraphCallbackHandler
else:
    class _CallbackBase(_GraphCallbackHandler, BaseCallbackHandler):  # type: ignore[misc, valid-type]
        pass


_MESSAGE_ROLE_MAP = {
    "ai": "assistant",
    "human": "user",
    "chat": "user",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}
_EMPTY_VALUES = (None, "", (), [])
STATUS_CODE_ATTR = "status_code"
_JSON_CODE_FENCE_RE = re.compile(
    r"^\s*(?P<fence>`{3,}|~{3,})[ \t]*(?P<language>jsonc?)?[ \t]*\r?\n"
    r"(?P<body>.*?)(?:\r?\n)?(?P=fence)\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class _RunRecord:
    run_id: str
    trace_id: str
    span_id: str
    parent_run_id: str | None
    parent_span_id: str | None
    name: str
    entity_path: str
    log_type: str
    span_kind: str
    start_ns: int
    input_value: Any = None
    serialized: Any = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    framework: str = "langchain"
    extra_attributes: dict[str, Any] = field(default_factory=dict)
    streamed_tokens: list[str] = field(default_factory=list)


def _run_id_to_hex(run_id: Any) -> str:
    if run_id is None:
        return uuid4().hex
    if isinstance(run_id, UUID):
        return run_id.hex
    value = str(run_id)
    try:
        return UUID(value).hex
    except (TypeError, ValueError):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return digest[:32]


def _derive_span_id(*parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    span_id = digest[:16]
    if int(span_id, 16) == 0:
        return "0000000000000001"
    return span_id


def _get_active_otel_parent() -> tuple[str, str] | None:
    try:
        span_context = trace.get_current_span().get_span_context()
    except Exception:
        return None

    trace_id = getattr(span_context, "trace_id", 0)
    span_id = getattr(span_context, "span_id", 0)
    if not trace_id or not span_id:
        return None
    return format_trace_id(trace_id), format_span_id(span_id)


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _to_json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=_json_default)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _safe_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return None


def _message_to_dict(message: Any) -> dict[str, Any]:
    message_dict = _safe_dict(message)
    if message_dict is None:
        message_dict = {}

    role = (
        message_dict.get("role")
        or message_dict.get("type")
        or getattr(message, "role", None)
        or getattr(message, "type", None)
    )
    content = message_dict.get("content", getattr(message, "content", None))
    normalized: dict[str, Any] = {
        "role": _MESSAGE_ROLE_MAP.get(str(role), str(role)) if role else "unknown",
        "content": content,
    }

    for key in ("id", "name", "tool_call_id"):
        value = message_dict.get(key, getattr(message, key, None))
        if value not in _EMPTY_VALUES:
            normalized[key] = value

    tool_calls = message_dict.get("tool_calls", getattr(message, "tool_calls", None))
    if tool_calls not in _EMPTY_VALUES:
        normalized["tool_calls"] = _serialize_value(tool_calls)

    additional_kwargs = message_dict.get(
        "additional_kwargs", getattr(message, "additional_kwargs", None)
    )
    if isinstance(additional_kwargs, Mapping):
        for key in ("tool_calls", "function_call"):
            if key in additional_kwargs and key not in normalized:
                normalized[key] = _serialize_value(additional_kwargs[key])

    return normalized


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_value(dataclasses.asdict(value))
    value_dict = _safe_dict(value)
    if value_dict is not None:
        return _serialize_value(value_dict)
    if hasattr(value, "page_content"):
        payload = {
            "page_content": getattr(value, "page_content", None),
            "metadata": getattr(value, "metadata", None),
        }
        doc_id = getattr(value, "id", None)
        if doc_id not in _EMPTY_VALUES:
            payload["id"] = doc_id
        return _serialize_value(payload)
    return str(value)


def _strip_json_code_fence(value: str) -> str:
    match = _JSON_CODE_FENCE_RE.match(value)
    if not match:
        return value

    body = match.group("body").strip()
    if match.group("language"):
        return body

    try:
        json.loads(body)
    except (TypeError, ValueError):
        return value
    return body


def _normalize_output_for_logging(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_json_code_fence(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_output_for_logging(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_output_for_logging(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_output_for_logging(item) for item in value]
    return value


def _normalize_chat_messages(messages: Any) -> list[list[dict[str, Any]]]:
    conversations = messages if isinstance(messages, list) else [messages]
    normalized: list[list[dict[str, Any]]] = []
    for conversation in conversations:
        if not isinstance(conversation, list):
            conversation = [conversation]
        normalized.append([_message_to_dict(message) for message in conversation])
    return normalized


def _extract_name(serialized: Any, fallback: str) -> str:
    if isinstance(serialized, Mapping):
        for key in ("name", "id", "type"):
            value = serialized.get(key)
            if isinstance(value, str) and value:
                if key == "id" and "." in value:
                    return value.rsplit(".", 1)[-1]
                return value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
                return str(value[-1])
        kwargs = serialized.get("kwargs")
        if isinstance(kwargs, Mapping):
            for key in ("name", "model", "model_name", "repo_id"):
                value = kwargs.get(key)
                if isinstance(value, str) and value:
                    return value
    return fallback


def _extract_model(serialized: Any, response: Any = None, metadata: Mapping[str, Any] | None = None) -> str | None:
    candidates: list[Any] = []
    if isinstance(metadata, Mapping):
        candidates.extend(
            metadata.get(key)
            for key in ("ls_model_name", "model", "model_name")
        )
    if isinstance(serialized, Mapping):
        kwargs = serialized.get("kwargs")
        if isinstance(kwargs, Mapping):
            candidates.extend(
                kwargs.get(key)
                for key in ("model", "model_name", "repo_id")
            )
        candidates.extend(serialized.get(key) for key in ("model", "model_name"))

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        candidates.extend(llm_output.get(key) for key in ("model_name", "model"))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    payloads: list[Any] = []
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        payloads.extend(
            llm_output.get(key)
            for key in ("token_usage", "usage", "usage_metadata")
        )
        payloads.append(llm_output)

    generations = getattr(response, "generations", None)
    if isinstance(generations, list) and generations:
        first_generation = generations[0][0] if isinstance(generations[0], list) and generations[0] else generations[0]
        message = getattr(first_generation, "message", None)
        if message is not None:
            payloads.append(getattr(message, "usage_metadata", None))
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, Mapping):
                payloads.extend(
                    response_metadata.get(key)
                    for key in ("token_usage", "usage", "usage_metadata")
                )

    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        prompt_tokens = _coerce_int(
            payload.get("prompt_tokens", payload.get("input_tokens"))
        )
        completion_tokens = _coerce_int(
            payload.get("completion_tokens", payload.get("output_tokens"))
        )
        total_tokens = _coerce_int(payload.get("total_tokens"))
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if (
            prompt_tokens is not None
            or completion_tokens is not None
            or total_tokens is not None
        ):
            return prompt_tokens, completion_tokens, total_tokens

    return None, None, None


def _generation_to_message(generation: Any) -> dict[str, Any]:
    message = getattr(generation, "message", None)
    if message is not None:
        return _message_to_dict(message)
    text = getattr(generation, "text", None)
    if text is not None:
        return {"role": "assistant", "content": text}
    return {"role": "assistant", "content": _serialize_value(generation)}


def _extract_llm_output(response: Any) -> tuple[Any, list[dict[str, Any]]]:
    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        normalized_batches = []
        completion_messages = []
        for batch in generations:
            batch_items = batch if isinstance(batch, list) else [batch]
            normalized_batch = [_generation_to_message(item) for item in batch_items]
            normalized_batches.append(normalized_batch)
            completion_messages.extend(normalized_batch)
        return normalized_batches, completion_messages

    serialized = _serialize_value(response)
    message = (
        serialized
        if isinstance(serialized, Mapping) and "content" in serialized
        else {"role": "assistant", "content": serialized}
    )
    return serialized, [dict(message)]


def _extract_tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            continue
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            tool_call = dict(raw_tool_call)
            if "function" not in tool_call and "name" in tool_call:
                tool_call = {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name"),
                        "arguments": _to_json_string(
                            tool_call.get("args", tool_call.get("arguments"))
                        ),
                    },
                }
            tool_calls.append(_serialize_value(tool_call))
    return tool_calls or None


def _extract_tool_names_from_serialized(serialized: Any) -> list[str] | None:
    if not isinstance(serialized, Mapping):
        return None
    tools = serialized.get("tools") or serialized.get("functions")
    if not isinstance(tools, list):
        kwargs = serialized.get("kwargs")
        tools = kwargs.get("tools") if isinstance(kwargs, Mapping) else None
    if not isinstance(tools, list):
        return None
    names = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names or None


def _detect_framework(
    *,
    serialized: Any = None,
    tags: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> str:
    haystack: list[str] = []
    haystack.extend(str(tag).lower() for tag in tags or [])
    if isinstance(metadata, Mapping):
        haystack.extend(str(key).lower() for key in metadata)
        haystack.extend(str(value).lower() for value in metadata.values() if isinstance(value, str))
    if isinstance(serialized, Mapping):
        haystack.append(json.dumps(_serialize_value(serialized), default=str).lower())
    if name:
        haystack.append(name.lower())
    text = " ".join(haystack)
    if "langflow" in text:
        return "langflow"
    if "langgraph" in text or "graph:" in text or "__pregel" in text:
        return "langgraph"
    return "langchain"


def _set_if_present(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value not in _EMPTY_VALUES:
        attrs[key] = value


def _set_error_attributes(
    attrs: dict[str, Any],
    error: BaseException | None,
    *,
    status_code: int = 500,
) -> str | None:
    if error is None:
        return None

    error_message = str(error)
    attrs.setdefault(ERROR_MESSAGE_ATTR, error_message)
    attrs.setdefault(STATUS_CODE_ATTR, status_code if status_code >= 400 else 500)
    return error_message


def _callback_list_contains(callbacks: list[Any], handler: Any) -> bool:
    return any(
        callback is handler or isinstance(callback, RespanCallbackHandler)
        for callback in callbacks
    )


def add_respan_callback(
    config: Mapping[str, Any] | None = None,
    handler: "RespanCallbackHandler | None" = None,
) -> dict[str, Any]:
    """Return a RunnableConfig copy with a Respan callback handler attached."""
    callback_handler = handler or get_callback_handler()
    new_config = dict(config or {})
    callbacks = new_config.get("callbacks")
    new_config["callbacks"] = _with_respan_callback(callbacks, callback_handler)
    return new_config


def _with_respan_callback(callbacks: Any, handler: "RespanCallbackHandler") -> Any:
    if callbacks is None:
        return [handler]

    if isinstance(callbacks, tuple):
        callback_list = list(callbacks)
        if _callback_list_contains(callback_list, handler):
            return callbacks
        return [*callback_list, handler]

    if isinstance(callbacks, list):
        if _callback_list_contains(callbacks, handler):
            return callbacks
        return [*callbacks, handler]

    existing_handlers = getattr(callbacks, "handlers", None)
    if isinstance(existing_handlers, list):
        if not _callback_list_contains(existing_handlers, handler):
            callbacks.add_handler(handler, inherit=True) if hasattr(callbacks, "add_handler") else existing_handlers.append(handler)
        return callbacks

    if isinstance(callbacks, BaseCallbackHandler):
        if isinstance(callbacks, RespanCallbackHandler):
            return [callbacks]
        return [callbacks, handler]

    return [callbacks, handler]


def get_callback_handler(**kwargs: Any) -> "RespanCallbackHandler":
    """Create a Respan callback handler for explicit LangChain/LangGraph config."""
    kwargs.setdefault("group_langflow_root_runs", True)
    return RespanCallbackHandler(**kwargs)


class RespanCallbackHandler(_CallbackBase):  # type: ignore[misc, valid-type]
    """LangChain callback handler that emits spans into the Respan OTEL pipeline."""

    raise_error = False
    run_inline = True

    def __init__(
        self,
        *,
        include_content: bool = True,
        include_metadata: bool = True,
        group_langflow_root_runs: bool = False,
        max_cached_runs: int = 4096,
    ) -> None:
        super().__init__()
        self.include_content = include_content
        self.include_metadata = include_metadata
        self.group_langflow_root_runs = group_langflow_root_runs
        self.max_cached_runs = max_cached_runs
        self._runs: dict[str, _RunRecord] = {}
        self._run_trace_ids: OrderedDict[str, str] = OrderedDict()
        self._run_paths: OrderedDict[str, str] = OrderedDict()
        self._langflow_trace_id = uuid4().hex

    def _remember_run(self, record: _RunRecord) -> None:
        self._run_trace_ids[record.run_id] = record.trace_id
        self._run_paths[record.run_id] = record.entity_path
        self._run_trace_ids.move_to_end(record.run_id)
        self._run_paths.move_to_end(record.run_id)
        while len(self._run_trace_ids) > self.max_cached_runs:
            self._run_trace_ids.popitem(last=False)
        while len(self._run_paths) > self.max_cached_runs:
            self._run_paths.popitem(last=False)

    def _start_run(
        self,
        *,
        run_id: Any,
        parent_run_id: Any,
        name: str,
        log_type: str,
        span_kind: str,
        input_value: Any = None,
        serialized: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        extra_attributes: dict[str, Any] | None = None,
    ) -> None:
        run_hex = _run_id_to_hex(run_id)
        parent_hex = _run_id_to_hex(parent_run_id) if parent_run_id is not None else None
        framework = _detect_framework(
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            name=name,
        )
        active_parent = _get_active_otel_parent() if parent_hex is None else None
        fallback_trace_id = (
            self._langflow_trace_id
            if (
                framework == "langflow"
                and self.group_langflow_root_runs
                and active_parent is None
                and parent_hex is None
            )
            else parent_hex or run_hex
        )
        trace_id = (
            self._runs[parent_hex].trace_id
            if parent_hex in self._runs
            else self._run_trace_ids.get(
                parent_hex or "",
                active_parent[0] if active_parent else fallback_trace_id,
            )
        )
        parent_span_id = (
            _derive_span_id(parent_hex)
            if parent_hex
            else active_parent[1] if active_parent else None
        )
        span_id = _derive_span_id(run_hex)
        parent_path = (
            self._runs[parent_hex].entity_path
            if parent_hex in self._runs
            else self._run_paths.get(parent_hex or "")
        )
        entity_path = f"{parent_path}.{name}" if parent_path else name
        self._runs[run_hex] = _RunRecord(
            run_id=run_hex,
            trace_id=trace_id,
            span_id=span_id,
            parent_run_id=parent_hex,
            parent_span_id=parent_span_id,
            name=name,
            entity_path=entity_path,
            log_type=log_type,
            span_kind=span_kind,
            start_ns=time.time_ns(),
            input_value=input_value,
            serialized=_serialize_value(serialized),
            tags=tags,
            metadata=metadata,
            framework=framework,
            extra_attributes=extra_attributes or {},
        )

    def _build_attributes(
        self,
        record: _RunRecord,
        *,
        output_value: Any = None,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
            RESPAN_LOG_TYPE: record.log_type,
            SpanAttributes.TRACELOOP_SPAN_KIND: record.span_kind,
            SpanAttributes.TRACELOOP_ENTITY_NAME: record.name,
            SpanAttributes.TRACELOOP_ENTITY_PATH: record.entity_path,
            LANGCHAIN_RUN_ID_ATTR: record.run_id,
            LANGCHAIN_FRAMEWORK_ATTR: record.framework,
        }
        _set_if_present(attrs, LANGCHAIN_PARENT_RUN_ID_ATTR, record.parent_run_id)

        if self.include_metadata:
            _set_if_present(attrs, LANGCHAIN_TAGS_ATTR, _to_json_string(record.tags))
            _set_if_present(attrs, LANGCHAIN_METADATA_ATTR, _to_json_string(record.metadata))
            _set_if_present(attrs, LANGCHAIN_SERIALIZED_ATTR, _to_json_string(record.serialized))

        if self.include_content:
            input_string = _to_json_string(record.input_value)
            output_string = _to_json_string(_normalize_output_for_logging(output_value))
            _set_if_present(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, input_string)
            _set_if_present(attrs, RESPAN_OVERRIDE_INPUT_ATTR, input_string)
            _set_if_present(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_string)
            _set_if_present(attrs, RESPAN_OVERRIDE_OUTPUT_ATTR, output_string)

        attrs.update(record.extra_attributes)
        return attrs

    def _end_run(
        self,
        *,
        run_id: Any,
        output_value: Any = None,
        error: BaseException | None = None,
        extra_attributes: dict[str, Any] | None = None,
    ) -> bool:
        run_hex = _run_id_to_hex(run_id)
        record = self._runs.pop(run_hex, None)
        if record is None:
            return False

        if record.streamed_tokens and output_value in _EMPTY_VALUES:
            output_value = "".join(record.streamed_tokens)

        if extra_attributes:
            record.extra_attributes.update(extra_attributes)

        attrs = self._build_attributes(record, output_value=output_value)
        error_message = _set_error_attributes(attrs, error)
        span = build_readable_span(
            name=record.name,
            trace_id=record.trace_id,
            span_id=record.span_id,
            parent_id=record.parent_span_id,
            start_time_ns=record.start_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            status_code=500 if error else 200,
            error_message=error_message,
        )
        self._remember_run(record)
        return inject_span(span)

    def _emit_event_span(
        self,
        *,
        parent_run_id: Any = None,
        name: str,
        log_type: str = LOG_TYPE_TASK,
        span_kind: str = LOG_TYPE_TASK,
        input_value: Any = None,
        output_value: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        error: BaseException | None = None,
        extra_attributes: dict[str, Any] | None = None,
    ) -> bool:
        parent_hex = _run_id_to_hex(parent_run_id) if parent_run_id is not None else None
        active_parent = _get_active_otel_parent() if parent_hex is None else None
        trace_id = (
            self._runs[parent_hex].trace_id
            if parent_hex in self._runs
            else self._run_trace_ids.get(
                parent_hex or "",
                active_parent[0] if active_parent else parent_hex or uuid4().hex,
            )
        )
        parent_span_id = (
            _derive_span_id(parent_hex)
            if parent_hex
            else active_parent[1] if active_parent else None
        )
        parent_path = (
            self._runs[parent_hex].entity_path
            if parent_hex in self._runs
            else self._run_paths.get(parent_hex or "")
        )
        entity_path = f"{parent_path}.{name}" if parent_path else name
        span_key = f"{trace_id}:{parent_hex}:{name}:{time.time_ns()}"
        record = _RunRecord(
            run_id=_run_id_to_hex(span_key),
            trace_id=trace_id,
            span_id=_derive_span_id(span_key),
            parent_run_id=parent_hex,
            parent_span_id=parent_span_id,
            name=name,
            entity_path=entity_path,
            log_type=log_type,
            span_kind=span_kind,
            start_ns=time.time_ns(),
            input_value=input_value,
            tags=tags,
            metadata=metadata,
            framework=_detect_framework(tags=tags, metadata=metadata, name=name),
            extra_attributes=extra_attributes or {},
        )
        attrs = self._build_attributes(record, output_value=output_value)
        error_message = _set_error_attributes(attrs, error)
        span = build_readable_span(
            name=name,
            trace_id=trace_id,
            span_id=record.span_id,
            parent_id=parent_span_id,
            start_time_ns=record.start_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            status_code=500 if error else 200,
            error_message=error_message,
        )
        return inject_span(span)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or _extract_name(serialized, "chain")
        is_root = parent_run_id is None
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            log_type=LOG_TYPE_WORKFLOW if is_root else LOG_TYPE_TASK,
            span_kind=LOG_TYPE_WORKFLOW if is_root else LOG_TYPE_TASK,
            input_value=_serialize_value(inputs),
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, output_value=_serialize_value(outputs))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, error=error)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        normalized_messages = _normalize_chat_messages(messages)
        first_conversation = normalized_messages[0] if normalized_messages else []
        extra_attrs: dict[str, Any] = {
            LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        }
        model = _extract_model(serialized, metadata=metadata)
        _set_if_present(extra_attrs, LLM_REQUEST_MODEL, model)
        _set_if_present(extra_attrs, RESPAN_OVERRIDE_MODEL_ATTR, model)
        for index, message in enumerate(first_conversation):
            for key, value in message.items():
                _set_if_present(extra_attrs, f"{GEN_AI_PROMPT_PREFIX}.{index}.{key}", _to_json_string(value) if isinstance(value, (dict, list)) else value)
        tool_names = _extract_tool_names_from_serialized(serialized)
        _set_if_present(extra_attrs, RESPAN_SPAN_TOOLS, _to_json_string(tool_names))

        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_extract_name(serialized, "chat_model"),
            log_type=LOG_TYPE_CHAT,
            span_kind=LLMRequestTypeValues.CHAT.value,
            input_value=normalized_messages,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            extra_attributes=extra_attrs,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        extra_attrs: dict[str, Any] = {LLM_REQUEST_TYPE: LOG_TYPE_COMPLETION}
        model = _extract_model(serialized, metadata=metadata)
        _set_if_present(extra_attrs, LLM_REQUEST_MODEL, model)
        _set_if_present(extra_attrs, RESPAN_OVERRIDE_MODEL_ATTR, model)
        for index, prompt in enumerate(prompts or []):
            extra_attrs[f"{GEN_AI_PROMPT_PREFIX}.{index}.role"] = "user"
            extra_attrs[f"{GEN_AI_PROMPT_PREFIX}.{index}.content"] = prompt

        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_extract_name(serialized, "llm"),
            log_type=LOG_TYPE_COMPLETION,
            span_kind=LOG_TYPE_COMPLETION,
            input_value=prompts,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            extra_attributes=extra_attrs,
        )

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._runs.get(_run_id_to_hex(run_id))
        if record is not None and token:
            record.streamed_tokens.append(token)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run_hex = _run_id_to_hex(run_id)
        record = self._runs.get(run_hex)
        output_payload, completion_messages = _extract_llm_output(response)
        completion_messages = _normalize_output_for_logging(completion_messages)
        extra_attrs: dict[str, Any] = {}
        if record is not None:
            model = _extract_model(record.serialized, response=response, metadata=record.metadata)
            _set_if_present(extra_attrs, LLM_REQUEST_MODEL, model)
            _set_if_present(extra_attrs, RESPAN_OVERRIDE_MODEL_ATTR, model)

        for index, message in enumerate(completion_messages):
            for key, value in message.items():
                _set_if_present(extra_attrs, f"{GEN_AI_COMPLETION_PREFIX}.{index}.{key}", _to_json_string(value) if isinstance(value, (dict, list)) else value)

        if completion_messages:
            tool_calls = _extract_tool_calls_from_messages(completion_messages)
            _set_if_present(extra_attrs, RESPAN_SPAN_TOOL_CALLS, _to_json_string(tool_calls))

        prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
        for key, value in (
            (LLM_USAGE_PROMPT_TOKENS, prompt_tokens),
            (GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens),
            (RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR, prompt_tokens),
            (LLM_USAGE_COMPLETION_TOKENS, completion_tokens),
            (GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens),
            (RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR, completion_tokens),
            (GEN_AI_USAGE_TOTAL_TOKENS, total_tokens),
            (RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR, total_tokens),
        ):
            _set_if_present(extra_attrs, key, value)

        self._end_run(
            run_id=run_id,
            output_value=output_payload,
            extra_attributes=extra_attrs,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, error=error)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or _extract_name(serialized, "tool")
        input_value = _serialize_value(inputs) if inputs is not None else input_str
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            log_type=LOG_TYPE_TOOL,
            span_kind=LOG_TYPE_TOOL,
            input_value=input_value,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            extra_attributes={
                GEN_AI_TOOL_NAME: name,
                GEN_AI_TOOL_CALL_ARGUMENTS: _to_json_string(input_value),
            },
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(
            run_id=run_id,
            output_value=_serialize_value(output),
            extra_attributes={GEN_AI_TOOL_CALL_RESULT: _to_json_string(_serialize_value(output))},
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, error=error)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=kwargs.get("name") or _extract_name(serialized, "retriever"),
            log_type=LOG_TYPE_TASK,
            span_kind=LOG_TYPE_TASK,
            input_value=query,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
        )

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, output_value=_serialize_value(list(documents)))

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, error=error)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = getattr(action, "tool", None) or "agent_action"
        tool_input = getattr(action, "tool_input", None)
        log = getattr(action, "log", None)
        self._emit_event_span(
            parent_run_id=run_id or parent_run_id,
            name=str(tool_name),
            log_type=LOG_TYPE_TOOL,
            span_kind=LOG_TYPE_TOOL,
            input_value=_serialize_value(tool_input),
            output_value=log,
            tags=tags,
            metadata=metadata,
            extra_attributes={
                GEN_AI_TOOL_NAME: str(tool_name),
                GEN_AI_TOOL_CALL_ARGUMENTS: _to_json_string(_serialize_value(tool_input)),
            },
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        output = getattr(finish, "return_values", finish)
        self._emit_event_span(
            parent_run_id=run_id or parent_run_id,
            name="agent_finish",
            log_type=LOG_TYPE_AGENT,
            span_kind=LOG_TYPE_AGENT,
            output_value=_serialize_value(output),
            tags=tags,
            metadata=metadata,
        )

    def on_text(self, text: str, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._runs.get(_run_id_to_hex(run_id))
        if record is not None and text:
            record.streamed_tokens.append(text)

    def on_retry(self, retry_state: Any, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._runs.get(_run_id_to_hex(run_id))
        if record is None:
            return
        retries = record.extra_attributes.setdefault("langchain.retry_count", 0)
        record.extra_attributes["langchain.retry_count"] = retries + 1
        record.extra_attributes["langchain.retry_state"] = _to_json_string(_serialize_value(retry_state))

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit_event_span(
            parent_run_id=run_id,
            name=name,
            log_type=LOG_TYPE_CUSTOM,
            span_kind=LOG_TYPE_TASK,
            input_value=_serialize_value(data),
            tags=tags,
            metadata=metadata,
        )

    def on_interrupt(self, event: Any) -> None:
        self._emit_graph_lifecycle_event("langgraph.interrupt", event)

    def on_resume(self, event: Any) -> None:
        self._emit_graph_lifecycle_event("langgraph.resume", event)

    def _emit_graph_lifecycle_event(self, name: str, event: Any) -> None:
        event_payload = _serialize_value(event)
        run_id = event_payload.get("run_id") if isinstance(event_payload, Mapping) else None
        self._emit_event_span(
            parent_run_id=run_id,
            name=name,
            log_type=LOG_TYPE_TASK,
            span_kind=LOG_TYPE_TASK,
            input_value=event_payload,
            metadata={"framework": "langgraph"},
        )
