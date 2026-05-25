"""Braintrust instrumentation plugin for Respan."""

from __future__ import annotations

import datetime
import json
import logging
import math
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
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
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.utils.span_factory import (
    build_readable_span,
    inject_span,
    read_propagated_attributes,
)

from respan_instrumentation_braintrust._constants import (
    BRAINTRUST_DEFAULT_SPAN_NAME,
    BRAINTRUST_ENTITY_PATH,
    BRAINTRUST_METADATA_PREFIX,
    BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE,
)

logger = logging.getLogger(__name__)

_GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
_GEN_AI_USAGE_INPUT_TOKENS = getattr(
    TLSpanAttributes,
    "LLM_USAGE_INPUT_TOKENS",
    "gen_ai.usage.input_tokens",
)
_GEN_AI_USAGE_OUTPUT_TOKENS = getattr(
    TLSpanAttributes,
    "LLM_USAGE_OUTPUT_TOKENS",
    "gen_ai.usage.output_tokens",
)
_LLM_USAGE_TOTAL_TOKENS = TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS


@dataclass(frozen=True)
class _BufferedBraintrustItem:
    item: Any
    propagated_attributes: dict[str, Any]


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return None


def _format_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, uuid.UUID):
        return value.hex
    if isinstance(value, str):
        try:
            return uuid.UUID(value).hex
        except ValueError:
            return value
    return str(value)


def _timestamp_to_ns(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        if not math.isfinite(value):
            return None
        return int(value * 1_000_000_000)
    if isinstance(value, datetime.datetime):
        return int(value.astimezone(datetime.timezone.utc).timestamp() * 1_000_000_000)
    return None


def _sanitize_json(value: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str | int | bool) or value is None:
        return value
    if isinstance(value, datetime.datetime):
        return value.astimezone(datetime.timezone.utc).isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            return "[CYCLE]"
        seen.add(object_id)
        return {str(key): _sanitize_json(val, seen=seen) for key, val in value.items()}
    if isinstance(value, list | tuple | set):
        object_id = id(value)
        if object_id in seen:
            return ["[CYCLE]"]
        seen.add(object_id)
        return [_sanitize_json(item, seen=seen) for item in value]
    return str(value)


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_sanitize_json(serialize_value(value)), default=str)


def _extract_nested_mapping(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _extract_model(record: Mapping[str, Any], span_attributes: Mapping[str, Any]) -> str | None:
    model = _coerce_str(record.get("model"))
    if model:
        return model

    for source in (record.get("metadata"), span_attributes, record.get("metrics")):
        if not isinstance(source, Mapping):
            continue
        for key in ("model", "model_name", "llm_model", "model_id"):
            model = _coerce_str(source.get(key))
            if model:
                return model

        invocation_params = source.get("invocation_params") or source.get(
            "invocation_parameters"
        )
        model = _coerce_str(_extract_nested_mapping(invocation_params, "model"))
        if model:
            return model

    return None


def _extract_workflow_name(
    record: Mapping[str, Any],
    span_attributes: Mapping[str, Any],
    extra_attributes: Mapping[str, Any] | None,
) -> str | None:
    for source in (record.get("metadata"), span_attributes):
        if not isinstance(source, Mapping):
            continue
        workflow_name = _coerce_str(source.get("workflow_name"))
        if workflow_name:
            return workflow_name

    if extra_attributes is not None:
        workflow_name = _coerce_str(extra_attributes.get(RESPAN_TRACE_GROUP_ID))
        if workflow_name:
            return workflow_name

    return None


def _read_tokens(source: Any) -> tuple[int | None, int | None]:
    if not isinstance(source, Mapping):
        return None, None

    prompt = _coerce_int(source.get("prompt_tokens"))
    completion = _coerce_int(source.get("completion_tokens"))
    if prompt is None and completion is None:
        prompt = _coerce_int(source.get("input_tokens"))
        completion = _coerce_int(source.get("output_tokens"))
    return prompt, completion


def _extract_token_usage(record: Mapping[str, Any]) -> tuple[int | None, int | None]:
    for source in (record.get("metrics"), record.get("metadata")):
        prompt_tokens, completion_tokens = _read_tokens(source)
        if prompt_tokens is not None or completion_tokens is not None:
            return prompt_tokens, completion_tokens
        if isinstance(source, Mapping):
            for nested_key in ("usage", "tokens", "token_usage"):
                prompt_tokens, completion_tokens = _read_tokens(source.get(nested_key))
                if prompt_tokens is not None or completion_tokens is not None:
                    return prompt_tokens, completion_tokens
    return None, None


def _normalize_messages(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]
        return _normalize_messages(parsed)

    if isinstance(value, Mapping):
        messages = value.get("messages")
        if messages is not None:
            return _normalize_messages(messages)
        if "role" in value or "content" in value:
            return [dict(value)]
        if "input" in value:
            return _normalize_messages(value["input"])

    if isinstance(value, list):
        messages = [dict(item) for item in value if isinstance(item, Mapping)]
        if messages:
            return messages

    return None


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_sanitize_json(serialize_value(value)), default=str)


def _set_prompt_attributes(attrs: dict[str, Any], input_value: Any) -> None:
    messages = _normalize_messages(input_value)
    if not messages:
        if input_value is not None:
            attrs[f"{_GEN_AI_PROMPT_PREFIX}0.role"] = "user"
            attrs[f"{_GEN_AI_PROMPT_PREFIX}0.content"] = _message_content(input_value)
        return

    for index, message in enumerate(messages):
        prefix = f"{_GEN_AI_PROMPT_PREFIX}{index}."
        role = _coerce_str(message.get("role")) or "user"
        attrs[f"{prefix}role"] = role
        if message.get("content") is not None:
            attrs[f"{prefix}content"] = _message_content(message.get("content"))
        if message.get("tool_calls") is not None:
            attrs[f"{prefix}tool_calls"] = json.dumps(
                _sanitize_json(serialize_value(message.get("tool_calls"))),
                default=str,
            )


def _extract_completion_message(output_value: Any) -> dict[str, Any] | None:
    if output_value is None:
        return None
    if isinstance(output_value, str):
        try:
            parsed = json.loads(output_value)
        except json.JSONDecodeError:
            return {"role": "assistant", "content": output_value}
        return _extract_completion_message(parsed)
    if isinstance(output_value, Mapping):
        if "role" in output_value or "content" in output_value or "tool_calls" in output_value:
            return dict(output_value)
        choices = output_value.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, Mapping):
                message = first_choice.get("message") or first_choice.get("delta")
                if isinstance(message, Mapping):
                    return dict(message)
                if first_choice.get("text") is not None:
                    return {"role": "assistant", "content": first_choice.get("text")}
        if output_value.get("output") is not None:
            return _extract_completion_message(output_value.get("output"))
    return {"role": "assistant", "content": _message_content(output_value)}


def _set_completion_attributes(attrs: dict[str, Any], output_value: Any) -> None:
    message = _extract_completion_message(output_value)
    if message is None:
        return

    attrs[f"{_GEN_AI_COMPLETION_PREFIX}0.role"] = (
        _coerce_str(message.get("role")) or "assistant"
    )
    if message.get("content") is not None:
        attrs[f"{_GEN_AI_COMPLETION_PREFIX}0.content"] = _message_content(
            message.get("content")
        )
    if message.get("tool_calls") is not None:
        attrs[f"{_GEN_AI_COMPLETION_PREFIX}0.tool_calls"] = json.dumps(
            _sanitize_json(serialize_value(message.get("tool_calls"))),
            default=str,
        )


def _build_metadata(record: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    base_metadata = record.get("metadata")
    if isinstance(base_metadata, Mapping):
        metadata.update(base_metadata)
    elif base_metadata is not None:
        metadata[f"{BRAINTRUST_METADATA_PREFIX}metadata"] = base_metadata

    for source_key, metadata_key in (
        ("tags", "tags"),
        ("scores", "scores"),
        ("metrics", "metrics"),
        ("span_attributes", "span_attributes"),
        ("context", "context"),
    ):
        if record.get(source_key) is not None:
            metadata[f"{BRAINTRUST_METADATA_PREFIX}{metadata_key}"] = record.get(
                source_key
            )

    if record.get("id") is not None:
        metadata[f"{BRAINTRUST_METADATA_PREFIX}log_id"] = _format_id(record.get("id"))

    for field in ("project_id", "experiment_id", "dataset_id", "org_id"):
        if record.get(field) is not None:
            metadata[f"{BRAINTRUST_METADATA_PREFIX}{field}"] = _format_id(
                record.get(field)
            )

    if not metadata:
        return None
    return _sanitize_json(metadata)


def _record_mapping(item: Any) -> Mapping[str, Any] | None:
    if isinstance(item, Mapping):
        return item
    get_method = getattr(item, "get", None)
    if callable(get_method):
        record = get_method()
        if isinstance(record, Mapping):
            return record
    return None


def _build_span_from_record(
    record: Mapping[str, Any],
    *,
    extra_attributes: Mapping[str, Any] | None = None,
    masking_function: Callable[[Any], Any] | None = None,
) -> ReadableSpan:
    span_attributes = record.get("span_attributes")
    if not isinstance(span_attributes, Mapping):
        span_attributes = {}

    span_type = _coerce_str(span_attributes.get("type"))
    normalized_span_type = span_type.lower() if span_type else ""
    log_type = BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE.get(
        normalized_span_type,
        LOG_TYPE_TASK,
    )

    span_parents = record.get("span_parents")
    parent_id = None
    if isinstance(span_parents, list | tuple) and span_parents:
        parent_id = _format_id(span_parents[0])

    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    start_time_ns = _timestamp_to_ns(metrics.get("start"))
    end_time_ns = _timestamp_to_ns(metrics.get("end"))

    span_name = (
        _coerce_str(span_attributes.get("name"))
        or _coerce_str(record.get("name"))
        or BRAINTRUST_DEFAULT_SPAN_NAME
    )
    input_value = record.get("input")
    output_value = record.get("output")
    metadata = _build_metadata(record)
    model = _extract_model(record, span_attributes)
    workflow_name = _extract_workflow_name(record, span_attributes, extra_attributes)
    prompt_tokens, completion_tokens = _extract_token_usage(record)
    total_tokens = (
        None
        if prompt_tokens is None and completion_tokens is None
        else (prompt_tokens or 0) + (completion_tokens or 0)
    )

    if masking_function is not None:
        input_value = _apply_masking(masking_function, input_value, "input")
        output_value = _apply_masking(masking_function, output_value, "output")
        metadata = _apply_masking(masking_function, metadata, "metadata")

    attrs: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: log_type,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH: (
            BRAINTRUST_ENTITY_PATH if parent_id else ""
        ),
    }

    input_string = _json_string(input_value)
    output_string = _json_string(output_value)
    if input_string is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = input_string
    if output_string is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output_string
    if metadata is not None:
        attrs[RESPAN_METADATA] = json.dumps(metadata, default=str)

    if log_type == LOG_TYPE_CHAT:
        attrs[LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        attrs[GEN_AI_SYSTEM] = "braintrust"
        _set_prompt_attributes(attrs, input_value)
        _set_completion_attributes(attrs, output_value)
        if model is not None:
            attrs[LLM_REQUEST_MODEL] = model
        if prompt_tokens is not None:
            attrs[LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
            attrs[_GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        if completion_tokens is not None:
            attrs[LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
            attrs[_GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        if total_tokens is not None:
            attrs[_LLM_USAGE_TOTAL_TOKENS] = total_tokens

    if extra_attributes:
        attrs.update(extra_attributes)
    if workflow_name is not None:
        attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name

    status_code = 500 if record.get("error") else 200
    error_message = _coerce_str(record.get("error"))

    return build_readable_span(
        name=span_name,
        trace_id=_format_id(record.get("root_span_id")),
        span_id=_format_id(record.get("span_id")),
        parent_id=parent_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        attributes=attrs,
        status_code=status_code,
        error_message=error_message,
    )


def _apply_masking(
    masking_function: Callable[[Any], Any],
    value: Any,
    field_name: str,
) -> Any:
    try:
        return masking_function(value)
    except Exception as exc:  # pragma: no cover - defensive
        return f"ERROR: Failed to mask field '{field_name}' - {type(exc).__name__}"


class BraintrustInstrumentor:
    """Respan instrumentor for Braintrust."""

    name = "braintrust"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: list[_BufferedBraintrustItem] = []
        self._previous_logger: Any | None = None
        self._masking_function: Callable[[Any], Any] | None = None
        self._is_instrumented = False
        self._braintrust: Any | None = None
        self._merge_row_batch: Callable[[Any], Any] | None = None
        self._extract_attachments: Callable[[Any, list[Any]], Any] | None = None

    def __enter__(self) -> "BraintrustInstrumentor":
        self.activate()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.deactivate()

    @property
    def is_instrumented(self) -> bool:
        return self._is_instrumented

    def activate(self) -> None:
        if self._is_instrumented:
            return

        try:
            import braintrust
        except ImportError:
            logger.warning("Failed to activate Braintrust instrumentation: braintrust is not installed")
            return

        self._braintrust = braintrust
        self._load_optional_braintrust_helpers()

        state = braintrust._internal_get_global_state()
        override_bg_logger = getattr(state, "_override_bg_logger", None)
        if override_bg_logger is None:
            logger.warning(
                "Failed to activate Braintrust instrumentation: unsupported Braintrust logger state"
            )
            return

        self._previous_logger = getattr(override_bg_logger, "logger", None)
        override_bg_logger.logger = self
        self._is_instrumented = True
        logger.info("Braintrust instrumentation activated")

    def deactivate(self) -> None:
        if not self._is_instrumented or self._braintrust is None:
            self._is_instrumented = False
            return

        try:
            state = self._braintrust._internal_get_global_state()
            override_bg_logger = getattr(state, "_override_bg_logger", None)
            if getattr(override_bg_logger, "logger", None) is self:
                override_bg_logger.logger = self._previous_logger
        except Exception:
            logger.exception("Failed to restore Braintrust logger")
        finally:
            self._previous_logger = None
            self._is_instrumented = False
            logger.info("Braintrust instrumentation deactivated")

    def enforce_queue_size_limit(self, enforce: bool) -> None:
        del enforce

    def set_masking_function(self, masking_function: Callable[[Any], Any] | None) -> None:
        self._masking_function = masking_function

    def log(self, *args: Any) -> None:
        propagated_attributes = read_propagated_attributes()
        with self._lock:
            self._buffer.extend(
                _BufferedBraintrustItem(
                    item=arg,
                    propagated_attributes=dict(propagated_attributes),
                )
                for arg in args
            )

    def flush(self, batch_size: int | None = None) -> None:
        del batch_size
        with self._lock:
            if not self._buffer:
                return
            items = self._buffer
            self._buffer = []

        records_with_attributes = [
            (record, buffered_item.propagated_attributes)
            for buffered_item in items
            if (record := _record_mapping(buffered_item.item)) is not None
        ]

        if self._merge_row_batch is not None:
            merged_records = self._merge_row_batch(
                [record for record, _attributes in records_with_attributes]
            )
            records_with_attributes = [
                (
                    record,
                    records_with_attributes[index][1]
                    if index < len(records_with_attributes)
                    else {},
                )
                for index, record in enumerate(merged_records)
            ]

        attachments: list[Any] = []
        for record, propagated_attributes in records_with_attributes:
            if self._extract_attachments is not None:
                self._extract_attachments(record, attachments)
            span = _build_span_from_record(
                record,
                extra_attributes=propagated_attributes,
                masking_function=self._masking_function,
            )
            if not inject_span(span):
                logger.warning("Failed to export Braintrust span %r", span.name)

    def _load_optional_braintrust_helpers(self) -> None:
        try:
            from braintrust.logger import _extract_attachments
        except (ImportError, AttributeError):
            self._extract_attachments = None
        else:
            self._extract_attachments = _extract_attachments

        try:
            from braintrust.merge_row_batch import merge_row_batch
        except (ImportError, AttributeError):
            self._merge_row_batch = None
        else:
            self._merge_row_batch = merge_row_batch


RespanBraintrustInstrumentor = BraintrustInstrumentor
