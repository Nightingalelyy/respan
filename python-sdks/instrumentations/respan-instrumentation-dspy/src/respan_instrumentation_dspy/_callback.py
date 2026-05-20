"""DSPy callback that emits native Respan spans."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_dspy._constants import (
    ASSISTANT_ROLE,
    DSPY_ADAPTER_SPAN_NAME,
    DSPY_CALL_KIND_ADAPTER_FORMAT,
    DSPY_CALL_KIND_ADAPTER_PARSE,
    DSPY_CALL_KIND_EVALUATE,
    DSPY_CALL_KIND_LANGUAGE_MODEL,
    DSPY_CALL_KIND_MODULE,
    DSPY_CALL_KIND_TOOL,
    DSPY_EVALUATE_SPAN_NAME,
    DSPY_LANGUAGE_MODEL_SPAN_NAME,
    DSPY_MODULE_SPAN_NAME,
    DSPY_TOOL_SPAN_NAME,
)
from respan_instrumentation_dspy._utils import (
    add_lm_request_attributes,
    add_lm_usage_attributes,
    content_to_string,
    extract_first_completion,
    normalize_messages,
    output_to_json,
    output_to_plain_value,
    safe_json,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
    generate_unique_id,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

_MODULE_TASK_CLASS_NAMES = frozenset(
    {
        "ChainOfThought",
        "Predict",
        "ProgramOfThought",
        "Retry",
        "TypedPredictor",
    }
)
_MODULE_AGENT_CLASS_NAMES = frozenset({"ReAct"})


class _CallState:
    """Internal call state used to stitch DSPy nested callbacks into one trace."""

    def __init__(
        self,
        *,
        call_id: str,
        call_kind: str,
        instance: Any,
        inputs: Mapping[str, Any],
        started_at_ns: int,
        trace_id: str | None,
        span_id: str,
        parent_id: str | None,
        history_index: int | None = None,
    ) -> None:
        self.call_id = call_id
        self.call_kind = call_kind
        self.instance = instance
        self.inputs = inputs
        self.started_at_ns = started_at_ns
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.history_index = history_index


def _get_active_dspy_call_id() -> str | None:
    try:
        callback_module = importlib.import_module("dspy.utils.callback")
        active_call_id = getattr(callback_module, "ACTIVE_CALL_ID", None)
        if active_call_id is None:
            return None
        value = active_call_id.get()
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _get_current_otel_parent() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    try:
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0)
    span_id = getattr(span_context, "span_id", 0)
    if not isinstance(trace_id, int) or not isinstance(span_id, int):
        return None, None
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _module_display_name(instance: Any) -> str:
    class_name = type(instance).__name__
    stage = getattr(instance, "stage", None)
    if isinstance(stage, str) and stage:
        return f"{class_name}:{stage}"
    return class_name


def _tool_display_name(instance: Any) -> str:
    name = getattr(instance, "name", None)
    if isinstance(name, str) and name:
        return name
    function = getattr(instance, "func", None)
    function_name = getattr(function, "__name__", None)
    if isinstance(function_name, str) and function_name:
        return function_name
    return type(instance).__name__


def _adapter_display_name(instance: Any, call_kind: str) -> str:
    method_name = "format" if call_kind == DSPY_CALL_KIND_ADAPTER_FORMAT else "parse"
    return f"{type(instance).__name__}.{method_name}"


def _module_log_type(instance: Any) -> str:
    class_name = type(instance).__name__
    module_name = type(instance).__module__
    if class_name in _MODULE_AGENT_CLASS_NAMES:
        return LOG_TYPE_AGENT
    if class_name in _MODULE_TASK_CLASS_NAMES or module_name.startswith("dspy."):
        return LOG_TYPE_TASK
    return LOG_TYPE_AGENT


def _get_history_index(instance: Any) -> int | None:
    history = getattr(instance, "history", None)
    if isinstance(history, list):
        return len(history)
    return None


def _get_lm_history_entry(state: _CallState) -> Mapping[str, Any] | None:
    history = getattr(state.instance, "history", None)
    if not isinstance(history, list) or state.history_index is None:
        return None
    if len(history) <= state.history_index:
        return None
    history_entry = history[state.history_index]
    return history_entry if isinstance(history_entry, Mapping) else None


def _tool_input_value(*, name: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "arguments": _tool_arguments(inputs=inputs),
    }


def _tool_arguments(*, inputs: Mapping[str, Any]) -> Any:
    kwargs = inputs.get("kwargs")
    args = inputs.get("args")
    if isinstance(kwargs, Mapping):
        if args:
            return {
                "args": output_to_plain_value(value=args),
                "kwargs": output_to_plain_value(value=kwargs),
            }
        return output_to_plain_value(value=kwargs)
    return output_to_plain_value(value=inputs)


class DSPyInstrumentationCallback:
    """DSPy callback handler that emits Respan-compatible OTEL spans."""

    def __init__(self, *, include_content: bool = True) -> None:
        self._include_content = include_content
        self._active_calls: dict[str, _CallState] = {}
        self._lock = threading.RLock()

    def _start_call(
        self,
        *,
        call_id: str,
        call_kind: str,
        instance: Any,
        inputs: Mapping[str, Any],
    ) -> None:
        parent_call_id = _get_active_dspy_call_id()
        with self._lock:
            parent_state = (
                self._active_calls.get(parent_call_id)
                if parent_call_id is not None
                else None
            )
            if parent_state is not None:
                trace_id = parent_state.trace_id
                parent_id = parent_state.span_id
            else:
                trace_id, parent_id = _get_current_otel_parent()
                if trace_id is None:
                    trace_id = generate_unique_id()

            state = _CallState(
                call_id=call_id,
                call_kind=call_kind,
                instance=instance,
                inputs=dict(inputs),
                started_at_ns=time.time_ns(),
                trace_id=trace_id,
                span_id=generate_unique_id()[:16],
                parent_id=parent_id,
                history_index=(
                    _get_history_index(instance=instance)
                    if call_kind == DSPY_CALL_KIND_LANGUAGE_MODEL
                    else None
                ),
            )
            self._active_calls[call_id] = state

    def _end_call(
        self,
        *,
        call_id: str,
        outputs: Any,
        exception: Exception | None,
    ) -> None:
        with self._lock:
            state = self._active_calls.pop(call_id, None)

        if state is None:
            logger.debug("DSPy callback ended without start state for %s", call_id)
            return

        attributes = self._build_attributes(
            state=state,
            outputs=outputs,
            exception=exception,
        )
        span = build_readable_span(
            name=self._span_name(state=state),
            trace_id=state.trace_id,
            span_id=state.span_id,
            parent_id=state.parent_id,
            start_time_ns=state.started_at_ns,
            end_time_ns=time.time_ns(),
            attributes=attributes,
            status_code=500 if exception is not None else 200,
            error_message=str(exception) if exception is not None else None,
        )
        inject_span(span=span)

    def _base_attributes(
        self,
        *,
        log_type: str,
        entity_name: str,
        input_value: Any,
        output_value: Any,
    ) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
            RESPAN_LOG_TYPE: log_type,
            SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
            SpanAttributes.TRACELOOP_ENTITY_PATH: entity_name,
        }
        if self._include_content:
            attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
                value=input_value
            )
            attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output_to_json(
                value=output_value
            )
        return attributes

    def _build_attributes(
        self,
        *,
        state: _CallState,
        outputs: Any,
        exception: Exception | None,
    ) -> dict[str, Any]:
        if state.call_kind == DSPY_CALL_KIND_LANGUAGE_MODEL:
            return self._build_lm_attributes(
                state=state,
                outputs=outputs,
                exception=exception,
            )
        if state.call_kind == DSPY_CALL_KIND_TOOL:
            return self._build_tool_attributes(
                state=state,
                outputs=outputs,
                exception=exception,
            )
        if state.call_kind in {
            DSPY_CALL_KIND_ADAPTER_FORMAT,
            DSPY_CALL_KIND_ADAPTER_PARSE,
        }:
            return self._build_adapter_attributes(
                state=state,
                outputs=outputs,
                exception=exception,
            )
        if state.call_kind == DSPY_CALL_KIND_EVALUATE:
            return self._build_evaluate_attributes(
                state=state,
                outputs=outputs,
                exception=exception,
            )
        return self._build_module_attributes(
            state=state,
            outputs=outputs,
            exception=exception,
        )

    def _build_lm_attributes(
        self,
        *,
        state: _CallState,
        outputs: Any,
        exception: Exception | None,
    ) -> dict[str, Any]:
        prompt = state.inputs.get("prompt")
        messages = normalize_messages(
            prompt=prompt,
            messages=state.inputs.get("messages"),
        )
        output_value = str(exception) if exception is not None else outputs
        attributes = self._base_attributes(
            log_type=LOG_TYPE_CHAT,
            entity_name=DSPY_LANGUAGE_MODEL_SPAN_NAME,
            input_value=messages or prompt,
            output_value=output_value,
        )
        add_lm_request_attributes(
            attributes=attributes,
            instance=state.instance,
            inputs=state.inputs,
        )

        if self._include_content:
            for message_index, message in enumerate(messages):
                prompt_prefix = f"{SpanAttributes.LLM_PROMPTS}.{message_index}"
                attributes[f"{prompt_prefix}.role"] = str(message.get("role") or "user")
                attributes[f"{prompt_prefix}.content"] = content_to_string(
                    value=message.get("content")
                )

            completion_content = (
                str(exception)
                if exception is not None
                else extract_first_completion(outputs=outputs)
            )
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = ASSISTANT_ROLE
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = completion_content

        history_entry = _get_lm_history_entry(state=state)
        if history_entry is not None:
            usage = history_entry.get("usage")
            if isinstance(usage, Mapping):
                add_lm_usage_attributes(attributes=attributes, usage=usage)
        return attributes

    def _build_module_attributes(
        self,
        *,
        state: _CallState,
        outputs: Any,
        exception: Exception | None,
    ) -> dict[str, Any]:
        entity_name = _module_display_name(instance=state.instance)
        output_value = str(exception) if exception is not None else outputs
        return self._base_attributes(
            log_type=_module_log_type(instance=state.instance),
            entity_name=entity_name,
            input_value=state.inputs,
            output_value=output_value,
        )

    def _build_tool_attributes(
        self,
        *,
        state: _CallState,
        outputs: Any,
        exception: Exception | None,
    ) -> dict[str, Any]:
        entity_name = _tool_display_name(instance=state.instance)
        output_value = str(exception) if exception is not None else outputs
        return self._base_attributes(
            log_type=LOG_TYPE_TOOL,
            entity_name=entity_name,
            input_value=_tool_input_value(name=entity_name, inputs=state.inputs),
            output_value=output_value,
        )

    def _build_adapter_attributes(
        self,
        *,
        state: _CallState,
        outputs: Any,
        exception: Exception | None,
    ) -> dict[str, Any]:
        entity_name = _adapter_display_name(
            instance=state.instance,
            call_kind=state.call_kind,
        )
        output_value = str(exception) if exception is not None else outputs
        return self._base_attributes(
            log_type=LOG_TYPE_TASK,
            entity_name=entity_name,
            input_value=state.inputs,
            output_value=output_value,
        )

    def _build_evaluate_attributes(
        self,
        *,
        state: _CallState,
        outputs: Any,
        exception: Exception | None,
    ) -> dict[str, Any]:
        entity_name = type(state.instance).__name__
        output_value = str(exception) if exception is not None else outputs
        return self._base_attributes(
            log_type=LOG_TYPE_TASK,
            entity_name=entity_name,
            input_value=state.inputs,
            output_value=output_to_plain_value(value=output_value),
        )

    @staticmethod
    def _span_name(*, state: _CallState) -> str:
        if state.call_kind == DSPY_CALL_KIND_LANGUAGE_MODEL:
            return DSPY_LANGUAGE_MODEL_SPAN_NAME
        if state.call_kind == DSPY_CALL_KIND_TOOL:
            return DSPY_TOOL_SPAN_NAME
        if state.call_kind in {
            DSPY_CALL_KIND_ADAPTER_FORMAT,
            DSPY_CALL_KIND_ADAPTER_PARSE,
        }:
            return DSPY_ADAPTER_SPAN_NAME
        if state.call_kind == DSPY_CALL_KIND_EVALUATE:
            return DSPY_EVALUATE_SPAN_NAME
        return DSPY_MODULE_SPAN_NAME

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start_call(
            call_id=call_id,
            call_kind=DSPY_CALL_KIND_MODULE,
            instance=instance,
            inputs=inputs,
        )

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end_call(call_id=call_id, outputs=outputs, exception=exception)

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start_call(
            call_id=call_id,
            call_kind=DSPY_CALL_KIND_LANGUAGE_MODEL,
            instance=instance,
            inputs=inputs,
        )

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end_call(call_id=call_id, outputs=outputs, exception=exception)

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start_call(
            call_id=call_id,
            call_kind=DSPY_CALL_KIND_TOOL,
            instance=instance,
            inputs=inputs,
        )

    def on_tool_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end_call(call_id=call_id, outputs=outputs, exception=exception)

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start_call(
            call_id=call_id,
            call_kind=DSPY_CALL_KIND_ADAPTER_FORMAT,
            instance=instance,
            inputs=inputs,
        )

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end_call(call_id=call_id, outputs=outputs, exception=exception)

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start_call(
            call_id=call_id,
            call_kind=DSPY_CALL_KIND_ADAPTER_PARSE,
            instance=instance,
            inputs=inputs,
        )

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end_call(call_id=call_id, outputs=outputs, exception=exception)

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        self._start_call(
            call_id=call_id,
            call_kind=DSPY_CALL_KIND_EVALUATE,
            instance=instance,
            inputs=inputs,
        )

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        self._end_call(call_id=call_id, outputs=outputs, exception=exception)
