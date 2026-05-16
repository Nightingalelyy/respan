"""Guardrails AI instrumentation plugin for Respan."""

import ast
import functools
import importlib
import json
import logging
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode

from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_GUARDRAIL
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

GUARDRAILS_INSTRUMENTATION_NAME = "guardrails"
GUARDRAILS_RUNTIME_MODULE = "guardrails"
GUARDRAILS_GUARD_CLASS = "Guard"

_GUARD_METHODS = ("__call__", "parse", "validate")
_METHOD_LABELS = {
    "__call__": "call",
    "parse": "parse",
    "validate": "validate",
}
_RESPAN_WRAPPED_ATTRIBUTE = "_respan_guardrails_wrapped"
_GUARDRAILS_SPAN_TYPE = "type"
_GUARDRAILS_SPAN_TYPE_PREFIX = "guardrails/"
_GUARDRAILS_LLM_CALL_TYPE = "guardrails/guard/step/call"
_GUARDRAILS_INPUT_VALUE = "input.value"
_GUARDRAILS_OUTPUT_VALUE = "output.value"
_GUARDRAILS_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
_GUARDRAILS_LLM_INPUT_MESSAGES_PREFIX = "llm.input_messages."
_GUARDRAILS_LLM_OUTPUT_MESSAGES_PREFIX = "llm.output_messages."
_GUARDRAILS_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
_GUARDRAILS_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
_GUARDRAILS_LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
_GEN_AI_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_LLM_USAGE_TOTAL_TOKENS = SpanAttributes.LLM_USAGE_TOTAL_TOKENS


def _load_guardrails_guard_class() -> type:
    guardrails_module = importlib.import_module(GUARDRAILS_RUNTIME_MODULE)
    return getattr(guardrails_module, GUARDRAILS_GUARD_CLASS)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {
            str(item_key): _json_safe(item_value)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return repr(value)


def _json_string(value: Any) -> str:
    return json.dumps(_json_safe(value), default=repr)


def _parse_invocation_parameters(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or value == "":
        return {}

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}

    return parsed if isinstance(parsed, dict) else {}


def _int_value(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _translate_guardrails_message_attrs(
    attrs: dict[str, Any],
    source_prefix: str,
    target_prefix: str,
) -> None:
    for key, value in list(attrs.items()):
        if not key.startswith(source_prefix):
            continue

        suffix = key[len(source_prefix):]
        parts = suffix.split(".", 2)
        if len(parts) != 3 or not parts[0].isdigit() or parts[1] != "message":
            continue

        field_name = parts[2]
        if field_name not in {"role", "content"}:
            continue

        attrs.setdefault(f"{target_prefix}{parts[0]}.{field_name}", value)


def _has_guardrails_llm_attrs(attrs: dict[str, Any]) -> bool:
    if attrs.get(_GUARDRAILS_LLM_INVOCATION_PARAMETERS):
        return True
    if attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_PROMPT) is not None:
        return True
    if attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_COMPLETION) is not None:
        return True
    if attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_TOTAL) is not None:
        return True
    return any(
        key.startswith(_GUARDRAILS_LLM_INPUT_MESSAGES_PREFIX)
        or key.startswith(_GUARDRAILS_LLM_OUTPUT_MESSAGES_PREFIX)
        for key in attrs
    )


def _result_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"result": repr(result)}
    for attribute_name in (
        "validation_passed",
        "validated_output",
        "raw_llm_output",
        "error",
    ):
        if hasattr(result, attribute_name):
            payload[attribute_name] = getattr(result, attribute_name)
    return payload


def _span_input_payload(
    method_label: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": method_label,
        "args": [_json_safe(argument) for argument in args],
        "kwargs": {
            str(argument_name): _json_safe(argument_value)
            for argument_name, argument_value in kwargs.items()
        },
    }


def _set_guardrail_span_attributes(
    span: trace.Span,
    span_name: str,
    input_payload: dict[str, Any],
) -> None:
    span.set_attribute(RESPAN_LOG_TYPE, LOG_TYPE_GUARDRAIL)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, "")
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, _json_string(input_payload))


def _build_guard_method_wrapper(
    method_name: str,
    original_method: Callable[..., Any],
) -> Callable[..., Any]:
    method_label = _METHOD_LABELS[method_name]
    span_name = f"guardrails.{method_label}"

    @functools.wraps(original_method)
    def wrapped_method(guard_instance: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = trace.get_tracer(__name__)
        input_payload = _span_input_payload(
            method_label=method_label,
            args=args,
            kwargs=kwargs,
        )

        with tracer.start_as_current_span(span_name) as span:
            _set_guardrail_span_attributes(
                span=span,
                span_name=span_name,
                input_payload=input_payload,
            )
            try:
                result = original_method(guard_instance, *args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    _json_string(
                        {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                )
                raise

            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_string(_result_payload(result)),
            )
            return result

    setattr(wrapped_method, _RESPAN_WRAPPED_ATTRIBUTE, True)
    return wrapped_method


class GuardrailsSpanProcessor(SpanProcessor):
    """Normalize Guardrails internal OTEL spans for the Respan backend."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        guardrails_type = attrs.get(_GUARDRAILS_SPAN_TYPE)
        if not (
            isinstance(guardrails_type, str)
            and guardrails_type.startswith(_GUARDRAILS_SPAN_TYPE_PREFIX)
        ):
            return

        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            f"guardrails.{span.name}",
        )
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")

        input_value = attrs.get(_GUARDRAILS_INPUT_VALUE)
        if input_value is not None:
            attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_INPUT, str(input_value))

        output_value = attrs.get(_GUARDRAILS_OUTPUT_VALUE)
        if output_value is not None:
            attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(output_value))

        if guardrails_type != _GUARDRAILS_LLM_CALL_TYPE or not _has_guardrails_llm_attrs(
            attrs
        ):
            attrs.setdefault(RESPAN_LOG_TYPE, LOG_TYPE_GUARDRAIL)
            attrs.setdefault(SpanAttributes.TRACELOOP_SPAN_KIND, LOG_TYPE_GUARDRAIL)
            span._attributes = attrs
            return

        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_CHAT
        attrs.setdefault(SpanAttributes.TRACELOOP_SPAN_KIND, LOG_TYPE_CHAT)
        attrs.setdefault(LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)

        invocation_parameters = _parse_invocation_parameters(
            attrs.get(_GUARDRAILS_LLM_INVOCATION_PARAMETERS)
        )
        model = invocation_parameters.get("model")
        if model:
            attrs.setdefault(LLM_REQUEST_MODEL, model)

        temperature = invocation_parameters.get("temperature")
        if temperature is not None:
            attrs.setdefault(SpanAttributes.LLM_REQUEST_TEMPERATURE, temperature)

        prompt_tokens = _int_value(attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_PROMPT))
        if prompt_tokens is not None:
            attrs.setdefault(LLM_USAGE_PROMPT_TOKENS, prompt_tokens)

        completion_tokens = _int_value(
            attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_COMPLETION)
        )
        if completion_tokens is not None:
            attrs.setdefault(LLM_USAGE_COMPLETION_TOKENS, completion_tokens)

        total_tokens = _int_value(attrs.get(_GUARDRAILS_LLM_TOKEN_COUNT_TOTAL))
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if total_tokens is not None:
            attrs.setdefault(_LLM_USAGE_TOTAL_TOKENS, total_tokens)

        _translate_guardrails_message_attrs(
            attrs=attrs,
            source_prefix=_GUARDRAILS_LLM_INPUT_MESSAGES_PREFIX,
            target_prefix=_GEN_AI_PROMPT_PREFIX,
        )
        _translate_guardrails_message_attrs(
            attrs=attrs,
            source_prefix=_GUARDRAILS_LLM_OUTPUT_MESSAGES_PREFIX,
            target_prefix=_GEN_AI_COMPLETION_PREFIX,
        )

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class GuardrailsInstrumentor:
    """Respan instrumentor for Guardrails AI.

    Wraps the stable public ``Guard`` execution methods and emits Respan
    ``guardrail`` spans into the active OpenTelemetry pipeline.
    """

    name = GUARDRAILS_INSTRUMENTATION_NAME
    _span_processor = GuardrailsSpanProcessor()
    _span_processor_registered = False

    def __init__(self) -> None:
        self._guard_class: type | None = None
        self._original_methods: dict[str, Callable[..., Any]] = {}
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _instrument_guard_class(self, guard_class: type) -> None:
        for method_name in _GUARD_METHODS:
            original_method = getattr(guard_class, method_name, None)
            if original_method is None:
                continue
            if getattr(original_method, _RESPAN_WRAPPED_ATTRIBUTE, False):
                continue

            self._original_methods[method_name] = original_method
            setattr(
                guard_class,
                method_name,
                _build_guard_method_wrapper(
                    method_name=method_name,
                    original_method=original_method,
                ),
            )

    @classmethod
    def _register_span_processor(cls) -> None:
        tracer_provider = trace.get_tracer_provider()
        active_span_processor = getattr(
            tracer_provider,
            "_active_span_processor",
            None,
        )
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if processors is not None:
            active_span_processor._span_processors = (
                cls._span_processor,
                *(
                    processor
                    for processor in processors
                    if processor is not cls._span_processor
                ),
            )
            cls._span_processor_registered = True
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(cls._span_processor)
            cls._span_processor_registered = True

    @classmethod
    def _remove_span_processor(cls) -> None:
        tracer_provider = trace.get_tracer_provider()
        active_span_processor = getattr(
            tracer_provider,
            "_active_span_processor",
            None,
        )
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is not None:
            active_span_processor._span_processors = tuple(
                processor
                for processor in processors
                if processor is not cls._span_processor
            )
        cls._span_processor_registered = False

    def activate(self) -> None:
        """Instrument Guardrails public Guard methods."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Guardrails instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            guard_class = _load_guardrails_guard_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Guardrails instrumentation — missing runtime dependency: %s",
                exc,
            )
            return

        self._guard_class = guard_class
        self._register_span_processor()
        self._instrument_guard_class(guard_class=guard_class)
        self._is_instrumented = True
        logger.info("Guardrails instrumentation activated")

    def deactivate(self) -> None:
        """Restore original Guardrails methods."""
        if self._guard_class is not None:
            for method_name, original_method in self._original_methods.items():
                current_method = getattr(self._guard_class, method_name, None)
                if getattr(current_method, _RESPAN_WRAPPED_ATTRIBUTE, False):
                    setattr(self._guard_class, method_name, original_method)

        self._guard_class = None
        self._original_methods.clear()
        self._remove_span_processor()
        self._is_instrumented = False
        logger.info("Guardrails instrumentation deactivated")
