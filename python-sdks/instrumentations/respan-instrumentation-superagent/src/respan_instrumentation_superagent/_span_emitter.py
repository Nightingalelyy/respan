"""Emit Superagent safety-agent calls as Respan OTEL spans."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_GUARDRAIL, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA_GUARDRAIL_NAME,
    RESPAN_METADATA_TRIGGERED,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_superagent._constants import (
    GUARD_METHOD,
    SUPERAGENT_INSTRUMENTATION_NAME,
    SUPERAGENT_METADATA_CLASSIFICATION,
    SUPERAGENT_METADATA_INTEGRATION,
    SUPERAGENT_METADATA_METHOD,
    SUPERAGENT_METADATA_MODEL,
    SUPERAGENT_METADATA_REDACT_FINDINGS,
    SUPERAGENT_METADATA_USAGE,
)
from respan_instrumentation_superagent._serialization import (
    extract_model,
    normalize_call_input,
    safe_error_message,
    safe_json_dumps,
    safe_text,
)

logger = logging.getLogger(__name__)


def _current_trace_context() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    if current_span is None:
        return None, None

    span_context = current_span.get_span_context()
    if not span_context or not span_context.is_valid:
        return None, None

    return (
        format_trace_id(span_context.trace_id),
        format_span_id(span_context.span_id),
    )


def _get_attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    try:
        return getattr(value, name, None)
    except BaseException:  # noqa: BLE001 - hostile SDK result objects must fail closed
        return None


def _operation_log_type(method_name: str) -> str:
    if method_name == GUARD_METHOD:
        return LOG_TYPE_GUARDRAIL
    return LOG_TYPE_TOOL


def _error_status_code(error: BaseException) -> int:
    try:
        direct = getattr(error, "status_code", None)
    except BaseException:  # noqa: BLE001 - hostile provider errors must fail closed
        direct = None
    if isinstance(direct, int) and 400 <= direct <= 599:
        return direct
    try:
        response = getattr(error, "response", None)
        nested = (
            getattr(response, "status_code", None) if response is not None else None
        )
    except BaseException:  # noqa: BLE001 - hostile provider errors must fail closed
        nested = None
    return nested if isinstance(nested, int) and 400 <= nested <= 599 else 500


def _add_result_metadata(attrs: dict[str, Any], method_name: str, result: Any) -> None:
    usage = _get_attr(result, "usage")
    if usage is not None:
        attrs[SUPERAGENT_METADATA_USAGE] = safe_json_dumps(usage)
    if method_name == GUARD_METHOD:
        classification = _get_attr(result, "classification")
        if isinstance(classification, str) and classification:
            attrs[SUPERAGENT_METADATA_CLASSIFICATION] = safe_text(classification)
            attrs[RESPAN_METADATA_TRIGGERED] = classification == "block"
        attrs[RESPAN_METADATA_GUARDRAIL_NAME] = "superagent.guard"
        return

    if method_name == "redact":
        findings = _get_attr(result, "findings")
        if findings is not None:
            attrs[SUPERAGENT_METADATA_REDACT_FINDINGS] = safe_json_dumps(findings)


def build_superagent_span_attributes(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Build canonical Respan attributes for a Superagent operation."""
    operation_name = f"superagent.{method_name}"
    model = extract_model(args=args, kwargs=kwargs)
    attrs: dict[str, Any] = {
        RESPAN_LOG_TYPE: _operation_log_type(method_name),
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: safe_json_dumps(
            normalize_call_input(
                method_name=method_name,
                args=args,
                kwargs=kwargs,
            )
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: safe_json_dumps(result),
        SUPERAGENT_METADATA_INTEGRATION: SUPERAGENT_INSTRUMENTATION_NAME,
        SUPERAGENT_METADATA_METHOD: method_name,
    }

    if model:
        attrs[SUPERAGENT_METADATA_MODEL] = model

    _add_result_metadata(attrs, method_name, result)
    return attrs


def emit_superagent_span(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    start_time_ns: int,
    end_time_ns: int,
    error: Exception | None = None,
) -> bool:
    """Emit one completed Superagent operation into the active OTEL pipeline."""
    try:
        trace_id, parent_id = _current_trace_context()
        attrs = build_superagent_span_attributes(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result=result,
        )

        if error is not None:
            message = safe_error_message(error)
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json_dumps(
                {"error": type(error).__name__, "message": message}
            )

        span = build_readable_span(
            name=f"superagent.{method_name}",
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes=attrs,
            status_code=_error_status_code(error) if error else 200,
            error_message=safe_error_message(error) if error else None,
        )
        return inject_span(span)
    except Exception:
        logger.debug("Failed to emit Superagent span", exc_info=True)
        return False
