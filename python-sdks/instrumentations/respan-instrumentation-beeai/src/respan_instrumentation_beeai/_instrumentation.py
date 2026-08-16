"""BeeAI Framework instrumentation plugin for Respan."""

import importlib
import json
import logging
from threading import Lock
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues
from openinference.semconv.trace import SpanAttributes as OISpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

BEEAI_INSTRUMENTATION_NAME = "beeai"
OPENINFERENCE_BEEAI_MODULE = "openinference.instrumentation.beeai"
OPENINFERENCE_BEEAI_PROCESSOR_MODULE = (
    "openinference.instrumentation.beeai.processors.base"
)
OPENINFERENCE_BEEAI_SPAN_MODULE = "openinference.instrumentation.beeai._span"
_OFF_CONTRACT_ALIAS_KEYS = (
    RESPAN_SPAN_TOOLS,
    RESPAN_SPAN_TOOL_CALLS,
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
)
_GEN_AI_MESSAGE_PREFIXES = (
    f"{TLSpanAttributes.LLM_PROMPTS}.",
    f"{TLSpanAttributes.LLM_COMPLETIONS}.",
)
_TOOL_CALLS_SUFFIX = ".tool_calls"

_BEEAI_PATCH_LOCK = Lock()
_BEEAI_PATCH_REFCOUNT = 0
_BEEAI_PATCHES: list[tuple[type, str, Any, Any]] = []


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        next_error = getattr(current, "__cause__", None)
        if not isinstance(next_error, BaseException):
            next_error = getattr(current, "__context__", None)
        if not isinstance(next_error, BaseException):
            next_error = getattr(current, "_predecessor", None)
        current = next_error if isinstance(next_error, BaseException) else None
    return chain


def _exception_status_code(error: BaseException) -> int:
    for item in _exception_chain(error):
        for value in (
            getattr(item, "status_code", None),
            getattr(getattr(item, "response", None), "status_code", None),
            getattr(getattr(item, "response", None), "status", None),
        ):
            if isinstance(value, int) and value >= 400:
                return value
    return 500


def _exception_message(error: BaseException) -> str:
    explain = getattr(error, "explain", None)
    if callable(explain):
        try:
            message = explain()
        except Exception:
            message = None
        if isinstance(message, str) and message:
            return message
    return str(error) or type(error).__name__


def _event_error(value: Any) -> BaseException | None:
    if isinstance(value, BaseException):
        return value
    error = getattr(value, "error", None)
    return error if isinstance(error, BaseException) else None


def _record_active_parent_exception(error: BaseException) -> None:
    active_span = trace.get_current_span()
    is_recording = getattr(active_span, "is_recording", None)
    if not callable(is_recording) or not is_recording():
        return

    message = _exception_message(error)
    current_status = getattr(getattr(active_span, "status", None), "status_code", None)
    if current_status != StatusCode.ERROR:
        active_span.record_exception(error)
    active_span.set_status(Status(StatusCode.ERROR, message))
    active_span.set_attribute("status_code", _exception_status_code(error))
    active_span.set_attribute("error.message", message)


def _record_exception_wrapper(original: Any) -> Any:
    def record_exception(span: Any, error: BaseException) -> None:
        original(span, error)
        _record_active_parent_exception(error)
        attrs = getattr(span, "attributes", None)
        if not isinstance(attrs, dict):
            return
        # Error text is diagnostic data, not an assistant completion. Keeping
        # it in output causes platform token/cost estimation for failed calls.
        attrs.pop(OISpanAttributes.OUTPUT_VALUE, None)
        attrs.pop(OISpanAttributes.OUTPUT_MIME_TYPE, None)
        attrs["status_code"] = _exception_status_code(error)
        attrs["error.message"] = _exception_message(error)

    return record_exception


def _child_wrapper(original: Any) -> Any:
    def child(
        span: Any,
        name: str | None = None,
        event: tuple[Any, Any] | None = None,
    ) -> Any:
        error = _event_error(event[0]) if event is not None else None
        if error is None:
            return original(span, name=name, event=event)

        meta = event[1]
        span.add_event(
            name or getattr(meta, "name", None) or "error",
            {"error.message": _exception_message(error)},
            getattr(meta, "created_at", None),
        )
        span.record_exception(error)
        # Error events describe the owning operation; they are not another
        # model invocation. Returning the owner also supports upstream callers
        # that immediately call record_exception() on the child result.
        return span

    return child


def _end_wrapper(original: Any) -> Any:
    async def end(processor: Any, event: Any, meta: Any) -> None:
        await original(processor, event, meta)
        output = getattr(event, "output", None)
        get_text_content = getattr(output, "get_text_content", None)
        if (
            getattr(processor.span, "kind", None) == OpenInferenceSpanKindValues.LLM
            and callable(get_text_content)
        ):
            content = get_text_content()
            if content:
                processor.span.set_attribute(
                    f"{OISpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.content",
                    content,
                )
        error = _event_error(event)
        if error is not None:
            # Upstream currently resets ERROR to OK when output is also set.
            processor.span.record_exception(error)

    return end


def _patch_beeai_processors() -> None:
    global _BEEAI_PATCH_REFCOUNT

    with _BEEAI_PATCH_LOCK:
        if _BEEAI_PATCH_REFCOUNT == 0:
            processor_module = importlib.import_module(
                OPENINFERENCE_BEEAI_PROCESSOR_MODULE
            )
            span_module = importlib.import_module(OPENINFERENCE_BEEAI_SPAN_MODULE)
            patch_specs = (
                (
                    span_module.SpanWrapper,
                    "record_exception",
                    _record_exception_wrapper,
                ),
                (span_module.SpanWrapper, "child", _child_wrapper),
                (processor_module.Processor, "end", _end_wrapper),
            )
            for owner, attribute, wrapper_factory in patch_specs:
                original = getattr(owner, attribute)
                patched = wrapper_factory(original)
                setattr(owner, attribute, patched)
                _BEEAI_PATCHES.append((owner, attribute, original, patched))
        _BEEAI_PATCH_REFCOUNT += 1


def _unpatch_beeai_processors() -> None:
    global _BEEAI_PATCH_REFCOUNT

    with _BEEAI_PATCH_LOCK:
        if _BEEAI_PATCH_REFCOUNT == 0:
            return
        _BEEAI_PATCH_REFCOUNT -= 1
        if _BEEAI_PATCH_REFCOUNT != 0:
            return
        for owner, attribute, original, patched in reversed(_BEEAI_PATCHES):
            if getattr(owner, attribute) is patched:
                setattr(owner, attribute, original)
        _BEEAI_PATCHES.clear()


def _load_openinference_beeai_class() -> type:
    beeai_module = importlib.import_module(OPENINFERENCE_BEEAI_MODULE)
    return beeai_module.BeeAIInstrumentor


def _is_beeai_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", None)
    return scope_name == OPENINFERENCE_BEEAI_MODULE


def _is_gen_ai_tool_calls_attr(key: str) -> bool:
    return key.endswith(_TOOL_CALLS_SUFFIX) and key.startswith(_GEN_AI_MESSAGE_PREFIXES)


def _safe_json_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class _BeeAIOffContractAliasProcessor(SpanProcessor):
    """Remove shared OpenInference aliases from BeeAI spans only."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if not _is_beeai_span(span):
            return

        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        for key in _OFF_CONTRACT_ALIAS_KEYS:
            attrs.pop(key, None)

        for key, value in list(attrs.items()):
            if _is_gen_ai_tool_calls_attr(key):
                attrs[key] = _safe_json_str(value)

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _active_span_processors() -> tuple[Any, Any]:
    tracer_provider = trace.get_tracer_provider()
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return active_span_processor, processors


class BeeAIInstrumentor:
    """Respan instrumentor for BeeAI Framework.

    Activates the OpenInference BeeAI instrumentor and registers Respan's
    OpenInference translator so BeeAI spans reach the Respan OTLP pipeline
    with the expected ``traceloop.*``, ``gen_ai.*``, and ``respan.*`` fields.
    """

    name = BEEAI_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._cleanup_processor = None
        self._processors_patched = False
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument BeeAI via OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "BeeAI instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            beeai_instrumentor_class = _load_openinference_beeai_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate BeeAI instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            _patch_beeai_processors()
            self._processors_patched = True
            self._delegate = OpenInferenceInstrumentor(
                beeai_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            self._register_cleanup_processor()
            self._is_instrumented = True
            logger.info("BeeAI instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up BeeAI instrumentation")
            self._delegate = None
            self._cleanup_processor = None
            if self._processors_patched:
                _unpatch_beeai_processors()
                self._processors_patched = False
            self._is_instrumented = False
            logger.exception("Failed to activate BeeAI instrumentation")

    def _register_cleanup_processor(self) -> None:
        translator_getter = getattr(OpenInferenceInstrumentor, "_get_translator", None)
        if translator_getter is None:
            return

        translator = translator_getter()
        active_span_processor, processors = _active_span_processors()
        if active_span_processor is None or processors is None:
            return

        cleanup_processor = _BeeAIOffContractAliasProcessor()
        rebuilt_processors = []
        inserted = False

        for processor in processors:
            if isinstance(processor, _BeeAIOffContractAliasProcessor):
                continue
            rebuilt_processors.append(processor)
            if processor is translator:
                rebuilt_processors.append(cleanup_processor)
                inserted = True

        if inserted:
            active_span_processor._span_processors = tuple(rebuilt_processors)
            self._cleanup_processor = cleanup_processor

    def _unregister_cleanup_processor(self) -> None:
        if self._cleanup_processor is None:
            return

        active_span_processor, processors = _active_span_processors()
        if active_span_processor is not None and processors is not None:
            active_span_processor._span_processors = tuple(
                processor
                for processor in processors
                if processor is not self._cleanup_processor
            )
        self._cleanup_processor = None

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        self._unregister_cleanup_processor()
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate BeeAI instrumentation")
        if self._processors_patched:
            _unpatch_beeai_processors()
            self._processors_patched = False
        self._delegate = None
        self._is_instrumented = False
        logger.info("BeeAI instrumentation deactivated")
