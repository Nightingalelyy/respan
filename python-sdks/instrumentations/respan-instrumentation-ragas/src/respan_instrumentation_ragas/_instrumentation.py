"""Ragas evaluation and experiment instrumentation."""

from __future__ import annotations

import contextvars
import functools
import importlib
import importlib.metadata
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes

from respan_instrumentation_ragas._serialization import (
    exception_message,
    json_string,
    safe_text,
    sensitive_key,
)

logger = logging.getLogger(__name__)

RAGAS_INSTRUMENTATION_NAME = "ragas"
_LOCK = threading.RLock()
_REFCOUNT = 0
_CAPTURE_CONTENT = True
_ENABLED = False
_EVALUATION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "respan_ragas_evaluation_depth", default=0
)
_METRIC_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "respan_ragas_metric_depth", default=0
)


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


_PATCHES: list[_Patch] = []


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _entity(kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if kind.startswith("metric"):
        metric = args[0] if args else None
        try:
            configured_name = getattr(metric, "name", None)
        except Exception:  # noqa: BLE001 - vendor objects are untrusted.
            configured_name = None
        name = safe_text(configured_name) or type(metric).__name__
        return f"{name}.batch" if kind == "metric_batch" else name
    if kind == "experiment_row":
        wrapper = args[0] if args else None
        try:
            name = getattr(wrapper, "__name__", "item")
        except Exception:  # noqa: BLE001
            name = "item"
        return f"ragas.experiment.row.{safe_text(name) or 'item'}"
    if kind == "experiment_run":
        wrapper = args[0] if args else None
        name = kwargs.get("name")
        try:
            wrapper_name = getattr(wrapper, "__name__", "run")
        except Exception:  # noqa: BLE001
            wrapper_name = "run"
        return f"ragas.experiment.{safe_text(name or wrapper_name) or 'run'}"
    return safe_text(kwargs.get("experiment_name")) or "ragas.evaluate"


def _input(kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    positional = (
        args[1:]
        if kind.startswith("metric") or kind in {"experiment_row", "experiment_run"}
        else args
    )
    return {"args": positional, "kwargs": kwargs}


def _prepare_span(
    span: Any,
    *,
    entity_name: str,
    kind: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    has_parent: bool,
) -> None:
    span.set_attribute(RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    span.set_attribute(RESPAN_LOG_TYPE, "task")
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_PATH,
        entity_name if has_parent else "",
    )
    if _CAPTURE_CONTENT:
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json_string(_input(kind, args, kwargs)),
        )
    for key, value in read_propagated_attributes().items():
        if key.startswith("respan."):
            if sensitive_key(key):
                serialized = "[REDACTED]"
            elif isinstance(value, str):
                serialized = safe_text(value)
            elif isinstance(value, bool | int | float):
                serialized = value
            else:
                serialized = json_string(value)
            span.set_attribute(key, serialized)


def _status_code(value: Any, *, default: int) -> int:
    try:
        response = getattr(value, "response", None)
    except Exception:  # noqa: BLE001 - exception objects are untrusted.
        response = None
    candidates = [value, response]
    for candidate in candidates:
        for name in ("status_code", "status"):
            try:
                code = getattr(candidate, name, None)
                if isinstance(code, int):
                    return code
            except Exception:  # noqa: BLE001,S112 - vendor status properties are untrusted.
                continue
    return default


def _finish_span(span: Any, result: Any) -> None:
    span.set_attribute("status_code", _status_code(result, default=200))
    if _CAPTURE_CONTENT:
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json_string(result))


def _mark_error(span: Any, exc: BaseException) -> None:
    status_code = _status_code(exc, default=500)
    if status_code < 400:
        status_code = 500
    message = exception_message(exc)
    span.set_attribute("status_code", status_code)
    span.set_attribute(ERROR_MESSAGE_ATTR, message)
    if _CAPTURE_CONTENT:
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json_string(
                {
                    "status": "error",
                    "error": type(exc).__name__,
                    "message": message,
                }
            ),
        )
    span.set_status(Status(StatusCode.ERROR, message))
    span.add_event(
        "exception",
        {
            "exception.type": f"{type(exc).__module__}.{type(exc).__name__}",
            "exception.message": message,
        },
    )


def _has_active_parent() -> bool:
    try:
        return bool(trace.get_current_span().get_span_context().is_valid)
    except Exception:  # noqa: BLE001 - non-recording contexts are root candidates.
        return False


def _tracer() -> Any:
    try:
        version = importlib.metadata.version("respan-instrumentation-ragas")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return trace.get_tracer(RAGAS_INSTRUMENTATION_NAME, version)


def _depth_for_kind(kind: str) -> contextvars.ContextVar[int] | None:
    if kind == "evaluation":
        return _EVALUATION_DEPTH
    if kind.startswith("metric"):
        return _METRIC_DEPTH
    return None


def _sync_wrapper(original: Callable[..., Any], *, kind: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _ENABLED:
            return original(*args, **kwargs)
        depth = _depth_for_kind(kind)
        if depth is not None and depth.get() > 0:
            return original(*args, **kwargs)
        token = depth.set(depth.get() + 1) if depth is not None else None
        entity_name = _entity(kind, args, kwargs)
        has_parent = _has_active_parent()
        try:
            tracer = _tracer()
            with tracer.start_as_current_span(
                f"{entity_name}.task",
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                _prepare_span(
                    span,
                    entity_name=entity_name,
                    kind=kind,
                    args=args,
                    kwargs=kwargs,
                    has_parent=has_parent,
                )
                try:
                    result = original(*args, **kwargs)
                except BaseException as exc:
                    _mark_error(span, exc)
                    raise
                _finish_span(span, result)
                return result
        finally:
            if token is not None and depth is not None:
                depth.reset(token)

    wrapper.__respan_ragas_wrapper__ = True  # type: ignore[attr-defined]
    return wrapper


def _async_wrapper(original: Callable[..., Any], *, kind: str) -> Callable[..., Any]:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _ENABLED:
            return await original(*args, **kwargs)
        depth = _depth_for_kind(kind)
        if depth is not None and depth.get() > 0:
            return await original(*args, **kwargs)
        token = depth.set(depth.get() + 1) if depth is not None else None
        entity_name = _entity(kind, args, kwargs)
        has_parent = _has_active_parent()
        try:
            tracer = _tracer()
            with tracer.start_as_current_span(
                f"{entity_name}.task",
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                _prepare_span(
                    span,
                    entity_name=entity_name,
                    kind=kind,
                    args=args,
                    kwargs=kwargs,
                    has_parent=has_parent,
                )
                try:
                    result = await original(*args, **kwargs)
                except BaseException as exc:
                    _mark_error(span, exc)
                    raise
                _finish_span(span, result)
                return result
        finally:
            if token is not None and depth is not None:
                depth.reset(token)

    wrapper.__respan_ragas_wrapper__ = True  # type: ignore[attr-defined]
    return wrapper


def _patch(owner: Any, name: str, *, kind: str) -> None:
    original = getattr(owner, name, None)
    if original is None or getattr(original, "__respan_ragas_wrapper__", False):
        return
    factory = _async_wrapper if inspect.iscoroutinefunction(original) else _sync_wrapper
    replacement = factory(original, kind=kind)
    setattr(owner, name, replacement)
    _PATCHES.append(_Patch(owner, name, original, replacement))


def _subclasses(base: type[Any]) -> set[type[Any]]:
    discovered: set[type[Any]] = set()
    pending = list(base.__subclasses__())
    while pending:
        candidate = pending.pop()
        if candidate in discovered:
            continue
        discovered.add(candidate)
        pending.extend(candidate.__subclasses__())
    return discovered


def _install_patches() -> None:
    ragas = importlib.import_module("ragas")
    evaluation = importlib.import_module("ragas.evaluation")
    metrics_base = importlib.import_module("ragas.metrics.base")
    experiment = importlib.import_module("ragas.experiment")

    for owner in (ragas, evaluation):
        _patch(owner, "evaluate", kind="evaluation")
        _patch(owner, "aevaluate", kind="evaluation")

    single_turn = metrics_base.SingleTurnMetric
    _patch(single_turn, "single_turn_score", kind="metric")
    _patch(single_turn, "single_turn_ascore", kind="metric")
    multi_turn = metrics_base.MultiTurnMetric
    _patch(multi_turn, "multi_turn_score", kind="metric")
    _patch(multi_turn, "multi_turn_ascore", kind="metric")
    importlib.import_module("ragas.metrics.collections")
    collections_base = importlib.import_module("ragas.metrics.collections.base")
    collection_metric = collections_base.BaseMetric
    _patch(collection_metric, "score", kind="metric")
    _patch(collection_metric, "batch_score", kind="metric_batch")
    _patch(collection_metric, "abatch_score", kind="metric_batch")
    for metric_class in _subclasses(collection_metric):
        if "ascore" in metric_class.__dict__:
            _patch(metric_class, "ascore", kind="metric")

    wrapper = experiment.ExperimentWrapper
    _patch(wrapper, "__call__", kind="experiment_row")
    _patch(wrapper, "arun", kind="experiment_run")


def _remove_patches() -> None:
    for patch in reversed(_PATCHES):
        if getattr(patch.owner, patch.name, None) is patch.replacement:
            setattr(patch.owner, patch.name, patch.original)
    _PATCHES.clear()


class RagasInstrumentor:
    """Instrument supported Ragas evaluation, metric, and experiment APIs."""

    name = RAGAS_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        global _CAPTURE_CONTENT, _ENABLED, _REFCOUNT

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("ragas")
        except ImportError as exc:
            logger.warning("Ragas instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if _REFCOUNT == 0:
                _CAPTURE_CONTENT = self._capture_content
                try:
                    _install_patches()
                except Exception:
                    _remove_patches()
                    raise
                _ENABLED = True
            elif _CAPTURE_CONTENT != self._capture_content:
                raise ValueError(
                    "all active RagasInstrumentor instances must use the same "
                    "capture_content setting"
                )
            _REFCOUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        global _ENABLED, _REFCOUNT

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT == 0:
                _ENABLED = False
                _remove_patches()
