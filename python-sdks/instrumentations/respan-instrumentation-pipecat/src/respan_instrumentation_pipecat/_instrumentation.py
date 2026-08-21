"""Pipecat instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_pipecat._observer_hooks import (
    ObserverHook,
    install_observer_hook,
    remove_observer_hook,
)
from respan_instrumentation_pipecat._translator import PipecatOpenInferenceTranslator

logger = logging.getLogger(__name__)

PIPECAT_INSTRUMENTATION_NAME = "pipecat"
OPENINFERENCE_PIPECAT_MODULE = "openinference.instrumentation.pipecat"
OPENINFERENCE_PIPECAT_OBSERVER_MODULE = (
    "openinference.instrumentation.pipecat._observer"
)

_LOCK = threading.RLock()
_REFCOUNT = 0
_PROCESSOR: PipecatOpenInferenceTranslator | None = None
_PROVIDER: Any = None
_UPSTREAM: Any = None
_OBSERVER_MODULE: Any = None
_OBSERVER_CONTEXT_ORIGINAL: Any = None
_OBSERVER_HOOK: ObserverHook | None = None
_CONFIG: dict[str, Any] | None = None


def _load_openinference_pipecat_class() -> type:
    pipecat_module = importlib.import_module(OPENINFERENCE_PIPECAT_MODULE)
    return pipecat_module.PipecatInstrumentor


def _load_openinference_pipecat_observer_module() -> Any:
    return importlib.import_module(OPENINFERENCE_PIPECAT_OBSERVER_MODULE)


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _active_processors(provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active = getattr(provider, "_active_span_processor", None)
    processors = getattr(active, "_span_processors", None) if active else None
    return active, processors


def _register_processor(
    provider: Any, processor: PipecatOpenInferenceTranslator
) -> None:
    active, processors = _active_processors(provider)
    if active is not None and processors is not None:
        remaining = tuple(
            existing for existing in processors if existing is not processor
        )
        active._span_processors = (processor, *remaining)
    elif hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)


def _unregister_processor(
    provider: Any, processor: PipecatOpenInferenceTranslator
) -> None:
    active, processors = _active_processors(provider)
    if active is not None and processors is not None:
        active._span_processors = tuple(
            existing for existing in processors if existing is not processor
        )


def _same_config(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key, value in left.items():
        if value is right[key]:
            continue
        try:
            if value == right[key]:
                continue
        except Exception:  # noqa: BLE001 - hostile configuration equality is rejected
            return False
        return False
    return True


def _patch_observer(observer_module: Any) -> tuple[Any, ObserverHook]:
    original_context = observer_module.Context
    observer_module.Context = context_api.get_current
    try:
        hook = install_observer_hook(observer_module)
    except Exception:
        if observer_module.Context is context_api.get_current:
            observer_module.Context = original_context
        raise
    return original_context, hook


def _restore_observer(
    observer_module: Any,
    original_context: Any,
    hook: ObserverHook | None,
) -> None:
    remove_observer_hook(hook)
    if (
        observer_module is not None
        and observer_module.Context is context_api.get_current
    ):
        observer_module.Context = original_context


class PipecatInstrumentor:
    """Activate OpenInference Pipecat once and normalize its spans for Respan."""

    name = PIPECAT_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._is_instrumented = False

    def activate(self) -> None:
        """Instrument Pipecat with shared, transactional lifecycle ownership."""
        global _CONFIG, _OBSERVER_CONTEXT_ORIGINAL, _OBSERVER_HOOK
        global _OBSERVER_MODULE, _PROCESSOR, _PROVIDER, _REFCOUNT, _UPSTREAM

        if self._is_instrumented:
            return
        if not _is_respan_tracing_enabled():
            logger.info(
                "Pipecat instrumentation skipped because Respan tracing is disabled"
            )
            return
        with _LOCK:
            if self._is_instrumented:
                return
            if _REFCOUNT:
                if _CONFIG is None or not _same_config(
                    _CONFIG, self._instrumentor_kwargs
                ):
                    logger.warning(
                        "Pipecat instrumentation is already active with different settings"
                    )
                    return
                _REFCOUNT += 1
                self._is_instrumented = True
                return

            try:
                instrumentor_class = _load_openinference_pipecat_class()
                observer_module = _load_openinference_pipecat_observer_module()
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Pipecat instrumentation — missing dependency: %s",
                    exc,
                )
                return

            provider = trace.get_tracer_provider()
            processor = PipecatOpenInferenceTranslator()
            upstream = instrumentor_class()
            original_context: Any = None
            hook: ObserverHook | None = None
            registered = False
            try:
                _register_processor(provider, processor)
                registered = True
                original_context, hook = _patch_observer(observer_module)
                upstream.instrument(
                    tracer_provider=provider,
                    **self._instrumentor_kwargs,
                )
            except Exception:
                try:
                    upstream.uninstrument()
                except Exception:
                    logger.exception("Failed to roll back Pipecat instrumentation")
                if hook is not None or original_context is not None:
                    _restore_observer(observer_module, original_context, hook)
                if registered:
                    _unregister_processor(provider, processor)
                logger.exception("Failed to activate Pipecat instrumentation")
                return

            _CONFIG = dict(self._instrumentor_kwargs)
            _OBSERVER_CONTEXT_ORIGINAL = original_context
            _OBSERVER_HOOK = hook
            _OBSERVER_MODULE = observer_module
            _PROCESSOR = processor
            _PROVIDER = provider
            _UPSTREAM = upstream
            _REFCOUNT = 1
            self._is_instrumented = True
            logger.info("Pipecat instrumentation activated")

    def deactivate(self) -> None:
        """Release one owner and remove only the final shared activation."""
        global _CONFIG, _OBSERVER_CONTEXT_ORIGINAL, _OBSERVER_HOOK
        global _OBSERVER_MODULE, _PROCESSOR, _PROVIDER, _REFCOUNT, _UPSTREAM

        if not self._is_instrumented:
            return
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            if _UPSTREAM is not None:
                try:
                    _UPSTREAM.uninstrument()
                except Exception:
                    logger.exception("Failed to deactivate Pipecat instrumentation")
            if _PROCESSOR is not None and _PROVIDER is not None:
                _unregister_processor(_PROVIDER, _PROCESSOR)
            _restore_observer(
                _OBSERVER_MODULE,
                _OBSERVER_CONTEXT_ORIGINAL,
                _OBSERVER_HOOK,
            )
            _CONFIG = None
            _OBSERVER_CONTEXT_ORIGINAL = None
            _OBSERVER_HOOK = None
            _OBSERVER_MODULE = None
            _PROCESSOR = None
            _PROVIDER = None
            _UPSTREAM = None
            logger.info("Pipecat instrumentation deactivated")

    def instrument(self) -> None:
        """Alias used by direct OpenTelemetry-style integrations."""
        self.activate()

    def uninstrument(self) -> None:
        """Alias used by direct OpenTelemetry-style integrations."""
        self.deactivate()
