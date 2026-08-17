"""Lifecycle for Portkey's OpenInference adapter and Respan contract layer."""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

from opentelemetry import trace
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_instrumentation_openinference._translator import OpenInferenceTranslator
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_portkey._constants import (
    OPENINFERENCE_PORTKEY_MODULE,
    PORTKEY_INSTRUMENTATION_NAME,
)
from respan_instrumentation_portkey._processor import PortkeySpanContractProcessor
from respan_instrumentation_portkey._streaming import (
    StreamHooks,
    install_stream_hooks,
    remove_stream_hooks,
)

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_REFCOUNT = 0
_CONFIG: dict[str, Any] | None = None
_DELEGATE: OpenInferenceInstrumentor | None = None
_PROCESSOR: PortkeySpanContractProcessor | None = None
_PROVIDER: Any = None
_STREAM_HOOKS: StreamHooks | None = None


def _load_openinference_portkey_class() -> type:
    return importlib.import_module(OPENINFERENCE_PORTKEY_MODULE).PortkeyInstrumentor


def _processors(provider: Any) -> tuple[Any, ...] | None:
    active = getattr(provider, "_active_span_processor", None)
    current = getattr(active, "_span_processors", None) if active else None
    return tuple(current) if current is not None else None


def _set_processors(provider: Any, processors: tuple[Any, ...]) -> bool:
    active = getattr(provider, "_active_span_processor", None)
    if active is None or getattr(active, "_span_processors", None) is None:
        return False
    active._span_processors = processors
    return True


def _register_after_translator(provider: Any, processor: Any) -> None:
    processors = _processors(provider)
    if processors is None:
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(processor)
        return
    remaining = tuple(item for item in processors if item is not processor)
    for index, item in enumerate(remaining):
        if isinstance(item, OpenInferenceTranslator):
            _set_processors(
                provider,
                (*remaining[: index + 1], processor, *remaining[index + 1 :]),
            )
            return
    _set_processors(provider, (*remaining, processor))


def _unregister(provider: Any, processor: Any) -> None:
    processors = _processors(provider)
    if processors is not None:
        _set_processors(
            provider, tuple(item for item in processors if item is not processor)
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
        except Exception:  # noqa: BLE001
            return False
        return False
    return True


def _tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    return tracer is None or bool(getattr(tracer, "is_enabled", True))


class PortkeyInstrumentor:
    """Activate one shared Portkey runtime per process."""

    name = PORTKEY_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._delegate: OpenInferenceInstrumentor | None = None
        self._contract_processor: PortkeySpanContractProcessor | None = None
        self._is_instrumented = False

    def activate(self) -> None:
        global _CONFIG, _DELEGATE, _PROCESSOR, _PROVIDER, _REFCOUNT, _STREAM_HOOKS

        if self._is_instrumented:
            return
        if not _tracing_enabled():
            logger.info(
                "Portkey instrumentation skipped because Respan tracing is disabled"
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
                        "Portkey instrumentation is already active with different settings"
                    )
                    return
                _REFCOUNT += 1
                self._delegate = _DELEGATE
                self._contract_processor = _PROCESSOR
                self._is_instrumented = True
                return
            try:
                instrumentor_class = _load_openinference_portkey_class()
            except ImportError as exc:
                logger.warning(
                    "Failed to activate Portkey instrumentation - missing dependency: %s",
                    exc,
                )
                return

            provider = trace.get_tracer_provider()
            delegate = OpenInferenceInstrumentor(
                instrumentor_class, **self._instrumentor_kwargs
            )
            processor = PortkeySpanContractProcessor()
            hooks: StreamHooks | None = None
            registered = False
            try:
                delegate.activate()
                _register_after_translator(provider, processor)
                registered = True
                hooks = install_stream_hooks(provider)
            except Exception:
                remove_stream_hooks(hooks)
                if registered:
                    _unregister(provider, processor)
                try:
                    delegate.deactivate()
                except Exception:
                    logger.exception("Failed to roll back Portkey instrumentation")
                logger.exception("Failed to activate Portkey instrumentation")
                return

            _CONFIG = dict(self._instrumentor_kwargs)
            _DELEGATE = delegate
            _PROCESSOR = processor
            _PROVIDER = provider
            _REFCOUNT = 1
            _STREAM_HOOKS = hooks
            self._delegate = delegate
            self._contract_processor = processor
            self._is_instrumented = True
            logger.info("Portkey instrumentation activated")

    def deactivate(self) -> None:
        global _CONFIG, _DELEGATE, _PROCESSOR, _PROVIDER, _REFCOUNT, _STREAM_HOOKS

        if not self._is_instrumented:
            return
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            self._delegate = None
            self._contract_processor = None
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            remove_stream_hooks(_STREAM_HOOKS)
            if _PROCESSOR is not None and _PROVIDER is not None:
                _unregister(_PROVIDER, _PROCESSOR)
            if _DELEGATE is not None:
                try:
                    _DELEGATE.deactivate()
                except Exception:
                    logger.exception("Failed to deactivate Portkey instrumentation")
            _CONFIG = None
            _DELEGATE = None
            _PROCESSOR = None
            _PROVIDER = None
            _STREAM_HOOKS = None
            logger.info("Portkey instrumentation deactivated")

    def instrument(self) -> None:
        self.activate()

    def uninstrument(self) -> None:
        self.deactivate()
