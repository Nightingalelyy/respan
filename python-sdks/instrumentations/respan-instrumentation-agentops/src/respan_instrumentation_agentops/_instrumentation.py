"""Lifecycle management for AgentOps decorator tracing."""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

from opentelemetry import trace

from respan_instrumentation_agentops._constants import AGENTOPS_INSTRUMENTATION_NAME
from respan_instrumentation_agentops._processor import AgentOpsSpanProcessor
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_REFCOUNT = 0
_PROCESSOR: AgentOpsSpanProcessor | None = None
_PROVIDER: Any = None
_AGENTOPS_CORE: Any = None
_AGENTOPS_PROVIDER_PROXY: Any = None
_OWNED_CORE_INITIALIZATION = False
_PREVIOUS_CORE_STATE: tuple[Any, ...] | None = None


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _active_span_processors(provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active = getattr(provider, "_active_span_processor", None)
    processors = getattr(active, "_span_processors", None) if active else None
    return active, processors


def _register_first(provider: Any, processor: AgentOpsSpanProcessor) -> None:
    active, processors = _active_span_processors(provider)
    if active is not None and processors is not None:
        remaining = tuple(item for item in processors if item is not processor)
        active._span_processors = (processor, *remaining)
    elif hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)


def _unregister(provider: Any, processor: AgentOpsSpanProcessor) -> None:
    active, processors = _active_span_processors(provider)
    if active is not None and processors is not None:
        active._span_processors = tuple(
            item for item in processors if item is not processor
        )


class _ProviderProxy:
    """Let AgentOps flush Respan spans without owning Respan provider shutdown."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def force_flush(self, *args: Any, **kwargs: Any) -> Any:
        force_flush = getattr(self._provider, "force_flush", None)
        if callable(force_flush):
            return force_flush(*args, **kwargs)
        return True

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _enable_agentops_core(provider: Any) -> None:
    global _AGENTOPS_CORE, _AGENTOPS_PROVIDER_PROXY
    global _OWNED_CORE_INITIALIZATION, _PREVIOUS_CORE_STATE

    core_module = importlib.import_module("agentops.sdk.core")
    agentops_core = core_module.tracer
    _AGENTOPS_CORE = agentops_core
    if bool(getattr(agentops_core, "initialized", False)):
        return

    _PREVIOUS_CORE_STATE = (
        getattr(agentops_core, "_initialized", False),
        getattr(agentops_core, "provider", None),
        getattr(agentops_core, "_meter_provider", None),
    )
    _AGENTOPS_PROVIDER_PROXY = _ProviderProxy(provider)
    agentops_core.provider = _AGENTOPS_PROVIDER_PROXY
    agentops_core._meter_provider = None
    agentops_core._initialized = True
    _OWNED_CORE_INITIALIZATION = True


def _restore_agentops_core() -> None:
    global _AGENTOPS_CORE, _AGENTOPS_PROVIDER_PROXY
    global _OWNED_CORE_INITIALIZATION, _PREVIOUS_CORE_STATE

    if (
        _OWNED_CORE_INITIALIZATION
        and _AGENTOPS_CORE is not None
        and _PREVIOUS_CORE_STATE is not None
        and getattr(_AGENTOPS_CORE, "provider", None) is _AGENTOPS_PROVIDER_PROXY
    ):
        initialized, provider, meter_provider = _PREVIOUS_CORE_STATE
        _AGENTOPS_CORE._initialized = initialized
        _AGENTOPS_CORE.provider = provider
        _AGENTOPS_CORE._meter_provider = meter_provider
    _AGENTOPS_CORE = None
    _AGENTOPS_PROVIDER_PROXY = None
    _OWNED_CORE_INITIALIZATION = False
    _PREVIOUS_CORE_STATE = None


class AgentOpsInstrumentor:
    """Route AgentOps decorators into Respan and normalize their native attrs."""

    name = AGENTOPS_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        """Activate AgentOps decorator tracing without an AgentOps exporter."""
        global _PROCESSOR, _PROVIDER, _REFCOUNT

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("agentops")
        except ImportError as exc:
            logger.warning("AgentOps instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if _REFCOUNT == 0:
                _PROVIDER = trace.get_tracer_provider()
                _PROCESSOR = AgentOpsSpanProcessor(
                    capture_content=self._capture_content
                )
                _register_first(_PROVIDER, _PROCESSOR)
                _enable_agentops_core(_PROVIDER)
            elif _PROCESSOR is not None and (
                _PROCESSOR.capture_content != self._capture_content
            ):
                logger.warning(
                    "AgentOps is already active; the first capture_content setting wins"
                )
            _REFCOUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        """Remove Respan normalization and restore AgentOps core ownership."""
        global _PROCESSOR, _PROVIDER, _REFCOUNT

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _REFCOUNT = max(0, _REFCOUNT - 1)
            if _REFCOUNT:
                return
            if _PROCESSOR is not None and _PROVIDER is not None:
                _unregister(_PROVIDER, _PROCESSOR)
            _restore_agentops_core()
            _PROCESSOR = None
            _PROVIDER = None
