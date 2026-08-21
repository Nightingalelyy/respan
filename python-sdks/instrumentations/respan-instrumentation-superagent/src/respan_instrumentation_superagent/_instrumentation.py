"""Superagent safety-agent instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import inspect
import logging
import time
from collections.abc import Callable
from threading import RLock
from typing import Any

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_superagent._constants import (
    SAFETY_AGENT_CLIENT_MODULE,
    SAFETY_CLIENT_CLASS_NAME,
    SUPERAGENT_INSTRUMENTATION_NAME,
    SUPPORTED_METHODS,
)
from respan_instrumentation_superagent._span_emitter import emit_superagent_span

logger = logging.getLogger(__name__)

_PATCH_LOCK = RLock()
_PATCHED_CLASS: type[Any] | None = None
_ORIGINAL_METHODS: dict[str, Callable[..., Any]] = {}
_INSTALLED_METHODS: dict[str, Callable[..., Any]] = {}
_ACTIVE_INSTANCES = 0
_ACTIVE_METHODS: tuple[str, ...] | None = None
_PATCH_GENERATION = 0


def _load_safety_client_class() -> type[Any]:
    module = importlib.import_module(SAFETY_AGENT_CLIENT_MODULE)
    safety_client_class = getattr(module, SAFETY_CLIENT_CLASS_NAME, None)
    if safety_client_class is None:
        raise AttributeError(f"{SAFETY_AGENT_CLIENT_MODULE}.{SAFETY_CLIENT_CLASS_NAME}")
    return safety_client_class


def _wrap_method(
    method_name: str,
    original: Callable[..., Any],
    generation: int,
) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(original):

        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if _ACTIVE_INSTANCES == 0 or _PATCH_GENERATION != generation:
                return await original(self, *args, **kwargs)
            start_time_ns = time.time_ns()
            try:
                result = await original(self, *args, **kwargs)
            except Exception as exc:
                emit_superagent_span(
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                    result=None,
                    start_time_ns=start_time_ns,
                    end_time_ns=time.time_ns(),
                    error=exc,
                )
                raise

            emit_superagent_span(
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                result=result,
                start_time_ns=start_time_ns,
                end_time_ns=time.time_ns(),
            )
            return result

        return async_wrapper

    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _ACTIVE_INSTANCES == 0 or _PATCH_GENERATION != generation:
            return original(self, *args, **kwargs)
        start_time_ns = time.time_ns()
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            emit_superagent_span(
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                result=None,
                start_time_ns=start_time_ns,
                end_time_ns=time.time_ns(),
                error=exc,
            )
            raise

        emit_superagent_span(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result=result,
            start_time_ns=start_time_ns,
            end_time_ns=time.time_ns(),
        )
        return result

    return sync_wrapper


def _patch_safety_client(
    safety_client_class: type[Any],
    method_names: tuple[str, ...],
) -> bool:
    global _PATCHED_CLASS, _PATCH_GENERATION

    _PATCH_GENERATION += 1
    generation = _PATCH_GENERATION
    patched_any = False
    for method_name in method_names:
        original = getattr(safety_client_class, method_name, None)
        if original is None or not callable(original):
            logger.debug(
                "Skipping Superagent method %s; it is not available on %s",
                method_name,
                SAFETY_CLIENT_CLASS_NAME,
            )
            continue

        if method_name not in _ORIGINAL_METHODS:
            _ORIGINAL_METHODS[method_name] = original
            wrapper = _wrap_method(
                method_name=method_name,
                original=original,
                generation=generation,
            )
            _INSTALLED_METHODS[method_name] = wrapper
            setattr(
                safety_client_class,
                method_name,
                wrapper,
            )

        patched_any = True

    if patched_any:
        _PATCHED_CLASS = safety_client_class

    return patched_any


def _restore_safety_client() -> None:
    global _PATCHED_CLASS, _ACTIVE_METHODS

    if _PATCHED_CLASS is None:
        _ORIGINAL_METHODS.clear()
        _INSTALLED_METHODS.clear()
        _ACTIVE_METHODS = None
        return

    for method_name, original in _ORIGINAL_METHODS.items():
        if getattr(_PATCHED_CLASS, method_name, None) is _INSTALLED_METHODS.get(
            method_name
        ):
            setattr(_PATCHED_CLASS, method_name, original)

    _ORIGINAL_METHODS.clear()
    _INSTALLED_METHODS.clear()
    _PATCHED_CLASS = None
    _ACTIVE_METHODS = None


class SuperagentInstrumentor:
    """Respan instrumentor for the Superagent ``safety-agent`` SDK."""

    name = SUPERAGENT_INSTRUMENTATION_NAME

    def __init__(self, *, methods: tuple[str, ...] | None = None) -> None:
        self._methods = methods or SUPPORTED_METHODS
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Monkey-patch ``safety_agent.client.SafetyClient`` methods."""
        global _ACTIVE_INSTANCES, _ACTIVE_METHODS

        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Superagent instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            safety_client_class = _load_safety_client_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate Superagent instrumentation — missing dependency: %s",
                exc,
            )
            return

        with _PATCH_LOCK:
            normalized_methods = tuple(dict.fromkeys(self._methods))
            if _ACTIVE_INSTANCES:
                if _ACTIVE_METHODS != normalized_methods:
                    raise ValueError(
                        "Superagent instrumentation is already active with "
                        "different methods"
                    )
                _ACTIVE_INSTANCES += 1
                self._is_instrumented = True
                return
            if _patch_safety_client(safety_client_class, normalized_methods):
                _ACTIVE_METHODS = normalized_methods
                _ACTIVE_INSTANCES += 1
                self._is_instrumented = True
                logger.info("Superagent instrumentation activated")
            else:
                logger.warning(
                    "Failed to activate Superagent instrumentation — no compatible methods found"
                )

    def deactivate(self) -> None:
        """Restore the original Superagent client methods."""
        global _ACTIVE_INSTANCES

        if not self._is_instrumented:
            return

        with _PATCH_LOCK:
            _ACTIVE_INSTANCES = max(0, _ACTIVE_INSTANCES - 1)
            if _ACTIVE_INSTANCES == 0:
                _restore_safety_client()

        self._is_instrumented = False
        logger.info("Superagent instrumentation deactivated")
