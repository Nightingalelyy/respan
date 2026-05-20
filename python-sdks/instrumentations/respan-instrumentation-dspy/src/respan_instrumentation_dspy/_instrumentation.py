"""DSPy instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from respan_instrumentation_dspy._callback import DSPyInstrumentationCallback
from respan_instrumentation_dspy._constants import DSPY_INSTRUMENTATION_NAME

logger = logging.getLogger(__name__)

_UNSET = object()


class DSPyInstrumentor:
    """Respan instrumentor for DSPy.

    Registers a DSPy callback that converts native DSPy module, LM, tool,
    adapter, and evaluation callbacks into canonical Respan spans.
    """

    name = DSPY_INSTRUMENTATION_NAME

    def __init__(
        self,
        target: Any | None = None,
        *,
        include_content: bool = True,
    ) -> None:
        self._target = target
        self._include_content = include_content
        self._callback: DSPyInstrumentationCallback | None = None
        self._dspy_module: Any = None
        self._is_instrumented = False
        self._target_had_callbacks = False

    @staticmethod
    def _append_callback(callbacks: Any, callback: DSPyInstrumentationCallback) -> list[Any]:
        if callbacks is None:
            return [callback]
        callback_list = list(callbacks)
        if callback not in callback_list:
            callback_list.append(callback)
        return callback_list

    @staticmethod
    def _remove_callback(callbacks: Any, callback: DSPyInstrumentationCallback) -> list[Any]:
        if callbacks is None:
            return []
        return [
            existing_callback
            for existing_callback in callbacks
            if existing_callback is not callback
        ]

    def activate(self) -> None:
        """Register the Respan DSPy callback globally or on a target object."""
        if self._is_instrumented:
            return

        try:
            dspy_module = importlib.import_module("dspy")
        except ImportError as exc:
            logger.warning(
                "Failed to activate DSPy instrumentation — missing dependency: %s",
                exc,
            )
            return

        self._dspy_module = dspy_module
        self._callback = DSPyInstrumentationCallback(
            include_content=self._include_content,
        )

        if self._target is None:
            self._activate_global()
        else:
            self._activate_target()

        self._is_instrumented = True
        logger.info("DSPy instrumentation activated")

    def _activate_global(self) -> None:
        current_callbacks = self._dspy_module.settings.get("callbacks", [])
        callbacks = self._append_callback(
            callbacks=current_callbacks,
            callback=self._callback,
        )
        self._dspy_module.configure(callbacks=callbacks)

    def _activate_target(self) -> None:
        current_callbacks = getattr(self._target, "callbacks", _UNSET)
        self._target_had_callbacks = current_callbacks is not _UNSET
        callbacks = self._append_callback(
            callbacks=None if current_callbacks is _UNSET else current_callbacks,
            callback=self._callback,
        )
        setattr(self._target, "callbacks", callbacks)

    def deactivate(self) -> None:
        """Remove the Respan DSPy callback."""
        if not self._is_instrumented or self._callback is None:
            return

        if self._target is None:
            self._deactivate_global()
        else:
            self._deactivate_target()

        self._callback = None
        self._is_instrumented = False
        logger.info("DSPy instrumentation deactivated")

    def _deactivate_global(self) -> None:
        current_callbacks = self._dspy_module.settings.get("callbacks", [])
        callbacks = self._remove_callback(
            callbacks=current_callbacks,
            callback=self._callback,
        )
        self._dspy_module.configure(callbacks=callbacks)

    def _deactivate_target(self) -> None:
        current_callbacks = getattr(self._target, "callbacks", [])
        callbacks = self._remove_callback(
            callbacks=current_callbacks,
            callback=self._callback,
        )
        if not self._target_had_callbacks and not callbacks:
            try:
                delattr(self._target, "callbacks")
            except AttributeError:
                pass
            return
        setattr(self._target, "callbacks", callbacks)


DspyInstrumentor = DSPyInstrumentor
