"""Respan instrumentor for LangChain-compatible callback managers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from respan_instrumentation_langchain._callback import (
    RespanCallbackHandler,
    _with_respan_callback,
    add_respan_callback,
)

logger = logging.getLogger(__name__)
_MISSING = object()


class LangChainInstrumentor:
    """Respan instrumentor for LangChain, LangGraph, and Langflow Python flows."""

    name = "langchain"

    def __init__(
        self,
        *,
        callback_handler: RespanCallbackHandler | None = None,
        include_content: bool = True,
        include_metadata: bool = True,
    ) -> None:
        self._handler = callback_handler or RespanCallbackHandler(
            include_content=include_content,
            include_metadata=include_metadata,
        )
        self._is_instrumented = False
        self._patched_manager_classes: list[tuple[type, Any]] = []
        self._patched_langgraph_functions: list[tuple[Any, str, Any]] = []

    @property
    def callback_handler(self) -> RespanCallbackHandler:
        """Return the callback handler installed by this instrumentor."""
        return self._handler

    def _patch_callback_manager(self, manager_cls: type) -> None:
        original_descriptor = manager_cls.__dict__.get("configure", _MISSING)
        if original_descriptor is _MISSING:
            original_callable = getattr(manager_cls, "configure")

            def _call_original(cls, inheritable_callbacks, local_callbacks, *args, **kwargs):
                return original_callable(
                    inheritable_callbacks,
                    local_callbacks,
                    *args,
                    **kwargs,
                )
        else:
            original_func = (
                original_descriptor.__func__
                if isinstance(original_descriptor, classmethod)
                else original_descriptor
            )

            def _call_original(cls, inheritable_callbacks, local_callbacks, *args, **kwargs):
                return original_func(
                    cls,
                    inheritable_callbacks,
                    local_callbacks,
                    *args,
                    **kwargs,
                )

        handler = self._handler

        def _patched_configure(
            cls,
            inheritable_callbacks=None,
            local_callbacks=None,
            *args,
            **kwargs,
        ):
            inheritable_callbacks = _with_respan_callback(
                inheritable_callbacks,
                handler,
            )
            return _call_original(
                cls,
                inheritable_callbacks,
                local_callbacks,
                *args,
                **kwargs,
            )

        setattr(manager_cls, "configure", classmethod(_patched_configure))
        self._patched_manager_classes.append((manager_cls, original_descriptor))

    def _patch_langchain(self) -> bool:
        try:
            from langchain_core.callbacks.manager import (
                AsyncCallbackManager,
                CallbackManager,
            )
        except ImportError as exc:
            logger.warning(
                "Failed to activate LangChain instrumentation — missing dependency: %s",
                exc,
            )
            return False

        self._patch_callback_manager(CallbackManager)
        self._patch_callback_manager(AsyncCallbackManager)
        return True

    def _patch_langgraph(self) -> None:
        try:
            import langgraph.callbacks as callbacks_module
        except ImportError:
            return

        for function_name in (
            "get_sync_graph_callback_manager_for_config",
            "get_async_graph_callback_manager_for_config",
        ):
            original = getattr(callbacks_module, function_name, None)
            if original is None:
                continue

            def _patched(config, *args, __original=original, **kwargs):
                if isinstance(config, Mapping):
                    config = add_respan_callback(config, self._handler)
                else:
                    config = add_respan_callback(handler=self._handler)
                return __original(config, *args, **kwargs)

            setattr(callbacks_module, function_name, _patched)
            self._patched_langgraph_functions.append(
                (callbacks_module, function_name, original)
            )

    def activate(self) -> None:
        """Patch LangChain callback manager configuration to include Respan."""
        if self._is_instrumented:
            return
        if not self._patch_langchain():
            return
        self._patch_langgraph()
        self._is_instrumented = True
        logger.info("LangChain instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched callback manager functions."""
        if not self._is_instrumented:
            return

        for manager_cls, original_descriptor in reversed(self._patched_manager_classes):
            if original_descriptor is _MISSING:
                delattr(manager_cls, "configure")
            else:
                setattr(manager_cls, "configure", original_descriptor)
        self._patched_manager_classes.clear()

        for module, function_name, original in reversed(self._patched_langgraph_functions):
            setattr(module, function_name, original)
        self._patched_langgraph_functions.clear()

        self._is_instrumented = False
        logger.info("LangChain instrumentation deactivated")
