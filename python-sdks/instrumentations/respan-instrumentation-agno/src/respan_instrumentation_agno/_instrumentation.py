"""Native Agno instrumentation plugin for Respan."""

import functools
import importlib
import inspect
import logging
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from types import MethodType
from typing import Any

from respan_instrumentation_agno._constants import (
    AGNO_AGENT_CLASS_NAME,
    AGNO_AGENT_MODULE,
    AGNO_INSTRUMENTATION_NAME,
    AGNO_TARGET_AGENT,
    AGNO_TARGET_TEAM,
    AGNO_TEAM_CLASS_NAME,
    AGNO_TEAM_MODULE,
    ARUN_METHOD_NAME,
    EVENT_KEY,
    INPUT_KEY,
    RESPAN_AGNO_ORIGINALS_ATTR,
    RESPAN_AGNO_WRAPPED_ATTR,
    RUN_OUTPUT_MARKER_KEYS,
    RUN_METHOD_NAME,
)
from respan_instrumentation_agno._otel_emitter import (
    create_agno_run_context,
    emit_agno_error,
    emit_agno_run,
    use_agno_run_context,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_input_value(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get(INPUT_KEY)


def _is_sync_stream_result(result: Any) -> bool:
    if isinstance(result, (str, bytes, bytearray, list, tuple, dict)):
        return False
    return isinstance(result, Iterator)


def _is_async_stream_result(result: Any) -> bool:
    if inspect.isawaitable(result):
        return False
    return hasattr(result, "__aiter__")


def _is_run_output(item: Any) -> bool:
    if _object_value(value=item, key=EVENT_KEY) is not None:
        return False
    return any(
        _object_value(value=item, key=key) is not None for key in RUN_OUTPUT_MARKER_KEYS
    )


def _last_run_output(items: list[Any]) -> Any | None:
    for item in reversed(items):
        if _is_run_output(item=item):
            return item
    return None


def _emit_completed_run(
    *,
    target: Any,
    target_kind: str,
    input_value: Any,
    output: Any | None,
    events: list[Any] | None,
    started_at_ns: int,
) -> None:
    try:
        emit_agno_run(
            target=target,
            target_kind=target_kind,
            input_value=input_value,
            output=output,
            events=events,
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
        )
    except Exception:
        logger.exception("Failed to emit Agno run spans")


def _emit_failed_run(
    *,
    target: Any,
    target_kind: str,
    input_value: Any,
    exception: Exception,
    started_at_ns: int,
) -> None:
    try:
        emit_agno_error(
            target=target,
            target_kind=target_kind,
            input_value=input_value,
            exception=exception,
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
        )
    except Exception:
        logger.exception("Failed to emit failed Agno run span")


def _wrap_sync_stream(
    *,
    iterator: Iterator[Any],
    target: Any,
    target_kind: str,
    input_value: Any,
    started_at_ns: int,
    run_context: Any,
) -> Iterator[Any]:
    items: list[Any] = []
    with use_agno_run_context(run_context=run_context):
        try:
            for item in iterator:
                items.append(item)
                yield item
        except Exception as exception:
            _emit_failed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                exception=exception,
                started_at_ns=started_at_ns,
            )
            raise
        else:
            _emit_completed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                output=_last_run_output(items=items),
                events=items,
                started_at_ns=started_at_ns,
            )


async def _wrap_async_stream(
    *,
    async_iterator: Any,
    target: Any,
    target_kind: str,
    input_value: Any,
    started_at_ns: int,
    run_context: Any,
) -> AsyncIterator[Any]:
    items: list[Any] = []
    with use_agno_run_context(run_context=run_context):
        try:
            async for item in async_iterator:
                items.append(item)
                yield item
        except Exception as exception:
            _emit_failed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                exception=exception,
                started_at_ns=started_at_ns,
            )
            raise
        else:
            _emit_completed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                output=_last_run_output(items=items),
                events=items,
                started_at_ns=started_at_ns,
            )


def _wrap_sync_method(
    *,
    original_method: Any,
    target_kind: str,
    is_bound_method: bool,
) -> Any:
    @functools.wraps(original_method)
    def wrapped_sync_method(*args: Any, **kwargs: Any) -> Any:
        target = args[0]
        call_args = args[1:] if is_bound_method else args
        input_args = args[1:] if is_bound_method else args[1:]
        input_value = _extract_input_value(args=input_args, kwargs=kwargs)
        started_at_ns = time.time_ns()
        run_context = create_agno_run_context(
            target_kind=target_kind,
            started_at_ns=started_at_ns,
        )

        with use_agno_run_context(run_context=run_context):
            try:
                result = original_method(*call_args, **kwargs)
            except Exception as exception:
                _emit_failed_run(
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    exception=exception,
                    started_at_ns=started_at_ns,
                )
                raise

            if _is_sync_stream_result(result=result):
                return _wrap_sync_stream(
                    iterator=result,
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    started_at_ns=started_at_ns,
                    run_context=run_context,
                )

            _emit_completed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                output=result,
                events=None,
                started_at_ns=started_at_ns,
            )
            return result

    setattr(wrapped_sync_method, RESPAN_AGNO_WRAPPED_ATTR, True)
    return wrapped_sync_method


def _wrap_async_method(
    *,
    original_method: Any,
    target_kind: str,
    is_bound_method: bool,
) -> Any:
    @functools.wraps(original_method)
    def wrapped_async_method(*args: Any, **kwargs: Any) -> Any:
        target = args[0]
        call_args = args[1:] if is_bound_method else args
        input_args = args[1:] if is_bound_method else args[1:]
        input_value = _extract_input_value(args=input_args, kwargs=kwargs)
        started_at_ns = time.time_ns()
        run_context = create_agno_run_context(
            target_kind=target_kind,
            started_at_ns=started_at_ns,
        )

        with use_agno_run_context(run_context=run_context):
            try:
                result = original_method(*call_args, **kwargs)
            except Exception as exception:
                _emit_failed_run(
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    exception=exception,
                    started_at_ns=started_at_ns,
                )
                raise

            if _is_async_stream_result(result=result):
                return _wrap_async_stream(
                    async_iterator=result,
                    target=target,
                    target_kind=target_kind,
                    input_value=input_value,
                    started_at_ns=started_at_ns,
                    run_context=run_context,
                )

            if inspect.isawaitable(result):

                async def await_and_emit() -> Any:
                    with use_agno_run_context(run_context=run_context):
                        try:
                            output = await result
                        except Exception as exception:
                            _emit_failed_run(
                                target=target,
                                target_kind=target_kind,
                                input_value=input_value,
                                exception=exception,
                                started_at_ns=started_at_ns,
                            )
                            raise

                        _emit_completed_run(
                            target=target,
                            target_kind=target_kind,
                            input_value=input_value,
                            output=output,
                            events=None,
                            started_at_ns=started_at_ns,
                        )
                        return output

                return await_and_emit()

            _emit_completed_run(
                target=target,
                target_kind=target_kind,
                input_value=input_value,
                output=result,
                events=None,
                started_at_ns=started_at_ns,
            )
            return result

    setattr(wrapped_async_method, RESPAN_AGNO_WRAPPED_ATTR, True)
    return wrapped_async_method


class AgnoInstrumentor:
    """Respan instrumentor for Agno.

    This native integration patches Agno's public run methods and emits
    Respan-compatible OTEL spans directly. It intentionally does not use
    ``openinference-instrumentation-agno``.
    """

    name = AGNO_INSTRUMENTATION_NAME

    def __init__(
        self,
        agent: Any | None = None,
        *,
        include_teams: bool = True,
    ) -> None:
        self._agent = agent
        self._include_teams = include_teams
        self._patches: list[tuple[Any, dict[str, Any]]] = []
        self._is_instrumented = False

    def _patch_target(
        self,
        *,
        target: Any,
        target_kind: str,
        is_bound_method: bool,
    ) -> bool:
        if getattr(target, RESPAN_AGNO_WRAPPED_ATTR, False):
            return False

        originals: dict[str, Any] = {}
        if hasattr(target, RUN_METHOD_NAME):
            original_run = getattr(target, RUN_METHOD_NAME)
            originals[RUN_METHOD_NAME] = original_run
            wrapped_run = _wrap_sync_method(
                original_method=original_run,
                target_kind=target_kind,
                is_bound_method=is_bound_method,
            )
            if is_bound_method:
                wrapped_run = MethodType(wrapped_run, target)
            setattr(target, RUN_METHOD_NAME, wrapped_run)

        if hasattr(target, ARUN_METHOD_NAME):
            original_arun = getattr(target, ARUN_METHOD_NAME)
            originals[ARUN_METHOD_NAME] = original_arun
            wrapped_arun = _wrap_async_method(
                original_method=original_arun,
                target_kind=target_kind,
                is_bound_method=is_bound_method,
            )
            if is_bound_method:
                wrapped_arun = MethodType(wrapped_arun, target)
            setattr(target, ARUN_METHOD_NAME, wrapped_arun)

        if not originals:
            return False

        setattr(target, RESPAN_AGNO_ORIGINALS_ATTR, originals)
        setattr(target, RESPAN_AGNO_WRAPPED_ATTR, True)
        self._patches.append((target, originals))
        return True

    @staticmethod
    def _load_agent_class() -> type:
        agent_module = importlib.import_module(AGNO_AGENT_MODULE)
        return getattr(agent_module, AGNO_AGENT_CLASS_NAME)

    @staticmethod
    def _load_team_class() -> type:
        team_module = importlib.import_module(AGNO_TEAM_MODULE)
        return getattr(team_module, AGNO_TEAM_CLASS_NAME)

    def activate(self) -> None:
        """Activate Agno instrumentation."""
        if self._is_instrumented:
            return

        if not _is_respan_tracing_enabled():
            logger.info(
                "Agno instrumentation skipped because Respan tracing is disabled"
            )
            return

        if self._agent is not None:
            self._patch_target(
                target=self._agent,
                target_kind=AGNO_TARGET_AGENT,
                is_bound_method=True,
            )
            self._is_instrumented = bool(self._patches)
            return

        try:
            agent_class = self._load_agent_class()
        except ImportError as exception:
            logger.warning(
                f"Failed to activate Agno instrumentation - missing dependency: {exception}"
            )
            return

        self._patch_target(
            target=agent_class,
            target_kind=AGNO_TARGET_AGENT,
            is_bound_method=False,
        )

        if self._include_teams:
            try:
                team_class = self._load_team_class()
            except ImportError:
                logger.info(
                    "Agno Team instrumentation skipped because Team is unavailable"
                )
            else:
                self._patch_target(
                    target=team_class,
                    target_kind=AGNO_TARGET_TEAM,
                    is_bound_method=False,
                )

        self._is_instrumented = bool(self._patches)
        if self._is_instrumented:
            logger.info("Agno instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched Agno run methods."""
        for target, originals in reversed(self._patches):
            for method_name, original_method in originals.items():
                setattr(target, method_name, original_method)
            if hasattr(target, RESPAN_AGNO_ORIGINALS_ATTR):
                delattr(target, RESPAN_AGNO_ORIGINALS_ATTR)
            if hasattr(target, RESPAN_AGNO_WRAPPED_ATTR):
                delattr(target, RESPAN_AGNO_WRAPPED_ATTR)

        self._patches.clear()
        self._is_instrumented = False
        logger.info("Agno instrumentation deactivated")
