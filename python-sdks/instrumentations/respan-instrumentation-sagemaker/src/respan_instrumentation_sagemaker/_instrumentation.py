"""AWS SageMaker Runtime instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from types import TracebackType
from typing import Any, Self

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_sagemaker._constants import (
    BODY_KEY,
    SAGEMAKER_INSTRUMENTATION_NAME,
    SAGEMAKER_RUNTIME_SERVICE_NAME,
    STREAMING_OPERATIONS,
    SUPPORTED_OPERATIONS,
)
from respan_instrumentation_sagemaker._otel_emitter import emit_sagemaker_span
from respan_instrumentation_sagemaker._translator import (
    SageMakerStreamAccumulator,
    capture_invoke_response_payload,
)

logger = logging.getLogger(__name__)

_original_make_api_call = None
_patched_make_api_call = None
_activation_count = 0
_activation_lock = threading.RLock()


def _error_details(exc: BaseException) -> tuple[str, int]:
    try:
        response = getattr(exc, "response", None)
    except Exception:  # noqa: BLE001 - provider exception properties are untrusted
        response = None
    status_code = _status_code_from_response(response)
    if status_code < 400:
        status_code = 500
    error = response.get("Error") if isinstance(response, Mapping) else None
    if isinstance(error, Mapping):
        error_type = error.get("Code")
        message = error.get("Message")
        return (
            f"{error_type or type(exc).__name__}: {message or 'request failed'}",
            status_code,
        )
    return type(exc).__name__, status_code


def _load_base_client_class() -> type[Any]:
    module = importlib.import_module("botocore.client")
    base_client = getattr(module, "BaseClient", None)
    if base_client is None:
        raise AttributeError("botocore.client.BaseClient")
    return base_client


def _is_sagemaker_runtime_client(client: Any) -> bool:
    service_model = getattr(getattr(client, "meta", None), "service_model", None)
    return (
        getattr(service_model, "service_name", None) == SAGEMAKER_RUNTIME_SERVICE_NAME
    )


def _status_code_from_response(response: Any) -> int:
    if not isinstance(response, Mapping):
        return 200
    response_metadata = response.get("ResponseMetadata")
    if not isinstance(response_metadata, Mapping):
        return 200
    value = response_metadata.get("HTTPStatusCode")
    return value if isinstance(value, int) else 200


class _InstrumentedEventStream:
    def __init__(
        self,
        *,
        stream: Iterable[Any],
        operation_name: str,
        api_params: Mapping[str, Any] | None,
        start_ns: int,
        trace_id: str | None,
        parent_id: str | None,
    ) -> None:
        self._stream = stream
        self._operation_name = operation_name
        self._api_params = api_params
        self._start_ns = start_ns
        self._trace_id = trace_id
        self._parent_id = parent_id
        self._accumulator = SageMakerStreamAccumulator()
        self._emitted = False

    def __iter__(self) -> Iterator[Any]:
        stream_iterable = (
            [self._stream] if isinstance(self._stream, Mapping) else self._stream
        )
        try:
            for event in stream_iterable:
                self._accumulator.add_event(event)
                yield event
        except BaseException as exc:
            error_message, status_code = _error_details(exc)
            self._emit(error_message=error_message, status_code=status_code)
            raise
        else:
            self._emit()

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._emit()

    def __enter__(self) -> Self:
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc is not None:
            message, status = _error_details(exc)
            self._emit(error_message=message, status_code=status)
        else:
            self._emit()
        exit_method = getattr(self._stream, "__exit__", None)
        if callable(exit_method):
            return bool(exit_method(exc_type, exc, traceback))
        self.close()
        return False

    def _emit(
        self, *, error_message: str | None = None, status_code: int = 200
    ) -> None:
        if self._emitted:
            return
        self._emitted = True
        emit_sagemaker_span(
            operation_name=self._operation_name,
            api_params=self._api_params,
            start_ns=self._start_ns,
            stream_events=self._accumulator,
            error_message=error_message,
            status_code=status_code,
            trace_id=self._trace_id,
            parent_id=self._parent_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_streaming_response(
    *,
    response: Any,
    operation_name: str,
    api_params: Mapping[str, Any] | None,
    start_ns: int,
    trace_id: str | None,
    parent_id: str | None,
) -> Any:
    if not isinstance(response, dict):
        return response

    stream = response.get(BODY_KEY)
    if stream is None:
        emit_sagemaker_span(
            operation_name=operation_name,
            api_params=api_params,
            start_ns=start_ns,
            response_payload=response,
            status_code=_status_code_from_response(response),
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    response[BODY_KEY] = _InstrumentedEventStream(
        stream=stream,
        operation_name=operation_name,
        api_params=api_params,
        start_ns=start_ns,
        trace_id=trace_id,
        parent_id=parent_id,
    )
    return response


def _wrap_make_api_call(original: Any) -> Any:
    def wrapper(
        self: Any, operation_name: str, api_params: Mapping[str, Any] | None = None
    ) -> Any:
        if (
            operation_name not in SUPPORTED_OPERATIONS
            or not _is_sagemaker_runtime_client(self)
        ):
            return original(self, operation_name, api_params)

        start_ns = time.time_ns()
        from respan_instrumentation_sagemaker._otel_emitter import (
            _current_trace_parent_ids,
        )

        trace_id, parent_id = _current_trace_parent_ids()
        try:
            response = original(self, operation_name, api_params)
        except Exception as exc:
            error_message, status_code = _error_details(exc)
            emit_sagemaker_span(
                operation_name=operation_name,
                api_params=api_params,
                start_ns=start_ns,
                error_message=error_message,
                status_code=status_code,
                trace_id=trace_id,
                parent_id=parent_id,
            )
            raise

        if operation_name in STREAMING_OPERATIONS:
            return _wrap_streaming_response(
                response=response,
                operation_name=operation_name,
                api_params=api_params,
                start_ns=start_ns,
                trace_id=trace_id,
                parent_id=parent_id,
            )

        response, response_payload = capture_invoke_response_payload(response)
        if response_payload is None:
            response_payload = response
        emit_sagemaker_span(
            operation_name=operation_name,
            api_params=api_params,
            start_ns=start_ns,
            response_payload=response_payload,
            status_code=_status_code_from_response(response),
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return response

    return wrapper


class SageMakerInstrumentor:
    """Respan instrumentor for the AWS SageMaker Runtime boto3 client."""

    name = SAGEMAKER_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._is_instrumented = False

    def activate(self) -> None:
        """Monkey-patch botocore's SageMaker Runtime call path."""
        global _activation_count, _original_make_api_call, _patched_make_api_call

        try:
            base_client = _load_base_client_class()
        except (AttributeError, ImportError) as exc:
            logger.warning(
                "Failed to activate SageMaker instrumentation - missing dependency: %s",
                exc,
            )
            return
        with _activation_lock:
            if self._is_instrumented:
                return
            if _activation_count == 0:
                _original_make_api_call = base_client._make_api_call
                _patched_make_api_call = _wrap_make_api_call(_original_make_api_call)
                base_client._make_api_call = _patched_make_api_call
            _activation_count += 1
            RespanTracer().get_tracer()
            self._is_instrumented = True
        logger.info("SageMaker instrumentation activated")

    def deactivate(self) -> None:
        """Restore botocore's original call path."""
        global _activation_count, _original_make_api_call, _patched_make_api_call

        if not self._is_instrumented:
            return

        with _activation_lock:
            self._is_instrumented = False
            _activation_count = max(_activation_count - 1, 0)
            if _activation_count:
                return
            try:
                base_client = _load_base_client_class()
                if (
                    _original_make_api_call is not None
                    and base_client._make_api_call is _patched_make_api_call
                ):
                    base_client._make_api_call = _original_make_api_call
            except Exception:
                logger.debug(
                    "Failed to deactivate SageMaker instrumentation", exc_info=True
                )
            finally:
                _original_make_api_call = None
                _patched_make_api_call = None
                logger.info("SageMaker instrumentation deactivated")
