"""Native Instructor instrumentation plugin for Respan."""

import functools
import importlib
import inspect
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextvars import ContextVar
from types import NoneType, UnionType
from typing import (
    Any,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    RESPAN_LOG_TYPE,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

INSTRUCTOR_INSTRUMENTATION_NAME = "instructor"
INSTRUCTOR_MODULE = "instructor"
INSTRUCTOR_CORE_PATCH_MODULE = "instructor.core.patch"
INSTRUCTOR_CORE_CLIENT_MODULE = "instructor.core.client"
RESPAN_INSTRUCTOR_INSTRUMENTED_ATTR = "_respan_instructor_instrumented"
INSTRUCTOR_CALL_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "respan_instructor_call_context",
    default=None,
)

_OPENAI_PROVIDER = "openai"
_ASSISTANT_ROLE = "assistant"


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _is_wrapped(callable_object: Any) -> bool:
    return bool(getattr(callable_object, RESPAN_INSTRUCTOR_INSTRUMENTED_ATTR, False))


def _mark_wrapped(callable_object: Callable[..., Any]) -> Callable[..., Any]:
    setattr(callable_object, RESPAN_INSTRUCTOR_INSTRUMENTED_ATTR, True)
    return callable_object


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    raw_value = getattr(value, "value", value)
    if raw_value is None:
        return None
    return str(raw_value)


def _response_model_name(response_model: Any) -> str | None:
    if response_model is None:
        return None
    model_name = getattr(response_model, "__name__", None)
    if model_name:
        return str(model_name)
    return str(response_model)


def _json_stringify(value: Any) -> str:
    try:
        return json.dumps(obj=value, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _serialize_value_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_stringify(value.model_dump())
    return _json_stringify(serialize_value(value=value))


def _message_content_to_string(content: Any) -> str:
    if isinstance(content, str):
        return content
    return _serialize_value_to_string(content)


def _json_schema_for_annotation(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {Required, NotRequired}:
        wrapped_annotation = args[0] if args else Any
        return _json_schema_for_annotation(wrapped_annotation)

    if origin is None:
        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation is None or annotation is NoneType:
            return {"type": "null"}
        if is_typeddict(annotation):
            return _typed_dict_json_schema(annotation)
        return {}

    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        if values:
            value_type = type(values[0])
            if value_type is str:
                schema["type"] = "string"
            elif value_type is int:
                schema["type"] = "integer"
            elif value_type is float:
                schema["type"] = "number"
            elif value_type is bool:
                schema["type"] = "boolean"
        return schema

    if origin in {Union, UnionType}:
        return {"anyOf": [_json_schema_for_annotation(arg) for arg in args]}

    if origin in {list, tuple, set}:
        item_annotation = args[0] if args else Any
        return {
            "type": "array",
            "items": _json_schema_for_annotation(item_annotation),
        }

    if origin in {dict, Mapping}:
        return {"type": "object"}

    return {}


def _typed_dict_json_schema(response_model: Any) -> dict[str, Any]:
    try:
        annotations = get_type_hints(response_model, include_extras=True)
    except (NameError, TypeError):
        annotations = getattr(response_model, "__annotations__", {})
    required_keys = getattr(response_model, "__required_keys__", frozenset())
    properties = {
        field_name: _json_schema_for_annotation(field_type) | {
            "title": field_name.replace("_", " ").title(),
        }
        for field_name, field_type in annotations.items()
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "title": _response_model_name(response_model),
    }
    if required_keys:
        schema["required"] = sorted(required_keys)
    return schema


def _response_model_function_schema(response_model: Any) -> list[dict[str, Any]] | None:
    if response_model is None:
        return None

    model_json_schema = getattr(response_model, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
    elif is_typeddict(response_model):
        schema = _typed_dict_json_schema(response_model)
    else:
        return None

    model_name = _response_model_name(response_model) or "response_model"
    description = schema.get("description") if isinstance(schema, dict) else None

    return [
        {
            "type": "function",
            "function": {
                "name": model_name,
                "description": description or f"Structured response for {model_name}",
                "parameters": schema,
            },
        }
    ]


def _request_function_schema(arguments: Mapping[str, Any]) -> Any | None:
    response_model_schema = _response_model_function_schema(arguments.get("response_model"))
    if response_model_schema is not None:
        return response_model_schema

    tools = arguments.get("tools")
    if isinstance(tools, (list, tuple)) and tools:
        return tools

    return None


def _extract_base_url_provider(client: Any) -> str | None:
    base_url = getattr(client, "base_url", None)
    if base_url is None:
        return None

    normalized_base_url = str(base_url).lower()
    if "anthropic" in normalized_base_url:
        return "anthropic"
    if "google" in normalized_base_url or "generativelanguage" in normalized_base_url:
        return "google"
    if "azure" in normalized_base_url:
        return "azure"
    if "openai" in normalized_base_url:
        return _OPENAI_PROVIDER
    return None


def _extract_provider_from_callable(create_callable: Any) -> str | None:
    owner = getattr(create_callable, "__self__", None)
    if owner is None:
        return None

    client = getattr(owner, "_client", None)
    if client is None:
        client = owner
    return _extract_base_url_provider(client)


def _extract_provider_from_instance(instance: Any) -> str | None:
    provider = _enum_value(getattr(instance, "provider", None))
    if provider:
        return provider
    client = getattr(instance, "client", None)
    return _extract_base_url_provider(client)


def _extract_direct_call_arguments(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    arguments = dict(kwargs)
    if args and "response_model" not in arguments:
        arguments["response_model"] = args[0]
    if len(args) > 1 and "messages" not in arguments:
        arguments["messages"] = args[1]
    return arguments


def _extract_patch_arguments(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    arguments = dict(kwargs)
    if args and "client" not in arguments:
        arguments["client"] = args[0]
    if len(args) > 1 and "create" not in arguments:
        arguments["create"] = args[1]
    if len(args) > 2 and "mode" not in arguments:
        arguments["mode"] = args[2]
    return arguments


def _extract_method_call_arguments(
    method: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        bound_arguments = inspect.signature(method).bind_partial(
            instance,
            *args,
            **kwargs,
        )
    except TypeError:
        return _extract_direct_call_arguments(args=args, kwargs=kwargs)

    arguments = dict(bound_arguments.arguments)
    arguments.pop("self", None)
    nested_kwargs = arguments.pop("kwargs", None)
    if isinstance(nested_kwargs, Mapping):
        arguments.update(nested_kwargs)
    return arguments


def _method_call_context(
    *,
    operation_name: str,
    arguments: Mapping[str, Any],
    mode: Any | None,
) -> dict[str, Any]:
    existing_context = INSTRUCTOR_CALL_CONTEXT.get()
    context = dict(existing_context) if isinstance(existing_context, Mapping) else {}
    for key, value in arguments.items():
        if context.get(key) is None and value is not None:
            context[key] = value
    context.setdefault("operation_name", operation_name)
    if mode is not None:
        context.setdefault("mode", mode)
    return context


def _merge_call_context(arguments: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    merged_arguments = dict(arguments)
    context = INSTRUCTOR_CALL_CONTEXT.get()
    if not isinstance(context, Mapping):
        return merged_arguments, None

    context_response_model = context.get("response_model")
    if (
        context_response_model is not None
        and _response_model_function_schema(merged_arguments.get("response_model"))
        is None
    ):
        merged_arguments["response_model"] = context_response_model

    for key in ("messages", "model", "response_model", "tools"):
        if merged_arguments.get(key) is None and context.get(key) is not None:
            merged_arguments[key] = context[key]
    return merged_arguments, context


def _iter_with_call_context(
    iterable: Iterator[Any],
    context: Mapping[str, Any],
) -> Iterator[Any]:
    token = INSTRUCTOR_CALL_CONTEXT.set(dict(context))
    try:
        yield from iterable
    finally:
        INSTRUCTOR_CALL_CONTEXT.reset(token)


async def _async_iter_with_call_context(
    async_iterable: AsyncIterator[Any],
    context: Mapping[str, Any],
) -> AsyncIterator[Any]:
    token = INSTRUCTOR_CALL_CONTEXT.set(dict(context))
    try:
        async for item in async_iterable:
            yield item
    finally:
        INSTRUCTOR_CALL_CONTEXT.reset(token)


def _build_span_attributes(
    operation_name: str,
    arguments: Mapping[str, Any],
    provider: str | None = None,
    mode: Any | None = None,
) -> dict[str, Any]:
    response_model = arguments.get("response_model")
    messages = arguments.get("messages")
    model = arguments.get("model")
    response_model_name = _response_model_name(response_model)
    mode_value = _enum_value(mode)

    input_payload = {
        "messages": messages,
        "model": model,
        "response_model": response_model_name,
        "mode": mode_value,
    }
    attributes: dict[str, Any] = {
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _serialize_value_to_string(
            {key: value for key, value in input_payload.items() if value is not None}
        ),
    }

    if provider:
        attributes[GEN_AI_SYSTEM] = provider.lower()
    if model:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = str(model)

    if isinstance(messages, list):
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if role is not None:
                attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role"] = str(
                    role
                )
            if content is not None:
                attributes[
                    f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content"
                ] = _message_content_to_string(content)
            if tool_calls is not None:
                attributes[
                    f"{SpanAttributes.LLM_PROMPTS}.{message_index}.tool_calls"
                ] = _serialize_value_to_string(tool_calls)

    function_schema = _request_function_schema(arguments)
    if function_schema is not None:
        attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = _serialize_value_to_string(
            function_schema
        )

    return attributes


def _set_success_attributes(span: Any, result: Any) -> None:
    output = _serialize_value_to_string(result)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output)
    span.set_attribute(
        f"{SpanAttributes.LLM_COMPLETIONS}.0.role",
        _ASSISTANT_ROLE,
    )
    span.set_attribute(
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
        output,
    )
    span.set_status(trace.StatusCode.OK)


def _set_error_status(span: Any, exception: Exception) -> None:
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
    span.record_exception(exception)


def _iter_with_span(
    iterator: Iterator[Any],
    span: Any,
    span_context: Any,
) -> Iterator[Any]:
    output_items: list[Any] = []
    span_closed = False
    try:
        for item in iterator:
            output_items.append(item)
            yield item
    except Exception as exception:
        _set_error_status(span=span, exception=exception)
        span_context.__exit__(type(exception), exception, exception.__traceback__)
        span_closed = True
        raise
    else:
        _set_success_attributes(span=span, result=output_items)
        span_context.__exit__(None, None, None)
        span_closed = True
    finally:
        if not span_closed:
            _set_success_attributes(span=span, result=output_items)
            span_context.__exit__(None, None, None)


async def _async_iter_with_span(
    async_iterator: AsyncIterator[Any],
    span: Any,
    span_context: Any,
) -> AsyncIterator[Any]:
    output_items: list[Any] = []
    span_closed = False
    try:
        async for item in async_iterator:
            output_items.append(item)
            yield item
    except Exception as exception:
        _set_error_status(span=span, exception=exception)
        span_context.__exit__(type(exception), exception, exception.__traceback__)
        span_closed = True
        raise
    else:
        _set_success_attributes(span=span, result=output_items)
        span_context.__exit__(None, None, None)
        span_closed = True
    finally:
        if not span_closed:
            _set_success_attributes(span=span, result=output_items)
            span_context.__exit__(None, None, None)


class InstructorInstrumentor:
    """Respan instrumentor for Instructor.

    This is a native wrapper around Instructor's public patch/create paths.
    It does not depend on ``openinference-instrumentation-instructor``.
    """

    name = INSTRUCTOR_INSTRUMENTATION_NAME

    def __init__(self) -> None:
        self._patches: list[tuple[Any, str, Any]] = []
        self._is_instrumented = False

    def _remember_patch(self, owner: Any, attribute_name: str, value: Any) -> None:
        original_value = getattr(owner, attribute_name)
        if _is_wrapped(original_value):
            return
        setattr(owner, attribute_name, value)
        self._patches.append((owner, attribute_name, original_value))

    def _start_span(
        self,
        operation_name: str,
        arguments: Mapping[str, Any],
        provider: str | None = None,
        mode: Any | None = None,
    ):
        tracer = trace.get_tracer(instrumenting_module_name=__name__)
        return tracer.start_as_current_span(
            name=operation_name,
            attributes=_build_span_attributes(
                operation_name=operation_name,
                arguments=arguments,
                provider=provider,
                mode=mode,
            ),
            record_exception=False,
            set_status_on_exception=False,
        )

    def _wrap_create_callable(
        self,
        original_create: Callable[..., Any],
        operation_name: str,
        provider: str | None = None,
        mode: Any | None = None,
    ) -> Callable[..., Any]:
        if _is_wrapped(original_create):
            return original_create

        if inspect.iscoroutinefunction(original_create):

            @functools.wraps(original_create)
            async def wrapped_async_create(*args: Any, **kwargs: Any) -> Any:
                arguments = _extract_direct_call_arguments(args=args, kwargs=kwargs)
                arguments, context = _merge_call_context(arguments)
                effective_operation_name = (
                    context.get("operation_name", operation_name)
                    if isinstance(context, Mapping)
                    else operation_name
                )
                effective_mode = (
                    mode
                    if mode is not None or not isinstance(context, Mapping)
                    else context.get("mode")
                )
                span_context = self._start_span(
                    operation_name=effective_operation_name,
                    arguments=arguments,
                    provider=provider,
                    mode=effective_mode,
                )
                span = span_context.__enter__()
                try:
                    result = await original_create(*args, **kwargs)
                except Exception as exception:
                    _set_error_status(span=span, exception=exception)
                    span_context.__exit__(
                        type(exception),
                        exception,
                        exception.__traceback__,
                    )
                    raise
                if isinstance(result, AsyncIterator):
                    return _async_iter_with_span(result, span, span_context)
                _set_success_attributes(span=span, result=result)
                span_context.__exit__(None, None, None)
                return result

            return _mark_wrapped(wrapped_async_create)

        @functools.wraps(original_create)
        def wrapped_create(*args: Any, **kwargs: Any) -> Any:
            arguments = _extract_direct_call_arguments(args=args, kwargs=kwargs)
            arguments, context = _merge_call_context(arguments)
            effective_operation_name = (
                context.get("operation_name", operation_name)
                if isinstance(context, Mapping)
                else operation_name
            )
            effective_mode = (
                mode
                if mode is not None or not isinstance(context, Mapping)
                else context.get("mode")
            )
            span_context = self._start_span(
                operation_name=effective_operation_name,
                arguments=arguments,
                provider=provider,
                mode=effective_mode,
            )
            span = span_context.__enter__()
            try:
                result = original_create(*args, **kwargs)
            except Exception as exception:
                _set_error_status(span=span, exception=exception)
                span_context.__exit__(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
                raise
            if isinstance(result, Iterator):
                return _iter_with_span(result, span, span_context)
            _set_success_attributes(span=span, result=result)
            span_context.__exit__(None, None, None)
            return result

        return _mark_wrapped(wrapped_create)

    def _wrap_patch_function(
        self,
        original_patch: Callable[..., Any],
    ) -> Callable[..., Any]:
        if _is_wrapped(original_patch):
            return original_patch

        @functools.wraps(original_patch)
        def wrapped_patch(*args: Any, **kwargs: Any) -> Any:
            patched_target = original_patch(*args, **kwargs)
            arguments = _extract_patch_arguments(args=args, kwargs=kwargs)
            client = arguments.get("client")
            create_callable = arguments.get("create")
            mode = arguments.get("mode")

            if create_callable is not None and callable(patched_target):
                provider = _extract_provider_from_callable(create_callable)
                return self._wrap_create_callable(
                    original_create=patched_target,
                    operation_name="instructor.patch",
                    provider=provider or _OPENAI_PROVIDER,
                    mode=mode,
                )

            if client is not None:
                create_owner = getattr(
                    getattr(
                        getattr(patched_target, "chat", None),
                        "completions",
                        None,
                    ),
                    "create",
                    None,
                )
                if callable(create_owner):
                    wrapped_create = self._wrap_create_callable(
                        original_create=create_owner,
                        operation_name="instructor.patch",
                        provider=_extract_base_url_provider(patched_target)
                        or _OPENAI_PROVIDER,
                        mode=mode,
                    )
                    patched_target.chat.completions.create = wrapped_create
            return patched_target

        return _mark_wrapped(wrapped_patch)

    def _wrap_instructor_method(
        self,
        original_method: Callable[..., Any],
        operation_name: str,
    ) -> Callable[..., Any]:
        if _is_wrapped(original_method):
            return original_method

        if inspect.iscoroutinefunction(original_method):

            @functools.wraps(original_method)
            async def wrapped_async_method(
                instance: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                create_function = getattr(instance, "create_fn", None)
                if _is_wrapped(create_function):
                    arguments = _extract_method_call_arguments(
                        method=original_method,
                        instance=instance,
                        args=args,
                        kwargs=kwargs,
                    )
                    if arguments.get("model") is None:
                        arguments["model"] = getattr(instance, "default_model", None)
                    context = _method_call_context(
                        operation_name=operation_name,
                        arguments=arguments,
                        mode=getattr(instance, "mode", None),
                    )
                    token = INSTRUCTOR_CALL_CONTEXT.set(context)
                    try:
                        result = await original_method(instance, *args, **kwargs)
                    finally:
                        INSTRUCTOR_CALL_CONTEXT.reset(token)
                    if isinstance(result, AsyncIterator):
                        return _async_iter_with_call_context(result, context)
                    return result

                arguments = _extract_method_call_arguments(
                    method=original_method,
                    instance=instance,
                    args=args,
                    kwargs=kwargs,
                )
                if arguments.get("model") is None:
                    arguments["model"] = getattr(instance, "default_model", None)
                with self._start_span(
                    operation_name=operation_name,
                    arguments=arguments,
                    provider=_extract_provider_from_instance(instance)
                    or _OPENAI_PROVIDER,
                    mode=getattr(instance, "mode", None),
                ) as span:
                    try:
                        result = await original_method(instance, *args, **kwargs)
                    except Exception as exception:
                        _set_error_status(span=span, exception=exception)
                        raise
                    _set_success_attributes(span=span, result=result)
                    return result

            return _mark_wrapped(wrapped_async_method)

        @functools.wraps(original_method)
        def wrapped_method(instance: Any, *args: Any, **kwargs: Any) -> Any:
            create_function = getattr(instance, "create_fn", None)
            if _is_wrapped(create_function):
                arguments = _extract_method_call_arguments(
                    method=original_method,
                    instance=instance,
                    args=args,
                    kwargs=kwargs,
                )
                if arguments.get("model") is None:
                    arguments["model"] = getattr(instance, "default_model", None)
                context = _method_call_context(
                    operation_name=operation_name,
                    arguments=arguments,
                    mode=getattr(instance, "mode", None),
                )
                token = INSTRUCTOR_CALL_CONTEXT.set(context)
                try:
                    result = original_method(instance, *args, **kwargs)
                finally:
                    INSTRUCTOR_CALL_CONTEXT.reset(token)
                if isinstance(result, Iterator) and operation_name.endswith(
                    "create_iterable"
                ):
                    return _iter_with_call_context(iter(result), context)
                return result

            arguments = _extract_method_call_arguments(
                method=original_method,
                instance=instance,
                args=args,
                kwargs=kwargs,
            )
            if arguments.get("model") is None:
                arguments["model"] = getattr(instance, "default_model", None)
            with self._start_span(
                operation_name=operation_name,
                arguments=arguments,
                provider=_extract_provider_from_instance(instance) or _OPENAI_PROVIDER,
                mode=getattr(instance, "mode", None),
            ) as span:
                try:
                    result = original_method(instance, *args, **kwargs)
                except Exception as exception:
                    _set_error_status(span=span, exception=exception)
                    raise
                _set_success_attributes(span=span, result=result)
                return result

        return _mark_wrapped(wrapped_method)

    def _patch_method(
        self,
        owner: Any,
        method_name: str,
        operation_name: str,
    ) -> None:
        original_method = getattr(owner, method_name, None)
        if original_method is None:
            return
        wrapped_method = self._wrap_instructor_method(
            original_method=original_method,
            operation_name=operation_name,
        )
        self._remember_patch(
            owner=owner,
            attribute_name=method_name,
            value=wrapped_method,
        )

    def activate(self) -> None:
        """Activate native Instructor instrumentation."""
        if self._is_instrumented:
            return

        if not _is_respan_tracing_enabled():
            logger.info(
                "Instructor instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            instructor_module = importlib.import_module(INSTRUCTOR_MODULE)
            patch_module = importlib.import_module(INSTRUCTOR_CORE_PATCH_MODULE)
            client_module = importlib.import_module(INSTRUCTOR_CORE_CLIENT_MODULE)
        except ImportError as import_error:
            logger.warning(
                "Failed to activate Instructor instrumentation — missing dependency: %s",
                import_error,
            )
            return

        wrapped_patch = self._wrap_patch_function(patch_module.patch)
        self._remember_patch(
            owner=patch_module,
            attribute_name="patch",
            value=wrapped_patch,
        )
        self._remember_patch(
            owner=instructor_module,
            attribute_name="patch",
            value=wrapped_patch,
        )

        for method_name in (
            "create",
            "create_partial",
            "create_iterable",
            "create_with_completion",
        ):
            self._patch_method(
                owner=client_module.Instructor,
                method_name=method_name,
                operation_name=f"instructor.{method_name}",
            )
            self._patch_method(
                owner=client_module.AsyncInstructor,
                method_name=method_name,
                operation_name=f"instructor.async_{method_name}",
            )

        self._is_instrumented = True
        logger.info("Instructor instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate native Instructor instrumentation."""
        for owner, attribute_name, original_value in reversed(self._patches):
            setattr(owner, attribute_name, original_value)
        self._patches.clear()
        self._is_instrumented = False
        logger.info("Instructor instrumentation deactivated")
