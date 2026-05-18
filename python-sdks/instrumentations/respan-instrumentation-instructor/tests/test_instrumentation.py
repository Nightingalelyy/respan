import asyncio
import json
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Literal
from typing import TypedDict

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_instructor import InstructorInstrumentor
from respan_instrumentation_instructor import _instrumentation
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer


class FakeMode:
    value = "tool_call"


class FakeProvider:
    value = "openai"


class UserResult:
    @classmethod
    def model_json_schema(cls):
        return {
            "title": "UserResult",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def __init__(self, name: str = "Ada") -> None:
        self.name = name

    def model_dump(self):
        return {"name": self.name}


class TicketResult(TypedDict):
    customer: str
    priority: Literal["low", "medium", "high"]
    follow_up_hours: int
    next_step: str | None


class FakeSpan:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = dict(attributes)
        self.status = None
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exception):
        self.exceptions.append(exception)


class FakeSpanContext:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exception_type, exception_value, traceback):
        return False


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, attributes, **kwargs):
        span = FakeSpan(name=name, attributes=attributes)
        self.spans.append(span)
        return FakeSpanContext(span=span)


def _fake_create(**kwargs):
    return UserResult()


async def _fake_async_create(**kwargs):
    return UserResult(name="Grace")


def _fake_iter_create(**kwargs):
    return iter([{"task": "send checklist"}, {"task": "finish dashboard"}])


def _install_fake_tracer(monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr(
        target=_instrumentation.trace,
        name="get_tracer",
        value=lambda instrumenting_module_name: tracer,
    )
    return tracer


def _install_fake_instructor_modules(monkeypatch):
    def patch(client=None, create=None, mode=FakeMode()):
        create_callable = create
        if create_callable is None:
            create_callable = client.chat.completions.create

        if asyncio.iscoroutinefunction(create_callable):

            async def new_create(*args, **kwargs):
                return await create_callable(*args, **kwargs)

        else:

            def new_create(*args, **kwargs):
                return create_callable(*args, **kwargs)

        if client is not None:
            client.chat.completions.create = new_create
            return client
        return new_create

    class FakeInstructor:
        def __init__(self, create_function):
            self.create_fn = create_function
            self.provider = FakeProvider()
            self.mode = FakeMode()
            self.default_model = "gpt-4o-mini"

        def create(self, response_model=None, messages=None, **kwargs):
            return self.create_fn(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )

        def create_partial(self, response_model=None, messages=None, **kwargs):
            return self.create(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )

        def create_iterable(self, messages=None, response_model=None, **kwargs):
            def iterator():
                yield self.create(
                    response_model=object,
                    messages=messages,
                    **kwargs,
                )

            return iterator()

        def create_with_completion(self, messages=None, response_model=None, **kwargs):
            result = self.create(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )
            return result, None

    class FakeAsyncInstructor(FakeInstructor):
        async def create(self, response_model=None, messages=None, **kwargs):
            return await self.create_fn(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )

        async def create_partial(self, response_model=None, messages=None, **kwargs):
            return await self.create(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )

        async def create_iterable(self, messages=None, response_model=None, **kwargs):
            return await self.create(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )

        async def create_with_completion(
            self,
            messages=None,
            response_model=None,
            **kwargs,
        ):
            result = await self.create(
                response_model=response_model,
                messages=messages,
                **kwargs,
            )
            return result, None

    instructor_module = ModuleType("instructor")
    core_module = ModuleType("instructor.core")
    patch_module = ModuleType(_instrumentation.INSTRUCTOR_CORE_PATCH_MODULE)
    client_module = ModuleType(_instrumentation.INSTRUCTOR_CORE_CLIENT_MODULE)

    instructor_module.patch = patch
    patch_module.patch = patch
    client_module.Instructor = FakeInstructor
    client_module.AsyncInstructor = FakeAsyncInstructor
    core_module.patch = patch_module
    core_module.client = client_module
    instructor_module.core = core_module

    monkeypatch.setitem(
        dic=sys.modules,
        name="instructor",
        value=instructor_module,
    )
    monkeypatch.setitem(
        dic=sys.modules,
        name="instructor.core",
        value=core_module,
    )
    monkeypatch.setitem(
        dic=sys.modules,
        name=_instrumentation.INSTRUCTOR_CORE_PATCH_MODULE,
        value=patch_module,
    )
    monkeypatch.setitem(
        dic=sys.modules,
        name=_instrumentation.INSTRUCTOR_CORE_CLIENT_MODULE,
        value=client_module,
    )

    return SimpleNamespace(
        instructor_module=instructor_module,
        patch_module=patch_module,
        client_module=client_module,
    )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


def test_patch_create_emits_native_respan_chat_span(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_create, mode=FakeMode())
    result = create(
        response_model=UserResult,
        messages=[{"role": "user", "content": "Extract Ada Lovelace."}],
        model="gpt-4o-mini",
    )

    assert result.model_dump() == {"name": "Ada"}
    assert len(tracer.spans) == 1
    attributes = tracer.spans[0].attributes
    assert attributes[RESPAN_LOG_TYPE] == "chat"
    assert attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attributes[GEN_AI_SYSTEM] == "openai"
    assert attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert (
        attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Extract Ada Lovelace."
    )
    assert attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == '{"name":"Ada"}'
    assert "UserResult" in attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    assert "model" not in attributes
    assert "prompt_tokens" not in attributes
    assert "tool_calls" not in attributes


def test_patch_create_includes_typed_dict_function_schema(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_create, mode=FakeMode())
    create(
        response_model=TicketResult,
        messages=[{"role": "user", "content": "Extract support ticket."}],
        model="gpt-4o-mini",
    )

    functions = tracer.spans[0].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    schema = json.loads(functions)[0]["function"]["parameters"]
    assert schema["title"] == "TicketResult"
    assert schema["properties"]["customer"]["type"] == "string"
    assert schema["properties"]["priority"]["type"] == "string"
    assert schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]
    assert schema["properties"]["follow_up_hours"]["type"] == "integer"
    assert schema["properties"]["next_step"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]


def test_patch_create_resolves_postponed_typed_dict_annotations(monkeypatch):
    class FutureTicket(TypedDict):
        priority: Literal["low", "high"]
        tags: list[str]

    monkeypatch.setattr(
        FutureTicket,
        "__annotations__",
        {"priority": "Literal['low', 'high']", "tags": "list[str]"},
    )
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_create, mode=FakeMode())
    create(
        response_model=FutureTicket,
        messages=[{"role": "user", "content": "Extract support ticket."}],
        model="gpt-4o-mini",
    )

    functions = tracer.spans[0].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    schema = json.loads(functions)[0]["function"]["parameters"]
    assert schema["properties"]["priority"]["type"] == "string"
    assert schema["properties"]["priority"]["enum"] == ["low", "high"]
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"]["type"] == "string"


def test_patch_create_preserves_prebuilt_tool_schema(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ActionItem",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                },
            },
        }
    ]

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_create, mode=FakeMode())
    create(
        tools=tools,
        messages=[{"role": "user", "content": "Extract action items."}],
        model="gpt-4o-mini",
    )

    functions = tracer.spans[0].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    assert json.loads(functions) == tools


def test_patch_create_records_consumed_iterable_output(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_iter_create, mode=FakeMode())
    result = list(
        create(
            response_model=TicketResult,
            messages=[{"role": "user", "content": "Extract action items."}],
            model="gpt-4o-mini",
        )
    )

    assert result == [{"task": "send checklist"}, {"task": "finish dashboard"}]
    assert tracer.spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '[{"task":"send checklist"},{"task":"finish dashboard"}]'
    )


def test_instructor_create_uses_wrapped_create_fn_without_duplicate_span(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_create, mode=FakeMode())
    client = fake.client_module.Instructor(create_function=create)
    client.create(
        response_model=UserResult,
        messages=[{"role": "user", "content": "Extract Ada Lovelace."}],
        model="gpt-4o-mini",
    )

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "instructor.create"
    assert "UserResult" in tracer.spans[0].attributes[
        SpanAttributes.LLM_REQUEST_FUNCTIONS
    ]


def test_create_iterable_preserves_method_context_for_wrapped_create_fn(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_create, mode=FakeMode())
    client = fake.client_module.Instructor(create_function=create)
    list(
        client.create_iterable(
            response_model=TicketResult,
            messages=[{"role": "user", "content": "Extract action items."}],
            model="gpt-4o-mini",
        )
    )

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "instructor.create_iterable"
    functions = tracer.spans[0].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    schema = json.loads(functions)[0]["function"]["parameters"]
    assert schema["title"] == "TicketResult"
    assert schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]


def test_instructor_create_emits_span_for_unwrapped_create_fn(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    client = fake.client_module.Instructor(create_function=_fake_create)
    client.create(
        response_model=UserResult,
        messages=[{"role": "user", "content": "Extract Ada Lovelace."}],
    )

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "instructor.create"
    assert tracer.spans[0].attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"


def test_patch_client_emits_span(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create)),
    )

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    patched_client = fake.instructor_module.patch(client=client, mode=FakeMode())
    patched_client.chat.completions.create(
        response_model=UserResult,
        messages=[{"role": "user", "content": "Extract Ada Lovelace."}],
        model="gpt-4o-mini",
    )

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "instructor.patch"
    assert tracer.spans[0].attributes[GEN_AI_SYSTEM] == "openai"


def test_patch_async_create_emits_span(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    tracer = _install_fake_tracer(monkeypatch)

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()

    create = fake.instructor_module.patch(create=_fake_async_create, mode=FakeMode())
    result = asyncio.run(
        create(
            response_model=UserResult,
            messages=[{"role": "user", "content": "Extract Grace Hopper."}],
            model="gpt-4o-mini",
        )
    )

    assert result.model_dump() == {"name": "Grace"}
    assert len(tracer.spans) == 1
    assert tracer.spans[0].attributes[
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
    ] == '{"name":"Grace"}'


def test_deactivate_restores_patches(monkeypatch):
    fake = _install_fake_instructor_modules(monkeypatch)
    original_patch = fake.patch_module.patch
    original_create = fake.client_module.Instructor.create

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()
    instrumentor.deactivate()

    assert fake.patch_module.patch is original_patch
    assert fake.instructor_module.patch is original_patch
    assert fake.client_module.Instructor.create is original_create


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_instructor_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = InstructorInstrumentor()
    with caplog.at_level("INFO"):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert fake.patch_module.patch is fake.instructor_module.patch
    assert "Instructor instrumentation skipped" in caplog.text


def test_activate_logs_warning_when_dependency_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        raise ImportError(module_name)

    monkeypatch.setattr(
        target=_instrumentation.importlib,
        name="import_module",
        value=import_module_raises,
    )

    instrumentor = InstructorInstrumentor()
    with caplog.at_level("WARNING"):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "Failed to activate Instructor instrumentation" in caplog.text
