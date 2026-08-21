"""Contract and lifecycle tests for the native OpenAI instrumentation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from opentelemetry.trace import StatusCode

from respan_instrumentation_openai import _instrumentation as instrumentation
from respan_instrumentation_openai import _otel_emitter as emitter
from respan_instrumentation_openai import _translator as translator
from respan_instrumentation_openai._instrumentation import OpenAIInstrumentor
from respan_instrumentation_openai._serialization import error_message, json_string


@pytest.fixture(autouse=True)
def clean_instrumentation(monkeypatch):
    instrumentation._remove_patches()
    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(
        OpenAIInstrumentor,
        "_is_respan_tracing_enabled",
        staticmethod(lambda: True),
    )
    yield
    instrumentation._remove_patches()
    instrumentation._REFCOUNT = 0


def _chat_response(*, tool_calls=None):
    return SimpleNamespace(
        id="chatcmpl-123",
        model="gpt-4.1-nano-2025-04-14",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content="Hello there", tool_calls=tool_calls
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )


def _responses_response(*, output=None):
    return SimpleNamespace(
        id="resp-1",
        model="gpt-4.1-mini",
        output_text="done",
        output=output or [],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
    )


def test_chat_attrs_follow_canonical_contract_and_keep_turn_calls_separate():
    historical = {
        "id": "call_old",
        "type": "function",
        "function": {"name": "old_tool", "arguments": '{"old":true}'},
    }
    current = SimpleNamespace(
        id="call_new",
        function=SimpleNamespace(name="lookup", arguments='{"city":"Paris"}'),
    )
    attrs = emitter.build_chat_attrs(
        request_kwargs={
            "model": "gpt-4.1-nano",
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [historical]},
                {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        response=_chat_response(tool_calls=[current]),
    )

    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs["llm.request.type"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4.1-nano"
    assert attrs["gen_ai.response.id"] == "chatcmpl-123"
    assert attrs["gen_ai.is_streaming"] is False
    assert attrs["gen_ai.usage.input_tokens"] == 8
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["gen_ai.usage.prompt_tokens"] == 8
    assert attrs["gen_ai.usage.completion_tokens"] == 3
    assert attrs["llm.usage.total_tokens"] == 11
    assert json.loads(attrs["gen_ai.prompt.0.tool_calls"])[0]["id"] == "call_old"
    assert json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["id"] == "call_new"
    assert json.loads(attrs["llm.request.functions"])[0]["function"]["name"] == "lookup"
    assert (
        json.loads(attrs["traceloop.entity.output"])["tool_calls"][0]["id"]
        == "call_new"
    )

    forbidden = {
        "traceloop.span.kind",
        "respan.span.tools",
        "respan.span.tool_calls",
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
    }
    assert forbidden.isdisjoint(attrs)


def test_managed_prompt_request_preserves_bounded_redacted_input():
    attrs = emitter.build_chat_attrs(
        request_kwargs={
            "model": "placeholder",
            "messages": [],
            "extra_body": {
                "prompt": {
                    "prompt_id": "prompt_123",
                    "variables": {
                        "feature_request": "Add order notifications",
                        "api_key": "sk-secret",
                    },
                }
            },
        },
        response=_chat_response(),
    )
    assert json.loads(attrs["traceloop.entity.input"]) == {
        "prompt": {
            "prompt_id": "prompt_123",
            "variables": {
                "api_key": "[REDACTED]",
                "feature_request": "Add order notifications",
            },
        }
    }
    assert "gen_ai.prompt.0.content" not in attrs


def test_responses_attrs_use_chat_contract_and_canonicalize_tools():
    call = SimpleNamespace(
        type="function_call",
        call_id="call_weather",
        name="get_weather",
        arguments='{"city":"Paris"}',
    )
    attrs = emitter.build_response_attrs(
        request_kwargs={
            "model": "gpt-4.1-mini",
            "instructions": "Be concise",
            "input": "weather",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
        },
        response=_responses_response(output=[call]),
    )
    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.1.role"] == "user"
    tool = json.loads(attrs["llm.request.functions"])[0]
    assert tool["function"]["name"] == "get_weather"
    current = json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]
    assert current["id"] == "call_weather"
    assert json.loads(current["function"]["arguments"]) == {"city": "Paris"}
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 4
    assert attrs["gen_ai.response.id"] == "resp-1"


def test_embedding_attrs_keep_provider_vectors_and_real_usage():
    response = SimpleNamespace(
        model="text-embedding-3-small",
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )
    attrs = emitter.build_embedding_attrs(
        request_kwargs={"model": "text-embedding-3-small", "input": "reset password"},
        response=response,
    )
    assert attrs["llm.request.type"] == "embedding"
    assert attrs["respan.entity.log_type"] == "embedding"
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert json.loads(attrs["traceloop.entity.output"]) == [[0.1, 0.2, 0.3]]


def test_serialization_is_bounded_redacted_and_never_calls_hostile_string_methods():
    class Hostile:
        def __str__(self):
            raise AssertionError("must not stringify")

        def __repr__(self):
            raise AssertionError("must not repr")

    encoded = json_string(
        {
            "authorization": "Bearer raw-secret",
            "nested": {"api_key": "sk-live", "value": "token=raw-token"},
            "payload": "x" * 50_000,
            "hostile": Hostile(),
        }
    )
    assert len(encoded.encode("utf-8")) <= 16_000
    decoded = json.loads(encoded)
    assert decoded["truncated"] is True
    assert "sk-live" not in encoded
    assert "raw-secret" not in encoded
    assert "raw-token" not in encoded


def test_serialization_consumes_only_bounded_sequence_prefix():
    class BoundedSequence(Sequence[int]):
        def __getitem__(self, index):
            if index > 50:
                raise AssertionError("serializer consumed beyond its lookahead")
            return index

        def __len__(self):
            raise AssertionError("serializer must not request sequence length")

    decoded = json.loads(json_string({"items": BoundedSequence()}))
    assert decoded["items"][:2] == [0, 1]
    assert decoded["items"][-1] == {"truncated_items": True}
    assert len(decoded["items"]) == 51


def test_serialization_bounds_multibyte_values_in_utf8_bytes():
    encoded = json_string({"emoji": "U0001f600" * 16_000})
    assert len(encoded.encode("utf-8")) <= 16_000
    assert json.loads(encoded)["truncated"] is True


def test_plain_prompt_completion_and_error_attributes_are_redacted_and_byte_bounded():
    secret = "api_key=sk-secret "
    multibyte = "U0001f600" * 20_000
    response = _chat_response()
    response.choices[0].message.content = secret + multibyte
    attrs = emitter.build_chat_attrs(
        request_kwargs={
            "model": "gpt",
            "messages": [{"role": "user", "content": secret + multibyte}],
        },
        response=response,
    )
    for key in ("gen_ai.prompt.0.content", "gen_ai.completion.0.content"):
        assert "sk-secret" not in attrs[key]
        assert len(attrs[key].encode("utf-8")) <= 16_000
    rendered_error = error_message(RuntimeError(secret + multibyte))
    assert "sk-secret" not in rendered_error
    assert len(rendered_error.encode("utf-8")) <= 2_000


def test_suppression_context_prevents_duplicate_emission(monkeypatch):
    emitted = []
    monkeypatch.setitem(
        instrumentation._KINDS,
        "chat",
        (lambda **kwargs: emitted.append(kwargs), instrumentation._aggregate_chat),
    )

    def original(_self, **_kwargs):
        return _chat_response()

    wrapper = instrumentation._make_sync_wrapper(original, kind="chat")
    with instrumentation.suppress_openai_instrumentation():
        wrapper(object(), model="gpt-4.1-nano", messages=[])
    assert emitted == []
    wrapper(object(), model="gpt-4.1-nano", messages=[])
    assert len(emitted) == 1


def test_lifecycle_is_reference_counted_and_patches_parse_surfaces():
    from openai.resources.chat.completions import Completions
    from openai.resources.responses.responses import Responses

    original_chat_create = Completions.create
    original_chat_parse = Completions.parse
    original_response_parse = Responses.parse
    first = OpenAIInstrumentor()
    second = OpenAIInstrumentor()

    first.activate()
    first.activate()
    second.activate()
    assert instrumentation._REFCOUNT == 2
    assert Completions.create is not original_chat_create
    assert Completions.parse is not original_chat_parse
    assert Responses.parse is not original_response_parse

    first.deactivate()
    first.deactivate()
    assert instrumentation._REFCOUNT == 1
    assert Completions.parse is not original_chat_parse
    second.deactivate()
    assert instrumentation._REFCOUNT == 0
    assert Completions.create is original_chat_create
    assert Completions.parse is original_chat_parse
    assert Responses.parse is original_response_parse
    assert instrumentation._INSTALLED_METHODS == {}


def test_deactivate_does_not_clobber_foreign_post_patch_wrapper():
    from openai.resources.chat.completions import Completions

    original = Completions.create
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    key = (Completions, "create")
    assert Completions.create is instrumentation._INSTALLED_METHODS[key]

    def foreign_wrapper(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    try:
        Completions.create = foreign_wrapper
        instrumentor.deactivate()
        assert Completions.create is foreign_wrapper
        assert key not in instrumentation._ORIGINAL_METHODS
        assert key not in instrumentation._INSTALLED_METHODS
    finally:
        Completions.create = original
        instrumentor.deactivate()


def test_partial_activation_failure_rolls_back_every_new_patch(monkeypatch):
    original_patch = instrumentation._patch
    attempts = 0

    def fail_second(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("patch failed")
        return original_patch(*args, **kwargs)

    monkeypatch.setattr(instrumentation, "_patch", fail_second)
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    assert instrumentor._is_instrumented is False
    assert instrumentation._REFCOUNT == 0
    assert instrumentation._ORIGINAL_METHODS == {}
    assert instrumentation._INSTALLED_METHODS == {}


def test_sync_stream_early_close_closes_source_and_emits_once(monkeypatch):
    emitted = []
    monkeypatch.setitem(
        instrumentation._KINDS,
        "chat",
        (lambda **kwargs: emitted.append(kwargs), instrumentation._aggregate_chat),
    )

    class Source:
        def __init__(self):
            self.items = iter([SimpleNamespace(choices=[], model="gpt", usage=None)])
            self.closed = 0

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.items)

        def close(self):
            self.closed += 1

    source = Source()
    stream = instrumentation._SyncStreamWrapper(
        source,
        kind="chat",
        request_kwargs={"model": "gpt", "messages": [], "stream": True},
        start_ns=1,
        trace_id=None,
        parent_id=None,
    )
    next(stream)
    stream.close()
    stream.close()
    assert source.closed == 1
    assert len(emitted) == 1
    assert emitted[0]["response"]["model"] == "gpt"


@pytest.mark.asyncio
async def test_async_stream_cancellation_closes_source_and_emits_one_error(monkeypatch):
    emitted = []
    monkeypatch.setitem(
        instrumentation._KINDS,
        "response",
        (lambda **kwargs: emitted.append(kwargs), instrumentation._aggregate_response),
    )

    class Source:
        def __init__(self):
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

        async def close(self):
            self.closed += 1

    source = Source()
    stream = instrumentation._AsyncStreamWrapper(
        source,
        kind="response",
        request_kwargs={"model": "gpt", "input": "hi", "stream": True},
        start_ns=1,
        trace_id=None,
        parent_id=None,
    )
    with pytest.raises(asyncio.CancelledError):
        await stream.__anext__()
    await stream.aclose()
    assert source.closed == 1
    assert len(emitted) == 1
    assert emitted[0]["status_code"] == 499
    assert emitted[0]["error_type"] == "CancelledError"


def test_hostile_exception_properties_cannot_mask_provider_error():
    class HostileError(Exception):
        def __getattribute__(self, name):
            if name in {"args", "message", "response", "status_code"}:
                raise RuntimeError("hostile property")
            return super().__getattribute__(name)

    error = HostileError()
    assert instrumentation._error_kwargs(error) == {
        "error_message": "HostileError",
        "error_type": "HostileError",
        "status_code": 500,
    }


def test_emit_error_preserves_precise_status_and_standard_otel_error(monkeypatch):
    captured = []
    monkeypatch.setattr(emitter, "inject_span", lambda span: captured.append(span))
    emitter.emit_chat_span(
        request_kwargs={"model": "gpt", "messages": []},
        start_ns=1,
        error_message="invalid credentials",
        error_type="AuthenticationError",
        status_code=401,
    )
    span = captured[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 401
    assert span.attributes["error.message"] == "invalid credentials"
    assert json.loads(span.attributes["traceloop.entity.output"])["status_code"] == 401


def test_stream_accumulator_bounds_text_and_tool_arguments():
    accumulator = instrumentation._StreamAccumulator("chat")
    delta = SimpleNamespace(
        content="x" * 30_000,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="call_1",
                function=SimpleNamespace(name="lookup", arguments="y" * 30_000),
            )
        ],
    )
    accumulator.add(
        SimpleNamespace(
            id="chat_1",
            model="gpt",
            usage=None,
            choices=[SimpleNamespace(delta=delta)],
        )
    )
    response = accumulator.response()
    message = response["choices"][0]["message"]
    assert len(message["content"]) == 12_000
    assert len(message["tool_calls"][0]["function"]["arguments"]) == 8_000


def test_response_tool_history_does_not_leak_into_current_turn_calls():
    history = [
        {
            "type": "function_call",
            "call_id": "call_old",
            "name": "lookup",
            "arguments": '{"city":"Rome"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_old",
            "output": "sunny",
        },
    ]
    attrs = emitter.build_response_attrs(
        request_kwargs={"model": "gpt", "input": history},
        response=_responses_response(),
    )
    assert "gen_ai.completion.0.tool_calls" not in attrs
    assert json.loads(attrs["gen_ai.prompt.0.tool_calls"])[0]["id"] == "call_old"


def test_tool_argument_secrets_are_redacted():
    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="lookup", arguments='{"api_key":"sk-secret","city":"Paris"}'
        ),
    )
    normalized = translator.extract_chat_tool_calls(_chat_response(tool_calls=[call]))
    arguments = json.loads(normalized[0]["function"]["arguments"])
    assert arguments == {"api_key": "[REDACTED]", "city": "Paris"}
