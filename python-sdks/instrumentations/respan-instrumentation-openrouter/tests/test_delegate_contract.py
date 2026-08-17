from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.completions import Completions as LegacyCompletions
from openai.resources.responses import Responses
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_STREAM,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    StatusCode,
    TraceFlags,
    TraceState,
    use_span,
)
from respan_instrumentation_openai import OpenAIInstrumentor as DirectOpenAIInstrumentor
from respan_instrumentation_openai import _instrumentation as openai_instrumentation
from respan_instrumentation_openai import _otel_emitter as openai_emitter
from respan_instrumentation_openrouter import OpenRouterInstrumentor, _instrumentation
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TEXT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


def test_declared_versions_match_the_validated_delegate_surface() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = project["tool"]["poetry"]["dependencies"]

    assert dependencies["openai"] == ">=3.0.0,<4.0.0"
    assert dependencies["respan-instrumentation-openai"] == ">=1.2.1,<2.0.0"


class FakeTracerProvider:
    def __init__(self) -> None:
        self._active_span_processor = SimpleNamespace(_span_processors=("exporter",))


def _response(content: str = "Hello from OpenRouter") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-openrouter-test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=1,
        model="openai/gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=7,
            completion_tokens=5,
            total_tokens=12,
        ),
    )


def _chunk(
    content: str, *, usage: CompletionUsage | None = None
) -> ChatCompletionChunk:
    choices = (
        [
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(content=content),
                finish_reason=None,
            )
        ]
        if content
        else []
    )
    return ChatCompletionChunk(
        id="chatcmpl-openrouter-stream-test",
        choices=choices,
        created=1,
        model="openai/gpt-4o-mini",
        object="chat.completion.chunk",
        usage=usage,
    )


def _tool_response() -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-openrouter-tool-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        }
    )


def _tool_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-openrouter-stream-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ]
                    },
                }
            ],
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "object": "chat.completion.chunk",
        }
    )


@pytest.fixture
def capture_delegate_spans(monkeypatch):
    owner = _instrumentation._ACTIVE_BRIDGE_OWNER
    if owner is not None:
        owner.deactivate()

    provider = FakeTracerProvider()
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer_provider",
        lambda: provider,
    )
    captured: list[ReadableSpan] = []
    instrumentor = OpenRouterInstrumentor()

    def install_capture() -> None:
        assert instrumentor._processor is not None

        def inject(span: ReadableSpan) -> bool:
            instrumentor._processor.on_end(span)
            captured.append(span)
            return True

        monkeypatch.setattr(openai_emitter, "inject_span", inject)

    yield instrumentor, captured, install_capture

    instrumentor.deactivate()


def test_current_sdk_sync_delegate_emits_one_real_readable_span(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return _response()

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()

    result = Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert isinstance(result, ChatCompletion)
    assert len(captured) == 1
    span = captured[0]
    assert isinstance(span, ReadableSpan)
    assert span.attributes[GEN_AI_PROVIDER_NAME] == "openrouter"
    assert span.attributes[SpanAttributes.LLM_SYSTEM] == "openrouter"
    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "openai/gpt-4o-mini"
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "openrouter.chat"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert span.instrumentation_scope.name == "respan.instrumentation.openrouter"


def test_released_delegate_without_suppression_helper_remains_supported(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return _response("released delegate compatibility")

    monkeypatch.delattr(
        openai_instrumentation,
        "_is_openai_instrumentation_suppressed",
        raising=False,
    )
    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()

    response = Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "compatibility"}],
    )

    assert response.choices[0].message.content == "released delegate compatibility"
    assert len(captured) == 1
    assert captured[0].attributes[GEN_AI_PROVIDER_NAME] == "openrouter"


def test_current_sdk_legacy_completion_uses_canonical_text_contract(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return SimpleNamespace(
            id="cmpl-openrouter-test",
            model="openai/gpt-4o-mini",
            choices=[SimpleNamespace(text="legacy completion")],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )

    monkeypatch.setattr(LegacyCompletions, "create", create)
    instrumentor.activate()
    install_capture()

    result = LegacyCompletions.create(
        object(), model="openai/gpt-4o-mini", prompt="complete this"
    )

    assert result.choices[0].text == "legacy completion"
    assert len(captured) == 1
    span = captured[0]
    assert isinstance(span, ReadableSpan)
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "openrouter.completion"
    )
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""


def test_current_sdk_responses_uses_canonical_chat_contract(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return SimpleNamespace(
            id="resp-openrouter-test",
            model="openai/gpt-4o-mini",
            output_text="responses output",
            usage=SimpleNamespace(input_tokens=4, output_tokens=3, total_tokens=7),
        )

    monkeypatch.setattr(Responses, "create", create)
    instrumentor.activate()
    install_capture()

    result = Responses.create(
        object(), model="openai/gpt-4o-mini", input="respond to this"
    )

    assert result.output_text == "responses output"
    assert len(captured) == 1
    span = captured[0]
    assert isinstance(span, ReadableSpan)
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "openrouter.response"
    )
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""


def test_current_sdk_delegate_emits_canonical_tools_without_aliases(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return _tool_response()

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()
    tool_definition = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }

    Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "weather"}],
        tools=[tool_definition],
    )

    assert len(captured) == 1
    attrs = dict(captured[0].attributes)
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [tool_definition]
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]
    for alias in ("tools", "tool_calls", "respan.span.tools"):
        assert alias not in attrs


def test_stream_uses_call_time_parent_when_consumed_outside_context(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans
    chunks = [_chunk("parented")]

    def create(_self, **_kwargs):
        return iter(chunks)

    monkeypatch.setattr(Completions, "create", create)
    monkeypatch.setattr(openai_instrumentation, "_is_stream", lambda _value: True)
    instrumentor.activate()
    install_capture()
    parent_context = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )

    with use_span(NonRecordingSpan(parent_context)):
        stream = Completions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
    assert list(stream) == chunks

    assert len(captured) == 1
    assert captured[0].context.trace_id == parent_context.trace_id
    assert captured[0].parent is not None
    assert captured[0].parent.span_id == parent_context.span_id
    assert captured[0].attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == (
        "openrouter.chat"
    )


def test_sync_stream_finalizes_once_with_stream_flag(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans
    chunks = [
        _chunk("Trace "),
        _chunk("works."),
        _chunk(
            "",
            usage=CompletionUsage(
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
            ),
        ),
    ]

    def create(_self, **_kwargs):
        return iter(chunks)

    monkeypatch.setattr(Completions, "create", create)
    monkeypatch.setattr(openai_instrumentation, "_is_stream", lambda _value: True)
    instrumentor.activate()
    install_capture()

    stream = Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
    )
    assert list(stream) == chunks

    assert len(captured) == 1
    span = captured[0]
    assert span.attributes[GEN_AI_REQUEST_STREAM] is True
    assert span.attributes[SpanAttributes.LLM_IS_STREAMING] is True
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Trace works."
    )
    assert span.attributes["gen_ai.usage.input_tokens"] == 4
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


def test_long_stream_keeps_complete_bounded_content_terminal_tool_and_usage(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans
    chunks = [
        *[_chunk("x") for _ in range(150)],
        _tool_chunk(),
        _chunk(
            "",
            usage=CompletionUsage(
                prompt_tokens=19,
                completion_tokens=11,
                total_tokens=30,
            ),
        ),
    ]

    def create(_self, **_kwargs):
        return iter(chunks)

    monkeypatch.setattr(Completions, "create", create)
    monkeypatch.setattr(openai_instrumentation, "_is_stream", lambda _value: True)
    instrumentor.activate()
    install_capture()

    stream = Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "long stream"}],
        stream=True,
    )
    assert list(stream) == chunks

    assert len(captured) == 1
    attrs = dict(captured[0].attributes)
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "x" * 150
    assert attrs["gen_ai.usage.input_tokens"] == 19
    assert attrs["gen_ai.usage.output_tokens"] == 11
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        {
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Tokyo"}',
            },
        }
    ]


def test_sync_stream_close_finalizes_partial_output_once(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans
    chunks = [_chunk("partial"), _chunk(" ignored")]

    def create(_self, **_kwargs):
        return iter(chunks)

    monkeypatch.setattr(Completions, "create", create)
    monkeypatch.setattr(openai_instrumentation, "_is_stream", lambda _value: True)
    instrumentor.activate()
    install_capture()

    stream = Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
    )
    assert next(stream) == chunks[0]
    stream.close()

    assert len(captured) == 1
    assert captured[0].attributes[GEN_AI_REQUEST_STREAM] is True
    assert (
        captured[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "partial"
    )


def test_sync_stream_proxy_preserves_context_manager_and_delegate_attributes(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class ContextStream:
        marker = "delegate-marker"

        def __init__(self) -> None:
            self.closed = False
            self.entered = False
            self._chunks = iter([_chunk("partial"), _chunk(" ignored")])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._chunks)

        def close(self) -> None:
            self.closed = True

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            self.close()

    delegated_stream = ContextStream()

    def create(_self, **_kwargs):
        return delegated_stream

    monkeypatch.setattr(Completions, "create", create)
    monkeypatch.setattr(openai_instrumentation, "_is_stream", lambda _value: True)
    instrumentor.activate()
    install_capture()

    with Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
    ) as stream:
        assert stream.marker == "delegate-marker"
        assert next(stream) == _chunk("partial")

    assert delegated_stream.entered is True
    assert delegated_stream.closed is True
    assert len(captured) == 1
    assert captured[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "partial"
    )


def test_sync_provider_error_retains_precise_status(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class RateLimited(Exception):
        status_code = 429

    def create(_self, **_kwargs):
        raise RateLimited("Error code: 429 - deterministic provider limit")

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()

    with pytest.raises(RateLimited):
        Completions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "fail"}],
        )

    assert len(captured) == 1
    span = captured[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["http.response.status_code"] == 429
    assert "deterministic provider limit" in span.attributes["error.message"]


def test_nonstream_status_does_not_require_message_regex_or_exception_str(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class HostileRateLimit(Exception):
        status_code = 429

        def __str__(self) -> str:
            raise RuntimeError("hostile __str__ invoked")

    def create(_self, **_kwargs):
        raise HostileRateLimit("provider limit")

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()

    with pytest.raises(HostileRateLimit):
        Completions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "fail safely"}],
        )

    assert len(captured) == 1
    span = captured[0]
    assert span.attributes["http.response.status_code"] == 429
    assert span.attributes["error.message"] == "provider limit"
    assert span.status.status_code is StatusCode.ERROR


def test_explicit_generic_500_is_not_overridden_by_error_text(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class ProviderFailure(Exception):
        status_code = 500

    def create(_self, **_kwargs):
        raise ProviderFailure("OpenRouter provider cache HTTP 404 during formatting")

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()

    with pytest.raises(ProviderFailure):
        Completions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "fail safely"}],
        )

    assert len(captured) == 1
    assert captured[0].attributes["http.response.status_code"] == 500


def test_hostile_status_property_never_masks_original_provider_error(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class HostileStatus(Exception):
        @property
        def status_code(self):
            raise RuntimeError("hostile status property invoked")

        @property
        def response(self):
            raise RuntimeError("hostile response property invoked")

    def create(_self, **_kwargs):
        raise HostileStatus("safe diagnostic")

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()

    with pytest.raises(HostileStatus, match="safe diagnostic"):
        Completions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "fail safely"}],
        )

    assert len(captured) == 1
    assert captured[0].attributes["http.response.status_code"] == 500


def test_async_stream_finalizes_and_preserves_error_status(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class AsyncFailingStream:
        def __aiter__(self):
            async def iterate():
                yield _chunk("partial")
                error = RuntimeError("HTTP 503 OpenRouter unavailable")
                error.status_code = 503
                raise error

            return iterate()

    def create(_self, **_kwargs):
        return AsyncFailingStream()

    monkeypatch.setattr(AsyncCompletions, "create", create)
    instrumentor.activate()
    install_capture()

    async def consume() -> None:
        stream = await AsyncCompletions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        with pytest.raises(RuntimeError, match="OpenRouter unavailable"):
            async for _chunk_value in stream:
                await asyncio.sleep(0)

    asyncio.run(consume())

    assert len(captured) == 1
    span = captured[0]
    assert span.attributes[GEN_AI_REQUEST_STREAM] is True
    assert span.attributes["http.response.status_code"] == 503
    assert span.status.status_code is StatusCode.ERROR


def test_async_for_delegates_iterator_and_finalizes_full_output(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans
    chunks = [
        _chunk("async "),
        _chunk("complete"),
        _chunk(
            "",
            usage=CompletionUsage(
                prompt_tokens=8,
                completion_tokens=2,
                total_tokens=10,
            ),
        ),
    ]

    class AsyncIterableOnly:
        def __aiter__(self):
            async def iterate():
                for chunk in chunks:
                    yield chunk

            return iterate()

    def create(_self, **_kwargs):
        return AsyncIterableOnly()

    monkeypatch.setattr(AsyncCompletions, "create", create)
    instrumentor.activate()
    install_capture()

    async def consume() -> list[ChatCompletionChunk]:
        stream = await AsyncCompletions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        return [chunk async for chunk in stream]

    assert asyncio.run(consume()) == chunks
    assert len(captured) == 1
    attrs = dict(captured[0].attributes)
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == ("async complete")
    assert attrs["gen_ai.usage.input_tokens"] == 8
    assert attrs["gen_ai.usage.output_tokens"] == 2


def test_async_stream_aclose_finalizes_once_and_closes_delegate(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class CloseAwareAsyncStream:
        def __init__(self) -> None:
            self.closed = False
            self._chunks = iter([_chunk("partial"), _chunk(" ignored")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def aclose(self) -> None:
            self.closed = True

    delegated_stream = CloseAwareAsyncStream()

    def create(_self, **_kwargs):
        return delegated_stream

    monkeypatch.setattr(AsyncCompletions, "create", create)
    instrumentor.activate()
    install_capture()

    async def consume_one() -> None:
        stream = await AsyncCompletions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        assert await anext(stream) == _chunk("partial")
        await stream.aclose()

    asyncio.run(consume_one())

    assert delegated_stream.closed is True
    assert len(captured) == 1
    attrs = dict(captured[0].attributes)
    assert attrs[GEN_AI_REQUEST_STREAM] is True
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "partial"


def test_async_stream_proxy_preserves_context_manager(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class AsyncContextStream:
        def __init__(self) -> None:
            self.entered = False
            self.closed = False
            self._chunks = iter([_chunk("partial"), _chunk(" ignored")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def close(self) -> None:
            self.closed = True

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
            await self.close()

    delegated_stream = AsyncContextStream()

    def create(_self, **_kwargs):
        return delegated_stream

    monkeypatch.setattr(AsyncCompletions, "create", create)
    instrumentor.activate()
    install_capture()

    async def consume() -> None:
        stream = await AsyncCompletions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        async with stream as entered:
            assert entered is stream
            assert await anext(entered) == _chunk("partial")

    asyncio.run(consume())

    assert delegated_stream.entered is True
    assert delegated_stream.closed is True
    assert len(captured) == 1
    assert captured[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "partial"
    )


def test_async_stream_cancellation_emits_once_and_closes_delegate(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    class CancelledStream:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError("consumer cancelled")

        async def close(self) -> None:
            self.closed = True

    delegated_stream = CancelledStream()

    def create(_self, **_kwargs):
        return delegated_stream

    monkeypatch.setattr(AsyncCompletions, "create", create)
    instrumentor.activate()
    install_capture()

    async def consume() -> None:
        stream = await AsyncCompletions.create(
            object(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
        )
        with pytest.raises(asyncio.CancelledError):
            await anext(stream)

    asyncio.run(consume())

    assert delegated_stream.closed is True
    assert len(captured) == 1
    assert captured[0].status.status_code is StatusCode.ERROR
    assert captured[0].attributes["http.response.status_code"] == 500
    assert captured[0].attributes["error.message"] == "consumer cancelled"


def test_later_openai_owner_join_and_leave_keeps_openrouter_active(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, captured, install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return _response("joined owner")

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    install_capture()
    openrouter_wrapper = Completions.create
    independent = DirectOpenAIInstrumentor()

    independent.activate()
    assert Completions.create is openrouter_wrapper
    independent.deactivate()
    assert Completions.create is openrouter_wrapper

    result = Completions.create(
        object(),
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "still traced"}],
    )
    assert result.choices[0].message.content == "joined owner"
    assert len(captured) == 1


def test_openrouter_leaves_before_independent_openai_owner_without_harming_it(
    monkeypatch,
    capture_delegate_spans,
) -> None:
    instrumentor, _captured, _install_capture = capture_delegate_spans

    def create(_self, **_kwargs):
        return _response("independent owner")

    monkeypatch.setattr(Completions, "create", create)
    instrumentor.activate()
    independent = DirectOpenAIInstrumentor()
    independent.activate()

    instrumentor.deactivate()
    assert Completions.create is not create
    assert independent._is_instrumented is True

    independent.deactivate()
    assert Completions.create is create
