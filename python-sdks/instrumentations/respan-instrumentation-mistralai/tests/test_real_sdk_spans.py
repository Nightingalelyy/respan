from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respan_instrumentation_mistralai._instrumentation as instrumentation
from mistralai.client import Mistral
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_mistralai import MistralAIInstrumentor
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.constants.tracing import SAMPLE_RATE_ATTR

MODEL = "mistral-small-latest"
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return deterministic weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
OFF_CONTRACT_ALIASES = {
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


def _completion_response(
    *,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "cmpl_deterministic",
        "object": "chat.completion",
        "model": MODEL,
        "created": 1_710_000_000,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


def _stream_body(
    *,
    first: str,
    second: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    chunks = [
        {
            "id": "stream_deterministic",
            "object": "chat.completion.chunk",
            "model": MODEL,
            "created": 1_710_000_000,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": first},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "stream_deterministic",
            "object": "chat.completion.chunk",
            "model": MODEL,
            "created": 1_710_000_000,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": second},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def _sync_client(handler: Callable[[httpx.Request], httpx.Response]) -> Mistral:
    return Mistral(
        api_key="test-secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _async_client(handler: Callable[[httpx.Request], httpx.Response]) -> Mistral:
    async def async_handler(request: httpx.Request) -> httpx.Response:
        return handler(request)

    return Mistral(
        api_key="test-secret-key",
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(async_handler)),
    )


def _mistral_spans(spans: tuple[ReadableSpan, ...]) -> list[ReadableSpan]:
    return [
        span
        for span in spans
        if span.instrumentation_scope.name
        == instrumentation.OPENINFERENCE_MISTRALAI_MODULE
    ]


def test_current_sdk_exports_complete_canonical_spans(monkeypatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_translator_registered", False)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_translator", None)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_active_span_processors", [])
    monkeypatch.setattr(instrumentation, "_SHARED_DELEGATE", None)
    monkeypatch.setattr(instrumentation, "_SHARED_CLEANUP_PROCESSOR", None)
    monkeypatch.setattr(instrumentation, "_SHARED_STREAM_GUARD_PATCH", None)
    monkeypatch.setattr(instrumentation, "_SHARED_INSTRUMENTOR_KWARGS", None)
    monkeypatch.setattr(instrumentation, "_SHARED_REFCOUNT", 0)

    instrumentor = MistralAIInstrumentor()
    instrumentor.activate()
    parent_tracer = provider.get_tracer("test.mistralai.parents")
    parent_ids: set[int] = set()

    def sync_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompt = payload["messages"][-1]["content"]
        if prompt == "sync stream":
            return httpx.Response(
                200,
                content=_stream_body(
                    first="sync ",
                    second="complete",
                    prompt_tokens=11,
                    completion_tokens=3,
                ),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        if prompt == "weather in Paris":
            return httpx.Response(
                200,
                json=_completion_response(
                    content="",
                    prompt_tokens=13,
                    completion_tokens=5,
                    tool_calls=[
                        {
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                ),
                request=request,
            )
        if prompt == "provider failure":
            return httpx.Response(
                401,
                json={"message": "invalid api key", "request_id": "request_123"},
                request=request,
            )
        if prompt == "application failure":
            raise RuntimeError("deterministic Mistral transport failure")
        return httpx.Response(
            200,
            json=_completion_response(
                content="sync complete",
                prompt_tokens=7,
                completion_tokens=2,
            ),
            request=request,
        )

    def async_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_stream_body(
                first="async ",
                second="complete",
                prompt_tokens=17,
                completion_tokens=4,
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    with _sync_client(sync_handler) as client:
        with parent_tracer.start_as_current_span("root.sync") as parent:
            parent_ids.add(parent.get_span_context().span_id)
            response = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": "sync complete"}],
            )
            assert response.choices[0].message.content == "sync complete"

        with parent_tracer.start_as_current_span("root.sync-stream") as parent:
            parent_ids.add(parent.get_span_context().span_id)
            chunks = client.chat.stream(
                model=MODEL,
                messages=[{"role": "user", "content": "sync stream"}],
            )
            assert "".join(
                event.data.choices[0].delta.content or "" for event in chunks
            ) == ("sync complete")

        with parent_tracer.start_as_current_span("root.tool") as parent:
            parent_ids.add(parent.get_span_context().span_id)
            tool_response = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": "weather in Paris"}],
                tools=[TOOL],
                tool_choice="auto",
            )
            assert tool_response.choices[0].message.tool_calls[0].id == "call_weather_1"

        with (
            pytest.raises(Exception, match="Status 401"),
            parent_tracer.start_as_current_span("root.provider-error") as parent,
        ):
            parent_ids.add(parent.get_span_context().span_id)
            client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": "provider failure"}],
            )

        with (
            pytest.raises(RuntimeError, match="deterministic Mistral"),
            parent_tracer.start_as_current_span("root.application-error") as parent,
        ):
            parent_ids.add(parent.get_span_context().span_id)
            client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": "application failure"}],
            )

    async def run_async_stream() -> None:
        async with _async_client(async_handler) as client:
            with parent_tracer.start_as_current_span("root.async-stream") as parent:
                parent_ids.add(parent.get_span_context().span_id)
                chunks = await client.chat.stream_async(
                    model=MODEL,
                    messages=[{"role": "user", "content": "async stream"}],
                )
                content = ""
                async for event in chunks:
                    content += event.data.choices[0].delta.content or ""
                assert content == "async complete"

    asyncio.run(run_async_stream())

    instrumentor.deactivate()
    provider.force_flush()

    finished = exporter.get_finished_spans()
    mistral_spans = _mistral_spans(finished)
    assert len(mistral_spans) == 6
    assert len({span.context.span_id for span in mistral_spans}) == 6
    assert {span.parent.span_id for span in mistral_spans} == parent_ids

    parent_spans = {
        span.name: span
        for span in finished
        if span.instrumentation_scope.name == "test.mistralai.parents"
    }
    assert parent_spans["root.provider-error"].status.status_code is (
        trace.StatusCode.ERROR
    )
    assert parent_spans["root.application-error"].status.status_code is (
        trace.StatusCode.ERROR
    )

    by_prompt = {
        span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]: span
        for span in mistral_spans
    }
    sync_span = by_prompt["sync complete"]
    assert sync_span.attributes[SpanAttributes.LLM_IS_STREAMING] is False
    assert sync_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "sync complete"
    )
    assert sync_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert sync_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 2

    for prompt, content, prompt_tokens, completion_tokens in (
        ("sync stream", "sync complete", 11, 3),
        ("async stream", "async complete", 17, 4),
    ):
        span = by_prompt[prompt]
        assert span.attributes[SpanAttributes.LLM_IS_STREAMING] is True
        assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == content
        assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == prompt_tokens
        assert span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == (
            completion_tokens
        )
        assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == (
            prompt_tokens + completion_tokens
        )

    tool_span = by_prompt["weather in Paris"]
    assert json.loads(tool_span.attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [
        TOOL
    ]
    assert json.loads(
        tool_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    ) == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        }
    ]

    error_span = by_prompt["provider failure"]
    assert error_span.status.status_code is trace.StatusCode.ERROR
    assert error_span.attributes["status_code"] == 401
    assert json.loads(
        error_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "error": "SDKError",
        "message": "Mistral request failed with status 401",
        "status": "error",
        "status_code": 401,
    }

    application_error_span = by_prompt["application failure"]
    assert application_error_span.status.status_code is trace.StatusCode.ERROR
    assert application_error_span.attributes["status_code"] == 500
    assert json.loads(
        application_error_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "error": "RuntimeError",
        "message": "Mistral request failed",
        "status": "error",
        "status_code": 500,
    }

    for span in mistral_spans:
        attrs = dict(span.attributes)
        assert OFF_CONTRACT_ALIASES.isdisjoint(attrs)
        serialized_attrs = json.dumps(attrs, default=str)
        assert "test-secret-key" not in serialized_attrs
        assert "authorization" not in serialized_attrs.lower()

    native_sdk_spans = [
        span
        for span in finished
        if span.instrumentation_scope.name == instrumentation.MISTRALAI_SDK_TRACER_NAME
    ]
    assert all(span.attributes[SAMPLE_RATE_ATTR] == 0 for span in native_sdk_spans)


def test_current_sdk_finalizes_interrupted_streams(monkeypatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_translator_registered", False)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_translator", None)
    monkeypatch.setattr(OpenInferenceInstrumentor, "_active_span_processors", [])
    monkeypatch.setattr(instrumentation, "_SHARED_DELEGATE", None)
    monkeypatch.setattr(instrumentation, "_SHARED_CLEANUP_PROCESSOR", None)
    monkeypatch.setattr(instrumentation, "_SHARED_STREAM_GUARD_PATCH", None)
    monkeypatch.setattr(instrumentation, "_SHARED_INSTRUMENTOR_KWARGS", None)
    monkeypatch.setattr(instrumentation, "_SHARED_REFCOUNT", 0)

    instrumentor = MistralAIInstrumentor()
    instrumentor.activate()

    def sync_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_stream_body(
                first="partial ",
                second="content",
                prompt_tokens=5,
                completion_tokens=2,
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    with _sync_client(sync_handler) as client:
        early_stream = client.chat.stream(
            model=MODEL,
            messages=[{"role": "user", "content": "sync early close"}],
        )
        assert next(early_stream).data.choices[0].delta.content == "partial "
        early_stream.close()

        with (
            pytest.raises(GeneratorExit),
            client.chat.stream(
                model=MODEL,
                messages=[{"role": "user", "content": "sync generator exit"}],
            ) as generator_exit_stream,
        ):
            assert (
                next(generator_exit_stream).data.choices[0].delta.content == "partial "
            )
            raise GeneratorExit

    class BlockingAsyncStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False
            self.close_calls = 0
            self.wait_forever = asyncio.Event()

        async def __aiter__(self):
            partial = {
                "id": "stream_cancelled",
                "object": "chat.completion.chunk",
                "model": MODEL,
                "created": 1_710_000_000,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "partial "},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(partial)}\n\n".encode()
            await self.wait_forever.wait()

        async def aclose(self) -> None:
            self.closed = True
            self.close_calls += 1
            if self.close_calls == 1:
                raise asyncio.CancelledError("source cleanup cancellation")

    blocking_stream: BlockingAsyncStream | None = None

    def async_handler(request: httpx.Request) -> httpx.Response:
        nonlocal blocking_stream
        payload = json.loads(request.content)
        prompt = payload["messages"][-1]["content"]
        if prompt == "async cancellation":
            blocking_stream = BlockingAsyncStream()
            return httpx.Response(
                200,
                stream=blocking_stream,
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return sync_handler(request)

    async def run_async_interruptions() -> None:
        async with _async_client(async_handler) as client:
            early_stream = await client.chat.stream_async(
                model=MODEL,
                messages=[{"role": "user", "content": "async early close"}],
            )
            first = await anext(early_stream)
            assert first.data.choices[0].delta.content == "partial "
            await early_stream.aclose()

            cancelled_stream = await client.chat.stream_async(
                model=MODEL,
                messages=[{"role": "user", "content": "async cancellation"}],
            )
            first = await anext(cancelled_stream)
            assert first.data.choices[0].delta.content == "partial "
            pending = asyncio.create_task(anext(cancelled_stream))
            await asyncio.sleep(0)
            pending.cancel("original stream cancellation")
            with pytest.raises(
                asyncio.CancelledError,
                match="original stream cancellation",
            ):
                await pending

    asyncio.run(run_async_interruptions())
    instrumentor.deactivate()
    provider.force_flush()

    assert blocking_stream is not None
    assert blocking_stream.closed is True
    spans = _mistral_spans(exporter.get_finished_spans())
    assert len(spans) == 4
    by_prompt = {
        span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]: span
        for span in spans
    }
    for prompt in (
        "sync early close",
        "sync generator exit",
        "async early close",
    ):
        span = by_prompt[prompt]
        assert span.status.status_code is trace.StatusCode.OK
        assert span.attributes[SpanAttributes.LLM_IS_STREAMING] is True

    cancelled_span = by_prompt["async cancellation"]
    assert cancelled_span.status.status_code is trace.StatusCode.ERROR
    assert cancelled_span.attributes["status_code"] == 500
    assert json.loads(
        cancelled_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "error": "CancelledError",
        "message": "Mistral request failed",
        "status": "error",
        "status_code": 500,
    }
