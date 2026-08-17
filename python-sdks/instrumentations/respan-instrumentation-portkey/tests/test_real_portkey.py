from __future__ import annotations

import json
from collections import Counter

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import StatusCode
from portkey_ai import AsyncPortkey, Portkey
from portkey_ai._vendor.openai import AuthenticationError
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_instrumentation_portkey import PortkeyInstrumentor, _instrumentation
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


def _payload(request: httpx.Request) -> dict:
    return json.loads(request.content.decode())


def _response(request: httpx.Request) -> httpx.Response:
    body = _payload(request)
    model = body.get("model", "local-model")
    if model == "error-401":
        return httpx.Response(
            401,
            json={"error": {"message": "deterministic authorization failure"}},
            request=request,
        )
    if body.get("stream"):
        events = [
            {
                "id": "stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Portkey "}}
                ],
            },
            {
                "id": "stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "stream."},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
        ]
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        content += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            text=content,
            headers={"content-type": "text/event-stream"},
            request=request,
        )
    message: dict = {"role": "assistant", "content": "Portkey response."}
    finish = "stop"
    if body.get("tools"):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "weather-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Tokyo"}'},
                }
            ],
        }
        finish = "tool_calls"
    return httpx.Response(
        200,
        json={
            "id": "chat-1",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
        },
        request=request,
    )


@pytest.mark.asyncio
async def test_real_current_portkey_sync_async_stream_tool_and_error(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_instrumentation.trace, "get_tracer_provider", lambda: provider)
    OpenInferenceInstrumentor._translator_registered = False
    OpenInferenceInstrumentor._translator = None
    OpenInferenceInstrumentor._active_span_processors = []
    instrumentor = PortkeyInstrumentor()
    instrumentor.activate()
    tracer = provider.get_tracer("respan.portkey.test")
    sync_http = httpx.Client(transport=httpx.MockTransport(_response))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(_response))
    sync_client = Portkey(
        api_key="local-key", base_url="https://portkey.test", http_client=sync_http
    )
    async_client = AsyncPortkey(
        api_key="local-key", base_url="https://portkey.test", http_client=async_http
    )
    try:
        with tracer.start_as_current_span("sync.root"):
            sync_client.chat.completions.create(
                model="local-model", messages=[{"role": "user", "content": "sync"}]
            )
        with tracer.start_as_current_span("async.root"):
            await async_client.chat.completions.create(
                model="local-model", messages=[{"role": "user", "content": "async"}]
            )
        with tracer.start_as_current_span("stream.root"):
            stream = sync_client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": "stream"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            assert (
                "".join(
                    chunk.choices[0].delta.content or ""
                    for chunk in stream
                    if chunk.choices
                )
                == "Portkey stream."
            )
        with tracer.start_as_current_span("tool.root"):
            sync_client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": "weather"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        with (
            tracer.start_as_current_span("error.root"),
            pytest.raises(AuthenticationError),
        ):
            sync_client.chat.completions.create(
                model="error-401", messages=[{"role": "user", "content": "fail"}]
            )
    finally:
        sync_client.close()
        await async_client.close()
        instrumentor.deactivate()
        provider.force_flush()

    spans = list(exporter.get_finished_spans())
    names = Counter(span.name for span in spans)
    assert names == Counter(
        {
            "sync.root": 1,
            "async.root": 1,
            "stream.root": 1,
            "tool.root": 1,
            "error.root": 1,
            "Completions": 4,
            "AsyncCompletions": 1,
        }
    )
    assert len({span.context.span_id for span in spans}) == len(spans)
    all_ids = {span.context.span_id for span in spans}
    for span in spans:
        if span.parent is not None:
            assert span.parent.span_id in all_ids

    chats = [span for span in spans if span.name in {"Completions", "AsyncCompletions"}]
    assert all(span.attributes[RESPAN_LOG_TYPE] == "chat" for span in chats)
    assert all(span.attributes["gen_ai.provider.name"] == "portkey" for span in chats)
    assert all(
        TLSpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes for span in chats
    )
    stream_span = next(
        span
        for span in chats
        if span.attributes.get(TLSpanAttributes.LLM_IS_STREAMING) is True
    )
    assert stream_span.attributes.get(TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7, (
        stream_span.attributes
    )
    assert stream_span.attributes[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
    assert (
        "Portkey stream."
        in stream_span.attributes[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    tool_candidates = [
        span
        for span in chats
        if TLSpanAttributes.LLM_REQUEST_FUNCTIONS in span.attributes
    ]
    assert tool_candidates, [dict(span.attributes) for span in chats]
    tool_span = tool_candidates[0]
    assert (
        json.loads(tool_span.attributes[TLSpanAttributes.LLM_REQUEST_FUNCTIONS])[0][
            "function"
        ]["name"]
        == "weather"
    )
    calls = json.loads(
        tool_span.attributes[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert calls[0]["id"] == "weather-1"
    error_span = next(
        span for span in chats if span.status.status_code is StatusCode.ERROR
    )
    assert error_span.attributes["status_code"] == 401
    assert error_span.attributes["http.response.status_code"] == 401
    assert "authorization failure" in error_span.attributes[ERROR_MESSAGE_ATTR]
    provider.shutdown()
