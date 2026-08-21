"""Exported-span regressions against the real current OpenAI SDK types."""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest
from openai import AsyncOpenAI, AuthenticationError, OpenAI
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from respan_instrumentation_openai import _instrumentation as instrumentation
from respan_instrumentation_openai import _otel_emitter as emitter
from respan_instrumentation_openai._instrumentation import OpenAIInstrumentor


class Answer(BaseModel):
    answer: str


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


@pytest.fixture
def captured(monkeypatch):
    spans = []
    monkeypatch.setattr(emitter, "inject_span", lambda span: spans.append(span))
    return spans


def _chat_payload(*, parsed: bool = False, tool: bool = False) -> dict:
    message: dict = {
        "role": "assistant",
        "content": '{"answer":"yes"}' if parsed else "deterministic chat",
    }
    if tool:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        }
    return {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4.1-nano",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool else "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }


def _response_payload(*, parsed: bool = False, tool: bool = False) -> dict:
    output: list[dict]
    if tool:
        output = [
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
                "status": "completed",
            }
        ]
    else:
        output = [
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"answer":"yes"}'
                        if parsed
                        else "deterministic response",
                        "annotations": [],
                    }
                ],
            }
        ]
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4.1-nano",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1,
        "top_p": 1,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
    }


def _sync_handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    if request.url.path.endswith("/chat/completions"):
        if body.get("messages", [{}])[-1].get("content") == "fail":
            return httpx2.Response(
                401,
                json={
                    "error": {
                        "message": "invalid test credential",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx2.Response(
            200,
            json=_chat_payload(
                parsed="response_format" in body,
                tool=bool(body.get("tools")) and "response_format" not in body,
            ),
        )
    if request.url.path.endswith("/responses"):
        return httpx2.Response(
            200,
            json=_response_payload(
                parsed=bool(body.get("text", {}).get("format")),
                tool=bool(body.get("tools")),
            ),
        )
    raise AssertionError(request.url.path)


async def _async_handler(request: httpx2.Request) -> httpx2.Response:
    return _sync_handler(request)


def _sync_client(handler=_sync_handler) -> OpenAI:
    return OpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _async_client(handler=_async_handler) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://openai.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def test_real_sync_chat_create_parse_tool_and_401_export(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _sync_client()
    try:
        chat = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
        )
        parsed = client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "structured"}],
            response_format=Answer,
        )
        tool = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        with pytest.raises(AuthenticationError):
            client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": "fail"}],
            )
    finally:
        client.close()
        instrumentor.deactivate()

    assert chat.choices[0].message.content == "deterministic chat"
    assert parsed.choices[0].message.parsed == Answer(answer="yes")
    assert tool.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert len(captured) == 4
    assert [span.name for span in captured] == ["openai.chat"] * 4
    assert captured[-1].status.status_code is StatusCode.ERROR
    assert captured[-1].attributes["status_code"] == 401
    assert "invalid test credential" in captured[-1].attributes["error.message"]
    assert json.loads(captured[1].attributes["traceloop.entity.output"])["parsed"] == {
        "answer": "yes"
    }
    assert (
        json.loads(captured[2].attributes["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "call_weather"
    )


def test_real_sync_responses_create_parse_and_tool_export(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _sync_client()
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]
    try:
        response = client.responses.create(model="gpt-4.1-nano", input="hello")
        parsed = client.responses.parse(
            model="gpt-4.1-nano", input="structured", text_format=Answer
        )
        tool = client.responses.create(
            model="gpt-4.1-nano", input="weather", tools=tools
        )
    finally:
        client.close()
        instrumentor.deactivate()

    assert response.output_text == "deterministic response"
    assert parsed.output_parsed == Answer(answer="yes")
    assert tool.output[0].call_id == "call_weather"
    assert len(captured) == 3
    assert [span.name for span in captured] == ["openai.response"] * 3
    assert captured[1].attributes["gen_ai.usage.input_tokens"] == 5
    assert (
        json.loads(captured[2].attributes["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "call_weather"
    )


@pytest.mark.asyncio
async def test_real_async_chat_and_responses_parse_export(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _async_client()
    try:
        chat = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
        )
        parsed_chat = await client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "structured"}],
            response_format=Answer,
        )
        response = await client.responses.create(model="gpt-4.1-nano", input="hello")
        parsed_response = await client.responses.parse(
            model="gpt-4.1-nano", input="structured", text_format=Answer
        )
    finally:
        await client.close()
        instrumentor.deactivate()

    assert chat.choices[0].message.content == "deterministic chat"
    assert parsed_chat.choices[0].message.parsed == Answer(answer="yes")
    assert response.output_text == "deterministic response"
    assert parsed_response.output_parsed == Answer(answer="yes")
    assert [span.name for span in captured] == [
        "openai.chat",
        "openai.chat",
        "openai.response",
        "openai.response",
    ]


def test_real_chat_stream_exports_one_bounded_final_span(captured):
    chunks = [
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]

    def stream_handler(request: httpx2.Request) -> httpx2.Response:
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        content += "data: [DONE]\n\n"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content.encode(),
        )

    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _sync_client(stream_handler)
    try:
        stream = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = "".join(
            chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
        )
    finally:
        client.close()
        instrumentor.deactivate()

    assert text == "Hello"
    assert len(captured) == 1
    span = captured[0]
    assert span.attributes["gen_ai.is_streaming"] is True
    assert span.attributes["gen_ai.completion.0.content"] == "Hello"
    assert span.attributes["llm.usage.total_tokens"] == 6


@pytest.mark.asyncio
async def test_real_async_chat_stream_completion_exports_usage(captured):
    chunks = [
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]

    async def stream_handler(request: httpx2.Request) -> httpx2.Response:
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        content += "data: [DONE]\n\n"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content.encode(),
        )

    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _async_client(stream_handler)
    try:
        stream = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = ""
        async for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].delta.content or ""
    finally:
        await client.close()
        instrumentor.deactivate()

    assert text == "Hello"
    assert len(captured) == 1
    assert captured[0].attributes["gen_ai.completion.0.content"] == "Hello"
    assert captured[0].attributes["llm.usage.total_tokens"] == 6


@pytest.mark.asyncio
async def test_real_async_chat_stream_close_exports_one_partial_span(captured):
    chunks = [
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    async def stream_handler(request: httpx2.Request) -> httpx2.Response:
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        content += "data: [DONE]\n\n"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content.encode(),
        )

    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _async_client(stream_handler)
    try:
        stream = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        first = await stream.__anext__()
        await stream.close()
        await stream.aclose()
        assert stream.response.is_closed
    finally:
        await client.close()
        instrumentor.deactivate()

    assert first.choices[0].delta.content == "Hel"
    assert len(captured) == 1
    span = captured[0]
    assert span.attributes["gen_ai.is_streaming"] is True
    assert span.attributes["gen_ai.completion.0.content"] == "Hel"


@pytest.mark.asyncio
async def test_real_async_chat_stream_context_exit_closes_source(captured):
    chunks = [
        {
            "id": "chat_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "partial"},
                    "finish_reason": None,
                }
            ],
        }
    ]

    async def stream_handler(request: httpx2.Request) -> httpx2.Response:
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        content += "data: [DONE]\n\n"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content.encode(),
        )

    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _async_client(stream_handler)
    try:
        stream = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        async with stream:
            await stream.__anext__()
        assert stream.response.is_closed
    finally:
        await client.close()
        instrumentor.deactivate()

    assert len(captured) == 1
    assert captured[0].attributes["gen_ai.completion.0.content"] == "partial"


@pytest.mark.asyncio
async def test_real_async_chat_stream_cancellation_closes_source(captured):
    started = asyncio.Event()

    class BlockingStream(httpx2.AsyncByteStream):
        def __init__(self) -> None:
            self.close_count = 0

        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self) -> None:
            self.close_count += 1

    source = BlockingStream()

    async def stream_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=source,
        )

    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _async_client(stream_handler)
    try:
        stream = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(started.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert stream.response.is_closed
        assert source.close_count == 1
    finally:
        await client.close()
        instrumentor.deactivate()

    assert len(captured) == 1
    assert captured[0].status.status_code is StatusCode.ERROR
    assert captured[0].attributes["status_code"] == 499
