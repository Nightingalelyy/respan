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


def _events(kind):
    if kind == "chat":
        return [
            {
                "id": "chat_stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4.1-nano",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
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
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        ]
    final = _response_payload()
    final["output"][0]["content"][0]["text"] = "Hello world"
    initial = {**final, "status": "in_progress", "output": [], "usage": None}
    message = {**final["output"][0], "status": "in_progress", "content": []}
    part = {"type": "output_text", "text": "", "annotations": []}
    return [
        {"type": "response.created", "response": initial, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "item": message,
            "output_index": 0,
            "sequence_number": 1,
        },
        {
            "type": "response.content_part.added",
            "part": part,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
        },
        {
            "type": "response.output_text.delta",
            "delta": " world",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 4,
        },
        {"type": "response.completed", "response": final, "sequence_number": 5},
    ]


def _sse(events):
    return (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        + "data: [DONE]\n\n"
    ).encode()


@pytest.mark.parametrize("kind", ["chat", "response"])
@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_current_sync_stream_surface_preserves_results_and_finalizes_on_close(
    captured, kind, helper, partial
):
    class Source(httpx2.SyncByteStream):
        close_count = 0

        def __iter__(self):
            yield _sse(_events(kind))

        def close(self):
            self.close_count += 1

    source = Source()
    client = _sync_client(
        lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source
        )
    )
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    resource = client.chat.completions if kind == "chat" else client.responses
    kwargs = {"model": "gpt-4.1-nano"}
    if kind == "chat":
        kwargs["messages"] = [{"role": "user", "content": "hello"}]
    else:
        kwargs["input"] = "hello"
    manager = (
        resource.stream(**kwargs) if helper else resource.create(**kwargs, stream=True)
    )
    try:
        with manager as stream:
            for event in stream:
                event_type = getattr(event, "type", None)
                if partial and (
                    event_type in {"content.delta", "response.output_text.delta"}
                    or (kind == "chat" and not helper)
                ):
                    break
            if helper and not partial:
                final = (
                    stream.get_final_completion()
                    if kind == "chat"
                    else stream.get_final_response()
                )
                text = (
                    final.choices[0].message.content
                    if kind == "chat"
                    else final.output_text
                )
                assert text == "Hello world"
        stream.close()
        assert source.close_count == 1
        assert len(captured) == 1
    finally:
        client.close()
        instrumentor.deactivate()
    attrs = captured[0].attributes
    assert attrs["gen_ai.completion.0.content"] == (
        "Hello" if partial else "Hello world"
    )
    assert attrs["gen_ai.is_streaming"] is True
    if not partial:
        assert attrs["llm.usage.total_tokens"] == (6 if kind == "chat" else 8)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "response"])
@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_current_async_stream_surface_preserves_results_and_finalizes_on_close(
    captured, kind, helper, partial
):
    class Source(httpx2.AsyncByteStream):
        close_count = 0

        async def __aiter__(self):
            yield _sse(_events(kind))

        async def aclose(self):
            self.close_count += 1

    source = Source()
    client = _async_client(
        lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source
        )
    )
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    resource = client.chat.completions if kind == "chat" else client.responses
    kwargs = {"model": "gpt-4.1-nano"}
    if kind == "chat":
        kwargs["messages"] = [{"role": "user", "content": "hello"}]
    else:
        kwargs["input"] = "hello"
    manager = (
        resource.stream(**kwargs)
        if helper
        else await resource.create(**kwargs, stream=True)
    )
    try:
        async with manager as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if partial and (
                    event_type in {"content.delta", "response.output_text.delta"}
                    or (kind == "chat" and not helper)
                ):
                    break
            if helper and not partial:
                final = (
                    await stream.get_final_completion()
                    if kind == "chat"
                    else await stream.get_final_response()
                )
                text = (
                    final.choices[0].message.content
                    if kind == "chat"
                    else final.output_text
                )
                assert text == "Hello world"
        await stream.close()
        assert source.close_count == 1
        assert len(captured) == 1
    finally:
        await client.close()
        instrumentor.deactivate()
    attrs = captured[0].attributes
    assert attrs["gen_ai.completion.0.content"] == (
        "Hello" if partial else "Hello world"
    )
    assert attrs["gen_ai.is_streaming"] is True
    if not partial:
        assert attrs["llm.usage.total_tokens"] == (6 if kind == "chat" else 8)


def test_response_partial_tools_are_kept_separate_from_output_text(captured):
    events = _events("response")[:1] + [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "sequence_number": 2,
            "delta": '{"city":"Paris"}',
        },
        {
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "private reasoning",
        },
    ]
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    with (
        _sync_client(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(events)
            )
        ) as client,
        client.responses.create(
            model="gpt-4.1-nano", input="weather", stream=True
        ) as stream,
    ):
        list(stream)
    assert len(captured) == 1
    attrs = captured[0].attributes
    assert not attrs.get("gen_ai.completion.0.content")
    calls = json.loads(attrs["gen_ai.completion.0.tool_calls"])
    assert calls[0]["id"] == "call_weather"
    assert calls[0]["function"]["arguments"] == '{"city":"Paris"}'
    assert attrs["gen_ai.response.id"] == "resp_1"


def test_response_failed_event_marks_span_error_without_changing_sdk_events(captured):
    failed = {
        **_response_payload(),
        "status": "failed",
        "error": {
            "code": "server_error",
            "message": "test provider failure api_key=sk-private",
        },
    }
    events = [{"type": "response.failed", "response": failed, "sequence_number": 1}]
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    with (
        _sync_client(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(events)
            )
        ) as client,
        client.responses.create(
            model="gpt-4.1-nano", input="hello", stream=True
        ) as stream,
    ):
        assert [event.type for event in stream] == ["response.failed"]
    assert len(captured) == 1
    assert captured[0].status.status_code is StatusCode.ERROR
    assert (
        captured[0].attributes["error.message"]
        == "test provider failure api_key=[REDACTED]"
    )
    assert captured[0].attributes["error.type"] == "server_error"


@pytest.mark.asyncio
async def test_real_sync_and_async_completions_and_embeddings(captured):
    def handler(request):
        if request.url.path.endswith("/embeddings"):
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        return httpx2.Response(
            200,
            json={
                "id": "cmpl_1",
                "object": "text_completion",
                "created": 1,
                "model": "davinci-002",
                "choices": [
                    {
                        "index": 0,
                        "text": "completed",
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    with _sync_client(handler) as client:
        assert (
            client.completions.create(model="davinci-002", prompt="hello")
            .choices[0]
            .text
            == "completed"
        )
        assert client.embeddings.create(
            model="text-embedding-3-small", input="hello", encoding_format="float"
        ).data[0].embedding == [0.1, 0.2]
    async with _async_client(handler) as client:
        assert (
            await client.completions.create(model="davinci-002", prompt="hello")
        ).choices[0].text == "completed"
        assert (
            await client.embeddings.create(
                model="text-embedding-3-small", input="hello", encoding_format="float"
            )
        ).data[0].embedding == [0.1, 0.2]
    assert len(captured) == 4
    for span in captured[1::2]:
        assert json.loads(span.attributes["traceloop.entity.output"]) == [[0.1, 0.2]]
        assert span.attributes["gen_ai.usage.input_tokens"] == 2
