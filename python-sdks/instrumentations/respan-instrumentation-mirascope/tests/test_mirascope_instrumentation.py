from __future__ import annotations

import asyncio
import gc
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest
import respan_instrumentation_mirascope._instrumentation as instrumentation
from mirascope import llm
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE


class FakeResponse:
    provider_id = "openai"
    model_id = "openai/gpt-4.1-mini"
    content = "Hello"
    text = "Hello"
    tool_calls: ClassVar[list[object]] = []
    usage = SimpleNamespace(
        input_tokens=5,
        output_tokens=2,
        cache_read_tokens=1,
        cache_write_tokens=0,
    )


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.mirascope"),
    )
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", True)
    return exporter


def test_sync_call_has_canonical_chat_attributes(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    def call(self, content, **kwargs):
        return FakeResponse()

    wrapped = instrumentation._call_wrapper(call)
    response = wrapped(model, "Say hello", tools=[lambda city: city])
    assert response.text == "Hello"

    span = spans.get_finished_spans()[0]
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4.1-mini"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Say hello"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello"
    tool_definition = json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]
    assert tool_definition["type"] == "function"
    assert tool_definition["function"]["name"]
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 2
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
    assert "traceloop.span.kind" not in attrs


@pytest.mark.asyncio
async def test_async_call_error_is_recorded_and_reraised(spans) -> None:
    model = SimpleNamespace(model_id="anthropic/claude-sonnet-4-5")

    async def call(self, content, **kwargs):
        raise ValueError("deterministic provider failure")

    wrapped = instrumentation._async_call_wrapper(call)
    with pytest.raises(ValueError, match="deterministic provider failure"):
        await wrapped(model, "fail")

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "ValueError",
        "message": "deterministic provider failure",
        "status": "error",
    }


def test_sync_stream_span_ends_after_consumption(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    response = FakeResponse()
    response._chunk_iterator = iter(["a", "b"])

    def stream(self, content, **kwargs):
        return response

    wrapped = instrumentation._stream_wrapper(stream)
    returned = wrapped(model, "stream")
    assert spans.get_finished_spans() == ()
    assert list(returned._chunk_iterator) == ["a", "b"]
    assert len(spans.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_stream_error_marks_span(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    response = FakeResponse()

    async def chunks():
        yield "first"
        raise RuntimeError("stream failed")

    response._chunk_iterator = chunks()

    async def stream(self, content, **kwargs):
        return response

    wrapped = instrumentation._async_stream_wrapper(stream)
    returned = await wrapped(model, "stream")
    collected = []
    with pytest.raises(RuntimeError, match="stream failed"):
        async for chunk in returned._chunk_iterator:
            collected.append(chunk)
    assert collected == ["first"]
    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["status"] == "error"
    assert output["error"] == "RuntimeError"


def test_real_tool_execution_uses_tool_contract_without_transport_envelope(
    spans,
) -> None:
    @llm.tool
    def weather(city: str) -> dict[str, int]:
        """Look up deterministic weather."""
        assert city == "Paris"
        return {"temperature": 18}

    tool_call = llm.ToolCall(id="call-1", name="weather", args='{"city":"Paris"}')
    toolkit = llm.Toolkit(tools=[weather])
    wrapped = instrumentation._tool_wrapper(llm.Toolkit.execute)
    result = wrapped(toolkit, tool_call)
    assert result.result == {"temperature": 18}
    attrs = spans.get_finished_spans()[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "arguments": {"city": "Paris"},
        "name": "weather",
    }
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "temperature": 18
    }
    assert not any(key.endswith("tool_calls") for key in attrs)


def test_real_tool_error_keeps_result_and_marks_tool_span_failed(spans) -> None:
    @llm.tool
    def broken_tool(city: str) -> str:
        """Raise a deterministic application error."""
        raise ValueError(f"no weather for {city}")

    tool_call = llm.ToolCall(
        id="call-error",
        name="broken_tool",
        args='{"city":"Paris"}',
    )
    toolkit = llm.Toolkit(tools=[broken_tool])
    result = instrumentation._tool_wrapper(llm.Toolkit.execute)(toolkit, tool_call)
    assert result.error is not None

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "no weather for Paris"
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == (
        "no weather for Paris"
    )
    assert [event.name for event in span.events] == ["exception"]


def test_raised_tool_error_records_one_exception_event(spans) -> None:
    tool_call = llm.ToolCall(id="call-error", name="broken_tool", args="{}")

    def execute(self, call):
        raise RuntimeError("tool transport failed")

    with pytest.raises(RuntimeError, match="tool transport failed"):
        instrumentation._tool_wrapper(execute)(object(), tool_call)

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]


def test_capture_content_false_omits_messages_and_outputs(spans, monkeypatch) -> None:
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", False)
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    def call(self, content, **kwargs):
        return FakeResponse()

    instrumentation._call_wrapper(call)(model, "secret")
    attrs = spans.get_finished_spans()[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert not any(key.startswith(SpanAttributes.LLM_PROMPTS) for key in attrs)
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7


def test_chat_prompt_completion_and_error_attributes_are_bounded(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    response = FakeResponse()
    response.content = "y" * 40_000
    response.text = "y" * 40_000

    instrumentation._call_wrapper(lambda self, content: response)(
        model,
        "x" * 40_000,
    )
    attrs = spans.get_finished_spans()[0].attributes
    assert len(attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"]) < 8_100
    assert len(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]) < 8_100

    def fail(self, content):
        raise RuntimeError("z" * 40_000)

    with pytest.raises(RuntimeError):
        instrumentation._call_wrapper(fail)(model, "fail")
    error_attrs = spans.get_finished_spans()[-1].attributes
    assert len(error_attrs["error.message"]) < 8_100


def test_direct_model_provider_and_error_text_are_privacy_safe(spans) -> None:
    class SecretText:
        def __str__(self) -> str:
            return "api_key=direct-attribute-secret"

    model = SimpleNamespace(model_id=SecretText())
    response = FakeResponse()
    response.model_id = SecretText()
    response.provider_id = SecretText()

    instrumentation._call_wrapper(lambda self, content: response)(model, "hello")
    serialized_attrs = json.dumps(dict(spans.get_finished_spans()[0].attributes))
    assert "direct-attribute-secret" not in serialized_attrs
    assert "<SecretText>" in serialized_attrs

    def fail(self, content):
        raise RuntimeError("Authorization: Bearer direct-error-secret")

    with pytest.raises(RuntimeError):
        instrumentation._call_wrapper(fail)(model, "fail")
    error_span = spans.get_finished_spans()[-1]
    assert "direct-error-secret" not in error_span.attributes["error.message"]
    assert "[REDACTED]" in error_span.attributes["error.message"]


def test_broken_exception_string_cannot_mask_or_leak_call_error(spans) -> None:
    class BrokenStringError(RuntimeError):
        stringify_calls = 0

        def __str__(self) -> str:
            type(self).stringify_calls += 1
            raise AssertionError("secret-from-broken-exception-string")

    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    def fail(self, content):
        raise BrokenStringError("api_key=provider-secret")

    with pytest.raises(BrokenStringError):
        instrumentation._call_wrapper(fail)(model, "fail")

    span = spans.get_finished_spans()[0]
    assert BrokenStringError.stringify_calls == 0
    assert span.attributes["error.message"] == "api_key=[REDACTED]"
    assert span.events[0].attributes["exception.message"] == "api_key=[REDACTED]"


def test_broken_tool_error_string_is_never_invoked(spans) -> None:
    class BrokenStringError(RuntimeError):
        stringify_calls = 0

        def __str__(self) -> str:
            type(self).stringify_calls += 1
            raise AssertionError("token-from-broken-tool-error")

    tool_call = llm.ToolCall(id="call-error", name="broken_tool", args="{}")
    result = SimpleNamespace(
        error=BrokenStringError("token=tool-secret"),
        result="tool failed",
    )
    returned = instrumentation._tool_wrapper(lambda self, call: result)(
        object(), tool_call
    )

    assert returned is result
    span = spans.get_finished_spans()[0]
    assert BrokenStringError.stringify_calls == 0
    assert span.attributes["error.message"] == "token=[REDACTED]"
    assert span.events[0].attributes["exception.message"] == "token=[REDACTED]"


def test_real_response_emits_current_turn_tool_call_and_request_definition(
    spans,
) -> None:
    @llm.tool
    def weather(city: str) -> dict[str, int]:
        """Look up deterministic weather."""
        return {"temperature": 18}

    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    tool_call = llm.ToolCall(id="call-1", name="weather", args='{"city":"Paris"}')
    response = llm.Response(
        raw={},
        provider_id="openai",
        model_id="openai/gpt-4.1-mini",
        provider_model_name="gpt-4.1-mini",
        params={},
        tools=[weather],
        input_messages=[llm.messages.user("Weather in Paris?")],
        assistant_message=llm.messages.assistant(
            [tool_call],
            provider_id="openai",
            model_id="openai/gpt-4.1-mini",
        ),
        finish_reason=None,
        usage=llm.Usage(input_tokens=24, output_tokens=10),
    )

    def call(self, content, **kwargs):
        return response

    instrumentation._call_wrapper(call)(model, "Weather in Paris?", tools=[weather])
    attrs = spans.get_finished_spans()[0].attributes
    definitions = json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    calls = json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
    assert definitions == [
        {
            "function": {
                "description": "Look up deterministic weather.",
                "name": "weather",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {"city": {"title": "City", "type": "string"}},
                    "required": ["city"],
                    "type": "object",
                },
            },
            "type": "function",
        }
    ]
    assert calls == [
        {
            "function": {
                "arguments": '{"city":"Paris"}',
                "name": "weather",
            },
            "id": "call-1",
            "type": "function",
        }
    ]
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "content": "",
        "tool_calls": calls,
    }
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 34
    for alias in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
    ):
        assert alias not in attrs


def test_real_sync_stream_sets_stream_flag_and_finishes_content_and_usage(
    spans,
) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    response = llm.StreamResponse(
        provider_id="openai",
        model_id="openai/gpt-4.1-mini",
        provider_model_name="gpt-4.1-mini",
        params={},
        tools=None,
        input_messages=[llm.messages.user("Stream a reply")],
        chunk_iterator=iter(
            [
                llm.TextStartChunk(),
                llm.TextChunk(delta="Mirascope streaming works."),
                llm.TextEndChunk(),
                llm.UsageDeltaChunk(input_tokens=12, output_tokens=6),
            ]
        ),
    )

    returned = instrumentation._stream_wrapper(lambda self, content: response)(
        model, "Stream a reply"
    )
    assert list(returned.text_stream()) == ["Mirascope streaming works.", "\n"]
    attrs = spans.get_finished_spans()[0].attributes
    assert spans.get_finished_spans()[0].events == ()
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Mirascope streaming works."
    )
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 6


def test_real_sync_stream_close_finalizes_once_without_false_error(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    source_closed = False

    def chunks():
        nonlocal source_closed
        try:
            yield llm.TextStartChunk()
            yield llm.TextChunk(delta="Partial sync content.")
            yield llm.TextEndChunk()
        finally:
            source_closed = True

    response = llm.StreamResponse(
        provider_id="openai",
        model_id="openai/gpt-4.1-mini",
        provider_model_name="gpt-4.1-mini",
        params={},
        tools=None,
        input_messages=[llm.messages.user("Close sync stream")],
        chunk_iterator=chunks(),
    )

    returned = instrumentation._stream_wrapper(lambda self, content: response)(
        model, "Close sync stream"
    )
    chunk_stream = returned._chunk_iterator
    assert next(chunk_stream).type == "text_start_chunk"
    assert next(chunk_stream).delta == "Partial sync content."
    chunk_stream.close()
    chunk_stream.close()

    exported = spans.get_finished_spans()
    assert len(exported) == 1
    assert exported[0].status.status_code is not StatusCode.ERROR
    assert exported[0].events == ()
    assert exported[0].attributes["status_code"] == 200
    assert source_closed is True


def test_abandoned_sync_stream_closes_source_and_finalizes_once(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    source_closed = False

    def stream(self, content):
        nonlocal source_closed

        class Chunks:
            def __iter__(self):
                return self

            def __next__(self):
                return llm.TextStartChunk()

            def close(self) -> None:
                nonlocal source_closed
                source_closed = True

        return llm.StreamResponse(
            provider_id="openai",
            model_id="openai/gpt-4.1-mini",
            provider_model_name="gpt-4.1-mini",
            params={},
            tools=None,
            input_messages=[llm.messages.user("Abandon sync stream")],
            chunk_iterator=Chunks(),
        )

    wrapped = instrumentation._stream_wrapper(stream)
    returned = wrapped(model, "Abandon sync stream")
    del returned
    del wrapped
    gc.collect()

    exported = spans.get_finished_spans()
    assert source_closed is True
    assert len(exported) == 1
    assert exported[0].status.status_code is not StatusCode.ERROR


@pytest.mark.asyncio
async def test_real_async_stream_sets_stream_flag_and_finishes_content(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    async def chunks():
        yield llm.TextStartChunk()
        yield llm.TextChunk(delta="Async Mirascope stream.")
        yield llm.TextEndChunk()
        yield llm.UsageDeltaChunk(input_tokens=8, output_tokens=4)

    response = llm.AsyncStreamResponse(
        provider_id="openai",
        model_id="openai/gpt-4.1-mini",
        provider_model_name="gpt-4.1-mini",
        params={},
        tools=None,
        input_messages=[llm.messages.user("Stream async")],
        chunk_iterator=chunks(),
    )

    async def stream(self, content):
        return response

    returned = await instrumentation._async_stream_wrapper(stream)(
        model, "Stream async"
    )
    collected = [part async for part in returned.text_stream()]
    assert collected == ["Async Mirascope stream.", "\n"]
    attrs = spans.get_finished_spans()[0].attributes
    assert spans.get_finished_spans()[0].events == ()
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Async Mirascope stream."
    )
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 12


@pytest.mark.asyncio
async def test_real_async_stream_close_finalizes_once_without_false_error(
    spans,
) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    source_closed = False

    async def chunks():
        nonlocal source_closed
        try:
            yield llm.TextStartChunk()
            yield llm.TextChunk(delta="Partial async content.")
            yield llm.TextEndChunk()
        finally:
            source_closed = True

    response = llm.AsyncStreamResponse(
        provider_id="openai",
        model_id="openai/gpt-4.1-mini",
        provider_model_name="gpt-4.1-mini",
        params={},
        tools=None,
        input_messages=[llm.messages.user("Close async stream")],
        chunk_iterator=chunks(),
    )

    async def stream(self, content):
        return response

    returned = await instrumentation._async_stream_wrapper(stream)(
        model, "Close async stream"
    )
    chunk_stream = returned._chunk_iterator
    assert (await chunk_stream.__anext__()).type == "text_start_chunk"
    assert (await chunk_stream.__anext__()).delta == "Partial async content."
    await chunk_stream.aclose()
    await chunk_stream.aclose()

    exported = spans.get_finished_spans()
    assert len(exported) == 1
    assert exported[0].status.status_code is not StatusCode.ERROR
    assert exported[0].events == ()
    assert exported[0].attributes["status_code"] == 200
    assert source_closed is True


@pytest.mark.asyncio
async def test_abandoned_async_stream_closes_source_and_finalizes_once(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")
    source_closed = False

    async def stream(self, content):
        nonlocal source_closed

        class Chunks:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return llm.TextStartChunk()

            async def aclose(self) -> None:
                nonlocal source_closed
                source_closed = True

        return llm.AsyncStreamResponse(
            provider_id="openai",
            model_id="openai/gpt-4.1-mini",
            provider_model_name="gpt-4.1-mini",
            params={},
            tools=None,
            input_messages=[llm.messages.user("Abandon async stream")],
            chunk_iterator=Chunks(),
        )

    wrapped = instrumentation._async_stream_wrapper(stream)
    returned = await wrapped(model, "Abandon async stream")
    del returned
    del wrapped
    gc.collect()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    exported = spans.get_finished_spans()
    assert source_closed is True
    assert len(exported) == 1
    assert exported[0].status.status_code is not StatusCode.ERROR


def test_activated_real_model_and_toolkit_export_exact_connected_spans(
    monkeypatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.mirascope.activated"),
    )
    monkeypatch.setattr(instrumentation, "_is_respan_tracing_enabled", lambda: True)

    @llm.tool
    def weather(city: str) -> dict[str, int]:
        """Return activated deterministic weather."""
        return {"temperature": 18 if city == "Paris" else 0}

    class ActivatedProvider:
        id = "respan-activated"
        default_scope = "respan-activated/"
        client = None

        def call(
            self,
            *,
            model_id,
            messages,
            toolkit,
            format=None,
            **params,
        ):
            tool_call = llm.ToolCall(
                id="activated-call-1",
                name="weather",
                args='{"city":"Paris"}',
            )
            return llm.Response(
                raw={"fixture": "activated"},
                provider_id=self.id,
                model_id=model_id,
                provider_model_name=model_id.split("/", 1)[-1],
                params=params,
                tools=toolkit,
                format=format,
                input_messages=messages,
                assistant_message=llm.messages.assistant(
                    [tool_call],
                    provider_id=self.id,
                    model_id=model_id,
                ),
                finish_reason=None,
                usage=llm.Usage(input_tokens=9, output_tokens=3),
            )

    llm.register_provider(ActivatedProvider(), scope="respan-activated/")
    model = llm.Model("respan-activated/model")
    instrumentor = instrumentation.MirascopeInstrumentor()
    instrumentor.activate()
    instrumentor.activate()
    try:
        tracer = provider.get_tracer("test.mirascope.parent")
        with tracer.start_as_current_span("activated.workflow") as parent:
            response = model.call("Weather in Paris?", tools=[weather])
            outputs = response.execute_tools()
            assert outputs[0].result == {"temperature": 18}
            parent_span_id = parent.get_span_context().span_id
    finally:
        instrumentor.deactivate()
        instrumentor.deactivate()

    exported = exporter.get_finished_spans()
    assert len(exported) == 3
    assert len({span.context.span_id for span in exported}) == 3
    parent = next(span for span in exported if span.name == "activated.workflow")
    children = [span for span in exported if span is not parent]
    assert {span.attributes[RESPAN_LOG_TYPE] for span in children} == {"chat", "tool"}
    assert {span.parent.span_id for span in children} == {parent_span_id}
    assert all("traceloop.span.kind" not in span.attributes for span in children)


def test_raised_chat_exception_natively_marks_enclosing_span_error(spans) -> None:
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    class ProviderError(RuntimeError):
        status_code = 503

    def call(self, content):
        raise ProviderError("provider unavailable")

    tracer = instrumentation.trace.get_tracer("test.mirascope.parent")
    with pytest.raises(ProviderError), tracer.start_as_current_span("workflow"):
        instrumentation._call_wrapper(call)(model, "fail")

    exported = spans.get_finished_spans()
    chat = next(span for span in exported if span.name.endswith(".chat"))
    parent = next(span for span in exported if span.name == "workflow")
    assert chat.status.status_code is StatusCode.ERROR
    assert chat.attributes["status_code"] == 503
    assert parent.status.status_code is StatusCode.ERROR
    assert chat.parent.span_id == parent.context.span_id
