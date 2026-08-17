from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry.trace.status import StatusCode
from respan_instrumentation_ollama import OllamaInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

MODEL = "llama3.2"
PRIVATE_HEADER = "Bearer ollama-test-private-value"


def get_weather(city: str) -> str:
    """Return deterministic weather for a city."""
    return f"Sunny in {city}"


class _OllamaCompatibilityHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0") or "0")
        payload = json.loads(self.rfile.read(content_length) or b"{}")

        if self.path == "/api/chat":
            self._chat(payload)
            return
        if self.path == "/api/generate":
            self._generate(payload)
            return
        if self.path in {"/api/embed", "/api/embeddings"}:
            self._write_json(
                {
                    "model": payload.get("model") or MODEL,
                    "embeddings": [[0.1, 0.2, 0.3]],
                    "embedding": [0.1, 0.2, 0.3],
                    "prompt_eval_count": 4,
                    "done": True,
                }
            )
            return
        self._write_json({"error": "not found"}, status_code=404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _chat(self, payload: dict[str, Any]) -> None:
        messages = payload.get("messages") or []
        if any(
            "force failure" in str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        ):
            self._write_json({"error": "compat provider unavailable"}, status_code=503)
            return

        if any(
            message.get("role") == "tool"
            for message in messages
            if isinstance(message, dict)
        ):
            message = {
                "role": "assistant",
                "content": "The weather in Tokyo is sunny.",
            }
        elif payload.get("tools"):
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "Tracing works."}

        self._write_json(
            {
                "model": payload.get("model") or MODEL,
                "created_at": "2026-08-17T00:00:00Z",
                "message": message,
                "done": True,
                "prompt_eval_count": 9,
                "eval_count": 7,
            }
        )

    def _generate(self, payload: dict[str, Any]) -> None:
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.end_headers()
            for chunk in (
                {
                    "model": payload.get("model") or MODEL,
                    "response": "Streaming ",
                    "done": False,
                },
                {
                    "model": payload.get("model") or MODEL,
                    "response": "works.",
                    "done": True,
                    "prompt_eval_count": 6,
                    "eval_count": 5,
                },
            ):
                self.wfile.write(json.dumps(chunk).encode() + b"\n")
            return
        self._write_json(
            {
                "model": payload.get("model") or MODEL,
                "response": "Generation works.",
                "done": True,
                "prompt_eval_count": 6,
                "eval_count": 5,
            }
        )

    def _write_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def ollama_compat_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaCompatibilityHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_current_ollama_client_exports_complete_connected_contract(
    monkeypatch: pytest.MonkeyPatch,
    ollama_compat_server: str,
) -> None:
    ollama = pytest.importorskip("ollama")
    captured_spans: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_ollama._otel_emitter.inject_span",
        lambda span: captured_spans.append(span),
    )

    instrumentor = OllamaInstrumentor()
    instrumentor.activate()
    client = ollama.Client(
        host=ollama_compat_server,
        headers={"Authorization": PRIVATE_HEADER},
    )
    trace_id = int("1" * 32, 16)
    parent_span_id = int("2" * 16, 16)
    parent_context = SpanContext(
        trace_id=trace_id,
        span_id=parent_span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    try:
        with trace.use_span(NonRecordingSpan(parent_context), end_on_exit=False):
            chat_response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "Say hello."}],
            )
            stream_chunks = list(
                client.generate(
                    model=MODEL,
                    prompt="Stream a reply.",
                    stream=True,
                )
            )
            tool_response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "Weather in Tokyo?"}],
                tools=[get_weather],
            )
            tool_calls = tool_response.message.tool_calls or []
            assert len(tool_calls) == 1
            assistant_message = tool_response.message.model_dump(exclude_none=True)
            tool_result = get_weather(**tool_calls[0].function.arguments)
            final_response = client.chat(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "Weather in Tokyo?"},
                    assistant_message,
                    {
                        "role": "tool",
                        "tool_name": "get_weather",
                        "content": tool_result,
                    },
                ],
            )
            embed_response = client.embed(model="nomic-embed-text", input="hello")
            with pytest.raises(ollama.ResponseError) as error_info:
                client.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": "force failure"}],
                )
    finally:
        client.close()
        instrumentor.deactivate()

    assert chat_response.message.content == "Tracing works."
    assert "".join(chunk.response for chunk in stream_chunks) == "Streaming works."
    assert final_response.message.content == "The weather in Tokyo is sunny."
    assert embed_response.prompt_eval_count == 4
    assert error_info.value.status_code == 503
    assert len(captured_spans) == 6

    span_ids = {span.context.span_id for span in captured_spans}
    assert len(span_ids) == len(captured_spans)
    for span in captured_spans:
        attrs = dict(span.attributes or {})
        assert span.context.trace_id == trace_id
        assert span.parent is not None
        assert span.parent.span_id == parent_span_id
        assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
        assert PRIVATE_HEADER not in json.dumps(attrs, default=str)

    chat_attrs = dict(captured_spans[0].attributes or {})
    assert chat_attrs[RESPAN_LOG_TYPE] == "chat"
    assert chat_attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Say hello."
    assert chat_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Tracing works."

    stream_attrs = dict(captured_spans[1].attributes or {})
    assert stream_attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert stream_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "Streaming works."
    )
    assert stream_attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 11

    first_tool_attrs = dict(captured_spans[2].attributes or {})
    definitions = json.loads(first_tool_attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    current_calls = json.loads(
        first_tool_attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert definitions[0]["function"]["name"] == "get_weather"
    assert definitions[0]["function"]["parameters"]["properties"]["city"] == {
        "type": "string"
    }
    assert current_calls == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]

    final_tool_attrs = dict(captured_spans[3].attributes or {})
    assert f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls" in final_tool_attrs
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in final_tool_attrs

    error_span = captured_spans[-1]
    assert error_span.status.status_code is StatusCode.ERROR
    assert error_span.attributes["status_code"] == 503
    assert "compat provider unavailable" in error_span.attributes["error.message"]

    banned_aliases = {
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
    for span in captured_spans:
        assert banned_aliases.isdisjoint(span.attributes or {})
