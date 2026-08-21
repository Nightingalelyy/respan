from __future__ import annotations

import asyncio
import json
from types import ModuleType, SimpleNamespace
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_livekit import _instrumentation, _otel_emitter
from respan_instrumentation_livekit._constants import (
    ATTR_LLM_METRICS,
    EVENT_GEN_AI_CHOICE,
    EVENT_GEN_AI_USER_MESSAGE,
    LIVEKIT_RESPAN_PROVIDER_NAME_ATTR,
    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
)
from respan_instrumentation_livekit._processor import LiveKitSpanProcessor
from respan_instrumentation_livekit._translator import (
    build_livekit_llm_attrs,
    build_tool_span_attrs,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.exporters.respan import _convert_attributes

_OFF_CONTRACT_ALIASES = {
    "completion_tokens",
    "has_tool_calls",
    "model",
    "parallel_tool_calls",
    "prompt_tokens",
    "respan.span.handoffs",
    "respan.span.tool_calls",
    "respan.span.tools",
    "span_tools",
    "tool_calls",
    "tools",
    "total_request_tokens",
}


class _Event:
    def __init__(self, name: str, attributes: dict[str, Any]) -> None:
        self.name = name
        self.attributes = attributes


class _Span:
    def __init__(
        self,
        *,
        name: str = "llm_request",
        attrs: dict[str, Any] | None = None,
        events: list[_Event] | None = None,
    ) -> None:
        self._name = name
        self._attributes = attrs or {}
        self.events = events or []
        self.context = SimpleNamespace(
            trace_id=int("0" * 31 + "a", 16),
            span_id=int("0" * 15 + "b", 16),
        )

    @property
    def name(self) -> str:
        return self._name


def test_processor_translates_livekit_llm_span_to_respan_contract(monkeypatch):
    tool_schema = {
        "type": "function",
        "function": {
            "name": "lookup_room",
            "description": "Lookup a room.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {"name": "lookup_room", "arguments": '{"room":"blue"}'},
    }
    span = _Span(
        attrs={
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat",
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
            GenAIAttributes.GEN_AI_REQUEST_MODEL: "gpt-4o-mini",
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 10,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 4,
            SpanAttributes.LLM_IS_STREAMING: True,
            ATTR_LLM_METRICS: json.dumps(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_cached_tokens": 2,
                }
            ),
            LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR: json.dumps([tool_schema]),
            RESPAN_TRACE_GROUP_ID: "livekit_03_tool_calling",
        },
        events=[
            _Event(EVENT_GEN_AI_USER_MESSAGE, {"content": "Which room is blue?"}),
            _Event(
                EVENT_GEN_AI_CHOICE,
                {
                    "role": "assistant",
                    "content": "I will check the room.",
                    "tool_calls": [json.dumps(tool_call)],
                },
            ),
        ],
    )

    registered_parents = []
    monkeypatch.setattr(
        "respan_instrumentation_livekit._processor.register_livekit_tool_parent_context",
        lambda **kwargs: registered_parents.append(kwargs),
    )

    LiveKitSpanProcessor().on_end(span)  # type: ignore[arg-type]
    attrs = span._attributes

    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert span.name == "livekit_03_tool_calling"
    assert attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "livekit_03_tool_calling"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Which room is blue?"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
        "I will check the room."
    )
    assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]) == [
        tool_call
    ]
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [tool_schema]
    assert LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR not in attrs
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 10
    assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 14
    assert registered_parents == [
        {
            "call_id": "call_123",
            "trace_id": "0" * 31 + "a",
            "parent_id": "0" * 15 + "b",
            "custom_identifier": None,
        }
    ]
    assert _OFF_CONTRACT_ALIASES.isdisjoint(attrs)


def test_stream_and_tool_contract_survives_otlp_attribute_serialization():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "lookup_room",
            "description": "Lookup a room.",
            "parameters": {
                "type": "object",
                "properties": {"room": {"type": "string"}},
                "required": ["room"],
            },
        },
    }
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {"name": "lookup_room", "arguments": '{"room":"blue"}'},
    }
    attrs = build_livekit_llm_attrs(
        span_name="llm_request",
        attrs={
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat",
            SpanAttributes.LLM_IS_STREAMING: True,
            LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR: json.dumps([tool_schema]),
        },
        events=[
            _Event(
                EVENT_GEN_AI_CHOICE,
                {"content": "I will check.", "tool_calls": [json.dumps(tool_call)]},
            )
        ],
    )

    exported = {item["key"]: item["value"] for item in _convert_attributes(attrs)}

    assert exported[SpanAttributes.LLM_IS_STREAMING] == {"boolValue": True}
    assert json.loads(
        exported[SpanAttributes.LLM_REQUEST_FUNCTIONS]["stringValue"]
    ) == [tool_schema]
    assert json.loads(
        exported[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]["stringValue"]
    ) == [tool_call]
    assert _OFF_CONTRACT_ALIASES.isdisjoint(exported)


def test_gateway_plugin_provider_overrides_base_url_host():
    gateway_llm = type(
        "LLM",
        (),
        {
            "__module__": "livekit.plugins.openai.llm",
            "model": "gpt-4o-mini",
            "provider": "api.respan.ai",
        },
    )()

    assert _instrumentation._provider_name_from_llm(gateway_llm) == "openai"

    attrs = build_livekit_llm_attrs(
        span_name="llm_request",
        attrs={
            GenAIAttributes.GEN_AI_OPERATION_NAME: "chat",
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "api.respan.ai",
            LIVEKIT_RESPAN_PROVIDER_NAME_ATTR: "openai",
        },
        events=[],
    )
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"


def test_build_tool_span_attrs_uses_tool_contract_without_aliases():
    attrs = build_tool_span_attrs(
        tool_name="lookup_room",
        arguments='{"room":"blue"}',
        output={"temperature": 21},
        call_id="call_123",
    )

    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_room"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "lookup_room"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "name": "lookup_room",
        "arguments": {"room": "blue"},
    }
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "temperature": 21
    }
    assert _OFF_CONTRACT_ALIASES.isdisjoint(attrs)


def test_emit_tool_span_uses_registered_parent_before_current_parent(monkeypatch):
    captured = {}

    def fake_build_readable_span(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"name": name, **kwargs}

    class _FakeCurrentSpan:
        def get_span_context(self):
            return SimpleNamespace(
                trace_id=int("0" * 31 + "1", 16),
                span_id=int("0" * 15 + "2", 16),
            )

    monkeypatch.setattr(
        _otel_emitter.trace, "get_current_span", lambda: _FakeCurrentSpan()
    )
    monkeypatch.setattr(_otel_emitter, "build_readable_span", fake_build_readable_span)
    monkeypatch.setattr(_otel_emitter, "inject_span", lambda span: True)
    monkeypatch.setattr(
        _otel_emitter,
        "read_propagated_attributes",
        lambda: {RESPAN_TRACE_GROUP_ID: "livekit_03_tool_calling"},
    )

    _otel_emitter._TOOL_PARENT_CONTEXTS.clear()
    _otel_emitter.register_livekit_tool_parent_context(
        call_id="call_1",
        trace_id="0" * 31 + "a",
        parent_id="0" * 15 + "b",
    )

    _otel_emitter.emit_livekit_tool_span(
        tool_name="lookup_room",
        arguments={},
        output="ok",
        call_id="call_1",
        start_time_ns=100,
    )

    assert captured["name"] == "livekit_03_tool_calling.lookup_room"
    assert captured["trace_id"] == "0" * 31 + "a"
    assert captured["parent_id"] == "0" * 15 + "b"
    assert captured["start_time_ns"] == 100
    assert captured["status_code"] == 200
    assert (
        captured["attributes"][SpanAttributes.TRACELOOP_WORKFLOW_NAME]
        == "livekit_03_tool_calling"
    )


def test_emit_tool_span_sets_error_attrs(monkeypatch):
    captured = {}

    def fake_build_readable_span(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return {"name": name, **kwargs}

    monkeypatch.setattr(_otel_emitter.trace, "get_current_span", lambda: None)
    monkeypatch.setattr(_otel_emitter, "build_readable_span", fake_build_readable_span)
    monkeypatch.setattr(_otel_emitter, "inject_span", lambda span: True)

    _otel_emitter.emit_livekit_tool_span(
        tool_name="missing_tool",
        arguments={},
        output=None,
        call_id="call_missing",
        start_time_ns=100,
        error=ValueError("Unknown function: missing_tool"),
    )

    assert captured["status_code"] == 500
    assert captured["error_message"] == "Unknown function: missing_tool"
    assert captured["attributes"]["status_code"] == 500
    assert captured["attributes"]["error.message"] == "Unknown function: missing_tool"


def test_instrumentor_patches_and_restores_livekit_modules(monkeypatch):
    async def original_execute_function_call(tool_call, tool_ctx, *, call_ctx=None):
        return SimpleNamespace(
            fnc_call=SimpleNamespace(
                name=tool_call.name,
                arguments=tool_call.arguments,
                call_id=tool_call.call_id,
            ),
            fnc_call_out=SimpleNamespace(output="ok", is_error=False),
            raw_output="ok",
            raw_exception=None,
        )

    class FakeLLMStream:
        async def _main_task(self):
            return None

    original_main_task = FakeLLMStream._main_task

    fake_telemetry = ModuleType("livekit.agents.telemetry")
    fake_telemetry.provider = None
    fake_telemetry.set_tracer_provider = lambda provider: setattr(
        fake_telemetry, "provider", provider
    )

    fake_llm = ModuleType("livekit.agents.llm")
    fake_llm.execute_function_call = original_execute_function_call
    fake_llm.LLMStream = FakeLLMStream

    fake_utils = ModuleType("livekit.agents.llm.utils")
    fake_utils.execute_function_call = original_execute_function_call

    module_map = {
        "livekit.agents.telemetry": fake_telemetry,
        "livekit.agents.llm": fake_llm,
        "livekit.agents.llm.utils": fake_utils,
    }
    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        lambda name: module_map[name],
    )
    monkeypatch.setattr(_instrumentation, "_register_processor", lambda processor: None)
    monkeypatch.setattr(
        _instrumentation, "_unregister_processor", lambda processor: None
    )
    emitted = []
    stream_attrs = {}

    class FakeCurrentSpan:
        def set_attribute(self, key, value):
            stream_attrs[key] = value

    monkeypatch.setattr(
        _instrumentation.trace,
        "get_current_span",
        lambda: FakeCurrentSpan(),
    )
    monkeypatch.setattr(
        _instrumentation,
        "emit_livekit_tool_span",
        lambda **kwargs: emitted.append(kwargs),
    )

    instrumentor = _instrumentation.LiveKitInstrumentor()
    instrumentor.activate()

    assert fake_telemetry.provider is not None
    assert fake_utils.execute_function_call is not original_execute_function_call
    assert fake_llm.execute_function_call is fake_utils.execute_function_call
    assert FakeLLMStream._main_task is not original_main_task

    stream = FakeLLMStream()
    stream._tools = []
    asyncio.run(stream._main_task())
    assert stream_attrs[SpanAttributes.LLM_IS_STREAMING] is True

    result = asyncio.run(
        fake_utils.execute_function_call(
            SimpleNamespace(name="lookup_room", arguments="{}", call_id="call_1"),
            object(),
        )
    )
    assert result.raw_output == "ok"
    assert emitted[0]["tool_name"] == "lookup_room"

    instrumentor.deactivate()

    assert fake_utils.execute_function_call is original_execute_function_call
    assert fake_llm.execute_function_call is original_execute_function_call
    assert FakeLLMStream._main_task is original_main_task
    assert _instrumentation._ORIGINAL_LLM_STREAM_MAIN_TASK is None


def test_real_livekit_stream_exports_translated_stream_and_provider():
    from livekit.agents import llm, telemetry
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

    class GatewayLLM(llm.LLM):
        __module__ = "livekit.plugins.openai.llm"

        def __init__(self) -> None:
            super().__init__()
            self._model = "gpt-4o-mini"

        @property
        def model(self) -> str:
            return self._model

        @property
        def provider(self) -> str:
            return "api.respan.ai"

        def chat(self, *, chat_ctx, tools=None, conn_options=None, **_kwargs):
            return GatewayStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
            )

    class GatewayStream(llm.LLMStream):
        async def _run(self) -> None:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="gateway-stream",
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content="streamed gateway response",
                    ),
                )
            )
            await asyncio.sleep(0)
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="gateway-stream",
                    usage=llm.CompletionUsage(
                        prompt_tokens=7,
                        completion_tokens=3,
                        total_tokens=10,
                    ),
                )
            )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(LiveKitSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous_provider = telemetry.tracer._tracer_provider
    telemetry.set_tracer_provider(provider)
    _instrumentation._patch_llm_stream_main_task(llm.LLMStream)
    try:

        async def collect_gateway_response():
            chat_ctx = llm.ChatContext.empty()
            chat_ctx.add_message(role="user", content="stream a response")
            return await GatewayLLM().chat(chat_ctx=chat_ctx).collect()

        response = asyncio.run(collect_gateway_response())
        assert response.text == "streamed gateway response"

        provider.force_flush()
        chat_spans = [
            span
            for span in exporter.get_finished_spans()
            if span.attributes.get(RESPAN_LOG_TYPE) == "chat"
        ]
        assert len(chat_spans) == 1
        attrs = chat_spans[0].attributes
        assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
        assert attrs[SpanAttributes.LLM_SYSTEM] == "openai"
        assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
            "content": "streamed gateway response"
        }
        assert LIVEKIT_RESPAN_PROVIDER_NAME_ATTR not in attrs
    finally:
        _instrumentation._restore_llm_stream_main_task()
        telemetry.tracer.set_provider(previous_provider)
        provider.shutdown()
