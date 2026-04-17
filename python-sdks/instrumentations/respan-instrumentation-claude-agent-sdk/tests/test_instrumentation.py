import asyncio
import builtins
import json
import sys
from types import ModuleType, SimpleNamespace

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

from respan_instrumentation_claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
    _instrumentation,
    _processor,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_TYPE,
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


def _make_fake_tracer_provider(*, composite: bool = True) -> SimpleNamespace:
    added_processors: list[object] = []
    tracer_provider = SimpleNamespace(added_processors=added_processors)

    if composite:
        tracer_provider._active_span_processor = SimpleNamespace(_span_processors=())

    def add_span_processor(processor: object) -> None:
        added_processors.append(processor)

    tracer_provider.add_span_processor = add_span_processor
    return tracer_provider


def _install_fake_claude_agent_sdk_modules(
    monkeypatch,
    *,
    instrument_error: Exception | None = None,
) -> SimpleNamespace:
    output_messages_attr = "gen_ai.output.messages"
    usage_input_tokens_attr = "gen_ai.usage.input_tokens"
    usage_output_tokens_attr = "gen_ai.usage.output_tokens"
    usage_cache_creation_tokens_attr = "gen_ai.usage.cache_creation_input_tokens"
    usage_cache_read_tokens_attr = "gen_ai.usage.cache_read_input_tokens"

    class FakeClaudeAgentSdkInstrumentor:
        def __init__(self):
            self.instrument_kwargs = None
            self.uninstrument_calls = 0

        def instrument(self, **kwargs):
            self.instrument_kwargs = kwargs
            if instrument_error is not None:
                raise instrument_error

        def uninstrument(self):
            self.uninstrument_calls += 1

        def _wrap_client_query(self, wrapped, instance, args, kwargs):
            instance._otel_invocation_ctx = kwargs.get("otel_invocation_ctx")
            return wrapped(*args, **kwargs)

        async def _instrumented_receive_response(
            self,
            wrapped,
            instance,
            args,
            kwargs,
        ):
            async for message in wrapped(*args, **kwargs):
                yield message

    def _original_set_response_content(span, content):
        span.set_attribute(
            output_messages_attr,
            json.dumps([{"role": "assistant", "content": content}], default=str),
        )

    def _original_set_result_attributes(span, result_message):
        usage = getattr(result_message, "usage", None) or {}
        total_input_tokens = (
            (usage.get("input_tokens", 0) or 0)
            + (usage.get("cache_creation_input_tokens", 0) or 0)
            + (usage.get("cache_read_input_tokens", 0) or 0)
        )
        span.set_attribute(usage_input_tokens_attr, total_input_tokens)
        span.set_attribute(usage_output_tokens_attr, usage.get("output_tokens", 0) or 0)
        if usage.get("cache_creation_input_tokens", 0):
            span.set_attribute(
                usage_cache_creation_tokens_attr,
                usage["cache_creation_input_tokens"],
            )
        if usage.get("cache_read_input_tokens", 0):
            span.set_attribute(
                usage_cache_read_tokens_attr,
                usage["cache_read_input_tokens"],
            )

    claude_package = ModuleType("opentelemetry.instrumentation.claude_agent_sdk")
    claude_package.__path__ = []
    claude_package.ClaudeAgentSdkInstrumentor = FakeClaudeAgentSdkInstrumentor

    constants_module = ModuleType(
        "opentelemetry.instrumentation.claude_agent_sdk._constants"
    )
    constants_module.GEN_AI_OUTPUT_MESSAGES = output_messages_attr
    constants_module.GEN_AI_USAGE_INPUT_TOKENS = usage_input_tokens_attr
    constants_module.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
        usage_cache_creation_tokens_attr
    )
    constants_module.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = (
        usage_cache_read_tokens_attr
    )

    context_state = {"value": None}

    context_module = ModuleType(
        "opentelemetry.instrumentation.claude_agent_sdk._context"
    )

    def get_invocation_context():
        return context_state["value"]

    def set_invocation_context(value):
        context_state["value"] = value

    context_module.get_invocation_context = get_invocation_context
    context_module.set_invocation_context = set_invocation_context

    spans_module = ModuleType("opentelemetry.instrumentation.claude_agent_sdk._spans")
    spans_module._to_serializable = lambda value: value
    spans_module.set_response_content = _original_set_response_content
    spans_module.set_result_attributes = _original_set_result_attributes

    instrumentor_module = ModuleType(
        "opentelemetry.instrumentation.claude_agent_sdk._instrumentor"
    )
    instrumentor_module.ClaudeAgentSdkInstrumentor = FakeClaudeAgentSdkInstrumentor
    instrumentor_module.set_response_content = _original_set_response_content
    instrumentor_module.set_result_attributes = _original_set_result_attributes

    claude_sdk_module = ModuleType("claude_agent_sdk")
    claude_sdk_module.__path__ = []
    internal_module = ModuleType("claude_agent_sdk._internal")
    internal_module.__path__ = []
    query_module = ModuleType("claude_agent_sdk._internal.query")

    class FakeQuery:
        def __init__(self):
            self._otel_invocation_ctx = None

        async def _handle_control_request(self, request):
            return context_module.get_invocation_context()

    query_module.Query = FakeQuery

    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk",
        claude_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._constants",
        constants_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._context",
        context_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._spans",
        spans_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.instrumentation.claude_agent_sdk._instrumentor",
        instrumentor_module,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", claude_sdk_module)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk._internal", internal_module)
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk._internal.query",
        query_module,
    )

    return SimpleNamespace(
        instrumentor_class=FakeClaudeAgentSdkInstrumentor,
        context_module=context_module,
        query_class=FakeQuery,
        spans_module=spans_module,
        instrumentor_module=instrumentor_module,
        original_set_response_content=_original_set_response_content,
        original_set_result_attributes=_original_set_result_attributes,
    )


def _make_span(
    *,
    name: str,
    attributes: dict[str, object] | None = None,
    trace_id: int = 1,
    span_id: int = 1,
    parent_span_id: int | None = None,
    start_time: int = 10,
) -> SimpleNamespace:
    attrs = dict(attributes or {})
    span_context = SimpleNamespace(trace_id=trace_id, span_id=span_id)
    return SimpleNamespace(
        name=name,
        _attributes=attrs,
        attributes=attrs,
        parent=(
            SimpleNamespace(span_id=parent_span_id)
            if parent_span_id is not None
            else None
        ),
        start_time=start_time,
        end_time=start_time + 1,
        get_span_context=lambda: span_context,
    )


def test_package_exports_instrumentor():
    assert ClaudeAgentSDKInstrumentor is _instrumentation.ClaudeAgentSDKInstrumentor
    assert ClaudeAgentSDKInstrumentor.name == "claude-agent-sdk"


def test_instrumentation_helpers_read_attrs_and_parse_json():
    span_with_public_attrs = SimpleNamespace(attributes={"key": "value"})
    span_with_private_attrs = SimpleNamespace(_attributes={"key": "private"})

    assert _instrumentation._safe_json_loads('{"a": 1}') == {"a": 1}
    assert _instrumentation._safe_json_loads("plain-text") is None
    assert _instrumentation._get_span_attr_value(span_with_public_attrs, "key") == "value"
    assert _instrumentation._get_span_attr_value(span_with_private_attrs, "key") == "private"


def test_register_and_unregister_processor_keep_processor_first():
    tracer_provider = _make_fake_tracer_provider()
    existing_processor = object()
    tracer_provider._active_span_processor._span_processors = (existing_processor,)
    processor = object()

    ClaudeAgentSDKInstrumentor._register_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )
    ClaudeAgentSDKInstrumentor._register_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )

    assert tracer_provider._active_span_processor._span_processors == (
        processor,
        existing_processor,
    )

    ClaudeAgentSDKInstrumentor._unregister_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )

    assert tracer_provider._active_span_processor._span_processors == (
        existing_processor,
    )


def test_register_processor_falls_back_to_add_span_processor():
    tracer_provider = _make_fake_tracer_provider(composite=False)
    processor = object()

    ClaudeAgentSDKInstrumentor._register_processor(
        tracer_provider=tracer_provider,
        processor=processor,
    )

    assert tracer_provider.added_processors == [processor]


def test_activate_patches_helpers_and_restores_originals(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(monkeypatch)

    instrumentor = ClaudeAgentSDKInstrumentor(
        agent_name="demo-agent",
        capture_content=True,
    )
    instrumentor.activate()

    assert instrumentor._is_instrumented is True
    assert instrumentor._otel_instrumentor.instrument_kwargs == {
        "tracer_provider": tracer_provider,
        "agent_name": "demo-agent",
        "capture_content": True,
    }
    assert fake.spans_module.set_response_content is not fake.original_set_response_content
    assert fake.spans_module.set_result_attributes is not fake.original_set_result_attributes

    fake_query = fake.query_class()
    fake_query._otel_invocation_ctx = SimpleNamespace(marker="client-session")
    seen_ctx = asyncio.run(fake_query._handle_control_request({"request": {}}))
    assert seen_ctx.marker == "client-session"

    span = SimpleNamespace(attributes={})
    span.set_attribute = lambda key, value: span.attributes.__setitem__(key, value)

    fake.spans_module.set_response_content(
        span,
        [{"type": "tool_use", "id": "toolu_123", "name": "calculator"}],
    )
    fake.spans_module.set_response_content(
        span,
        [{"type": "text", "text": "The tip is $18.00."}],
    )
    fake.spans_module.set_result_attributes(
        span,
        SimpleNamespace(
            usage={
                "input_tokens": 4,
                "cache_creation_input_tokens": 291,
                "cache_read_input_tokens": 39025,
                "output_tokens": 121,
            },
            total_cost_usd=0.04241955,
        ),
    )

    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_123", "name": "calculator"}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The tip is $18.00."}],
        },
    ]
    assert span.attributes["gen_ai.usage.input_tokens"] == 4
    assert span.attributes["cost"] == 0.04241955

    instrumentor.deactivate()

    assert instrumentor._is_instrumented is False
    assert tracer_provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert fake.spans_module.set_result_attributes is fake.original_set_result_attributes
    restored_ctx = asyncio.run(fake_query._handle_control_request({"request": {}}))
    assert restored_ctx is None


def test_activate_logs_warning_when_dependency_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "opentelemetry.instrumentation.claude_agent_sdk":
            raise ImportError("missing claude agent sdk instrumentation")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "missing dependency" in caplog.text


def test_activate_cleans_up_when_upstream_instrument_fails(monkeypatch, caplog):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_claude_agent_sdk_modules(
        monkeypatch,
        instrument_error=RuntimeError("boom"),
    )

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert instrumentor._otel_instrumentor is None
    assert tracer_provider._active_span_processor._span_processors == ()
    assert fake.spans_module.set_response_content is fake.original_set_response_content
    assert "Failed to activate Claude Agent SDK instrumentation" in caplog.text


def test_deactivate_is_a_noop_when_not_instrumented():
    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.deactivate()

    assert instrumentor._is_instrumented is False


def test_processor_helpers_parse_and_normalize_values():
    assert _processor._safe_json_loads('{"a": 1}') == {"a": 1}
    assert _processor._safe_json_loads("({'a': 1})") == {"a": 1}
    assert _processor._safe_json_loads("plain-text") is None
    assert _processor._json_string({"a": 1}) == '{"a": 1}'
    assert _processor._json_string('[{"a":1}]') == '[{"a": 1}]'
    assert _processor._json_string("plain-text") == "plain-text"


def test_extract_usage_normalizes_cached_prompt_tokens():
    prompt_tokens, completion_tokens, cache_hit_tokens, cache_creation_tokens = (
        _processor._extract_usage(
            {
                "gen_ai.usage.input_tokens": 39320,
                "gen_ai.usage.cache_read_input_tokens": 39025,
                "gen_ai.usage.cache_creation_input_tokens": 291,
                "gen_ai.usage.output_tokens": 121,
            }
        )
    )

    assert prompt_tokens == 4
    assert completion_tokens == 121
    assert cache_hit_tokens == 39025
    assert cache_creation_tokens == 291


def test_extract_input_output_prefers_messages_and_value_fallbacks():
    input_value, output_value = _processor._extract_input_output(
        {
            "gen_ai.system_instructions": "Always be concise.",
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        }
    )

    assert json.loads(input_value) == [
        {"role": "system", "content": "Always be concise."},
        {"role": "user", "content": "hi"},
    ]
    assert json.loads(output_value) == [{"role": "assistant", "content": "hello"}]

    fallback_input, fallback_output = _processor._extract_input_output(
        {
            "input.value": {"prompt": "fallback"},
            "output.value": {"answer": "ok"},
        }
    )

    assert json.loads(fallback_input) == {"prompt": "fallback"}
    assert json.loads(fallback_output) == {"answer": "ok"}


def test_extract_tools_and_tool_calls_normalize_supported_shapes():
    tools = _processor._extract_tools(
        {
            "gen_ai.tool.definitions": json.dumps(
                [
                    "get_weather",
                    {"name": "calculator", "input_schema": {"type": "object"}},
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_customer",
                            "description": "Find a customer.",
                            "parameters": {"type": "object"},
                            "strict": True,
                        },
                    },
                ]
            )
        }
    )

    assert tools == [
        {"type": "function", "function": {"name": "get_weather"}},
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Find a customer.",
                "parameters": {"type": "object"},
                "strict": True,
            },
        },
    ]

    tool_calls = _processor._extract_tool_calls(
        {
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "calculator",
                                "input": {"expression": "120 * 0.15"},
                            },
                            {"type": "text", "text": "The tip is $18.00."},
                        ],
                    }
                ]
            )
        }
    )

    assert tool_calls == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "120 * 0.15"}',
            },
        }
    ]


def test_extract_existing_tool_calls_and_key_helpers():
    span = _make_span(name="tool-span", trace_id=22, span_id=33, parent_span_id=11)

    assert _processor._extract_existing_tool_calls(
        {"tool_calls": [{"id": "override"}]}
    ) == [{"id": "override"}]
    assert _processor._extract_existing_tool_calls(
        {RESPAN_SPAN_TOOL_CALLS: '[{"id":"parsed"}]'}
    ) == [{"id": "parsed"}]
    assert _processor._get_span_key(span) == (22, 33)
    assert _processor._get_parent_span_key(span) == (22, 11)


def test_build_tool_call_from_tool_span_attrs_and_merge_tool_calls():
    built_tool_call = _processor._build_tool_call_from_tool_span_attrs(
        {
            SpanAttributes.TRACELOOP_ENTITY_NAME: "calculator",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: {"expression": "120 * 0.15"},
            "gen_ai.tool.call.id": "toolu_123",
        }
    )

    assert built_tool_call == {
        "id": "toolu_123",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "120 * 0.15"}',
        },
    }

    merged_tool_calls = _processor._merge_tool_calls(
        [built_tool_call],
        [built_tool_call, {"id": "toolu_456", "function": {"name": "search", "arguments": "{}"}}],
    )

    assert merged_tool_calls == [
        built_tool_call,
        {"id": "toolu_456", "function": {"name": "search", "arguments": "{}"}},
    ]


def test_is_claude_agent_sdk_span_recognizes_supported_shapes():
    invoke_span = _make_span(name="invoke_agent weather")
    tool_span = _make_span(name="execute_tool calculator")
    other_span = _make_span(name="http.request")

    assert _processor.is_claude_agent_sdk_span(
        invoke_span,
        {"gen_ai.operation.name": "invoke_agent"},
    )
    assert _processor.is_claude_agent_sdk_span(
        tool_span,
        {"gen_ai.tool.name": "calculator"},
    )
    assert _processor.is_claude_agent_sdk_span(other_span, {}) is False


def test_enrich_claude_agent_sdk_span_maps_agent_fields():
    span = _make_span(
        name="invoke_agent weather_agent",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": "anthropic",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.conversation.id": "session-123",
            "gen_ai.system_instructions": "Always call tools first.",
            "gen_ai.input.messages": '[{"role":"user","content":"weather?"}]',
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Tokyo is sunny."}],
                    },
                ]
            ),
            "gen_ai.tool.definitions": '[{"name":"get_weather"}]',
            "gen_ai.response.model": "claude-sonnet-4-5",
            "gen_ai.usage.input_tokens": 19,
            "gen_ai.usage.output_tokens": 7,
        },
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes[RESPAN_LOG_METHOD] == LogMethodChoices.TRACING_INTEGRATION.value
    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.AGENT.value
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather_agent"
    assert span._attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "weather_agent"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "system", "content": "Always call tools first."},
        {"role": "user", "content": "weather?"},
    ]
    assert span._attributes["model"] == "claude-sonnet-4-5"
    assert span._attributes["prompt_tokens"] == 19
    assert span._attributes["completion_tokens"] == 7
    assert span._attributes["total_request_tokens"] == 26
    assert span._attributes[RESPAN_SESSION_ID] == "session-123"
    assert json.loads(span._attributes[RESPAN_SPAN_TOOLS]) == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]
    assert json.loads(span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert "gen_ai.agent.name" not in span._attributes
    assert "gen_ai.input.messages" not in span._attributes
    assert "gen_ai.output.messages" not in span._attributes
    assert "tool_calls" not in span._attributes


def test_enrich_claude_agent_sdk_span_maps_tool_fields():
    span = _make_span(
        name="execute_tool calculator",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "calculator",
            "gen_ai.tool.call.arguments": {"expression": "120 * 0.15"},
            "gen_ai.tool.call.result": {"content": [{"type": "text", "text": "18"}]},
            "gen_ai.response.model": "claude-sonnet-4-5",
            "gen_ai.usage.input_tokens": 6,
            "gen_ai.usage.output_tokens": 2,
            LLM_REQUEST_TYPE: "chat",
        },
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TOOL.value
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "calculator"
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == {
        "expression": "120 * 0.15"
    }
    assert json.loads(span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "content": [{"type": "text", "text": "18"}]
    }
    assert span._attributes["model"] == "claude-sonnet-4-5"
    assert span._attributes["prompt_tokens"] == 6
    assert span._attributes["completion_tokens"] == 2
    assert LLM_REQUEST_TYPE not in span._attributes
    assert span._attributes["tools"] == [
        {"type": "function", "function": {"name": "calculator"}}
    ]
    assert GEN_AI_TOOL_NAME not in span._attributes
    assert GEN_AI_TOOL_CALL_ARGUMENTS not in span._attributes
    assert GEN_AI_TOOL_CALL_RESULT not in span._attributes


def test_enrich_claude_agent_sdk_span_leaves_unrelated_spans_untouched():
    span = _make_span(
        name="http.request",
        attributes={"http.method": "POST"},
    )

    _processor.enrich_claude_agent_sdk_span(span)

    assert span._attributes == {"http.method": "POST"}


def test_span_processor_on_end_merges_pending_tool_calls_into_parent_agent_span():
    processor = _processor.ClaudeAgentSDKSpanProcessor()

    tool_span = _make_span(
        name="execute_tool get_weather",
        trace_id=55,
        span_id=2,
        parent_span_id=1,
        start_time=20,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "get_weather",
            "gen_ai.tool.call.arguments": {"city": "Tokyo"},
            "gen_ai.tool.call.result": {"temperature": "22C"},
            "gen_ai.tool.call.id": "toolu_123",
        },
    )
    agent_span = _make_span(
        name="invoke_agent weather_agent",
        trace_id=55,
        span_id=1,
        start_time=10,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    }
                ]
            ),
        },
    )

    processor.on_start(tool_span)
    processor.on_end(tool_span)
    processor.on_end(agent_span)

    assert json.loads(agent_span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]


def test_span_processor_on_end_injects_final_chat_child_for_tool_turn(monkeypatch):
    processor = _processor.ClaudeAgentSDKSpanProcessor()
    built_span_call: dict[str, object] = {}
    injected_spans: list[object] = []

    def fake_build_readable_span(name: str, **kwargs):
        built_span_call["name"] = name
        built_span_call["kwargs"] = kwargs
        return SimpleNamespace(name=name, **kwargs)

    monkeypatch.setattr(_processor, "build_readable_span", fake_build_readable_span)
    monkeypatch.setattr(
        _processor,
        "inject_span",
        lambda span: injected_spans.append(span) or True,
    )

    agent_span = _make_span(
        name="invoke_agent weather_agent",
        trace_id=88,
        span_id=7,
        start_time=100,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "weather_agent",
            "gen_ai.system": "anthropic",
            "gen_ai.input.messages": '[{"role":"user","content":"weather?"}]',
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "Tokyo"},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Tokyo is sunny."}],
                    },
                ]
            ),
            "gen_ai.response.model": "claude-sonnet-4-5",
        },
    )

    processor.on_end(agent_span)

    assert json.loads(agent_span._attributes[RESPAN_SPAN_TOOL_CALLS]) == [
        {
            "id": "toolu_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        }
    ]
    assert built_span_call["name"] == "assistant_message"

    child_span_kwargs = built_span_call["kwargs"]
    child_attrs = child_span_kwargs["attributes"]
    assert child_span_kwargs["trace_id"] == format(88, "032x")
    assert child_span_kwargs["parent_id"] == format(7, "016x")
    assert child_attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert child_attrs[LLM_REQUEST_TYPE] == "chat"
    assert child_attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "assistant_message"
    assert child_attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == "assistant_message"
    assert child_attrs["gen_ai.completion.0.role"] == "assistant"
    assert child_attrs["gen_ai.completion.0.content"] == "Tokyo is sunny."
    assert child_attrs["gen_ai.request.model"] == "claude-sonnet-4-5"
    assert child_attrs["gen_ai.system"] == "anthropic"
    assert json.loads(child_attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
        {"role": "user", "content": "weather?"}
    ]
    assert json.loads(child_attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Tokyo is sunny."}],
    }
    assert injected_spans and injected_spans[0].name == "assistant_message"


def test_span_processor_shutdown_clears_pending_calls_and_force_flush_returns_true():
    processor = _processor.ClaudeAgentSDKSpanProcessor()
    tool_span = _make_span(
        name="execute_tool calculator",
        trace_id=77,
        span_id=4,
        parent_span_id=3,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "calculator",
            "gen_ai.tool.call.arguments": {"expression": "2 + 2"},
        },
    )

    processor.on_end(tool_span)

    assert processor._pending_tool_calls_by_parent

    processor.shutdown()

    assert processor._pending_tool_calls_by_parent == {}
    assert processor.force_flush() is True
