import asyncio
import logging
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_mcp import MCPInstrumentor
from respan_instrumentation_mcp import _instrumentation
from respan_instrumentation_mcp._instrumentation import (
    MCP_CLIENT_SESSION_MODULE,
    OPENINFERENCE_MCP_MODULE,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name):
        span = FakeSpan(name)
        self.spans.append(span)
        yield span


def _install_fake_modules(monkeypatch):
    class FakeOpenInferenceMCPInstrumentor:
        pass

    class FakeOpenInferenceInstrumentor:
        created = []

        def __init__(self, instrumentor_class, **kwargs):
            self.instrumentor_class = instrumentor_class
            self.kwargs = kwargs
            self.is_activated = False
            self.is_deactivated = False
            self.__class__.created.append(self)

        def activate(self):
            self.is_activated = True

        def deactivate(self):
            self.is_deactivated = True

    class FakeClientSession:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name="summarize_city", description="demo")]
            )

        async def call_tool(self, name, arguments=None):
            return SimpleNamespace(content=[{"type": "text", "text": f"{name}: ok"}])

    openinference_module = ModuleType("openinference")
    openinference_instrumentation_module = ModuleType("openinference.instrumentation")
    openinference_mcp_module = ModuleType(OPENINFERENCE_MCP_MODULE)
    openinference_mcp_module.MCPInstrumentor = FakeOpenInferenceMCPInstrumentor
    openinference_instrumentation_module.mcp = openinference_mcp_module

    mcp_module = ModuleType("mcp")
    mcp_client_module = ModuleType("mcp.client")
    mcp_client_session_module = ModuleType(MCP_CLIENT_SESSION_MODULE)
    mcp_client_session_module.ClientSession = FakeClientSession
    mcp_client_module.session = mcp_client_session_module
    mcp_module.client = mcp_client_module

    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation",
        openinference_instrumentation_module,
    )
    monkeypatch.setitem(sys.modules, OPENINFERENCE_MCP_MODULE, openinference_mcp_module)
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_module)
    monkeypatch.setitem(sys.modules, MCP_CLIENT_SESSION_MODULE, mcp_client_session_module)
    monkeypatch.setattr(
        _instrumentation,
        "OpenInferenceInstrumentor",
        FakeOpenInferenceInstrumentor,
    )

    return SimpleNamespace(
        client_session_class=FakeClientSession,
        openinference_mcp_class=FakeOpenInferenceMCPInstrumentor,
        delegate_class=FakeOpenInferenceInstrumentor,
    )


@pytest.fixture(autouse=True)
def reset_state():
    RespanTracer.reset_instance()
    MCPInstrumentor._patches_applied = False
    yield
    RespanTracer.reset_instance()
    MCPInstrumentor._patches_applied = False


def test_activate_delegates_to_openinference_and_traces_client_calls(monkeypatch):
    fake = _install_fake_modules(monkeypatch)
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    instrumentor = MCPInstrumentor(capture_headers=True)
    instrumentor.activate()

    delegate = fake.delegate_class.created[0]
    assert delegate.instrumentor_class is fake.openinference_mcp_class
    assert delegate.kwargs == {"capture_headers": True}
    assert delegate.is_activated is True
    assert instrumentor._is_instrumented is True

    session = fake.client_session_class()
    result = asyncio.run(session.call_tool("summarize_city", {"city": "Paris"}))

    assert result.content[0]["text"] == "summarize_city: ok"
    span = fake_tracer.spans[0]
    assert span.name == "mcp.tool.summarize_city"
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "summarize_city"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "summarize_city"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '{"arguments": {"city": "Paris"}, "name": "summarize_city"}'
    )
    assert "openinference.span.kind" not in span.attributes
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes

    instrumentor.deactivate()

    assert delegate.is_deactivated is True
    assert instrumentor._is_instrumented is False


def test_trace_list_tools_uses_task_log_type(monkeypatch):
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    async def wrapped():
        return SimpleNamespace(tools=[SimpleNamespace(name="first")])

    instrumentor = MCPInstrumentor()
    result = asyncio.run(
        instrumentor._trace_async_method("list_tools", wrapped, (), {})
    )

    assert result.tools[0].name == "first"
    span = fake_tracer.spans[0]
    assert span.name == "mcp.list_tools"
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "mcp.list_tools"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '{"method": "list_tools"}'
    )
    assert '"tools": [{"name": "first"}]' in span.attributes[
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT
    ]


def test_trace_error_sets_status_and_error_output(monkeypatch):
    fake_tracer = FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _name: fake_tracer)

    async def wrapped(*args, **kwargs):
        raise RuntimeError("tool failed")

    instrumentor = MCPInstrumentor()

    with pytest.raises(RuntimeError, match="tool failed"):
        asyncio.run(
            instrumentor._trace_async_method(
                "call_tool",
                wrapped,
                ("broken_tool",),
                {"arguments": {"x": 1}},
            )
        )

    span = fake_tracer.spans[0]
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert span.exceptions and isinstance(span.exceptions[0], RuntimeError)
    assert span.status.status_code.name == "ERROR"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '{"error": "RuntimeError", "message": "tool failed"}'
    )


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake = _install_fake_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = MCPInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert fake.delegate_class.created == []
    assert instrumentor._is_instrumented is False
    assert "MCP instrumentation skipped because Respan tracing is disabled" in caplog.text
