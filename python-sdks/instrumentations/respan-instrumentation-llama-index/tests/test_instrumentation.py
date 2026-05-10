import logging
from types import SimpleNamespace

import pytest
from llama_index.core import instrumentation
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TEXT,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    LLM_REQUEST_TYPE,
    RESPAN_LOG_TYPE,
)
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

from respan_instrumentation_llama_index import LlamaIndexInstrumentor
from respan_instrumentation_llama_index import _instrumentation
from respan_instrumentation_llama_index._handlers import (
    RespanLlamaIndexEventHandler,
    RespanLlamaIndexSpanHandler,
)
from respan_instrumentation_llama_index._serialization import extract_usage


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="llama-index-test",
        api_key="test-key",
        is_auto_instrument=False,
        is_batching_enabled=False,
    )
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter=exporter)
    )
    yield exporter
    telemetry.flush()


def test_activate_registers_native_handlers():
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.activate()

    assert instrumentor._span_handler in instrumentation.root_dispatcher.span_handlers
    assert instrumentor._event_handler in instrumentation.root_dispatcher.event_handlers
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert (
        instrumentor._span_handler not in instrumentation.root_dispatcher.span_handlers
    )
    assert (
        instrumentor._event_handler
        not in instrumentation.root_dispatcher.event_handlers
    )
    assert instrumentor._is_instrumented is False


def test_activate_is_idempotent():
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.activate()
    instrumentor.activate()

    assert (
        instrumentation.root_dispatcher.span_handlers.count(instrumentor._span_handler)
        == 1
    )
    assert (
        instrumentation.root_dispatcher.event_handlers.count(
            instrumentor._event_handler
        )
        == 1
    )

    instrumentor.deactivate()


def test_activate_skips_when_respan_tracing_is_disabled(caplog):
    RespanTracer(is_enabled=False)

    instrumentor = LlamaIndexInstrumentor()
    with caplog.at_level(logging.INFO):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert (
        "LlamaIndex instrumentation skipped because Respan tracing is disabled"
        in caplog.text
    )


def test_activate_logs_warning_when_dependency_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        raise ImportError(module_name)

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )
    instrumentor = LlamaIndexInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate LlamaIndex instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_span_handler_emits_workflow_and_task_spans(span_exporter):
    handler = RespanLlamaIndexSpanHandler()
    root_bound_args = SimpleNamespace(args=("question",), kwargs={})
    child_bound_args = SimpleNamespace(args=(), kwargs={"top_k": 2})

    handler.span_enter(
        id_="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
        bound_args=root_bound_args,
        parent_id=None,
    )
    handler.span_enter(
        id_="BaseRetriever.retrieve-22222222-2222-2222-2222-222222222222",
        bound_args=child_bound_args,
        parent_id="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
    )
    handler.span_exit(
        id_="BaseRetriever.retrieve-22222222-2222-2222-2222-222222222222",
        bound_args=child_bound_args,
        result=["node"],
    )
    handler.span_exit(
        id_="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
        bound_args=root_bound_args,
        result="answer",
    )

    spans = span_exporter.get_finished_spans()
    attrs_by_name = {span.name: span.attributes for span in spans}

    assert (
        attrs_by_name["RetrieverQueryEngine.query"][RESPAN_LOG_TYPE]
        == LOG_TYPE_WORKFLOW
    )
    assert attrs_by_name["BaseRetriever.retrieve"][RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert (
        attrs_by_name["RetrieverQueryEngine.query"][
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        ]
        == '"answer"'
    )


def test_chat_events_emit_canonical_llm_span(span_exporter):
    handler = RespanLlamaIndexEventHandler()
    user_message = SimpleNamespace(role="user", content="Hello")
    assistant_message = SimpleNamespace(role="assistant", content="Hi")
    response = SimpleNamespace(
        message=assistant_message,
        raw={
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
        },
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-1",
            messages=[user_message],
            model_dict={"class_name": "OpenAI", "model_name": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-1",
            response=response,
        )
    )

    spans = span_exporter.get_finished_spans()
    chat_span = next(span for span in spans if span.name == "llama_index.chat")
    attributes = chat_span.attributes

    assert attributes[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attributes[LLM_REQUEST_TYPE] == "chat"
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert attributes["gen_ai.prompt.0.role"] == "user"
    assert attributes["gen_ai.prompt.0.content"] == "Hello"
    assert attributes["gen_ai.completion.0.role"] == "assistant"
    assert attributes["gen_ai.completion.0.content"] == "Hi"
    assert attributes["gen_ai.usage.prompt_tokens"] == 3
    assert attributes["gen_ai.usage.completion_tokens"] == 2
    assert attributes["llm.usage.total_tokens"] == 5


def test_chat_events_can_disable_content_capture(span_exporter):
    handler = RespanLlamaIndexEventHandler(capture_content=False)
    user_message = SimpleNamespace(role="user", content="hidden")
    response = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content="also hidden"),
        raw={},
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatStartEvent",
            span_id="span-1",
            messages=[user_message],
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMChatEndEvent",
            span_id="span-1",
            response=response,
        )
    )

    chat_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.chat"
    )

    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in chat_span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in chat_span.attributes


def test_completion_events_emit_text_span(span_exporter):
    handler = RespanLlamaIndexEventHandler()
    response = SimpleNamespace(
        text="A short completion.",
        raw={
            "usage": {
                "input_tokens": 4,
                "output_tokens": 3,
            }
        },
    )

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="span-2",
            prompt="Complete this sentence",
            model_dict={"class_name": "OpenAI", "model": "gpt-4o-mini"},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionEndEvent",
            span_id="span-2",
            response=response,
        )
    )

    text_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.completion"
    )
    attributes = text_span.attributes

    assert attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert attributes[LLM_REQUEST_TYPE] == "completion"
    assert attributes["gen_ai.prompt.0.content"] == "Complete this sentence"
    assert attributes["gen_ai.completion.0.content"] == "A short completion."
    assert attributes["llm.usage.total_tokens"] == 7


def test_embedding_events_emit_embedding_span_without_vectors(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "EmbeddingStartEvent",
            span_id="span-3",
            model_dict={
                "class_name": "OpenAIEmbedding",
                "model": "text-embedding-3-small",
            },
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "EmbeddingEndEvent",
            span_id="span-3",
            chunks=["alpha", "beta"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
    )

    embedding_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.embedding"
    )
    attributes = embedding_span.attributes

    assert attributes[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert attributes[LLM_REQUEST_TYPE] == "embedding"
    assert "embedding_count" in attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert "0.1" not in attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]


def test_tool_event_emits_tool_span(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "AgentToolCallEvent",
            tool=SimpleNamespace(name="lookup_order"),
            arguments='{"order_id": "ord_123"}',
        )
    )

    tool_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.tool.lookup_order"
    )

    assert tool_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert "ord_123" in tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]


def test_tool_event_respects_content_capture_setting(span_exporter):
    handler = RespanLlamaIndexEventHandler(capture_content=False)

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "AgentToolCallEvent",
            tool=SimpleNamespace(name="lookup_order"),
            arguments='{"order_id": "ord_123"}',
        )
    )

    tool_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.tool.lookup_order"
    )

    assert tool_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in tool_span.attributes


def test_exception_event_marks_open_event_span_error(span_exporter):
    handler = RespanLlamaIndexEventHandler()

    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionStartEvent",
            span_id="span-4",
            prompt="raise",
            model_dict={},
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "ExceptionEvent",
            span_id="span-4",
            exception=RuntimeError("llama failure"),
        )
    )
    handler.handle(
        SimpleNamespace(
            class_name=lambda: "LLMCompletionEndEvent",
            span_id="span-4",
            response=SimpleNamespace(text="fallback", raw={}),
        )
    )

    completion_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "llama_index.completion"
    )

    assert completion_span.status.status_code == StatusCode.ERROR


def test_extract_usage_ignores_fractional_token_counts():
    prompt_tokens, completion_tokens, total_tokens = extract_usage(
        response={
            "usage": {
                "prompt_tokens": 3.5,
                "completion_tokens": 2,
                "total_tokens": 5.5,
            }
        }
    )

    assert prompt_tokens is None
    assert completion_tokens == 2
    assert total_tokens == 2
