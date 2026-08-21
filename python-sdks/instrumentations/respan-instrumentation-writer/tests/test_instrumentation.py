from __future__ import annotations

import asyncio
import json
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from respan_instrumentation_writer import (
    WriterInstrumentor,
    _instrumentation,
    _otel_emitter,
)
from respan_instrumentation_writer._constants import (
    WRITER_PARSE_PDF_TOOL_NAME,
    WRITER_WEB_SEARCH_TOOL_NAME,
)
from respan_instrumentation_writer._translator import (
    build_application_generate_attrs,
    build_completion_attrs,
    build_graph_question_attrs,
    build_tool_attrs,
    build_translation_attrs,
    build_vision_attrs,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TEXT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.exporters.respan import _export_span_name

OFF_CONTRACT_ALIASES = {
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "tools",
    "tool_calls",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
}


@pytest.fixture(autouse=True)
def reset_state() -> None:
    RespanTracer.reset_instance()
    _instrumentation._activation_count = 0
    _instrumentation._activation_installing = False
    _instrumentation._installed_methods.clear()
    yield
    RespanTracer.reset_instance()
    for name in list(vars(_instrumentation)):
        if name.startswith("_original_"):
            setattr(_instrumentation, name, None)
    _instrumentation._activation_count = 0
    _instrumentation._activation_installing = False
    _instrumentation._installed_methods.clear()
    for module_name in list(sys.modules):
        if module_name == "writerai" or module_name.startswith("writerai."):
            sys.modules.pop(module_name, None)


@pytest.fixture()
def captured_spans(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_writer_span",
        lambda **kwargs: spans.append(kwargs),
    )
    return spans


def _chat_response(model: str = "palmyra-x5") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="Use the forecast tool.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="get_weather",
                                arguments='{"city":"Tokyo"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            prompt_token_details=SimpleNamespace(cached_tokens=3),
        ),
    )


def _chat_stream_chunks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            model="palmyra-x5",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(role="assistant", content="Hello "),
                )
            ],
        ),
        SimpleNamespace(
            model="palmyra-x5",
            choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="stream"))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        ),
    ]


def _install_fake_writer_modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    class ChatResource:
        def chat(self, **kwargs: Any) -> Any:
            if kwargs.get("stream") is True:
                return iter(_chat_stream_chunks())
            return _chat_response(model=kwargs.get("model", "palmyra-x5"))

    class AsyncChatResource:
        async def chat(self, **kwargs: Any) -> Any:
            return _chat_response(model=kwargs.get("model", "palmyra-x5"))

    class CompletionsResource:
        def create(self, **kwargs: Any) -> Any:
            if kwargs.get("stream") is True:
                return iter(
                    [SimpleNamespace(value="alpha "), SimpleNamespace(value="beta")]
                )
            return SimpleNamespace(
                model=kwargs.get("model"),
                choices=[SimpleNamespace(text="Generated completion")],
            )

    class AsyncCompletionsResource:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                model=kwargs.get("model"),
                choices=[SimpleNamespace(text="Generated completion")],
            )

    class GraphsResource:
        def question(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                answer="Graph answer", question=kwargs.get("question")
            )

    class AsyncGraphsResource:
        async def question(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                answer="Graph answer", question=kwargs.get("question")
            )

    class ApplicationsResource:
        def generate_content(self, application_id: str, **kwargs: Any) -> Any:
            if kwargs.get("stream") is True:
                return iter(
                    [
                        SimpleNamespace(suggestion="Application "),
                        SimpleNamespace(suggestion="stream"),
                    ]
                )
            return SimpleNamespace(title="Summary", suggestion="Application answer")

    class AsyncApplicationsResource:
        async def generate_content(self, application_id: str, **kwargs: Any) -> Any:
            return SimpleNamespace(title="Summary", suggestion="Application answer")

    class VisionResource:
        def analyze(self, **kwargs: Any) -> Any:
            return SimpleNamespace(data="Vision answer")

    class AsyncVisionResource:
        async def analyze(self, **kwargs: Any) -> Any:
            return SimpleNamespace(data="Vision answer")

    class TranslationResource:
        def translate(self, **kwargs: Any) -> Any:
            return SimpleNamespace(data="Bonjour")

    class AsyncTranslationResource:
        async def translate(self, **kwargs: Any) -> Any:
            return SimpleNamespace(data="Bonjour")

    class ToolsResource:
        def web_search(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                query=kwargs.get("query"), answer="Search answer", sources=[]
            )

        def parse_pdf(self, file_id: str, **kwargs: Any) -> Any:
            return SimpleNamespace(content="PDF text")

    class AsyncToolsResource:
        async def web_search(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                query=kwargs.get("query"), answer="Search answer", sources=[]
            )

        async def parse_pdf(self, file_id: str, **kwargs: Any) -> Any:
            return SimpleNamespace(content="PDF text")

    module_specs = {
        "writerai.resources.chat": {
            "ChatResource": ChatResource,
            "AsyncChatResource": AsyncChatResource,
        },
        "writerai.resources.completions": {
            "CompletionsResource": CompletionsResource,
            "AsyncCompletionsResource": AsyncCompletionsResource,
        },
        "writerai.resources.graphs": {
            "GraphsResource": GraphsResource,
            "AsyncGraphsResource": AsyncGraphsResource,
        },
        "writerai.resources.applications.applications": {
            "ApplicationsResource": ApplicationsResource,
            "AsyncApplicationsResource": AsyncApplicationsResource,
        },
        "writerai.resources.vision": {
            "VisionResource": VisionResource,
            "AsyncVisionResource": AsyncVisionResource,
        },
        "writerai.resources.translation": {
            "TranslationResource": TranslationResource,
            "AsyncTranslationResource": AsyncTranslationResource,
        },
        "writerai.resources.tools": {
            "ToolsResource": ToolsResource,
            "AsyncToolsResource": AsyncToolsResource,
        },
    }

    package_names = [
        "writerai",
        "writerai.resources",
        "writerai.resources.applications",
    ]
    for package_name in package_names:
        monkeypatch.setitem(sys.modules, package_name, ModuleType(package_name))

    for module_name, attrs in module_specs.items():
        module = ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        monkeypatch.setitem(sys.modules, module_name, module)

    return SimpleNamespace(
        ChatResource=ChatResource,
        AsyncChatResource=AsyncChatResource,
        CompletionsResource=CompletionsResource,
        ApplicationsResource=ApplicationsResource,
        ToolsResource=ToolsResource,
    )


def _chat_request_kwargs() -> dict[str, Any]:
    return {
        "model": "palmyra-x5",
        "messages": [{"role": "user", "content": "Use a tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    }


def test_sync_chat_emits_canonical_attrs_without_aliases(
    monkeypatch: pytest.MonkeyPatch,
    captured_spans: list[dict[str, Any]],
) -> None:
    fake = _install_fake_writer_modules(monkeypatch)
    instrumentor = WriterInstrumentor()
    instrumentor.activate()

    response = fake.ChatResource().chat(**_chat_request_kwargs())

    assert response.choices[0].message.content == "Use the forecast tool."
    assert len(captured_spans) == 1
    attrs = captured_spans[0]["attrs"]
    assert captured_spans[0]["name"] == "writer.chat"
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[TLSpanAttributes.LLM_SYSTEM] == "writer"
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "writer"
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "palmyra-x5"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == "Use a tool."
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    assert (
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "Use the forecast tool."
    )
    assert (
        json.loads(attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"]["name"]
        == "get_weather"
    )
    assert (
        json.loads(attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == "call_1"
    )
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
    assert attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 8
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 20
    assert attrs[TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert TLSpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    instrumentor.deactivate()


def test_streaming_chat_emits_after_consumption(
    monkeypatch: pytest.MonkeyPatch,
    captured_spans: list[dict[str, Any]],
) -> None:
    fake = _install_fake_writer_modules(monkeypatch)
    WriterInstrumentor().activate()

    stream = fake.ChatResource().chat(**_chat_request_kwargs(), stream=True)
    assert [chunk.choices[0].delta.content for chunk in stream] == ["Hello ", "stream"]

    attrs = captured_spans[0]["attrs"]
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello stream"
    assert attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 5
    assert attrs[TLSpanAttributes.LLM_IS_STREAMING] is True
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_async_chat_is_patched(
    monkeypatch: pytest.MonkeyPatch,
    captured_spans: list[dict[str, Any]],
) -> None:
    fake = _install_fake_writer_modules(monkeypatch)
    WriterInstrumentor().activate()

    async def run() -> None:
        response = await fake.AsyncChatResource().chat(**_chat_request_kwargs())
        assert response.model == "palmyra-x5"

    asyncio.run(run())

    assert len(captured_spans) == 1
    assert captured_spans[0]["attrs"][TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"


def test_completion_mapper_uses_text_log_type_without_aliases() -> None:
    attrs = build_completion_attrs(
        request_kwargs={"model": "palmyra-x5", "prompt": "Write a headline."},
        response_or_chunks=SimpleNamespace(
            model="palmyra-x5",
            choices=[SimpleNamespace(text="Tracing works")],
        ),
    )

    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
    assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "palmyra-x5"
    assert attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] == "Write a headline."
    assert attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] == "Tracing works"
    assert not OFF_CONTRACT_ALIASES.intersection(attrs)


def test_other_writer_text_operations_map_to_canonical_attrs() -> None:
    graph_attrs = build_graph_question_attrs(
        request_kwargs={"graph_ids": ["graph_1"], "question": "What changed?"},
        response_or_chunks=SimpleNamespace(answer="The docs changed."),
    )
    app_attrs = build_application_generate_attrs(
        request_kwargs={
            "application_id": "app_1",
            "inputs": [{"id": "topic", "value": "AI"}],
        },
        response_or_chunks=SimpleNamespace(suggestion="Application output"),
    )
    vision_attrs = build_vision_attrs(
        request_kwargs={
            "model": "palmyra-vision",
            "prompt": "Describe {{image}}",
            "variables": [],
        },
        response=SimpleNamespace(data="Vision output"),
    )
    translation_attrs = build_translation_attrs(
        request_kwargs={
            "model": "palmyra-translate",
            "text": "Hello",
            "source_language_code": "en",
            "target_language_code": "fr",
            "formality": False,
            "length_control": False,
            "mask_profanity": False,
        },
        response=SimpleNamespace(data="Bonjour"),
    )

    for attrs in (graph_attrs, app_attrs, vision_attrs, translation_attrs):
        assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TEXT
        assert attrs[TLSpanAttributes.LLM_REQUEST_TYPE] == "chat"
        assert not OFF_CONTRACT_ALIASES.intersection(attrs)

    assert (
        graph_attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "The docs changed."
    )
    assert graph_attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "writer-graph"
    assert (
        app_attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "Application output"
    )
    assert app_attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "writer-application"
    assert vision_attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "palmyra-vision"
    assert translation_attrs[TLSpanAttributes.LLM_REQUEST_MODEL] == "palmyra-translate"


def test_direct_writer_tools_emit_tool_spans_without_llm_tool_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_attrs = build_tool_attrs(
        tool_name=WRITER_WEB_SEARCH_TOOL_NAME,
        request_kwargs={"query": "Respan tracing", "include_answer": True},
        response=SimpleNamespace(
            query="Respan tracing", answer="Search output", sources=[]
        ),
    )
    pdf_attrs = build_tool_attrs(
        tool_name=WRITER_PARSE_PDF_TOOL_NAME,
        request_kwargs={"file_id": "file_1", "format": "markdown"},
        response=SimpleNamespace(content="PDF output"),
    )

    for attrs, legacy_name, semantic_detail in (
        (web_attrs, WRITER_WEB_SEARCH_TOOL_NAME, "web_search"),
        (pdf_attrs, WRITER_PARSE_PDF_TOOL_NAME, "parse_pdf"),
    ):
        assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
        assert attrs[RESPAN_INTERNAL_SPAN_NAME_KIND] == LOG_TYPE_TOOL
        assert attrs[RESPAN_INTERNAL_SPAN_NAME_DETAIL] == semantic_detail
        assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT]
        assert attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        assert "gen_ai.tool.name" not in attrs
        assert "gen_ai.tool.call.arguments" not in attrs
        assert not OFF_CONTRACT_ALIASES.intersection(attrs)

        span = SimpleNamespace(name=legacy_name, attributes=attrs)
        assert _export_span_name(span) == f"tool.{semantic_detail}"

        monkeypatch.setenv("RESPAN_SPAN_NAME_STYLE", "legacy")
        assert _export_span_name(span) == legacy_name
        monkeypatch.delenv("RESPAN_SPAN_NAME_STYLE")


def test_completion_and_application_streams_emit_canonical_stream_flag(
    monkeypatch: pytest.MonkeyPatch,
    captured_spans: list[dict[str, Any]],
) -> None:
    fake = _install_fake_writer_modules(monkeypatch)
    instrumentor = WriterInstrumentor()
    instrumentor.activate()

    completion = fake.CompletionsResource().create(
        model="palmyra-x5",
        prompt="stream",
        stream=True,
    )
    application = fake.ApplicationsResource().generate_content(
        "app_1",
        inputs=[{"id": "topic", "value": "tracing"}],
        stream=True,
    )
    list(completion)
    list(application)

    assert len(captured_spans) == 2
    assert all(
        span["attrs"][TLSpanAttributes.LLM_IS_STREAMING] is True
        for span in captured_spans
    )
    instrumentor.deactivate()


def test_two_writer_instances_keep_patches_until_final_deactivate(
    monkeypatch: pytest.MonkeyPatch,
    captured_spans: list[dict[str, Any]],
) -> None:
    fake = _install_fake_writer_modules(monkeypatch)
    original = fake.ChatResource.chat
    first = WriterInstrumentor()
    second = WriterInstrumentor()
    first.activate()
    second.activate()

    first.deactivate()
    fake.ChatResource().chat(**_chat_request_kwargs())
    assert len(captured_spans) == 1
    assert fake.ChatResource.chat is not original

    second.deactivate()
    assert fake.ChatResource.chat is original


def test_writer_stream_keeps_call_time_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_writer_modules(monkeypatch)
    parent_ids: list[int] = []
    monkeypatch.setattr(
        _instrumentation,
        "emit_writer_span",
        lambda **_kwargs: parent_ids.append(
            trace.get_current_span().get_span_context().span_id
        ),
    )
    WriterInstrumentor().activate()
    parent_span_id = 0x4321
    parent = NonRecordingSpan(
        SpanContext(
            trace_id=0xABCDEF1234567890ABCDEF1234567890,
            span_id=parent_span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=None,
        )
    )
    token = context_api.attach(trace.set_span_in_context(parent))
    try:
        stream = fake.ChatResource().chat(**_chat_request_kwargs(), stream=True)
    finally:
        context_api.detach(token)

    list(stream)
    assert parent_ids == [parent_span_id]


def test_writer_stream_retention_is_bounded() -> None:
    chunks: list[Any] = []
    for index in range(_instrumentation._MAX_STREAM_CHUNKS + 20):
        _instrumentation._append_stream_chunk(chunks, index)

    assert len(chunks) == _instrumentation._MAX_STREAM_CHUNKS
    assert chunks[-1] == _instrumentation._MAX_STREAM_CHUNKS + 19


def test_writer_error_span_retains_provider_status_for_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []
    monkeypatch.setattr(
        _otel_emitter,
        "inject_span",
        lambda *, span: captured.append(span),
    )

    _otel_emitter.emit_writer_span(
        name="writer.chat",
        attrs={},
        start_ns=time.time_ns(),
        error_message="RateLimitError: provider limit",
        status_code=429,
    )

    assert len(captured) == 1
    assert captured[0].attributes["status_code"] == 429
    assert captured[0].status.is_ok is False
