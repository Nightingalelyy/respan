from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_vertexai import VertexAIInstrumentor, _instrumentation
from respan_instrumentation_vertexai._constants import (
    GENERATE_CONTENT_METHOD_NAME,
    GENERATIVE_MODEL_CLASS_NAME,
    VERTEXAI_GENERATIVE_MODELS_MODULE,
)
from respan_instrumentation_vertexai._serialization import json_dumps


def _obj(**values: Any) -> SimpleNamespace:
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _clean_runtime() -> Any:
    _instrumentation._reset_runtime_for_tests()
    yield
    _instrumentation._reset_runtime_for_tests()


def _current_model_class() -> type[Any]:
    module = pytest.importorskip(VERTEXAI_GENERATIVE_MODELS_MODULE)
    return getattr(module, GENERATIVE_MODEL_CLASS_NAME)


def _response(text: str, *, usage: Any = None) -> Any:
    content = _obj(parts=[_obj(text=text)], role="model")
    return _obj(
        text=text,
        candidates=[_obj(content=content)],
        usage_metadata=usage,
    )


def test_real_current_vertex_stream_exports_canonical_readable_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_class = _current_model_class()
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_vertexai._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    def fake_generate(self: Any, contents: Any, **kwargs: Any) -> Any:
        del self, contents, kwargs
        return iter(
            [
                _response("Vertex "),
                _response(
                    "streamed.",
                    usage=_obj(
                        prompt_token_count=8,
                        candidates_token_count=3,
                        thoughts_token_count=1,
                        total_token_count=12,
                    ),
                ),
            ]
        )

    monkeypatch.setattr(model_class, GENERATE_CONTENT_METHOD_NAME, fake_generate)
    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()
    instance = _obj(_model_name="gemini-2.5-flash", _tools=None)
    provider = TracerProvider()
    with provider.get_tracer(__name__).start_as_current_span("parent") as parent:
        stream = getattr(model_class, GENERATE_CONTENT_METHOD_NAME)(
            instance,
            "Say streamed.",
            stream=True,
        )
        assert [chunk for chunk in stream]
        parent_context = parent.get_span_context()
    instrumentor.deactivate()

    assert len(emitted) == 1
    span = emitted[0]
    attrs = span.attributes
    assert span.parent.span_id == parent_context.span_id
    assert attrs[SpanAttributes.LLM_IS_STREAMING] is True
    assert attrs[gen_ai_attributes.GEN_AI_PROVIDER_NAME] == "google_vertex_ai"
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Vertex streamed."
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs


def test_real_current_vertex_tool_exports_canonical_function_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = pytest.importorskip(VERTEXAI_GENERATIVE_MODELS_MODULE)
    model_class = getattr(module, GENERATIVE_MODEL_CLASS_NAME)
    function_declaration = module.FunctionDeclaration(
        name="get_weather",
        description="Return deterministic weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    tool = module.Tool(function_declarations=[function_declaration])
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_vertexai._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    def fake_generate(self: Any, contents: Any, **kwargs: Any) -> Any:
        del self, contents, kwargs
        return _response("Tool definition captured.")

    monkeypatch.setattr(model_class, GENERATE_CONTENT_METHOD_NAME, fake_generate)
    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()
    getattr(model_class, GENERATE_CONTENT_METHOD_NAME)(
        _obj(_model_name="gemini-2.5-flash", _tools=[tool]),
        "Use get_weather.",
    )
    instrumentor.deactivate()

    assert len(emitted) == 1
    definitions = json.loads(
        emitted[0].attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS]
    )
    assert definitions == [
        {
            "function": {
                "description": "Return deterministic weather for a city.",
                "name": "get_weather",
                "parameters": {
                    "properties": {"city": {"type": "STRING"}},
                    "property_ordering": ["city"],
                    "required": ["city"],
                    "type": "OBJECT",
                },
            },
            "type": "function",
        }
    ]


def test_real_current_vertex_lifecycle_is_shared_and_foreign_safe() -> None:
    model_class = _current_model_class()
    original = getattr(model_class, GENERATE_CONTENT_METHOD_NAME)
    first = VertexAIInstrumentor()
    second = VertexAIInstrumentor()
    first.activate()
    installed = getattr(model_class, GENERATE_CONTENT_METHOD_NAME)
    second.activate()
    first.deactivate()
    assert getattr(model_class, GENERATE_CONTENT_METHOD_NAME) is installed
    second.deactivate()
    assert getattr(model_class, GENERATE_CONTENT_METHOD_NAME) is original

    first.activate()
    installed = getattr(model_class, GENERATE_CONTENT_METHOD_NAME)

    def foreign(self: Any, *args: Any, **kwargs: Any) -> Any:
        return installed(self, *args, **kwargs)

    setattr(model_class, GENERATE_CONTENT_METHOD_NAME, foreign)
    first.deactivate()
    assert getattr(model_class, GENERATE_CONTENT_METHOD_NAME) is foreign
    setattr(model_class, GENERATE_CONTENT_METHOD_NAME, original)


def test_vertex_error_status_privacy_and_hostile_exception_are_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_class = _current_model_class()
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_vertexai._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    class ProviderError(Exception):
        status_code = 503

        def __str__(self) -> str:
            raise AssertionError("instrumentation called hostile __str__")

    error = ProviderError("password=plain-secret provider unavailable")

    def fail(self: Any, contents: Any, **kwargs: Any) -> Any:
        del self, contents, kwargs
        raise error

    monkeypatch.setattr(model_class, GENERATE_CONTENT_METHOD_NAME, fail)
    instrumentor = VertexAIInstrumentor()
    instrumentor.activate()
    with pytest.raises(ProviderError) as caught:
        getattr(model_class, GENERATE_CONTENT_METHOD_NAME)(
            _obj(_model_name="gemini-2.5-flash"),
            "hello",
        )
    instrumentor.deactivate()
    assert caught.value is error
    assert len(emitted) == 1
    span = emitted[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["http.response.status_code"] == 503
    assert "plain-secret" not in json.dumps(dict(span.attributes), sort_keys=True)


def test_vertex_json_capture_is_valid_bounded_and_redacted() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError

        __repr__ = __str__

    value = json_dumps(
        {
            "api_key": "plain-secret",
            "nested": {"auth_token": "token-value", "object": Hostile()},
            "unicode": "😀" * 20_000,
        }
    )
    assert len(value.encode("utf-8")) <= 16_000
    assert "plain-secret" not in value
    assert "token-value" not in value
    assert json.loads(value)
