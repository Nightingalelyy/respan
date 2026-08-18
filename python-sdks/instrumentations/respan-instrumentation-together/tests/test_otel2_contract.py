from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_together import TogetherInstrumentor, _instrumentation
from respan_instrumentation_together._constants import (
    COMPLETIONS_RESOURCE_CLASS_NAME,
    CREATE_METHOD_NAME,
    TOGETHER_CHAT_COMPLETIONS_MODULE,
)
from respan_instrumentation_together._serialization import json_dumps


def _obj(**values: Any) -> SimpleNamespace:
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _clean_runtime() -> Any:
    _instrumentation._reset_runtime_for_tests()
    yield
    _instrumentation._reset_runtime_for_tests()


def _current_chat_class() -> type[Any]:
    module = pytest.importorskip(TOGETHER_CHAT_COMPLETIONS_MODULE)
    return getattr(module, COMPLETIONS_RESOURCE_CLASS_NAME)


def test_real_current_together_stream_exports_canonical_readable_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_class = _current_chat_class()
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_together._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    def fake_create(self: Any, **kwargs: Any) -> Any:
        del self, kwargs
        return iter(
            [
                _obj(
                    choices=[_obj(delta=_obj(content="Together "), finish_reason=None)],
                    usage=None,
                ),
                _obj(
                    choices=[
                        _obj(delta=_obj(content="streamed."), finish_reason="stop")
                    ],
                    usage=_obj(
                        prompt_tokens=9,
                        completion_tokens=4,
                        total_tokens=13,
                    ),
                ),
            ]
        )

    monkeypatch.setattr(chat_class, CREATE_METHOD_NAME, fake_create)
    instrumentor = TogetherInstrumentor()
    instrumentor.activate()
    resource = object.__new__(chat_class)
    provider = TracerProvider()
    with provider.get_tracer(__name__).start_as_current_span("parent") as parent:
        stream = resource.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say streamed."}],
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
    assert attrs[gen_ai_attributes.GEN_AI_PROVIDER_NAME] == "together"
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 9
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Together streamed."
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs


def test_real_current_together_lifecycle_is_shared_and_foreign_safe() -> None:
    chat_class = _current_chat_class()
    original = getattr(chat_class, CREATE_METHOD_NAME)
    first = TogetherInstrumentor()
    second = TogetherInstrumentor()
    first.activate()
    installed = getattr(chat_class, CREATE_METHOD_NAME)
    second.activate()
    first.deactivate()
    assert getattr(chat_class, CREATE_METHOD_NAME) is installed
    second.deactivate()
    assert getattr(chat_class, CREATE_METHOD_NAME) is original

    first.activate()
    installed = getattr(chat_class, CREATE_METHOD_NAME)

    def foreign(self: Any, *args: Any, **kwargs: Any) -> Any:
        return installed(self, *args, **kwargs)

    setattr(chat_class, CREATE_METHOD_NAME, foreign)
    first.deactivate()
    assert getattr(chat_class, CREATE_METHOD_NAME) is foreign
    setattr(chat_class, CREATE_METHOD_NAME, original)


def test_provider_error_keeps_precise_status_without_calling_hostile_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_class = _current_chat_class()
    emitted: list[Any] = []
    monkeypatch.setattr(
        "respan_instrumentation_together._otel_emitter.inject_span",
        lambda span: emitted.append(span),
    )

    class ProviderError(Exception):
        status_code = 429

        def __str__(self) -> str:
            raise AssertionError("instrumentation called hostile __str__")

    error = ProviderError('{"api_key":"plain-secret","message":"limited"}')

    def fail(self: Any, **kwargs: Any) -> Any:
        del self, kwargs
        raise error

    monkeypatch.setattr(chat_class, CREATE_METHOD_NAME, fail)
    instrumentor = TogetherInstrumentor()
    instrumentor.activate()
    with pytest.raises(ProviderError) as caught:
        object.__new__(chat_class).create(
            model="model",
            messages=[{"role": "user", "content": "hello"}],
        )
    instrumentor.deactivate()
    assert caught.value is error
    assert len(emitted) == 1
    span = emitted[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["http.response.status_code"] == 429
    serialized = json.dumps(dict(span.attributes), sort_keys=True)
    assert "plain-secret" not in serialized


def test_bounded_json_redacts_nested_secrets_and_never_uses_repr() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError

        __repr__ = __str__

    serialized = json_dumps(
        {
            "client_secret": "plain-secret",
            "nested": {"authorization": "Basic abcdef", "object": Hostile()},
            "unicode": "😀" * 20_000,
        }
    )
    assert len(serialized.encode("utf-8")) <= 16_000
    assert "plain-secret" not in serialized
    assert "abcdef" not in serialized
    assert json.loads(serialized)
