"""Policy/lifetime checks using the released native Helicone helpers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from helicone_helpers import HeliconeManualLogger, manual_logger
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_helicone import (
    HeliconeInstrumentor,
    _emitter,
    _instrumentation,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer


class CountingSampler(Sampler):
    def __init__(self, decision=Decision.RECORD_AND_SAMPLE):
        self.decision = decision
        self.calls = []

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ):
        self.calls.append((parent_context, name, attributes))
        return SamplingResult(self.decision)

    def get_description(self):
        return "Helicone native policy test sampler"


@pytest.fixture
def native(monkeypatch):
    sampler = CountingSampler()
    spans = []
    posts = []

    class Collector(SpanProcessor):
        def on_end(self, span):
            spans.append(span)

    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(Collector())
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(RespanTracer, "_instance", None)
    for key in ("TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        manual_logger.requests,
        "post",
        lambda _url, **kwargs: (
            posts.append(kwargs) or SimpleNamespace(status_code=200, text="ok")
        ),
    )
    plugin = HeliconeInstrumentor()
    plugin.activate()
    logger = HeliconeManualLogger(api_key="synthetic-native-fixture")
    yield SimpleNamespace(
        plugin=plugin,
        logger=logger,
        provider=provider,
        sampler=sampler,
        spans=spans,
        posts=posts,
    )
    plugin.deactivate()
    _instrumentation._restore_all()
    _instrumentation._REFCOUNT = 0
    _instrumentation._CONFIG = None
    _instrumentation._GENERATION = None
    provider.shutdown()


REQUEST = {
    "model": "policy-model",
    "messages": [{"role": "user", "content": "private-native-fixture"}],
}
RESPONSE = {
    "model": "policy-model",
    "choices": [
        {"message": {"role": "assistant", "content": "private-native-fixture"}}
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}


def invoke(native, operation=None):
    sentinel = object()

    def default(recorder):
        recorder.append_results(RESPONSE)
        return sentinel

    result = native.logger.log_request(REQUEST, operation or default, provider="openai")
    if operation is None:
        assert result is sentinel
    assert len(native.posts) == 1
    return result


def assert_private(spans):
    assert spans
    for span in spans:
        attrs = span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
        assert "private-native-fixture" not in json.dumps(dict(attrs))


@pytest.mark.parametrize("decision", [Decision.DROP, Decision.RECORD_ONLY])
def test_native_sampler_precedes_all_payload_normalization(
    native, monkeypatch, decision
):
    native.sampler.decision = decision
    monkeypatch.setattr(
        _emitter,
        "_normalized_response",
        lambda *_args: pytest.fail("unsampled payload was normalized"),
    )
    invoke(native)
    assert len(native.sampler.calls) == 1
    assert len(native.spans) == (1 if decision == Decision.RECORD_ONLY else 0)
    if native.spans:
        assert not native.spans[0].context.trace_flags.sampled
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in native.spans[0].attributes


@pytest.mark.parametrize("key", ["TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT"])
def test_native_environment_ceiling_cannot_be_overridden(native, monkeypatch, key):
    monkeypatch.setenv(key, "false")
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
    try:
        invoke(native)
    finally:
        context.detach(token)
    assert_private(native.spans)
    assert native.spans[0].attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0


def test_native_private_constructor_cannot_be_overridden(native):
    native.plugin.deactivate()
    native.plugin = HeliconeInstrumentor(capture_content=False)
    native.plugin.activate()
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
    try:
        invoke(native)
    finally:
        context.detach(token)
    assert_private(native.spans)


def test_native_suppression_and_runtime_disabled_keep_native_behavior(native):
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        invoke(native)
    finally:
        context.detach(token)
    assert not native.spans and not native.sampler.calls
    native.posts.clear()
    RespanTracer._instance = SimpleNamespace(is_enabled=False)
    invoke(native)
    assert not native.spans and not native.sampler.calls


def test_native_late_recorder_privacy_veto_survives_context_detachment(native):
    def operation(recorder):
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            recorder.append_results(RESPONSE)
        finally:
            context.detach(token)
        return "native result"

    assert invoke(native, operation) == "native result"
    assert_private(native.spans)


@pytest.mark.parametrize("phase", ["creation", "chunk", "send"])
def test_native_deferred_builder_privacy_is_irreversible(native, phase):
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if phase == "creation"
        else None
    )
    try:
        builder = native.logger.new_builder(REQUEST)
    finally:
        if token is not None:
            context.detach(token)
    if phase == "chunk":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        token = None
    try:
        assert (
            builder.add_chunk(
                {"choices": [{"delta": {"content": "private-native-fixture"}}]}
            )
            is None
        )
    finally:
        if token is not None:
            context.detach(token)
    assert not native.spans
    if phase == "send":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        token = None
    try:
        assert asyncio.run(builder.send_log()) is None
    finally:
        if token is not None:
            context.detach(token)
    assert len(native.posts) == 1
    assert_private(native.spans)


@pytest.mark.parametrize("phase", ["creation", "send"])
def test_native_deferred_builder_suppression_is_irreversible(native, phase):
    token = (
        context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        if phase == "creation"
        else None
    )
    try:
        builder = native.logger.log_builder(REQUEST)
    finally:
        if token is not None:
            context.detach(token)
    builder.add_response(RESPONSE)
    token = (
        context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        if phase == "send"
        else None
    )
    try:
        asyncio.run(builder.send_log())
    finally:
        if token is not None:
            context.detach(token)
    assert len(native.posts) == 1
    assert not native.spans and not native.sampler.calls


def test_native_private_ancestor_cannot_be_reenabled_by_child(native):
    tracer = native.provider.get_tracer("native-parent")
    with tracer.start_as_current_span("parent") as parent:
        parent.set_attribute("allow_trace_content", False)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
        try:
            invoke(native)
        finally:
            context.detach(token)
    assert_private(native.spans[:-1])
    assert native.spans[0].parent.span_id == parent.get_span_context().span_id


def test_native_deferred_parent_sampling_context_is_preserved(native):
    native.sampler.decision = Decision.DROP
    tracer = native.provider.get_tracer("native-parent")
    with tracer.start_as_current_span("parent"):
        builder = native.logger.log_builder(REQUEST)
    native.sampler.decision = Decision.RECORD_AND_SAMPLE
    native.provider.sampler = ParentBased(native.sampler)
    builder.add_response(RESPONSE)
    asyncio.run(builder.send_log())
    assert len(native.posts) == 1
    assert not native.spans


def test_native_deactivation_inside_admitted_operation_drains_once(native):
    def operation(recorder):
        recorder.append_results(RESPONSE)
        native.plugin.deactivate()
        return "native result"

    assert invoke(native, operation) == "native result"
    assert len(native.spans) == 1
    assert_private(native.spans)
    assert native.spans[0].attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0


def test_native_old_generation_builder_does_not_attach_to_reactivated_owner(native):
    builder = native.logger.log_builder(REQUEST)
    assert not native.spans and not native.sampler.calls
    native.plugin.deactivate()
    native.plugin = HeliconeInstrumentor()
    native.plugin.activate()
    builder.add_response(RESPONSE)
    asyncio.run(builder.send_log())
    assert len(native.posts) == 1
    assert not native.spans and not native.sampler.calls


def test_native_generator_result_identity_and_error_identity_are_unchanged(native):
    def values():
        yield "native lazy result"

    generator = values()
    assert invoke(native, lambda _recorder: generator) is generator
    assert next(generator) == "native lazy result"
    native.posts.clear()
    error = ValueError("native source error")

    def failed(_recorder):
        raise error

    with pytest.raises(ValueError) as caught:
        native.logger.log_request(REQUEST, failed, provider="openai")
    assert caught.value is error
    assert native.posts == []
    assert native.spans[-1].attributes[ERROR_MESSAGE] == "native source error"
    assert native.spans[-1].attributes[ERROR_TYPE] == "ValueError"


def test_native_default_attribute_budget_preserves_complete_canonical_bodies(native):
    request = {
        "model": "policy-model",
        "messages": [{"role": "user", "content": str(i)} for i in range(180)],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": str(i),
                    "parameters": {
                        "type": "object",
                        "properties": {str(j): {"type": "string"} for j in range(160)},
                    },
                },
            }
            for i in range(75)
        ],
    }
    response = {
        "model": "policy-model",
        "choices": [{"message": {"role": "assistant", "content": "x" * 40000}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    def operation(recorder):
        recorder.append_results(response)

    native.logger.log_request(request, operation, provider="openai")
    attrs = native.spans[0].attributes
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == request
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == response
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == request["tools"]
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 0
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "policy-model"
    assert len(attrs) <= 128


@pytest.mark.parametrize(
    "phase", ["builder_send", "builder_mutation", "recorder_append"]
)
def test_native_late_current_private_span_veto_is_irreversible(native, phase):
    tracer = native.provider.get_tracer("private-current-parent")
    builder = native.logger.log_builder(REQUEST)
    if phase == "recorder_append":

        def operation(recorder):
            with tracer.start_as_current_span("private-source") as parent:
                parent.set_attribute("allow_trace_content", False)
                recorder.append_results(RESPONSE)
            return "native result"

        invoke(native, operation)
    else:
        with tracer.start_as_current_span("private-source") as parent:
            parent.set_attribute("allow_trace_content", False)
            builder.add_response(RESPONSE)
            if phase == "builder_send":
                asyncio.run(builder.send_log())
        if phase == "builder_mutation":
            asyncio.run(builder.send_log())
    assert len(native.posts) == 1
    assert_private([span for span in native.spans if span.name == "llm.policy-model"])


def test_native_direct_manual_interval_is_retained(native):
    assert (
        native.logger.send_log(
            "openai",
            REQUEST,
            RESPONSE,
            {"start_time": 123.25, "end_time": 125.5},
        )
        is None
    )
    assert len(native.posts) == 1
    span = native.spans[0]
    assert span.start_time == 123_250_000_000
    assert span.end_time == 125_500_000_000


def test_native_deferred_builder_interval_is_retained(native):
    builder = native.logger.log_builder(REQUEST)
    builder.add_response(RESPONSE)
    expected_start = int(builder.start_time / 1000 * 1_000_000_000)
    expected_end = int(builder.end_time / 1000 * 1_000_000_000)
    assert not native.spans
    asyncio.run(builder.send_log())
    span = native.spans[0]
    assert abs(span.start_time - expected_start) <= 2000
    assert span.end_time == expected_end
    assert span.end_time >= span.start_time
