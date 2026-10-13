import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanLimits, SpanProcessor, TracerProvider


@pytest.fixture
def otel_spans(monkeypatch):
    spans = []

    class Collector(SpanProcessor):
        def on_end(self, span):
            spans.append(span)

    provider = TracerProvider(span_limits=SpanLimits(max_attributes=4096))
    provider.add_span_processor(Collector())
    tracer = provider.get_tracer("helicone-contract-tests")
    monkeypatch.setattr(trace, "get_tracer", lambda *_args, **_kwargs: tracer)
    yield spans
    provider.shutdown()
