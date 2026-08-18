from __future__ import annotations

import asyncio
import json
from collections import Counter

import ragas
import respan_instrumentation_ragas._instrumentation as instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from ragas import EvaluationDataset
from ragas.backends.inmemory import InMemoryBackend
from ragas.dataset import Dataset
from ragas.metrics import ExactMatch
from respan_instrumentation_ragas import RagasInstrumentor
from respan_instrumentation_ragas._serialization import json_string
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


def test_real_current_evaluate_and_experiment_export_connected_spans(
    monkeypatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("ragas", "0.1.0"),
    )
    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(instrumentation, "_PATCHES", [])
    monkeypatch.setattr(instrumentation, "_ENABLED", False)

    adapter = RagasInstrumentor()
    adapter.activate()
    evaluation_dataset = EvaluationDataset.from_list(
        [{"user_input": "Capital?", "response": "Paris", "reference": "Paris"}]
    )
    result = ragas.evaluate(
        evaluation_dataset,
        metrics=[ExactMatch()],
        show_progress=False,
    )
    assert result["exact_match"] == [1.0]

    backend = InMemoryBackend()

    @ragas.experiment(backend=backend, name_prefix="offline")
    async def answer(row):
        return {"answer": row["answer"]}

    dataset = Dataset(
        name="questions",
        backend=backend,
        data=[{"answer": "Paris"}, {"answer": "Rome"}],
    )
    experiment = asyncio.run(answer.arun(dataset, name="two-rows"))
    assert len(experiment) == 2
    adapter.deactivate()

    spans = exporter.get_finished_spans()
    names = Counter(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] for span in spans
    )
    assert names == Counter(
        {
            "exact_match": 1,
            "ragas.evaluate": 1,
            "ragas.experiment.two-rows": 1,
            "ragas.experiment.row.answer": 2,
        }
    )
    assert len({span.context.span_id for span in spans}) == 5
    assert all(span.attributes[RESPAN_LOG_TYPE] == "task" for span in spans)
    assert all("traceloop.span.kind" not in span.attributes for span in spans)
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == 2
    assert all(
        root.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "" for root in roots
    )
    parents = {span.context.span_id for span in spans}
    assert all(span.parent is None or span.parent.span_id in parents for span in spans)
    assert all(span.instrumentation_scope.name == "ragas" for span in spans)


def test_ragas_serialization_is_valid_bounded_and_private() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError("must not stringify")

        def __repr__(self) -> str:
            raise AssertionError("must not repr")

    encoded = json_string(
        {
            "api_token": "plain-secret",
            "content": "😀" * 5_000,
            "hostile": Hostile(),
            "nonfinite": float("inf"),
        }
    )
    assert len(encoded.encode("utf-8")) <= 16_000
    assert "plain-secret" not in encoded
    assert json.loads(encoded)["nonfinite"] is None
