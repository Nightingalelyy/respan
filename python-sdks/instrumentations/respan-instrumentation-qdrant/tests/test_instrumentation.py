import json
import sys
from collections import Counter
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_qdrant import QdrantInstrumentor, _instrumentation
from respan_instrumentation_qdrant._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
)
from respan_instrumentation_qdrant._serialization import json_dumps
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


class _FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.status = None
        self.exceptions = []
        self.events = []
        self.parent = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))


class _FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **_kwargs):
        span = _FakeSpan(name)
        self.spans.append(span)
        yield span


def _install_fake_qdrant(monkeypatch):
    module = ModuleType("qdrant_client")

    class QdrantClient:
        def create_collection(self, collection_name, vectors_config=None):
            return {"collection_name": collection_name, "created": True}

        def upsert(self, collection_name, points, api_key=None):
            return {"collection_name": collection_name, "points": points}

        def query_points(self, collection_name, query, limit=10):
            if collection_name == "missing":
                raise RuntimeError("collection does not exist")
            return {"points": [{"id": 1, "score": 0.99}], "query": query}

    class AsyncQdrantClient:
        async def upsert(self, collection_name, points):
            return {"collection_name": collection_name, "count": len(points)}

        async def query_points(self, collection_name, query, limit=10):
            return {"points": [{"id": 2, "score": 0.9}], "query": query}

    module.QdrantClient = QdrantClient
    module.AsyncQdrantClient = AsyncQdrantClient
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    return QdrantClient, AsyncQdrantClient


@pytest.fixture(autouse=True)
def reset_instrumentor():
    QdrantInstrumentor._patches_applied = False
    QdrantInstrumentor._activation_count = 0
    QdrantInstrumentor._patched_targets = []
    QdrantInstrumentor._capture_content_config = None
    yield
    for patch in reversed(QdrantInstrumentor._patched_targets):
        try:
            _instrumentation.unwrap(patch.target, patch.attribute)
        except Exception:  # noqa: BLE001,S110 - test cleanup must be best effort.
            pass
    QdrantInstrumentor._patches_applied = False
    QdrantInstrumentor._activation_count = 0
    QdrantInstrumentor._patched_targets = []
    QdrantInstrumentor._capture_content_config = None


def _assert_contract(span):
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert SpanAttributes.TRACELOOP_ENTITY_PATH in attrs
    for banned_alias in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    ):
        assert banned_alias not in attrs


def test_package_exports_qdrant_instrumentor():
    assert QdrantInstrumentor is _instrumentation.QdrantInstrumentor
    assert QdrantInstrumentor.name == "qdrant"


def test_sync_operations_emit_canonical_task_spans(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.instrument()

    client = QdrantClient()
    client.create_collection("docs", vectors_config={"size": 3})
    client.upsert(
        "docs",
        [{"id": 1, "vector": [0.1, 0.2, 0.3]}],
        api_key="do-not-export",
    )
    result = client.query_points("docs", [0.1, 0.2, 0.3], limit=1)

    assert result["points"][0]["id"] == 1
    assert [span.name for span in tracer.spans] == [
        "qdrant.create_collection",
        "qdrant.upsert",
        "qdrant.query_points",
    ]
    for span in tracer.spans:
        _assert_contract(span)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
        assert span.status.status_code is StatusCode.OK
    assert (
        "[REDACTED]"
        in tracer.spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_async_operations_are_awaited_and_traced(monkeypatch):
    _, AsyncQdrantClient = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()

    result = await AsyncQdrantClient().upsert(
        "docs",
        [{"id": 2, "vector": [0.3, 0.2, 0.1]}],
    )

    assert result["count"] == 1
    assert [span.name for span in tracer.spans] == ["qdrant.upsert"]
    _assert_contract(tracer.spans[0])
    instrumentor.deactivate()


def test_errors_are_recorded_and_reraised(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="collection does not exist"):
        QdrantClient().query_points("missing", [0.0, 0.0, 0.0])

    span = tracer.spans[-1]
    assert span.status.status_code is StatusCode.ERROR
    assert not span.exceptions
    assert span.events[0][0] == "exception"
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "collection does not exist"
    assert (
        "collection does not exist"
        in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    instrumentor.deactivate()


def test_capture_content_false_omits_inputs_and_outputs(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = QdrantInstrumentor(capture_content=False)
    instrumentor.activate()

    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])

    attrs = tracer.spans[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    _assert_contract(tracer.spans[0])
    instrumentor.deactivate()


def test_large_vectors_are_bounded_with_stable_previews(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()
    vector = [0.125] * (MAX_ATTRIBUTE_CHARS + 17)
    tags = [f"tag-{index}" for index in range(MAX_PREVIEW_ITEMS + 17)]

    QdrantClient().upsert(
        "docs",
        [{"id": 1, "vector": vector, "payload": {"tags": tags}}],
    )

    span = tracer.spans[-1]
    entity_input = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    entity_output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert entity_input["points"][0]["vector"]["count"] == len(vector)
    assert len(entity_input["points"][0]["vector"]["items"]) == MAX_PREVIEW_ITEMS
    assert entity_output["points"][0]["vector"]["count"] == len(vector)
    assert (
        len(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT].encode("utf-8"))
        <= MAX_ATTRIBUTE_CHARS
    )
    assert (
        len(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT].encode("utf-8"))
        <= MAX_ATTRIBUTE_CHARS
    )
    assert entity_input["points"][0]["payload"]["tags"]["truncated"] is True
    instrumentor.deactivate()


def test_lifecycle_is_idempotent_and_reference_counted(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    first = QdrantInstrumentor()
    second = QdrantInstrumentor()

    first.activate()
    first.activate()
    second.activate()
    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])
    assert len(tracer.spans) == 1

    first.deactivate()
    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])
    assert len(tracer.spans) == 2

    second.deactivate()
    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])
    assert len(tracer.spans) == 2


def test_active_instances_reject_capture_mismatch(monkeypatch):
    _install_fake_qdrant(monkeypatch)
    first = QdrantInstrumentor(capture_content=True)
    second = QdrantInstrumentor(capture_content=False)
    first.activate()
    with pytest.raises(ValueError, match="same capture_content"):
        second.activate()
    first.deactivate()


def test_serializer_is_bounded_redacting_and_hostile_safe():
    class Hostile:
        def __str__(self):
            raise AssertionError("hostile __str__ called")

        def __repr__(self):
            raise AssertionError("hostile __repr__ called")

    encoded = json_dumps(
        {
            "api_key": "plain-secret",
            "nested": {"client_secret": "nested-secret"},
            "endpoint": "https://user:pass@example.test/v1?api_key=query-secret",
            "hostile": Hostile(),
            "unicode": "😀" * 10_000,
        }
    )
    assert len(encoded.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS
    for secret in ("plain-secret", "nested-secret", "query-secret", "user:pass"):
        assert secret not in encoded
    assert json.loads(encoded)


@pytest.mark.asyncio
async def test_real_current_clients_export_sync_async_success_and_failure(monkeypatch):
    from qdrant_client import AsyncQdrantClient, QdrantClient, models

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda name, version=None: provider.get_tracer(name, version),
    )
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()
    root_tracer = provider.get_tracer("qdrant-real-contract")

    with root_tracer.start_as_current_span("sync.root") as sync_root:
        client = QdrantClient(":memory:")
        client.create_collection(
            "docs",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )
        client.upsert(
            "docs",
            points=[
                models.PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3],
                    payload={"topic": "tracing"},
                )
            ],
        )
        query = client.query_points("docs", query=[0.1, 0.2, 0.3], limit=1)
        assert query.points[0].id == 1
        assert client.retrieve("docs", ids=[1])[0].payload == {"topic": "tracing"}
        assert client.scroll("docs", limit=1)[0][0].id == 1
        with pytest.raises(ValueError, match="missing_collection"):
            client.get_collection("missing_collection")
        client.close()

    with root_tracer.start_as_current_span("async.root") as async_root:
        async_client = AsyncQdrantClient(":memory:")
        await async_client.create_collection(
            "async_docs",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.DOT,
            ),
        )
        await async_client.upsert(
            "async_docs",
            points=[models.PointStruct(id=2, vector=[0.3, 0.2, 0.1])],
        )
        async_query = await async_client.query_points(
            "async_docs",
            query=[0.3, 0.2, 0.1],
            limit=1,
        )
        assert async_query.points[0].id == 2
        await async_client.close()

    assert provider.force_flush()
    spans = list(exporter.get_finished_spans())
    qdrant_spans = [
        span for span in spans if span.instrumentation_scope.name == "qdrant"
    ]
    names = Counter(span.name for span in qdrant_spans)
    assert names == Counter(
        {
            "qdrant.create_collection": 2,
            "qdrant.upsert": 2,
            "qdrant.query_points": 2,
            "qdrant.retrieve": 1,
            "qdrant.scroll": 1,
            "qdrant.get_collection": 1,
        }
    )
    assert len({span.context.span_id for span in qdrant_spans}) == 9
    sync_parent = sync_root.get_span_context().span_id
    async_parent = async_root.get_span_context().span_id
    assert all(
        span.parent.span_id in {sync_parent, async_parent} for span in qdrant_spans
    )
    failed = next(span for span in qdrant_spans if span.name == "qdrant.get_collection")
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.attributes["status_code"] == 500
    assert failed.events[0].attributes["exception.type"] == "ValueError"
    for span in qdrant_spans:
        _assert_contract(span)
        assert span.instrumentation_scope.version == "0.1.0"
        assert span.attributes["db.system"] == "qdrant"
        assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        assert SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes

    instrumentor.deactivate()
    provider.shutdown()
