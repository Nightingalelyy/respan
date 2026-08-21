import asyncio
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_milvus import MilvusInstrumentor
from respan_instrumentation_milvus import (
    _native_instrumentation as native_instrumentation,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class _Span:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exception):
        self.exceptions.append(exception)


class _Tracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **_kwargs):
        span = _Span(name)
        self.spans.append(span)
        yield span


def test_sync_and_async_milvus_operations(monkeypatch):
    pymilvus = ModuleType("pymilvus")

    class MilvusClient:
        def insert(self, collection_name, data):
            return {"insert_count": len(data), "collection": collection_name}

        def search(self, collection_name, data, limit=10):
            return [[{"id": 1, "distance": 0.01}]]

        def close(self):
            return None

    class AsyncMilvusClient:
        async def query(self, collection_name, filter=""):
            return [{"id": 1, "collection": collection_name, "filter": filter}]

        async def close(self):
            return None

    pymilvus.MilvusClient = MilvusClient
    pymilvus.AsyncMilvusClient = AsyncMilvusClient
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)

    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    MilvusInstrumentor._patches_applied = False
    instrumentor = MilvusInstrumentor()
    instrumentor.activate()

    assert MilvusClient().insert("docs", [{"id": 1}])["insert_count"] == 1
    assert MilvusClient().search("docs", [[0.1, 0.2]])[0]
    assert asyncio.run(AsyncMilvusClient().query("docs", "id == 1"))[0]["id"] == 1
    assert [span.name for span in tracer.spans] == [
        "milvus.client.insert",
        "milvus.client.search",
        "milvus.client.query",
    ]
    for span in tracer.spans:
        assert span.attributes[RESPAN_LOG_TYPE] == "task"
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
    assert not hasattr(MilvusClient.close, "__wrapped__")

    instrumentor.deactivate()


def test_identity_serialization_removes_endpoint_credentials():
    identity = SimpleNamespace(
        uri=(
            "https://uri-user:uri-password@milvus.example:19530/default"
            "?api_key=uri-secret#private"
        ),
        _config=SimpleNamespace(
            host=(
                "host-user:host-password@db.milvus.example:19530"
                "?token=host-secret#private"
            )
        ),
    )

    output = native_instrumentation._json_dumps(identity)

    assert json.loads(output) == {
        "host": "db.milvus.example:19530",
        "uri": "https://milvus.example:19530/default",
    }
    for secret in (
        "host-password",
        "host-secret",
        "host-user",
        "uri-password",
        "uri-secret",
        "uri-user",
    ):
        assert secret not in output


def test_real_client_exports_stable_success_and_error_spans(monkeypatch):
    from pymilvus import AsyncMilvusClient, DataType, MilvusClient
    from pymilvus.exceptions import DescribeCollectionException

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        native_instrumentation.trace,
        "get_tracer",
        lambda _: tracer_provider.get_tracer("respan-milvus-test"),
    )

    MilvusInstrumentor._patches_applied = False
    instrumentor = MilvusInstrumentor()
    instrumentor.activate()
    credential = "milvus-test-password"

    async def exercise_async_client(uri: str) -> None:
        client = AsyncMilvusClient(uri=uri)
        try:
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
            await client.create_collection(
                collection_name="async_docs",
                schema=schema,
            )
            inserted = await client.insert(
                collection_name="async_docs",
                data=[
                    {
                        "id": 2,
                        "vector": [0.0, 1.0],
                        "text": "async document",
                    }
                ],
            )
            assert inserted["insert_count"] == 1
            assert await client.query(
                collection_name="async_docs",
                filter="id == 2",
                output_fields=["id", "text"],
            ) == [{"id": 2, "text": "async document"}]

            with pytest.raises(DescribeCollectionException):
                await client.describe_collection(collection_name="missing_async_docs")

            await client.drop_collection(collection_name="async_docs")
        finally:
            await client.close()

    try:
        with TemporaryDirectory(prefix="respan-milvus-test-") as directory:
            client = MilvusClient(uri=str(Path(directory) / "milvus.db"))
            tracer = tracer_provider.get_tracer("respan-milvus-real-client-test")
            with tracer.start_as_current_span("milvus.real.sync") as root_span:
                sync_root_id = root_span.get_span_context().span_id
                try:
                    schema = client.create_schema(
                        auto_id=False,
                        enable_dynamic_field=True,
                    )
                    schema.add_field("id", DataType.INT64, is_primary=True)
                    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
                    client.create_collection(collection_name="docs", schema=schema)
                    inserted = client.insert(
                        collection_name="docs",
                        data=[
                            {
                                "id": 1,
                                "vector": [1.0, 0.0],
                                "text": "safe document",
                                "password": credential,
                            }
                        ],
                    )
                    assert inserted["insert_count"] == 1
                    assert client.query(
                        collection_name="docs",
                        filter="id == 1",
                        output_fields=["id", "text"],
                    ) == [{"id": 1, "text": "safe document"}]

                    with pytest.raises(DescribeCollectionException):
                        client.describe_collection(collection_name="missing_docs")

                    client.drop_collection(collection_name="docs")
                finally:
                    client.close()

        with TemporaryDirectory(prefix="respan-milvus-async-test-") as directory:
            tracer = tracer_provider.get_tracer("respan-milvus-real-client-test")
            with tracer.start_as_current_span("milvus.real.async") as root_span:
                async_root_id = root_span.get_span_context().span_id
                asyncio.run(exercise_async_client(str(Path(directory) / "milvus.db")))

        finished_spans = exporter.get_finished_spans()
        expected_names = Counter(
            {
                "milvus.client.create_collection": 2,
                "milvus.client.describe_collection": 2,
                "milvus.client.drop_collection": 2,
                "milvus.client.insert": 2,
                "milvus.client.query": 2,
                "milvus.real.async": 1,
                "milvus.real.sync": 1,
            }
        )
        assert len(finished_spans) == sum(expected_names.values())
        assert Counter(span.name for span in finished_spans) == expected_names
        assert len({span.context.span_id for span in finished_spans}) == len(
            finished_spans
        )
        instrumented_spans = [
            span for span in finished_spans if span.name.startswith("milvus.client.")
        ]
        assert all(
            span.parent is not None
            and span.parent.span_id in {sync_root_id, async_root_id}
            for span in instrumented_spans
        )
        spans_by_parent = {
            parent_id: {
                span.name: span
                for span in instrumented_spans
                if span.parent is not None and span.parent.span_id == parent_id
            }
            for parent_id in (sync_root_id, async_root_id)
        }
        expected_client_names = {
            "milvus.client.create_collection",
            "milvus.client.describe_collection",
            "milvus.client.drop_collection",
            "milvus.client.insert",
            "milvus.client.query",
        }
        assert set(spans_by_parent[sync_root_id]) == expected_client_names
        assert set(spans_by_parent[async_root_id]) == expected_client_names
        spans = spans_by_parent[sync_root_id]
        async_spans = spans_by_parent[async_root_id]

        create_input = json.loads(
            spans["milvus.client.create_collection"].attributes[
                SpanAttributes.TRACELOOP_ENTITY_INPUT
            ]
        )
        assert create_input["schema"]["fields"][1]["params"]["dim"] == 2

        insert_span = spans["milvus.client.insert"]
        assert insert_span.status.status_code == StatusCode.OK
        insert_input = insert_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
        insert_output = insert_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        assert credential not in insert_input
        assert json.loads(insert_input)["data"][0]["password"] == "<redacted>"
        assert json.loads(insert_output)["ids"] == [1]
        assert "0x" not in insert_output

        query_span = spans["milvus.client.query"]
        assert query_span.status.status_code == StatusCode.OK
        assert json.loads(
            query_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        ) == [{"id": 1, "text": "safe document"}]

        failed_span = spans["milvus.client.describe_collection"]
        assert failed_span.status.status_code == StatusCode.ERROR
        failed_output = json.loads(
            failed_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )
        assert failed_output["error"] == "DescribeCollectionException"
        assert "missing_docs" in failed_output["message"]

        async_insert_span = async_spans["milvus.client.insert"]
        assert async_insert_span.status.status_code == StatusCode.OK
        assert json.loads(
            async_insert_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )["ids"] == [2]

        async_query_span = async_spans["milvus.client.query"]
        assert async_query_span.status.status_code == StatusCode.OK
        assert json.loads(
            async_query_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        ) == [{"id": 2, "text": "async document"}]

        async_failed_span = async_spans["milvus.client.describe_collection"]
        assert async_failed_span.status.status_code == StatusCode.ERROR
        async_failed_output = json.loads(
            async_failed_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )
        assert async_failed_output["error"] == "DescribeCollectionException"
        assert "missing_async_docs" in async_failed_output["message"]

        forbidden_aliases = {
            "has_tool_calls",
            "model",
            "span_tools",
            "tool_calls",
            "tools",
        }
        for span in instrumented_spans:
            assert span.attributes[RESPAN_LOG_TYPE] == "task"
            assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
            assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
            assert SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes
            assert forbidden_aliases.isdisjoint(span.attributes)
    finally:
        instrumentor.deactivate()
        tracer_provider.shutdown()
