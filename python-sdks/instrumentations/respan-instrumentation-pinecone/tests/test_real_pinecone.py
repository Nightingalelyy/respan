from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from pinecone import Index
from pinecone.async_client.async_index import AsyncIndex
from pinecone.exceptions import PineconeApiException
from respan_instrumentation_pinecone import PineconeInstrumentor
from respan_instrumentation_pinecone import (
    _native_instrumentation as native_instrumentation,
)
from respan_instrumentation_pinecone._serialization import json_dumps
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class _Handler(BaseHTTPRequestHandler):
    server_version = "PineconeContractFixture/1.0"

    def log_message(self, *_args) -> None:
        return

    def _json(self, status: int, value: object) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path.startswith("/vectors/fetch"):
            self._json(
                200,
                {
                    "namespace": "demo",
                    "vectors": {
                        "doc-1": {
                            "id": "doc-1",
                            "values": [0.1, 0.2],
                            "metadata": {"topic": "tracing"},
                        }
                    },
                },
            )
            return
        self._json(404, {"message": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        if self.path == "/describe_index_stats":
            self._json(
                200,
                {
                    "dimension": 2,
                    "indexFullness": 0,
                    "namespaces": {},
                    "totalVectorCount": 1,
                },
            )
        elif self.path == "/vectors/upsert":
            self._json(200, {"upsertedCount": 1})
        elif self.path == "/query":
            self._json(
                200,
                {
                    "namespace": "demo",
                    "matches": [
                        {
                            "id": "doc-1",
                            "score": 0.99,
                            "values": [0.1, 0.2],
                            "metadata": {"topic": "tracing"},
                        }
                    ],
                },
            )
        elif self.path == "/vectors/delete":
            self._json(503, {"message": "deterministic Pinecone outage"})
        else:
            self._json(404, {"message": "not found"})


@contextmanager
def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def exporter(monkeypatch):
    provider = TracerProvider()
    span_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", provider.get_tracer)
    yield provider, span_exporter
    provider.shutdown()


def test_real_current_sdk_exports_sync_async_and_failure(exporter):
    provider, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    try:
        with _server() as host:
            index = Index(
                host=host, api_key="pinecone-contract-secret", ssl_verify=False
            )

            async def async_fetch() -> object:
                async_index = AsyncIndex(
                    host=host,
                    api_key="pinecone-contract-secret",
                    ssl_verify=False,
                )
                try:
                    return await async_index.fetch(ids=["doc-1"], namespace="demo")
                finally:
                    await async_index.close()

            tracer = provider.get_tracer("pinecone.contract")
            with tracer.start_as_current_span("root") as root:
                index.describe_index_stats()
                index.upsert(
                    vectors=[{"id": "doc-1", "values": [0.1, 0.2]}],
                    namespace="demo",
                )
                index.query(
                    vector=[0.1, 0.2],
                    top_k=1,
                    namespace="demo",
                    include_metadata=True,
                    include_values=True,
                )
                asyncio.run(async_fetch())
                with pytest.raises(PineconeApiException):
                    index.delete(ids=["doc-1"], namespace="demo")

            spans = span_exporter.get_finished_spans()
            names = Counter(span.name for span in spans)
            assert names == Counter(
                {
                    "root": 1,
                    "pinecone.index.describe_index_stats": 1,
                    "pinecone.index.upsert": 1,
                    "pinecone.index.query": 1,
                    "pinecone.index.fetch": 1,
                    "pinecone.index.delete": 1,
                }
            )
            assert len({span.context.span_id for span in spans}) == len(spans)
            client_spans = [span for span in spans if span.name != "root"]
            assert all(
                span.parent.span_id == root.context.span_id for span in client_spans
            )
            assert all(
                span.attributes[RESPAN_LOG_TYPE] == "task" for span in client_spans
            )
            assert all(
                SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes
                for span in client_spans
            )
            assert all(
                span.attributes[OTelSpanAttributes.DB_SYSTEM] == "pinecone"
                for span in client_spans
            )
            for span in client_spans:
                assert json.loads(
                    span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
                )
                if SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes:
                    json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])

            failed = next(span for span in client_spans if span.name.endswith("delete"))
            assert failed.status.status_code is StatusCode.ERROR
            assert failed.attributes[HTTP_RESPONSE_STATUS_CODE] == 503
            assert (
                "deterministic Pinecone outage" in failed.attributes[ERROR_MESSAGE_ATTR]
            )
            assert failed.events[0].name == "exception"
            exported = json.dumps([dict(span.attributes) for span in spans])
            assert "pinecone-contract-secret" not in exported
            assert "0x" not in exported
    finally:
        instrumentor.deactivate()


def test_serializer_is_bounded_redacted_and_never_calls_repr_or_str():
    calls = {"repr": 0, "str": 0}

    class Hostile:
        def __repr__(self):
            calls["repr"] += 1
            raise AssertionError("repr must not be called")

        def __str__(self):
            calls["str"] += 1
            raise AssertionError("str must not be called")

    encoded = json_dumps(
        {
            "api_key": "plain-secret",
            "client_secret": "another-secret",
            "nested": {"auth_token": "token-value", "value": Hostile()},
            "unicode": "😀" * 10_000,
        }
    )
    assert len(encoded.encode("utf-8")) <= 16_000
    parsed = json.loads(encoded)
    assert parsed
    assert "plain-secret" not in encoded
    assert "another-secret" not in encoded
    assert "token-value" not in encoded
    assert calls == {"repr": 0, "str": 0}


def test_multiple_instances_share_lifecycle_and_reject_config_mismatch(monkeypatch):
    class IndexFixture:
        def query(self, vector, top_k):
            return {"matches": [], "vector": vector, "top_k": top_k}

    module = SimpleNamespace(Index=IndexFixture)
    monkeypatch.setattr(
        native_instrumentation.importlib,
        "import_module",
        lambda name: (
            module
            if name == "pinecone.index"
            else (_ for _ in ()).throw(ImportError(name))
        ),
    )
    first = PineconeInstrumentor()
    second = PineconeInstrumentor()
    first.activate()
    second.activate()
    try:
        assert PineconeInstrumentor._activation_count == 2
        first.deactivate()
        assert PineconeInstrumentor._activation_count == 1
        assert PineconeInstrumentor._patches_applied is True
        with pytest.raises(ValueError):
            PineconeInstrumentor(capture_content=False).activate()
    finally:
        second.deactivate()
    assert PineconeInstrumentor._activation_count == 0
    assert PineconeInstrumentor._patches_applied is False
