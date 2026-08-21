import json
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_marqo import MarqoInstrumentor
from respan_instrumentation_marqo import (
    _native_instrumentation as native_instrumentation,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class _MarqoLoopbackHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return

    def _send_json(self, status, payload):
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == "/":
            self._send_json(200, {"version": "3.18.2"})
            return
        if self.path == "/indexes/docs/health":
            self._send_json(
                503,
                {
                    "message": "marqo unavailable",
                    "code": "service_unavailable",
                    "type": "service_unavailable",
                    "link": "",
                },
            )
            return
        self._send_json(404, {"message": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/indexes/docs/search":
            self._send_json(
                200,
                {
                    "hits": [{"_id": "doc-1", "_score": 0.9}],
                    "processingTimeMs": 1,
                },
            )
            return
        self._send_json(404, {"message": "not found"})


@contextmanager
def _marqo_loopback_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MarqoLoopbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


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


def test_search_and_embed_emit_canonical_spans(monkeypatch):
    marqo = ModuleType("marqo")
    marqo.__path__ = []
    client_module = ModuleType("marqo.client")
    index_module = ModuleType("marqo.index")

    class Client:
        def index(self, index_name):
            return Index(index_name)

    class Index:
        def __init__(self, index_name="docs"):
            self.index_name = index_name

        def search(self, q, limit=10):
            return {"hits": [{"_id": "doc-1", "_score": 0.9}], "query": q}

        def embed(self, content):
            return {"embeddings": [[0.1, 0.2]], "content": content}

        def health(self):
            raise RuntimeError("marqo unavailable")

    client_module.Client = Client
    index_module.Index = Index
    marqo.client = client_module
    marqo.index = index_module
    for module in (marqo, client_module, index_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    MarqoInstrumentor._patches_applied = False
    instrumentor = MarqoInstrumentor()
    instrumentor.activate()
    second_instrumentor = MarqoInstrumentor()
    second_instrumentor.activate()

    index = Client().index("docs")
    assert index.search("observability")["hits"]
    assert index.embed(["hello"])["embeddings"]
    assert [span.name for span in tracer.spans] == [
        "marqo.client.index",
        "marqo.index.search",
        "marqo.index.embed",
    ]
    assert tracer.spans[1].attributes[RESPAN_LOG_TYPE] == "task"
    assert tracer.spans[2].attributes[RESPAN_LOG_TYPE] == "embedding"
    assert tracer.spans[2].attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    for span in tracer.spans:
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes

    with pytest.raises(RuntimeError, match="marqo unavailable"):
        index.health()
    failed_span = tracer.spans[-1]
    assert failed_span.name == "marqo.index.health"
    assert failed_span.status.status_code == StatusCode.ERROR
    assert [str(exc) for exc in failed_span.exceptions] == ["marqo unavailable"]

    instrumentor.deactivate()
    traced_count = len(tracer.spans)
    assert index.search("still active")["hits"]
    assert len(tracer.spans) == traced_count + 1

    second_instrumentor.deactivate()
    traced_count = len(tracer.spans)
    assert index.search("deactivated")["hits"]
    assert len(tracer.spans) == traced_count


def test_identity_serialization_removes_endpoint_credentials():
    identity = SimpleNamespace(
        uri=(
            "https://uri-user:uri-password@marqo.example:8443/indexes/docs"
            "?api_key=uri-secret#private"
        ),
        _config=SimpleNamespace(
            host=(
                "host-user:host-password@api.marqo.example:9443"
                "?token=host-secret#private"
            )
        ),
    )

    output = native_instrumentation._json_dumps(identity)

    assert json.loads(output) == {
        "host": "api.marqo.example:9443",
        "uri": "https://marqo.example:8443/indexes/docs",
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
    import marqo
    from marqo.errors import MarqoWebError

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        native_instrumentation.trace,
        "get_tracer",
        lambda _: tracer_provider.get_tracer("respan-marqo-test"),
    )

    MarqoInstrumentor._patches_applied = False
    instrumentor = MarqoInstrumentor()
    instrumentor.activate()
    try:
        with _marqo_loopback_url() as url:
            client = marqo.Client(url=url)
            index = client.index("docs")
            assert index.search("observability")["hits"] == [
                {"_id": "doc-1", "_score": 0.9}
            ]
            with pytest.raises(MarqoWebError, match="marqo unavailable"):
                index.health()

        finished_spans = exporter.get_finished_spans()
        expected_names = Counter(
            {
                "marqo.client.index": 1,
                "marqo.index.health": 1,
                "marqo.index.search": 1,
            }
        )
        assert len(finished_spans) == sum(expected_names.values())
        assert Counter(span.name for span in finished_spans) == expected_names
        spans = {span.name: span for span in finished_spans}

        index_output = spans["marqo.client.index"].attributes[
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        ]
        assert json.loads(index_output) == {"index_name": "docs"}
        assert "0x" not in index_output

        search_span = spans["marqo.index.search"]
        assert search_span.status.status_code == StatusCode.OK
        assert json.loads(
            search_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )["hits"] == [{"_id": "doc-1", "_score": 0.9}]

        failed_span = spans["marqo.index.health"]
        assert failed_span.status.status_code == StatusCode.ERROR
        failed_output = json.loads(
            failed_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )
        assert failed_output["error"] == "MarqoWebError"
        assert "marqo unavailable" in failed_output["message"]

        forbidden_aliases = {
            "has_tool_calls",
            "model",
            "span_tools",
            "tool_calls",
            "tools",
        }
        for span in spans.values():
            assert span.attributes[RESPAN_LOG_TYPE] == "task"
            assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
            assert forbidden_aliases.isdisjoint(span.attributes)
    finally:
        instrumentor.deactivate()
        tracer_provider.shutdown()
