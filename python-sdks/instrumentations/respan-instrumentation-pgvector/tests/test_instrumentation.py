import json
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from respan_instrumentation_pgvector import PGVectorInstrumentor
from respan_instrumentation_pgvector import _instrumentation
from respan_instrumentation_pgvector._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    FunctionPatchSpec,
    MethodPatchSpec,
)
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

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class _FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **_kwargs):
        span = _FakeSpan(name)
        self.spans.append(span)
        yield span


def _install_fake_modules(monkeypatch):
    psycopg = ModuleType("psycopg")
    pgvector_psycopg = ModuleType("pgvector.psycopg")

    class Cursor:
        result_vector = [0.1, 0.2, 0.3]
        rowcount = 2
        statusmessage = "SELECT 2"
        description = None

        def execute(self, query, params=None):
            if "BROKEN" in query:
                raise RuntimeError("invalid vector query")
            return self

        def executemany(self, query, params_seq):
            return self

        def fetchall(self):
            return [(1, self.result_vector), (2, list(reversed(self.result_vector)))]

        def fetchmany(self, size=1):
            return self.fetchall()[:size]

        def fetchone(self):
            return self.fetchall()[0]

    class Connection:
        def execute(self, query, params=None):
            return Cursor().execute(query, params)

    class AsyncCursor:
        rowcount = 1
        statusmessage = "SELECT 1"
        description = None

        async def execute(self, query, params=None):
            return self

        async def executemany(self, query, params_seq):
            return self

        async def fetchall(self):
            return [(3, [0.4, 0.5, 0.6])]

        async def fetchmany(self, size=1):
            return (await self.fetchall())[:size]

        async def fetchone(self):
            return (await self.fetchall())[0]

    class AsyncConnection:
        async def execute(self, query, params=None):
            return await AsyncCursor().execute(query, params)

    def register_vector(connection):
        return None

    async def register_vector_async(connection):
        return None

    for name, value in (
        ("Cursor", Cursor),
        ("Connection", Connection),
        ("AsyncCursor", AsyncCursor),
        ("AsyncConnection", AsyncConnection),
    ):
        setattr(psycopg, name, value)
    pgvector_psycopg.register_vector = register_vector
    pgvector_psycopg.register_vector_async = register_vector_async
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", pgvector_psycopg)

    method_specs = (
        MethodPatchSpec("psycopg", "Connection", "connection", ("execute",)),
        MethodPatchSpec(
            "psycopg",
            "Cursor",
            "cursor",
            ("execute", "executemany", "fetchall", "fetchmany", "fetchone"),
        ),
        MethodPatchSpec("psycopg", "AsyncConnection", "connection", ("execute",), True),
        MethodPatchSpec(
            "psycopg",
            "AsyncCursor",
            "cursor",
            ("execute", "executemany", "fetchall", "fetchmany", "fetchone"),
            True,
        ),
    )
    function_specs = (
        FunctionPatchSpec("pgvector.psycopg", "register_vector", "register_vector"),
        FunctionPatchSpec(
            "pgvector.psycopg",
            "register_vector_async",
            "register_vector",
            True,
        ),
    )
    monkeypatch.setattr(_instrumentation, "METHOD_PATCH_SPECS", method_specs)
    monkeypatch.setattr(_instrumentation, "FUNCTION_PATCH_SPECS", function_specs)
    return psycopg, pgvector_psycopg


@pytest.fixture(autouse=True)
def reset_instrumentor():
    PGVectorInstrumentor._patches_applied = False
    PGVectorInstrumentor._activation_count = 0
    PGVectorInstrumentor._patched_targets = []
    yield
    for target, method in reversed(PGVectorInstrumentor._patched_targets):
        _instrumentation.unwrap(target, method)
    PGVectorInstrumentor._patches_applied = False
    PGVectorInstrumentor._activation_count = 0
    PGVectorInstrumentor._patched_targets = []


def _assert_contract(span):
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH]
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


def test_package_exports_pgvector_instrumentor():
    assert PGVectorInstrumentor is _instrumentation.PGVectorInstrumentor
    assert PGVectorInstrumentor.name == "pgvector"


def test_registration_execute_and_fetch_emit_canonical_spans(monkeypatch):
    psycopg, pgvector_psycopg = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()

    connection = psycopg.Connection()
    pgvector_psycopg.register_vector(connection)
    cursor = connection.execute(
        "SELECT id, embedding FROM items ORDER BY embedding <-> %s LIMIT 2",
        ([0.1, 0.2, 0.3],),
    )
    rows = cursor.fetchall()

    assert rows[0][1] == [0.1, 0.2, 0.3]
    assert [span.name for span in tracer.spans] == [
        "pgvector.register_vector",
        "pgvector.connection.execute",
        "pgvector.cursor.fetchall",
    ]
    for span in tracer.spans:
        _assert_contract(span)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
        assert span.status.status_code is StatusCode.OK
    assert (
        "embedding <->"
        in tracer.spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert "0.3" in tracer.spans[2].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    instrumentor.deactivate()


@pytest.mark.asyncio
async def test_async_execute_and_fetch_are_awaited(monkeypatch):
    psycopg, pgvector_psycopg = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = PGVectorInstrumentor()
    instrumentor.instrument()

    connection = psycopg.AsyncConnection()
    await pgvector_psycopg.register_vector_async(connection)
    cursor = await connection.execute(
        "SELECT embedding FROM items ORDER BY embedding <=> %s LIMIT 1",
        ([0.4, 0.5, 0.6],),
    )
    rows = await cursor.fetchall()

    assert rows == [(3, [0.4, 0.5, 0.6])]
    assert [span.name for span in tracer.spans] == [
        "pgvector.register_vector",
        "pgvector.connection.execute",
        "pgvector.cursor.fetchall",
    ]
    instrumentor.uninstrument()


def test_error_status_and_capture_content_control(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = PGVectorInstrumentor(capture_content=False)
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="invalid vector query"):
        psycopg.Cursor().execute("BROKEN VECTOR QUERY")

    span = tracer.spans[-1]
    assert span.status.status_code is StatusCode.ERROR
    assert span.exceptions
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "invalid vector query"
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    _assert_contract(span)
    instrumentor.deactivate()


def test_full_vectors_survive_canonical_input_and_output(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()
    vector = [0.125] * (MAX_ATTRIBUTE_CHARS + 17)
    labels = [f"label-{index}" for index in range(MAX_PREVIEW_ITEMS + 17)]
    psycopg.Cursor.result_vector = vector

    cursor = psycopg.Cursor()
    cursor.execute(
        "SELECT embedding FROM items ORDER BY embedding <-> %s LIMIT 1",
        (vector, labels),
    )
    cursor.fetchone()

    entity_input = json.loads(
        tracer.spans[-2].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    entity_output = json.loads(
        tracer.spans[-1].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    assert entity_input["params"][0] == vector
    assert entity_output[1] == vector
    assert len(entity_output[1]) == len(vector)
    assert (
        len(tracer.spans[-2].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        > MAX_ATTRIBUTE_CHARS
    )
    assert (
        len(tracer.spans[-1].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        > MAX_ATTRIBUTE_CHARS
    )
    assert entity_input["params"][1]["truncated"] is True
    instrumentor.deactivate()


def test_lifecycle_is_idempotent_and_reference_counted(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    first = PGVectorInstrumentor()
    second = PGVectorInstrumentor()
    first.activate()
    first.activate()
    second.activate()

    psycopg.Cursor().execute("SELECT embedding FROM items")
    assert len(tracer.spans) == 1
    first.deactivate()
    psycopg.Cursor().execute("SELECT embedding FROM items")
    assert len(tracer.spans) == 2
    second.deactivate()
    psycopg.Cursor().execute("SELECT embedding FROM items")
    assert len(tracer.spans) == 2
