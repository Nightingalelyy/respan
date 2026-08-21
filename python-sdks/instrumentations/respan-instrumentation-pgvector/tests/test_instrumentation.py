import json
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode
from respan_instrumentation_pgvector import PGVectorInstrumentor, _instrumentation
from respan_instrumentation_pgvector._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    MAX_STRING_CHARS,
    PGVECTOR_INSTRUMENTATION_NAME,
    PGVECTOR_INSTRUMENTATION_VERSION,
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
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc, attributes=None):
        self.exceptions.append((exc, attributes or {}))

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))

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
        result_vector: ClassVar[list[float]] = [0.1, 0.2, 0.3]
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
    PGVectorInstrumentor._capture_content_config = None
    yield
    PGVectorInstrumentor._patched_targets = PGVectorInstrumentor._unwrap_targets(
        PGVectorInstrumentor._patched_targets
    )
    PGVectorInstrumentor._patches_applied = False
    PGVectorInstrumentor._activation_count = 0
    PGVectorInstrumentor._patched_targets = []
    PGVectorInstrumentor._capture_content_config = None


def _assert_contract(span):
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] in {
        "",
        attrs[SpanAttributes.TRACELOOP_ENTITY_NAME],
    }
    assert attrs[OTelSpanAttributes.DB_SYSTEM] == "postgresql"
    assert attrs[OTelSpanAttributes.DB_OPERATION]
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
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


def test_readme_keeps_dsn_outside_the_decorated_workflow_boundary():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "def run_query(dsn" not in readme
    assert "dsn = os.environ" in readme
    assert "def run_query(query_vector: list[float], limit: int)" in readme
    assert "run_query([0.1, 0.2, 0.3], 3)" in readme


def test_registration_execute_and_fetch_emit_canonical_spans(monkeypatch):
    psycopg, pgvector_psycopg = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
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
    assert [
        span.attributes[OTelSpanAttributes.DB_OPERATION] for span in tracer.spans
    ] == [
        "register_vector",
        "SELECT",
        "fetchall",
    ]
    for span in tracer.spans:
        _assert_contract(span)
        assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
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
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
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
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = PGVectorInstrumentor(capture_content=False)
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="invalid vector query"):
        psycopg.Cursor().execute("BROKEN VECTOR QUERY")

    span = tracer.spans[-1]
    assert span.status.status_code is StatusCode.ERROR
    assert span.events[0][0] == "exception"
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "invalid vector query"
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    _assert_contract(span)
    instrumentor.deactivate()


def test_vectors_and_other_sequences_are_bounded_in_canonical_content(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
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
    assert entity_input["params"][0] == {
        "count": len(vector),
        "items": vector[:MAX_PREVIEW_ITEMS],
        "kind": "vector",
        "truncated": True,
    }
    assert entity_output[1] == {
        "count": len(vector),
        "items": vector[:MAX_PREVIEW_ITEMS],
        "kind": "vector",
        "truncated": True,
    }
    assert entity_input["params"][1]["truncated"] is True
    for span in tracer.spans[-2:]:
        for key in (
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        ):
            value = span.attributes.get(key)
            if value is not None:
                assert len(value.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS
                json.loads(value)
    instrumentor.deactivate()


def test_lifecycle_is_idempotent_and_reference_counted(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
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


def test_serializer_never_uses_arbitrary_repr_and_redacts_credentials():
    class ConnectionLike:
        def __init__(self):
            self.repr_calls = 0

        def __repr__(self):
            self.repr_calls += 1
            return (
                "<Connection postgresql://admin:secret@localhost/private at 0x1234abcd>"
            )

    connection = ConnectionLike()
    payload = {
        "connection": connection,
        "dsn": "postgresql://admin:secret@localhost/private",
        "authorization": "Bearer should-not-survive",
        "nested": {"password": "should-not-survive"},
        "message": "object at 0x1234abcd",
    }

    serialized = _instrumentation._json_dumps(payload)
    decoded = json.loads(serialized)

    assert connection.repr_calls == 0
    assert decoded["connection"]["type"].endswith("ConnectionLike")
    assert decoded["dsn"] == "<redacted>"
    assert decoded["authorization"] == "<redacted>"
    assert decoded["nested"]["password"] == "<redacted>"
    assert decoded["message"] == "object at 0x<redacted>"
    assert "secret" not in serialized
    assert "localhost" not in serialized
    assert "0x1234abcd" not in serialized
    assert len(serialized.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS


def test_serializer_handles_cycles_depth_and_multibyte_bounds():
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    recursive["text"] = "界" * (MAX_ATTRIBUTE_CHARS * 2)

    serialized = _instrumentation._json_dumps(recursive)

    assert len(serialized.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS
    assert json.loads(serialized)["truncated"] is True
    assert "0x" not in serialized


def test_serializer_handles_hostile_containers_and_non_finite_numbers():
    class HostileMapping(dict):
        def items(self):
            raise RuntimeError("postgresql://admin:secret@localhost/private")

    class HostileSequence(Sequence):
        def __getitem__(self, _index):
            raise RuntimeError("password=secret")

        def __len__(self):
            return MAX_PREVIEW_ITEMS + 1

        def __iter__(self):
            raise RuntimeError("password=secret")

    decoded = json.loads(
        _instrumentation._json_dumps(
            {
                "mapping": HostileMapping(),
                "sequence": HostileSequence(),
                "numbers": [float("nan"), float("inf"), float("-inf")],
            }
        )
    )

    assert decoded["mapping"]["type"].endswith("HostileMapping")
    assert decoded["sequence"]["type"].endswith("HostileSequence")
    assert decoded["numbers"] == ["nan", "inf", "-inf"]
    assert "secret" not in json.dumps(decoded)


def test_serializer_bounds_vendor_vectors_without_eager_conversion(monkeypatch):
    from pgvector import Vector

    vector = Vector([index / 100 for index in range(MAX_PREVIEW_ITEMS + 17)])
    conversion_calls = 0

    def forbidden_to_list(_self):
        nonlocal conversion_calls
        conversion_calls += 1
        raise AssertionError("full vector conversion must not run")

    monkeypatch.setattr(Vector, "to_list", forbidden_to_list)
    payload = json.loads(_instrumentation._json_dumps({"embedding": vector}))

    assert conversion_calls == 0
    preview = payload["embedding"]
    assert preview["count"] == MAX_PREVIEW_ITEMS + 17
    assert preview["kind"] == "vector"
    assert preview["truncated"] is True
    assert preview["items"] == pytest.approx(
        [index / 100 for index in range(MAX_PREVIEW_ITEMS)]
    )

    class VendorLookalike:
        __module__ = "pgvector.future"

        def __init__(self):
            self.calls = 0

        def to_list(self):
            self.calls += 1
            return list(range(MAX_PREVIEW_ITEMS * 100))

    lookalike = VendorLookalike()
    summary = json.loads(_instrumentation._json_dumps(lookalike))
    assert lookalike.calls == 0
    assert summary["type"].endswith("VendorLookalike")


def test_serializer_does_not_call_custom_length_for_truncation_metadata():
    class CountingSequence(Sequence):
        def __init__(self):
            self.length_calls = 0

        def __getitem__(self, index):
            if index >= MAX_PREVIEW_ITEMS + 17:
                raise IndexError
            return float(index)

        def __len__(self):
            self.length_calls += 1
            raise AssertionError("custom __len__ must not run")

        def __iter__(self):
            return iter(float(index) for index in range(MAX_PREVIEW_ITEMS + 17))

    sequence = CountingSequence()
    payload = json.loads(_instrumentation._json_dumps({"vector": sequence}))

    assert sequence.length_calls == 0
    assert payload["vector"]["count"] == f">{MAX_PREVIEW_ITEMS}"
    assert len(payload["vector"]["items"]) == MAX_PREVIEW_ITEMS


def test_error_event_never_calls_raw_exception_recording_or_leaks_broken_str(
    monkeypatch,
):
    class BrokenStringError(RuntimeError):
        string_calls = 0

        def __str__(self):
            type(self).string_calls += 1
            raise RuntimeError("postgresql://admin:secret@localhost/private")

    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)

    def fail():
        raise BrokenStringError()

    with pytest.raises(BrokenStringError):
        PGVectorInstrumentor()._trace_sync("cursor", "execute", fail, (), {})

    span = tracer.spans[-1]
    assert span.exceptions == []
    assert BrokenStringError.string_calls == 0
    assert span.events[0][0] == "exception"
    event_attributes = span.events[0][1]
    assert event_attributes[OTelSpanAttributes.EXCEPTION_TYPE].endswith(
        "BrokenStringError"
    )
    assert event_attributes[OTelSpanAttributes.EXCEPTION_MESSAGE].endswith(
        "BrokenStringError"
    )
    assert "secret" not in json.dumps(event_attributes)
    assert "postgresql://" not in json.dumps(event_attributes)
    assert OTelSpanAttributes.EXCEPTION_STACKTRACE not in event_attributes


def test_real_otel_failure_event_contains_only_sanitized_content(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("respan-pgvector-error-test")
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)

    def fail():
        raise RuntimeError("postgresql://admin:secret@localhost/private at 0x1234abcd")

    with pytest.raises(RuntimeError):
        PGVectorInstrumentor()._trace_sync("cursor", "execute", fail, (), {})

    provider.force_flush()
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "<redacted-dsn>"
    assert span.attributes["error.message"] == "<redacted-dsn>"
    assert len(span.events) == 1
    event = span.events[0]
    assert event.name == "exception"
    event_text = json.dumps(dict(event.attributes))
    assert "postgresql://" not in event_text
    assert "secret" not in event_text
    assert "0x1234abcd" not in event_text
    assert OTelSpanAttributes.EXCEPTION_STACKTRACE not in event.attributes


def test_multibyte_error_status_and_attributes_are_byte_bounded(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("respan-pgvector-utf8-error-test")
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)

    def fail():
        raise RuntimeError("😀" * (MAX_STRING_CHARS * 2))

    with pytest.raises(RuntimeError):
        PGVectorInstrumentor()._trace_sync("cursor", "execute", fail, (), {})

    provider.force_flush()
    span = exporter.get_finished_spans()[0]
    direct_values = (
        span.status.description,
        span.attributes["error.message"],
        span.events[0].attributes[OTelSpanAttributes.EXCEPTION_MESSAGE],
    )
    assert all(value.endswith("...[truncated]") for value in direct_values)
    assert all(
        len(value.encode("utf-8")) <= MAX_STRING_CHARS for value in direct_values
    )
    output = span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert len(output.encode("utf-8")) <= MAX_ATTRIBUTE_CHARS
    json.loads(output)


def test_activation_rolls_back_partial_patches(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    original_wrap = _instrumentation.wrap_function_wrapper
    calls = 0

    def flaky_wrap(module, target, wrapper):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic patch failure")
        return original_wrap(module, target, wrapper)

    monkeypatch.setattr(_instrumentation, "wrap_function_wrapper", flaky_wrap)

    with pytest.raises(RuntimeError, match="synthetic patch failure"):
        PGVectorInstrumentor().activate()

    psycopg.Connection().execute("SELECT 1")
    assert not tracer.spans
    assert PGVectorInstrumentor._patched_targets == []
    assert PGVectorInstrumentor._patches_applied is False
    assert PGVectorInstrumentor._activation_count == 0


def test_concurrent_activation_and_deactivation_keep_one_wrapper(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentors = [PGVectorInstrumentor() for _ in range(12)]

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda item: item.activate(), instrumentors))

    assert PGVectorInstrumentor._activation_count == len(instrumentors)
    psycopg.Cursor().execute("SELECT 1")
    assert len(tracer.spans) == 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda item: item.deactivate(), instrumentors))

    assert PGVectorInstrumentor._activation_count == 0
    assert PGVectorInstrumentor._patched_targets == []
    psycopg.Cursor().execute("SELECT 1")
    assert len(tracer.spans) == 1


def test_active_instances_reject_mismatched_capture_content(monkeypatch):
    _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    without_content = PGVectorInstrumentor(capture_content=False)
    with_content = PGVectorInstrumentor(capture_content=True)
    without_content.activate()

    with pytest.raises(ValueError, match="same capture_content setting"):
        with_content.activate()

    assert PGVectorInstrumentor._activation_count == 1
    assert with_content._is_instrumented is False
    without_content.deactivate()
    with_content.activate()
    assert PGVectorInstrumentor._capture_content_config is True
    with_content.deactivate()


def test_deactivation_does_not_remove_a_later_foreign_wrapper(monkeypatch):
    psycopg, _ = _install_fake_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()

    def foreign_wrapper(wrapped, _instance, args, kwargs):
        return wrapped(*args, **kwargs)

    _instrumentation.wrap_function_wrapper(
        "psycopg",
        "Cursor.execute",
        foreign_wrapper,
    )
    instrumentor.deactivate()

    assert PGVectorInstrumentor._patches_applied is False
    assert len(PGVectorInstrumentor._patched_targets) == 1
    psycopg.Cursor().execute("SELECT 1")
    assert len(tracer.spans) == 0

    _instrumentation.unwrap(psycopg.Cursor, "execute")
    instrumentor.activate()
    psycopg.Cursor().execute("SELECT 1")
    assert len(tracer.spans) == 1
    instrumentor.deactivate()
    assert PGVectorInstrumentor._patched_targets == []


def test_real_otel_export_preserves_hierarchy_content_and_no_duplicates(monkeypatch):
    psycopg, pgvector_psycopg = _install_fake_modules(monkeypatch)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("respan-pgvector-test")
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()

    with tracer.start_as_current_span("workflow"):
        connection = psycopg.Connection()
        pgvector_psycopg.register_vector(connection)
        rows = connection.execute(
            "SELECT embedding FROM items ORDER BY embedding <-> %s LIMIT 1",
            ([0.1, 0.2, 0.3],),
        ).fetchall()

    instrumentor.deactivate()
    provider.force_flush()
    spans = list(exporter.get_finished_spans())
    operation_spans = [span for span in spans if span.name.startswith("pgvector.")]

    assert rows[0][1] == [0.1, 0.2, 0.3]
    assert [span.name for span in operation_spans] == [
        "pgvector.register_vector",
        "pgvector.connection.execute",
        "pgvector.cursor.fetchall",
    ]
    assert len({span.context.span_id for span in operation_spans}) == 3
    workflow = next(span for span in spans if span.name == "workflow")
    assert all(
        span.parent.span_id == workflow.context.span_id for span in operation_spans
    )
    assert all(span.kind is SpanKind.CLIENT for span in operation_spans)
    assert all(span.status.status_code is StatusCode.OK for span in operation_spans)
    for span in operation_spans:
        _assert_contract(span)
        assert (
            span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
            == span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        )
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])


def test_canonical_scope_version_and_root_nested_entity_paths(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer_calls: list[tuple[str, str]] = []

    def get_tracer(name: str, instrumentation_version: str):
        tracer_calls.append((name, instrumentation_version))
        return provider.get_tracer(name, instrumentation_version)

    monkeypatch.setattr(_instrumentation.trace, "get_tracer", get_tracer)
    instrumentor = PGVectorInstrumentor()

    def succeed():
        return None

    instrumentor._trace_sync("cursor", "fetchone", succeed, (), {})
    outer_tracer = provider.get_tracer("pgvector-test-parent")
    with outer_tracer.start_as_current_span("workflow"):
        instrumentor._trace_sync("cursor", "fetchone", succeed, (), {})

    provider.force_flush()
    operation_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "pgvector.cursor.fetchone"
    ]
    assert len(operation_spans) == 2
    root = next(span for span in operation_spans if span.parent is None)
    nested = next(span for span in operation_spans if span.parent is not None)
    assert root.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert nested.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == (
        "pgvector.cursor.fetchone"
    )
    assert tracer_calls == [
        (PGVECTOR_INSTRUMENTATION_NAME, PGVECTOR_INSTRUMENTATION_VERSION),
        (PGVECTOR_INSTRUMENTATION_NAME, PGVECTOR_INSTRUMENTATION_VERSION),
    ]
    assert PGVECTOR_INSTRUMENTATION_VERSION == version(
        "respan-instrumentation-pgvector"
    )
    for span in operation_spans:
        assert span.instrumentation_scope.name == PGVECTOR_INSTRUMENTATION_NAME
        assert span.instrumentation_scope.version == PGVECTOR_INSTRUMENTATION_VERSION


@pytest.mark.asyncio
async def test_current_pgvector_and_psycopg_exports_are_wrapped_and_restored(
    monkeypatch,
):
    import pgvector.psycopg as pgvector_psycopg
    import psycopg

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("respan-pgvector-current-sdk-test")
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda *_: tracer)
    original_register = pgvector_psycopg.register_vector
    original_register_async = pgvector_psycopg.register_vector_async
    original_execute = psycopg.Connection.execute
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()

    assert pgvector_psycopg.register_vector is not original_register
    assert pgvector_psycopg.register_vector_async is not original_register_async
    assert psycopg.Connection.execute is not original_execute
    with pytest.raises((AttributeError, TypeError)):
        pgvector_psycopg.register_vector(object())
    with pytest.raises((AttributeError, TypeError)):
        await pgvector_psycopg.register_vector_async(object())

    instrumentor.deactivate()
    provider.force_flush()
    spans = list(exporter.get_finished_spans())

    assert pgvector_psycopg.register_vector is original_register
    assert pgvector_psycopg.register_vector_async is original_register_async
    assert psycopg.Connection.execute is original_execute
    assert [span.name for span in spans] == [
        "pgvector.register_vector",
        "pgvector.register_vector",
    ]
    assert all(span.status.status_code is StatusCode.ERROR for span in spans)
    for span in spans:
        _assert_contract(span)
        assert len(span.events) == 1
        payload = span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
        assert "0x" not in payload
        assert "postgresql://" not in payload
        assert json.loads(payload)["context"]["type"] == "object"


@pytest.mark.skipif(
    not os.getenv("PGVECTOR_DSN"), reason="PGVECTOR_DSN is not configured"
)
def test_live_psycopg_sync_registration_server_cursor_and_rollback(monkeypatch):
    import pgvector.psycopg as pgvector_psycopg
    import psycopg

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda *_: provider.get_tracer("respan-pgvector-live-sync"),
    )
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()
    connection = psycopg.connect(os.environ["PGVECTOR_DSN"])
    try:
        extension = connection.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if not extension:
            pytest.skip("the configured PostgreSQL database does not install pgvector")
        pgvector_psycopg.register_vector(connection)
        connection.execute(
            "CREATE TEMP TABLE respan_pgvector_test (embedding vector(3))"
        )
        connection.execute(
            "INSERT INTO respan_pgvector_test VALUES (%s)",
            ([0.1, 0.2, 0.3],),
        )
        with connection.cursor(name="respan_pgvector_server_cursor") as cursor:
            cursor.execute("SELECT embedding FROM respan_pgvector_test")
            assert cursor.fetchall()[0][0] == [0.1, 0.2, 0.3]
        connection.rollback()
    finally:
        connection.close()
        instrumentor.deactivate()
    provider.force_flush()
    assert any(
        span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME)
        == "pgvector.server_cursor.fetchall"
        for span in exporter.get_finished_spans()
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("PGVECTOR_DSN"), reason="PGVECTOR_DSN is not configured"
)
async def test_live_psycopg_async_registration_and_rollback(monkeypatch):
    import pgvector.psycopg as pgvector_psycopg
    import psycopg

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        _instrumentation.trace,
        "get_tracer",
        lambda *_: provider.get_tracer("respan-pgvector-live-async"),
    )
    instrumentor = PGVectorInstrumentor()
    instrumentor.activate()
    connection = await psycopg.AsyncConnection.connect(os.environ["PGVECTOR_DSN"])
    try:
        extension = await connection.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if not await extension.fetchone():
            pytest.skip("the configured PostgreSQL database does not install pgvector")
        await pgvector_psycopg.register_vector_async(connection)
        await connection.execute(
            "CREATE TEMP TABLE respan_pgvector_async_test (embedding vector(3))"
        )
        await connection.execute(
            "INSERT INTO respan_pgvector_async_test VALUES (%s)",
            ([0.4, 0.5, 0.6],),
        )
        cursor = await connection.execute(
            "SELECT embedding FROM respan_pgvector_async_test"
        )
        assert (await cursor.fetchone())[0] == [0.4, 0.5, 0.6]
        await connection.rollback()
    finally:
        await connection.close()
        instrumentor.deactivate()
    provider.force_flush()
    assert any(
        span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME)
        == "pgvector.connection.execute"
        for span in exporter.get_finished_spans()
    )
