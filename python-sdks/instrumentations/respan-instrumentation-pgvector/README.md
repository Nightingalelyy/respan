# respan-instrumentation-pgvector

Respan instrumentation for pgvector applications using psycopg 3. It traces
sync and async pgvector type registration, connection/cursor execution,
executemany, and result fetching as canonical Respan task spans. Psycopg 2 type
registration is also detected when the optional `psycopg2` extra is installed.

## Install

```bash
pip install respan-ai respan-instrumentation-pgvector pgvector "psycopg[binary]"
```

For pgvector's psycopg 2 adapter:

```bash
pip install "respan-instrumentation-pgvector[psycopg2]"
```

## Quickstart

```python
import os

import psycopg
import pgvector.psycopg as pgvector_psycopg
from respan import Respan, workflow
from respan_instrumentation_pgvector import PGVectorInstrumentor

respan = Respan(instrumentations=[PGVectorInstrumentor()])
dsn = os.environ["PGVECTOR_DSN"]


@workflow(name="pgvector_similarity_query_workflow")
def run_query(query_vector: list[float], limit: int):
    with psycopg.connect(dsn) as connection:
        pgvector_psycopg.register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, embedding FROM items ORDER BY embedding <-> %s LIMIT %s",
                (query_vector, limit),
            )
            return cursor.fetchall()


try:
    print(run_query([0.1, 0.2, 0.3], 3))
finally:
    try:
        respan.flush()
    finally:
        respan.shutdown()
```

Pass `capture_content=False` to omit SQL arguments, parameters, and returned
rows while retaining operation names, status, and PostgreSQL attributes.
Activation/deactivation and `instrument()` / `uninstrument()` are idempotent.
All concurrently active instrumentors must use the same `capture_content`
setting; a conflicting activation is rejected instead of silently using the
first instance's setting.
SQLAlchemy applications that use the psycopg 3 driver are covered at the
underlying connection/cursor layer.

Captured values are valid JSON and capped at 16,000 UTF-8 bytes. Collections
and vectors include at most 128 preview items plus their count/truncation
metadata. Connection and other unsupported SDK objects are represented by a
stable type name rather than arbitrary `repr()` output; DSNs, credentials,
secrets, and process-specific memory addresses are redacted. Operation identity
is retained in `traceloop.entity.name` and standard database attributes. The
default semantic exporter displays these auto-emitted operation nodes as the
contract-defined bare `task` span name.

See `respan-example-projects/python/tracing/pgvector` for setup, distance
operators, sync/async, mutation, bulk insert, and deterministic error examples.
