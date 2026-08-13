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
import psycopg
import pgvector.psycopg as pgvector_psycopg
from respan import Respan, workflow
from respan_instrumentation_pgvector import PGVectorInstrumentor

respan = Respan(instrumentations=[PGVectorInstrumentor()])


@workflow(name="pgvector_similarity_query_workflow")
def run_query(dsn: str):
    with psycopg.connect(dsn) as connection:
        pgvector_psycopg.register_vector(connection)
        return connection.execute(
            "SELECT id, embedding FROM items ORDER BY embedding <-> %s LIMIT 3",
            ([0.1, 0.2, 0.3],),
        ).fetchall()


print(run_query("postgresql://localhost/postgres"))
respan.shutdown()
```

Pass `capture_content=False` to omit SQL arguments, parameters, and returned
rows while retaining operation names, status, and PostgreSQL attributes.
Activation/deactivation and `instrument()` / `uninstrument()` are idempotent.
SQLAlchemy applications that use the psycopg 3 driver are covered at the
underlying connection/cursor layer.

See `respan-example-projects/python/tracing/pgvector` for setup, distance
operators, sync/async, mutation, bulk insert, and deterministic error examples.
