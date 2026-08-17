"""pgvector and psycopg SDK-specific instrumentation constants."""

from importlib.metadata import PackageNotFoundError, version
from typing import NamedTuple


class MethodPatchSpec(NamedTuple):
    module: str
    class_name: str
    label: str
    methods: tuple[str, ...]
    is_async: bool = False


class FunctionPatchSpec(NamedTuple):
    module: str
    function_name: str
    label: str
    is_async: bool = False


PGVECTOR_INSTRUMENTATION_NAME = "pgvector"
try:
    PGVECTOR_INSTRUMENTATION_VERSION = version("respan-instrumentation-pgvector")
except PackageNotFoundError:  # pragma: no cover - source-only imports
    PGVECTOR_INSTRUMENTATION_VERSION = "unknown"

CURSOR_OPERATIONS = (
    "execute",
    "executemany",
    "fetchall",
    "fetchmany",
    "fetchone",
)
CONNECTION_OPERATIONS = ("execute",)

METHOD_PATCH_SPECS = (
    MethodPatchSpec(
        "psycopg",
        "Connection",
        "connection",
        CONNECTION_OPERATIONS,
    ),
    MethodPatchSpec(
        "psycopg",
        "Cursor",
        "cursor",
        CURSOR_OPERATIONS,
    ),
    MethodPatchSpec(
        "psycopg",
        "ServerCursor",
        "server_cursor",
        CURSOR_OPERATIONS,
    ),
    MethodPatchSpec(
        "psycopg",
        "AsyncConnection",
        "connection",
        CONNECTION_OPERATIONS,
        True,
    ),
    MethodPatchSpec(
        "psycopg",
        "AsyncCursor",
        "cursor",
        CURSOR_OPERATIONS,
        True,
    ),
    MethodPatchSpec(
        "psycopg",
        "AsyncServerCursor",
        "server_cursor",
        CURSOR_OPERATIONS,
        True,
    ),
)

FUNCTION_PATCH_SPECS = (
    FunctionPatchSpec(
        "pgvector.psycopg",
        "register_vector",
        "register_vector",
    ),
    FunctionPatchSpec(
        "pgvector.psycopg",
        "register_vector_async",
        "register_vector",
        True,
    ),
    FunctionPatchSpec(
        "pgvector.psycopg2",
        "register_vector",
        "register_vector_psycopg2",
    ),
)

MAX_ATTRIBUTE_CHARS = 16_000
MAX_PREVIEW_ITEMS = 128
MAX_SERIALIZATION_DEPTH = 7
MAX_STRING_CHARS = 4_096
SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "connection_string",
    "credential",
    "dsn",
    "passfile",
    "sslkey",
    "password",
    "secret",
    "token",
)
