"""Real MCP 1.x stdio server for instrumentation integration tests."""

from __future__ import annotations

import sys

if "--exit-immediately" in sys.argv:
    raise SystemExit(3)

try:
    from openinference.instrumentation.mcp import (
        MCPInstrumentor as OpenInferenceMCPInstrumentor,
    )
except ImportError:
    OpenInferenceMCPInstrumentor = None
else:
    OpenInferenceMCPInstrumentor().instrument()

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings

# MCP 1.x with current Pydantic otherwise warns while resolving the generic
# lifespan annotation. Rebuild before FastMCP constructs its Settings instance.
Settings.model_rebuild(force=True)

server = FastMCP("respan-mcp-instrumentation-test")


@server.tool()
def summarize_city(city: str) -> str:
    return f"{city}: ready"


@server.tool()
def current_trace_id() -> str:
    try:
        from opentelemetry import trace
    except ImportError:
        return ""

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return ""
    return f"{span_context.trace_id:032x}"


if __name__ == "__main__":
    server.run(transport="stdio")
