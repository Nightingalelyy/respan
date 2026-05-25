# respan-instrumentation-mcp

Respan instrumentation plugin for [Model Context Protocol](https://modelcontextprotocol.io/) Python applications.

The package enables upstream OpenInference MCP transport context propagation and adds Respan spans for common `mcp.ClientSession` operations such as tool listing, tool calls, resource reads, and prompt fetches.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-mcp mcp
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates tracing export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

```python
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from respan import Respan, workflow
from respan_instrumentation_mcp import MCPInstrumentor

respan = Respan(instrumentations=[MCPInstrumentor()])


@workflow(name="mcp_tool_call_workflow")
async def run_mcp_client() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([tool.name for tool in tools.tools])
            result = await session.call_tool(
                "summarize_city",
                arguments={"city": "Paris"},
            )
            print(result)


asyncio.run(run_mcp_client())
respan.flush()
respan.shutdown()
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects) for runnable MCP scripts.
