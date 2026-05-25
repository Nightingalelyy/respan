# @respan/instrumentation-mcp

Respan instrumentation plugin for the [Model Context Protocol](https://modelcontextprotocol.io/) TypeScript SDK.

The package adds Respan spans for common MCP client operations and server-side tool handlers from `@modelcontextprotocol/sdk`.

## Install

```bash
npm install @respan/respan @respan/instrumentation-mcp @modelcontextprotocol/sdk
```

## Quickstart

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { Respan } from "@respan/respan";
import { MCPInstrumentor } from "@respan/instrumentation-mcp";

const respan = new Respan({
  instrumentations: [new MCPInstrumentor()],
});
await respan.initialize();

const transport = new StdioClientTransport({
  command: "node",
  args: ["server.js"],
});

const client = new Client({ name: "demo-client", version: "1.0.0" });
await client.connect(transport);

const tools = await client.listTools();
console.log(tools.tools.map((tool) => tool.name));

const result = await client.callTool({
  name: "summarize_city",
  arguments: { city: "Paris" },
});
console.log(result);

await respan.flush();
```

## Captured Operations

- `Client.connect()`
- `Client.listTools()`
- `Client.callTool()`
- `Client.listResources()`
- `Client.readResource()`
- `Client.listResourceTemplates()`
- `Client.listPrompts()`
- `Client.getPrompt()`
- `McpServer.registerTool()` handlers
- `McpServer.tool()` handlers

Client tool calls and server tool handlers are emitted as `tool` spans. List, resource, prompt, and connect operations are emitted as `task` spans.
