import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";

import { MCPInstrumentor } from "../dist/index.js";

const capturedSpans = [];
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

class FakeSpan {
  constructor(name, attributes) {
    this.name = name;
    this.attributes = { ...attributes };
    this.ended = false;
    this.exceptions = [];
    this.status = undefined;
  }

  setAttribute(key, value) {
    this.attributes[key] = value;
  }

  recordException(error) {
    this.exceptions.push(error);
  }

  setStatus(status) {
    this.status = status;
  }

  end() {
    this.ended = true;
    capturedSpans.push(this);
  }
}

function resetTracer() {
  capturedSpans.length = 0;
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  trace.setGlobalTracerProvider({
    getTracer() {
      return {
        startActiveSpan(name, optionsOrFn, maybeFn) {
          const fn = typeof optionsOrFn === "function" ? optionsOrFn : maybeFn;
          const options = typeof optionsOrFn === "function" ? {} : optionsOrFn;
          const span = new FakeSpan(name, options?.attributes ?? {});
          return fn(span);
        },
      };
    },
  });
}

function createFakeClientClass() {
  return class FakeClient {
    async connect() {
      return undefined;
    }

    async listTools() {
      return {
        tools: [{ name: "summarize_city", description: "Summarize a city" }],
      };
    }

    async callTool(params) {
      return {
        content: [{ type: "text", text: `${params.name}: ok` }],
      };
    }

    async readResource(params) {
      return { contents: [{ uri: params.uri, text: "resource text" }] };
    }

    async getPrompt(params) {
      return {
        messages: [{ role: "user", content: { type: "text", text: params.name } }],
      };
    }
  };
}

function createFakeServerClass() {
  return class FakeMcpServer {
    registerTool(name, config, cb) {
      return { name, config, handler: cb };
    }

    tool(name, ...args) {
      return { name, handler: args.at(-1) };
    }
  };
}

test.after(() => {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

test("patches MCP Client.callTool as a tool span with canonical attrs", async () => {
  resetTracer();
  const FakeClient = createFakeClientClass();
  const instrumentor = new MCPInstrumentor({
    clientModule: { Client: FakeClient },
    serverModule: {},
  });
  await instrumentor.activate();

  const client = new FakeClient();
  const result = await client.callTool({
    name: "summarize_city",
    arguments: { city: "Paris" },
  });

  assert.equal(result.content[0].text, "summarize_city: ok");
  assert.equal(capturedSpans.length, 1);

  const span = capturedSpans[0];
  assert.equal(span.name, "mcp.tool.summarize_city");
  assert.equal(span.ended, true);
  assert.equal(
    span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE],
    "tool",
  );
  assert.equal(span.attributes["traceloop.entity.name"], "summarize_city");
  assert.equal(span.attributes["traceloop.entity.path"], "summarize_city");
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.input"]), {
    name: "summarize_city",
    arguments: { city: "Paris" },
  });
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.output"]), {
    content: [{ type: "text", text: "summarize_city: ok" }],
  });
  assert.equal(span.attributes["traceloop.span.kind"], undefined);
  assert.equal(span.attributes.tool_calls, undefined);
  assert.equal(span.attributes["mcp.method"], "callTool");
  assert.equal(span.attributes["mcp.operation"], "client.call_tool");

  instrumentor.deactivate();
});

test("patches MCP Client.listTools as a task span", async () => {
  resetTracer();
  const FakeClient = createFakeClientClass();
  const instrumentor = new MCPInstrumentor({
    clientModule: { Client: FakeClient },
    serverModule: {},
  });
  await instrumentor.activate();

  const client = new FakeClient();
  const result = await client.listTools();

  assert.equal(result.tools[0].name, "summarize_city");
  const span = capturedSpans[0];
  assert.equal(span.name, "mcp.list_tools");
  assert.equal(
    span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE],
    "task",
  );
  assert.equal(span.attributes["traceloop.entity.name"], "mcp.list_tools");
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.input"]), {
    method: "listTools",
    params: {},
  });

  instrumentor.deactivate();
});

test("records MCP client errors on the span before rethrowing", async () => {
  resetTracer();
  class ErrorClient {
    async callTool() {
      throw new RuntimeError("tool failed");
    }
  }
  class RuntimeError extends Error {
    constructor(message) {
      super(message);
      this.name = "RuntimeError";
    }
  }

  const instrumentor = new MCPInstrumentor({
    clientModule: { Client: ErrorClient },
    serverModule: {},
  });
  await instrumentor.activate();

  await assert.rejects(
    () => new ErrorClient().callTool({ name: "broken_tool" }),
    /tool failed/,
  );

  const span = capturedSpans[0];
  assert.equal(span.name, "mcp.tool.broken_tool");
  assert.equal(span.status.code, 2);
  assert.equal(span.status.message, "tool failed");
  assert.equal(span.exceptions.length, 1);
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.output"]), {
    error: "RuntimeError",
    message: "tool failed",
  });

  instrumentor.deactivate();
});

test("patches MCP Client resource and prompt methods with stable names", async () => {
  resetTracer();
  const FakeClient = createFakeClientClass();
  const instrumentor = new MCPInstrumentor({
    clientModule: { Client: FakeClient },
    serverModule: {},
  });
  await instrumentor.activate();

  const client = new FakeClient();
  await client.readResource({ uri: "demo://city/paris" });
  await client.getPrompt({ name: "city_brief", arguments: { city: "Paris" } });

  assert.equal(capturedSpans[0].name, "mcp.resource.read");
  assert.equal(capturedSpans[0].attributes["traceloop.entity.name"], "demo://city/paris");
  assert.equal(capturedSpans[0].attributes["mcp.operation"], "client.read_resource");
  assert.equal(capturedSpans[1].name, "mcp.prompt.city_brief");
  assert.equal(capturedSpans[1].attributes["traceloop.entity.name"], "city_brief");
  assert.equal(capturedSpans[1].attributes["mcp.operation"], "client.get_prompt");

  instrumentor.deactivate();
});

test("wraps MCP server registerTool handlers as tool spans", async () => {
  resetTracer();
  const FakeServer = createFakeServerClass();
  const instrumentor = new MCPInstrumentor({
    clientModule: {},
    serverModule: { McpServer: FakeServer },
  });
  await instrumentor.activate();

  const server = new FakeServer();
  const registered = server.registerTool(
    "lookup_city",
    { description: "Lookup a city" },
    async (args) => ({
      content: [{ type: "text", text: `City: ${args.city}` }],
    }),
  );

  const result = await registered.handler({ city: "Paris" });

  assert.equal(result.content[0].text, "City: Paris");
  const span = capturedSpans[0];
  assert.equal(span.name, "mcp.tool.lookup_city");
  assert.equal(
    span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE],
    "tool",
  );
  assert.equal(span.attributes["traceloop.entity.name"], "lookup_city");
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.input"]), {
    name: "lookup_city",
    arguments: { city: "Paris" },
  });

  instrumentor.deactivate();
});

test("wraps legacy MCP server tool handlers as tool spans", async () => {
  resetTracer();
  const FakeServer = createFakeServerClass();
  const instrumentor = new MCPInstrumentor({
    clientModule: {},
    serverModule: { McpServer: FakeServer },
  });
  await instrumentor.activate();

  const server = new FakeServer();
  const registered = server.tool(
    "legacy_echo",
    "Echo an input value",
    {},
    async () => ({ content: [{ type: "text", text: "legacy echo ok" }] }),
  );

  const result = await registered.handler({});

  assert.equal(result.content[0].text, "legacy echo ok");
  const span = capturedSpans[0];
  assert.equal(span.name, "mcp.tool.legacy_echo");
  assert.equal(span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], "tool");
  assert.equal(span.attributes["traceloop.entity.name"], "legacy_echo");
  assert.equal(span.attributes["mcp.operation"], "server.tool");

  instrumentor.deactivate();
});

test("deactivate restores patched client methods", async () => {
  resetTracer();
  const FakeClient = createFakeClientClass();
  const original = FakeClient.prototype.callTool;
  const instrumentor = new MCPInstrumentor({
    clientModule: { Client: FakeClient },
    serverModule: {},
  });

  await instrumentor.activate();
  assert.notEqual(FakeClient.prototype.callTool, original);

  instrumentor.deactivate();
  assert.equal(FakeClient.prototype.callTool, original);

  const client = new FakeClient();
  await client.callTool({ name: "not_traced" });
  assert.equal(capturedSpans.length, 0);
});
