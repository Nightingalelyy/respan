import assert from "node:assert/strict";
import test from "node:test";
import { context, trace, TraceFlags, SpanStatusCode } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  AlwaysOffSampler,
  AlwaysOnSampler,
  SamplingDecision,
  SimpleSpanProcessor,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as Cursor from "@cursor/sdk";
import { CursorSDKInstrumentor } from "../dist/index.js";
import { nativeFixture } from "./native_fixture.mjs";

async function setup({
  traceContent = true,
  sampler = new AlwaysOnSampler(),
} = {}) {
  trace.disable();
  context.disable();
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    sampler,
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.setGlobalTracerProvider(provider);
  context.setGlobalContextManager(
    new AsyncLocalStorageContextManager().enable(),
  );
  const fixture = await nativeFixture();
  const plugin = new CursorSDKInstrumentor({
    sdkModule: { ...Cursor },
    agentName: "fixture-agent",
    traceContent,
  });
  plugin.activate();
  return {
    fixture,
    plugin,
    exporter,
    provider,
    spans: () => exporter.getFinishedSpans(),
    async close() {
      plugin.deactivate();
      await fixture.close();
      await provider.shutdown();
      trace.disable();
      context.disable();
    },
  };
}
async function stream(s, options = {}) {
  const agent = await Cursor.Agent.create({ ...s.fixture.options, ...options });
  const run = await agent.send(options.prompt ?? "fixture prompt");
  const events = [];
  for await (const event of run.stream()) events.push(event);
  const result = await run.wait();
  agent.close();
  return { run, events, result };
}
function content(spans) {
  return spans.flatMap((span) =>
    Object.entries(span.attributes).filter(([key]) =>
      [
        "traceloop.entity.input",
        "traceloop.entity.output",
        "respan.metadata",
      ].includes(key),
    ),
  );
}

test("real released Cursor native stream preserves agent, tools, IDs, aggregate usage and results", async () => {
  const s = await setup();
  try {
    const { events, result } = await stream(s);
    assert.equal(result.result, "native Cursor answer");
    assert.equal(events.filter((e) => e.type === "tool_call").length, 2);
    const spans = s.spans();
    assert.equal(
      spans.filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      1,
    );
    assert.equal(
      spans.filter((v) => v.attributes["respan.entity.log_type"] === "chat")
        .length,
      0,
    );
    const agent = spans.find(
      (v) => v.attributes["respan.entity.log_type"] === "agent",
    );
    const tool = spans.find(
      (v) => v.attributes["respan.entity.log_type"] === "tool",
    );
    assert.equal(tool.parentSpanContext.spanId, agent.spanContext().spanId);
    assert.equal(tool.attributes["gen_ai.tool.call.id"], "tool-" + result.id);
    assert.deepEqual(JSON.parse(tool.attributes["traceloop.entity.output"]), {
      value: false,
      empty: "",
      zero: 0,
      items: [],
    });
    const metadata = JSON.parse(agent.attributes["respan.metadata"]);
    assert.equal(metadata.cursor_agent_id, s.fixture.options.agentId);
    assert.equal(metadata.cursor_run_id, result.id);
    if (result.usage) assert.deepEqual(metadata.cursor_usage, result.usage);
    else assert.equal(metadata.cursor_usage, undefined);
    for (const span of spans) {
      assert.equal(span.attributes["traceloop.span.kind"], undefined);
      assert.equal(span.attributes["gen_ai.system"], undefined);
    }
  } finally {
    await s.close();
  }
});
test("wait-only observes real native callbacks and returns exact native result shape", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    let callbacks = 0;
    const run = await agent.send("fixture", {
      onDelta() {
        callbacks++;
      },
    });
    const result = await run.wait();
    assert.equal(result.status, "finished");
    assert.ok(callbacks > 0);
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      1,
    );
    agent.close();
  } finally {
    await s.close();
  }
});
test("Agent.prompt uses one real agent span without nested duplicates", async () => {
  const s = await setup();
  try {
    const result = await Cursor.Agent.prompt("fixture", s.fixture.options);
    assert.equal(result.status, "finished");
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      1,
    );
  } finally {
    await s.close();
  }
});
test("large native tool payload is complete under default OTel attribute limits", async () => {
  const s = await setup();
  try {
    await stream(s, { prompt: "large" });
    const tool = s
      .spans()
      .find((v) => v.attributes["respan.entity.log_type"] === "tool");
    assert.equal(
      JSON.parse(tool.attributes["traceloop.entity.input"]).arguments.items
        .length,
      80,
    );
    assert.equal(
      JSON.parse(tool.attributes["traceloop.entity.output"]).items.length,
      5001,
    );
  } finally {
    await s.close();
  }
});
test("constructor content false suppresses native payloads", async () => {
  const s = await setup({ traceContent: false });
  try {
    await stream(s);
    assert.deepEqual(content(s.spans()), []);
  } finally {
    await s.close();
  }
});
test("environment false is an immutable ceiling", async () => {
  process.env.RESPAN_TRACE_CONTENT = "false";
  const s = await setup();
  try {
    await stream(s);
    assert.deepEqual(content(s.spans()), []);
  } finally {
    delete process.env.RESPAN_TRACE_CONTENT;
    await s.close();
  }
});
test("canonical context false wins over constructor true", async () => {
  const s = await setup();
  try {
    await context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
      () => stream(s),
    );
    assert.deepEqual(content(s.spans()), []);
  } finally {
    await s.close();
  }
});
test("tracing suppression does not capture native payloads", async () => {
  const s = await setup();
  try {
    await context.with(suppressTracing(context.active()), () => stream(s));
    assert.deepEqual(content(s.spans()), []);
  } finally {
    await s.close();
  }
});
test("DROP sampler runs before instrumentation payload serialization", async () => {
  let serialized = 0,
    sampled = 0;
  const s = await setup({
    sampler: {
      shouldSample() {
        sampled++;
        return { decision: SamplingDecision.NOT_RECORD };
      },
      toString() {
        return "fixture-drop";
      },
    },
  });
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const message = {
      text: "fixture",
      toJSON() {
        serialized++;
        return { text: this.text };
      },
    };
    const run = await agent.send(message);
    await run.wait();
    assert.ok(sampled > 0);
    assert.equal(serialized, 0);
    assert.equal(s.spans().length, 0);
    agent.close();
  } finally {
    await s.close();
  }
});
test("RECORD_ONLY sampler never captures payloads", async () => {
  const s = await setup({
    sampler: {
      shouldSample() {
        return { decision: SamplingDecision.RECORD };
      },
      toString() {
        return "fixture-record";
      },
    },
  });
  try {
    await stream(s);
    assert.deepEqual(content(s.spans()), []);
  } finally {
    await s.close();
  }
});
test("actual span guards deny late setters, direct fields, events and status messages", async () => {
  const s = await setup({ traceContent: false });
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture", {
      onDelta() {
        const span = trace.getSpan(context.active());
        span?.setAttributes({
          "traceloop.entity.input": "private",
          "gen_ai.usage.evil": "private",
        });
        span?.addEvent("private", { secret: "private" });
        span?.setStatus({ code: 2, message: "private" });
        if (span?.attributes)
          span.attributes = {
            "traceloop.entity.output": "private",
            evil: "private",
          };
      },
    });
    await run.wait();
    assert.deepEqual(content(s.spans()), []);
    for (const span of s.spans()) {
      assert.equal(span.attributes.evil, undefined);
      assert.equal(span.status.message, undefined);
      assert.equal(span.events.length, 0);
    }
    agent.close();
  } finally {
    await s.close();
  }
});
test("native structured failure maps ERROR and preserves native wait result", async () => {
  const s = await setup();
  try {
    const { result } = await stream(s, { prompt: "error" });
    assert.equal(result.status, "error");
    const agent = s
      .spans()
      .find((v) => v.attributes["respan.entity.log_type"] === "agent");
    assert.equal(agent.status.code, SpanStatusCode.ERROR);
    assert.equal(agent.attributes["traceloop.entity.output"], undefined);
  } finally {
    await s.close();
  }
});
test("deactivation drains a run and restores only owned methods", async () => {
  const s = await setup();
  try {
    const original = Cursor.Agent.create;
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture");
    s.plugin.deactivate();
    assert.equal(s.plugin.isActive(), false);
    await run.wait();
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      1,
    );
    s.plugin.activate();
    assert.equal(s.plugin.isActive(), true);
    agent.close();
  } finally {
    await s.close();
  }
});
test("multiple owners share one patch and any false owner denies capture", async () => {
  const s = await setup();
  const second = new CursorSDKInstrumentor({
    sdkModule: { ...Cursor },
    traceContent: false,
  });
  second.activate();
  try {
    await stream(s);
    assert.deepEqual(content(s.spans()), []);
    s.plugin.deactivate();
    assert.equal(second.isActive(), true);
  } finally {
    second.deactivate();
    await s.close();
  }
});
test("foreign patches survive deactivation", async () => {
  const s = await setup();
  const before = Cursor.Agent.create;
  const foreign = function (...args) {
    return before.apply(this, args);
  };
  Cursor.Agent.create = foreign;
  try {
    s.plugin.deactivate();
    assert.equal(Cursor.Agent.create, foreign);
  } finally {
    Cursor.Agent.create = before;
    await s.close();
  }
});

test("multimodal native input and model override survive without inventing a provider", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const message = {
      text: "image fixture",
      images: [
        {
          data: "Zml4dHVyZQ==",
          mimeType: "image/png",
          dimension: { width: 1, height: 1 },
        },
      ],
    };
    const run = await agent.send(
      message,
      Cursor.JsonlLocalAgentStore
        ? { model: { id: "cursor-fixture-next" } }
        : undefined,
    );
    await run.wait();
    const span = s
      .spans()
      .find((v) => v.attributes["respan.entity.log_type"] === "agent");
    assert.deepEqual(
      JSON.parse(span.attributes["traceloop.entity.input"]),
      message,
    );
    assert.equal(
      JSON.parse(span.attributes["respan.metadata"]).cursor_model,
      Cursor.JsonlLocalAgentStore
        ? "cursor-fixture-next"
        : "cursor-fixture-model",
    );
    agent.close();
  } finally {
    await s.close();
  }
});
test("run stream generator identity and return behavior remain native", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture");
    const iterator = run.stream();
    assert.equal(iterator[Symbol.asyncIterator](), iterator);
    assert.equal(typeof iterator.return, "function");
    await iterator.next();
    const result = await iterator.return();
    assert.equal(result.done, true);
    await run.wait();
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      1,
    );
    agent.close();
  } finally {
    await s.close();
  }
});
test("consumer break does not complete a still-running native run", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture");
    for await (const event of run.stream()) break;
    await run.wait();
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      1,
    );
    agent.close();
  } finally {
    await s.close();
  }
});
test("late canonical context false cannot be reversed for an existing run", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture");
    await context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
      () => run.wait(),
    );
    assert.deepEqual(content(s.spans()), []);
    agent.close();
  } finally {
    await s.close();
  }
});
test("native cancel returns undefined and closes the recorded run as cancelled", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture");
    assert.equal(await run.cancel(), undefined);
    const span = s
      .spans()
      .find((v) => v.attributes["respan.entity.log_type"] === "agent");
    assert.equal(span.status.code, SpanStatusCode.ERROR);
    assert.equal(
      JSON.parse(span.attributes["respan.metadata"]).cursor_status,
      "cancelled",
    );
    agent.close();
  } finally {
    await s.close();
  }
});
test("real native getRun waits a detached cloud handle", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    const run = await agent.send("fixture");
    await run.wait();
    const restored = await Cursor.Agent.getRun(run.id, {
      runtime: "cloud",
      agentId: agent.agentId,
      apiKey: "key-respan-cursor-fixture",
    });
    await restored.wait();
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      2,
    );
    agent.close();
  } finally {
    await s.close();
  }
});

test(
  "actual local SDK preserves custom-tool schema, annotations, context and zero usage",
  { skip: !Cursor.JsonlLocalAgentStore },
  async () => {
    const { nativeLocalFixture } = await import("./native_fixture.mjs");
    const s = await setup();
    let observed;
    const tools = {
      lookup_fixture: {
        inputSchema: { type: "object" },
        outputSchema: { type: "object" },
        annotations: { readOnlyHint: true },
        execute(args, ctx) {
          observed = { args, ctx };
          return { value: 0, empty: "", enabled: false };
        },
      },
    };
    const local = await nativeLocalFixture(Cursor, tools);
    try {
      const agent = await Cursor.Agent.create(local.options);
      const run = await agent.send("fixture");
      const result = await run.wait();
      await agent[Symbol.asyncDispose]();
      assert.deepEqual(observed.ctx, {
        toolCallId: "native-local-call",
        sessionId: "native-child-session",
      });
      const agentSpan = s
        .spans()
        .find((v) => v.attributes["respan.entity.log_type"] === "agent");
      const metadata = JSON.parse(agentSpan.attributes["respan.metadata"]);
      assert.deepEqual(metadata.cursor_usage, result.usage);
      assert.equal(metadata.cursor_usage.totalTokens, 0);
      assert.deepEqual(metadata.cursor_tools[0].outputSchema, {
        type: "object",
      });
      assert.deepEqual(metadata.cursor_tools[0].annotations, {
        readOnlyHint: true,
      });
      const tool = s
        .spans()
        .find((v) => v.attributes["respan.entity.log_type"] === "tool");
      assert.equal(tool.attributes["gen_ai.tool.call.id"], "native-local-call");
    } finally {
      await local.close();
      await s.close();
    }
  },
);

test(
  "native local custom-tool falsy thrown error preserves tool failure",
  { skip: !Cursor.JsonlLocalAgentStore },
  async () => {
    const { nativeLocalFixture } = await import("./native_fixture.mjs");
    const s = await setup();
    const local = await nativeLocalFixture(Cursor, {
      lookup_fixture: {
        execute() {
          throw 0;
        },
      },
    });
    try {
      const agent = await Cursor.Agent.create(local.options);
      const run = await agent.send("fixture");
      assert.equal((await run.wait()).status, "error");
      await agent[Symbol.asyncDispose]();
      const tool = s
        .spans()
        .find((v) => v.attributes["respan.entity.log_type"] === "tool");
      assert.equal(tool.status.code, SpanStatusCode.ERROR);
      assert.equal(tool.attributes["traceloop.entity.output"], "0");
    } finally {
      await local.close();
      await s.close();
    }
  },
);

test("each actual native tool consults its own sampler under a DROP agent", async () => {
  const names = [];
  const s = await setup({
    sampler: {
      shouldSample(_c, _t, name) {
        names.push(name);
        return {
          decision: name.endsWith(".agent")
            ? SamplingDecision.NOT_RECORD
            : SamplingDecision.RECORD_AND_SAMPLED,
        };
      },
      toString() {
        return "child-sampler";
      },
    },
  });
  try {
    await stream(s);
    assert.ok(names.some((name) => name.endsWith(".tool")));
    assert.equal(
      s.spans().filter((v) => v.attributes["respan.entity.log_type"] === "tool")
        .length,
      1,
    );
  } finally {
    await s.close();
  }
});
test("private native tool topology and invocation ID survive payload denial", async () => {
  const s = await setup({ traceContent: false });
  try {
    const { result } = await stream(s);
    const tool = s
      .spans()
      .find((v) => v.attributes["respan.entity.log_type"] === "tool");
    assert.ok(tool);
    assert.equal(tool.attributes["gen_ai.tool.call.id"], "tool-" + result.id);
    assert.deepEqual(content(s.spans()), []);
  } finally {
    await s.close();
  }
});
test("reactivation reinstruments the same actual native agent", async () => {
  const s = await setup();
  try {
    const agent = await Cursor.Agent.create(s.fixture.options);
    await (await agent.send("first")).wait();
    s.plugin.deactivate();
    s.plugin.activate();
    await (await agent.send("second")).wait();
    assert.equal(
      s
        .spans()
        .filter((v) => v.attributes["respan.entity.log_type"] === "agent")
        .length,
      2,
    );
    agent.close();
  } finally {
    await s.close();
  }
});
