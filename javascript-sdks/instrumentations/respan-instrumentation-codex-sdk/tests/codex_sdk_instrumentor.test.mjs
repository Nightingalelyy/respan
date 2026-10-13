import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { context, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SimpleSpanProcessor,
  InMemorySpanExporter,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
const native = await import(
  process.env.CODEX_NATIVE_MODULE ?? "@openai/codex-sdk"
);
const { CodexSDKInstrumentor } = await import(
  process.env.CODEX_TARGET_MODULE ?? "../dist/index.js"
);
const latest = process.env.CODEX_NATIVE_VERSION !== "0.141.0";
let decision = SamplingDecision.RECORD_AND_SAMPLED;
let samples = [];
let starts = [];
let onStart;
const exporter = new InMemorySpanExporter();
const provider = new BasicTracerProvider({
  sampler: {
    shouldSample(_ctx, _trace, name, _kind, attrs) {
      samples.push({ name, attrs });
      return { decision };
    },
    toString() {
      return "native-test-sampler";
    },
  },
  spanProcessors: [
    {
      onStart(s, c) {
        starts.push(s);
        onStart?.(s, c);
      },
      onEnd() {},
      forceFlush: async () => {},
      shutdown: async () => {},
    },
    new SimpleSpanProcessor(exporter),
  ],
});
trace.setGlobalTracerProvider(provider);
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
const owners = [];
const fixture = new URL("./fixtures/controlled-codex.cjs", import.meta.url)
  .pathname;
function codex(scenario = "all", extra = {}) {
  return new native.Codex({
    codexPathOverride: fixture,
    env: {
      PATH: process.env.PATH,
      CODEX_FIXTURE_SCENARIO: scenario,
      CODEX_FIXTURE_CACHE_WRITE: latest ? "1" : "0",
      ...extra,
    },
  });
}
function thread(scenario = "all", extra = {}) {
  return codex(scenario, extra).startThread({
    model: "controlled-model",
    skipGitRepoCheck: true,
  });
}
async function activate(options = {}) {
  const owner = new CodexSDKInstrumentor({ sdkModule: native, ...options });
  owners.push(owner);
  await owner.activate();
  return owner;
}
function spans() {
  return exporter.getFinishedSpans();
}
function agent() {
  return spans().find(
    (s) => s.attributes["respan.entity.log_type"] === "agent",
  );
}
function metadata(s) {
  return JSON.parse(s.attributes["respan.metadata"]);
}
function assertContract() {
  for (const s of spans()) {
    for (const key of [
      "traceloop.span.kind",
      "respan.span.tools",
      "respan.span.tool_calls",
      "tools",
      "tool_calls",
      "model",
      "prompt_tokens",
      "completion_tokens",
      "status_code",
    ])
      assert.equal(s.attributes[key], undefined, key);
    assert.equal(s.attributes["gen_ai.tool.call.id"], undefined);
    assert.ok(
      !Object.keys(s.attributes).some(
        (k) =>
          k.startsWith("gen_ai.prompt") || k.startsWith("gen_ai.completion"),
      ),
    );
  }
  assert.ok(
    !spans().some((s) => s.attributes["respan.entity.log_type"] === "chat"),
  );
}
test.beforeEach(() => {
  exporter.reset();
  samples = [];
  starts = [];
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  onStart = undefined;
});
test.afterEach(() => {
  for (const owner of owners.splice(0)) owner.deactivate();
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.after(async () => {
  await provider.shutdown();
  context.disable();
  trace.disable();
});
test("native buffered turn maps all real items and aggregate usage without an invented LLM", async () => {
  await activate();
  const result = await thread().run("controlled input");
  assert.equal(result.items.length, 8);
  assert.equal(spans().length, 8);
  assert.equal(samples.length, 8);
  const a = agent();
  assert.deepEqual(metadata(a).codex_turn_usage, {
    input_tokens: 31,
    output_tokens: 9,
    cached_input_tokens: 4,
    ...(latest ? { cache_write_input_tokens: 2 } : {}),
    reasoning_output_tokens: 3,
  });
  assert.equal(metadata(a).codex_model, "controlled-model");
  assert.equal(
    a.attributes["respan.threads.thread_identifier"],
    "controlled-codex-thread",
  );
  assert.equal(a.attributes["gen_ai.request.model"], undefined);
  for (const s of spans().filter((s) => s !== a))
    assert.equal(s.parentSpanContext.spanId, a.spanContext().spanId);
  assertContract();
});
test("native MCP input/result retain false zero null definitions vector image and _meta", async () => {
  await activate();
  await thread().run("controlled");
  const m = spans().find((s) => metadata(s).codex_item_id === "mcp-1");
  const input = JSON.parse(m.attributes["traceloop.entity.input"]);
  const output = JSON.parse(m.attributes["traceloop.entity.output"]);
  assert.equal(input.name, "mcp__fixture__lookup");
  assert.equal(input.arguments.definitions.length, 82);
  assert.equal(input.arguments.false, false);
  assert.equal(input.arguments.zero, 0);
  assert.equal(input.arguments.empty, "");
  assert.equal(input.arguments.nullable, null);
  assert.equal(output.structured_content.vector.length, 5001);
  assert.equal(output.content[1].type, "image");
  assert.deepEqual(output._meta, latest ? { controlled: true } : undefined);
  assertContract();
});
test("native multimodal input retains typed local_image parts and order", async () => {
  await activate();
  await thread().run([
    { type: "text", text: "first" },
    { type: "local_image", path: "/controlled/image.png" },
    { type: "text", text: "last" },
  ]);
  assert.deepEqual(
    JSON.parse(agent().attributes["traceloop.entity.input"])[0].content,
    [
      { type: "text", text: "first" },
      { type: "local_image", path: "/controlled/image.png" },
      { type: "text", text: "last" },
    ],
  );
});
test("native streamed iterator retains identity and event objects", async () => {
  await activate();
  const t = thread();
  const result = await t.runStreamed("controlled");
  const iterator = result.events;
  assert.equal(iterator[Symbol.asyncIterator](), iterator);
  const first = await iterator.next();
  assert.equal(first.value.type, "thread.started");
  for await (const e of iterator) assert.ok(e.type);
  assert.equal(result.events, iterator);
  assert.equal(spans().length, 8);
  assertContract();
});
test("native buffered failure keeps observed tool then native throw and ERROR agent", async () => {
  await activate();
  await assert.rejects(
    thread("failure").run("controlled"),
    /controlled turn failure/,
  );
  assert.equal(spans().length, 2);
  assert.equal(agent().status.code, 2);
  assert.equal(agent().attributes["traceloop.entity.output"], undefined);
});
test("native streamed failure remains an event and does not become an adapter throw", async () => {
  await activate();
  const { events } = await thread("failure").runStreamed("controlled");
  const collected = [];
  for await (const e of events) collected.push(e);
  assert.equal(collected.at(-1).type, "turn.failed");
  assert.equal(agent().status.code, 2);
});
test("native nonfatal ErrorItem marks only actual task error", async () => {
  await activate();
  await thread().run("controlled");
  assert.equal(agent().status.code, 0);
  const err = spans().find((s) => metadata(s).codex_item_type === "error");
  assert.equal(err.status.code, 2);
});
for (const scenario of ["invalid-json", "process-error"])
  test(`native ${scenario} throw parity`, async () => {
    const bare = await thread(scenario)
      .run("controlled")
      .catch((e) => e);
    await activate();
    const wrapped = await thread(scenario)
      .run("controlled")
      .catch((e) => e);
    assert.equal(wrapped.message, bare.message);
    assert.equal(agent().status.code, 2);
  });
test("native stream error preserves event and final ERROR", async () => {
  await activate();
  const { events } = await thread("stream-error").runStreamed("controlled");
  for await (const _ of events) {
  }
  assert.equal(agent().status.code, 2);
});
test("native early iterator return drains the started agent exactly once", async () => {
  await activate();
  const { events } = await thread().runStreamed("controlled");
  await events.next();
  await events.return();
  await events.return();
  assert.equal(spans().length, 1);
  assert.equal(agent().status.code, 0);
});
test("native iterator return before first next closes admitted turn", async () => {
  await activate();
  const { events } = await thread().runStreamed("controlled");
  await events.return();
  assert.equal(spans().length, 1);
});
test("native AbortSignal rejection is preserved and closes actual span", async () => {
  await activate();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    thread().run("controlled", { signal: controller.signal }),
    (e) => e.name === "AbortError",
  );
  assert.equal(agent().status.code, 2);
});
test("native schema is passed unchanged and response text retained", async () => {
  const request = path.join(
    await fs.mkdtemp(path.join(os.tmpdir(), "t7-codex-schema-")),
    "request.json",
  );
  await activate();
  const schema = {
    type: "object",
    properties: { ok: { type: "boolean" } },
    required: ["ok"],
  };
  const r = await thread("structured", {
    CODEX_FIXTURE_REQUEST_PATH: request,
  }).run("structured", { outputSchema: schema });
  assert.equal(r.finalResponse, '{"ok":true,"value":0}');
  assert.deepEqual(
    JSON.parse(await fs.readFile(request, "utf8")).schema,
    schema,
  );
});
test("native resumed thread identity is retained", async () => {
  await activate();
  const c = codex();
  const first = c.startThread();
  await first.run("first");
  await c.resumeThread(first.id).run("second");
  assert.equal(
    spans().filter((s) => s.attributes["respan.entity.log_type"] === "agent")
      .length,
    2,
  );
  assert.ok(
    spans().every(
      (s) =>
        s.attributes["respan.entity.log_type"] !== "agent" ||
        s.attributes["respan.threads.thread_identifier"] === first.id,
    ),
  );
});
test("native empty finalResponse stays an explicit empty value", async () => {
  await activate();
  await thread("empty").run("controlled");
  assert.equal(agent().attributes["traceloop.entity.output"], '""');
});
for (const value of [SamplingDecision.NOT_RECORD, SamplingDecision.RECORD])
  test(`actual sampler ${value} avoids adapter payload getters`, async () => {
    decision = value;
    await activate();
    let reads = 0;
    const input = [
      {
        type: "text",
        get text() {
          reads++;
          return "native-input";
        },
        get extra() {
          throw new Error("adapter getter");
        },
      },
    ];
    await thread().run(input);
    assert.equal(reads, 1);
    assert.equal(spans().length, 0);
    assert.ok(
      !JSON.stringify(starts.map((s) => s.attributes)).includes(
        "controlled MCP result",
      ),
    );
    assert.equal(samples.length, 8);
  });
for (const lane of [
  "option",
  "respan-env",
  "traceloop-env",
  "context",
  "suppression",
  "parent",
])
  test(`${lane} privacy ceiling suppresses actual span payload`, async () => {
    if (lane === "respan-env") process.env.RESPAN_TRACE_CONTENT = "false";
    if (lane === "traceloop-env") process.env.TRACELOOP_TRACE_CONTENT = "false";
    await activate(lane === "option" ? { traceContent: false } : {});
    let ctx = context.active();
    if (lane === "context")
      ctx = ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
    if (lane === "suppression") ctx = suppressTracing(ctx);
    if (lane === "parent") {
      const p = trace.getTracer("test").startSpan("parent");
      p.setAttribute("allow_trace_content", false);
      ctx = trace.setSpan(ctx, p);
    }
    await context.with(ctx, () => thread().run("private input"));
    for (const s of spans()) {
      assert.equal(s.attributes["traceloop.entity.input"], undefined);
      assert.equal(s.attributes["traceloop.entity.output"], undefined);
    }
    if (lane !== "suppression")
      assert.equal(metadata(agent()).codex_turn_usage.cached_input_tokens, 4);
    else assert.equal(spans().length, 0);
  });
test("late actual parent false cannot be restored by true", async () => {
  await activate();
  const parent = trace.getTracer("test").startSpan("parent");
  const streamed = await context.with(
    trace.setSpan(context.active(), parent),
    () => thread().runStreamed("private input"),
  );
  await streamed.events.next();
  parent.setAttribute("allow_trace_content", false);
  await streamed.events.next();
  parent.setAttribute("allow_trace_content", true);
  for await (const _ of streamed.events) {
  }
  assert.equal(agent().attributes["traceloop.entity.input"], undefined);
  assert.ok(
    spans().every((s) => s.attributes["traceloop.entity.output"] === undefined),
  );
});
test("actual recording span blocks late payload attributes events and status", async () => {
  await activate({ traceContent: false });
  onStart = (s) => {
    if (s.name === "codex-agent")
      setTimeout(() => {
        s.setAttribute("private", ["secret"]);
        s.addEvent("secret", { private: "secret" });
        s.setStatus({ code: 2, message: "secret" });
        s.attributes["private-direct"] = "secret";
        s.events = [{ name: "secret" }];
        s.status = { code: 2, message: "secret" };
      }, 0);
  };
  await thread().run("controlled");
  assert.ok(
    !JSON.stringify(
      spans().map((s) => ({
        attrs: s.attributes,
        events: s.events,
        status: s.status,
      })),
    ).includes("secret"),
  );
});
test("deactivation drains original buffered call and restores public methods", async () => {
  const original = native.Thread.prototype.run;
  const owner = await activate();
  const pending = thread().run("controlled");
  owner.deactivate();
  await pending;
  assert.equal(native.Thread.prototype.run, original);
  assert.equal(spans().length, 8);
});
test("two owners apply most restrictive content and remaining owner recovers", async () => {
  const a = await activate();
  const b = await activate({ traceContent: false });
  await thread().run("first");
  assert.equal(agent().attributes["traceloop.entity.input"], undefined);
  b.deactivate();
  exporter.reset();
  await thread().run("second");
  assert.ok(agent().attributes["traceloop.entity.input"]);
  assert.equal(a.isActive(), true);
});
test("activation coalesces and pending cancellation allows later reactivation", async () => {
  const owner = new CodexSDKInstrumentor();
  owners.push(owner);
  const a = owner.activate();
  assert.equal(owner.activate(), a);
  owner.deactivate();
  await a;
  assert.equal(owner.isActive(), false);
  await owner.activate();
  assert.equal(owner.isActive(), true);
});
test("foreign method replacement is retained after deactivation", async () => {
  const original = native.Thread.prototype.run;
  const owner = await activate();
  const foreign = function () {
    return "foreign";
  };
  native.Thread.prototype.run = foreign;
  owner.deactivate();
  assert.equal(native.Thread.prototype.run, foreign);
  native.Thread.prototype.run = original;
});
test("captureItemSpans false emits only actual turn", async () => {
  await activate({ captureItemSpans: false });
  await thread().run("controlled");
  assert.equal(spans().length, 1);
});
test(
  "current native configOverrides threadSource and cyber program reach transport",
  { skip: !latest },
  async () => {
    const request = path.join(
      await fs.mkdtemp(path.join(os.tmpdir(), "t7-codex-current-")),
      "request.json",
    );
    await activate();
    const c = new native.Codex({
      codexPathOverride: fixture,
      configOverrides: ['controlled_flag="raw"'],
      env: {
        PATH: process.env.PATH,
        CODEX_FIXTURE_REQUEST_PATH: request,
        CODEX_FIXTURE_CACHE_WRITE: "1",
      },
    });
    await c
      .startThread({
        threadSource: "controlled-source",
        modelReasoningEffort: "ultra",
      })
      .run("controlled", { cyberAccessProgram: "standard" });
    const { args } = JSON.parse(await fs.readFile(request, "utf8"));
    assert.ok(args.includes('controlled_flag="raw"'));
    assert.ok(args.some((v) => v.includes("controlled-source")));
    assert.ok(args.some((v) => v.includes("standard")));
    assert.equal(
      metadata(agent()).codex_turn_usage.cache_write_input_tokens,
      2,
    );
  },
);

test("independently loaded native SDK prototypes have independent patch owners", async () => {
  const moduleURL =
    process.env.CODEX_NATIVE_MODULE ?? import.meta.resolve("@openai/codex-sdk");
  const second = await import(moduleURL + "?independent=1");
  assert.notEqual(second.Thread, native.Thread);
  const original = second.Thread.prototype.run;
  const first = await activate();
  const other = new CodexSDKInstrumentor({ sdkModule: second });
  owners.push(other);
  await other.activate();
  await new second.Codex({
    codexPathOverride: fixture,
    env: { PATH: process.env.PATH },
  })
    .startThread()
    .run("controlled");
  first.deactivate();
  assert.notEqual(second.Thread.prototype.run, original);
  other.deactivate();
  assert.equal(second.Thread.prototype.run, original);
  assert.equal(spans().length, 8);
});
test("late consuming context false vetoes captured content permanently", async () => {
  await activate();
  const stream = await thread().runStreamed("private initial input");
  await stream.events.next();
  await context.with(
    context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
    () => stream.events.next(),
  );
  for await (const _ of stream.events) {
  }
  assert.equal(agent().attributes["traceloop.entity.input"], undefined);
  assert.ok(
    spans().every((s) => s.attributes["traceloop.entity.output"] === undefined),
  );
});
test("canonical inherited metadata merges without flattened aliases", async () => {
  onStart = (s) =>
    s.setAttribute(
      "respan.metadata",
      JSON.stringify({ run_id: "controlled-run", nested: { zero: 0 } }),
    );
  await activate();
  await thread().run("controlled");
  assert.equal(metadata(agent()).run_id, "controlled-run");
  assert.deepEqual(metadata(agent()).nested, { zero: 0 });
  assert.ok(
    !Object.keys(agent().attributes).some((k) =>
      k.startsWith("respan.metadata."),
    ),
  );
});
test("nonconfigurable actual span attributes obey privacy finalization", async () => {
  onStart = (s) => {
    s.setAttribute("private-constructor", "secret constructor");
    Object.defineProperty(s, "attributes", {
      value: s.attributes,
      writable: true,
      configurable: false,
    });
  };
  await activate({ traceContent: false });
  await thread().run("controlled");
  assert.ok(
    spans().every((s) => s.attributes["private-constructor"] === undefined),
  );
});
