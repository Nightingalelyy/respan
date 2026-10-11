import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import OpenAI from "openai";
import { BraintrustInstrumentor } from "../dist/index.js";
process.env.BRAINTRUST_ENVIRONMENT_TYPE = "development";
process.env.BRAINTRUST_ENVIRONMENT_NAME = "fixture";
const bt = await import("braintrust");
let decision = SamplingDecision.RECORD_AND_SAMPLED;
let sampler = () => decision;
let starts = [],
  ended = [],
  sampled = [],
  requests = [];
let startHook;
const provider = new BasicTracerProvider({
  sampler: {
    shouldSample(ctx, traceId, name, kind, attrs) {
      sampled.push({ name, attrs });
      return { decision: sampler(ctx, name) };
    },
    toString() {
      return "native-fixture";
    },
  },
  spanProcessors: [
    {
      onStart(s) {
        starts.push(s);
        startHook?.(s);
      },
      onEnd(s) {
        ended.push(s);
      },
      forceFlush: async () => {},
      shutdown: async () => {},
    },
  ],
});
trace.setGlobalTracerProvider(provider);
const manager = new AsyncLocalStorageContextManager().enable();
context.setGlobalContextManager(manager);
let base, server;
const completion = {
  id: "fixture-chat",
  object: "chat.completion",
  created: 1,
  model: "fixture-model",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "fixture response",
        tool_calls: [
          {
            id: "call_fixture",
            type: "function",
            function: { name: "fixture_tool", arguments: '{"value":0}' },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 17 },
};
test.before(async () => {
  server = http.createServer(async (req, res) => {
    requests.push(req.url);
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
    res.setHeader("content-type", "application/json");
    if (req.url === "/api/apikey/login")
      return res.end(
        JSON.stringify({
          org_info: [{ id: "fixture-org", name: "fixture", api_url: base }],
        }),
      );
    if (req.url.startsWith("/api/project"))
      return res.end(
        JSON.stringify({ id: "fixture-project", name: "fixture", project: {} }),
      );
    if (body.model === "fixture-error") {
      res.statusCode = 400;
      return res.end(
        JSON.stringify({
          error: {
            message: "controlled provider error",
            type: "invalid_request_error",
          },
        }),
      );
    }
    if (body.stream) {
      res.setHeader("content-type", "text/event-stream");
      for (const chunk of [
        {
          id: "fixture-stream",
          object: "chat.completion.chunk",
          created: 1,
          model: "fixture-model",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "first" },
              finish_reason: null,
            },
          ],
        },
        {
          id: "fixture-stream",
          object: "chat.completion.chunk",
          created: 1,
          model: "fixture-model",
          choices: [
            { index: 0, delta: { content: " second" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 0, completion_tokens: 2, total_tokens: 9 },
        },
      ])
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      return res.end("data: [DONE]\n\n");
    }
    res.end(JSON.stringify(completion));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${server.address().port}`;
});
test.after(async () => {
  await new Promise((r) => server.close(r));
  await provider.shutdown();
  manager.disable();
});
test.beforeEach(() => {
  starts = [];
  ended = [];
  sampled = [];
  requests = [];
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  sampler = () => decision;
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
async function fixture(options, fn) {
  const instrumentor = new BraintrustInstrumentor(options);
  await instrumentor.activate();
  const logger = bt.initLogger({
    projectId: "00000000-0000-4000-8000-000000000001",
    projectName: "fixture",
    apiKey: "fixture-only",
    appUrl: base,
  });
  try {
    await fn(logger, instrumentor);
    await logger.flush();
    await instrumentor.flush();
  } finally {
    await instrumentor.deactivate();
  }
}
test("native logger/traced/log preserve values, partial flush, explicit usage and canonical metadata", async () => {
  const result = { value: 0 };
  await fixture({}, async (logger) => {
    assert.equal(
      await logger.traced(
        async (span) => {
          span.log({
            input: [{ role: "user", content: "input" }],
            metadata: {
              model: "vendor/model",
              marker: "native",
              nested: { value: false },
            },
            metrics: { prompt_tokens: 3, completion_tokens: 2, tokens: 17 },
            tags: ["a"],
            scores: { quality: 0 },
            expected: null,
          });
          await logger.flush();
          assert.equal(ended.length, 0);
          span.log({
            output: { role: "assistant", content: "output" },
            metadata: { second: true },
            tags: ["b"],
          });
          return result;
        },
        { name: "native-llm", type: "llm" },
      ),
      result,
    );
  });
  assert.equal(sampled.length, 1);
  assert.equal(ended.length, 1);
  const a = ended[0].attributes;
  assert.equal(a["llm.usage.total_tokens"], 17);
  assert.equal(a["gen_ai.system"], undefined);
  assert.equal(a["gen_ai.request.model"], "vendor/model");
  assert.equal(a.status_code, undefined);
  const m = JSON.parse(a["respan.metadata"]);
  assert.deepEqual(m.nested, { value: false });
  assert.equal(m.second, true);
  assert.equal(m.braintrust_expected, null);
  assert.deepEqual(m.braintrust_tags, ["a", "b"]);
  assert.ok(m.braintrust_span_id);
  assert.equal(
    Object.keys(a).some((k) => k.startsWith("respan.metadata.")),
    false,
  );
});
test("native parent and tool IDs map to real OTel-generated contexts", async () => {
  let parentId, childId;
  await fixture({}, async (logger) => {
    await logger.traced(
      async (parent) => {
        parentId = parent.spanId;
        await parent.traced(
          async (child) => {
            childId = child.spanId;
            child.log({
              input: { value: 0 },
              output: false,
              metadata: { tool_call_id: "call_actual" },
            });
          },
          { name: "native-tool", type: "tool" },
        );
      },
      { name: "native-parent", type: "task" },
    );
  });
  const p = ended.find((s) => s.name === "native-parent"),
    c = ended.find((s) => s.name === "native-tool");
  assert.equal(
    c.parentSpanContext?.spanId ?? c.parentSpanId,
    p.spanContext().spanId,
  );
  assert.equal(c.spanContext().traceId, p.spanContext().traceId);
  assert.equal(
    JSON.parse(p.attributes["respan.metadata"]).braintrust_span_id,
    parentId,
  );
  assert.equal(
    JSON.parse(c.attributes["respan.metadata"]).braintrust_span_id,
    childId,
  );
  assert.equal(c.attributes["gen_ai.tool.call.id"], "call_actual");
  assert.deepEqual(JSON.parse(c.attributes["traceloop.entity.input"]), {
    name: "native-tool",
    arguments: { value: 0 },
  });
  assert.equal(c.attributes["traceloop.entity.output"], "false");
});
for (const [label, choice] of [
  ["DROP", SamplingDecision.NOT_RECORD],
  ["RECORD_ONLY", SamplingDecision.RECORD],
])
  test(`${label} samples before native copies and skips bridge lazy resolution`, async () => {
    decision = choice;
    let reads = 0;
    await fixture({}, async (logger) => {
      await logger.traced(
        async (span) => {
          span.log({
            input: {
              get value() {
                reads++;
                assert.ok(sampled.length > 0);
                return "private";
              },
            },
            output: "private",
          });
        },
        { name: "native-denied", type: "llm" },
      );
    });
    assert.equal(sampled.length, 1);
    assert.equal(requests.length, 0);
    for (const s of ended)
      assert.equal(s.attributes["traceloop.entity.input"], undefined);
    assert.ok(reads >= 1);
  });
for (const lane of [
  "constructor",
  "respan-env",
  "traceloop-env",
  "context",
  "suppression",
  "ancestor",
])
  test(`native content ceiling ${lane} stays immutable`, async () => {
    if (lane === "respan-env") process.env.RESPAN_TRACE_CONTENT = "false";
    if (lane === "traceloop-env") process.env.TRACELOOP_TRACE_CONTENT = "false";
    const p = trace.getTracer("fixture").startSpan("external-parent");
    if (lane === "ancestor") p.setAttribute("allow_trace_content", false);
    let ctx = trace.setSpan(context.active(), p);
    if (lane === "context")
      ctx = ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
    if (lane === "suppression") ctx = suppressTracing(ctx);
    await context.with(ctx, () =>
      fixture({ traceContent: lane !== "constructor" }, async (logger) => {
        const s = logger.startSpan({ name: "native-private", type: "llm" });
        delete process.env.RESPAN_TRACE_CONTENT;
        delete process.env.TRACELOOP_TRACE_CONTENT;
        p.setAttribute("allow_trace_content", true);
        await context.with(
          context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
          async () => {
            s.log({ input: "private", output: "private" });
            s.end();
          },
        );
      }),
    );
    const s = ended.find((i) => i.name === "native-private");
    if (s) assert.equal(s.attributes["traceloop.entity.input"], undefined);
    p.end();
  });
test("actual native RecordingSpan guards late attributes/events/status and replacement", async () => {
  let reads = 0;
  await fixture({ traceContent: false }, async (logger) => {
    const native = logger.startSpan({ name: "late-private", type: "llm" });
    const actual = starts.at(-1);
    actual.setAttributes({
      get unknown() {
        reads++;
        return "private";
      },
      "gen_ai.usage.fiction": {
        get value() {
          reads++;
          return 1;
        },
      },
    });
    actual.attributes = { secret: "private", "gen_ai.usage.input_tokens": 0 };
    actual.events = [{ name: "private", attributes: { private: true } }];
    actual.status = { code: "private", message: "private" };
    actual.setStatus({
      get code() {
        reads++;
        return 2;
      },
      get message() {
        reads++;
        return "private";
      },
    });
    assert.throws(
      () =>
        Object.defineProperty(actual, "attributes", {
          value: { private: true },
        }),
      TypeError,
    );
    native.end();
  });
  assert.equal(reads, 0);
  assert.equal(ended[0].attributes.secret, undefined);
  assert.equal(ended[0].attributes["gen_ai.usage.fiction"], undefined);
  assert.deepEqual(ended[0].events, []);
  assert.equal(typeof ended[0].status.code, "number");
  assert.equal(ended[0].status.message, undefined);
});
test("late parent false stays sticky and exporter ambient suppression preserves admitted content", async () => {
  const p = trace.getTracer("fixture").startSpan("external");
  await context.with(trace.setSpan(context.active(), p), () =>
    fixture({}, async (logger) => {
      const s = logger.startSpan({ name: "late-veto", type: "llm" });
      p.setAttribute("allow_trace_content", false);
      s.log({ input: "private" });
      p.setAttribute("allow_trace_content", true);
      s.log({ output: "private" });
      s.end();
    }),
  );
  assert.equal(
    ended.find((s) => s.name === "late-veto").attributes[
      "traceloop.entity.input"
    ],
    undefined,
  );
  p.end();
  await fixture({}, async (logger, i) => {
    await logger.traced(
      async (s) => s.log({ input: "admitted", output: "admitted" }),
      { name: "ambient-export", type: "llm" },
    );
    await context.with(suppressTracing(context.active()), () => i.flush());
  });
  assert.equal(
    JSON.parse(
      ended.find((s) => s.name === "ambient-export").attributes[
        "traceloop.entity.output"
      ],
    ),
    "admitted",
  );
});
test("native child sampler runs independently of denied parent sampling", async () => {
  sampler = (_ctx, name) =>
    name === "sampled-child"
      ? SamplingDecision.RECORD_AND_SAMPLED
      : SamplingDecision.NOT_RECORD;
  await fixture({}, async (logger) => {
    await logger.traced(
      async (p) =>
        p.traced(async (c) => c.log({ output: "actual child" }), {
          name: "sampled-child",
          type: "task",
        }),
      { name: "dropped-parent", type: "task" },
    );
  });
  assert.equal(sampled.length, 2);
  assert.equal(ended.length, 1);
  assert.equal(
    ended[0].attributes["traceloop.entity.output"],
    '"actual child"',
  );
});
test("native errors preserve original error identity without invented completion or HTTP status", async () => {
  const error = new Error("controlled native failure");
  await fixture({}, async (logger) => {
    await assert.rejects(
      logger.traced(
        async () => {
          throw error;
        },
        { name: "native-error", type: "llm" },
      ),
      (e) => e === error,
    );
  });
  assert.equal(ended[0].status.code, 2);
  assert.equal(ended[0].attributes.status_code, undefined);
  assert.equal(ended[0].attributes["gen_ai.completion.0.content"], undefined);
});
test("80 messages,82 tools,multimodal output and5001 vectors stay complete at default128 attributes", async () => {
  const messages = Array.from({ length: 80 }, (_, i) => ({
    role: "user",
    content: [
      { type: "text", text: String(i) },
      {
        type: "image_url",
        image_url: { url: `data:image/png;base64,fixture${i}` },
      },
    ],
  }));
  const tools = Array.from({ length: 82 }, (_, i) => ({
    type: "function",
    function: { name: `tool_${i}`, parameters: { type: "object" } },
  }));
  const vector = Array.from({ length: 5001 }, (_, i) => i / 10);
  await fixture({}, async (logger) => {
    await logger.traced(
      async (s) =>
        s.log({
          input: messages,
          output: {
            role: "assistant",
            content: [
              { type: "text", text: "output" },
              {
                type: "image_url",
                image_url: { url: "data:image/png;base64,output" },
              },
            ],
            tool_calls: [
              {
                id: "actual_call",
                type: "function",
                function: { name: "tool_81", arguments: "{}" },
              },
            ],
          },
          metadata: { model: "fixture-model", provider: "openai", tools },
          metrics: { prompt_tokens: 0, completion_tokens: 0, tokens: 0 },
        }),
      { name: "large-llm", type: "llm" },
    );
    await logger.traced(
      async (s) =>
        s.log({
          input: ["embedding"],
          output: vector,
          metadata: { model: "fixture-embedding" },
        }),
      { name: "native-vector", type: "embedding" },
    );
  });
  const a = ended.find((s) => s.name === "large-llm").attributes;
  assert.deepEqual(JSON.parse(a["traceloop.entity.input"]), messages);
  assert.deepEqual(JSON.parse(a["llm.request.functions"]), tools);
  assert.equal(JSON.parse(a["traceloop.entity.output"]).content.length, 2);
  assert.equal(
    JSON.parse(a["gen_ai.completion.0.tool_calls"])[0].id,
    "actual_call",
  );
  assert.equal(a["llm.usage.total_tokens"], 0);
  assert.ok(Object.keys(a).length <= 128);
  assert.deepEqual(
    JSON.parse(
      ended.find((s) => s.name === "native-vector").attributes[
        "traceloop.entity.output"
      ],
    ),
    vector,
  );
});
for (const value of [false, 0, "", null, [], {}])
  test(`native task preserves ${JSON.stringify(value)} input/output`, async () => {
    await fixture({}, async (logger) => {
      await logger.traced(async (s) => s.log({ input: value, output: value }), {
        name: "native-scalar",
        type: "task",
      });
    });
    assert.deepEqual(
      JSON.parse(ended[0].attributes["traceloop.entity.input"]),
      value,
    );
    assert.deepEqual(
      JSON.parse(ended[0].attributes["traceloop.entity.output"]),
      value,
    );
    assert.equal(ended[0].attributes["llm.request.type"], undefined);
  });
test("real Braintrust/OpenAI HTTP wrapper preserves response,tools,SSE and controlled errors", async () => {
  await fixture({}, async (logger) => {
    const openai = bt.wrapOpenAI(
      new OpenAI({
        apiKey: "fixture-only",
        baseURL: `${base}/v1`,
        maxRetries: 0,
      }),
    );
    await logger.traced(
      async () => {
        const response = await openai.chat.completions.create({
          model: "fixture-model",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "multimodal" },
                {
                  type: "image_url",
                  image_url: { url: "data:image/png;base64,fixture" },
                },
              ],
            },
          ],
          tools: [
            {
              type: "function",
              function: {
                name: "fixture_tool",
                parameters: { type: "object" },
              },
            },
          ],
        });
        assert.deepEqual(response, completion);
        const stream = await openai.chat.completions.create({
          model: "fixture-model",
          messages: [{ role: "user", content: "stream" }],
          stream: true,
        });
        assert.equal(typeof stream.controller.abort, "function");
        const chunks = [];
        for await (const chunk of stream) chunks.push(chunk);
        assert.equal(chunks.length, 2);
        await assert.rejects(
          openai.chat.completions.create({
            model: "fixture-error",
            messages: [{ role: "user", content: "error" }],
          }),
          (e) => e.status === 400,
        );
      },
      { name: "wrapper-parent", type: "task" },
    );
  });
  const llms = ended.filter(
    (s) => s.attributes["respan.entity.log_type"] === "chat",
  );
  assert.equal(llms.length, 3);
  const normal = llms.find(
    (s) => s.attributes["llm.usage.total_tokens"] === 17,
  );
  assert.equal(
    JSON.parse(normal.attributes["traceloop.entity.input"])[0].content.length,
    2,
  );
  assert.equal(
    JSON.parse(normal.attributes["gen_ai.completion.0.tool_calls"])[0].id,
    "call_fixture",
  );
  assert.ok(sampled.some((s) => s.name === "Chat Completion"));
  assert.ok(llms.some((s) => s.status.code === 2));
});
test("activation coalesces,cancels,reactivates;owners coexist and foreign override survives", async () => {
  const a = new BraintrustInstrumentor(),
    b = new BraintrustInstrumentor();
  const activation = a.activate();
  assert.equal(a.activate(), activation);
  await a.deactivate();
  assert.equal(a.isActive(), false);
  await a.activate();
  await b.activate();
  await a.deactivate();
  assert.equal(b.isActive(), true);
  const state = bt._internalGetGlobalState();
  const foreign = {
    log() {},
    flush: async () => {},
    pendingFlushBytes: () => 0,
    flushBackpressureBytes: () => Infinity,
    setMaskingFunction() {},
  };
  state.setOverrideBgLogger(foreign);
  await b.deactivate();
  assert.equal(state.bgLogger(), foreign);
  state.setOverrideBgLogger(null);
});
test("deactivation drains in-flight native span without early invented completion", async () => {
  const i = new BraintrustInstrumentor();
  await i.activate();
  const l = bt.initLogger({
    projectId: "00000000-0000-4000-8000-000000000001",
    projectName: "fixture",
    apiKey: "fixture-only",
    appUrl: base,
  });
  const s = l.startSpan({ name: "inflight", type: "task" });
  s.log({ input: "before" });
  await i.deactivate();
  assert.equal(ended.length, 0);
  s.log({ output: "after" });
  s.end();
  await l.flush();
  assert.equal(ended.length, 1);
  assert.equal(ended[0].attributes["traceloop.entity.output"], '"after"');
});
test("initial native identity/model metadata is visible to the actual sampler", async () => {
  await fixture({}, async (logger) => {
    await logger.traced(async (s) => s.log({ output: "actual" }), {
      name: "initial-model",
      type: "llm",
      event: {
        metadata: { model: "initial-model-id", provider: "literal-provider" },
      },
    });
  });
  const a = sampled.find((s) => s.name === "initial-model").attrs;
  assert.equal(a["gen_ai.request.model"], "initial-model-id");
  assert.equal(a["gen_ai.system"], "literal-provider");
  assert.ok(JSON.parse(a["respan.metadata"]).braintrust_span_id);
});

test("native privacy scrubs a nonconfigurable RecordingSpan replacement installed onStart", async () => {
  let reads = 0;
  startHook = (s) =>
    Object.defineProperty(s, "attributes", {
      configurable: false,
      writable: true,
      enumerable: true,
      value: {
        get secret() {
          reads++;
          return "private";
        },
        "gen_ai.usage.fiction": {
          get value() {
            reads++;
            return "private";
          },
        },
      },
    });
  try {
    await fixture({ traceContent: false }, async (logger) => {
      await logger.traced(
        async (s) => s.log({ input: "private", output: "private" }),
        { name: "locked-native", type: "llm" },
      );
    });
  } finally {
    startHook = undefined;
  }
  assert.equal(reads, 0);
  assert.equal(ended[0].attributes.secret, undefined);
  assert.equal(ended[0].attributes["gen_ai.usage.fiction"], undefined);
});

test("native preactivation and active masking apply to merged payload fields and survive restoration", async () => {
  const seen = [];
  const mask = (value) => {
    seen.push(value);
    return typeof value === "string" ? "redacted" : value;
  };
  bt.setMaskingFunction(mask);
  try {
    await fixture({}, async (logger) => {
      const native = logger.startSpan({
        name: "masked-native",
        type: "task",
        event: { input: "private" },
      });
      await logger.flush();
      bt.setMaskingFunction(mask);
      native.log({ output: "private", metadata: { marker: "preserved" } });
      native.end();
    });
    assert.equal(ended[0].attributes["traceloop.entity.input"], '"redacted"');
    assert.equal(ended[0].attributes["traceloop.entity.output"], '"redacted"');
    assert.equal(
      JSON.parse(ended[0].attributes["respan.metadata"]).marker,
      "preserved",
    );
    assert.ok(
      seen.every(
        (value) =>
          !value || typeof value !== "object" || value.span_id === undefined,
      ),
    );
  } finally {
    bt.setMaskingFunction(null);
  }
});

test("denied native content never executes bridge-owned masking callbacks", async () => {
  let calls = 0;
  bt.setMaskingFunction((value) => {
    calls++;
    return value;
  });
  try {
    await fixture({ traceContent: false }, async (logger) => {
      await logger.traced(
        async (span) => span.log({ input: "private", output: "private" }),
        { name: "private-mask", type: "task" },
      );
    });
  } finally {
    bt.setMaskingFunction(null);
  }
  assert.equal(calls, 0);
});

test("peer: late canonical context on actual RecordingSpan imposes sticky ceiling", async () => {
  await fixture({}, async (logger) => {
    const native = logger.startSpan({ name: "peer-late-context", type: "llm" });
    const actual = starts.at(-1);
    context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
      () => actual.setAttribute("traceloop.entity.input", "late private write"),
    );
    native.log({ input: "native private", output: "native private" });
    native.end();
  });
  const span = ended.find((s) => s.name === "peer-late-context");
  console.log(
    JSON.stringify({
      peer: "late-context",
      input: span.attributes["traceloop.entity.input"],
      output: span.attributes["traceloop.entity.output"],
    }),
  );
  assert.equal(span.attributes["traceloop.entity.input"], undefined);
});

test("peer: direct false on guarded native parent is observed by its child", async () => {
  await fixture({}, async (logger) => {
    await logger.traced(
      async (parent) => {
        const actual = starts.at(-1);
        actual.attributes.allow_trace_content = false;
        await parent.traced(
          async (child) =>
            child.log({ input: "native private", output: "native private" }),
          { name: "peer-parent-child", type: "llm" },
        );
      },
      { name: "peer-parent", type: "task" },
    );
  });
  const span = ended.find((s) => s.name === "peer-parent-child");
  console.log(
    JSON.stringify({
      peer: "guarded-parent",
      input: span.attributes["traceloop.entity.input"],
      output: span.attributes["traceloop.entity.output"],
    }),
  );
  assert.equal(span.attributes["traceloop.entity.input"], undefined);
});

test("peer: second active owner with traceContent false is a ceiling", async () => {
  const permissive = new BraintrustInstrumentor();
  await permissive.activate();
  try {
    await fixture({ traceContent: false }, async (logger) => {
      await logger.traced(
        async (span) =>
          span.log({ input: "native private", output: "native private" }),
        { name: "peer-two-owners", type: "llm" },
      );
    });
    const span = ended.find((s) => s.name === "peer-two-owners");
    console.log(
      JSON.stringify({
        peer: "multiple-owners",
        input: span.attributes["traceloop.entity.input"],
        output: span.attributes["traceloop.entity.output"],
      }),
    );
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
  } finally {
    await permissive.deactivate();
  }
});
