import assert from "node:assert/strict";
import test from "node:test";
import Together from "together-ai";
import { context, trace, TraceFlags } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SimpleSpanProcessor,
  InMemorySpanExporter,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import {
  TogetherAIInstrumentor,
  TogetherAIInstrumentation,
} from "../dist/index.js";

const chat = (
  message = { role: "assistant", content: "ANSWER" },
  usage = { prompt_tokens: 4, completion_tokens: 5, total_tokens: 17 },
) => ({
  id: "response-actual",
  model: "response-model",
  choices: [{ index: 0, finish_reason: "stop", message }],
  usage,
});
const environments = [];
test.afterEach(async () => {
  for (const e of environments.splice(0)) await e.done();
});
function setup(
  options = {},
  decision = SamplingDecision.RECORD_AND_SAMPLED,
  onStart = () => {},
) {
  trace.disable();
  context.disable();
  context.setGlobalContextManager(
    new AsyncLocalStorageContextManager().enable(),
  );
  const exporter = new InMemorySpanExporter(),
    native = [];
  const sampler = {
    shouldSample() {
      return { decision };
    },
    toString() {
      return "fixture";
    },
  };
  const provider = new BasicTracerProvider({
    sampler,
    spanProcessors: [
      {
        onStart,
        onEnd(s) {
          native.push(s);
        },
        async shutdown() {},
        async forceFlush() {},
      },
      new SimpleSpanProcessor(exporter),
    ],
  });
  trace.setGlobalTracerProvider(provider);
  const plugin = new TogetherAIInstrumentor(options);
  const env = {
    exporter,
    native,
    provider,
    plugin,
    sampler,
    async done() {
      plugin.deactivate();
      await provider.shutdown();
      trace.disable();
      context.disable();
    },
  };
  environments.push(env);
  return env;
}
function client(response = chat(), capture = []) {
  return new Together({
    apiKey: "fixture-only",
    maxRetries: 0,
    fetch: async (url, opts) => {
      capture.push({
        url,
        body: typeof opts.body === "string" ? JSON.parse(opts.body) : opts.body,
      });
      return response instanceof Response
        ? response
        : Response.json(
            typeof response === "function"
              ? await response(url, opts)
              : response,
          );
    },
  });
}
const request = {
  model: "request-model",
  messages: [{ role: "user", content: "PROMPT" }],
};
test("native public aliases and idempotent shared activation", async () => {
  assert.equal(TogetherAIInstrumentation, TogetherAIInstrumentor);
  const e = setup();
  await Promise.all([e.plugin.activate(), e.plugin.activate()]);
  assert.equal(e.plugin.isActive(), true);
  await client().chat.completions.create(request);
  assert.equal(e.native.length, 1);
  await e.done();
});
test("native chat keeps actual response, input, output, model and explicit total", async () => {
  const e = setup();
  await e.plugin.activate();
  const c = client();
  const response = await c.chat.completions.create(request);
  assert.equal(response.choices[0].message.content, "ANSWER");
  const a = e.native[0].attributes;
  assert.equal(a["gen_ai.request.model"], "request-model");
  assert.equal(a["gen_ai.response.model"], "response-model");
  assert.equal(a["llm.usage.total_tokens"], 17);
  assert.equal(a["gen_ai.usage.input_tokens"], 4);
  assert.equal(JSON.parse(a["traceloop.entity.input"])[0].content, "PROMPT");
  await e.done();
});
test("native incomplete usage does not invent totals", async () => {
  const e = setup();
  await e.plugin.activate();
  await client(
    chat(undefined, { prompt_tokens: 0, completion_tokens: 5 }),
  ).chat.completions.create(request);
  assert.equal(e.native[0].attributes["gen_ai.usage.input_tokens"], 0);
  assert.equal(e.native[0].attributes["llm.usage.total_tokens"], undefined);
  await e.done();
});
test("native histories are context without fabricated executions", async () => {
  const e = setup();
  await e.plugin.activate();
  await client().chat.completions.create({
    ...request,
    messages: [
      ...request.messages,
      {
        role: "assistant",
        tool_calls: [
          {
            id: "old-call",
            type: "function",
            function: { name: "old-tool", arguments: "{}" },
          },
        ],
      },
      { role: "tool", tool_call_id: "old-call", content: "old result" },
    ],
  });
  assert.equal(e.native.length, 1);
  assert.equal(
    JSON.parse(e.native[0].attributes["traceloop.entity.input"])[2]
      .tool_call_id,
    "old-call",
  );
  assert.equal(
    e.native[0].attributes["gen_ai.completion.0.tool_calls"],
    undefined,
  );
  await e.done();
});
test("native current multimodal and reasoning stay distinct", async () => {
  const e = setup();
  await e.plugin.activate();
  await client(
    chat({
      role: "assistant",
      content: "answer",
      reasoning: "reason",
      reasoning_content: "reason2",
    }),
  ).chat.completions.create({
    ...request,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "prompt" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,fixture" },
          },
          {
            type: "input_audio",
            input_audio: { data: "fixture", format: "wav" },
          },
        ],
      },
    ],
  });
  const a = e.native[0].attributes;
  assert.equal(JSON.parse(a["traceloop.entity.input"])[0].content.length, 3);
  const output = JSON.parse(a["traceloop.entity.output"])[0];
  assert.equal(output.reasoning, "reason");
  assert.equal(output.reasoning_content, "reason2");
  assert.equal(a["gen_ai.completion.0.content"], "answer");
  await e.done();
});
test("native current tool IDs and full definitions survive default attribute budget", async () => {
  const e = setup();
  await e.plugin.activate();
  const tools = Array.from({ length: 80 }, (_, i) => ({
    type: "function",
    function: {
      name: `tool_${i}`,
      parameters: {
        type: "object",
        properties: { x: { type: "integer" } },
        additionalProperties: false,
      },
    },
  }));
  const messages = Array.from({ length: 80 }, (_, i) => ({
    role: "user",
    content: `message-${i}`,
  }));
  await client(
    chat({
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "new-call",
          type: "function",
          function: { name: "tool_79", arguments: '{"x":0}' },
        },
      ],
    }),
  ).chat.completions.create({ ...request, messages, tools });
  const a = e.native[0].attributes;
  assert.equal(JSON.parse(a["traceloop.entity.input"]).length, 80);
  assert.equal(JSON.parse(a["llm.request.functions"]).length, 80);
  assert.equal(
    JSON.parse(a["gen_ai.completion.0.tool_calls"])[0].id,
    "new-call",
  );
  assert.equal(a["llm.usage.total_tokens"], 17);
  await e.done();
});
test("native APIPromise withResponse keeps result and helper identities", async () => {
  const e = setup();
  await e.plugin.activate();
  const pending = client().chat.completions.create(request);
  assert.equal(typeof pending.asResponse, "function");
  assert.equal(typeof pending.withResponse, "function");
  const response = await pending.withResponse();
  assert.equal(response.data.choices[0].message.content, "ANSWER");
  assert.ok(response.response instanceof Response);
  assert.equal(e.native[0].attributes["llm.usage.total_tokens"], 17);
  assert.equal(
    JSON.parse(e.native[0].attributes["traceloop.entity.output"])[0].content,
    "ANSWER",
  );
  assert.equal(e.native.length, 1);
  await pending;
  assert.equal(e.native.length, 1);
  await e.done();
});
test("native asResponse leaves body unread and records no invented completion", async () => {
  const e = setup();
  await e.plugin.activate();
  const response = await client().chat.completions.create(request).asResponse();
  assert.equal(response.bodyUsed, false);
  assert.equal((await response.json()).choices[0].message.content, "ANSWER");
  assert.equal(e.native[0].attributes["traceloop.entity.output"], undefined);
  await e.done();
});
const sse = (values) =>
  new Response(
    values.map((v) => `data: ${JSON.stringify(v)}\n\n`).join("") +
      "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } },
  );
test("native SSE stream preserves controller, chunks, independent choices and reasoning", async () => {
  const chunks = [
    {
      id: "stream-id",
      model: "stream-model",
      choices: [
        { index: 0, delta: { content: "A", reasoning: "R" } },
        { index: 1, delta: { content: "B" } },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: { content: "C", reasoning_content: "R2" },
          finish_reason: "stop",
        },
        { index: 1, delta: { content: "D" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 5, total_tokens: 17 },
    },
  ];
  const e = setup();
  await e.plugin.activate();
  const stream = await client(sse(chunks)).chat.completions.create({
    ...request,
    stream: true,
  });
  assert.ok(stream.controller instanceof AbortController);
  const native = [];
  for await (const chunk of stream) native.push(chunk);
  assert.deepEqual(native, chunks);
  const output = JSON.parse(e.native[0].attributes["traceloop.entity.output"]);
  assert.equal(output[0].content, "AC");
  assert.equal(output[0].reasoning, "R");
  assert.equal(output[1].content, "BD");
  await e.done();
});
test("native SSE break ends exactly once without replacing iterator", async () => {
  const e = setup();
  await e.plugin.activate();
  const stream = await client(
    sse([
      { choices: [{ index: 0, delta: { content: "first" } }] },
      { choices: [{ index: 0, delta: { content: "second" } }] },
    ]),
  ).chat.completions.create({ ...request, stream: true });
  for await (const chunk of stream) {
    assert.ok(chunk);
    break;
  }
  assert.equal(e.native.length, 1);
  assert.equal(
    JSON.parse(e.native[0].attributes["traceloop.entity.output"])[0].content,
    "first",
  );
  await e.done();
});
test("native HTTP error preserves status and does not invent output", async () => {
  const e = setup();
  await e.plugin.activate();
  await assert.rejects(
    client(
      Response.json(
        { error: { message: "controlled", type: "fixture" } },
        { status: 429 },
      ),
    ).chat.completions.create(request),
    (err) => err.status === 429,
  );
  assert.equal(e.native[0].status.code, 2);
  assert.equal(e.native[0].attributes.status_code, 429);
  assert.equal(e.native[0].attributes["traceloop.entity.output"], undefined);
  await e.done();
});
for (const [label, options] of [
  ["constructor", { traceContent: false }],
  ["input lane", { recordInputs: false }],
  ["output lane", { recordOutputs: false }],
])
  test(`native ${label} privacy ceiling`, async () => {
    const e = setup(options);
    await e.plugin.activate();
    await context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
      () => client().chat.completions.create(request),
    );
    const a = e.native[0].attributes;
    assert.equal(
      a["traceloop.entity.input"] === undefined,
      options.traceContent === false || options.recordInputs === false,
    );
    assert.equal(
      a["traceloop.entity.output"] === undefined,
      options.traceContent === false || options.recordOutputs === false,
    );
    await e.done();
  });
for (const [label, decision] of [
  ["DROP", SamplingDecision.NOT_RECORD],
  ["RECORD_ONLY", SamplingDecision.RECORD],
])
  test(`actual ${label} sampler precedes bridge traversal`, async () => {
    let calls = 0,
      atSample;
    const e = setup({}, decision);
    e.sampler.shouldSample = () => {
      atSample = calls;
      return { decision };
    };
    await e.plugin.activate();
    const body = {
      ...request,
      messages: [
        {
          role: "user",
          content: {
            toJSON() {
              calls++;
              return "native payload";
            },
          },
        },
      ],
    };
    await client().chat.completions.create(body);
    assert.equal(atSample, 0);
    assert.equal(calls, 1);
    for (const s of e.native)
      assert.equal(s.attributes["traceloop.entity.input"], undefined);
    assert.equal(e.exporter.getFinishedSpans().length, 0);
    await e.done();
  });
test("canonical context and suppression are immutable capture ceilings", async () => {
  const e = setup();
  await e.plugin.activate();
  await context.with(
    context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
    () => client().chat.completions.create(request),
  );
  assert.equal(e.native[0].attributes["traceloop.entity.input"], undefined);
  await context.with(suppressTracing(context.active()), () =>
    client().chat.completions.create(request),
  );
  assert.equal(
    e.native.at(-1).attributes["traceloop.entity.output"],
    undefined,
  );
  await e.done();
});
test("late ancestor veto guards the actual span attrs, events and status", async () => {
  let actual;
  const e = setup({}, SamplingDecision.RECORD_AND_SAMPLED, (s) => {
    actual = s;
  });
  await e.plugin.activate();
  const parent = trace.getTracer("test").startSpan("parent");
  await context.with(trace.setSpan(context.active(), parent), () =>
    client(() => {
      parent.setAttribute("allow_trace_content", false);
      actual.setAttribute("secret", "PRIVATE");
      actual.addEvent("PRIVATE", { secret: "PRIVATE" });
      actual.setStatus({ code: 2, message: "PRIVATE" });
      return chat();
    }).chat.completions.create(request),
  );
  const s = e.native.find((s) => s.name.includes("together"));
  assert.equal(s.attributes.secret, undefined);
  assert.equal(s.attributes["traceloop.entity.input"], undefined);
  assert.equal(s.events.length, 0);
  assert.equal(s.status.message, undefined);
  parent.end();
  await e.done();
});
test("native embedding vector is complete and generic resources still work", async () => {
  const e = setup();
  await e.plugin.activate();
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  await client({
    model: "embed",
    data: [{ index: 0, embedding: vector }],
    usage: { prompt_tokens: 0, total_tokens: 0 },
  }).embeddings.create({ model: "embed", input: ["a", "b"] });
  const a = e.native[0].attributes;
  assert.equal(
    JSON.parse(a["traceloop.entity.output"])[0].embedding.length,
    5001,
  );
  assert.equal(a["gen_ai.usage.input_tokens"], 0);
  await client({ data: [{ b64_json: "fixture" }] }).images.generate({
    model: "image",
    prompt: "image prompt",
  });
  await client({ results: [{ index: 0, relevance_score: 0.5 }] }).rerank.create(
    { model: "rank", query: "q", documents: ["d"] },
  );
  assert.equal(e.native.length, 3);
  await e.done();
});
test("multiple owners retain restrictive options and foreign patches", async () => {
  const e = setup();
  const second = new TogetherAIInstrumentor({ traceContent: false });
  await e.plugin.activate();
  await second.activate();
  await client().chat.completions.create(request);
  assert.equal(e.native[0].attributes["traceloop.entity.input"], undefined);
  e.plugin.deactivate();
  assert.equal(second.isActive(), true);
  await client().chat.completions.create(request);
  assert.equal(e.native.length, 2);
  second.deactivate();
  await client().chat.completions.create(request);
  assert.equal(e.native.length, 2);
  await e.done();
});
test("deactivation drains already started native calls and supports reactivation", async () => {
  const e = setup();
  await e.plugin.activate();
  let release;
  const pending = client(
    async () =>
      await new Promise((resolve) => {
        release = () => resolve(chat());
      }),
  ).chat.completions.create(request);
  const awaited = pending.then((x) => x);
  while (!release) await new Promise((resolve) => setImmediate(resolve));
  e.plugin.deactivate();
  release();
  await awaited;
  assert.equal(e.native.length, 1);
  await e.plugin.activate();
  await client().chat.completions.create(request);
  assert.equal(e.native.length, 2);
  await e.done();
});

for (const path of ["catch-success", "finally-success", "finally-error"])
  test(`native peer helper ${path}`, async () => {
    const e = setup();
    await e.plugin.activate();
    const pending = client(
      path === "finally-error"
        ? Response.json({ error: { message: "controlled" } }, { status: 429 })
        : chat(),
    ).chat.completions.create(request);
    let callbacks = 0;
    if (path === "catch-success")
      assert.equal(
        (await pending.catch(() => null)).choices[0].message.content,
        "ANSWER",
      );
    else if (path === "finally-error")
      await assert.rejects(
        pending.finally(() => {
          callbacks++;
        }),
        (error) => error.status === 429,
      );
    else
      assert.equal(
        (
          await pending.finally(() => {
            callbacks++;
          })
        ).choices[0].message.content,
        "ANSWER",
      );
    assert.equal(e.native.length, 1);
    assert.equal(e.native[0].status.code, path === "finally-error" ? 2 : 0);
    if (path !== "finally-error")
      assert.equal(e.native[0].attributes["llm.usage.total_tokens"], 17);
    await e.done();
  });
test("native strict function definition remains canonical", async () => {
  const e = setup();
  await e.plugin.activate();
  await client().chat.completions.create({
    ...request,
    tools: [
      {
        type: "function",
        function: {
          name: "f",
          strict: true,
          parameters: {
            type: "object",
            properties: { x: { type: "boolean" } },
          },
        },
      },
    ],
  });
  assert.equal(
    JSON.parse(e.native[0].attributes["llm.request.functions"])[0].function
      .strict,
    true,
  );
  await e.done();
});
test("native locked writable actual attrs replacement is scrubbed under false", async () => {
  let actual;
  const e = setup(
    { traceContent: false },
    SamplingDecision.RECORD_AND_SAMPLED,
    (span) => {
      actual = span;
      Object.defineProperty(span, "attributes", {
        value: span.attributes,
        configurable: false,
        writable: true,
        enumerable: true,
      });
    },
  );
  await e.plugin.activate();
  await client(() => {
    actual.attributes = {
      "traceloop.entity.input": "PRIVATE_LOCKED",
      unknown: "PRIVATE_LOCKED",
    };
    return chat();
  }).chat.completions.create(request);
  assert.ok(!JSON.stringify(e.native[0].attributes).includes("PRIVATE_LOCKED"));
  await e.done();
});
test("late native source suppression veto survives later admitted context", async () => {
  let actual;
  const e = setup({}, SamplingDecision.RECORD_AND_SAMPLED, (span) => {
    actual = span;
  });
  await e.plugin.activate();
  await client(() => {
    context.with(suppressTracing(context.active()), () =>
      actual.setAttribute("private", "PRIVATE_SUPPRESSED"),
    );
    return chat();
  }).chat.completions.create(request);
  const a = e.native[0].attributes;
  assert.equal(a.private, undefined);
  assert.equal(a["traceloop.entity.input"], undefined);
  assert.equal(a["traceloop.entity.output"], undefined);
  await e.done();
});
