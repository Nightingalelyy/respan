import assert from "node:assert/strict";
import test from "node:test";
import * as legacy from "@google-cloud/vertexai";
import * as genai from "@google/genai";
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
import {
  VertexAIInstrumentor,
  buildGenerateContentAttrs,
} from "../dist/index.js";
const originalFetch = globalThis.fetch;
const envs = [];
test.afterEach(async () => {
  for (const e of envs.splice(0)) await e.done();
  globalThis.fetch = originalFetch;
});
const authClient = {
  async getAccessToken() {
    return { token: "fixture-only" };
  },
  async getRequestHeaders() {
    return new Headers({ Authorization: "Bearer fixture-only" });
  },
};
function response(
  parts = [{ text: "ANSWER" }],
  usage = { promptTokenCount: 4, candidatesTokenCount: 5, totalTokenCount: 17 },
) {
  return {
    candidates: [
      { index: 0, finishReason: "STOP", content: { role: "model", parts } },
    ],
    usageMetadata: usage,
    modelVersion: "native-response-model",
    responseId: "actual-response",
  };
}
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
    native = [],
    requests = [];
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
  const plugin = new VertexAIInstrumentor({
    sdkModule: legacy,
    genaiModule: genai,
    ...options,
  });
  const e = {
    exporter,
    native,
    requests,
    sampler,
    provider,
    plugin,
    async done() {
      plugin.deactivate();
      await provider.shutdown();
      trace.disable();
      context.disable();
    },
  };
  envs.push(e);
  return e;
}
function transport(e, res = response()) {
  globalThis.fetch = async (url, opts) => {
    const body =
      typeof opts.body === "string" ? JSON.parse(opts.body) : opts.body;
    e.requests.push({ url: String(url), body });
    const value =
      typeof res === "function" ? await res(String(url), body) : res;
    return value instanceof Response ? value : Response.json(value);
  };
}
function modern(vertexai = true) {
  return new genai.GoogleGenAI(
    vertexai
      ? {
          vertexai: true,
          project: "fixture-project",
          location: "us-central1",
          googleAuthOptions: { authClient },
        }
      : { apiKey: "fixture-only" },
  );
}
function old(preview = false) {
  const c = new legacy.VertexAI({
    project: "fixture-project",
    location: "us-central1",
    googleAuthOptions: { authClient },
  });
  return (preview ? c.preview : c).getGenerativeModel({
    model: "gemini-fixture",
  });
}
const request = { model: "gemini-fixture", contents: "PROMPT" };
test("native legacy GA and Preview current generation both emit", async () => {
  const e = setup();
  transport(e);
  await e.plugin.activate();
  await old().generateContent("PROMPT");
  await old(true).generateContent("PREVIEW");
  assert.equal(e.native.length, 2);
  assert.equal(e.requests.length, 2);
  for (const s of e.native)
    assert.equal(s.attributes["llm.usage.total_tokens"], 17);
});
test("modern Vertex generation keeps public response helpers and explicit provider usage", async () => {
  const e = setup();
  transport(e);
  await e.plugin.activate();
  const r = await modern().models.generateContent(request);
  assert.equal(r.text, "ANSWER");
  assert.equal(r.responseId, "actual-response");
  const a = e.native[0].attributes;
  assert.equal(a["gen_ai.system"], "google");
  assert.equal(a["gen_ai.request.model"], "gemini-fixture");
  assert.equal(a["llm.usage.total_tokens"], 17);
  assert.equal(a["gen_ai.response.model"], "native-response-model");
  assert.equal(JSON.parse(a["traceloop.entity.input"])[0].content, "PROMPT");
});
test("modern Gemini Developer backend is untouched", async () => {
  const e = setup();
  transport(e);
  await e.plugin.activate();
  assert.equal(
    (await modern(false).models.generateContent(request)).text,
    "ANSWER",
  );
  assert.equal(e.native.length, 0);
});
test("modern typed parts preserve image, inline audio, reasoning and signature", async () => {
  const e = setup();
  transport(
    e,
    response([
      { text: "thought", thought: true, thoughtSignature: "actual-signature" },
      { text: "answer" },
      { inlineData: { mimeType: "image/png", data: "fixture" } },
    ]),
  );
  await e.plugin.activate();
  await modern().models.generateContent({
    ...request,
    contents: [
      {
        role: "user",
        parts: [
          { text: "question" },
          { inlineData: { mimeType: "audio/wav", data: "fixture-audio" } },
        ],
      },
    ],
  });
  const a = e.native[0].attributes;
  assert.equal(
    JSON.parse(a["traceloop.entity.input"])[0].content[1].inline_data.data,
    "fixture-audio",
  );
  const output = JSON.parse(a["traceloop.entity.output"])[0].content;
  assert.equal(output[0].thought, true);
  assert.equal(output[0].thoughtSignature, "actual-signature");
  assert.equal(output[2].inlineData.data, "fixture");
});
test("modern complete histories and75tool schemas precede optional attrs", async () => {
  const e = setup();
  transport(
    e,
    response([
      { functionCall: { id: "native-call", name: "tool_79", args: { x: 0 } } },
    ]),
  );
  await e.plugin.activate();
  const tools = [
    {
      functionDeclarations: Array.from({ length: 80 }, (_, i) => ({
        name: `tool_${i}`,
        parameters: {
          type: "OBJECT",
          properties: { x: { type: "INTEGER" } },
          required: ["x"],
        },
      })),
    },
  ];
  await modern().models.generateContent({
    ...request,
    contents: Array.from({ length: 80 }, (_, i) => ({
      role: "user",
      parts: [{ text: `history-${i}` }],
    })),
    config: { tools },
  });
  const a = e.native[0].attributes;
  assert.equal(JSON.parse(a["traceloop.entity.input"]).length, 80);
  assert.equal(JSON.parse(a["llm.request.functions"]).length, 80);
  assert.equal(
    JSON.parse(a["gen_ai.completion.0.tool_calls"])[0].id,
    "native-call",
  );
  assert.equal(a["llm.usage.total_tokens"], 17);
});
test("modern native chat captures actual preceding turns with tool IDs", async () => {
  const e = setup();
  let turn = 0;
  transport(e, () =>
    ++turn === 1
      ? response([
          {
            functionCall: {
              id: "chat-call",
              name: "weather",
              args: { city: "Tokyo" },
            },
          },
        ])
      : response(),
  );
  await e.plugin.activate();
  const chat = modern().chats.create({ model: "gemini-fixture" });
  await chat.sendMessage({ message: "weather?" });
  await chat.sendMessage({
    message: [
      {
        functionResponse: {
          id: "chat-call",
          name: "weather",
          response: { temperature: 0 },
        },
      },
    ],
  });
  assert.equal(e.native.length, 2);
  const input = JSON.parse(e.native[1].attributes["traceloop.entity.input"]);
  assert.ok(JSON.stringify(input).includes("chat-call"));
  assert.equal(input.length, 3);
  assert.equal(
    e.native.filter((s) => s.attributes["respan.entity.log_type"] === "tool")
      .length,
    0,
  );
});
test("modern native AFC executes callback once and traces only two actual model calls", async () => {
  const e = setup();
  let turns = 0,
    called = 0;
  transport(e, () =>
    ++turns === 1
      ? response([
          {
            functionCall: {
              id: "afc-call",
              name: "weather",
              args: { city: "Tokyo" },
            },
          },
        ])
      : response([{ text: "AFC answer" }]),
  );
  await e.plugin.activate();
  const callable = {
    async tool() {
      return {
        functionDeclarations: [
          {
            name: "weather",
            parameters: {
              type: "OBJECT",
              properties: { city: { type: "STRING" } },
            },
          },
        ],
      };
    },
    async callTool(calls) {
      called++;
      assert.equal(calls[0].id, "afc-call");
      return [
        {
          functionResponse: {
            id: calls[0].id,
            name: calls[0].name,
            response: { temperature: 0 },
          },
        },
      ];
    },
  };
  const r = await modern().models.generateContent({
    ...request,
    config: { tools: [callable] },
  });
  assert.equal(r.text, "AFC answer");
  assert.equal(called, 1);
  assert.equal(e.requests.length, 2);
  assert.equal(e.native.length, 2);
  assert.ok(
    e.native.every((s) => s.attributes["respan.entity.log_type"] === "chat"),
  );
  assert.ok(
    e.native[1].attributes["traceloop.entity.input"].includes("afc-call"),
  );
});
const sse = (values) =>
  new Response(
    values.map((value) => `data: ${JSON.stringify(value)}\n\n`).join(""),
    { headers: { "content-type": "text/event-stream" } },
  );
test("modern SSE preserves native generator and parts while streaming two candidates", async () => {
  const chunks = [
    response([{ text: "A" }], undefined),
    {
      candidates: [
        { index: 0, content: { role: "model", parts: [{ text: "C" }] } },
        { index: 1, content: { role: "model", parts: [{ text: "B" }] } },
      ],
      usageMetadata: {
        promptTokenCount: 4,
        candidatesTokenCount: 5,
        totalTokenCount: 17,
      },
    },
  ];
  const e = setup();
  transport(e, sse(chunks));
  await e.plugin.activate();
  const stream = await modern().models.generateContentStream(request);
  const iterator = stream[Symbol.asyncIterator]();
  assert.equal(iterator, stream);
  const values = [];
  for await (const value of iterator) values.push(value);
  assert.equal(values.length, 2);
  const output = JSON.parse(e.native[0].attributes["traceloop.entity.output"]);
  assert.equal(output[0].content.length, 2);
  assert.equal(output[1].content[0].text, "B");
});
test("modern early stream return closes exactly one real call", async () => {
  const e = setup();
  transport(
    e,
    sse([response([{ text: "first" }]), response([{ text: "second" }])]),
  );
  await e.plugin.activate();
  const stream = await modern().models.generateContentStream(request);
  for await (const chunk of stream) {
    assert.ok(chunk);
    break;
  }
  assert.equal(e.native.length, 1);
  assert.equal(
    JSON.parse(e.native[0].attributes["traceloop.entity.output"])[0].content[0]
      .text,
    "first",
  );
});
test("legacy streamed result preserves response and stream contracts", async () => {
  const e = setup();
  transport(e, sse([response([{ text: "A" }]), response([{ text: "B" }])]));
  await e.plugin.activate();
  const r = await old().generateContentStream("PROMPT");
  const chunks = [];
  for await (const chunk of r.stream) chunks.push(chunk);
  const aggregated = await r.response;
  assert.equal(chunks.length, 2);
  assert.equal(aggregated.candidates[0].content.parts[0].text, "AB");
  assert.equal(e.native.length, 1);
});
test("modern embeddings retain5001dimensions and actual statistics", async () => {
  const e = setup();
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  transport(e, {
    predictions: [
      {
        embeddings: {
          values: vector,
          statistics: { token_count: 7, truncated: false },
        },
      },
    ],
  });
  await e.plugin.activate();
  const r = await modern().models.embedContent({
    model: "text-embedding-005",
    contents: "embedding input",
  });
  assert.equal(r.embeddings[0].values.length, 5001);
  const a = e.native[0].attributes;
  assert.equal(a["respan.entity.log_type"], "embedding");
  assert.equal(JSON.parse(a["traceloop.entity.output"])[0].values.length, 5001);
  assert.equal(a["llm.usage.total_tokens"], undefined);
  assert.equal(a["gen_ai.usage.input_tokens"], 7);
});
test("modern image generation preserves actual typed native output", async () => {
  const e = setup();
  transport(e, {
    predictions: [
      { bytesBase64Encoded: "fixture-image", mimeType: "image/png" },
    ],
  });
  await e.plugin.activate();
  const r = await modern().models.generateImages({
    model: "imagen-4.0-generate-001",
    prompt: "fixture image",
  });
  assert.equal(r.generatedImages[0].image.imageBytes, "fixture-image");
  assert.equal(e.native[0].attributes["respan.entity.log_type"], "generation");
  assert.ok(
    e.native[0].attributes["traceloop.entity.output"].includes("fixture-image"),
  );
});
test("modern video start preserves pending operation without invented media completion", async () => {
  const e = setup();
  transport(e, {
    name: "projects/fixture/locations/us-central1/publishers/google/models/veo/operations/actual-operation",
    done: false,
  });
  await e.plugin.activate();
  const r = await modern().models.generateVideos({
    model: "veo-3.0-generate-001",
    source: { prompt: "fixture video" },
  });
  assert.equal(r.done, false);
  const a = e.native[0].attributes;
  assert.ok(a["traceloop.entity.output"].includes("actual-operation"));
  assert.ok(!a["traceloop.entity.output"].includes("generatedVideos"));
});
test("modern provider error keeps status, rejects natively and omits output", async () => {
  const e = setup();
  transport(
    e,
    Response.json(
      {
        error: {
          code: 429,
          message: "controlled error",
          status: "RESOURCE_EXHAUSTED",
        },
      },
      { status: 429 },
    ),
  );
  await e.plugin.activate();
  await assert.rejects(
    modern().models.generateContent(request),
    (error) => error.status === 429,
  );
  assert.equal(e.native[0].status.code, 2);
  assert.equal(e.native[0].attributes.status_code, 429);
  assert.equal(e.native[0].attributes["traceloop.entity.output"], undefined);
});
for (const [label, options] of [
  ["constructor", { traceContent: false }],
  ["input lane", { recordInputs: false }],
  ["output lane", { recordOutputs: false }],
])
  test(`native ${label} ceiling`, async () => {
    const e = setup(options);
    transport(e);
    await e.plugin.activate();
    await context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
      () => modern().models.generateContent(request),
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
  });
for (const [label, decision] of [
  ["DROP", SamplingDecision.NOT_RECORD],
  ["RECORD_ONLY", SamplingDecision.RECORD],
])
  test(`real ${label} sampler precedes adapter serialization`, async () => {
    let copies = 0,
      observed;
    const e = setup({}, decision);
    e.sampler.shouldSample = () => {
      observed = copies;
      return { decision };
    };
    transport(e);
    const nativeRequest = {
      ...request,
      contents: {
        role: "user",
        parts: [
          {
            text: "native",
            toJSON() {
              copies++;
              return { text: "native" };
            },
          },
        ],
      },
    };
    await modern().models.generateContent(nativeRequest);
    const bareCopies = copies;
    copies = 0;
    await e.plugin.activate();
    await modern().models.generateContent(nativeRequest);
    assert.equal(observed, 0);
    assert.equal(copies, bareCopies);
    for (const s of e.native)
      assert.equal(s.attributes["traceloop.entity.input"], undefined);
    assert.equal(e.exporter.getFinishedSpans().length, 0);
  });
test("canonical context and suppression prevent capture", async () => {
  const e = setup();
  transport(e);
  await e.plugin.activate();
  await context.with(
    context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
    () => modern().models.generateContent(request),
  );
  assert.equal(e.native[0].attributes["traceloop.entity.input"], undefined);
  await context.with(suppressTracing(context.active()), () =>
    modern().models.generateContent(request),
  );
  assert.equal(
    e.native.at(-1).attributes["traceloop.entity.output"],
    undefined,
  );
});
test("late private ancestor sanitizes actual attributes, events, status and nonconfig initialattrs", async () => {
  let actual;
  const e = setup({}, SamplingDecision.RECORD_AND_SAMPLED, (span) => {
    actual = span;
    Object.defineProperty(span, "attributes", {
      value: span.attributes,
      writable: true,
      configurable: false,
      enumerable: true,
    });
  });
  transport(e, () => {
    parent.setAttribute("allow_trace_content", false);
    actual.setAttribute("private", "secret");
    actual.addEvent("secret", { private: "secret" });
    actual.setStatus({ code: 2, message: "secret" });
    return response();
  });
  await e.plugin.activate();
  const parent = trace.getTracer("test").startSpan("parent");
  await context.with(trace.setSpan(context.active(), parent), () =>
    modern().models.generateContent(request),
  );
  const s = e.native[0];
  assert.equal(s.attributes.private, undefined);
  assert.equal(s.attributes["traceloop.entity.output"], undefined);
  assert.equal(s.events.length, 0);
  assert.equal(s.status.message, undefined);
  parent.end();
});
test("independent owner privacy and final deactivation restore native methods", async () => {
  const e = setup();
  transport(e);
  await e.plugin.activate();
  const second = new VertexAIInstrumentor({
    sdkModule: legacy,
    genaiModule: genai,
    traceContent: false,
  });
  await second.activate();
  await modern().models.generateContent(request);
  assert.equal(e.native[0].attributes["traceloop.entity.input"], undefined);
  e.plugin.deactivate();
  assert.equal(second.isActive(), true);
  await modern().models.generateContent(request);
  second.deactivate();
  await modern().models.generateContent(request);
  assert.equal(e.native.length, 2);
});
test("activation cancellation, coalescing and reactivation do not duplicate patches", async () => {
  const e = setup();
  transport(e);
  const pending = e.plugin.activate();
  e.plugin.deactivate();
  await pending;
  assert.equal(e.plugin.isActive(), false);
  await Promise.all([e.plugin.activate(), e.plugin.activate()]);
  await modern().models.generateContent(request);
  assert.equal(e.native.length, 1);
});

for (const [method, args] of [
  [
    "editImage",
    {
      model: "imagen-3.0-capability-001",
      prompt: "edit fixture",
      referenceImages: [],
    },
  ],
  [
    "upscaleImage",
    {
      model: "imagen-4.0-upscale-preview",
      image: { imageBytes: "fixture", mimeType: "image/png" },
      upscaleFactor: "x2",
    },
  ],
  [
    "recontextImage",
    {
      model: "imagen-product-recontext-preview-06-30",
      source: {
        productImages: [
          { productImage: { imageBytes: "fixture", mimeType: "image/png" } },
        ],
      },
    },
  ],
  [
    "segmentImage",
    {
      model: "image-segmentation-001",
      source: { image: { imageBytes: "fixture", mimeType: "image/png" } },
      config: { mode: "foreground" },
    },
  ],
])
  test(`modern native ${method} inference`, async () => {
    const e = setup();
    transport(e, {
      predictions: [
        { bytesBase64Encoded: "fixture-image", mimeType: "image/png" },
      ],
    });
    await e.plugin.activate();
    const value = await modern().models[method](args);
    assert.ok(value);
    assert.equal(e.requests.length, 1);
    assert.equal(e.native.length, 1);
    assert.equal(
      e.native[0].attributes["respan.entity.log_type"],
      "generation",
    );
    assert.ok(
      e.native[0].attributes["traceloop.entity.input"].includes(args.model),
    );
  });

test("independent actual legacy and modern modules release only their owned patches", async () => {
  const e = setup();
  transport(e);
  const a = new VertexAIInstrumentor({ sdkModule: legacy, genaiModule: {} });
  const b = new VertexAIInstrumentor({ sdkModule: {}, genaiModule: genai });
  await a.activate();
  await b.activate();
  await old().generateContent("owned legacy");
  await modern().models.generateContent(request);
  assert.equal(e.native.length, 2);
  a.deactivate();
  await old().generateContent("unowned legacy");
  await modern().models.generateContent(request);
  assert.equal(e.native.length, 3);
  b.deactivate();
});

test("public legacy helper preserves canonical native usage and request parameters", async () => {
  const e = setup();
  transport(e);
  await e.plugin.activate();
  const r = await modern().models.generateContent(request);
  const attrs = buildGenerateContentAttrs({
    requestPayload: {
      model: request.model,
      contents: request.contents,
      generationConfig: { temperature: 0, maxOutputTokens: 64 },
    },
    responseOrChunks: r,
  });
  assert.equal(attrs["gen_ai.usage.input_tokens"], 4);
  assert.equal(attrs["llm.usage.total_tokens"], 17);
  assert.equal(attrs["gen_ai.request.temperature"], 0);
  assert.equal(attrs["gen_ai.request.max_tokens"], 64);
  assert.equal(
    JSON.parse(attrs["traceloop.entity.output"])[0].content[0].text,
    "ANSWER",
  );
});

test("native legacy aggregate-only await observes unchanged Promise identity", async () => {
  const e = setup();
  transport(e, sse([response([{ text: "aggregate" }])]));
  await e.plugin.activate();
  const result = await old().generateContentStream("actual prompt");
  const nativePromise = result.response;
  const aggregate = await result.response;
  assert.equal(result.response, nativePromise);
  assert.equal(aggregate.candidates[0].content.parts[0].text, "aggregate");
  assert.equal(e.native.length, 1);
  assert.equal(e.native[0].attributes["llm.usage.total_tokens"], 17);
});
test("native GenAI locked writable actual attrs replacement is scrubbed under false", async () => {
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
  transport(e, () => {
    actual.attributes = {
      "traceloop.entity.output": "PRIVATE_LOCKED",
      unknown: "PRIVATE_LOCKED",
    };
    return response();
  });
  await e.plugin.activate();
  await modern().models.generateContent(request);
  assert.ok(!JSON.stringify(e.native[0].attributes).includes("PRIVATE_LOCKED"));
});
