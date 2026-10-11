import { MimeType } from "@arizeai/openinference-semantic-conventions";
import assert from "node:assert/strict";
import test from "node:test";
import {
  context,
  trace,
  ROOT_CONTEXT,
  SpanStatusCode,
} from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import {
  SemanticConventions as OI,
  OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";
import {
  withSpan,
  getOutputAttributes,
  traceTool,
  getLLMAttributes,
  getEmbeddingAttributes,
  getRetrieverAttributes,
  OITracer,
  setMetadata,
} from "@arizeai/openinference-core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import {
  OpenInferenceTranslator,
  prepareOpenInferenceSpanForExport,
} from "../dist/index.js";

function setup(options = {}, sampler) {
  trace.disable();
  context.disable();
  context.setGlobalContextManager(
    new AsyncLocalStorageContextManager().enable(),
  );
  const translator = new OpenInferenceTranslator(options);
  const native = [];
  const exported = [];
  const provider = new BasicTracerProvider({
    sampler,
    spanProcessors: [
      {
        onStart: translator.onStart.bind(translator),
        onEnd(span) {
          translator.onEnd(span);
          native.push(span);
          exported.push(prepareOpenInferenceSpanForExport(span));
        },
        shutdown: async () => {},
        forceFlush: async () => {},
      },
    ],
  });
  trace.setGlobalTracerProvider(provider);
  return { provider, native, exported };
}
const completion = () =>
  getLLMAttributes({
    modelName: "requested",
    requestModelName: "requested",
    responseModelName: "actual",
    provider: "openai",
    inputMessages: Array.from({ length: 80 }, (_, i) => ({
      role: "user",
      content: `input-${i}`,
    })),
    outputMessages: [
      {
        role: "assistant",
        content: "",
        toolCalls: [
          {
            id: "native-call",
            function: { name: "lookup", arguments: '{"q":"x"}' },
          },
        ],
      },
    ],
    tools: Array.from({ length: 75 }, (_, i) => ({
      jsonSchema: {
        type: "function",
        function: { name: `tool-${i}`, parameters: { type: "object" } },
      },
    })),
    tokenCount: { prompt: 0, completion: 0, total: 0 },
  });

test("real released helper retains 80 messages, 75 tools, output and zero usage at the default native budget", async () => {
  const env = setup();
  const call = withSpan(() => false, {
    name: "native-llm",
    kind: OpenInferenceSpanKind.LLM,
    processOutput: completion,
  });
  assert.equal(call(), false);
  const attrs = env.exported[0].attributes;
  assert.equal(JSON.parse(attrs["traceloop.entity.input"]).length, 80);
  assert.equal(JSON.parse(attrs["llm.request.functions"]).length, 75);
  assert.equal(
    JSON.parse(attrs["traceloop.entity.output"])[0].tool_calls[0].id,
    "native-call",
  );
  assert.equal(
    attrs["gen_ai.completion.0.tool_calls"],
    '[{"type":"function","function":{"name":"lookup","arguments":"{\\"q\\":\\"x\\"}"},"id":"native-call"}]',
  );
  assert.equal(attrs["gen_ai.request.model"], "requested");
  assert.equal(
    attrs["gen_ai.response.model"],
    env.native[0].attributes[OI.LLM_RESPONSE_MODEL_NAME] === undefined
      ? undefined
      : "actual",
  );
  assert.equal(attrs["gen_ai.usage.input_tokens"], 0);
  assert.equal(attrs["llm.usage.total_tokens"], 0);
  assert.ok(env.native[0].droppedAttributesCount > 0);
  assert.equal(
    env.native[0].attributes["llm.input_messages.0.message.role"],
    "user",
  );
  for (const key of [
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "llm.tools.0.tool.json_schema",
  ])
    assert.equal(attrs[key], undefined);
  await env.provider.shutdown();
});

test("released helpers fill absent fields and preserve native adapter canonical values, including falsy values", async () => {
  const env = setup();
  const input = JSON.stringify({ prompt: "question", history: ["earlier"] });
  const output = JSON.stringify({ role: "assistant", content: "answer" });
  for (const value of [output, "false", "0", "null", ""]) {
    withSpan(() => false, {
      name: "native-llm",
      kind: OpenInferenceSpanKind.LLM,
      attributes: {
        "respan.entity.log_type": "agent",
        "traceloop.entity.name": "NativeAgent",
        "traceloop.entity.path": "",
        "traceloop.entity.input": input,
        "traceloop.entity.output": value,
        "gen_ai.request.model": "canonical-model",
        "gen_ai.usage.input_tokens": 0,
        "gen_ai.prompt.0.content": false,
        "respan.metadata": '{"collision":"canonical"}',
      },
      processOutput: () => ({
        ...getLLMAttributes({
          modelName: "raw-model",
          provider: "openai",
          inputMessages: [{ role: "user", content: "raw input" }],
          outputMessages: [{ role: "assistant", content: "raw output" }],
          tokenCount: { prompt: 9, completion: 7 },
        }),
        [OI.METADATA]: '{"collision":"raw","upstream":true}',
      }),
    })();
    const attrs = env.exported.at(-1).attributes;
    assert.equal(attrs["respan.entity.log_type"], "agent");
    assert.equal(attrs["traceloop.entity.name"], "NativeAgent");
    assert.equal(attrs["traceloop.entity.path"], "");
    assert.equal(attrs["traceloop.entity.input"], input);
    assert.equal(attrs["traceloop.entity.output"], value);
    assert.equal(attrs["gen_ai.request.model"], "canonical-model");
    assert.equal(attrs["gen_ai.usage.input_tokens"], 0);
    assert.equal(attrs["gen_ai.prompt.0.content"], false);
    assert.equal(attrs["gen_ai.prompt.0.role"], "user");
    assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
    assert.deepEqual(JSON.parse(attrs["respan.metadata"]), {
      collision: "canonical",
      upstream: true,
    });
  }
  await env.provider.shutdown();
});

test("real embedding builder preserves its model and 5001-dimensional vector", async () => {
  const env = setup();
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  const call = withSpan(() => vector, {
    name: "embedding",
    kind: OpenInferenceSpanKind.EMBEDDING,
    processOutput: () =>
      getEmbeddingAttributes({
        modelName: "embedding-native",
        embeddings: [{ text: "hello", vector }],
      }),
  });
  assert.equal(call(), vector);
  const attrs = env.exported[0].attributes;
  assert.equal(attrs["gen_ai.request.model"], "embedding-native");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [vector]);
  assert.equal(attrs["embedding.model_name"], undefined);
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  await env.provider.shutdown();
});

test("real tool and retrieval helpers preserve execution arguments, false/zero and document output", async () => {
  const env = setup();
  const result = { ok: false, count: 0, value: null };
  const tool = traceTool((args) => result, {
    name: "lookup",
    attributes: { [OI.TOOL_CALL_ID]: "native-call" },
  });
  assert.equal(tool({ q: "x" }), result);
  const attrs = env.exported[0].attributes;
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), {
    name: "lookup",
    arguments: { q: "x" },
  });
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), result);
  assert.equal(attrs["gen_ai.tool.call.id"], "native-call");
  assert.equal(attrs["llm.request.type"], undefined);
  withSpan(() => false, {
    name: "retrieve",
    kind: OpenInferenceSpanKind.RETRIEVER,
    processOutput: () =>
      getRetrieverAttributes({
        documents: [
          { id: "doc", content: "text", score: 0, metadata: { ok: false } },
        ],
      }),
  })();
  assert.deepEqual(
    JSON.parse(env.exported[1].attributes["traceloop.entity.output"]),
    [{ id: "doc", content: "text", score: 0, metadata: { ok: false } }],
  );
  await env.provider.shutdown();
});

test("multimodal and historical versus current calls remain separate canonical messages", async () => {
  const env = setup();
  withSpan(() => null, {
    name: "multimodal",
    kind: OpenInferenceSpanKind.LLM,
    processOutput: () =>
      getLLMAttributes({
        inputMessages: [
          {
            role: "assistant",
            toolCalls: [
              { id: "history", function: { name: "old", arguments: "{}" } },
            ],
          },
          {
            role: "user",
            contents: [
              { type: "text", text: "question" },
              {
                type: "image",
                image: { url: "https://example.invalid/image.png" },
              },
            ],
          },
        ],
        outputMessages: [
          {
            role: "assistant",
            content: "answer",
            toolCalls: [
              { id: "current", function: { name: "new", arguments: "{}" } },
            ],
          },
        ],
      }),
  })();
  const attrs = env.exported[0].attributes;
  assert.equal(
    JSON.parse(attrs["gen_ai.prompt.0.tool_calls"])[0].id,
    "history",
  );
  assert.equal(
    JSON.parse(attrs["gen_ai.completion.0.tool_calls"])[0].id,
    "current",
  );
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.1.content"])[1], {
    type: "image_url",
    image_url: { url: "https://example.invalid/image.png" },
  });
  await env.provider.shutdown();
});

test("metadata merges inherited canonical JSON and upstream context metadata", async () => {
  const env = setup();
  context.with(setMetadata(context.active(), { upstream: 1 }), () =>
    withSpan(() => false, {
      name: "metadata",
      attributes: { "respan.metadata": '{"inherited":true}' },
    })(),
  );
  assert.deepEqual(JSON.parse(env.exported[0].attributes["respan.metadata"]), {
    inherited: true,
    upstream: 1,
  });
  await env.provider.shutdown();
});

for (const option of [
  { traceContent: false },
  { recordInputs: false },
  { recordOutputs: false },
])
  test(`actual span guards preserve independent capture ceilings ${JSON.stringify(option)}`, async () => {
    const env = setup(option);
    let actual;
    const tracer = new OITracer({ tracer: trace.getTracer("custom-native") });
    tracer.startActiveSpan(
      "private",
      {
        attributes: { [OI.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM },
      },
      (span) => {
        actual = trace.getSpan(context.active());
        span.setAttribute(OI.INPUT_VALUE, "PRIVATE_INPUT");
        span.setAttribute(OI.OUTPUT_VALUE, "PRIVATE_OUTPUT");
        span.setAttribute(OI.LLM_MODEL_NAME, "safe-model");
        actual.attributes = {
          ...actual.attributes,
          unknown: "PRIVATE_UNKNOWN",
        };
        actual.status = { code: 2, message: "PRIVATE_STATUS" };
        actual.events = [{ name: "PRIVATE_EVENT" }];
        actual.setAttribute("allow_trace_content", true);
        span.end();
      },
    );
    const attrs = env.exported[0].attributes;
    assert.equal(attrs["gen_ai.request.model"], "safe-model");
    assert.equal(
      attrs["traceloop.entity.input"],
      option.recordOutputs === false ? "PRIVATE_INPUT" : undefined,
    );
    assert.equal(
      attrs["traceloop.entity.output"],
      option.recordInputs === false ? "PRIVATE_OUTPUT" : undefined,
    );
    assert.equal(attrs.unknown, undefined);
    assert.deepEqual(env.exported[0].status, { code: 2 });
    assert.deepEqual(env.exported[0].events, []);
    assert.throws(() =>
      Object.defineProperty(actual, "attributes", {
        value: { unknown: "PRIVATE_REPLACEMENT" },
        configurable: false,
      }),
    );
    await env.provider.shutdown();
  });

test("denied late native setAttributes skips unknown getters and toJSON", async () => {
  const env = setup({ traceContent: false });
  let visits = 0;
  const span = trace.getTracer("native").startSpan("private");
  span.setAttribute(OI.OPENINFERENCE_SPAN_KIND, "CHAIN");
  const poison = {
    get unknown() {
      visits++;
      throw Error("private");
    },
    [OI.INPUT_VALUE]: {
      toJSON() {
        visits++;
        return "private";
      },
    },
  };
  span.setAttributes(poison);
  span.setStatus({
    code: 2,
    get message() {
      visits++;
      return "private";
    },
  });
  span.end();
  assert.equal(visits, 0);
  assert.equal(env.exported[0].attributes["traceloop.entity.input"], undefined);
  await env.provider.shutdown();
});

for (const lane of ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
  test(`environment ${lane}=false remains sticky`, async () => {
    const env = setup();
    process.env[lane] = "false";
    const span = new OITracer({ tracer: trace.getTracer("native") }).startSpan(
      "private",
      {
        attributes: {
          [OI.OPENINFERENCE_SPAN_KIND]: "CHAIN",
          [OI.INPUT_VALUE]: "PRIVATE_INPUT",
        },
      },
    );
    process.env[lane] = "true";
    span.setAttribute(OI.OUTPUT_VALUE, "PRIVATE_OUTPUT");
    span.end();
    delete process.env[lane];
    assert.equal(
      JSON.stringify(env.exported[0].attributes).includes("PRIVATE_"),
      false,
    );
    await env.provider.shutdown();
  });

test("late parent and canonical context vetoes cannot be reversed", async () => {
  const env = setup();
  const parent = trace.getTracer("parent").startSpan("parent");
  context.with(trace.setSpan(context.active(), parent), () => {
    const span = new OITracer({ tracer: trace.getTracer("native") }).startSpan(
      "child",
      {
        attributes: {
          [OI.OPENINFERENCE_SPAN_KIND]: "CHAIN",
          [OI.INPUT_VALUE]: "PRIVATE_INPUT",
        },
      },
    );
    parent.setAttribute("allow_trace_content", false);
    span.setAttribute(OI.OUTPUT_VALUE, "PRIVATE_OUTPUT");
    parent.setAttribute("allow_trace_content", true);
    span.end();
  });
  const attrs = env.exported[0].attributes;
  assert.equal(JSON.stringify(attrs).includes("PRIVATE_"), false);
  const ctx = ROOT_CONTEXT.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
  context.with(ctx, () =>
    traceTool(() => "PRIVATE_OUTPUT", { name: "context-private" })(
      "PRIVATE_INPUT",
    ),
  );
  assert.equal(
    JSON.stringify(env.exported[1].attributes).includes("PRIVATE_"),
    false,
  );
  parent.end();
  await env.provider.shutdown();
});

test("RECORD_ONLY never retains payload and exporter suppression does not revoke admitted spans", async () => {
  let env = setup(
    {},
    {
      shouldSample: () => ({ decision: SamplingDecision.RECORD }),
      toString: () => "record-only",
    },
  );
  traceTool(() => "PRIVATE_OUTPUT", { name: "record-only" })("PRIVATE_INPUT");
  assert.equal(env.native[0].attributes[OI.INPUT_VALUE], undefined);
  await env.provider.shutdown();
  env = setup();
  const span = new OITracer({ tracer: trace.getTracer("native") }).startSpan(
    "admitted",
    {
      attributes: {
        [OI.OPENINFERENCE_SPAN_KIND]: "CHAIN",
        [OI.INPUT_VALUE]: "kept",
      },
    },
  );
  context.with(suppressTracing(context.active()), () => span.end());
  assert.equal(env.exported[0].attributes["traceloop.entity.input"], "kept");
  await env.provider.shutdown();
});

test("native helper errors retain status without fabricating completion or changing thrown identity", async () => {
  const env = setup();
  const error = Error("controlled failure");
  const fn = withSpan(
    async () => {
      throw error;
    },
    { name: "failure", kind: OpenInferenceSpanKind.LLM },
  );
  await assert.rejects(fn(), (value) => value === error);
  assert.equal(env.exported[0].status.code, SpanStatusCode.ERROR);
  assert.equal(
    env.exported[0].attributes["traceloop.entity.output"],
    undefined,
  );
  await env.provider.shutdown();
});

test("DROP invokes no bridge start/capture; released helper serialization remains upstream-owned", async () => {
  const env = setup(
    {},
    {
      shouldSample: () => ({ decision: SamplingDecision.NOT_RECORD }),
      toString: () => "drop",
    },
  );
  let visits = 0;
  withSpan(() => false, { name: "native-drop" })({
    toJSON() {
      visits++;
      return { private: "fixture" };
    },
  });
  assert.equal(visits, 1);
  assert.equal(env.native.length, 0);
  await env.provider.shutdown();
});

test("initial-options native truncation remains an explicit producer boundary", async () => {
  const env = setup();
  const span = trace
    .getTracer("native-initial-options")
    .startSpan("initial-options", {
      attributes: {
        [OI.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        ...completion(),
      },
    });
  span.end();
  assert.ok(
    JSON.parse(env.exported[0].attributes["traceloop.entity.input"]).length <
      80,
  );
  assert.equal(
    env.exported[0].attributes["traceloop.entity.output"],
    undefined,
  );
  assert.equal(env.exported[0].attributes["llm.request.functions"], undefined);
  await env.provider.shutdown();
});

test("unrelated native spans retain their attribute object and native mutation behavior", async () => {
  const env = setup({ traceContent: false });
  const span = trace.getTracer("unrelated").startSpan("unrelated");
  const attributes = span.attributes;
  span.setAttribute("app.value", "ordinary");
  assert.equal(span.attributes, attributes);
  span.end();
  assert.equal(env.exported[0].attributes["app.value"], "ordinary");
  await env.provider.shutdown();
});

test("explicit released output helper preserves null, while denied safe keys accept only scalar primitives", async () => {
  let env = setup();
  const tool = traceTool((_args) => null, {
    name: "null-result",
    processOutput: (output) =>
      getOutputAttributes({
        value: JSON.stringify(output),
        mimeType: MimeType.JSON,
      }),
  });
  assert.equal(tool({}), null);
  assert.equal(env.exported[0].attributes["traceloop.entity.output"], "null");
  await env.provider.shutdown();
  env = setup({ traceContent: false });
  const span = trace.getTracer("native").startSpan("private");
  span.setAttribute(OI.OPENINFERENCE_SPAN_KIND, "LLM");
  span.setAttribute(OI.LLM_MODEL_NAME, ["PRIVATE_SAFE_KEY_ARRAY"]);
  span.setAttribute(OI.LLM_TOKEN_COUNT_PROMPT, [12, 34]);
  span.end();
  assert.equal(env.exported[0].attributes["gen_ai.request.model"], undefined);
  assert.equal(
    env.exported[0].attributes["gen_ai.usage.input_tokens"],
    undefined,
  );
  await env.provider.shutdown();
});

test("private export ReadableSpan clone rejects late attributes, events, status text and hard replacements", async () => {
  const env = setup({ traceContent: false });
  traceTool((_input) => "PRIVATE_OUTPUT", { name: "private-export" })(
    "PRIVATE_INPUT",
  );
  const span = env.exported[0];
  span.attributes = { ...span.attributes, unknown: "PRIVATE_LATE_ATTRIBUTE" };
  span.status = { code: 2, message: "PRIVATE_LATE_STATUS" };
  span.events = [{ name: "PRIVATE_LATE_EVENT" }];
  assert.equal(
    JSON.stringify({
      attributes: span.attributes,
      status: span.status,
      events: span.events,
    }).includes("PRIVATE_"),
    false,
  );
  assert.equal(span.status.code, 2);
  assert.throws(() =>
    Object.defineProperty(span, "attributes", {
      value: { unknown: "PRIVATE_REPLACEMENT" },
      configurable: false,
    }),
  );
  await env.provider.shutdown();
});

test("peer: recordInputs false must remove native embedding text while retaining vector", async () => {
  const env = setup({ recordInputs: false });
  withSpan(() => [0, 1], {
    name: "peer-private-embedding-input",
    kind: OpenInferenceSpanKind.EMBEDDING,
    processOutput: () =>
      getEmbeddingAttributes({
        modelName: "native-model",
        embeddings: [{ text: "private embedding input", vector: [0, 1] }],
      }),
  })();
  const attrs = env.native[0].attributes;
  console.log(
    JSON.stringify({
      peer: "embedding-input-lane",
      nativeText: attrs["embedding.embeddings.0.embedding.text"],
      exportedInput: env.exported[0].attributes["traceloop.entity.input"],
      exportedOutput: env.exported[0].attributes["traceloop.entity.output"],
    }),
  );
  assert.equal(attrs["embedding.embeddings.0.embedding.text"], undefined);
  assert.deepEqual(
    JSON.parse(env.exported[0].attributes["traceloop.entity.output"]),
    [[0, 1]],
  );
  await env.provider.shutdown();
});

test("peer: recordOutputs false preserves legitimate native embedding input text", async () => {
  const env = setup({ recordOutputs: false });
  withSpan(() => [0, 1], {
    name: "peer-embedding-output-denied",
    kind: OpenInferenceSpanKind.EMBEDDING,
    processOutput: () =>
      getEmbeddingAttributes({
        modelName: "native-model",
        embeddings: [{ text: "actual embedding input", vector: [0, 1] }],
      }),
  })();
  console.log(
    JSON.stringify({
      peer: "embedding-output-lane",
      input: env.exported[0].attributes["traceloop.entity.input"],
    }),
  );
  assert.deepEqual(
    JSON.parse(env.exported[0].attributes["traceloop.entity.input"]),
    ["actual embedding input"],
  );
  assert.equal(
    env.exported[0].attributes["traceloop.entity.output"],
    undefined,
  );
  await env.provider.shutdown();
});

test("peer: nonconfigurable native attribute replacement must not retain denied native input", async () => {
  trace.disable();
  const translator = new OpenInferenceTranslator({ traceContent: false });
  let actual;
  const provider = new BasicTracerProvider({
    spanProcessors: [
      {
        onStart(span, ctx) {
          actual = span;
          Object.defineProperty(span, "attributes", {
            value: span.attributes,
            writable: true,
            configurable: false,
            enumerable: true,
          });
          translator.onStart(span, ctx);
        },
        onEnd(span) {
          translator.onEnd(span);
        },
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  });
  trace.setGlobalTracerProvider(provider);
  const native = new OITracer({ tracer: trace.getTracer("peer-native") });
  const span = native.startSpan("peer-locked", {
    attributes: getLLMAttributes({
      inputMessages: [{ role: "user", content: "native private input" }],
      outputMessages: [{ role: "assistant", content: "native private output" }],
    }),
  });
  span.end();
  console.log(
    JSON.stringify({
      peer: "nonconfig-native",
      input: actual.attributes["llm.input_messages.0.message.content"],
      output: actual.attributes["llm.output_messages.0.message.content"],
    }),
  );
  assert.equal(
    actual.attributes["llm.input_messages.0.message.content"],
    undefined,
  );
  assert.equal(
    actual.attributes["llm.output_messages.0.message.content"],
    undefined,
  );
  await provider.shutdown();
});

test("peer: immutable native attribute data fails closed without retaining denied native input", async () => {
  trace.disable();
  const translator = new OpenInferenceTranslator({ traceContent: false });
  let actual;
  const provider = new BasicTracerProvider({
    spanProcessors: [
      {
        onStart(span, ctx) {
          actual = span;
          Object.defineProperty(span, "attributes", {
            value: span.attributes,
            writable: false,
            configurable: false,
            enumerable: true,
          });
          translator.onStart(span, ctx);
        },
        onEnd(span) {
          translator.onEnd(span);
        },
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  });
  trace.setGlobalTracerProvider(provider);
  const native = new OITracer({ tracer: trace.getTracer("peer-native") });
  const span = native.startSpan("peer-locked", {
    attributes: getLLMAttributes({
      inputMessages: [{ role: "user", content: "native private input" }],
      outputMessages: [{ role: "assistant", content: "native private output" }],
    }),
  });
  span.end();
  console.log(
    JSON.stringify({
      peer: "nonconfig-native",
      input: actual.attributes["llm.input_messages.0.message.content"],
      output: actual.attributes["llm.output_messages.0.message.content"],
    }),
  );
  assert.equal(
    actual.attributes["llm.input_messages.0.message.content"],
    undefined,
  );
  assert.equal(
    actual.attributes["llm.output_messages.0.message.content"],
    undefined,
  );
  await provider.shutdown();
});
