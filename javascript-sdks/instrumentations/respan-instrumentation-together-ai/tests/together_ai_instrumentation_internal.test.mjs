import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { JsonTraceSerializer } from "@opentelemetry/otlp-transformer";

import { patchResourceMethod } from "../dist/_patching.js";
const CHAT_SPEC = {
  kind: "chat",
  method: "create",
  spanName: "together.chat.completions",
  logType: "chat",
  requestType: "chat",
};

const EMBEDDING_SPEC = {
  kind: "embedding",
  method: "create",
  spanName: "together.embeddings",
  logType: "embedding",
  requestType: "embedding",
};

const NON_CHAT_CASES = [
  {
    label: "image generation",
    spec: {
      kind: "image",
      method: "generate",
      spanName: "together.images.generate",
      logType: "generation",
      requestType: "image",
    },
    request: {
      model: "black-forest-labs/FLUX.1-schnell-Free",
      prompt: "A compact tracing diagram",
    },
    response: {
      model: "black-forest-labs/FLUX.1-schnell-Free",
      data: [{ url: "https://example.invalid/generated.png" }],
    },
  },
  {
    label: "rerank",
    spec: {
      kind: "rerank",
      method: "create",
      spanName: "together.rerank",
      logType: "custom",
      requestType: "rerank",
    },
    request: {
      model: "Salesforce/Llama-Rank-v1",
      query: "observability",
      documents: ["Tracing", "Metrics"],
    },
    response: {
      model: "Salesforce/Llama-Rank-v1",
      results: [{ index: 0, relevance_score: 0.98 }],
      usage: { prompt_tokens: 10, completion_tokens: 0, total_tokens: 10 },
    },
    usage: { input: 10, output: 0, total: 10 },
  },
  {
    label: "speech",
    spec: {
      kind: "speech",
      method: "create",
      spanName: "together.audio.speech",
      logType: "speech",
      requestType: "speech",
    },
    request: { model: "cartesia/sonic", input: "Trace this sentence." },
    response: { model: "cartesia/sonic", audio: "base64-audio" },
  },
  {
    label: "transcription",
    spec: {
      kind: "transcription",
      method: "create",
      spanName: "together.audio.transcriptions",
      logType: "transcription",
      requestType: "transcription",
    },
    request: { model: "openai/whisper-large-v3", file: "demo.wav" },
    response: { model: "openai/whisper-large-v3", text: "Trace received." },
  },
  {
    label: "translation",
    spec: {
      kind: "translation",
      method: "create",
      spanName: "together.audio.translations",
      logType: "custom",
      requestType: "translation",
    },
    request: { model: "openai/whisper-large-v3", file: "demo.wav" },
    response: { model: "openai/whisper-large-v3", text: "Trace received." },
  },
];

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

test.before(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return {
        activeSpanProcessor: {
          onEnd(span) {
            captureState.spans.push(span);
          },
        },
      };
    },
  });
});

test.after(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

test("patchResourceMethod emits canonical chat attributes without off-contract aliases", async () => {
  captureState.spans = [];

  const resourcePrototype = {
    create(_body) {
      return Promise.resolve({
        model: "meta-llama/test",
        choices: [
          {
            message: {
              role: "assistant",
              content: "I will call the weather tool.",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: "{\"city\":\"Tokyo\"}",
                  },
                },
              ],
            },
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 7,
          total_tokens: 19,
        },
      });
    },
  };

  const patchedTarget = patchResourceMethod(resourcePrototype, CHAT_SPEC);
  assert.ok(patchedTarget);

  const result = await resourcePrototype.create({
    model: "meta-llama/test",
    messages: [
      { role: "user", content: "Use the weather tool for Tokyo." },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_existing",
            type: "function",
            function: {
              name: "get_weather",
              arguments: "{\"city\":\"Osaka\"}",
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_existing",
        content: "{\"forecast\":\"clear\"}",
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather by city.",
          parameters: {
            type: "object",
            properties: {
              city: { type: "string" },
            },
          },
        },
      },
    ],
  }).then((value) => value);

  assert.equal(result.model, "meta-llama/test");
  assert.equal(captureState.spans.length, 2);

  const toolSpan = captureState.spans[0];
  assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "get_weather");
  assert.equal(toolSpan.attributes.tool_calls, undefined);
  assert.equal(toolSpan.attributes["respan.span.tool_calls"], undefined);

  const chatSpan = captureState.spans[1];
  const attrs = chatSpan.attributes;
  assert.equal(chatSpan.instrumentationScope?.name, "@respan/instrumentation-together-ai");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["gen_ai.system"], "together");
  assert.equal(attrs["gen_ai.request.model"], "meta-llama/test");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Use the weather tool for Tokyo." },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_existing",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Osaka\"}",
          },
        },
      ],
    },
    {
      role: "tool",
      content: "{\"forecast\":\"clear\"}",
      tool_call_id: "call_existing",
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather by city.",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
        },
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 7);
  assert.equal(attrs["llm.usage.total_tokens"], 19);

  for (const bannedKey of [
    "respan.span.tools",
    "respan.span.tool_calls",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
  ]) {
    assert.equal(attrs[bannedKey], undefined, `${bannedKey} should not be emitted`);
  }

  resourcePrototype.create = patchedTarget.original;
});

test("streaming chat calls emit after async iteration completes", async () => {
  captureState.spans = [];

  const stream = {
    async *[Symbol.asyncIterator]() {
      yield {
        model: "meta-llama/stream",
        choices: [{ delta: { role: "assistant", content: "Hello " }, finish_reason: null }],
      };
      yield {
        model: "meta-llama/stream",
        choices: [{ delta: { content: "stream" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      };
    },
  };

  const resourcePrototype = {
    create(_body) {
      return Promise.resolve(stream);
    },
  };

  const patchedTarget = patchResourceMethod(resourcePrototype, CHAT_SPEC);
  const result = await resourcePrototype.create({
    model: "meta-llama/stream",
    stream: true,
    messages: [{ role: "user", content: "Say hello." }],
  }).then((value) => value);

  for await (const _chunk of result) {
    // Consume the stream to trigger final span emission.
  }

  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["gen_ai.completion.0.content"], "Hello stream");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 3);

  resourcePrototype.create = patchedTarget.original;
});

test("embedding spans preserve vectors in traceloop entity output", async () => {
  captureState.spans = [];

  const resourcePrototype = {
    create(_body) {
      return Promise.resolve({
        model: "togethercomputer/m2-bert-80M-8k-retrieval",
        data: [
          {
            index: 0,
            embedding: [0.1, 0.2, 0.3],
          },
        ],
      });
    },
  };

  const patchedTarget = patchResourceMethod(resourcePrototype, EMBEDDING_SPEC);
  await resourcePrototype.create({
    model: "togethercomputer/m2-bert-80M-8k-retrieval",
    input: "semantic search text",
  }).then((value) => value);

  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.request.model"], "togethercomputer/m2-bert-80M-8k-retrieval");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [
    { index: 0, embedding: [0.1, 0.2, 0.3] },
  ]);

  resourcePrototype.create = patchedTarget.original;
});

function serializedSpan(readableSpan) {
  const bytes = JsonTraceSerializer.serializeRequest([readableSpan]);
  assert.ok(bytes?.length, "OTLP serializer produced a payload");
  const request = JSON.parse(Buffer.from(bytes).toString("utf8"));
  const spans = (request.resourceSpans ?? []).flatMap((resourceSpans) =>
    (resourceSpans.scopeSpans ?? []).flatMap((scopeSpans) =>
      (scopeSpans.spans ?? []).map((span) => ({
        ...span,
        scopeName: scopeSpans.scope?.name,
      })),
    ),
  );
  assert.equal(spans.length, 1);
  return spans[0];
}

function serializedAttribute(span, key) {
  const value = span.attributes?.find((attribute) => attribute.key === key)?.value;
  if (!value) return undefined;
  return value.stringValue ??
    value.intValue ??
    value.doubleValue ??
    value.boolValue ??
    value.arrayValue;
}

test("non-chat spans retain canonical model and usage through OTel 2 OTLP serialization", async (t) => {
  for (const testCase of NON_CHAT_CASES) {
    await t.test(testCase.label, async () => {
      captureState.spans = [];
      const resourcePrototype = {
        [testCase.spec.method](_body) {
          return Promise.resolve(testCase.response);
        },
      };
      const patchedTarget = patchResourceMethod(resourcePrototype, testCase.spec);
      assert.ok(patchedTarget);

      await resourcePrototype[testCase.spec.method](testCase.request).then((value) => value);

      assert.equal(captureState.spans.length, 1);
      const emitted = captureState.spans[0];
      const serialized = serializedSpan(emitted);
      assert.equal(serialized.scopeName, "@respan/instrumentation-together-ai");
      assert.equal(
        serializedAttribute(serialized, "respan.entity.log_type"),
        testCase.spec.logType,
      );
      assert.equal(
        serializedAttribute(serialized, "llm.request.type"),
        testCase.spec.requestType,
      );
      assert.equal(
        serializedAttribute(serialized, "gen_ai.request.model"),
        testCase.request.model,
      );
      assert.ok(serializedAttribute(serialized, "traceloop.entity.input"));
      assert.ok(serializedAttribute(serialized, "traceloop.entity.output"));

      if (testCase.usage) {
        assert.equal(
          Number(serializedAttribute(serialized, "gen_ai.usage.input_tokens")),
          testCase.usage.input,
        );
        assert.equal(
          Number(serializedAttribute(serialized, "gen_ai.usage.output_tokens")),
          testCase.usage.output,
        );
        assert.equal(
          Number(serializedAttribute(serialized, "llm.usage.total_tokens")),
          testCase.usage.total,
        );
      }

      for (const bannedKey of [
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
      ]) {
        assert.equal(serializedAttribute(serialized, bannedKey), undefined);
      }

      resourcePrototype[testCase.spec.method] = patchedTarget.original;
    });
  }
});
