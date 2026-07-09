import test from "node:test";
import assert from "node:assert/strict";

import { VercelAITranslator } from "../dist/_translator.js";

function runTranslator(name, attributes, options = {}) {
  const span = {
    name,
    instrumentationLibrary: options.instrumentationLibrary,
    instrumentationScope: options.instrumentationScope ?? { name: "ai" },
    attributes: { ...attributes },
  };
  const writableSpan = {
    name,
    instrumentationLibrary: options.instrumentationLibrary,
    instrumentationScope: options.instrumentationScope ?? { name: "ai" },
    setAttribute(key, value) {
      span.attributes[key] = value;
    },
  };

  const translator = new VercelAITranslator();
  translator.onStart(writableSpan, undefined);
  translator.onEnd(span);

  return span.attributes;
}

const offContractAliasKeys = [
  "tools",
  "tool_calls",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  "respan.span.tools",
  "respan.span.tool_calls",
  "respan.span.handoffs",
];

function assertNoOffContractAliases(attrs) {
  for (const key of offContractAliasKeys) {
    assert.equal(attrs[key], undefined, `${key} should be stripped`);
  }
}

function assertNoRawAIAttrs(attrs) {
  const rawKeys = Object.keys(attrs).filter((key) => key.startsWith("ai."));
  assert.deepEqual(rawKeys, []);
}

const baseLLMSpan = {
  "ai.model.id": "gpt-4o-mini",
  "ai.model.provider": "openai.chat",
  "ai.prompt.messages": JSON.stringify([{ role: "user", content: "hi" }]),
  "ai.response.text": "hello",
  "gen_ai.usage.input_tokens": 5,
  "gen_ai.usage.output_tokens": 3,
  "ai.usage.totalTokens": 8,
  "traceloop.span.kind": "task",
};

test("ai.generateText.doGenerate is classified as LLM text, not task", () => {
  const attrs = runTranslator("ai.generateText.doGenerate", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "hi");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "hello");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 5);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 3);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 5);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 3);
  assert.equal(attrs["llm.usage.total_tokens"], 8);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("ai.streamText.doStream is classified as LLM text, not task", () => {
  const attrs = runTranslator("ai.streamText.doStream", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["llm.is_streaming"], true);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("ai.response is classified as LLM text, not response", () => {
  const attrs = runTranslator("ai.response", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("parent streamText operation spans are enriched when they carry telemetry", () => {
  const attrs = runTranslator("ai.streamText", {
    "ai.model.id": "gpt-4o-mini",
    "ai.model.provider": "openai.chat",
    "ai.prompt.messages": JSON.stringify([{ role: "user", content: "stream this" }]),
    "ai.response.text": "streamed response",
    "ai.usage.promptTokens": 9,
    "ai.usage.completionTokens": 6,
    "ai.usage.cachedInputTokens": 2,
    "traceloop.span.kind": "task",
    model: "gpt-4o-mini",
    prompt_tokens: 9,
  });

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["llm.is_streaming"], true);
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.prompt.0.content"], "stream this");
  assert.equal(attrs["gen_ai.completion.0.content"], "streamed response");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 9);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 6);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 9);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 6);
  assert.equal(attrs["llm.usage.total_tokens"], 15);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 2);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("object generation spans map object output into canonical completion fields", () => {
  const responseObject = { answer: "Tokyo", confidence: 0.98 };
  const attrs = runTranslator("ai.generateObject.doGenerate", {
    "ai.model.id": "gpt-4o-mini",
    "ai.model.provider": "openai.chat",
    "ai.prompt": "return JSON",
    "ai.response.object": JSON.stringify(responseObject),
    "ai.usage.inputTokens": 11,
    "ai.usage.outputTokens": 7,
    "ai.schema": JSON.stringify({ type: "object" }),
    "traceloop.span.kind": "task",
  });

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "return JSON");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], JSON.stringify(responseObject));
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    role: "assistant",
    content: JSON.stringify(responseObject),
  });
  assert.equal(attrs["gen_ai.usage.input_tokens"], 11);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
  assert.equal(attrs["llm.usage.total_tokens"], 18);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("streamObject parent operation spans map object output and streaming state", () => {
  const responseObject = { status: "ok" };
  const attrs = runTranslator("ai.streamObject", {
    "ai.model.id": "gpt-4o-mini",
    "ai.prompt.messages": JSON.stringify([{ role: "user", content: "stream JSON" }]),
    "ai.response.object": JSON.stringify(responseObject),
    "ai.usage.promptTokens": 4,
    "ai.usage.completionTokens": 5,
    "traceloop.span.kind": "task",
  });

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["llm.is_streaming"], true);
  assert.equal(attrs["gen_ai.prompt.0.content"], "stream JSON");
  assert.equal(attrs["gen_ai.completion.0.content"], JSON.stringify(responseObject));
  assert.equal(attrs["gen_ai.usage.input_tokens"], 4);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 5);
  assert.equal(attrs["llm.usage.total_tokens"], 9);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("ai.embed.doEmbed is classified as embedding without synthetic usage fields", () => {
  const attrs = runTranslator("ai.embed.doEmbed", {
    "ai.model.id": "text-embedding-3-small",
    "ai.values": [JSON.stringify("embed this")],
    "ai.embeddings": [JSON.stringify([0.1, 0.2, 0.3])],
    "ai.usage.tokens": 7,
    "traceloop.span.kind": "task",
  });

  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.request.model"], "text-embedding-3-small");
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], undefined);
  assert.equal(attrs["llm.model_name"], undefined);
  assert.equal(attrs.model, undefined);
  assert.equal(attrs["llm.token_count.prompt"], undefined);
  assert.equal(attrs.prompt_tokens, undefined);
  assert.equal(attrs.total_request_tokens, undefined);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assert.equal(attrs["ai.usage.tokens"], undefined);
  assert.equal(attrs["ai.embeddings"], undefined);

  // The embedded text and the vector are captured into input/output (not
  // dropped) — debuggable RAG data; vendor keys are remapped then stripped.
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), ["embed this"]);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [[0.1, 0.2, 0.3]]);
  assert.equal(attrs["ai.values"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("parent embedMany operation spans capture values and vectors then strip raw ai keys", () => {
  const attrs = runTranslator("ai.embedMany", {
    "ai.model.id": "text-embedding-3-small",
    "ai.model.provider": "openai.embedding",
    "ai.values": [JSON.stringify("first"), JSON.stringify("second")],
    "ai.embeddings": [JSON.stringify([0.1, 0.2]), JSON.stringify([0.3, 0.4])],
    "ai.usage.tokens": 99,
    "traceloop.span.kind": "task",
  });

  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.request.model"], "text-embedding-3-small");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), ["first", "second"]);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [[0.1, 0.2], [0.3, 0.4]]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], undefined);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("LLM spans emit tool definitions and tool calls in canonical fields only", () => {
  const tool = {
    type: "function",
    name: "weather",
    description: "Return weather.",
    inputSchema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  };
  const toolCall = {
    id: "call_weather",
    type: "function",
    function: {
      name: "weather",
      arguments: JSON.stringify({ city: "Tokyo" }),
    },
  };

  const attrs = runTranslator("ai.generateText.doGenerate", {
    "ai.model.id": "gpt-4o",
    "ai.prompt.messages": JSON.stringify([{ role: "user", content: "weather in Tokyo" }]),
    "ai.prompt.tools": [JSON.stringify(tool)],
    "ai.response.toolCalls": JSON.stringify([toolCall]),
    "gen_ai.usage.input_tokens": 12,
    "gen_ai.usage.output_tokens": 4,
    tools: [tool],
    tool_calls: [toolCall],
    "respan.span.tools": [tool],
    "respan.span.tool_calls": [toolCall],
    span_tools: [tool],
    has_tool_calls: true,
    parallel_tool_calls: false,
  });

  const expectedTools = [
    {
      type: "function",
      function: {
        name: "weather",
        description: "Return weather.",
        parameters: tool.inputSchema,
      },
    },
  ];

  // Canonical fields only (per docs/SPAN_CONTRACT.md):
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "weather in Tokyo");
  assert.equal(JSON.parse(attrs["llm.request.functions"]).length, 1);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), expectedTools);
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [toolCall]);

  // Off-contract aliases must NOT be set:
  assertNoOffContractAliases(attrs);

  // Vendor-specific raw attrs stripped:
  assert.equal(attrs["ai.prompt.tools"], undefined);
  assert.equal(attrs["ai.response.toolCalls"], undefined);
  assertNoRawAIAttrs(attrs);
});

test("final text step does not echo prompt-history tool calls into completion", () => {
  const toolCall = {
    id: "call_weather",
    type: "function",
    function: {
      name: "weather",
      arguments: JSON.stringify({ city: "Tokyo" }),
    },
  };

  const attrs = runTranslator("ai.generateText.doGenerate", {
    "ai.model.id": "gpt-4o",
    "ai.prompt.messages": JSON.stringify([
      { role: "user", content: "weather in Tokyo" },
      { role: "assistant", content: "", tool_calls: [toolCall] },
      {
        role: "tool",
        tool_call_id: "call_weather",
        name: "weather",
        content: JSON.stringify({ city: "Tokyo", condition: "clear" }),
      },
    ]),
    "ai.response.text": "Tokyo is clear.",
  });

  // This turn emitted plain text, not a new tool call. The assistant's
  // earlier tool_calls remain in the prompt history (gen_ai.prompt.*),
  // not on this span's completion / top-level tool_calls fields.
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs["gen_ai.completion.0.tool_calls"], undefined);
  assert.equal(attrs.has_tool_calls, undefined);
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.1.tool_calls"]), [toolCall]);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    role: "assistant",
    content: "Tokyo is clear.",
  });
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("ai.toolCall spans carry input/output only — no tool_calls aliases", () => {
  const attrs = runTranslator("ai.toolCall", {
    "ai.toolCall.id": "call_weather",
    "ai.toolCall.name": "weather",
    "ai.toolCall.args": JSON.stringify({ city: "Tokyo" }),
    "ai.toolCall.result": JSON.stringify({ city: "Tokyo", condition: "clear" }),
  });

  // The span's existence + log_type=tool IS the tool call.
  assert.equal(attrs["respan.entity.log_type"], "tool");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), {
    name: "weather",
    arguments: { city: "Tokyo" },
  });
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    city: "Tokyo",
    condition: "clear",
  });

  // Tool execution spans must NOT carry tool_calls aliases:
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.span_tools, undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("AI SDK 7 chat spans map gen_ai message parts into canonical chat fields", () => {
  const tool = {
    type: "function",
    name: "weather",
    description: "Return weather.",
    inputSchema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  };
  const attrs = runTranslator(
    "chat gpt-4o-mini",
    {
      "gen_ai.operation.name": "chat",
      "gen_ai.provider.name": "openai",
      "gen_ai.request.model": "gpt-4o-mini",
      "gen_ai.input.messages": JSON.stringify([
        { role: "user", parts: [{ type: "text", content: "weather in Tokyo" }] },
      ]),
      "gen_ai.output.messages": JSON.stringify([
        {
          role: "assistant",
          parts: [
            { type: "text", content: "Calling weather." },
            { type: "tool_call", id: "call_weather", name: "weather", arguments: { city: "Tokyo" } },
          ],
          finish_reason: "tool_calls",
        },
      ]),
      "gen_ai.tool.definitions": JSON.stringify([tool]),
      "gen_ai.usage.input_tokens": 12,
      "gen_ai.usage.output_tokens": 4,
      "traceloop.span.kind": "task",
    },
    { instrumentationScope: { name: "gen_ai" } },
  );

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "weather in Tokyo");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Calling weather.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      type: "function",
      id: "call_weather",
      function: {
        name: "weather",
        arguments: JSON.stringify({ city: "Tokyo" }),
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "weather",
        description: "Return weather.",
        parameters: tool.inputSchema,
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 4);
  assert.equal(attrs["llm.usage.total_tokens"], 16);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});

test("AI SDK 7 execute_tool spans carry input/output only", () => {
  const attrs = runTranslator(
    "execute_tool weather",
    {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "weather",
      "gen_ai.tool.call.id": "call_weather",
      "gen_ai.tool.type": "function",
      "gen_ai.tool.call.arguments": JSON.stringify({ city: "Tokyo" }),
      "gen_ai.tool.call.result": JSON.stringify({ city: "Tokyo", condition: "clear" }),
    },
    { instrumentationScope: { name: "gen_ai" } },
  );

  assert.equal(attrs["respan.entity.log_type"], "tool");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), {
    name: "weather",
    arguments: { city: "Tokyo" },
  });
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    city: "Tokyo",
    condition: "clear",
  });
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assertNoRawAIAttrs(attrs);
  assertNoOffContractAliases(attrs);
});
