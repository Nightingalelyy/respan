import test from "node:test";
import assert from "node:assert/strict";
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { generateText, streamText, embedMany, rerank, registerTelemetry, Output, jsonSchema, tool } from "ai";
import { MockLanguageModelV4, MockEmbeddingModelV4, MockRerankingModelV4 } from "ai/test";
import { OpenTelemetry } from "@ai-sdk/otel";
import { VercelAITranslator } from "../../dist/_translator.js";

const usage = {
  inputTokens: { total: 12, noCache: 10, cacheRead: 2, cacheWrite: 0 },
  outputTokens: { total: 7, text: 5, reasoning: 2 },
};
const finishReason = { unified: "stop", raw: "stop" };
const response = {
  content: [
    { type: "reasoning", text: "fixture reasoning" },
    { type: "text", text: "fixture answer" },
    { type: "file", mediaType: "image/png", data: { type: "data", data: "aGVsbG8=" } },
  ],
  finishReason, usage, warnings: [],
};

test("real AI SDK 7.0.112 and @ai-sdk/otel 1.0.112 emit compatible spans", async (t) => {
  const exporter = new InMemorySpanExporter();
  const translator = new VercelAITranslator();
  const provider = new BasicTracerProvider({ spanProcessors: [
    { onStart: (span, ctx) => translator.onStart(span, ctx), onEnd: span => translator.onEnd(span), forceFlush: async () => {}, shutdown: async () => {} },
    new SimpleSpanProcessor(exporter),
  ] });
  const integration = new OpenTelemetry({ tracer: provider.getTracer("gen_ai"), embedding: true, reranking: true });
  registerTelemetry(integration);
  const spans = () => exporter.getFinishedSpans();
  const chat = () => spans().find(span => span.name.startsWith("chat "));
  try {
    await t.test("separate instructions and mixed input/output media survive", async () => {
      await generateText({
        model: new MockLanguageModelV4({ doGenerate: response }),
        instructions: "system fixture",
        messages: [{ role: "user", content: [{ type: "text", text: "input fixture" }, { type: "file", mediaType: "image/png", data: Buffer.from("hello") }] }],
      });
      const attrs = chat().attributes;
      const input = JSON.parse(attrs["traceloop.entity.input"]);
      assert.equal(input[0].role, "system");
      assert.equal(input[0].content, "system fixture");
      assert.equal(input[1].content[1].content, "aGVsbG8=");
      const output = JSON.parse(attrs["traceloop.entity.output"]);
      assert.deepEqual(output.content.map(part => part.type), ["reasoning", "text", "blob"]);
      assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
      assert.equal(attrs["llm.is_streaming"], false);
    });
    exporter.reset();
    await t.test("stream spans use measured first-output latency", async () => {
      const result = streamText({ model: new MockLanguageModelV4({ doStream: { stream: new ReadableStream({ start(controller) {
        for (const part of [{ type: "stream-start", warnings: [] }, { type: "text-start", id: "1" }, { type: "text-delta", id: "1", delta: "stream fixture" }, { type: "text-end", id: "1" }, { type: "finish", finishReason, usage }]) controller.enqueue(part);
        controller.close();
      } }) } }), prompt: "stream fixture" });
      assert.equal(await result.text, "stream fixture");
      assert.equal(chat().attributes["llm.is_streaming"], true);
      assert.ok(Number(JSON.parse(chat().attributes["respan.metadata"]).time_to_first_token) >= 0);
    });
    exporter.reset();
    await t.test("rerank records documents and ordering", async () => {
      await rerank({ model: new MockRerankingModelV4({ doRerank: async () => ({ ranking: [{ index: 1, relevanceScore: 0.9 }] }) }), query: "second", documents: ["first", "second"] });
      const span = spans().find(span => span.attributes["traceloop.entity.output"]);
      assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.input"]), { documents: ["first", "second"] });
      assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.output"]), [{ index: 1, relevanceScore: 0.9 }]);
    });
    exporter.reset();
    await t.test("structured Output.object retains the final JSON", async () => {
      const result = await generateText({
        model: new MockLanguageModelV4({ doGenerate: { ...response, content: [{ type: "text", text: '{"answer":42}' }] } }),
        prompt: "Return JSON",
        output: Output.object({ schema: jsonSchema({ type: "object", properties: { answer: { type: "number" } }, required: ["answer"], additionalProperties: false }) }),
      });
      assert.deepEqual(result.output, { answer: 42 });
      assert.equal(JSON.parse(chat().attributes["traceloop.entity.output"]).content, '{"answer":42}');
    });
    exporter.reset();
    await t.test("approval-required tools retain the call without a false execution span", async () => {
      let executed = false;
      const result = await generateText({
        model: new MockLanguageModelV4({ doGenerate: { ...response, content: [{ type: "tool-call", toolCallId: "approval-call", toolName: "lookup", input: "{}" }], finishReason: { unified: "tool-calls", raw: "tool_calls" } } }),
        prompt: "Request approval",
        tools: { lookup: tool({ inputSchema: jsonSchema({ type: "object", properties: {} }), execute: async () => { executed = true; return "ok"; } }) },
        toolApproval: { lookup: "user-approval" },
      });
      assert.equal(executed, false);
      assert.ok(result.content.some(part => part.type === "tool-approval-request"));
      assert.equal(JSON.parse(chat().attributes["gen_ai.completion.0.tool_calls"])[0].id, "approval-call");
      assert.ok(!spans().some(span => span.attributes["respan.entity.log_type"] === "tool"));
    });
    exporter.reset();
    await t.test("embedding batches retain input and vectors", async () => {
      await embedMany({ model: new MockEmbeddingModelV4({ maxEmbeddingsPerCall: 10, doEmbed: { embeddings: [[0.1, 0.2], [0.3, 0.4]], usage: { tokens: 2 } } }), values: ["first", "second"] });
      assert.ok(spans().some(span => span.attributes["traceloop.entity.output"] === "[[0.1,0.2],[0.3,0.4]]"));
    });
    exporter.reset();
    await t.test("SDK content opt-out is respected", async () => {
      await generateText({ model: new MockLanguageModelV4({ doGenerate: response }), instructions: "PRIVATE_SYSTEM", prompt: "PRIVATE_PROMPT", telemetry: { recordInputs: false, recordOutputs: false } });
      assert.ok(spans().length > 0);
      for (const span of spans()) {
        assert.equal(span.attributes["traceloop.entity.input"], undefined);
        assert.equal(span.attributes["traceloop.entity.output"], undefined);
        assert.ok(!JSON.stringify(span.attributes).includes("PRIVATE_"));
      }
    });
    exporter.reset();
    await t.test("failed model calls retain OTel error status and exception content", async () => {
      await assert.rejects(generateText({ model: new MockLanguageModelV4({ doGenerate: async () => { throw new Error("fixture failure"); } }), prompt: "fail", maxRetries: 0 }), /fixture failure/);
      assert.equal(chat().status.code, 2);
      assert.equal(chat().status.message, "fixture failure");
      assert.ok(chat().events.some(event => event.attributes?.["exception.message"] === "fixture failure"));
    });
  } finally {
    const integrations = globalThis.AI_SDK_TELEMETRY_INTEGRATIONS;
    const index = integrations?.indexOf(integration) ?? -1;
    if (index >= 0) integrations.splice(index, 1);
    await provider.shutdown();
  }
});
