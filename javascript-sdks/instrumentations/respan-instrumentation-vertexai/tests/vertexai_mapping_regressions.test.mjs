import assert from "node:assert/strict";
import test from "node:test";
import * as genai from "@google/genai";
import { context, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SimpleSpanProcessor,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { VertexAIInstrumentor } from "../dist/index.js";

test("native GenAI JSON schemas, function thought signature and multimodal function response survive mapping", async () => {
  trace.disable();
  context.disable();
  context.setGlobalContextManager(
    new AsyncLocalStorageContextManager().enable(),
  );
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.setGlobalTracerProvider(provider);
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    requests.push(JSON.parse(options.body));
    return Response.json({
      candidates: [
        {
          index: 0,
          finishReason: "STOP",
          content: { role: "model", parts: [{ text: "controlled answer" }] },
        },
      ],
      usageMetadata: {
        promptTokenCount: 4,
        candidatesTokenCount: 5,
        totalTokenCount: 17,
      },
    });
  };
  const plugin = new VertexAIInstrumentor({ genaiModule: genai });
  const parametersJsonSchema = {
    type: "object",
    properties: { x: { type: "boolean" } },
    required: ["x"],
  };
  const responseJsonSchema = {
    type: "object",
    properties: { x: { type: "boolean" } },
  };
  const responseParts = [
    { inlineData: { data: "fixture-image", mimeType: "image/png" } },
  ];
  try {
    await plugin.activate();
    const client = new genai.GoogleGenAI({
      vertexai: true,
      project: "fixture-project",
      location: "us-central1",
      googleAuthOptions: {
        authClient: {
          async getAccessToken() {
            return { token: "fixture-only" };
          },
          async getRequestHeaders() {
            return new Headers({ Authorization: "Bearer fixture-only" });
          },
        },
      },
    });
    await client.models.generateContent({
      model: "fixture-model",
      contents: [
        {
          role: "model",
          parts: [
            {
              functionCall: {
                id: "call-native",
                name: "f",
                args: { x: false },
              },
              thoughtSignature: "native-thought-signature",
            },
          ],
        },
        {
          role: "user",
          parts: [
            {
              functionResponse: {
                id: "call-native",
                name: "f",
                response: { x: false },
                parts: responseParts,
              },
            },
          ],
        },
      ],
      config: {
        tools: [
          {
            functionDeclarations: [
              { name: "f", parametersJsonSchema, responseJsonSchema },
            ],
          },
        ],
      },
    });
    assert.equal(requests.length, 1);
    const declaration = requests[0].tools[0].functionDeclarations[0];
    assert.deepEqual(declaration.parametersJsonSchema, parametersJsonSchema);
    assert.deepEqual(declaration.responseJsonSchema, responseJsonSchema);
    assert.equal(
      requests[0].contents[0].parts[0].thoughtSignature,
      "native-thought-signature",
    );
    assert.deepEqual(
      requests[0].contents[1].parts[0].functionResponse.parts,
      responseParts,
    );
    const spans = exporter.getFinishedSpans();
    assert.equal(spans.length, 1);
    const attributes = spans[0].attributes;
    const canonical = JSON.parse(attributes["llm.request.functions"])[0]
      .function;
    assert.deepEqual(canonical.parameters, parametersJsonSchema);
    assert.deepEqual(canonical.responseJsonSchema, responseJsonSchema);
    const messages = JSON.parse(attributes["traceloop.entity.input"]);
    assert.equal(
      messages[0].content.thoughtSignature,
      "native-thought-signature",
    );
    assert.equal(messages[0].content.id, "call-native");
    assert.deepEqual(messages[1].content.function.parts, responseParts);
    assert.deepEqual(messages[1].content.function.response, { x: false });
    assert.ok(
      !spans.some(
        (span) => span.attributes["respan.entity.log_type"] === "tool",
      ),
    );
  } finally {
    plugin.deactivate();
    globalThis.fetch = originalFetch;
    await provider.shutdown();
    trace.disable();
    context.disable();
  }
});
