import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { BeeAIInstrumentation as OpenInferenceBeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
import * as beeaiFramework from "beeai-framework";
import { DummyChatModel } from "beeai-framework/adapters/dummy/backend/chat";
import { ChatModelOutput } from "beeai-framework/backend/chat";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";

import { BeeAIInstrumentor } from "../dist/index.js";

function resetGlobalTracerProvider(provider) {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  if (provider) {
    trace.setGlobalTracerProvider(provider);
  }
}

test("real Arize BeeAI spans retain late OTel 2.x input, model, and usage", async () => {
  const exporter = new InMemorySpanExporter();
  const onStartSnapshots = [];
  const startAuditProcessor = {
    onStart(span) {
      onStartSnapshots.push({
        name: span.name,
        attributes: { ...span.attributes },
      });
    },
    onEnd() {},
    async forceFlush() {},
    async shutdown() {},
  };
  const provider = new BasicTracerProvider({
    spanProcessors: [
      startAuditProcessor,
      new SimpleSpanProcessor(exporter),
    ],
  });
  const activeSpanProcessor = provider._activeSpanProcessor;
  const originalProcessorOnEnd = activeSpanProcessor.onEnd;
  resetGlobalTracerProvider(provider);

  const instrumentor = new BeeAIInstrumentor({
    sdkModule: beeaiFramework,
    instrumentationClass: OpenInferenceBeeAIInstrumentation,
  });

  try {
    await instrumentor.activate();

    const prompt = "Explain why tracing helps debugging.";
    const answer = "Tracing preserves each step and value.";
    const model = new DummyChatModel("gpt-4o-mini");
    model._create = async () => new ChatModelOutput(
      [new AssistantMessage(answer)],
      { promptTokens: 11, completionTokens: 7, totalTokens: 18 },
      "stop",
    );

    const response = await model.create({
      messages: [new UserMessage(prompt)],
    });
    assert.equal(response.getTextContent(), answer);

    await provider.forceFlush();

    const startSnapshot = onStartSnapshots.find((span) =>
      span.name.startsWith("backend.dummy.chat.start"),
    );
    assert.ok(startSnapshot, "expected the real BeeAI chat start span");
    assert.equal(startSnapshot.attributes.target, undefined);
    assert.equal(startSnapshot.attributes["input.value"], undefined);

    const successSpan = exporter.getFinishedSpans().find((span) =>
      span.attributes["respan.entity.log_type"] === "chat" &&
      span.attributes["traceloop.entity.name"] === "backend.dummy.chat.success",
    );
    assert.ok(successSpan, "expected the translated BeeAI chat success span");

    const attrs = successSpan.attributes;
    assert.equal(attrs["respan.entity.log_type"], "chat");
    assert.equal(attrs["llm.request.type"], "chat");
    assert.equal(attrs["gen_ai.system"], "dummy");
    assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(attrs["gen_ai.usage.input_tokens"], 11);
    assert.equal(attrs["gen_ai.usage.prompt_tokens"], 11);
    assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
    assert.equal(attrs["gen_ai.usage.completion_tokens"], 7);
    assert.equal(attrs["llm.usage.total_tokens"], 18);
    assert.equal(
      attrs["traceloop.entity.input"],
      JSON.stringify([{ role: "user", content: prompt }]),
    );
    assert.equal(
      attrs["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: answer }),
    );
    assert.equal(attrs["gen_ai.prompt.0.role"], "user");
    assert.equal(attrs["gen_ai.prompt.0.content"], prompt);
    assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
    assert.equal(attrs["gen_ai.completion.0.content"], answer);
    for (const rawAttribute of [
      "target",
      "traceId",
      "input.value",
      "output.value",
      "llm.model_name",
      "llm.provider",
      "llm.system",
      "llm.token_count.prompt",
      "llm.token_count.completion",
      "llm.token_count.total",
      "metadata.model_name",
    ]) {
      assert.equal(attrs[rawAttribute], undefined, `exported raw ${rawAttribute}`);
    }
  } finally {
    instrumentor.deactivate();
    try {
      assert.equal(
        activeSpanProcessor.onEnd,
        originalProcessorOnEnd,
        "deactivation must fully restore the pre-instrumentation processor hook",
      );
    } finally {
      await provider.shutdown();
      resetGlobalTracerProvider();
    }
  }
});
