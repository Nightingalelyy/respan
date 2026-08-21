import assert from "node:assert/strict";
import { AsyncLocalStorage } from "node:async_hooks";
import test from "node:test";

import { context, ROOT_CONTEXT, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { BeeAIInstrumentation as OpenInferenceBeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
import * as beeaiFramework from "beeai-framework";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";
import { DummyChatModel } from "beeai-framework/adapters/dummy/backend/chat";
import { ChatModelOutput } from "beeai-framework/backend/chat";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { CalculatorTool } from "beeai-framework/tools/calculator";

import { BeeAIInstrumentor } from "../dist/index.js";

class TestAsyncLocalContextManager {
  storage = new AsyncLocalStorage();

  active() {
    return this.storage.getStore() ?? ROOT_CONTEXT;
  }

  with(activeContext, fn, thisArg, ...args) {
    return this.storage.run(activeContext, () => fn.call(thisArg, ...args));
  }

  bind(activeContext, target) {
    if (typeof target !== "function") return target;
    return (...args) => this.with(activeContext, target, undefined, ...args);
  }

  enable() {
    return this;
  }

  disable() {
    this.storage.disable();
    return this;
  }
}

function resetGlobalTelemetry(provider, contextManager) {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  if (typeof context.disable === "function") {
    context.disable();
  }
  if (contextManager) {
    context.setGlobalContextManager(contextManager.enable());
  }
  if (provider) {
    trace.setGlobalTracerProvider(provider);
  }
}

function getParentSpanId(span) {
  return span.parentSpanId ?? span.parentSpanContext?.spanId;
}

test("real Arize BeeAI spans retain chat data and one complete ToolCallingAgent tree", async () => {
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
  const contextManager = new TestAsyncLocalContextManager();
  resetGlobalTelemetry(provider, contextManager);

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

    let agentModelCall = 0;
    const agentModel = new DummyChatModel("gpt-4o-mini");
    agentModel._create = async () => {
      agentModelCall += 1;
      const message = agentModelCall === 1
        ? new AssistantMessage({
            type: "tool-call",
            toolCallId: "call-calculator",
            toolName: "Calculator",
            args: { expression: "(19 + 23) * 2" },
          })
        : new AssistantMessage({
            type: "tool-call",
            toolCallId: "call-final-answer",
            toolName: "final_answer",
            args: { response: "84" },
          });
      return new ChatModelOutput(
        [message],
        {
          promptTokens: agentModelCall * 10,
          completionTokens: 2,
          totalTokens: agentModelCall * 10 + 2,
        },
        "tool-call",
      );
    };

    const agent = new ToolCallingAgent({
      llm: agentModel,
      memory: new UnconstrainedMemory(),
      tools: [new CalculatorTool()],
    });
    const agentPrompt = "Compute (19 + 23) * 2";
    const workflowTracer = trace.getTracer("beeai-otel2-integration");
    const agentResult = await workflowTracer.startActiveSpan(
      "beeai_tool_calling_agent.workflow.workflow",
      { attributes: { "traceloop.span.kind": "workflow" } },
      async (workflowSpan) => {
        try {
          return await agent.run({ prompt: agentPrompt });
        } finally {
          workflowSpan.end();
        }
      },
    );
    assert.equal(agentResult.result.text, "84");

    await provider.forceFlush();

    const finishedSpans = exporter.getFinishedSpans();
    const visibleSpans = finishedSpans.filter(
      (span) => !(
        Array.isArray(span.attributes["respan.processors"]) &&
        span.attributes["respan.processors"].length === 0
      ),
    );
    const workflowSpan = visibleSpans.find(
      (span) => span.name === "beeai_tool_calling_agent.workflow.workflow",
    );
    assert.ok(workflowSpan, "expected the enclosing workflow span");

    const agentTraceId = workflowSpan.spanContext().traceId;
    const allAgentTraceSpans = finishedSpans.filter(
      (span) => span.spanContext().traceId === agentTraceId,
    );
    const agentTraceSpans = visibleSpans.filter(
      (span) => span.spanContext().traceId === agentTraceId,
    );
    const agentSpans = agentTraceSpans.filter(
      (span) => span.attributes["respan.entity.log_type"] === "agent",
    );
    assert.equal(agentSpans.length, 1, "expected one recovered agent span");

    const agentSpan = agentSpans[0];
    assert.equal(agentSpan.name, "beeai-framework-main");
    assert.equal(getParentSpanId(agentSpan), workflowSpan.spanContext().spanId);
    assert.equal(agentSpan.attributes["traceloop.entity.name"], "ToolCallingAgent");
    const canonicalAgentInput = JSON.parse(
      agentSpan.attributes["traceloop.entity.input"],
    );
    assert.equal(canonicalAgentInput.prompt, agentPrompt);
    assert.deepEqual(
      canonicalAgentInput.history.map((message) => message.role),
      ["system", "user", "assistant", "tool", "assistant", "tool"],
    );
    assert.match(canonicalAgentInput.history[0].content, /final_answer/);
    assert.equal(
      canonicalAgentInput.history[1].content,
      `Your task: ${agentPrompt}\n`,
    );
    assert.deepEqual(
      canonicalAgentInput.history.slice(2).map((message) => message.content),
      ["", "", "", ""],
    );
    assert.deepEqual(
      JSON.parse(agentSpan.attributes["traceloop.entity.output"]),
      { role: "assistant", content: "84" },
    );
    for (const rawAttribute of [
      "input.value",
      "output.value",
      "history",
      "source",
      "beeai.version",
      "traceId",
    ]) {
      assert.equal(agentSpan.attributes[rawAttribute], undefined, `exported raw ${rawAttribute}`);
    }

    const agentChildren = agentTraceSpans.filter((span) =>
      ["chat", "tool"].includes(span.attributes["respan.entity.log_type"]),
    );
    assert.equal(
      agentChildren.filter((span) => span.attributes["respan.entity.log_type"] === "chat").length,
      2,
    );
    assert.equal(
      agentChildren.filter((span) => span.attributes["respan.entity.log_type"] === "tool").length,
      2,
    );
    for (const child of agentChildren) {
      assert.equal(
        getParentSpanId(child),
        agentSpan.spanContext().spanId,
        `${child.name} should be a direct agent child`,
      );
    }
    assert.equal(
      agentTraceSpans.filter((span) => span.name.startsWith("agent.toolCalling.")).length,
      0,
      "iteration callbacks must not create duplicate agent spans",
    );
    assert.equal(
      agentTraceSpans.filter((span) => span.name === "beeai-framework-main").length,
      1,
      "the promoted wrapper must be the sole agent boundary",
    );
    const upstreamSuccessCallbacks = allAgentTraceSpans.filter(
      (span) => span.name.startsWith("agent.toolCalling.success"),
    );
    assert.equal(
      upstreamSuccessCallbacks.length,
      2,
      "the real two-iteration callback path should be exercised",
    );
    for (const callbackSpan of upstreamSuccessCallbacks) {
      assert.deepEqual(callbackSpan.attributes["respan.processors"], []);
      assert.equal(callbackSpan.attributes["traceloop.entity.input"], undefined);
      assert.equal(callbackSpan.attributes["traceloop.entity.output"], undefined);
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
      contextManager.disable();
      resetGlobalTelemetry();
    }
  }
});
