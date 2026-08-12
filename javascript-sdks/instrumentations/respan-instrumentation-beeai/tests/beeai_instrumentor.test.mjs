import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { BeeAIInstrumentor } from "../dist/index.js";

const BEEAI_SCOPE_NAME = "@arizeai/openinference-instrumentation-beeai";

function makeSpan({
  name = "test-span",
  attributes = {},
  instrumentationScopeName = BEEAI_SCOPE_NAME,
  instrumentationLibraryName,
  traceId = "trace-test",
  spanId = `${name}-span`,
  parentSpanId,
} = {}) {
  const span = {
    name,
    parentSpanId,
    spanContext() {
      return {
        traceId,
        spanId,
        traceFlags: 1,
      };
    },
    attributes: { ...attributes },
  };

  if (instrumentationScopeName) {
    span.instrumentationScope = {
      name: instrumentationScopeName,
      version: "1.0.0",
    };
  }
  if (instrumentationLibraryName) {
    span.instrumentationLibrary = {
      name: instrumentationLibraryName,
      version: "1.0.0",
    };
  }

  return span;
}

function resetTracerProvider(provider) {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  if (provider) {
    trace.setGlobalTracerProvider(provider);
  }
}

function createFakeTracerProvider(processor) {
  return {
    activeSpanProcessor: processor,
    getTracer() {
      return {
        startSpan() {
          throw new Error("startSpan should not be called in this test");
        },
      };
    },
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

test("BeeAIInstrumentor delegates activation with the provided BeeAI module", async () => {
  class FakeBeeAIInstrumentation {}

  const calls = [];
  const sdkModule = { BeeAgent: class BeeAgent {} };
  const delegate = {
    activate() {
      calls.push(["activate"]);
    },
    deactivate() {
      calls.push(["deactivate"]);
    },
  };

  const instrumentor = new BeeAIInstrumentor({
    sdkModule,
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory(instrumentationClass, module) {
      calls.push(["factory", instrumentationClass, module]);
      return delegate;
    },
  });

  await instrumentor.activate();
  await instrumentor.activate();
  instrumentor.deactivate();

  assert.deepEqual(calls, [
    ["factory", FakeBeeAIInstrumentation, sdkModule],
    ["activate"],
    ["deactivate"],
  ]);
});

test("BeeAIInstrumentor coalesces concurrent activation", async () => {
  class FakeBeeAIInstrumentation {}

  const delegateGate = deferred();
  const calls = [];
  let factoryCalls = 0;
  const delegate = {
    activate() {
      calls.push("activate");
    },
    deactivate() {
      calls.push("deactivate");
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      factoryCalls += 1;
      return delegateGate.promise;
    },
  });

  const firstActivation = instrumentor.activate();
  const secondActivation = instrumentor.activate();
  assert.equal(factoryCalls, 1);

  delegateGate.resolve(delegate);
  await Promise.all([firstActivation, secondActivation]);
  assert.deepEqual(calls, ["activate"]);

  instrumentor.deactivate();
  assert.deepEqual(calls, ["activate", "deactivate"]);
});

test("BeeAIInstrumentor cancels a pending activation and can reactivate", async () => {
  class FakeBeeAIInstrumentation {}

  const firstDelegateGate = deferred();
  const calls = [];
  let factoryCalls = 0;
  const delegate = {
    activate() {
      calls.push("activate");
    },
    deactivate() {
      calls.push("deactivate");
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      factoryCalls += 1;
      return factoryCalls === 1 ? firstDelegateGate.promise : delegate;
    },
  });

  const cancelledActivation = instrumentor.activate();
  instrumentor.deactivate();
  firstDelegateGate.resolve(delegate);
  await cancelledActivation;
  assert.deepEqual(calls, []);

  await instrumentor.activate();
  assert.equal(factoryCalls, 2);
  assert.deepEqual(calls, ["activate"]);

  instrumentor.deactivate();
  assert.deepEqual(calls, ["activate", "deactivate"]);
});

test("BeeAIInstrumentor rolls back activation errors and permits retry", async () => {
  class FakeBeeAIInstrumentation {}

  const calls = [];
  let factoryCalls = 0;
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      factoryCalls += 1;
      const attempt = factoryCalls;
      return {
        activate() {
          calls.push(`activate-${attempt}`);
          if (attempt === 1) throw new Error("activation failed");
        },
        deactivate() {
          calls.push(`deactivate-${attempt}`);
        },
      };
    },
  });

  await assert.rejects(instrumentor.activate(), /activation failed/);
  await instrumentor.activate();
  instrumentor.deactivate();

  assert.deepEqual(calls, [
    "activate-1",
    "deactivate-1",
    "activate-2",
    "deactivate-2",
  ]);
});

test("BeeAIInstrumentor drains spans that started before deactivation", async () => {
  class FakeBeeAIInstrumentation {}

  const capturedSpans = [];
  const processor = {
    onStart() {},
    onEnd(span) {
      capturedSpans.push(span);
    },
  };
  const originalOnEnd = processor.onEnd;
  resetTracerProvider(createFakeTracerProvider(processor));

  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory: () => ({ activate() {}, deactivate() {} }),
  });

  try {
    await instrumentor.activate();
    const translatedAfterDeactivate = makeSpan({
      name: "backend.openai.chat.success-drain",
      attributes: {
        target: "backend.openai.chat.success",
        traceId: "vendor-drain-trace",
        "input.value": JSON.stringify([{ role: "user", content: "hello" }]),
        "output.value": JSON.stringify([{ role: "assistant", content: "hi" }]),
        "llm.model_name": "gpt-4o-mini",
      },
    });

    processor.onStart(translatedAfterDeactivate, {});
    const drainingOnEnd = processor.onEnd;
    instrumentor.deactivate();
    assert.equal(processor.onEnd, drainingOnEnd);

    processor.onEnd(translatedAfterDeactivate);
    assert.equal(processor.onEnd, originalOnEnd);
    assert.deepEqual(capturedSpans, [translatedAfterDeactivate]);
    assert.equal(translatedAfterDeactivate.attributes["respan.entity.log_type"], "chat");
    assert.equal(translatedAfterDeactivate.attributes["gen_ai.system"], "openai");
    assert.equal(translatedAfterDeactivate.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(translatedAfterDeactivate.attributes.target, undefined);
    assert.equal(translatedAfterDeactivate.attributes.traceId, undefined);
    assert.equal(translatedAfterDeactivate.attributes["llm.model_name"], undefined);
  } finally {
    instrumentor.deactivate();
    resetTracerProvider();
  }
});

test("BeeAIInstrumentor preserves processor wrappers installed after activation", async () => {
  class FakeBeeAIInstrumentation {}

  const processor = {
    onStart() {},
    onEnd() {},
  };
  resetTracerProvider(createFakeTracerProvider(processor));

  let nestedOriginalOnEnd;
  let nestedOnEnd;
  const delegate = {
    activate() {
      nestedOriginalOnEnd = processor.onEnd;
      nestedOnEnd = (span) => nestedOriginalOnEnd.call(processor, span);
      processor.onEnd = nestedOnEnd;
    },
    deactivate() {
      if (processor.onEnd === nestedOnEnd) {
        processor.onEnd = nestedOriginalOnEnd;
      }
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory: () => delegate,
  });
  const warnings = [];
  const originalWarn = console.warn;

  try {
    await instrumentor.activate();
    const beeAIOnStart = processor.onStart;
    const beeAIOnEnd = processor.onEnd;
    const externalOnStart = (span, context) => beeAIOnStart.call(processor, span, context);
    const externalOnEnd = (span) => beeAIOnEnd.call(processor, span);
    processor.onStart = externalOnStart;
    processor.onEnd = externalOnEnd;
    console.warn = (...args) => warnings.push(args);

    instrumentor.deactivate();

    assert.equal(processor.onStart, externalOnStart);
    assert.equal(processor.onEnd, externalOnEnd);
    assert.ok(warnings.length >= 1, "external hook conflict should be reported");
  } finally {
    console.warn = originalWarn;
    instrumentor.deactivate();
    resetTracerProvider();
  }
});

test("BeeAIInstrumentor bounds pending trace state without losing spans", async () => {
  class FakeBeeAIInstrumentation {}

  const capturedSpans = [];
  const processor = {
    onStart() {},
    onEnd(span) {
      capturedSpans.push(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory: () => ({ activate() {}, deactivate() {} }),
  });

  try {
    await instrumentor.activate();
    for (let index = 0; index < 300; index += 1) {
      const span = makeSpan({
        name: `backend.dummy.chat.success-${index}`,
        traceId: `otel-bounded-${index}`,
        spanId: `bounded-span-${index}`,
        attributes: {
          target: "backend.dummy.chat.success",
          traceId: `vendor-bounded-${index}`,
          "output.value": JSON.stringify([{ role: "assistant", content: String(index) }]),
          "llm.model_name": "gpt-4o-mini",
        },
      });
      processor.onStart(span, {});
      processor.onEnd(span);
    }

    assert.ok(capturedSpans.length > 0, "old pending traces should be evicted and exported");
    assert.ok(capturedSpans.length < 300, "recent pending traces should remain available");

    instrumentor.deactivate();
    assert.equal(capturedSpans.length, 300);
    for (const span of capturedSpans) {
      assert.equal(span.attributes.target, undefined);
      assert.equal(span.attributes.traceId, undefined);
      assert.equal(span.attributes["llm.model_name"], undefined);
    }
  } finally {
    instrumentor.deactivate();
    resetTracerProvider();
  }
});

test("BeeAIInstrumentor bounds one trace's pending span queue losslessly", async () => {
  class FakeBeeAIInstrumentation {}

  const capturedSpans = [];
  const processor = {
    onStart() {},
    onEnd(span) {
      capturedSpans.push(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory: () => ({ activate() {}, deactivate() {} }),
  });

  try {
    await instrumentor.activate();
    for (let index = 0; index < 100; index += 1) {
      const span = makeSpan({
        name: `backend.dummy.chat.success-shared-${index}`,
        traceId: "otel-shared-pending-trace",
        spanId: `shared-pending-span-${index}`,
        attributes: {
          target: "backend.dummy.chat.success",
          traceId: "vendor-shared-pending-trace",
          "output.value": JSON.stringify([{ role: "assistant", content: String(index) }]),
          "llm.model_name": "gpt-4o-mini",
        },
      });
      processor.onStart(span, {});
      processor.onEnd(span);
    }

    assert.ok(capturedSpans.length > 0, "old pending spans should be compacted and exported");
    assert.ok(capturedSpans.length < 100, "recent pending spans should remain correlatable");

    instrumentor.deactivate();
    assert.equal(capturedSpans.length, 100);
  } finally {
    instrumentor.deactivate();
    resetTracerProvider();
  }
});

test("BeeAIInstrumentor bounds dropped-parent history within one trace", async () => {
  class FakeBeeAIInstrumentation {}

  const capturedSpans = [];
  const processor = {
    onStart() {},
    onEnd(span) {
      if (span.attributes["respan.processors"]?.length === 0) return;
      capturedSpans.push(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: {},
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory: () => ({ activate() {}, deactivate() {} }),
  });

  try {
    await instrumentor.activate();
    for (let index = 0; index < 300; index += 1) {
      processor.onEnd(makeSpan({
        name: `backend.dummy.chat.start-bounded-${index}`,
        traceId: "otel-shared-parent-trace",
        spanId: `dropped-parent-${index}`,
        attributes: {
          target: "backend.dummy.chat.start",
          traceId: "vendor-shared-parent-trace",
        },
      }));
    }

    const childOfEvictedParent = makeSpan({
      name: "backend.dummy.chat.success-old-parent",
      traceId: "otel-shared-parent-trace",
      spanId: "old-parent-child",
      parentSpanId: "dropped-parent-0",
      attributes: {
        target: "backend.dummy.chat.success",
        traceId: "vendor-shared-parent-trace",
        "input.value": JSON.stringify([{ role: "user", content: "old" }]),
        "output.value": JSON.stringify([{ role: "assistant", content: "old" }]),
      },
    });
    const childOfRecentParent = makeSpan({
      name: "backend.dummy.chat.success-recent-parent",
      traceId: "otel-shared-parent-trace",
      spanId: "recent-parent-child",
      parentSpanId: "dropped-parent-299",
      attributes: {
        target: "backend.dummy.chat.success",
        traceId: "vendor-shared-parent-trace",
        "input.value": JSON.stringify([{ role: "user", content: "recent" }]),
        "output.value": JSON.stringify([{ role: "assistant", content: "recent" }]),
      },
    });
    processor.onEnd(childOfEvictedParent);
    processor.onEnd(childOfRecentParent);

    assert.equal(childOfEvictedParent.parentSpanId, "dropped-parent-0");
    assert.equal(childOfRecentParent.parentSpanId, undefined);
    assert.deepEqual(capturedSpans, [childOfEvictedParent, childOfRecentParent]);
  } finally {
    instrumentor.deactivate();
    resetTracerProvider();
  }
});


test("BeeAIInstrumentor exports only complete BeeAI event rows", async () => {
  class FakeBeeAIInstrumentation {}

  const capturedSpans = [];
  const startedSpans = [];
  const processor = {
    onStart(span) {
      startedSpans.push(span);
    },
    onEnd(span) {
      if (Array.isArray(span.attributes["respan.processors"]) && span.attributes["respan.processors"].length === 0) {
        return;
      }
      capturedSpans.push(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));

  const calls = [];
  const delegate = {
    activate() {
      calls.push("activate");
    },
    deactivate() {
      calls.push("deactivate");
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: { BeeAgent: class BeeAgent {} },
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      return delegate;
    },
  });

  try {
    await instrumentor.activate();

    const expression = "(19 + 23) * 2";
    const userMessage = {
      role: "user",
      content: [{ type: "text", text: `Compute ${expression}` }],
    };
    const toolCallMessage = {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "Calculator",
          args: { expression },
        },
      ],
    };
    const toolResultMessage = {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call-1",
          toolName: "Calculator",
          result: "84",
          isError: false,
        },
      ],
    };
    const agentStartState = {
      memory: { messages: [userMessage] },
      iteration: 1,
    };
    const agentSuccessState = {
      memory: { messages: [userMessage, toolCallMessage, toolResultMessage] },
      iteration: 1,
    };
    const finalToolCallMessage = {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          toolCallId: "call-final",
          toolName: "final_answer",
          args: { response: "84" },
        },
      ],
    };
    const finalToolResultMessage = {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call-final",
          toolName: "final_answer",
          result: "Message has been sent",
          isError: false,
        },
      ],
    };
    const finalState = {
      memory: {
        messages: [
          userMessage,
          toolCallMessage,
          toolResultMessage,
          finalToolCallMessage,
          finalToolResultMessage,
        ],
      },
      result: {
        role: "assistant",
        content: [{ type: "text", text: "84" }],
      },
      iteration: 2,
    };

    const normalizedToolCall = {
      id: "call-1",
      type: "function",
      function: { name: "Calculator", arguments: JSON.stringify({ expression }) },
    };
    const normalizedFinalToolCall = {
      id: "call-final",
      type: "function",
      function: { name: "final_answer", arguments: JSON.stringify({ response: "84" }) },
    };
    const normalizedUserMessages = [
      { role: "user", content: `Compute ${expression}` },
    ];
    const normalizedFollowupMessages = [
      { role: "user", content: `Compute ${expression}` },
      { role: "assistant", content: "", tool_calls: [normalizedToolCall] },
      {
        role: "tool",
        tool_call_id: "call-1",
        name: "Calculator",
        content: "84",
        is_error: false,
      },
    ];
    const normalizedAgentInput = {
      iteration: 1,
      messages: [
        { role: "user", content: `Compute ${expression}` },
        { role: "assistant", content: "", tool_calls: [normalizedToolCall] },
      ],
    };
    const basicUserMessage = {
      role: "user",
      content: [{ type: "text", text: "Explain tracing in one sentence." }],
    };
    const basicAssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: "Tracing shows each step and value in a run." }],
    };

    const basicStartSpan = makeSpan({
      name: "backend.dummy.chat.start-basic",
      instrumentationScopeName: null,
      instrumentationLibraryName: BEEAI_SCOPE_NAME,
      traceId: "otel-trace-basic",
      spanId: "basic-start-span",
      attributes: {},
    });
    const basicChatSpan = makeSpan({
      name: "backend.dummy.chat.success-basic",
      instrumentationScopeName: null,
      instrumentationLibraryName: BEEAI_SCOPE_NAME,
      traceId: "otel-trace-basic",
      spanId: "basic-chat-span",
      attributes: {
        target: "backend.dummy.chat.success",
        traceId: "trace-basic",
        "output.value": JSON.stringify([basicAssistantMessage]),
        "llm.model_name": "gpt-4o-mini",
        "metadata.model_name": "gpt-4o-mini",
        "llm.token_count.prompt": 7,
        "llm.token_count.completion": 9,
        "llm.token_count.total": 16,
        data: JSON.stringify({
          value: {
            messages: [basicAssistantMessage],
          },
        }),
      },
    });

    const agentStartSpan = makeSpan({
      name: "agent.toolCalling.start-1",
      traceId: "otel-trace-1",
      spanId: "start-span-1",
      parentSpanId: "framework-span",
      attributes: {
        target: "agent.toolCalling.start",
        traceId: "trace-1",
        data: JSON.stringify({}),
      },
    });
    const chatSpan = makeSpan({
      name: "backend.openai.chat.success-1",
      traceId: "otel-trace-1",
      spanId: "chat-span-1",
      parentSpanId: "start-span-1",
      attributes: {
        target: "backend.openai.chat.success",
        traceId: "trace-1",
        "input.value": JSON.stringify([{}]),
        "output.value": JSON.stringify([toolCallMessage]),
        "llm.input_messages.0.message.content": `Compute ${expression}`,
        "llm.output_messages.0.message.content": "",
        data: JSON.stringify({
          value: {
            model: "gpt-4o-mini",
            messages: [toolCallMessage],
            usage: { promptTokens: 18, completionTokens: 2, totalTokens: 20 },
          },
        }),
      },
    });
    const chat2Span = makeSpan({
      name: "backend.openai.chat.success-2",
      traceId: "otel-trace-1",
      spanId: "chat-span-2",
      parentSpanId: "agent-span-1",
      attributes: {
        target: "backend.openai.chat.success",
        traceId: "trace-1",
        "input.value": JSON.stringify([userMessage, toolCallMessage, toolResultMessage]),
        "output.value": JSON.stringify([finalToolCallMessage]),
        data: JSON.stringify({
          value: {
            model: "gpt-4o-mini",
            messages: [finalToolCallMessage],
            usage: { promptTokens: 22, completionTokens: 1, totalTokens: 23 },
          },
        }),
      },
    });
    const toolSpan = makeSpan({
      name: "tool.calculator.success-1",
      traceId: "otel-trace-1",
      spanId: "tool-span-1",
      parentSpanId: "chat-span-1",
      attributes: {
        target: "tool.calculator.success",
        traceId: "trace-1",
        data: JSON.stringify({
          input: { expression },
          output: { result: 84 },
        }),
      },
    });
    const agentSpan = makeSpan({
      name: "agent.toolCalling.success-1",
      traceId: "otel-trace-1",
      spanId: "agent-span-1",
      parentSpanId: "chat-span-1",
      attributes: {
        target: "agent.toolCalling.success",
        traceId: "trace-1",
        data: JSON.stringify({ state: agentSuccessState }),
      },
    });
    const emptyAgentSpan = makeSpan({
      name: "agent.toolCalling.success-empty",
      traceId: "otel-empty-agent-trace",
      spanId: "empty-agent-span",
      attributes: {
        target: "agent.toolCalling.success",
        traceId: "empty-agent-trace",
        data: JSON.stringify({}),
      },
    });
    const finishSpan = makeSpan({
      name: "tool.calculator.finish-1",
      traceId: "otel-trace-1",
      spanId: "finish-span-1",
      parentSpanId: "tool-span-1",
      attributes: {
        target: "tool.calculator.finish",
        traceId: "trace-1",
        metadata: JSON.stringify({
          state: agentSuccessState,
          toolCallMsg: {
            type: "tool-call",
            toolCallId: "call-1",
            toolName: "Calculator",
            args: { expression },
          },
        }),
      },
    });
    const finalAnswerSpan = makeSpan({
      name: "tool.dynamic.finalAnswer.success-1",
      traceId: "otel-trace-1",
      spanId: "final-answer-span-1",
      parentSpanId: "chat-span-2",
      attributes: {
        target: "tool.dynamic.finalAnswer.success",
        traceId: "trace-1",
        metadata: JSON.stringify({
          state: finalState,
          toolCallMsg: {
            type: "tool-call",
            toolCallId: "call-final",
            toolName: "final_answer",
            args: { response: "84" },
          },
        }),
        data: JSON.stringify({
          input: { response: "84" },
          output: { result: "Message has been sent" },
        }),
      },
    });
    const parentSpan = makeSpan({
      name: "beeai-framework-main",
      traceId: "otel-trace-1",
      spanId: "framework-span",
      parentSpanId: "workflow-span",
      attributes: {
        traceId: "trace-1",
        source: "ToolCallingAgent",
        "beeai.version": "0.1.13",
      },
    });
    const workflowSpan = makeSpan({
      name: "beeai_tool_calling_agent.workflow.workflow",
      instrumentationScopeName: "@respan/tracing",
      traceId: "otel-trace-1",
      spanId: "workflow-span",
      attributes: {
        "traceloop.span.kind": "workflow",
      },
    });

    processor.onStart(basicStartSpan, {});
    Object.assign(basicStartSpan.attributes, {
      target: "backend.dummy.chat.start",
      traceId: "trace-basic",
      "input.value": JSON.stringify([basicUserMessage]),
      "llm.model_name": "gpt-4o-mini",
    });
    processor.onEnd(basicStartSpan);
    processor.onEnd(basicChatSpan);
    processor.onStart(workflowSpan, {});
    processor.onStart(parentSpan, {});
    processor.onStart(agentStartSpan, {});
    processor.onEnd(agentStartSpan);
    processor.onEnd(chatSpan);
    processor.onEnd(toolSpan);
    processor.onEnd(agentSpan);
    processor.onEnd(emptyAgentSpan);
    processor.onEnd(chat2Span);
    processor.onEnd(finishSpan);
    processor.onEnd(finalAnswerSpan);
    processor.onEnd(parentSpan);

    assert.deepEqual(calls, ["activate"]);
    assert.equal(startedSpans.length, 4);
    assert.equal(
      capturedSpans.length,
      6,
      `captured spans: ${capturedSpans.map((span) => span.name).join(", ")}`,
    );
    assert.deepEqual(capturedSpans, [
      basicChatSpan,
      toolSpan,
      chatSpan,
      agentSpan,
      chat2Span,
      finalAnswerSpan,
    ]);
    assert.deepEqual(agentStartSpan.attributes["respan.processors"], []);
    assert.deepEqual(emptyAgentSpan.attributes["respan.processors"], []);
    assert.equal(emptyAgentSpan.attributes.target, undefined);
    assert.deepEqual(finishSpan.attributes["respan.processors"], []);
    assert.deepEqual(parentSpan.attributes["respan.processors"], []);

    assert.equal(basicChatSpan.attributes["respan.entity.log_type"], "chat");
    assert.equal(basicChatSpan.attributes["llm.request.type"], "chat");
    assert.equal(basicChatSpan.attributes["gen_ai.system"], "dummy");
    assert.equal(basicChatSpan.attributes["traceloop.entity.name"], "backend.dummy.chat.success");
    assert.equal(basicChatSpan.attributes["traceloop.entity.path"], "backend.dummy.chat.success-basic");
    assert.equal(basicChatSpan.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(basicChatSpan.attributes["gen_ai.usage.input_tokens"], 7);
    assert.equal(basicChatSpan.attributes["gen_ai.usage.prompt_tokens"], 7);
    assert.equal(basicChatSpan.attributes["gen_ai.usage.output_tokens"], 9);
    assert.equal(basicChatSpan.attributes["gen_ai.usage.completion_tokens"], 9);
    assert.equal(basicChatSpan.attributes["llm.usage.total_tokens"], 16);
    assert.equal(
      basicChatSpan.attributes["traceloop.entity.input"],
      JSON.stringify([{ role: "user", content: "Explain tracing in one sentence." }]),
    );
    assert.equal(
      basicChatSpan.attributes["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: "Tracing shows each step and value in a run." }),
    );
    assert.equal(basicChatSpan.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(
      basicChatSpan.attributes["gen_ai.completion.0.content"],
      "Tracing shows each step and value in a run.",
    );

    assert.equal(chatSpan.parentSpanId, "workflow-span");
    assert.equal(chatSpan.attributes["respan.entity.log_type"], "chat");
    assert.equal(chatSpan.attributes["llm.request.type"], "chat");
    assert.equal(chatSpan.attributes["gen_ai.system"], "openai");
    assert.equal(chatSpan.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(chatSpan.attributes["gen_ai.usage.input_tokens"], 18);
    assert.equal(chatSpan.attributes["gen_ai.usage.output_tokens"], 2);
    assert.equal(chatSpan.attributes["llm.usage.total_tokens"], 20);
    assert.equal(chatSpan.attributes["traceloop.entity.input"], JSON.stringify(normalizedUserMessages));
    assert.equal(
      chatSpan.attributes["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: "", tool_calls: [normalizedToolCall] }),
    );
    assert.equal(chatSpan.attributes["gen_ai.prompt.0.role"], "user");
    assert.equal(chatSpan.attributes["gen_ai.prompt.0.content"], `Compute ${expression}`);
    assert.equal(chatSpan.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(chatSpan.attributes["gen_ai.completion.0.content"], "");
    assert.equal(
      chatSpan.attributes["gen_ai.completion.0.tool_calls"],
      JSON.stringify([normalizedToolCall]),
    );

    assert.equal(toolSpan.parentSpanId, "workflow-span");
    assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
    assert.equal(toolSpan.attributes["traceloop.entity.input"], JSON.stringify({ expression }));
    assert.equal(toolSpan.attributes["traceloop.entity.output"], JSON.stringify({ result: 84 }));

    assert.equal(agentSpan.parentSpanId, "workflow-span");
    assert.equal(agentSpan.attributes["respan.entity.log_type"], "agent");
    assert.equal(agentSpan.attributes["traceloop.entity.input"], JSON.stringify(normalizedAgentInput));
    assert.equal(
      agentSpan.attributes["traceloop.entity.output"],
      JSON.stringify({ result: "84", is_error: false }),
    );

    assert.equal(chat2Span.parentSpanId, "workflow-span");
    assert.equal(chat2Span.attributes["respan.entity.log_type"], "chat");
    assert.equal(chat2Span.attributes["llm.request.type"], "chat");
    assert.equal(chat2Span.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(chat2Span.attributes["gen_ai.usage.input_tokens"], 22);
    assert.equal(chat2Span.attributes["gen_ai.usage.output_tokens"], 1);
    assert.equal(chat2Span.attributes["llm.usage.total_tokens"], 23);
    assert.equal(chat2Span.attributes["traceloop.entity.input"], JSON.stringify(normalizedFollowupMessages));
    assert.equal(
      chat2Span.attributes["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: "", tool_calls: [normalizedFinalToolCall] }),
    );
    assert.equal(chat2Span.attributes["gen_ai.prompt.0.role"], "user");
    assert.equal(chat2Span.attributes["gen_ai.prompt.0.content"], `Compute ${expression}`);
    assert.equal(chat2Span.attributes["gen_ai.prompt.1.role"], "assistant");
    assert.equal(chat2Span.attributes["gen_ai.prompt.1.content"], "");
    assert.equal(
      chat2Span.attributes["gen_ai.prompt.1.tool_calls"],
      JSON.stringify([normalizedToolCall]),
    );
    assert.equal(chat2Span.attributes["gen_ai.prompt.2.role"], "tool");
    assert.equal(chat2Span.attributes["gen_ai.prompt.2.content"], "84");
    assert.equal(chat2Span.attributes["gen_ai.prompt.2.tool_call_id"], "call-1");
    assert.equal(chat2Span.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(chat2Span.attributes["gen_ai.completion.0.content"], "");
    assert.equal(
      chat2Span.attributes["gen_ai.completion.0.tool_calls"],
      JSON.stringify([normalizedFinalToolCall]),
    );

    assert.equal(finalAnswerSpan.parentSpanId, "workflow-span");
    assert.equal(finalAnswerSpan.attributes["respan.entity.log_type"], "tool");
    assert.equal(finalAnswerSpan.attributes["traceloop.entity.input"], JSON.stringify({ response: "84" }));
    assert.equal(finalAnswerSpan.attributes["traceloop.entity.output"], "84");

    for (const span of capturedSpans) {
      for (const rawAttribute of [
        "target",
        "data",
        "metadata",
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
        assert.equal(
          span.attributes[rawAttribute],
          undefined,
          `${span.name} exported raw ${rawAttribute}`,
        );
      }
      assert.equal(span.attributes["llm.input_messages.0.message.content"], undefined);
      assert.equal(span.attributes["llm.output_messages.0.message.content"], undefined);
      for (const alias of [
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "tools",
        "tool_calls",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
        "respan.span.tools",
        "respan.span.tool_calls",
        "respan.span.handoffs",
      ]) {
        assert.equal(span.attributes[alias], undefined, `${span.name} emitted ${alias}`);
      }
    }

    instrumentor.deactivate();
    assert.deepEqual(calls, ["activate", "deactivate"]);

    const rawSpan = makeSpan({
      name: "backend.openai.chat.success-2",
      attributes: { target: "backend.openai.chat.success" },
    });
    processor.onEnd(rawSpan);
    assert.equal(rawSpan.attributes["respan.entity.log_type"], undefined);
  } finally {
    instrumentor.deactivate();
    resetTracerProvider();
  }
});
