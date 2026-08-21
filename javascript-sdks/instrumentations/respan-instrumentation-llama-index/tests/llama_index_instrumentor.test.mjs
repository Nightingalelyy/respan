import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { CallbackManager, Settings } from "llamaindex";

import { LlamaIndexInstrumentor } from "../dist/index.js";

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);
let originalCallbackManager;

test.before(() => {
  originalCallbackManager = Settings.callbackManager;
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
  Settings.callbackManager = originalCallbackManager;
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

test.beforeEach(() => {
  captureState.spans = [];
  Settings.callbackManager = new CallbackManager();
});

test("emits canonical chat span from LlamaIndex LLM callbacks", async () => {
  const instrumentor = new LlamaIndexInstrumentor({
    workflowName: "llama_index_ts_unit_chat",
  });
  await instrumentor.activate();

  Settings.callbackManager.dispatchEvent(
    "llm-start",
    {
      id: "llm-1",
      messages: [{ role: "user", content: "Say hello" }],
    },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "llm-end",
    {
      id: "llm-1",
      response: {
        raw: {
          model: "gpt-4.1-nano",
          usage: {
            prompt_tokens: 7,
            completion_tokens: 3,
            total_tokens: 10,
          },
        },
        message: {
          role: "assistant",
          content: "Hello",
        },
      },
    },
    true,
  );

  assert.equal(captureState.spans.length, 1);
  const [span] = captureState.spans;
  const attrs = span.attributes;
  assert.equal(span.instrumentationScope?.name, "@respan/instrumentation-llama-index");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["traceloop.entity.name"], "llamaindex.llm");
  assert.equal(attrs["traceloop.workflow.name"], "llama_index_ts_unit_chat");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4.1-nano");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Say hello");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Hello");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 3);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 3);
  assert.equal(attrs["llm.usage.total_tokens"], 10);
  assert.equal(attrs.model, undefined);
  assert.equal(attrs.prompt_tokens, undefined);
  assert.equal(attrs.completion_tokens, undefined);
  assert.equal(attrs.total_request_tokens, undefined);
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs.tool_calls, undefined);

  instrumentor.deactivate();
});

test("emits query, retrieve, and tool spans without off-contract aliases", async () => {
  const instrumentor = new LlamaIndexInstrumentor({
    workflowName: "llama_index_ts_unit_rag",
  });
  await instrumentor.activate();

  Settings.callbackManager.dispatchEvent(
    "query-start",
    { id: "query-1", query: "What is Respan?" },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "retrieve-start",
    { id: "retrieve-1", query: { query: "What is Respan?" } },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "retrieve-end",
    {
      id: "retrieve-1",
      query: { query: "What is Respan?" },
      nodes: [{ node: { text: "Respan traces AI apps." }, score: 0.9 }],
    },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "llm-tool-call",
    {
      toolCall: {
        id: "tool-1",
        name: "lookup_docs",
        input: { topic: "Respan" },
      },
    },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "llm-tool-result",
    {
      toolCall: {
        id: "tool-1",
        name: "lookup_docs",
        input: { topic: "Respan" },
      },
      toolResult: {
        output: { answer: "observability" },
        isError: false,
      },
    },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "query-end",
    { id: "query-1", response: { text: "Respan traces AI apps." } },
    true,
  );

  assert.equal(captureState.spans.length, 3);
  const retrieveSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "task",
  );
  const toolSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  const workflowSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "workflow",
  );

  assert.ok(retrieveSpan);
  assert.ok(toolSpan);
  assert.ok(workflowSpan);
  assert.equal(retrieveSpan.attributes["traceloop.entity.name"], "llamaindex.retrieve");
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "lookup_docs");
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.input"]), {
    name: "lookup_docs",
    arguments: "{\"topic\":\"Respan\"}",
  });
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.output"]), {
    answer: "observability",
  });
  assert.equal(toolSpan.attributes.tool_calls, undefined);
  assert.equal(toolSpan.attributes["respan.span.tool_calls"], undefined);
  assert.equal(workflowSpan.attributes["traceloop.workflow.name"], "llama_index_ts_unit_rag");

  instrumentor.deactivate();
});

test("deactivate removes handlers", async () => {
  const instrumentor = new LlamaIndexInstrumentor();
  await instrumentor.activate();
  instrumentor.deactivate();

  Settings.callbackManager.dispatchEvent(
    "query-start",
    { id: "query-2", query: "ignored" },
    true,
  );
  Settings.callbackManager.dispatchEvent(
    "query-end",
    { id: "query-2", response: "ignored" },
    true,
  );

  assert.equal(captureState.spans.length, 0);
});

test("rejected LLM calls emit failed chat and ancestor spans with sanitized errors", async () => {
  const instrumentor = new LlamaIndexInstrumentor({
    workflowName: "llama_index_ts_unit_failure",
  });
  await instrumentor.activate();

  const failingLlm = {
    model: "gpt-4.1-nano",
    async chat() {
      Settings.callbackManager.dispatchEvent(
        "query-start",
        { id: "query-failure", query: "Fail safely" },
        true,
      );
      Settings.callbackManager.dispatchEvent(
        "llm-start",
        {
          id: "llm-failure",
          messages: [{ role: "user", content: "Fail safely" }],
        },
        true,
      );
      throw new Error("401 invalid api_key=sk-super-secret-token");
    },
  };
  Settings.llm = failingLlm;

  await assert.rejects(() => Settings.llm.chat({ messages: [] }), /401 invalid/);

  assert.equal(captureState.spans.length, 2);
  const chatSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "chat",
  );
  const workflowSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "workflow",
  );
  assert.ok(chatSpan);
  assert.ok(workflowSpan);
  assert.equal(chatSpan.status.code, 2);
  assert.equal(workflowSpan.status.code, 2);
  assert.equal(chatSpan.attributes.status_code, 500);
  assert.equal(workflowSpan.attributes.status_code, 500);
  assert.equal(chatSpan.attributes["gen_ai.request.model"], "gpt-4.1-nano");
  assert.equal(chatSpan.attributes["gen_ai.prompt.0.content"], "Fail safely");
  assert.equal(chatSpan.attributes["error.message"], "401 invalid api_key=[REDACTED]");
  assert.equal(workflowSpan.attributes["error.message"], "401 invalid api_key=[REDACTED]");
  assert.equal(chatSpan.parentSpanContext?.spanId, workflowSpan.spanContext().spanId);

  instrumentor.deactivate();
  assert.equal(Object.hasOwn(failingLlm, "chat"), true);
  assert.equal(failingLlm.chat.name, "chat");
});

test("deactivate exports unfinished callbacks as failed spans instead of dropping them", async () => {
  const instrumentor = new LlamaIndexInstrumentor({
    workflowName: "llama_index_ts_unit_pending",
  });
  await instrumentor.activate();

  Settings.callbackManager.dispatchEvent(
    "llm-start",
    {
      id: "llm-pending",
      messages: [{ role: "user", content: "Pending request" }],
    },
    true,
  );
  instrumentor.deactivate();

  assert.equal(captureState.spans.length, 1);
  assert.equal(captureState.spans[0].status.code, 2);
  assert.equal(captureState.spans[0].attributes.status_code, 500);
  assert.match(captureState.spans[0].attributes["error.message"], /without a matching/);
});

test("correlates id-less start and end events by LlamaIndex event reason", async () => {
  const handlers = new Map();
  const callbackManager = {
    on(event, handler) {
      handlers.set(event, handler);
    },
    off(event) {
      handlers.delete(event);
    },
  };
  const instrumentor = new LlamaIndexInstrumentor({
    workflowName: "llama_index_ts_unit_idless",
    llamaIndexModule: { Settings: { callbackManager } },
  });
  await instrumentor.activate();

  const reason = { id: "stable-event-reason" };
  handlers.get("chunking-start")({
    detail: { chunks: ["input text"] },
    reason,
  });
  handlers.get("chunking-end")({
    detail: { chunks: ["output chunk"] },
    reason,
  });

  handlers.get("node-parsing-start")({
    detail: { documents: [{ text: "input document" }] },
    reason: null,
  });
  handlers.get("node-parsing-end")({
    detail: { nodes: [{ text: "output node" }] },
    reason: null,
  });

  assert.equal(captureState.spans.length, 2);
  for (const span of captureState.spans) {
    assert.equal(span.status.code, 1);
    assert.equal(span.attributes["error.message"], undefined);
  }
  instrumentor.deactivate();
  assert.equal(captureState.spans.length, 2);
});
