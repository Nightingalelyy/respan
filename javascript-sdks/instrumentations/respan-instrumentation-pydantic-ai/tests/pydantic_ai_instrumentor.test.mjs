import assert from "node:assert/strict";
import test from "node:test";

import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { RespanCompositeProcessor } from "../../../respan-tracing/dist/processor/composite.js";

import {
  PydanticAIInstrumentor,
  enrichPydanticAISpan,
  isPydanticAISpan,
} from "../dist/index.js";

const RESPAN_LOG_TYPE = "respan.entity.log_type";
const RESPAN_LOG_METHOD = "respan.entity.log_method";
const ENTITY_NAME = "traceloop.entity.name";
const ENTITY_PATH = "traceloop.entity.path";
const ENTITY_INPUT = "traceloop.entity.input";
const ENTITY_OUTPUT = "traceloop.entity.output";
const WORKFLOW_NAME = "traceloop.workflow.name";
const LLM_REQUEST_TYPE = "llm.request.type";
const LLM_REQUEST_FUNCTIONS = "llm.request.functions";
const LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens";

function makeSpan({ name, attributes = {}, scopeName = "pydantic-ai" }) {
  return {
    name,
    attributes: { ...attributes },
    instrumentationScope: { name: scopeName },
    spanContext() {
      return {
        traceId: "11111111111111111111111111111111",
        spanId: "2222222222222222",
        traceFlags: 1,
      };
    },
  };
}

function assertNoOffContractAliases(attrs) {
  for (const key of [
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "tools",
    "tool_calls",
    "span_tools",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "has_tool_calls",
    "parallel_tool_calls",
  ]) {
    assert.equal(attrs[key], undefined, `${key} should not be emitted`);
  }
}

test("enriches native Pydantic AI chat spans with canonical fields", () => {
  const span = makeSpan({
    name: "chat completion",
    attributes: {
      "gen_ai.system": "OpenAI",
      "gen_ai.operation.name": "chat",
      "gen_ai.request.model": "gpt-4o-mini",
      "gen_ai.input.messages": JSON.stringify([
        { role: "system", content: "Answer briefly." },
        { role: "user", content: "What is Respan?" },
      ]),
      "gen_ai.output.messages": JSON.stringify([
        { role: "assistant", content: "An AI observability platform." },
      ]),
      "gen_ai.tool.definitions": JSON.stringify([
        {
          name: "lookup_doc",
          description: "Look up a document",
          parameters: { type: "object" },
        },
      ]),
      "gen_ai.usage.input_tokens": 21,
      "gen_ai.usage.output_tokens": 9,
      "gen_ai.usage.total_tokens": 30,
      "model": "raw-alias-should-strip",
    },
  });

  enrichPydanticAISpan(span);
  const attrs = span.attributes;

  assert.equal(attrs[RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs[RESPAN_LOG_METHOD], "ts_tracing");
  assert.equal(attrs[ENTITY_NAME], "chat completion");
  assert.equal(attrs[ENTITY_PATH], "chat completion");
  assert.equal(attrs[LLM_REQUEST_TYPE], "chat");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Answer briefly.");
  assert.equal(attrs["gen_ai.prompt.1.role"], "user");
  assert.equal(attrs["gen_ai.prompt.1.content"], "What is Respan?");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "An AI observability platform.");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 21);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 21);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 9);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 9);
  assert.equal(attrs[LLM_USAGE_TOTAL_TOKENS], 30);
  assert.deepEqual(JSON.parse(attrs[LLM_REQUEST_FUNCTIONS]), [
    {
      type: "function",
      function: {
        name: "lookup_doc",
        description: "Look up a document",
        parameters: { type: "object" },
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs[ENTITY_INPUT]), [
    { role: "system", content: "Answer briefly." },
    { role: "user", content: "What is Respan?" },
  ]);
  assert.equal(attrs["gen_ai.input.messages"], undefined);
  assert.equal(attrs["gen_ai.output.messages"], undefined);
  assert.equal(attrs["gen_ai.tool.definitions"], undefined);
  assert.equal(attrs["traceloop.span.kind"], undefined);
  assertNoOffContractAliases(attrs);
});

test("enriches native Pydantic AI tool spans without LLM aliases", () => {
  const span = makeSpan({
    name: "execute tool add",
    attributes: {
      "gen_ai.tool.name": "add",
      "gen_ai.tool.call.arguments": JSON.stringify({ a: 1, b: 2 }),
      "gen_ai.tool.call.result": "3",
      "gen_ai.request.model": "gpt-4o-mini",
      "gen_ai.system": "openai",
      "span_tools": ["add"],
    },
  });

  enrichPydanticAISpan(span);
  const attrs = span.attributes;

  assert.equal(attrs[RESPAN_LOG_TYPE], "tool");
  assert.equal(attrs[ENTITY_NAME], "add");
  assert.equal(attrs[ENTITY_PATH], "add");
  assert.deepEqual(JSON.parse(attrs[ENTITY_INPUT]), {
    name: "add",
    arguments: { a: 1, b: 2 },
  });
  assert.equal(attrs[ENTITY_OUTPUT], "3");
  assert.equal(attrs["gen_ai.tool.name"], undefined);
  assert.equal(attrs["gen_ai.request.model"], undefined);
  assert.equal(attrs["gen_ai.system"], undefined);
  assertNoOffContractAliases(attrs);
});

test("enriches Pydantic AI-scoped OpenInference LLM spans", () => {
  const span = makeSpan({
    name: "openinference llm",
    scopeName: "@arizeai/openinference-instrumentation-pydantic-ai",
    attributes: {
      "openinference.span.kind": "LLM",
      "input.value": "What is the capital of France?",
      "output.value": "Paris",
      "llm.model_name": "gpt-4o-mini",
      "llm.provider": "OpenAI",
      "llm.token_count.prompt": 12,
      "llm.token_count.completion": 4,
      "llm.token_count.total": 16,
      "llm.input_messages.0.message.role": "user",
      "llm.input_messages.0.message.content": "What is the capital of France?",
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.content": "Paris",
      "tools": ["raw"],
    },
  });

  enrichPydanticAISpan(span);
  const attrs = span.attributes;

  assert.equal(attrs[RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs[LLM_REQUEST_TYPE], "chat");
  assert.equal(attrs[ENTITY_INPUT], "What is the capital of France?");
  assert.equal(attrs[ENTITY_OUTPUT], "Paris");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.provider.name"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "What is the capital of France?");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Paris");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 4);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 4);
  assert.equal(attrs[LLM_USAGE_TOTAL_TOKENS], 16);
  assert.equal(attrs["openinference.span.kind"], undefined);
  assert.equal(attrs["llm.input_messages.0.message.role"], undefined);
  assert.equal(attrs["llm.output_messages.0.message.content"], undefined);
  assertNoOffContractAliases(attrs);
});

test("does not translate generic GenAI spans without Pydantic AI evidence", () => {
  const span = makeSpan({
    name: "generic chat",
    scopeName: "@example/generic-genai",
    attributes: {
      "gen_ai.system": "openai",
      "gen_ai.operation.name": "chat",
      "gen_ai.request.model": "gpt-4o-mini",
    },
  });

  assert.equal(isPydanticAISpan(span), false);
  enrichPydanticAISpan(span);
  assert.equal(span.attributes[RESPAN_LOG_TYPE], undefined);
});

test("OTel 2.10 cached tracers translate complete content and drain in-flight spans", async () => {
  const delegatedSpans = [];
  const manager = {
    onStart() {},
    onEnd(span) {
      delegatedSpans.push(span);
    },
    async shutdown() {},
    async forceFlush() {},
  };
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  const tracerBeforeActivation = provider.getTracer("pydantic-ai");
  const activeProcessorBefore = provider._activeSpanProcessor;
  const instrumentor = new PydanticAIInstrumentor();
  try {
    instrumentor.activate();
    assert.equal(provider._activeSpanProcessor, activeProcessorBefore);

    tracerBeforeActivation.startSpan("chat completion", {
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", content: "hello" },
        ]),
        "gen_ai.output.messages": JSON.stringify([
          { role: "assistant", content: "hi" },
        ]),
      },
    }).end();

    assert.equal(delegatedSpans.length, 1);
    assert.equal(delegatedSpans[0].attributes[RESPAN_LOG_TYPE], "chat");
    assert.deepEqual(JSON.parse(delegatedSpans[0].attributes[ENTITY_INPUT]), [
      { role: "user", content: "hello" },
    ]);
    assert.deepEqual(JSON.parse(delegatedSpans[0].attributes[ENTITY_OUTPUT]), [
      { role: "assistant", content: "hi" },
    ]);
    assert.equal(
      delegatedSpans[0].attributes["gen_ai.input.messages"],
      undefined,
    );
    assert.equal(
      delegatedSpans[0].attributes["gen_ai.output.messages"],
      undefined,
    );
    assertNoOffContractAliases(delegatedSpans[0].attributes);

    const tracerAfterActivation = provider.getTracer("pydantic-ai.after");
    const draining = tracerAfterActivation.startSpan("chat completion", {
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", content: "drain me" },
        ]),
        "gen_ai.output.messages": JSON.stringify([
          { role: "assistant", content: "drained" },
        ]),
      },
    });
    instrumentor.deactivate();
    draining.end();
    assert.equal(delegatedSpans[1].attributes[RESPAN_LOG_TYPE], "chat");
    assert.match(delegatedSpans[1].attributes[ENTITY_OUTPUT], /drained/);

    tracerBeforeActivation.startSpan("chat completion", {
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "raw-after-deactivation",
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", content: "raw" },
        ]),
      },
    }).end();
    assert.equal(delegatedSpans[2].attributes[RESPAN_LOG_TYPE], undefined);
    assert.equal(
      delegatedSpans[2].attributes["gen_ai.input.messages"],
      JSON.stringify([{ role: "user", content: "raw" }]),
    );
  } finally {
    instrumentor.deactivate();
    await provider.shutdown();
  }

  const noHost = new PydanticAIInstrumentor();
  assert.throws(
    () => noHost.activate(),
    /No compatible Respan span-transformer host is active/,
  );
  assert.equal(noHost.isActive(), false);
});
