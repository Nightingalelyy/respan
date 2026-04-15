import assert from "node:assert/strict";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan } from "../src/utils/spanFactory.js";
import {
  isDirectRespanSpan,
  spanToRespanPayload,
} from "../src/exporters/respanHybridExporter.js";

const toolDefinitions = [
  {
    type: "function",
    function: {
      name: "mcp__demo__get_weather",
      description: "Get weather",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string" },
        },
      },
    },
  },
];

const toolCalls = [
  {
    id: "toolu_123",
    type: "function",
    function: {
      name: "mcp__demo__get_weather",
      arguments: '{"city":"Tokyo"}',
    },
  },
];

function buildTestSpan(attributes = {}) {
  return buildReadableSpan({
    name: "claude-agent-sdk-complex-edge-cases.agent",
    traceId: "1234567890abcdef1234567890abcdef",
    spanId: "1234567890abcdef",
    parentId: "abcdef1234567890",
    mergePropagated: false,
    attributes: {
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.AGENT,
      [RespanSpanAttributes.GEN_AI_SYSTEM]: "anthropic",
      [RespanSpanAttributes.LLM_SYSTEM]: "anthropic",
      [RespanSpanAttributes.LLM_REQUEST_TYPE]: "chat",
      [RespanSpanAttributes.GEN_AI_REQUEST_MODEL]: "claude-sonnet-4-6",
      [RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME]:
        "claude-agent-sdk-complex-edge-cases",
      [SpanAttributes.TRACELOOP_ENTITY_NAME]:
        "claude-agent-sdk-complex-edge-cases",
      [SpanAttributes.TRACELOOP_WORKFLOW_NAME]:
        "claude-agent-sdk-complex-edge-cases",
      [SpanAttributes.TRACELOOP_ENTITY_INPUT]: JSON.stringify([
        {
          role: "system",
          content: "Use tools when needed.",
        },
        {
          role: "user",
          content: "Use get_weather for Tokyo and summarize briefly.",
        },
      ]),
      [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: "Tokyo is sunny.",
      [RespanSpanAttributes.RESPAN_SPAN_TOOLS]: JSON.stringify(toolDefinitions),
      [RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS]: JSON.stringify(toolCalls),
      prompt_tokens: 42,
      completion_tokens: 9,
      total_request_tokens: 51,
      prompt_cache_hit_tokens: 1,
      prompt_cache_creation_tokens: 0,
      cost: 0.123,
      parallel_tool_calls: false,
      ...attributes,
    },
  });
}

const span = buildTestSpan();

assert.equal(isDirectRespanSpan(span), true);
assert.equal(
  isDirectRespanSpan(
    buildTestSpan({
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: "python_tracing",
    }),
  ),
  false,
);

const payload = spanToRespanPayload(span);

assert.equal(payload.log_type, RespanLogType.CHAT);
assert.deepEqual(payload.tools, toolDefinitions);
assert.deepEqual(payload.tool_calls, toolCalls);
assert.deepEqual(payload.prompt_messages, [
  {
    role: "system",
    content: "Use tools when needed.",
  },
  {
    role: "user",
    content: "Use get_weather for Tokyo and summarize briefly.",
  },
]);
assert.deepEqual(payload.completion_message, {
  role: "assistant",
  content: "Tokyo is sunny.",
  tool_calls: toolCalls,
});
assert.deepEqual(payload.output, {
  role: "assistant",
  content: "Tokyo is sunny.",
  tool_calls: toolCalls,
});
assert.deepEqual(payload.metadata, {
  agent_name: "claude-agent-sdk-complex-edge-cases",
  "traceloop.entity.name": "claude-agent-sdk-complex-edge-cases",
  "gen_ai.system": "anthropic",
  "llm.system": "anthropic",
});

const structuredOutputPayload = spanToRespanPayload(
  buildTestSpan({
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: JSON.stringify({
      role: "assistant",
      content: "Already structured.",
      tool_calls: toolCalls,
    }),
  }),
);

assert.deepEqual(structuredOutputPayload.completion_message, {
  role: "assistant",
  content: "Already structured.",
  tool_calls: toolCalls,
});
assert.deepEqual(structuredOutputPayload.output, {
  role: "assistant",
  content: "Already structured.",
  tool_calls: toolCalls,
});

const toolOnlyPayload = spanToRespanPayload(
  buildTestSpan({
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: "",
  }),
);

assert.deepEqual(toolOnlyPayload.completion_message, {
  role: "assistant",
  content: "",
  tool_calls: toolCalls,
});
assert.deepEqual(toolOnlyPayload.output, {
  role: "assistant",
  content: "",
  tool_calls: toolCalls,
});

const nonChatPayload = spanToRespanPayload(
  buildTestSpan({
    [RespanSpanAttributes.LLM_REQUEST_TYPE]: "response",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.AGENT,
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: JSON.stringify({
      content: "non-chat output",
    }),
  }),
);

assert.equal(nonChatPayload.log_type, RespanLogType.AGENT);
assert.equal(nonChatPayload.completion_message, undefined);
assert.deepEqual(nonChatPayload.output, {
  content: "non-chat output",
});

console.log("test_hybrid_exporter passed");
