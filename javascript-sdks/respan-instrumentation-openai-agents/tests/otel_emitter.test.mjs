import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { emitSdkItem } from "../dist/_otel_emitter.js";

const captureState = { spans: [] };

trace.setGlobalTracerProvider({
  getTracer() {
    return {};
  },
  activeSpanProcessor: {
    onEnd(span) {
      captureState.spans.push(span);
    },
  },
});

function emitAndCapture(item) {
  captureState.spans = [];
  emitSdkItem(item);
  assert.equal(captureState.spans.length, 1);
  return captureState.spans[0].attributes;
}

function makeBaseSpanData(spanData) {
  return {
    traceId: "trace_test_123",
    spanId: "span_test_456",
    parentId: "parent_test_789",
    started_at: new Date().toISOString(),
    ended_at: new Date().toISOString(),
    error: null,
    spanData,
  };
}

test("emit response preserves chat history and carries tool calls in output", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "response",
      _input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Tell me everything about Tokyo" }],
        },
        {
          type: "function_call",
          call_id: "call_weather",
          name: "get_weather",
          arguments: "{\"city\":\"Tokyo\"}",
        },
        {
          type: "function_call_output",
          call_id: "call_weather",
          output: "Sunny, 22°C in Tokyo",
        },
      ],
      _response: {
        model: "gpt-4o",
        output: [
          {
            type: "function_call",
            call_id: "call_stats",
            name: "get_city_stats",
            arguments: "{\"city\":\"Tokyo\"}",
          },
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Here is Tokyo info" }],
          },
        ],
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Get weather",
            parameters: { type: "object" },
          },
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 3,
        },
      },
    }),
  );

  const inputMessages = JSON.parse(attrs["traceloop.entity.input"]);
  assert.deepEqual(inputMessages, [
    { role: "user", content: "Tell me everything about Tokyo" },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_weather",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Tokyo\"}",
          },
        },
      ],
    },
    {
      role: "tool",
      content: "Sunny, 22°C in Tokyo",
      tool_call_id: "call_weather",
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [
    {
      role: "assistant",
      content: "Here is Tokyo info",
      tool_calls: [
        {
          id: "call_stats",
          type: "function",
          function: {
            name: "get_city_stats",
            arguments: "{\"city\":\"Tokyo\"}",
          },
        },
      ],
    },
  ]);
  assert.deepEqual(attrs.tools, [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather",
        parameters: { type: "object" },
      },
    },
  ]);
  assert.ok(!attrs["traceloop.entity.input"].includes("[object Object]"));
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit generation extracts tool calls without object stringification", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "generation",
      model: "gpt-4o",
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Use the tool" }],
        },
      ],
      output: [
        {
          type: "function_call",
          call_id: "call_docs",
          name: "search_docs",
          arguments: "{\"query\":\"otel\"}",
        },
      ],
      usage: {
        prompt_tokens: 8,
        completion_tokens: 2,
      },
    }),
  );

  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Use the tool" },
  ]);
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_docs",
          type: "function",
          function: {
            name: "search_docs",
            arguments: "{\"query\":\"otel\"}",
          },
        },
      ],
    },
  ]);
  assert.ok(!attrs["traceloop.entity.input"].includes("[object Object]"));
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit function serializes wrapped text tool output", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "function",
      name: "get_weather",
      input: { city: "Tokyo" },
      output: { type: "text", text: "Sunny, 22°C in Tokyo" },
    }),
  );

  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    role: "tool",
    content: "Sunny, 22°C in Tokyo",
  });
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit generation preserves boolean false output", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "generation",
      model: "gpt-4o",
      input: "Return false",
      output: false,
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
      },
    }),
  );

  assert.equal(attrs["traceloop.entity.output"], "false");
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit response preserves Chat Completions tool_calls messages", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "response",
      _input: [
        { role: "user", content: "Check Tokyo weather" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_weather_chat",
              type: "function",
              function: {
                name: "get_weather",
                arguments: "{\"city\":\"Tokyo\"}",
              },
            },
          ],
        },
      ],
      _response: {
        model: "gpt-4o",
        output: "Done",
        usage: {
          input_tokens: 5,
          output_tokens: 1,
        },
      },
    }),
  );

  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Check Tokyo weather" },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_weather_chat",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Tokyo\"}",
          },
        },
      ],
    },
  ]);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});
