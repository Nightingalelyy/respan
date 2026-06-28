import test from "node:test";
import assert from "node:assert/strict";

import {
  semanticSpanNameForSpan,
  transformReadableSpanName,
} from "../dist/processor/spanName.js";

function span(name, attributes = {}) {
  return {
    name,
    attributes,
  };
}

test("semantic span names use operation prefix and entity detail", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("triage-service.agent", {
        "traceloop.span.kind": "agent",
        "traceloop.entity.name": "triage-service",
      })
    ),
    "AGENT.triage-service"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("send_notification.tool", {
        "traceloop.span.kind": "tool",
        "traceloop.entity.name": "send_notification",
      })
    ),
    "TOOL.send_notification"
  );
});

test("semantic span names use integration hints and strip internal attrs", () => {
  const transformed = transformReadableSpanName(
    span("ai.generateText.doGenerate", {
      "respan.internal.span_name.kind": "generate",
      "respan.internal.span_name.detail": "doGenerate",
      "respan.entity.log_type": "text",
    }),
    "semantic"
  );

  assert.equal(transformed.name, "LLM");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
  assert.equal(transformed.attributes["respan.entity.log_type"], "text");
});
test("semantic span names collapse LLM-like log types to capital LLM", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("openai.chat", {
        "respan.entity.log_type": "chat",
      })
    ),
    "LLM"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("anthropic.generation", {
        "respan.entity.log_type": "generation",
      })
    ),
    "LLM"
  );
});

test("legacy span names only strip internal semantic hint attrs", () => {
  const transformed = transformReadableSpanName(
    span("ai.embed.doEmbed", {
      "respan.internal.span_name.kind": "embed",
      "respan.internal.span_name.detail": "doEmbed",
      "respan.entity.log_type": "embedding",
    }),
    "legacy"
  );

  assert.equal(transformed.name, "ai.embed.doEmbed");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
  assert.equal(transformed.attributes["respan.entity.log_type"], "embedding");
});


test("semantic span names let integration hints override generic operation-prefixed names", () => {
  const transformed = transformReadableSpanName(
    span("handoff.task", {
      "respan.internal.span_name.kind": "handoff",
      "respan.internal.span_name.detail": "triage-service_to_bank-service",
      "respan.entity.log_type": "handoff",
    }),
    "semantic"
  );

  assert.equal(transformed.name, "HANDOFF.triage-service_to_bank-service");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
});
