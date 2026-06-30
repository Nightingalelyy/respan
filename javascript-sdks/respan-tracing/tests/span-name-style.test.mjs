import test from "node:test";
import assert from "node:assert/strict";

import {
  semanticSpanNameForSpan,
  transformReadableSpanBatch,
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
    "agent.triage-service"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("send_notification.tool", {
        "traceloop.span.kind": "tool",
        "traceloop.entity.name": "send_notification",
      })
    ),
    "tool.send_notification"
  );
});

test("semantic span names use integration hints and strip internal attrs", () => {
  const transformed = transformReadableSpanName(
    span("ai.generateText.doGenerate", {
      "respan.internal.span_name.kind": "generate",
      "respan.internal.span_name.detail": "doGenerate",
      "respan.entity.log_type": "text",
      "gen_ai.request.model": "gpt-4o-mini",
    }),
    "semantic"
  );

  assert.equal(transformed.name, "llm.gpt-4o-mini");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
  assert.equal(transformed.attributes["respan.entity.log_type"], "text");
});
test("semantic span names use lowercase llm prefix with model suffix", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("openai.chat", {
        "respan.entity.log_type": "chat",
        "gen_ai.request.model": "gpt-4o",
      })
    ),
    "llm.gpt-4o"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("ai.generateText.doGenerate", {
        "respan.entity.log_type": "generation",
        "ai.model.id": "claude-3-5-sonnet",
      })
    ),
    "llm.claude-3-5-sonnet"
  );

  assert.equal(
    semanticSpanNameForSpan(
      span("llm.doGenerate", {
        "respan.entity.log_type": "text",
        "gen_ai.request.model": "gpt-4.1",
      })
    ),
    "llm.gpt-4.1"
  );
});

test("semantic span names keep embedding operation name", () => {
  assert.equal(
    semanticSpanNameForSpan(
      span("ai.embed.doEmbed", {
        "respan.entity.log_type": "embedding",
        "ai.model.id": "text-embedding-3-small",
      })
    ),
    "embedding"
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

  assert.equal(transformed.name, "handoff.triage-service_to_bank-service");
  assert.equal(transformed.attributes["respan.internal.span_name.kind"], undefined);
  assert.equal(transformed.attributes["respan.internal.span_name.detail"], undefined);
});

test("semantic exporter drops structural Vercel LLM wrappers and reparents child spans", () => {
  const agent = span("agent.triage-service", {
    "traceloop.span.kind": "agent",
    "traceloop.entity.name": "triage-service",
  });
  agent.parentSpanId = "root-span";
  agent.spanContext = () => ({ spanId: "agent-span" });

  const wrapper = span("ai.generateText", {
    "respan.entity.log_type": "text",
    "ai.telemetry.functionId": "triage-service",
    "resource.name": "triage-service",
  });
  wrapper.parentSpanId = "agent-span";
  wrapper.spanContext = () => ({ spanId: "wrapper-span" });

  const child = span("ai.generateText.doGenerate", {
    "respan.entity.log_type": "chat",
    "gen_ai.request.model": "gpt-4o",
    "traceloop.entity.input": "[{\"role\":\"user\",\"content\":\"hi\"}]",
    "traceloop.entity.output": "{\"role\":\"assistant\",\"content\":\"hello\"}",
  });
  child.parentSpanId = "wrapper-span";
  child.spanContext = () => ({ spanId: "child-span" });

  const exported = transformReadableSpanBatch([agent, wrapper, child], "semantic");

  assert.deepEqual(exported.map((item) => item.name), [
    "agent.triage-service",
    "llm.gpt-4o",
  ]);
  assert.equal(exported[1].parentSpanId, "agent-span");
});

test("semantic exporter renames Vercel embedding child spans to embedding", () => {
  const exported = transformReadableSpanBatch(
    [
      span("ai.embed.doEmbed", {
        "respan.entity.log_type": "embedding",
        "ai.model.id": "text-embedding-3-small",
      }),
    ],
    "semantic"
  );

  assert.equal(exported[0].name, "embedding");
});
