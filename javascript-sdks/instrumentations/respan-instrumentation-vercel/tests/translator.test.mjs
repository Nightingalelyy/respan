import test from "node:test";
import assert from "node:assert/strict";

import { VercelAITranslator } from "../dist/_translator.js";

/**
 * Build a minimal Vercel AI SDK-style span and run both translator phases on it.
 * Returns the mutated attribute map.
 */
function runTranslator(name, attributes) {
  const span = {
    name,
    instrumentationLibrary: { name: "ai" },
    attributes: { ...attributes },
  };
  const writableSpan = {
    name,
    setAttribute(k, v) { span.attributes[k] = v; },
  };
  const tr = new VercelAITranslator();
  tr.onStart(writableSpan);
  tr.onEnd(span);
  return span.attributes;
}

const baseLLMSpan = {
  "ai.model.id": "gpt-4o-mini",
  "ai.prompt.messages": JSON.stringify([
    { role: "user", content: "hi" },
  ]),
  "ai.response.text": "hello",
  "gen_ai.usage.input_tokens": 5,
  "gen_ai.usage.output_tokens": 3,
};

test("customer_params with documented {email, name} shape populates Customer columns", () => {
  const attrs = runTranslator("ai.generateText.doGenerate", {
    ...baseLLMSpan,
    "ai.telemetry.metadata.customer_params": JSON.stringify({
      customer_identifier: "user_42",
      email: "frank@respan.ai",
      name: "Frank",
    }),
  });

  assert.equal(attrs["respan.customer_params.customer_identifier"], "user_42");
  assert.equal(attrs["respan.customer_params.email"], "frank@respan.ai");
  assert.equal(attrs["respan.customer_params.name"], "Frank");
});

test("customer_params with legacy {customer_email, customer_name} aliases still works", () => {
  const attrs = runTranslator("ai.generateText.doGenerate", {
    ...baseLLMSpan,
    "ai.telemetry.metadata.customer_params": JSON.stringify({
      customer_identifier: "user_99",
      customer_email: "legacy@example.com",
      customer_name: "Legacy",
    }),
  });

  assert.equal(attrs["respan.customer_params.customer_identifier"], "user_99");
  assert.equal(attrs["respan.customer_params.email"], "legacy@example.com");
  assert.equal(attrs["respan.customer_params.name"], "Legacy");
});

test("ai.generateText.doGenerate is classified as LLM (text), not task", () => {
  const attrs = runTranslator("ai.generateText.doGenerate", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  // The composite processor reserves traceloop.span.kind for user-decorated
  // spans (withWorkflow / withTask / withAgent). Auto-emitted Vercel spans
  // must leave it unset, otherwise LLM spans get reclassified as tasks.
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("ai.streamText.doStream is also LLM with no traceloop.span.kind", () => {
  const attrs = runTranslator("ai.streamText.doStream", baseLLMSpan);

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("malformed customer_params JSON does not throw and skips customer fields", () => {
  const attrs = runTranslator("ai.generateText.doGenerate", {
    ...baseLLMSpan,
    "ai.telemetry.metadata.customer_params": "{ not json",
  });

  assert.equal(attrs["respan.customer_params.email"], undefined);
  assert.equal(attrs["respan.customer_params.name"], undefined);
  // Span itself should still be enriched normally
  assert.equal(attrs["respan.entity.log_type"], "text");
});
