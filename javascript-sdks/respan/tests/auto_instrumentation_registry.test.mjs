import assert from "node:assert/strict";
import test from "node:test";

import {
  AUTO_INSTRUMENTATION_REGISTRY,
  DIRECT_LLM_AUTO_INSTRUMENTATIONS,
} from "../dist/_auto_instrumentation_registry.js";

const registryById = new Map(
  AUTO_INSTRUMENTATION_REGISTRY.map((entry) => [entry.id, entry]),
);

test("clean committed direct LLM packages are enabled for onboarding", () => {
  const expected = {
    "aws-bedrock": [
      "@aws-sdk/client-bedrock-runtime",
      "@respan/instrumentation-aws-bedrock",
      "AWSBedrockInstrumentor",
    ],
    cohere: [
      "cohere-ai",
      "@respan/instrumentation-cohere",
      "CohereInstrumentor",
    ],
    "together-ai": [
      "together-ai",
      "@respan/instrumentation-together-ai",
      "TogetherAIInstrumentor",
    ],
    writer: [
      "writer-sdk",
      "@respan/instrumentation-writer",
      "WriterInstrumentor",
    ],
  };

  for (const [
    id,
    [sdkPackage, instrumentationPackage, instrumentorClass],
  ] of Object.entries(expected)) {
    const entry = registryById.get(id);
    assert.ok(entry, id + " must be present in the onboarding registry");
    assert.equal(entry.category, "direct-llm");
    assert.equal(entry.enabledByDefault, true);
    assert.equal(entry.sdkPackage, sdkPackage);
    assert.equal(entry.instrumentationPackage, instrumentationPackage);
    assert.equal(entry.instrumentorClass, instrumentorClass);
  }

  const directIds = new Set(
    DIRECT_LLM_AUTO_INSTRUMENTATIONS.map((entry) => entry.id),
  );
  for (const id of Object.keys(expected)) {
    assert.ok(directIds.has(id), id + " must be auto-discoverable");
  }
});

test("clean committed framework packages stay explicit-only", () => {
  const expected = {
    "codex-sdk": "agent-framework",
    "cursor-sdk": "agent-framework",
    livekit: "agent-framework",
    flue: "app-framework",
  };

  for (const [id, category] of Object.entries(expected)) {
    const entry = registryById.get(id);
    assert.ok(entry, id + " must be present in the onboarding registry");
    assert.equal(entry.category, category);
    assert.equal(entry.enabledByDefault, false);
    assert.ok(entry.autoDisabledReason);
  }
});

test("registry instrumentation packages are unique", () => {
  const packages = AUTO_INSTRUMENTATION_REGISTRY.map(
    (entry) => entry.instrumentationPackage,
  );
  assert.equal(new Set(packages).size, packages.length);
});
