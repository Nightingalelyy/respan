import assert from "node:assert/strict";
import test from "node:test";

import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";

import {
  getRegisteredSpanTransformerKeys,
  registerSpanTransformer,
} from "../dist/index.js";
import { RespanCompositeProcessor } from "../dist/processor/composite.js";

const LOG_TYPE = "respan.entity.log_type";
const ENTITY_INPUT = "traceloop.entity.input";
const ENTITY_OUTPUT = "traceloop.entity.output";
const RAW_MODEL = "gen_ai.request.model";

class RecordingManager {
  started = [];
  ended = [];

  onStart(span) {
    this.started.push(span);
  }

  onEnd(span) {
    this.ended.push(span);
  }

  async shutdown() {}
  async forceFlush() {}
}

function createHarness() {
  const manager = new RecordingManager();
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  return { composite, manager, provider };
}

test("registration fails when no compatible Respan tracing host is active", () => {
  assert.throws(
    () => registerSpanTransformer("test.no-host", {}),
    /No compatible Respan span-transformer host is active/,
  );
});

test("OTel 2.10 tracers created before and after activation use transformers without provider mutation", async () => {
  const { manager, provider } = createHarness();
  const tracerBeforeActivation = provider.getTracer("test.before-activation");
  const activeProcessorBefore = provider._activeSpanProcessor;
  let disposeCount = 0;

  const registration = registerSpanTransformer("test.otel2", {
    onStart(span) {
      span.setAttribute(LOG_TYPE, "task");
    },
    onEnd(span) {
      span.attributes[ENTITY_INPUT] = "translated input";
      delete span.attributes[RAW_MODEL];
    },
    prepareForExport(span) {
      return {
        ...span,
        attributes: {
          ...span.attributes,
          [ENTITY_OUTPUT]: "translated output",
        },
      };
    },
    dispose() {
      disposeCount += 1;
    },
  });

  assert.equal(provider._activeSpanProcessor, activeProcessorBefore);
  const tracerAfterActivation = provider.getTracer("test.after-activation");

  tracerBeforeActivation
    .startSpan("raw.before", { attributes: { [RAW_MODEL]: "before" } })
    .end();
  tracerAfterActivation
    .startSpan("raw.after", { attributes: { [RAW_MODEL]: "after" } })
    .end();

  assert.equal(manager.ended.length, 2);
  for (const span of manager.ended) {
    assert.equal(span.attributes[LOG_TYPE], "task");
    assert.equal(span.attributes[ENTITY_INPUT], "translated input");
    assert.equal(span.attributes[ENTITY_OUTPUT], "translated output");
    assert.equal(span.attributes[RAW_MODEL], undefined);
  }

  registration.unregister();
  assert.equal(disposeCount, 1);
  await provider.shutdown();
  assert.equal(disposeCount, 1, "dispose is idempotent across host shutdown");
});

test("unregistration excludes new spans while captured in-flight spans drain before disposal", async () => {
  const { manager, provider } = createHarness();
  const tracer = provider.getTracer("test.in-flight");
  let disposeCount = 0;

  const startedBeforeRegistration = tracer.startSpan("raw.pre-registration", {
    attributes: { [RAW_MODEL]: "pre-registration" },
  });

  const registration = registerSpanTransformer("test.in-flight", {
    onStart(span) {
      span.setAttribute(LOG_TYPE, "task");
    },
    onEnd(span) {
      span.attributes[ENTITY_INPUT] = "captured";
    },
    dispose() {
      disposeCount += 1;
    },
  });

  const captured = tracer.startSpan("raw.captured", {
    attributes: { [RAW_MODEL]: "captured" },
  });
  registration.unregister();
  assert.equal(disposeCount, 0, "captured span holds the disposal barrier");

  startedBeforeRegistration.end();
  captured.end();
  assert.equal(disposeCount, 1);

  tracer
    .startSpan("raw.post-unregister", {
      attributes: { [RAW_MODEL]: "post-unregister" },
    })
    .end();

  const byName = Object.fromEntries(manager.ended.map((span) => [span.name, span]));
  assert.equal(
    byName["raw.pre-registration"].attributes[ENTITY_INPUT],
    undefined,
    "a transformer cannot retroactively claim an already-started span",
  );
  assert.equal(byName["raw.captured"].attributes[ENTITY_INPUT], "captured");
  assert.equal(
    byName["raw.post-unregister"].attributes[ENTITY_INPUT],
    undefined,
  );

  await provider.shutdown();
});

test("reactivating the same transformer cancels pending disposal while an older span drains", async () => {
  const { manager, provider } = createHarness();
  const tracer = provider.getTracer("test.reactivation");
  let disposeCount = 0;
  const transformer = {
    onStart(span) {
      span.setAttribute(LOG_TYPE, "task");
    },
    onEnd(span) {
      span.attributes[ENTITY_INPUT] = "state is still available";
    },
    dispose() {
      disposeCount += 1;
    },
  };

  const first = registerSpanTransformer("test.reactivation", transformer);
  const olderSpan = tracer.startSpan("reactivation.older");
  first.unregister();
  assert.equal(disposeCount, 0);

  const second = registerSpanTransformer("test.reactivation", transformer);
  olderSpan.end();
  assert.equal(
    disposeCount,
    0,
    "the draining older span must not dispose newly active shared state",
  );

  tracer.startSpan("reactivation.newer").end();
  assert.equal(manager.ended.length, 2);
  assert.ok(
    manager.ended.every(
      (span) => span.attributes[ENTITY_INPUT] === "state is still available",
    ),
  );

  second.unregister();
  assert.equal(disposeCount, 1);
  await provider.shutdown();
});

test("keys execute deterministically, same-key owners are refcounted, and hook failures are isolated", async () => {
  const { manager, provider } = createHarness();
  const tracer = provider.getTracer("test.ordering");
  const firstSameKeyCalls = [];
  const ignoredSameKeyCalls = [];

  const z = registerSpanTransformer("test.z", {
    onStart() {
      throw new Error("start failure is isolated");
    },
    onEnd(span) {
      span.attributes.order = `${span.attributes.order ?? ""}z`;
    },
  });
  const a = registerSpanTransformer("test.a", {
    onEnd(span) {
      span.attributes.order = `${span.attributes.order ?? ""}a`;
      throw new Error("end failure is isolated");
    },
  });
  const first = registerSpanTransformer("test.same-key", {
    onEnd(span) {
      firstSameKeyCalls.push(span.name);
      span.attributes[LOG_TYPE] = "task";
    },
  });
  const second = registerSpanTransformer("test.same-key", {
    onEnd(span) {
      ignoredSameKeyCalls.push(span.name);
    },
  });

  assert.deepEqual(getRegisteredSpanTransformerKeys(), [
    "test.a",
    "test.same-key",
    "test.z",
  ]);

  tracer.startSpan("ordered.first").end();
  assert.equal(manager.ended[0].attributes.order, "az");
  assert.deepEqual(firstSameKeyCalls, ["ordered.first"]);
  assert.deepEqual(ignoredSameKeyCalls, []);

  first.unregister();
  tracer.startSpan("ordered.second").end();
  assert.deepEqual(firstSameKeyCalls, ["ordered.first", "ordered.second"]);
  assert.deepEqual(ignoredSameKeyCalls, []);

  second.unregister();
  a.unregister();
  z.unregister();
  assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
  await provider.shutdown();
});

test("final host shutdown disposes active registrations and resets availability", async () => {
  const { provider } = createHarness();
  let disposeCount = 0;
  const registration = registerSpanTransformer("test.host-shutdown", {
    dispose() {
      disposeCount += 1;
    },
  });

  await provider.shutdown();
  assert.equal(disposeCount, 1);
  assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
  registration.unregister();
  assert.equal(disposeCount, 1);
  assert.throws(
    () => registerSpanTransformer("test.after-host-shutdown", {}),
    /No compatible Respan span-transformer host is active/,
  );
});
