import assert from "node:assert/strict";
import test from "node:test";

import { ROOT_CONTEXT } from "@opentelemetry/api";

import { registerSpanTransformer } from "../dist/index.js";
import {
  getClient,
  shutdownTracing,
  startTracing,
} from "../dist/utils/tracing.js";

test("a failed NodeSDK start releases the transformer host and runtime singletons", async () => {
  const forcedFailure = new Error("forced context-manager enable failure");
  const contextManager = {
    active() {
      return ROOT_CONTEXT;
    },
    bind(_context, target) {
      return target;
    },
    disable() {
      return this;
    },
    enable() {
      throw forcedFailure;
    },
    with(_context, fn, thisArg, ...args) {
      return fn.apply(thisArg, args);
    },
  };
  const exporter = {
    export(_spans, callback) {
      callback({ code: 0 });
    },
    shutdown() {
      return Promise.resolve();
    },
  };

  await assert.rejects(
    startTracing({
      apiKey: "test-api-key-long-enough",
      appName: "start-failure-test",
      baseURL: "http://127.0.0.1",
      contextManager,
      disableBatch: true,
      exporter,
      silenceInitializationMessage: true,
    }),
    forcedFailure,
  );

  assert.equal(getClient(), undefined);
  assert.throws(
    () => registerSpanTransformer("test.dead-runtime", {}),
    /No compatible Respan span-transformer host is active/,
  );

  // Failure cleanup clears the singleton, so a later explicit shutdown is a
  // harmless no-op rather than a second processor shutdown.
  await shutdownTracing();
});
