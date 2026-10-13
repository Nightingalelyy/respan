import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import * as OpenAI from "openai";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import { Respan } from "@respan/respan";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { withSpan } from "@arizeai/openinference-core";
import { OpenInferenceInstrumentor } from "../dist/index.js";

function fixtureClient() {
  return new OpenAI.OpenAI({
    apiKey: "local-fixture-only",
    baseURL: "http://127.0.0.1:1/v1",
    fetch: async () =>
      new Response(
        JSON.stringify({
          id: "native-completion",
          object: "chat.completion",
          created: 0,
          model: "fixture-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "fixture answer" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
  });
}

test("released Respan registry + mature real OpenInference/OpenAI adapter: ownership, deactivation, drain and reactivation", async () => {
  const server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{}");
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const original = OpenAI.OpenAI.Chat.Completions.prototype.create;
  const first = new OpenInferenceInstrumentor(OpenAIInstrumentation, OpenAI);
  const second = new OpenInferenceInstrumentor(OpenAIInstrumentation, OpenAI);
  const respan = new Respan({
    apiKey: "local-fixture-only",
    baseURL: `http://127.0.0.1:${server.address().port}`,
    instrumentations: [first, second],
    silenceInitializationMessage: true,
  });
  const exporter = new InMemorySpanExporter();
  try {
    await respan.initialize();
    second.activate();
    respan.addProcessor({
      name: "capture",
      exporter,
      filter: () => true,
      disableBatch: true,
    });
    const patched = OpenAI.OpenAI.Chat.Completions.prototype.create;
    assert.notEqual(patched, original);
    first.activate();
    assert.equal(OpenAI.OpenAI.Chat.Completions.prototype.create, patched);
    const client = fixtureClient();
    const request = {
      model: "fixture-model",
      messages: [{ role: "user", content: "fixture question" }],
    };
    const promise = client.chat.completions.create(request);
    assert.equal(typeof promise.withResponse, "function");
    assert.equal(typeof promise.asResponse, "function");
    const result = await promise;
    assert.equal(result.choices[0].message.content, "fixture answer");
    first.deactivate();
    assert.equal(OpenAI.OpenAI.Chat.Completions.prototype.create, patched);
    second.deactivate();
    assert.equal(OpenAI.OpenAI.Chat.Completions.prototype.create, original);
    first.activate();
    let resolve;
    const gate = new Promise((done) => {
      resolve = done;
    });
    const pending = withSpan(
      async () => {
        await gate;
        return "drained output";
      },
      { name: "native-drain" },
    )();
    first.deactivate();
    first.activate();
    resolve();
    assert.equal(await pending, "drained output");
    await respan.flush();
    const span = exporter
      .getFinishedSpans()
      .find(
        (span) => span.attributes["traceloop.entity.name"] === "native-drain",
      );
    assert.ok(span);
    assert.equal(span.attributes["traceloop.entity.output"], "drained output");
    assert.equal(span.attributes["openinference.span.kind"], undefined);
    first.deactivate();
    assert.equal(OpenAI.OpenAI.Chat.Completions.prototype.create, original);

    first.activate();
    const ownedPatch = OpenAI.OpenAI.Chat.Completions.prototype.create;
    function foreignPatch(...args) {
      return ownedPatch.apply(this, args);
    }
    OpenAI.OpenAI.Chat.Completions.prototype.create = foreignPatch;
    first.deactivate();
    assert.equal(OpenAI.OpenAI.Chat.Completions.prototype.create, foreignPatch);
    OpenAI.OpenAI.Chat.Completions.prototype.create = original;

    let finish;
    let instruments = 0;
    let disables = 0;
    class AsyncInstrumentor {
      instrument() {
        instruments++;
        return new Promise((resolve) => {
          finish = resolve;
        });
      }
      uninstrument() {
        disables++;
      }
    }
    const asynchronous = new OpenInferenceInstrumentor(AsyncInstrumentor);
    const activation = asynchronous.activate();
    assert.equal(asynchronous.activate(), activation);
    assert.equal(instruments, 1);
    asynchronous.deactivate();
    finish();
    await activation;
    await Promise.resolve();
    assert.equal(asynchronous.isActive(), false);
    assert.equal(disables, 1);
    const next = asynchronous.activate();
    finish();
    await next;
    assert.equal(asynchronous.isActive(), true);
    asynchronous.deactivate();
    await Promise.resolve();
    assert.equal(disables, 2);
  } finally {
    await respan.shutdown();
    await new Promise((resolve) => server.close(resolve));
  }
});
