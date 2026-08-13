import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import assert from "node:assert/strict";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = resolve(root, "src/index.ts");
const require = createRequire(import.meta.url);

async function loadSdkPrototypes() {
  const [chatModule, embeddingsModule] = await Promise.all([
    import(pathToFileURL(require.resolve("@openrouter/sdk/sdk/chat.js")).href),
    import(pathToFileURL(require.resolve("@openrouter/sdk/sdk/embeddings.js")).href),
  ]);
  return {
    chat: chatModule.Chat.prototype,
    embeddings: embeddingsModule.Embeddings.prototype,
  };
}

test("exports an OpenRouter instrumentor", async () => {
  const mod = await import(pathToFileURL(resolve(root, "dist/index.js")));
  assert.equal(typeof mod.OpenRouterInstrumentor, "function");
  assert.equal(typeof mod.instrumentOpenRouter, "function");

  const instrumentor = new mod.OpenRouterInstrumentor();
  assert.equal(instrumentor.name, "openrouter");
  assert.equal(instrumentor.isActive(), false);
});

test("keeps shared SDK patches until the last instrumentor deactivates", async () => {
  const mod = await import(pathToFileURL(resolve(root, "dist/index.js")));
  const prototypes = await loadSdkPrototypes();
  const originalSend = prototypes.chat.send;
  const originalGenerate = prototypes.embeddings.generate;
  const first = new mod.OpenRouterInstrumentor();
  const second = new mod.OpenRouterInstrumentor();

  try {
    await Promise.all([first.activate(), first.activate()]);
    assert.equal(first.isActive(), true);
    assert.notEqual(prototypes.chat.send, originalSend);
    assert.notEqual(prototypes.embeddings.generate, originalGenerate);
    const patchedSend = prototypes.chat.send;
    const patchedGenerate = prototypes.embeddings.generate;

    await second.activate();
    assert.equal(second.isActive(), true);
    assert.equal(prototypes.chat.send, patchedSend);
    assert.equal(prototypes.embeddings.generate, patchedGenerate);

    await first.deactivate();
    assert.equal(first.isActive(), false);
    assert.equal(second.isActive(), true);
    assert.equal(prototypes.chat.send, patchedSend);
    assert.equal(prototypes.embeddings.generate, patchedGenerate);

    await second.deactivate();
    assert.equal(second.isActive(), false);
    assert.equal(prototypes.chat.send, originalSend);
    assert.equal(prototypes.embeddings.generate, originalGenerate);
  } finally {
    await first.deactivate();
    await second.deactivate();
  }
});

test("does not activate against an incompatible OpenRouter SDK shape", async () => {
  const mod = await import(pathToFileURL(resolve(root, "dist/index.js")));
  const prototypes = await loadSdkPrototypes();
  const originalSend = prototypes.chat.send;
  const originalGenerate = prototypes.embeddings.generate;
  const instrumentor = new mod.OpenRouterInstrumentor();

  prototypes.chat.send = undefined;
  prototypes.embeddings.generate = undefined;
  try {
    await instrumentor.activate();
    assert.equal(instrumentor.isActive(), false);
  } finally {
    await instrumentor.deactivate();
    prototypes.chat.send = originalSend;
    prototypes.embeddings.generate = originalGenerate;
  }
});

test("waits for patch work to settle before cleaning up a failed activation", async () => {
  const mod = await import(pathToFileURL(resolve(root, "dist/index.js")));
  const prototypes = await loadSdkPrototypes();
  const originalSend = prototypes.chat.send;
  const originalGenerate = prototypes.embeddings.generate;
  const instrumentor = new mod.OpenRouterInstrumentor();
  const patchChat = instrumentor.patchChat.bind(instrumentor);
  let releaseChat;
  const chatGate = new Promise((resolveGate) => {
    releaseChat = resolveGate;
  });

  instrumentor.patchChat = async () => {
    await chatGate;
    await patchChat();
  };
  instrumentor.patchEmbeddings = async () => {
    throw new Error("forced embeddings patch failure");
  };

  const activation = instrumentor.activate();
  await new Promise((resolveImmediate) => setImmediate(resolveImmediate));
  releaseChat();
  try {
    await assert.rejects(activation, /forced embeddings patch failure/);
    await new Promise((resolveImmediate) => setImmediate(resolveImmediate));

    assert.equal(instrumentor.isActive(), false);
    assert.equal(prototypes.chat.send, originalSend);
    assert.equal(prototypes.embeddings.generate, originalGenerate);
    assert.equal(prototypes.chat.__respanPatchedSend, undefined);
    assert.equal(prototypes.embeddings.__respanPatchedGenerate, undefined);
  } finally {
    await instrumentor.deactivate();
    instrumentor.restorePatches();
  }
});

test("uses canonical constants and avoids banned package-owned aliases", async () => {
  const source = await readFile(sourcePath, "utf8");
  assert.match(source, /@opentelemetry\/semantic-conventions\/incubating/);
  assert.match(source, /@traceloop\/ai-semantic-conventions/);
  assert.match(source, /@respan\/respan-sdk/);
  assert.doesNotMatch(source, /respan\.span\.tools/);
  assert.doesNotMatch(source, /respan\.span\.tool_calls/);
  assert.doesNotMatch(source, /\["tools"\]/);
  assert.doesNotMatch(source, /\["tool_calls"\]/);
  assert.doesNotMatch(source, /\["model"\]/);
  assert.doesNotMatch(source, /\["prompt_tokens"\]/);
  assert.doesNotMatch(source, /\["completion_tokens"\]/);
  assert.doesNotMatch(source, /\["total_tokens"\]/);
});
