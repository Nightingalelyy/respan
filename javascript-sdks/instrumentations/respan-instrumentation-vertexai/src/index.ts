import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { context } from "@opentelemetry/api";
import { type CaptureOptions } from "./_capture.js";
import {
  startOperation,
  finishOperation,
  type VertexOperation,
} from "./_otel_emitter.js";
import { wrapStream } from "./_streaming.js";
type Prototype = Record<string, any>;
export interface VertexAIModule {
  GenerativeModel?: { prototype?: Prototype };
  GenerativeModelPreview?: { prototype?: Prototype };
  ChatSession?: { prototype?: Prototype };
  ChatSessionPreview?: { prototype?: Prototype };
}
export interface GoogleGenAIModule {
  Models?: { prototype?: Prototype };
}
export interface VertexAIInstrumentorOptions extends CaptureOptions {
  sdkModule?: VertexAIModule;
  genaiModule?: GoogleGenAIModule;
}
interface Patch {
  prototype: Prototype;
  method: string;
  original: any;
  wrapped: any;
  owners: Set<VertexAIInstrumentor>;
}
const patches: Patch[] = [];
function options(owners: Set<VertexAIInstrumentor>): CaptureOptions {
  return {
    traceContent: [...owners].every(
      (owner) => owner.options.traceContent !== false,
    ),
    recordInputs: [...owners].every(
      (owner) => owner.options.recordInputs !== false,
    ),
    recordOutputs: [...owners].every(
      (owner) => owner.options.recordOutputs !== false,
    ),
  };
}
function instrumentResult(result: any, op: VertexOperation): any {
  if (result && typeof result[Symbol.asyncIterator] === "function")
    return wrapStream(result, op);
  if (result && typeof result === "object" && "response" in result) {
    const response = result.response;
    if (response && typeof response.then === "function") {
      // Native Promise await bypasses own .then overrides; retain its identity.
      response
        .then(
          (value: any) => {
            finishOperation(op, value);
          },
          (error: unknown) => {
            finishOperation(op, undefined, error, true);
          },
        )
        .catch(() => {});
      if (result.stream) wrapStream(result.stream, op);
      return result;
    }
    finishOperation(op, response);
    return result;
  }
  finishOperation(op, result);
  return result;
}
function patch(
  owner: VertexAIInstrumentor,
  prototype: Prototype | undefined,
  method: string,
  genai = false,
  embedding = false,
  chat = false,
  generation = false,
): void {
  if (!prototype || typeof prototype[method] !== "function") return;
  const existing = patches.find(
    (p) => p.prototype === prototype && p.method === method,
  );
  if (existing) {
    existing.owners.add(owner);
    return;
  }
  const owners = new Set([owner]);
  const original = prototype[method];
  const wrapped = function (this: any, ...args: any[]) {
    if (genai && this?.apiClient?.isVertexAI?.() !== true)
      return original.apply(this, args);
    let op: VertexOperation;
    try {
      op = startOperation(
        this,
        args,
        options(owners),
        genai,
        embedding,
        chat,
        generation,
      );
    } catch {
      return original.apply(this, args);
    }
    try {
      const result = context.with(op.ctx, () => original.apply(this, args));
      if (result && typeof result.then === "function") {
        return result.then(
          (value: any) => instrumentResult(value, op),
          (error: unknown) => {
            finishOperation(op, undefined, error, true);
            throw error;
          },
        );
      }
      return instrumentResult(result, op);
    } catch (error) {
      finishOperation(op, undefined, error, true);
      throw error;
    }
  };
  prototype[method] = wrapped;
  patches.push({ prototype, method, original, wrapped, owners });
}
async function load(name: string): Promise<any[]> {
  const result: any[] = [];
  for (const base of [
    import.meta.url,
    pathToFileURL(process.cwd() + "/__respan_runtime__.js").href,
  ])
    try {
      const require = createRequire(base);
      const entry = require.resolve(name);
      for (const candidate of [
        entry,
        entry.endsWith(".cjs") ? entry.slice(0, -4) + ".mjs" : entry,
      ]) {
        if (!existsSync(candidate)) continue;
        const module = await import(pathToFileURL(candidate).href);
        if (!result.includes(module)) result.push(module);
      }
    } catch {}
  return result;
}
/** Instruments released Vertex AI clients and Google GenAI calls using the Vertex backend. */
export class VertexAIInstrumentor {
  readonly name = "vertexai";
  private active = false;
  private generation = 0;
  private activating?: Promise<void>;
  constructor(readonly options: VertexAIInstrumentorOptions = {}) {}
  isActive(): boolean {
    return this.active;
  }
  activate(): Promise<void> {
    if (this.active) return Promise.resolve();
    if (this.activating) return this.activating;
    const generation = this.generation;
    const pending = this.activateOnce(generation).finally(() => {
      if (this.activating === pending) this.activating = undefined;
    });
    this.activating = pending;
    return pending;
  }
  private async activateOnce(generation: number): Promise<void> {
    const legacy = this.options.sdkModule
      ? [this.options.sdkModule]
      : await load("@google-cloud/vertexai");
    const modern = this.options.genaiModule
      ? [this.options.genaiModule]
      : await load("@google/genai");
    if (generation !== this.generation) return;
    for (const module of legacy) {
      for (const key of ["GenerativeModel", "GenerativeModelPreview"]) {
        const proto = module[key]?.prototype;
        patch(this, proto, "generateContent");
        patch(this, proto, "generateContentStream");
      }
      for (const key of ["ChatSession", "ChatSessionPreview"]) {
        const proto = module[key]?.prototype;
        patch(this, proto, "sendMessage", false, false, true);
        patch(this, proto, "sendMessageStream", false, false, true);
      }
    }
    for (const module of modern) {
      const proto = module.Models?.prototype;
      patch(this, proto, "generateContentInternal", true);
      patch(this, proto, "generateContentStreamInternal", true);
      patch(this, proto, "embedContentInternal", true, true);
      for (const method of [
        "generateImagesInternal",
        "editImageInternal",
        "upscaleImageInternal",
        "recontextImage",
        "segmentImage",
        "generateVideosInternal",
      ])
        patch(this, proto, method, true, false, false, true);
    }
    if (patches.some((p) => p.owners.has(this))) this.active = true;
  }
  deactivate(): void {
    this.generation++;
    this.active = false;
    for (const p of [...patches]) {
      p.owners.delete(this);
      if (p.owners.size) continue;
      if (p.prototype[p.method] === p.wrapped)
        p.prototype[p.method] = p.original;
      patches.splice(patches.indexOf(p), 1);
    }
  }
}
export { buildGenerateContentAttrs } from "./_otel_emitter.js";
export {
  extractToolCalls,
  extractTools,
  formatInput,
  formatOutput,
  normalizeInputMessages,
  requestPayloadFromCall,
} from "./_translator.js";
