import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { context } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { observeRequest, type Operation } from "./request.js";

export interface OpenAIInstrumentorOptions {
  /** Capture input and output. The active trace-content context takes precedence. */
  traceContent?: boolean;
  /** Optional SDK constructor or module for applications with multiple SDK copies. */
  openAI?: any;
}

interface Patch {
  original: any;
  wrapped: any;
  owners: Set<OpenAIInstrumentor>;
}

/** Instruments the OpenAI 4–7 resource APIs without replacing SDK result objects. */
export class OpenAIInstrumentor {
  readonly name = "openai";
  private static readonly patches = new Map<any, Patch>();
  private readonly targets = new Set<any>();
  private active = false;
  private activation?: Promise<void>;

  constructor(private readonly options: OpenAIInstrumentorOptions = {}) {}

  activate(): Promise<void> {
    // A deactivate may cancel an in-flight install. Recheck after it settles.
    if (this.activation) return this.activation.then(() => this.activate());
    if (this.active) return Promise.resolve();
    this.active = true;
    this.activation = this.install()
      .catch((error) => {
        this.deactivate();
        throw error;
      })
      .finally(() => {
        this.activation = undefined;
      });
    return this.activation;
  }

  private async install(): Promise<void> {
    const modules = this.options.openAI
      ? [this.options.openAI]
      : await importOpenAISdks();
    if (!this.active) return;
    for (const module of modules) {
      const OpenAI = module.OpenAI ?? module.default ?? module;
      for (const [target, operation] of [
        [OpenAI.Chat?.Completions?.prototype, "chat"],
        [OpenAI.Completions?.prototype, "text"],
        [OpenAI.Responses?.prototype, "responses"],
        [OpenAI.Embeddings?.prototype, "embedding"],
      ] as const) {
        if (typeof target?.create !== "function" || this.targets.has(target))
          continue;
        let patch = OpenAIInstrumentor.patches.get(target);
        if (!patch) {
          const original = target.create;
          const owners = new Set<OpenAIInstrumentor>();
          const wrapped = function (this: any, ...args: any[]) {
            const owner = owners.values().next().value;
            if (
              !owner ||
              isAzureOpenAIResource(this) ||
              isTracingSuppressed(context.active())
            ) {
              return original.apply(this, args);
            }
            return observeRequest(
              original,
              this,
              args,
              operation as Operation,
              owner.options.traceContent,
            );
          };
          patch = { original, wrapped, owners };
          target.create = wrapped;
          OpenAIInstrumentor.patches.set(target, patch);
        }
        patch.owners.add(this);
        this.targets.add(target);
      }
    }
  }

  deactivate(): void {
    this.active = false;
    for (const target of this.targets) {
      const patch = OpenAIInstrumentor.patches.get(target);
      if (!patch) continue;
      patch.owners.delete(this);
      if (patch.owners.size) continue;
      // Do not overwrite a wrapper installed by another library after activation.
      if (target.create === patch.wrapped) target.create = patch.original;
      OpenAIInstrumentor.patches.delete(target);
    }
    this.targets.clear();
  }
}

function isAzureOpenAIResource(receiver: any): boolean {
  const client = receiver?._client ?? receiver;
  for (
    let prototype = client;
    prototype;
    prototype = Object.getPrototypeOf(prototype)
  ) {
    if (prototype.constructor?.name === "AzureOpenAI") return true;
  }
  const baseURL =
    typeof client?.baseURL === "string" ? client.baseURL.toLowerCase() : "";
  return (
    typeof client?.apiVersion === "string" &&
    (baseURL.includes("azure") || Boolean(client.deploymentName))
  );
}

async function importOpenAISdks(): Promise<any[]> {
  let require = createRequire(`${process.cwd()}/package.json`);
  let resolved: string;
  try {
    resolved = require.resolve("openai");
  } catch {
    require = createRequire(import.meta.url);
    resolved = require.resolve("openai");
  }
  const modules = [require(resolved)];
  const esmEntry = join(dirname(resolved), "index.mjs");
  if (existsSync(esmEntry))
    modules.push(await import(pathToFileURL(esmEntry).href));
  return modules;
}
