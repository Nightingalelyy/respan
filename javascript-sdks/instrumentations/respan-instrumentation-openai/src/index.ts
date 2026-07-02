/**
 * Respan instrumentation plugin for the OpenAI SDK.
 *
 * Wraps `@traceloop/instrumentation-openai` in the Respan plugin protocol.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { OpenAIInstrumentor } from "@respan/instrumentation-openai";
 *
 * const respan = new Respan({
 *   instrumentations: [new OpenAIInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

export class OpenAIInstrumentor {
  public readonly name = "openai";
  private static readonly _sharedState = {
    activeInstances: 0,
    instrumentor: null as any,
    openAI: null as any,
  };

  private _isInstrumented = false;

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const sharedState = OpenAIInstrumentor._sharedState;

    if (sharedState.activeInstances === 0) {
      const { trace } = await import("@opentelemetry/api");
      const { OpenAIInstrumentation } = await import(
        "@traceloop/instrumentation-openai"
      );

      sharedState.instrumentor = new OpenAIInstrumentation();
      sharedState.instrumentor.setTracerProvider(trace.getTracerProvider());
      sharedState.openAI = (await importOpenAISdk()).default;
      sharedState.instrumentor.manuallyInstrument(sharedState.openAI);
    }

    sharedState.activeInstances += 1;
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = OpenAIInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0 || !sharedState.instrumentor) return;

    try {
      sharedState.instrumentor.unpatch({ OpenAI: sharedState.openAI });
    } catch {
      /* ignore */
    }

    sharedState.instrumentor = null;
    sharedState.openAI = null;
  }
}

async function importOpenAISdk(): Promise<any> {
  try {
    const hostRequire = createRequire(`${process.cwd()}/package.json`);
    const resolved = hostRequire.resolve("openai");
    const esmEntry = join(dirname(resolved), "index.mjs");
    const entry = existsSync(esmEntry) ? esmEntry : resolved;
    return await import(pathToFileURL(entry).href);
  } catch {
    return await import("openai");
  }
}
