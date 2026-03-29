/**
 * Respan instrumentation plugin for the Vercel AI SDK.
 *
 * Registers a {@link VercelAITranslator} SpanProcessor that enriches Vercel AI SDK
 * OTEL spans with Traceloop/GenAI semantic conventions so they flow through the
 * unified Respan OTEL pipeline.
 *
 * The Vercel AI SDK already emits OTEL spans natively. This instrumentation
 * translates its attribute schema (ai.model.id, ai.prompt.messages, ai.response.text)
 * into the Traceloop format the Respan backend expects (gen_ai.request.model,
 * traceloop.entity.input, traceloop.entity.output, etc.).
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { VercelAIInstrumentor } from "@respan/instrumentation-vercel";
 *
 * const respan = new Respan({
 *   instrumentations: [new VercelAIInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import { trace } from "@opentelemetry/api";
import { VercelAITranslator } from "./_translator.js";

export { VercelAITranslator } from "./_translator.js";
export { VERCEL_SPAN_CONFIG, VERCEL_PARENT_SPANS } from "./constants/index.js";

export class VercelAIInstrumentor {
  public readonly name = "vercel-ai";

  /** Class-level flag: only register the translator once across all instances */
  private static _translatorRegistered = false;

  activate(): void {
    if (VercelAIInstrumentor._translatorRegistered) return;

    // Walk the TracerProvider chain to find one that supports addSpanProcessor.
    // trace.getTracerProvider() returns a ProxyTracerProvider; the real
    // NodeTracerProvider lives at ._delegate (or ._delegate._tracerProvider).
    const tp = trace.getTracerProvider() as any;
    const provider =
      (typeof tp?.addSpanProcessor === "function" && tp) ||
      (typeof tp?._delegate?.addSpanProcessor === "function" && tp._delegate) ||
      (typeof tp?._delegate?._tracerProvider?.addSpanProcessor === "function" && tp._delegate._tracerProvider) ||
      null;

    if (provider) {
      provider.addSpanProcessor(new VercelAITranslator());
      VercelAIInstrumentor._translatorRegistered = true;
    }
  }

  deactivate(): void {
    // The translator processor is managed by the TracerProvider lifecycle.
    // No explicit cleanup needed — it shuts down with the SDK.
  }
}
