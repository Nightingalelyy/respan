import { trace } from "@opentelemetry/api";
import { OpenInferenceTranslator } from "./_translator.js";

export { OpenInferenceTranslator } from "./_translator.js";

/**
 * Generic Respan instrumentation wrapper for any OpenInference instrumentor.
 *
 * OpenInference instrumentors (from `@arizeai/openinference-instrumentation-*`)
 * may expose either:
 * - `.instrument({ tracerProvider })` — standard OTEL instrumentor interface
 * - SpanProcessor interface (added via `tracerProvider.addSpanProcessor()`)
 *
 * This wrapper detects the interface and handles both patterns.
 *
 * Automatically registers an {@link OpenInferenceTranslator} SpanProcessor that
 * converts OI span attributes to OpenLLMetry/Traceloop format before export.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { OpenInferenceInstrumentor } from "@respan/instrumentation-openinference";
 * import { GoogleADKInstrumentor } from "@arizeai/openinference-instrumentation-google-adk";
 *
 * const respan = new Respan({
 *   instrumentations: [new OpenInferenceInstrumentor(GoogleADKInstrumentor)],
 * });
 * await respan.initialize();
 * ```
 */
export class OpenInferenceInstrumentor {
  public readonly name: string;
  private _instrumentorClass: any;
  private _instrumentor: any = null;
  private _isInstrumented = false;
  private _isSpanProcessor = false;

  /** Class-level flag: only register the translator once across all instances */
  private static _translatorRegistered = false;

  private _sdkModule: any;

  /**
   * @param instrumentorClass - The OI instrumentor class (e.g. GoogleADKInstrumentor)
   * @param sdkModule - Optional SDK module for ESM manual instrumentation.
   *   Required for instrumentors that use `manuallyInstrument(module)` instead of
   *   auto-patching (e.g. Claude Agent SDK in ESM environments).
   */
  constructor(instrumentorClass: any, sdkModule?: any) {
    this._instrumentorClass = instrumentorClass;
    this._sdkModule = sdkModule;
    this.name = `openinference-${instrumentorClass.name || "unknown"}`;
  }

  activate(): void {
    this._instrumentor = new this._instrumentorClass();
    const tp = trace.getTracerProvider();

    // Register the OI → OpenLLMetry translator once
    if (!OpenInferenceInstrumentor._translatorRegistered && tp && typeof (tp as any).addSpanProcessor === "function") {
      (tp as any).addSpanProcessor(new OpenInferenceTranslator());
      OpenInferenceInstrumentor._translatorRegistered = true;
    }

    // Set tracer provider if the instrumentor supports it
    if (typeof this._instrumentor.setTracerProvider === "function") {
      this._instrumentor.setTracerProvider(tp);
    }

    // ESM manual instrumentation (e.g. Claude Agent SDK)
    if (this._sdkModule && typeof this._instrumentor.manuallyInstrument === "function") {
      this._instrumentor.manuallyInstrument(this._sdkModule);
    }
    // Standard OTEL auto-patching
    else if (typeof this._instrumentor.instrument === "function") {
      this._instrumentor.instrument({ tracerProvider: tp });
    }
    // SpanProcessor-based (e.g. pydantic-ai, strands-agents)
    else if (tp && typeof (tp as any).addSpanProcessor === "function") {
      (tp as any).addSpanProcessor(this._instrumentor);
      this._isSpanProcessor = true;
    }
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (this._isInstrumented && this._instrumentor) {
      try {
        if (this._isSpanProcessor) {
          this._instrumentor.shutdown();
        } else {
          this._instrumentor.uninstrument();
        }
      } catch {
        /* ignore */
      }
      this._isInstrumented = false;
    }
  }
}
