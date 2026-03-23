import { trace } from "@opentelemetry/api";

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

  constructor(instrumentorClass: any) {
    this._instrumentorClass = instrumentorClass;
    this.name = `openinference-${instrumentorClass.name || "unknown"}`;
  }

  activate(): void {
    this._instrumentor = new this._instrumentorClass();
    const tp = trace.getTracerProvider();

    if (typeof this._instrumentor.instrument === "function") {
      this._instrumentor.instrument({ tracerProvider: tp });
    } else if (tp && typeof (tp as any).addSpanProcessor === "function") {
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
