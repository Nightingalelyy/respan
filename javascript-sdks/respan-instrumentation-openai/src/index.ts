import { trace } from "@opentelemetry/api";

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
export class OpenAIInstrumentor {
  public readonly name = "openai";
  private _instrumentor: any = null;
  private _isInstrumented = false;

  async activate(): Promise<void> {
    const { OpenAIInstrumentation } = await import(
      "@traceloop/instrumentation-openai"
    );
    this._instrumentor = new OpenAIInstrumentation();
    const tp = trace.getTracerProvider();
    this._instrumentor.instrument({ tracerProvider: tp });
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (this._isInstrumented && this._instrumentor) {
      try {
        this._instrumentor.uninstrument();
      } catch {
        /* ignore */
      }
      this._isInstrumented = false;
    }
  }
}
