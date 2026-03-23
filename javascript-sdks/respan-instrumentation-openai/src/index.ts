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
    const { trace } = await import("@opentelemetry/api");
    const { OpenAIInstrumentation } = await import(
      "@traceloop/instrumentation-openai"
    );
    this._instrumentor = new OpenAIInstrumentation();

    // Point the instrumentor at the global TracerProvider (set by RespanTelemetry)
    this._instrumentor.setTracerProvider(trace.getTracerProvider());

    // manuallyInstrument patches the OpenAI module's prototypes directly
    const OpenAI = (await import("openai")).default;
    this._instrumentor.manuallyInstrument(OpenAI);
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (this._isInstrumented && this._instrumentor) {
      try {
        this._instrumentor.unpatch();
      } catch {
        /* ignore */
      }
      this._isInstrumented = false;
    }
  }
}
