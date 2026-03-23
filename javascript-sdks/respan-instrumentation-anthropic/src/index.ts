import { trace } from "@opentelemetry/api";

/**
 * Respan instrumentation plugin for the Anthropic SDK.
 *
 * Wraps `@traceloop/instrumentation-anthropic` in the Respan plugin protocol.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { AnthropicInstrumentor } from "@respan/instrumentation-anthropic";
 *
 * const respan = new Respan({
 *   instrumentations: [new AnthropicInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */
export class AnthropicInstrumentor {
  public readonly name = "anthropic";
  private _instrumentor: any = null;
  private _isInstrumented = false;

  async activate(): Promise<void> {
    const { AnthropicInstrumentation } = await import(
      "@traceloop/instrumentation-anthropic"
    );
    this._instrumentor = new AnthropicInstrumentation();
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
