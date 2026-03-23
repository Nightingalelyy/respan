import { trace } from "@opentelemetry/api";
import type { RespanInstrumentation } from "./_types.js";

/**
 * Bridges Traceloop / OpenLLMetry OTEL instrumentors into the Respan plugin system.
 *
 * Usage:
 * ```typescript
 * import { AnthropicInstrumentation } from "@traceloop/instrumentation-anthropic";
 * new OTELInstrumentor(AnthropicInstrumentation);
 * ```
 */
export class OTELInstrumentor implements RespanInstrumentation {
  public readonly name: string;
  private _instrumentorClass: any;
  private _instrumentor: any = null;
  private _isInstrumented = false;

  constructor(instrumentorClass: any) {
    this._instrumentorClass = instrumentorClass;
    this.name = `otel-${instrumentorClass.name || "unknown"}`;
  }

  activate(): void {
    this._instrumentor = new this._instrumentorClass();
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
