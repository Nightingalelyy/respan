import { trace } from "@opentelemetry/api";
import type { RespanInstrumentation } from "./_types.js";

/**
 * Bridges OpenInference (Arize Phoenix) instrumentors into the Respan plugin system.
 *
 * OpenInference instrumentors may implement either:
 * - `.instrument({ tracerProvider })` — standard OTEL instrumentor interface
 * - SpanProcessor interface (added via `tracerProvider.addSpanProcessor()`)
 *
 * Usage:
 * ```typescript
 * import { GoogleADKInstrumentor } from "@arizeai/openinference-instrumentation-google-adk";
 * new OpenInferenceInstrumentor(GoogleADKInstrumentor);
 * ```
 */
export class OpenInferenceInstrumentor implements RespanInstrumentation {
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
