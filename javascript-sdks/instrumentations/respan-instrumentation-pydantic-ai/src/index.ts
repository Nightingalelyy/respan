import {
  registerSpanTransformer,
  type RespanSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import {
  PydanticAISpanProcessor,
  type PydanticAISpanProcessorOptions,
  enrichPydanticAISpan,
  isPydanticAISpan,
  isPydanticAIOpenInferenceSpan,
  isPydanticAINativeSpan,
} from "./_processor.js";

export interface PydanticAIInstrumentorOptions
  extends PydanticAISpanProcessorOptions {}

const TRANSFORMER_KEY = "@respan/instrumentation-pydantic-ai";

/**
 * Respan instrumentation for Pydantic AI-compatible TypeScript spans.
 *
 * No local TypeScript Pydantic AI SDK surface is documented in this repo, so
 * this instrumentor deliberately does not patch unknown SDK APIs. It registers
 * a transformer that normalizes Pydantic AI native OTel spans and Pydantic
 * AI-scoped OpenInference spans before Respan filters and exports them.
 */
export class PydanticAIInstrumentor {
  public readonly name = "pydantic-ai";

  private readonly _options: Required<PydanticAISpanProcessorOptions>;
  private _processor: PydanticAISpanProcessor | null = null;
  private _transformer: RespanSpanTransformer | null = null;
  private _registration: SpanTransformerRegistration | null = null;
  private _isInstrumented = false;

  constructor(options: PydanticAIInstrumentorOptions = {}) {
    this._options = {
      includeNativeSpans: options.includeNativeSpans ?? true,
      includeOpenInferenceSpans: options.includeOpenInferenceSpans ?? true,
    };
  }

  activate(): void {
    if (this._isInstrumented) {
      return;
    }

    if (!this._processor) {
      this._processor = new PydanticAISpanProcessor(this._options);
    }
    const processor = this._processor;
    if (!this._transformer) {
      this._transformer = {
        onStart: (span, parentContext) =>
          processor.onStart(span, parentContext),
        onEnd: (span) => processor.onEnd(span),
        dispose: () => {
          void processor.shutdown();
        },
      };
    }
    this._registration = registerSpanTransformer(
      TRANSFORMER_KEY,
      this._transformer,
    );
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    this._registration?.unregister();
    this._registration = null;
    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }
}

export {
  PydanticAISpanProcessor,
  enrichPydanticAISpan,
  isPydanticAISpan,
  isPydanticAIOpenInferenceSpan,
  isPydanticAINativeSpan,
};
