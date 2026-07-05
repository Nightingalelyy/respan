import { trace } from "@opentelemetry/api";
import type { Context } from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
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

type ProcessorProperty = "activeSpanProcessor" | "_activeSpanProcessor";

interface ProcessorPatchState {
  originalProcessor: SpanProcessor;
  processorProperty: ProcessorProperty;
  provider: Record<string, unknown>;
  wrapper: PydanticAIProcessorWrapper;
}

const WRAPPED_PROCESSOR = Symbol.for(
  "respan.instrumentation.pydanticAI.wrappedProcessor",
);

class PydanticAIProcessorWrapper implements SpanProcessor {
  readonly [WRAPPED_PROCESSOR] = true;

  constructor(
    private readonly _delegate: SpanProcessor,
    private readonly _translator: SpanProcessor,
  ) {}

  onStart(span: Span, parentContext: Context): void {
    this._translator.onStart(span, parentContext);
    this._delegate.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this._translator.onEnd(span);
    this._delegate.onEnd(span);
  }

  shutdown(): Promise<void> {
    return this._delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    return this._delegate.forceFlush();
  }
}

/**
 * Respan instrumentation for Pydantic AI-compatible TypeScript spans.
 *
 * No local TypeScript Pydantic AI SDK surface is documented in this repo, so
 * this instrumentor deliberately does not patch unknown SDK APIs. It installs
 * a processor that normalizes Pydantic AI native OTel spans and Pydantic
 * AI-scoped OpenInference spans before Respan exports them.
 */
export class PydanticAIInstrumentor {
  public readonly name = "pydantic-ai";

  private static _patchCount = 0;
  private static _patchState: ProcessorPatchState | null = null;

  private readonly _options: Required<PydanticAISpanProcessorOptions>;
  private _processor: PydanticAISpanProcessor | null = null;
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
    this._installProcessor(this._processor);
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    this._restoreProcessor();
    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private _installProcessor(processor: PydanticAISpanProcessor): void {
    const provider = resolveWritableTracerProvider();
    const { activeProcessor, processorProperty } =
      resolveActiveSpanProcessor(provider);

    if ((activeProcessor as unknown as Record<symbol, unknown>)[WRAPPED_PROCESSOR]) {
      PydanticAIInstrumentor._patchCount += 1;
      return;
    }

    const wrapper = new PydanticAIProcessorWrapper(activeProcessor, processor);
    setActiveSpanProcessor(provider, processorProperty, wrapper);

    PydanticAIInstrumentor._patchState = {
      originalProcessor: activeProcessor,
      processorProperty,
      provider,
      wrapper,
    };
    PydanticAIInstrumentor._patchCount = 1;
  }

  private _restoreProcessor(): void {
    if (PydanticAIInstrumentor._patchCount === 0) {
      return;
    }

    PydanticAIInstrumentor._patchCount -= 1;
    if (PydanticAIInstrumentor._patchCount > 0) {
      return;
    }

    const patchState = PydanticAIInstrumentor._patchState;
    if (!patchState) {
      return;
    }

    if (patchState.provider[patchState.processorProperty] === patchState.wrapper) {
      setActiveSpanProcessor(
        patchState.provider,
        patchState.processorProperty,
        patchState.originalProcessor,
      );
    }
    PydanticAIInstrumentor._patchState = null;
  }
}

function resolveWritableTracerProvider(): Record<string, unknown> {
  const provider = trace.getTracerProvider() as unknown as Record<string, unknown>;
  const delegated = provider?._delegate as Record<string, unknown> | undefined;
  return delegated ?? provider;
}

function resolveActiveSpanProcessor(provider: Record<string, unknown>): {
  activeProcessor: SpanProcessor;
  processorProperty: ProcessorProperty;
} {
  for (const property of [
    "activeSpanProcessor",
    "_activeSpanProcessor",
  ] as const) {
    const candidate = provider[property];
    if (isSpanProcessor(candidate)) {
      return { activeProcessor: candidate, processorProperty: property };
    }
  }
  throw new Error(
    "PydanticAIInstrumentor requires an active OpenTelemetry SpanProcessor. Initialize Respan before activating this instrumentor.",
  );
}

function setActiveSpanProcessor(
  provider: Record<string, unknown>,
  property: ProcessorProperty,
  processor: SpanProcessor,
): void {
  try {
    provider[property] = processor;
    if (provider[property] === processor) {
      return;
    }
  } catch {
    // Fall through to defineProperty.
  }

  Object.defineProperty(provider, property, {
    configurable: true,
    value: processor,
    writable: true,
  });
}

function isSpanProcessor(value: unknown): value is SpanProcessor {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof (value as SpanProcessor).onStart === "function" &&
      typeof (value as SpanProcessor).onEnd === "function",
  );
}

export {
  PydanticAISpanProcessor,
  enrichPydanticAISpan,
  isPydanticAISpan,
  isPydanticAIOpenInferenceSpan,
  isPydanticAINativeSpan,
};
