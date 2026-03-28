import { trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { OpenInferenceTranslator } from "./_translator.js";

export { OpenInferenceTranslator } from "./_translator.js";

const OI_SPAN_KIND_ATTR = "openinference.span.kind";
const EMPTY_SCOPE_NAME = "";
const OTEL_RESOURCE_NOISE_PREFIXES = [
  "process.",
  "host.",
  "telemetry.sdk.",
];

function filterResourceAttributes(
  attrs: Record<string, any> | undefined,
): Record<string, any> | undefined {
  if (!attrs) return attrs;

  return Object.fromEntries(
    Object.entries(attrs).filter(
      ([key]) =>
        !OTEL_RESOURCE_NOISE_PREFIXES.some((prefix) => key.startsWith(prefix)),
    ),
  );
}

function sanitizeTranslatedSpanForExport(span: ReadableSpan): ReadableSpan {
  const clonedSpan = Object.assign(
    Object.create(Object.getPrototypeOf(span)),
    span,
  ) as ReadableSpan & {
    resource?: { attributes?: Record<string, any> };
    instrumentationLibrary?: { name?: string; version?: string };
    instrumentationScope?: { name?: string; version?: string };
  };

  const resource = (span as any).resource;
  const filteredResourceAttrs = filterResourceAttributes(resource?.attributes);
  if (resource && filteredResourceAttrs) {
    clonedSpan.resource = {
      ...resource,
      attributes: filteredResourceAttrs,
    };
  }

  const instrumentationLibrary = (span as any).instrumentationLibrary;
  if (instrumentationLibrary) {
    clonedSpan.instrumentationLibrary = {
      ...instrumentationLibrary,
      name: EMPTY_SCOPE_NAME,
      version: EMPTY_SCOPE_NAME,
    };
  }

  const instrumentationScope = (span as any).instrumentationScope;
  if (instrumentationScope) {
    clonedSpan.instrumentationScope = {
      ...instrumentationScope,
      name: EMPTY_SCOPE_NAME,
      version: EMPTY_SCOPE_NAME,
    };
  }

  return clonedSpan;
}

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
 * Automatically ensures the {@link OpenInferenceTranslator} runs before the
 * active Respan processor exports the span. This keeps the translation logic
 * in the OpenInference package while still allowing the core Respan pipeline
 * to see the translated attributes.
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
  private _ownsTranslatorHook = false;

  private static _translator = new OpenInferenceTranslator();
  private static _translatorHookRefCount = 0;
  private static _patchedProcessor: any = null;
  private static _originalOnEnd: ((span: ReadableSpan) => void) | null = null;
  private static _wrappedOnEnd: ((span: ReadableSpan) => void) | null = null;

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

  private static _getActiveSpanProcessor(): any {
    const tracerProvider = trace.getTracerProvider() as any;
    return (
      tracerProvider?.activeSpanProcessor ??
      tracerProvider?._delegate?.activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?.activeSpanProcessor
    );
  }

  private static _installTranslatorHook(): void {
    const processor = OpenInferenceInstrumentor._getActiveSpanProcessor();
    if (!processor || typeof processor.onEnd !== "function") {
      return;
    }

    if (OpenInferenceInstrumentor._patchedProcessor === processor) {
      return;
    }

    if (
      OpenInferenceInstrumentor._patchedProcessor &&
      OpenInferenceInstrumentor._originalOnEnd
    ) {
      if (
        !OpenInferenceInstrumentor._wrappedOnEnd ||
        OpenInferenceInstrumentor._patchedProcessor.onEnd ===
          OpenInferenceInstrumentor._wrappedOnEnd
      ) {
        OpenInferenceInstrumentor._patchedProcessor.onEnd =
          OpenInferenceInstrumentor._originalOnEnd;
      }
    }

    const originalOnEnd = processor.onEnd.bind(processor);
    const wrappedOnEnd = (span: ReadableSpan) => {
      const isOpenInferenceSpan =
        (span as any).attributes?.[OI_SPAN_KIND_ATTR] !== undefined;

      try {
        OpenInferenceInstrumentor._translator.onEnd(span);
      } catch {
        // Never block export if translation hits an unexpected span shape.
      }

      return originalOnEnd(
        isOpenInferenceSpan ? sanitizeTranslatedSpanForExport(span) : span,
      );
    };
    processor.onEnd = wrappedOnEnd;

    OpenInferenceInstrumentor._patchedProcessor = processor;
    OpenInferenceInstrumentor._originalOnEnd = originalOnEnd;
    OpenInferenceInstrumentor._wrappedOnEnd = wrappedOnEnd;
  }

  private static _removeTranslatorHook(): void {
    if (
      OpenInferenceInstrumentor._patchedProcessor &&
      OpenInferenceInstrumentor._originalOnEnd &&
      (
        !OpenInferenceInstrumentor._wrappedOnEnd ||
        OpenInferenceInstrumentor._patchedProcessor.onEnd ===
          OpenInferenceInstrumentor._wrappedOnEnd
      )
    ) {
      OpenInferenceInstrumentor._patchedProcessor.onEnd =
        OpenInferenceInstrumentor._originalOnEnd;
    }
    OpenInferenceInstrumentor._patchedProcessor = null;
    OpenInferenceInstrumentor._originalOnEnd = null;
    OpenInferenceInstrumentor._wrappedOnEnd = null;
  }

  activate(): void {
    this._instrumentor = new this._instrumentorClass();
    const tp = trace.getTracerProvider();

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

    OpenInferenceInstrumentor._installTranslatorHook();
    if (!this._ownsTranslatorHook) {
      OpenInferenceInstrumentor._translatorHookRefCount += 1;
      this._ownsTranslatorHook = true;
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

    if (this._ownsTranslatorHook) {
      OpenInferenceInstrumentor._translatorHookRefCount = Math.max(
        0,
        OpenInferenceInstrumentor._translatorHookRefCount - 1
      );
      this._ownsTranslatorHook = false;
    }

    if (OpenInferenceInstrumentor._translatorHookRefCount === 0) {
      OpenInferenceInstrumentor._removeTranslatorHook();
    }
  }
}
