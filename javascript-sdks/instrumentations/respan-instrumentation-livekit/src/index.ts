import { trace, type TracerProvider } from "@opentelemetry/api";
import * as RespanTracing from "@respan/tracing";
import type { SpanTransformerRegistration } from "@respan/tracing";
import { LIVEKIT_INSTRUMENTATION_NAME } from "./_constants.js";
import { translateLiveKitSpan } from "./_translator.js";
import { LiveKitSpanTransformer } from "./_transformer.js";

type AnyFunction = (...args: any[]) => any;
const LIVEKIT_TRANSFORMER_KEY = "@respan/instrumentation-livekit";
const SHARED_LIVEKIT_TRANSFORMER = new LiveKitSpanTransformer();

export interface LiveKitTelemetryModule {
  tracer?: {
    startSpan?: AnyFunction;
    startActiveSpan?: AnyFunction;
    startActiveSpanSync?: AnyFunction;
    setProvider?: (provider: TracerProvider) => void;
  };
  setTracerProvider?: (...args: any[]) => void;
}

export interface LiveKitAgentsModule {
  telemetry?: LiveKitTelemetryModule;
}

export interface LiveKitInstrumentorOptions {
  livekitModule?: LiveKitAgentsModule;
  telemetryModule?: LiveKitTelemetryModule;
  syncTracerProvider?: boolean;
}

export class LiveKitInstrumentor {
  public readonly name = LIVEKIT_INSTRUMENTATION_NAME;

  private readonly _livekitModule?: LiveKitAgentsModule;
  private readonly _telemetryModule?: LiveKitTelemetryModule;
  private readonly _syncTracerProvider: boolean;
  private _transformerRegistration: SpanTransformerRegistration | undefined;
  private _activationPromise: Promise<void> | undefined;
  private _activationGeneration = 0;
  private _isInstrumented = false;

  constructor(options: LiveKitInstrumentorOptions = {}) {
    this._livekitModule = options.livekitModule;
    this._telemetryModule = options.telemetryModule;
    this._syncTracerProvider = options.syncTracerProvider ?? true;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    // All callers in one activation wave share the same registry acquisition.
    // This prevents concurrent activate() calls from incrementing the shared
    // transformer's refcount twice and overwriting one registration handle.
    if (this._activationPromise) {
      return this._activationPromise;
    }

    const activationGeneration = ++this._activationGeneration;
    const activationPromise = this._activate(activationGeneration);
    this._activationPromise = activationPromise;
    try {
      await activationPromise;
    } finally {
      if (this._activationPromise === activationPromise) {
        this._activationPromise = undefined;
      }
    }
  }

  private async _activate(activationGeneration: number): Promise<void> {
    const telemetryModule = await this._resolveTelemetryModule();
    if (activationGeneration !== this._activationGeneration) {
      return;
    }
    if (!telemetryModule?.tracer) {
      return;
    }

    if (this._syncTracerProvider) {
      syncLiveKitTracerProvider(telemetryModule);
    }

    // Registration is synchronous. The generation check immediately before it
    // guarantees deactivate() cannot be overtaken by a late module resolution.
    const registration = registerLiveKitTransformer();
    if (activationGeneration !== this._activationGeneration) {
      registration.unregister();
      return;
    }

    this._transformerRegistration = registration;
    this._isInstrumented = true;
  }

  deactivate(): void {
    // Invalidate an activation that may still be awaiting the optional module.
    // Clearing the shared promise also permits a fresh activate() immediately.
    this._activationGeneration += 1;
    this._activationPromise = undefined;

    if (this._transformerRegistration) {
      this._transformerRegistration.unregister();
      this._transformerRegistration = undefined;
    }

    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private async _resolveTelemetryModule(): Promise<LiveKitTelemetryModule | undefined> {
    if (this._telemetryModule) {
      return this._telemetryModule;
    }
    if (this._livekitModule?.telemetry) {
      return this._livekitModule.telemetry;
    }

    try {
      const livekit = await import("@livekit/agents") as unknown as LiveKitAgentsModule;
      return livekit.telemetry;
    } catch {
      return undefined;
    }
  }

}

function syncLiveKitTracerProvider(telemetryModule: LiveKitTelemetryModule): void {
  const provider = resolveActiveTracerProvider();
  if (!provider) {
    return;
  }

  if (typeof telemetryModule.setTracerProvider === "function") {
    try {
      telemetryModule.setTracerProvider(provider);
      return;
    } catch {
      // Fall back to the tracer-level setter below.
    }
  }

  if (typeof telemetryModule.tracer?.setProvider === "function") {
    try {
      telemetryModule.tracer.setProvider(provider as TracerProvider);
    } catch {
      // Non-fatal: the method patch still works if LiveKit already uses the global provider.
    }
  }
}

function resolveActiveTracerProvider(): unknown {
  const provider = trace.getTracerProvider() as unknown as { _delegate?: unknown };
  return provider?._delegate ?? provider;
}

function registerLiveKitTransformer(): SpanTransformerRegistration {
  const registerSpanTransformer = (RespanTracing as Record<string, unknown>)[
    "registerSpanTransformer"
  ];
  if (typeof registerSpanTransformer !== "function") {
    throw new Error(
      "@respan/instrumentation-livekit requires a compatible @respan/tracing " +
      "runtime with registerSpanTransformer(). Upgrade @respan/tracing before activation.",
    );
  }

  return (registerSpanTransformer as typeof RespanTracing.registerSpanTransformer)(
    LIVEKIT_TRANSFORMER_KEY,
    SHARED_LIVEKIT_TRANSFORMER,
  );
}

export { LiveKitSpanTransformer, translateLiveKitSpan };
