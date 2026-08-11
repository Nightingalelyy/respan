/**
 * Respan instrumentation plugin for the Vercel AI SDK.
 *
 * Registers a {@link VercelAITranslator} with the Respan span-transformer
 * registry so Vercel AI SDK OTEL spans are enriched before filtering/export.
 *
 * AI SDK 4-6 emit OTEL spans when experimental_telemetry is enabled. AI SDK 7
 * emits them through @ai-sdk/otel, which this instrumentor registers when that
 * optional peer is installed. Both formats are translated into the canonical
 * Respan span contract.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { VercelAIInstrumentor } from "@respan/instrumentation-vercel";
 *
 * const respan = new Respan({
 *   instrumentations: [new VercelAIInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import * as RespanTracing from "@respan/tracing";
import type { SpanTransformerRegistration } from "@respan/tracing";
import {
  ensureAISDKTelemetry,
  releaseOwnedAISDKTelemetry,
  type OwnedAISDKTelemetryLease,
} from "./_ai_sdk_telemetry.js";
import { VercelAITranslator } from "./_translator.js";

export { VercelAITranslator } from "./_translator.js";
export {
  VERCEL_SPAN_CONFIG,
  VERCEL_PARENT_SPANS,
  VERCEL_STRUCTURAL_LLM_PARENT_SPANS,
} from "./constants/index.js";

export interface VercelAIInstrumentorOptions {
  /**
   * Automatically register @ai-sdk/otel when AI SDK 7 is detected.
   * Defaults to true. Set false when telemetry registration is managed by the app.
   */
  autoRegisterAISDKTelemetry?: boolean;
}

const VERCEL_TRANSFORMER_KEY = "@respan/instrumentation-vercel";
const SHARED_VERCEL_TRANSLATOR = new VercelAITranslator();

export class VercelAIInstrumentor {
  public readonly name = "vercel-ai";

  private readonly _autoRegisterAISDKTelemetry: boolean;
  private _aiSDKTelemetryLease: OwnedAISDKTelemetryLease | undefined;
  private _transformerRegistration: SpanTransformerRegistration | undefined;
  private _activationGeneration = 0;
  private _active = false;

  constructor(options: VercelAIInstrumentorOptions = {}) {
    this._autoRegisterAISDKTelemetry = options.autoRegisterAISDKTelemetry ?? true;
  }

  protected _ensureAISDKTelemetry(): ReturnType<typeof ensureAISDKTelemetry> {
    return ensureAISDKTelemetry();
  }

  async activate(): Promise<void> {
    if (this._active) {
      return;
    }

    this._active = true;
    const activationGeneration = ++this._activationGeneration;

    try {
      this._transformerRegistration = registerVercelTransformer();

      if (this._autoRegisterAISDKTelemetry && !this._aiSDKTelemetryLease) {
        const registration = await this._ensureAISDKTelemetry();

        // If deactivate() (or a newer activate()) ran while optional modules were
        // loading, immediately return this acquisition. Otherwise a late
        // registration could survive after the instrumentor was deactivated.
        if (activationGeneration !== this._activationGeneration) {
          if (registration.lease) {
            releaseOwnedAISDKTelemetry(registration.lease);
          }
          return;
        }

        this._aiSDKTelemetryLease = registration.lease;
      }
    } catch (error) {
      // A failed registry or optional-adapter activation must leave this
      // instance retryable and return every lease acquired by this attempt.
      if (activationGeneration === this._activationGeneration) {
        this._active = false;
        this._activationGeneration += 1;
        if (this._transformerRegistration) {
          this._transformerRegistration.unregister();
          this._transformerRegistration = undefined;
        }
        if (this._aiSDKTelemetryLease) {
          releaseOwnedAISDKTelemetry(this._aiSDKTelemetryLease);
          this._aiSDKTelemetryLease = undefined;
        }
      }
      throw error;
    }
  }
  deactivate(): void {
    this._active = false;
    this._activationGeneration += 1;

    if (this._transformerRegistration) {
      this._transformerRegistration.unregister();
      this._transformerRegistration = undefined;
    }

    if (this._aiSDKTelemetryLease) {
      releaseOwnedAISDKTelemetry(this._aiSDKTelemetryLease);
      this._aiSDKTelemetryLease = undefined;
    }
  }

  isActive(): boolean {
    return this._active;
  }
}

function registerVercelTransformer(): SpanTransformerRegistration {
  const registerSpanTransformer = (RespanTracing as Record<string, unknown>)[
    "registerSpanTransformer"
  ];
  if (typeof registerSpanTransformer !== "function") {
    throw new Error(
      "@respan/instrumentation-vercel requires a compatible @respan/tracing " +
      "runtime with registerSpanTransformer(). Upgrade @respan/tracing before activation.",
    );
  }

  return (registerSpanTransformer as typeof RespanTracing.registerSpanTransformer)(
    VERCEL_TRANSFORMER_KEY,
    SHARED_VERCEL_TRANSLATOR,
  );
}
