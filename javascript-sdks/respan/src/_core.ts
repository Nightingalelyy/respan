import { RespanTelemetry } from "@respan/tracing";
import type { RespanInstrumentation } from "./_types.js";

export interface RespanOptions {
  apiKey?: string;
  baseURL?: string;
  appName?: string;
  instrumentations?: RespanInstrumentation[];
  disabledInstrumentations?: string[];
  traceContent?: boolean;
  logLevel?: "debug" | "info" | "warn" | "error";
  silenceInitializationMessage?: boolean;
}

/**
 * Unified entry point for the Respan SDK.
 *
 * Creates a `RespanTelemetry` instance under the hood and activates
 * any instrumentation plugins passed in `options.instrumentations`.
 *
 * ```typescript
 * import { Respan, OTELInstrumentor } from "@respan/respan";
 * import { AnthropicInstrumentation } from "@traceloop/instrumentation-anthropic";
 *
 * const respan = new Respan({
 *   apiKey: "...",
 *   instrumentations: [new OTELInstrumentor(AnthropicInstrumentation)],
 * });
 * await respan.initialize();
 *
 * await respan.withWorkflow({ name: "my_flow" }, async () => { ... });
 * ```
 */
export class Respan {
  public readonly telemetry: RespanTelemetry;
  private _instrumentations: Map<string, RespanInstrumentation> = new Map();
  private _pendingInstrumentations: RespanInstrumentation[];

  constructor(options: RespanOptions = {}) {
    this._pendingInstrumentations = options.instrumentations ?? [];

    // Create RespanTelemetry (the OTEL engine)
    this.telemetry = new RespanTelemetry({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      appName: options.appName,
      traceContent: options.traceContent,
      logLevel: options.logLevel,
      disabledInstrumentations: options.disabledInstrumentations as any,
      silenceInitializationMessage: options.silenceInitializationMessage,
    });
  }

  /**
   * Initialize tracing and activate all instrumentation plugins.
   * Must be called (and awaited) before tracing begins.
   */
  async initialize(): Promise<void> {
    await this.telemetry.initialize();

    // Activate instrumentation plugins
    // (must happen after telemetry init so TracerProvider exists)
    for (const inst of this._pendingInstrumentations) {
      await this._activate(inst);
    }
    this._pendingInstrumentations = [];
  }

  // ── Re-exported decorator methods from telemetry ──────────────────────

  public get withWorkflow() {
    return this.telemetry.withWorkflow;
  }

  public get withTask() {
    return this.telemetry.withTask;
  }

  public get withAgent() {
    return this.telemetry.withAgent;
  }

  public get withTool() {
    return this.telemetry.withTool;
  }

  public get withRespanSpanAttributes() {
    return this.telemetry.withRespanSpanAttributes;
  }

  // ── Proxy helpers ─────────────────────────────────────────────────────

  public getClient() {
    return this.telemetry.getClient();
  }

  public getSpanBufferManager() {
    return this.telemetry.getSpanBufferManager();
  }

  public addProcessor(config: any): void {
    this.telemetry.addProcessor(config);
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────

  /**
   * Deactivate all plugins and flush telemetry.
   */
  async shutdown(): Promise<void> {
    // Deactivate plugins first
    for (const [, inst] of this._instrumentations) {
      try {
        await inst.deactivate();
      } catch {
        /* ignore */
      }
    }
    this._instrumentations.clear();

    await this.telemetry.shutdown();
  }

  // ── Private helpers ───────────────────────────────────────────────────

  private async _activate(inst: RespanInstrumentation): Promise<void> {
    if (this._instrumentations.has(inst.name)) {
      console.warn(
        `[Respan] Instrumentation "${inst.name}" is already active — skipping.`
      );
      return;
    }
    await inst.activate();
    this._instrumentations.set(inst.name, inst);
  }
}
