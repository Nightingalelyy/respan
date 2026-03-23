import { RespanTelemetry, propagateAttributes, logBatchResults as _logBatchResults } from "@respan/tracing";
import type { RespanParams } from "@respan/respan-sdk";
import type { BatchRequest, BatchResult } from "@respan/tracing";
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

  // ── Context propagation ──────────────────────────────────────────────

  /**
   * Run a function within a context that attaches Respan attributes to all
   * spans created within its scope.
   *
   * Attributes are propagated via OpenTelemetry context — safe for concurrent
   * async tasks. Nested calls merge attributes (inner wins). Metadata dicts
   * are merged, not replaced.
   *
   * @param attrs - Attributes to propagate: customer_identifier, customer_email,
   *   customer_name, thread_identifier, custom_identifier, group_identifier,
   *   environment, metadata (dict), prompt (dict with prompt_id + variables).
   * @param fn - The function to execute within the propagation scope.
   * @returns The result of `fn`.
   *
   * @example
   * ```typescript
   * await respan.propagateAttributes(
   *   { customer_identifier: "user_123", thread_identifier: "conv_abc" },
   *   async () => {
   *     const result = await Runner.run(agent, "Hello");
   *     return result;
   *   }
   * );
   *
   * await respan.propagateAttributes(
   *   { prompt: { prompt_id: "abc123", variables: { x: "y" } } },
   *   async () => {
   *     const result = await Runner.run(agent, "Hello");
   *     return result;
   *   }
   * );
   * ```
   */
  propagateAttributes<T>(attrs: Partial<RespanParams>, fn: () => T): T {
    return propagateAttributes(attrs, fn);
  }

  // ── Batch API logging ──────────────────────────────────────────────────

  /**
   * Log OpenAI Batch API results as individual chat completion spans.
   *
   * Trace linking (in priority order):
   * 1. **OTEL context** — when called inside a `withTask` / `withWorkflow`,
   *    auto-links to the active trace and nests completions under the current span.
   * 2. **Explicit `traceId`** — for async batches where results arrive in a
   *    separate process (e.g. 24 hours later).
   * 3. **Auto-generated** — creates a new standalone trace.
   *
   * @param requests - Original batch request objects (from the input JSONL).
   *   Each must have `custom_id` and `body.messages`.
   * @param results - Batch result objects (from the output JSONL).
   *   Each must have `custom_id` and `response.body`.
   * @param traceId - Optional explicit trace ID to link results to.
   */
  logBatchResults(
    requests: BatchRequest[],
    results: BatchResult[],
    traceId?: string
  ): void {
    _logBatchResults(requests, results, traceId);
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
