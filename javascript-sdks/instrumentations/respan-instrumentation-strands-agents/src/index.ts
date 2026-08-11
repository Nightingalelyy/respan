/**
 * Respan instrumentation plugin for the Strands Agents TypeScript SDK.
 *
 * Strands already emits OpenTelemetry spans for agents, model calls, tools,
 * graph/swarm orchestration, and node execution. This plugin registers a
 * transformer so those spans use the canonical Respan span contract before
 * filtering and export.
 */

import {
  registerSpanTransformer,
  type RespanSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import {
  enrichStrandsAgentsSpan,
  StrandsAgentsSpanProcessor,
} from "./_processor.js";
import { STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN } from "./_constants.js";

export interface StrandsAgentsInstrumentorOptions {
  includeToolDefinitions?: boolean;
}

const TRANSFORMER_KEY = "@respan/instrumentation-strands-agents";
let semconvOptInOwnerCount = 0;
let originalSemconvOptIn: string | undefined;

export class StrandsAgentsInstrumentor {
  public readonly name = "strands-agents";

  private readonly _includeToolDefinitions: boolean;
  private _ownsSemconvOptIn = false;
  private _processor: StrandsAgentsSpanProcessor | null = null;
  private _transformer: RespanSpanTransformer | null = null;
  private _registration: SpanTransformerRegistration | null = null;
  private _isInstrumented = false;

  constructor(options: StrandsAgentsInstrumentorOptions = {}) {
    this._includeToolDefinitions = options.includeToolDefinitions ?? true;
  }

  activate(): void {
    if (this._isInstrumented) {
      return;
    }

    this._enableSemconvOptIns();
    try {
      if (!this._processor) {
        this._processor = new StrandsAgentsSpanProcessor();
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
    } catch (error) {
      this._restoreSemconvOptIns();
      throw error;
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) {
      return;
    }

    this._registration?.unregister();
    this._registration = null;
    this._restoreSemconvOptIns();
    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private _enableSemconvOptIns(): void {
    if (!this._includeToolDefinitions || this._ownsSemconvOptIn) {
      return;
    }
    acquireSemconvOptIn();
    this._ownsSemconvOptIn = true;
  }

  private _restoreSemconvOptIns(): void {
    if (!this._ownsSemconvOptIn) {
      return;
    }
    this._ownsSemconvOptIn = false;
    releaseSemconvOptIn();
  }
}

function acquireSemconvOptIn(): void {
  if (semconvOptInOwnerCount === 0) {
    originalSemconvOptIn = process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
    const values = new Set(
      (originalSemconvOptIn ?? "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
    values.add(STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN);
    process.env.OTEL_SEMCONV_STABILITY_OPT_IN = [...values].sort().join(",");
  }
  semconvOptInOwnerCount += 1;
}

function releaseSemconvOptIn(): void {
  if (semconvOptInOwnerCount === 0) {
    return;
  }
  semconvOptInOwnerCount -= 1;
  if (semconvOptInOwnerCount > 0) {
    return;
  }

  if (originalSemconvOptIn === undefined) {
    delete process.env.OTEL_SEMCONV_STABILITY_OPT_IN;
  } else {
    process.env.OTEL_SEMCONV_STABILITY_OPT_IN = originalSemconvOptIn;
  }
  originalSemconvOptIn = undefined;
}

export { enrichStrandsAgentsSpan, StrandsAgentsSpanProcessor };
