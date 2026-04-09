/**
 * Respan instrumentation plugin for the OpenAI Agents SDK.
 *
 * Registers a TracingProcessor that converts OpenAI Agents SDK
 * traces/spans to OTEL ReadableSpan objects and injects them into
 * the unified OTEL pipeline.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents";
 *
 * const respan = new Respan({
 *   instrumentations: [new OpenAIAgentsInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import {
  setTraceProcessors,
  type TracingProcessor,
  type Trace,
  type Span,
} from "@openai/agents";
import { emitSdkItem } from "./_otel_emitter.js";

class _RespanTracingProcessor implements TracingProcessor {
  async onTraceStart(_trace: Trace): Promise<void> {
    // no-op
  }

  async onTraceEnd(traceObj: Trace): Promise<void> {
    emitSdkItem(traceObj);
  }

  async onSpanStart(_span: Span<any>): Promise<void> {
    // no-op
  }

  async onSpanEnd(span: Span<any>): Promise<void> {
    emitSdkItem(span);
  }

  async shutdown(): Promise<void> {
    // no-op
  }

  async forceFlush(): Promise<void> {
    // no-op
  }
}

export class OpenAIAgentsInstrumentor {
  public readonly name = "openai-agents";
  private _processor: _RespanTracingProcessor | null = null;

  activate(): void {
    this._processor = new _RespanTracingProcessor();
    setTraceProcessors([this._processor]);
  }

  deactivate(): void {
    this._processor = null;
  }
}
