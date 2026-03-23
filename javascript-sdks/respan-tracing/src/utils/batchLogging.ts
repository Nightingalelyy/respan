/**
 * Batch API result logging utilities.
 *
 * Logs OpenAI Batch API results as individual chat completion spans
 * injected into the OTEL pipeline.
 */

import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, injectSpan, ensureSpanId } from "./spanFactory.js";
import { getClient } from "./client.js";

export interface BatchRequest {
  custom_id: string;
  body?: {
    messages?: any[];
    model?: string;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface BatchResult {
  custom_id: string;
  response?: {
    status_code?: number;
    body?: {
      choices?: Array<{ message?: any; [key: string]: any }>;
      usage?: { prompt_tokens?: number; completion_tokens?: number };
      model?: string;
      created?: number;
      [key: string]: any;
    };
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Log OpenAI Batch API results as individual chat completion spans.
 *
 * Trace linking (in priority order):
 * 1. OTEL context — when called inside a `withTask` / `withWorkflow`,
 *    auto-links to the active trace.
 * 2. Explicit `traceId` — for async batches where results arrive later.
 * 3. Auto-generated — creates a new standalone trace.
 */
export function logBatchResults(
  requests: BatchRequest[],
  results: BatchResult[],
  traceId?: string
): void {
  const client = getClient();

  // Resolve trace context: OTEL > explicit > auto-generated.
  let otelTraceId = client.getCurrentTraceId();
  let otelSpanId = client.getCurrentSpanId();
  if (otelTraceId && /^0+$/.test(otelTraceId)) otelTraceId = undefined;
  if (otelSpanId && /^0+$/.test(otelSpanId)) otelSpanId = undefined;

  const resolvedTraceId = otelTraceId ?? traceId ?? undefined;
  const parentSpanId = otelSpanId ?? undefined;

  // Index requests by custom_id
  const requestsById = new Map<string, Record<string, any>>();
  for (const req of requests) {
    requestsById.set(req.custom_id, req.body ?? {});
  }

  // When no OTEL context, create a grouping span so completions are nested
  let groupSpanId: string | undefined;
  if (!otelSpanId) {
    groupSpanId = ensureSpanId();
  }

  const completionTimestamps: Date[] = [];

  for (const result of results) {
    const customId = result.custom_id ?? "";
    const response = result.response ?? {};
    const body = response.body ?? {};
    const statusCode = response.status_code ?? 200;

    const original = requestsById.get(customId) ?? {};
    const messages = original.messages ?? [];

    const choices = body.choices ?? [{}];
    const output = choices[0]?.message ?? {};
    const usage = body.usage ?? {};

    // Extract timestamp
    const created: number | undefined = body.created;
    let endTimeIso: string | undefined;
    if (created) {
      const ts = new Date(created * 1000);
      endTimeIso = ts.toISOString();
      completionTimestamps.push(ts);
    }

    const model = body.model ?? original.model ?? "";

    const span = buildReadableSpan({
      name: `batch:${customId}`,
      traceId: resolvedTraceId,
      parentId: groupSpanId ?? parentSpanId,
      endTimeIso,
      attributes: {
        "llm.request.type": "chat",
        "gen_ai.request.model": model,
        "gen_ai.usage.prompt_tokens": usage.prompt_tokens ?? 0,
        "gen_ai.usage.completion_tokens": usage.completion_tokens ?? 0,
        [SpanAttributes.TRACELOOP_ENTITY_INPUT]: JSON.stringify(messages),
        [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: JSON.stringify(output),
        [SpanAttributes.TRACELOOP_ENTITY_PATH]: "batch_results",
        [SpanAttributes.TRACELOOP_SPAN_KIND]: "task",
        [RespanSpanAttributes.RESPAN_LOG_TYPE]: "chat",
      },
      statusCode,
    });
    injectSpan(span);
  }

  // Create the grouping "batch_results" task span (when no OTEL context)
  if (groupSpanId) {
    let earliestIso: string | undefined;
    let latestIso: string | undefined;
    if (completionTimestamps.length > 0) {
      completionTimestamps.sort((a, b) => a.getTime() - b.getTime());
      earliestIso = completionTimestamps[0].toISOString();
      latestIso =
        completionTimestamps[completionTimestamps.length - 1].toISOString();
    }

    const parentSpan = buildReadableSpan({
      name: "batch_results.task",
      traceId: resolvedTraceId,
      spanId: groupSpanId,
      startTimeIso: earliestIso,
      endTimeIso: latestIso,
      attributes: {
        [SpanAttributes.TRACELOOP_SPAN_KIND]: "task",
        [SpanAttributes.TRACELOOP_ENTITY_NAME]: "batch_results",
        [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
        [RespanSpanAttributes.RESPAN_LOG_TYPE]: "task",
      },
    });
    injectSpan(parentSpan);
  }
}
