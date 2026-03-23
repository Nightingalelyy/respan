/**
 * Shared utilities for constructing and injecting ReadableSpan objects.
 *
 * Used by the `Respan` unified entry point (e.g. `logBatchResults`) and
 * instrumentation plugins to emit spans into the OTEL pipeline without
 * going through a live tracer context.
 */

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";

// ── ID helpers ──────────────────────────────────────────────────────────────

function hashStringToHexId(s: string, length: number): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  const hex = Math.abs(hash).toString(16).padStart(8, "0");
  return (hex + hex + hex + hex).slice(0, length);
}

function generateHexId(length: number): string {
  return Array.from({ length }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join("");
}

export function ensureTraceId(id?: string): string {
  if (!id) return generateHexId(32);
  if (/^[0-9a-f]{32}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 32);
}

export function ensureSpanId(id?: string): string {
  if (!id) return generateHexId(16);
  if (/^[0-9a-f]{16}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 16);
}

// ── Timestamp helpers ───────────────────────────────────────────────────────

export function parseISOToHrTime(
  iso: string | undefined
): [number, number] | null {
  if (!iso) return null;
  try {
    const ms = new Date(iso).getTime();
    const secs = Math.floor(ms / 1000);
    const nanos = (ms % 1000) * 1_000_000;
    return [secs, nanos];
  } catch {
    return null;
  }
}

// ── ReadableSpan builder ────────────────────────────────────────────────────

export interface BuildSpanOptions {
  name: string;
  traceId?: string;
  spanId?: string;
  parentId?: string;
  startTimeIso?: string;
  endTimeIso?: string;
  startTimeHr?: [number, number] | null;
  endTimeHr?: [number, number] | null;
  attributes: Record<string, any>;
  statusCode?: number;
  errorMessage?: string;
}

/**
 * Construct a ReadableSpan-compatible object with explicit IDs and attributes.
 */
export function buildReadableSpan(opts: BuildSpanOptions): ReadableSpan {
  const startTime =
    opts.startTimeHr ??
    parseISOToHrTime(opts.startTimeIso) ??
    hrTime();
  const endTime =
    opts.endTimeHr ??
    parseISOToHrTime(opts.endTimeIso) ??
    hrTime();

  const traceId = ensureTraceId(opts.traceId);
  const spanId = ensureSpanId(opts.spanId);
  const parentSpanId = opts.parentId
    ? ensureSpanId(opts.parentId)
    : undefined;

  const status =
    opts.errorMessage
      ? { code: SpanStatusCode.ERROR, message: opts.errorMessage }
      : opts.statusCode && opts.statusCode >= 400
        ? { code: SpanStatusCode.ERROR, message: `HTTP ${opts.statusCode}` }
        : { code: SpanStatusCode.OK, message: "" };

  return {
    name: opts.name,
    kind: SpanKind.INTERNAL,
    spanContext: () => ({
      traceId,
      spanId,
      traceFlags: 1,
      isRemote: false,
    }),
    parentSpanId,
    startTime,
    endTime,
    duration: hrTimeDuration(startTime, endTime),
    status,
    attributes: opts.attributes,
    links: [],
    events: [],
    resource: { attributes: {} } as any,
    instrumentationLibrary: {
      name: "@respan/tracing",
      version: "1.0.0",
    },
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as unknown as ReadableSpan;
}

// ── Inject into OTEL pipeline ───────────────────────────────────────────────

/**
 * Push a ReadableSpan through the active TracerProvider's processor chain.
 *
 * Returns true on success, false if no processor is available.
 */
export function injectSpan(span: ReadableSpan): boolean {
  const tp = trace.getTracerProvider() as any;
  const processor =
    tp?.activeSpanProcessor ?? tp?._delegate?.activeSpanProcessor;
  if (processor && typeof processor.onEnd === "function") {
    processor.onEnd(span);
    return true;
  }
  return false;
}
