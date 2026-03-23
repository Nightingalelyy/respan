/**
 * Emit OpenAI Agents SDK spans as OTEL ReadableSpan objects.
 *
 * Each per-type emitter converts an OpenAI Agents SDK Trace/Span into a
 * ReadableSpan with traceloop.* and gen_ai.* attributes, then injects it
 * into the OTEL pipeline via the active TracerProvider's span processor.
 */

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanSpanAttributes, RespanLogType } from "@respan/respan-sdk";
import type { Trace, Span } from "@openai/agents";

// ── Helpers ────────────────────────────────────────────────────────────────

function safeJson(obj: any): string {
  try {
    return JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    );
  } catch {
    return String(obj);
  }
}

function parseISOToHrTime(iso: string | undefined): [number, number] | null {
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

function generateHexId(length: number): string {
  return Array.from({ length }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join("");
}

function hashStringToHexId(s: string, length: number): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  // Pad the hash into a hex string of the target length
  const hex = Math.abs(hash).toString(16).padStart(8, "0");
  // Repeat/slice to desired length
  return (hex + hex + hex + hex).slice(0, length);
}

function ensureTraceId(id: string): string {
  // OTEL trace IDs are 32 hex chars
  if (/^[0-9a-f]{32}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 32);
}

function ensureSpanId(id: string): string {
  // OTEL span IDs are 16 hex chars
  if (/^[0-9a-f]{16}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 16);
}

// ── Base attribute builder ─────────────────────────────────────────────────

function baseAttrs(
  spanKind: string,
  entityName: string,
  entityPath: string,
  logType: string,
): Record<string, any> {
  return {
    [SpanAttributes.TRACELOOP_SPAN_KIND]: spanKind,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

// ── ReadableSpan builder ───────────────────────────────────────────────────

interface BuildSpanOptions {
  name: string;
  traceId: string;
  spanId: string;
  parentId?: string;
  startTimeHr?: [number, number] | null;
  endTimeHr?: [number, number] | null;
  attributes: Record<string, any>;
  statusCode?: number;
  errorMessage?: string;
}

function buildReadableSpan(opts: BuildSpanOptions): ReadableSpan {
  const startTime = opts.startTimeHr ?? hrTime();
  const endTime = opts.endTimeHr ?? hrTime();

  const traceId = ensureTraceId(opts.traceId);
  const spanId = ensureSpanId(opts.spanId);
  const parentSpanId = opts.parentId ? ensureSpanId(opts.parentId) : undefined;

  const status =
    opts.statusCode && opts.statusCode >= 400
      ? { code: SpanStatusCode.ERROR, message: opts.errorMessage ?? "" }
      : { code: SpanStatusCode.OK, message: "" };

  // Build a ReadableSpan-compatible object
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
      name: "@respan/instrumentation-openai-agents",
      version: "1.0.0",
    },
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as unknown as ReadableSpan;
}

// ── Inject into OTEL pipeline ──────────────────────────────────────────────

function injectSpan(span: ReadableSpan): void {
  const tp = trace.getTracerProvider() as any;
  const processor =
    tp?.activeSpanProcessor ?? tp?._delegate?.activeSpanProcessor;
  if (processor && typeof processor.onEnd === "function") {
    processor.onEnd(span);
  }
}

// ── Input/output formatting helpers ────────────────────────────────────────

function formatInputMessages(input: any): any[] | null {
  if (!input) return null;
  if (Array.isArray(input)) {
    return input.map((item: any) => {
      if (typeof item === "object" && item !== null && item.role) {
        return item;
      }
      return { role: "user", content: String(item) };
    });
  }
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }
  return null;
}

function formatOutput(output: any): any {
  if (!output) return null;
  if (Array.isArray(output)) {
    const messages: any[] = [];
    for (const item of output) {
      if (typeof item !== "object" || item === null) {
        messages.push({ role: "assistant", content: String(item) });
        continue;
      }
      const itemType = (item as any).type;
      if (itemType === "message" && (item as any).role === "assistant") {
        const content = Array.isArray((item as any).content)
          ? (item as any).content
              .map((c: any) =>
                typeof c === "object" && c !== null
                  ? c.text ?? String(c)
                  : String(c),
              )
              .join(" ")
          : String((item as any).content);
        messages.push({ role: "assistant", content });
      } else {
        messages.push(item);
      }
    }
    return messages;
  }
  return output;
}

// ── Per-type emitters ──────────────────────────────────────────────────────

function isTrace(item: any): item is Trace {
  return "traceId" in item && "name" in item && !("spanId" in item);
}

function isSpan(item: any): item is Span<any> {
  return "spanId" in item && "spanData" in item;
}

function emitTrace(traceObj: Trace): void {
  const attrs = baseAttrs(
    RespanLogType.WORKFLOW,
    traceObj.name || "trace",
    "",
    RespanLogType.WORKFLOW,
  );
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = traceObj.name || "trace";

  const span = buildReadableSpan({
    name: `${traceObj.name}.workflow`,
    traceId: traceObj.traceId,
    spanId: traceObj.traceId, // root span uses trace_id as span_id
    attributes: attrs,
  });
  injectSpan(span);
}

function emitAgent(item: Span<any>): void {
  const data = item.spanData as any;
  const name = data.name || "agent";
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const attrs = baseAttrs(RespanLogType.AGENT, name, name, RespanLogType.AGENT);
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = name;
  attrs["respan.metadata.agent_name"] = name;
  if (data.tools) attrs["respan.span.tools"] = safeJson(data.tools);
  if (data.handoffs) attrs["respan.span.handoffs"] = safeJson(data.handoffs);

  const span = buildReadableSpan({
    name: `${name}.agent`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitResponse(item: Span<any>): void {
  const data = item.spanData as any;
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const attrs = baseAttrs(RespanLogType.TASK, "response", "response", RespanLogType.RESPONSE);
  attrs[RespanSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[RespanSpanAttributes.LLM_SYSTEM] = "openai";

  // Input
  const inputMsgs = formatInputMessages(data._input);
  if (inputMsgs) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);
  }

  // Response data
  const resp = data._response;
  if (resp) {
    if (resp.model) attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = resp.model;

    if (resp.output) {
      const output = formatOutput(resp.output);
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);
    }

    if (resp.usage) {
      attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = resp.usage.input_tokens ?? 0;
      attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = resp.usage.output_tokens ?? 0;
    }
  }

  const span = buildReadableSpan({
    name: "openai.chat",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitFunction(item: Span<any>): void {
  const data = item.spanData as any;
  const name = data.name || "function";
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const attrs = baseAttrs(RespanLogType.TOOL, name, name, RespanLogType.TOOL);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson([
    { role: "tool", content: String(data.input ?? "") },
  ]);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
    role: "tool",
    content: String(data.output ?? ""),
  });

  const span = buildReadableSpan({
    name: `${name}.tool`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitGeneration(item: Span<any>): void {
  const data = item.spanData as any;
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const attrs = baseAttrs(RespanLogType.TASK, "generation", "generation", RespanLogType.GENERATION);
  attrs[RespanSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;

  if (data.model) attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = data.model;

  const inputMsgs = formatInputMessages(data.input);
  if (inputMsgs) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);
  }

  const output = formatOutput(data.output);
  if (output) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);
  }

  if (data.usage) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] =
      data.usage.prompt_tokens ?? data.usage.input_tokens ?? 0;
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] =
      data.usage.completion_tokens ?? data.usage.output_tokens ?? 0;
  }

  const span = buildReadableSpan({
    name: "openai.chat",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitHandoff(item: Span<any>): void {
  const data = item.spanData as any;
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const fromAgent = data.from_agent || "";
  const toAgent = data.to_agent || "";

  const attrs = baseAttrs(RespanLogType.TASK, "handoff", "handoff", RespanLogType.HANDOFF);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(fromAgent);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(toAgent);
  attrs["respan.metadata.from_agent"] = fromAgent;
  attrs["respan.metadata.to_agent"] = toAgent;

  const span = buildReadableSpan({
    name: "handoff.task",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitGuardrail(item: Span<any>): void {
  const data = item.spanData as any;
  const name = `guardrail:${data.name}`;
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const attrs = baseAttrs(RespanLogType.TASK, name, name, RespanLogType.GUARDRAIL);
  attrs["respan.metadata.guardrail_name"] = data.name;
  attrs["respan.metadata.triggered"] = String(data.triggered);

  const span = buildReadableSpan({
    name: `${name}.task`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitCustom(item: Span<any>): void {
  const data = item.spanData as any;
  const name = data.name || "custom";
  const json = JSON.parse(JSON.stringify(item));
  const startHr = parseISOToHrTime(json.started_at);
  const endHr = parseISOToHrTime(json.ended_at);

  const attrs = baseAttrs(RespanLogType.TASK, name, name, RespanLogType.CUSTOM);
  const customData = data.data || {};
  for (const [k, v] of Object.entries(customData)) {
    if (k === "model") attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = v;
    else if (k === "prompt_tokens") attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = v;
    else if (k === "completion_tokens")
      attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = v;
    else if (k === "input")
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(v);
    else if (k === "output")
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(v);
    else attrs[`respan.metadata.${k}`] = String(v);
  }

  const span = buildReadableSpan({
    name: `${name}.task`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr: startHr,
    endTimeHr: endHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

// ── Dispatcher ─────────────────────────────────────────────────────────────

export function emitSdkItem(item: Trace | Span<any>): void {
  if (isTrace(item)) {
    emitTrace(item);
    return;
  }

  if (!isSpan(item)) return;

  const spanData = item.spanData as any;
  const type = spanData?.type;

  try {
    if (type === "response") emitResponse(item);
    else if (type === "function") emitFunction(item);
    else if (type === "generation") emitGeneration(item);
    else if (type === "agent") emitAgent(item);
    else if (type === "handoff") emitHandoff(item);
    else if (typeof spanData?.triggered === "boolean") emitGuardrail(item);
    else if (spanData?.name && spanData?.data) emitCustom(item);
    else {
      console.debug(
        `[Respan] Unknown OpenAI Agents span data type: ${type}`,
      );
    }
  } catch (error) {
    console.error(`[Respan] Error emitting OpenAI Agents span:`, error);
  }
}
