import type { ExportResult } from "@opentelemetry/core";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  SpanAttributes,
  TraceloopSpanKindValues,
} from "@traceloop/ai-semantic-conventions";
import {
  RespanLogType,
  RespanSpanAttributes,
  type RespanSpanNameStyle,
} from "@respan/respan-sdk";

type SpanAttrs = Record<string, unknown>;

const INTERNAL_KIND_ATTR = RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND;
const INTERNAL_DETAIL_ATTR = RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL;

const INTERNAL_SPAN_NAME_ATTRS = [
  INTERNAL_KIND_ATTR,
  INTERNAL_DETAIL_ATTR,
] as const;

const SUFFIXED_OPERATIONS = new Set(["agent", "tool", "handoff"]);

function formatOperationName(operation: string): string {
  return operation.toUpperCase();
}

export function resolveSpanNameStyle(
  value?: RespanSpanNameStyle | string
): RespanSpanNameStyle {
  return value === "semantic" ? "semantic" : "legacy";
}

export function transformReadableSpanName(
  span: ReadableSpan,
  style: RespanSpanNameStyle | string | undefined
): ReadableSpan {
  const resolvedStyle = resolveSpanNameStyle(style);
  const attributes = stripInternalSemanticNameAttrs(span.attributes as SpanAttrs);
  const name =
    resolvedStyle === "semantic" ? semanticSpanNameForSpan(span) : span.name;

  if (name === span.name && attributes === span.attributes) {
    return span;
  }

  return cloneReadableSpan(span, name, attributes);
}

export function semanticSpanNameForSpan(span: ReadableSpan): string {
  const attrs = span.attributes as SpanAttrs;
  const operation = resolveOperation(attrs, span.name);
  const detail = resolveDetail(attrs, span.name, operation);

  const hasInternalHint =
    attrs[INTERNAL_KIND_ATTR] !== undefined || attrs[INTERNAL_DETAIL_ATTR] !== undefined;

  const prefix = formatOperationName(operation);
  if (!SUFFIXED_OPERATIONS.has(operation)) {
    return prefix;
  }

  if (!hasInternalHint && span.name.startsWith(`${prefix}.`)) {
    return span.name;
  }

  return `${prefix}.${detail}`;
}

export class SpanNameTransformingExporter implements SpanExporter {
  constructor(
    private readonly delegate: SpanExporter,
    private readonly style: RespanSpanNameStyle
  ) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this.delegate.export(
      spans.map((span) => transformReadableSpanName(span, this.style)),
      resultCallback
    );
  }

  shutdown(): Promise<void> {
    return this.delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    const maybeFlush = (this.delegate as { forceFlush?: () => Promise<void> })
      .forceFlush;
    return maybeFlush ? maybeFlush.call(this.delegate) : Promise.resolve();
  }
}

function stripInternalSemanticNameAttrs(attrs: SpanAttrs): SpanAttrs {
  let next: SpanAttrs | undefined;

  for (const key of INTERNAL_SPAN_NAME_ATTRS) {
    if (attrs[key] !== undefined) {
      next ??= { ...attrs };
      delete next[key];
    }
  }

  return next ?? attrs;
}

function cloneReadableSpan(
  span: ReadableSpan,
  name: string,
  attributes: SpanAttrs
): ReadableSpan {
  const clone = Object.create(Object.getPrototypeOf(span));
  Object.assign(clone, span);
  Object.defineProperty(clone, "name", {
    value: name,
    enumerable: true,
    configurable: true,
  });
  Object.defineProperty(clone, "attributes", {
    value: attributes,
    enumerable: true,
    configurable: true,
  });
  return clone as ReadableSpan;
}

function resolveOperation(attrs: SpanAttrs, spanName: string): string {
  const hintedKind = stringAttr(attrs, INTERNAL_KIND_ATTR);
  if (hintedKind) {
    return sanitizeNamePart(mapOperation(hintedKind), "span");
  }

  const tlKind = stringAttr(attrs, SpanAttributes.TRACELOOP_SPAN_KIND);
  if (tlKind) {
    return sanitizeNamePart(mapOperation(tlKind), "span");
  }

  const logType = stringAttr(attrs, RespanSpanAttributes.RESPAN_LOG_TYPE);
  if (logType) {
    return sanitizeNamePart(mapOperation(logType), "span");
  }

  return sanitizeNamePart(inferOperationFromName(spanName), "span");
}

function resolveDetail(
  attrs: SpanAttrs,
  spanName: string,
  operation: string
): string {
  const hintedDetail = stringAttr(attrs, INTERNAL_DETAIL_ATTR);
  if (hintedDetail) {
    return sanitizeNamePart(hintedDetail, "operation");
  }

  const entityName = stringAttr(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME);
  if (entityName) {
    return sanitizeNamePart(entityName, "operation");
  }

  return sanitizeNamePart(detailFromRawName(spanName, operation), "operation");
}

function stringAttr(attrs: SpanAttrs, key: string): string | undefined {
  const value = attrs[key];
  if (value === undefined || value === null) return undefined;
  return String(value);
}

function mapOperation(value: string): string {
  const normalized = value.toLowerCase();

  switch (normalized) {
    case TraceloopSpanKindValues.WORKFLOW:
    case RespanLogType.WORKFLOW:
      return "workflow";
    case TraceloopSpanKindValues.AGENT:
    case RespanLogType.AGENT:
      return "agent";
    case TraceloopSpanKindValues.TASK:
    case RespanLogType.TASK:
      return "task";
    case TraceloopSpanKindValues.TOOL:
    case RespanLogType.TOOL:
      return "tool";
    case RespanLogType.FUNCTION:
      return "function";
    case RespanLogType.HANDOFF:
      return "handoff";
    case RespanLogType.GUARDRAIL:
      return "guardrail";
    case RespanLogType.EMBEDDING:
    case "embedding":
      return "embed";
    case RespanLogType.TRANSCRIPTION:
      return "transcribe";
    case RespanLogType.SPEECH:
      return "speech";
    case RespanLogType.CHAT:
    case RespanLogType.TEXT:
    case RespanLogType.RESPONSE:
    case RespanLogType.GENERATION:
    case "generate":
    case "llm":
      return "llm";
    case RespanLogType.CUSTOM:
    case RespanLogType.UNKNOWN:
      return "span";
    default:
      return value;
  }
}

function inferOperationFromName(spanName: string): string {
  if (spanName.startsWith("ai.stream")) return "llm";
  if (spanName.startsWith("ai.embed")) return "embed";
  if (spanName.startsWith("ai.generate") || spanName.startsWith("ai.")) {
    return "llm";
  }

  const suffix = spanName.split(".").at(-1);
  if (suffix) {
    return mapOperation(suffix);
  }

  return "span";
}

function detailFromRawName(spanName: string, operation: string): string {
  if (spanName.startsWith("ai.")) {
    return spanName.split(".").at(-1) ?? spanName;
  }

  if (spanName.endsWith(`.${operation}`)) {
    return spanName.slice(0, -(operation.length + 1));
  }

  if (spanName.startsWith(`${operation}.`)) {
    return spanName.slice(operation.length + 1);
  }

  if (operation === "llm") {
    const parts = spanName.split(".").filter(Boolean);
    const suffix = parts.at(-1)?.toLowerCase();
    if (["chat", "generation", "completion", "response", "text", "llm"].includes(suffix ?? "")) {
      return parts.slice(0, -1).join(".") || spanName;
    }
    return parts.at(-1) ?? spanName;
  }

  if (operation === "handoff") {
    return spanName.replace(/^handoff\s*[:.-]?\s*/i, "");
  }

  return spanName;
}

function sanitizeNamePart(value: string, fallback: string): string {
  const sanitized = value
    .trim()
    .replace(/\s*(?:→|->)\s*/g, "_")
    .replace(/[^\w.-]+/g, "_")
    .replace(/^[_\-.]+|[_\-.]+$/g, "");

  return sanitized || fallback;
}
