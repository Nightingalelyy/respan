import { ExportResultCode, type ExportResult } from "@opentelemetry/core";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  RespanLogType,
  type RespanPayload,
  RespanSpanAttributes,
  resolveTracingIngestEndpoint,
} from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

type JsonRecord = Record<string, unknown>;
type DirectRespanPayload = Omit<
  RespanPayload,
  "input" | "output" | "prompt_messages" | "completion_message" | "metadata"
> & {
  input?: unknown;
  output?: unknown;
  prompt_messages?: JsonRecord[];
  completion_message?: JsonRecord;
  metadata?: JsonRecord;
};

const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing" as const;

interface RespanHybridSpanExporterOptions {
  apiKey: string;
  baseURL: string;
  headers?: Record<string, string>;
  otlpExporter: SpanExporter;
  fetchImpl?: typeof fetch;
}

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function safeParseJson(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }

  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function parseStructuredItems(value: unknown): JsonRecord[] | undefined {
  const parsed = safeParseJson(value);

  if (Array.isArray(parsed)) {
    const items = parsed
      .map((item) => safeParseJson(item))
      .filter((item): item is JsonRecord => isRecord(item));
    return items.length > 0 ? items : undefined;
  }

  return isRecord(parsed) ? [parsed] : undefined;
}

function hrTimeToIsoString(hrTime: [number, number] | undefined): string | undefined {
  if (!Array.isArray(hrTime) || hrTime.length !== 2) {
    return undefined;
  }

  const [seconds, nanos] = hrTime;
  return new Date(seconds * 1000 + nanos / 1_000_000).toISOString();
}

function hrTimeDurationToSeconds(duration: [number, number] | undefined): number | undefined {
  if (!Array.isArray(duration) || duration.length !== 2) {
    return undefined;
  }

  return duration[0] + duration[1] / 1_000_000_000;
}

function coerceNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  return undefined;
}

function coerceBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "string") {
    if (value === "true") return true;
    if (value === "false") return false;
  }

  return undefined;
}

function coerceString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }

  const stringValue = String(value);
  return stringValue === "" ? undefined : stringValue;
}

function isPromptMessageArray(value: unknown): value is JsonRecord[] {
  return (
    Array.isArray(value) &&
    value.every((item) => isRecord(item) && typeof item.role === "string")
  );
}

function buildCompletionMessage(
  rawOutput: unknown,
  toolCalls: JsonRecord[] | undefined,
): DirectRespanPayload["completion_message"] {
  if (isRecord(rawOutput)) {
    return {
      ...rawOutput,
      role: typeof rawOutput.role === "string" ? rawOutput.role : "assistant",
      ...(toolCalls && rawOutput.tool_calls === undefined
        ? { tool_calls: toolCalls }
        : {}),
    };
  }

  if (typeof rawOutput === "string" && rawOutput !== "") {
    return {
      role: "assistant",
      content: rawOutput,
      ...(toolCalls ? { tool_calls: toolCalls } : {}),
    };
  }

  if (toolCalls && toolCalls.length > 0) {
    return {
      role: "assistant",
      content: "",
      tool_calls: toolCalls,
    };
  }

  return undefined;
}

function inferLogType(attrs: Record<string, unknown>): DirectRespanPayload["log_type"] {
  const requestType = coerceString(attrs[RespanSpanAttributes.LLM_REQUEST_TYPE]);
  if (requestType === "chat") {
    return RespanLogType.CHAT;
  }

  const rawLogType = coerceString(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE]);
  return rawLogType ?? RespanLogType.CUSTOM;
}

function deriveStatusCode(span: ReadableSpan): number {
  const explicitStatus = coerceNumber((span.attributes as Record<string, unknown>).status_code);
  if (explicitStatus !== undefined) {
    return explicitStatus;
  }

  return span.status?.code === 2 ? 500 : 200;
}

function buildMetadata(attrs: Record<string, unknown>): JsonRecord | undefined {
  const metadata: JsonRecord = {};
  const metadataPrefix = `${RespanSpanAttributes.RESPAN_METADATA}.`;

  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith(metadataPrefix)) {
      metadata[key.slice(metadataPrefix.length)] = value;
    }
  }

  const entityName = attrs[SpanAttributes.TRACELOOP_ENTITY_NAME];
  if (entityName !== undefined) {
    metadata[SpanAttributes.TRACELOOP_ENTITY_NAME] = entityName;
  }

  const genAiSystem = attrs[RespanSpanAttributes.GEN_AI_SYSTEM];
  if (genAiSystem !== undefined) {
    metadata[RespanSpanAttributes.GEN_AI_SYSTEM] = genAiSystem;
  }

  const llmSystem = attrs[RespanSpanAttributes.LLM_SYSTEM];
  if (llmSystem !== undefined) {
    metadata[RespanSpanAttributes.LLM_SYSTEM] = llmSystem;
  }

  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

export function isDirectRespanSpan(span: ReadableSpan): boolean {
  return (
    span.attributes?.[RespanSpanAttributes.RESPAN_LOG_METHOD] ===
    RESPAN_LOG_METHOD_TS_TRACING
  );
}

export function spanToRespanPayload(span: ReadableSpan): DirectRespanPayload {
  const attrs = (span.attributes ?? {}) as Record<string, unknown>;
  const logType = inferLogType(attrs);
  const tools =
    parseStructuredItems(attrs[RespanSpanAttributes.RESPAN_SPAN_TOOLS]) ??
    parseStructuredItems(attrs.tools);
  const toolCalls =
    parseStructuredItems(attrs[RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS]) ??
    parseStructuredItems(attrs.tool_calls);
  const parsedInput = safeParseJson(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]);
  const parsedOutput = safeParseJson(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]);
  const completionMessage =
    logType === RespanLogType.CHAT
      ? buildCompletionMessage(parsedOutput, toolCalls)
      : undefined;
  const outputValue = completionMessage ?? parsedOutput;

  return {
    trace_unique_id: span.spanContext().traceId,
    span_unique_id: span.spanContext().spanId,
    span_parent_id: span.parentSpanId || undefined,
    span_name: span.name,
    span_workflow_name:
      coerceString(attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME]) || undefined,
    log_type: logType,
    start_time: hrTimeToIsoString(span.startTime),
    timestamp: hrTimeToIsoString(span.endTime),
    latency: hrTimeDurationToSeconds(span.duration),
    status_code: deriveStatusCode(span),
    error_message: coerceString(span.status?.message),
    input: parsedInput,
    output: outputValue,
    prompt_messages: isPromptMessageArray(parsedInput) ? parsedInput : undefined,
    completion_message: completionMessage,
    tools,
    tool_calls: toolCalls,
    parallel_tool_calls: coerceBoolean(attrs.parallel_tool_calls),
    model:
      coerceString(attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL]) ??
      coerceString(attrs.model),
    provider_id: coerceString(attrs[RespanSpanAttributes.GEN_AI_SYSTEM]),
    prompt_tokens:
      coerceNumber(attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS]) ??
      coerceNumber(attrs.prompt_tokens),
    completion_tokens:
      coerceNumber(attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS]) ??
      coerceNumber(attrs.completion_tokens),
    total_request_tokens: coerceNumber(attrs.total_request_tokens),
    prompt_cache_hit_tokens: coerceNumber(attrs.prompt_cache_hit_tokens),
    prompt_cache_creation_tokens: coerceNumber(attrs.prompt_cache_creation_tokens),
    cost: coerceNumber(attrs.cost),
    customer_identifier: coerceString(
      attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID],
    ),
    customer_email: coerceString(
      attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL],
    ),
    customer_name: coerceString(
      attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME],
    ),
    session_identifier: coerceString(
      attrs[RespanSpanAttributes.RESPAN_SESSION_ID],
    ),
    thread_identifier: coerceString(
      attrs[RespanSpanAttributes.RESPAN_THREADS_ID],
    ),
    trace_group_identifier: coerceString(
      attrs[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID],
    ),
    custom_identifier: coerceString(
      attrs[RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID],
    ),
    environment: coerceString(attrs[RespanSpanAttributes.RESPAN_ENVIRONMENT]),
    metadata: buildMetadata(attrs),
  };
}

export class RespanHybridSpanExporter implements SpanExporter {
  private readonly otlpExporter: SpanExporter;
  private readonly directEndpoint: string;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof fetch;

  constructor({
    apiKey,
    baseURL,
    headers = {},
    otlpExporter,
    fetchImpl = fetch,
  }: RespanHybridSpanExporterOptions) {
    this.otlpExporter = otlpExporter;
    this.directEndpoint = resolveTracingIngestEndpoint(baseURL);
    this.headers = {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      ...headers,
    };
    this.fetchImpl = fetchImpl;
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    const directSpans = spans.filter((span) => isDirectRespanSpan(span));
    const otlpSpans = spans.filter((span) => !isDirectRespanSpan(span));

    const tasks: Array<Promise<void>> = [];
    let exportError: Error | undefined;

    if (directSpans.length > 0) {
      tasks.push(
        this.exportDirectSpans(directSpans).catch((error: unknown) => {
          exportError = error instanceof Error ? error : new Error(String(error));
        }),
      );
    }

    if (otlpSpans.length > 0) {
      tasks.push(
        new Promise((resolve) => {
          this.otlpExporter.export(otlpSpans, (result) => {
            if (result.code !== ExportResultCode.SUCCESS && !exportError) {
              exportError =
                result.error instanceof Error
                  ? result.error
                  : new Error(String(result.error ?? "OTLP export failed"));
            }
            resolve();
          });
        }),
      );
    }

    if (tasks.length === 0) {
      resultCallback({ code: ExportResultCode.SUCCESS });
      return;
    }

    void Promise.all(tasks)
      .then(() => {
        if (exportError) {
          resultCallback({
            code: ExportResultCode.FAILED,
            error: exportError,
          });
          return;
        }

        resultCallback({ code: ExportResultCode.SUCCESS });
      })
      .catch((error: unknown) => {
        resultCallback({
          code: ExportResultCode.FAILED,
          error: error instanceof Error ? error : new Error(String(error)),
        });
      });
  }

  async shutdown(): Promise<void> {
    await this.otlpExporter.shutdown();
  }

  async forceFlush(): Promise<void> {
    await this.otlpExporter.forceFlush?.();
  }

  private async exportDirectSpans(spans: ReadableSpan[]): Promise<void> {
    const payload = {
      data: spans.map((span) => spanToRespanPayload(span)),
    };

    const response = await this.fetchImpl(this.directEndpoint, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const responseText = await response.text().catch(() => "");
      throw new Error(
        `Respan direct export failed with ${response.status}: ${responseText}`,
      );
    }
  }
}
