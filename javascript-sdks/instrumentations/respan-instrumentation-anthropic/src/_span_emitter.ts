import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ANTHROPIC_CHAT_ENTITY_NAME,
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  extractToolCalls,
  extractToolExecutions,
  formatInputMessages,
  formatOutputMessage,
  formatTools,
  resolveErrorStatusCode,
  safeJson,
  stringifyStructured,
  type ToolExecution,
} from "./_helpers.js";

const COMPLETION_ZERO = `${ATTR_GEN_AI_COMPLETION}.0`;
const COMPLETION_TOOL_CALLS = `${COMPLETION_ZERO}.tool_calls`;
const LLM_IS_STREAMING =
  (SpanAttributes as unknown as Record<string, string>).LLM_IS_STREAMING ??
  "llm.is_streaming";
const STATUS_CODE_ATTR = "status_code";

function buildInstrumentedReadableSpan(opts: {
  name: string;
  startTime: [number, number];
  endTime: [number, number];
  attributes: Record<string, any>;
  statusCode?: number;
  errorMessage?: string;
}): ReadableSpan {
  const activeSpanContext = trace.getSpan(context.active())?.spanContext();
  const span = buildReadableSpan({
    name: opts.name,
    traceId: activeSpanContext?.traceId,
    parentId: activeSpanContext?.spanId,
    startTimeHr: opts.startTime,
    endTimeHr: opts.endTime,
    attributes: opts.attributes,
    statusCode: opts.statusCode,
    errorMessage: opts.errorMessage,
  }) as ReadableSpan & {
    instrumentationScope?: { name: string; version?: string };
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };

  const originalSpanContext = span.spanContext.bind(span);
  const mutableSpan = span as typeof span & {
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };
  mutableSpan.spanContext = () => ({
    ...originalSpanContext(),
    traceFlags: activeSpanContext?.traceFlags ?? TraceFlags.SAMPLED,
  });
  mutableSpan.instrumentationScope = {
    name: INSTRUMENTATION_LIBRARY_NAME,
    version: PACKAGE_VERSION,
  };
  return mutableSpan;
}

function setPromptAttrs(attrs: Record<string, any>, messages: Record<string, any>[]): void {
  messages.forEach((message, index) => {
    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    attrs[`${prefix}.role`] = message.role;
    attrs[`${prefix}.content`] = stringifyStructured(message.content ?? "");
    if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      attrs[`${prefix}.tool_calls`] = safeJson(message.tool_calls);
    }
    if (message.tool_call_id) {
      attrs[`${prefix}.tool_call_id`] = String(message.tool_call_id);
    }
  });
}

function buildBaseChatAttrs(kwargs: Record<string, any>, model?: string): Record<string, any> {
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: ANTHROPIC_CHAT_ENTITY_NAME,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: ANTHROPIC_CHAT_ENTITY_NAME,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_SYSTEM]: "anthropic",
    [LLM_IS_STREAMING]: kwargs.stream === true,
  };

  const resolvedModel = model ?? kwargs.model;
  if (resolvedModel) {
    attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = resolvedModel;
  }

  const inputMessages = formatInputMessages(kwargs.messages ?? [], kwargs.system);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMessages);
  setPromptAttrs(attrs, inputMessages);

  const tools = formatTools(kwargs.tools);
  if (tools) {
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  }

  return attrs;
}

function buildSuccessAttrs(kwargs: Record<string, any>, message: any): Record<string, any> {
  const attrs = buildBaseChatAttrs(kwargs, message?.model ?? kwargs.model);
  const outputMessage = formatOutputMessage(message);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([outputMessage]);
  attrs[`${COMPLETION_ZERO}.role`] = outputMessage.role ?? "assistant";
  attrs[`${COMPLETION_ZERO}.content`] = stringifyStructured(
    outputMessage.content ?? "",
  );

  if (message?.usage) {
    const inputTokens = message.usage.input_tokens ?? 0;
    const outputTokens = message.usage.output_tokens ?? 0;
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = outputTokens;
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = inputTokens + outputTokens;
  }

  const toolCalls = extractToolCalls(message);
  if (toolCalls) {
    attrs[COMPLETION_TOOL_CALLS] = safeJson(toolCalls);
  }

  return attrs;
}

function buildErrorAttrs(kwargs: Record<string, any>): Record<string, any> {
  return buildBaseChatAttrs(kwargs);
}

function emitSpan(
  name: string,
  attrs: Record<string, any>,
  startTime: [number, number],
  errorMessage?: string,
  statusCode?: number,
): void {
  try {
    const span = buildInstrumentedReadableSpan({
      name,
      startTime,
      endTime: hrTime(),
      attributes: attrs,
      statusCode,
      errorMessage,
    });
    injectSpan(span);
  } catch {
    // Never break the application.
  }
}

export function emitSuccessSpan(
  kwargs: Record<string, any>,
  startTime: [number, number],
  message: any,
): void {
  try {
    emitSpan(
      ANTHROPIC_CHAT_ENTITY_NAME,
      buildSuccessAttrs(kwargs, message),
      startTime,
    );
  } catch {
    // Never break the application.
  }
}

export function emitErrorSpan(
  kwargs: Record<string, any>,
  startTime: [number, number],
  err: unknown,
): void {
  try {
    const errorMessage = String(err);
    const statusCode = resolveErrorStatusCode(err);
    const attrs = buildErrorAttrs(kwargs);
    attrs["error.message"] = errorMessage;
    attrs[STATUS_CODE_ATTR] = statusCode;
    emitSpan(ANTHROPIC_CHAT_ENTITY_NAME, attrs, startTime, errorMessage, statusCode);
  } catch {
    // Never break the application.
  }
}

export function emitToolSpan(toolExecution: ToolExecution): void {
  const startTime = hrTime();
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: toolExecution.name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: toolExecution.name,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJson([
      { role: "tool", content: stringifyStructured(toolExecution.input) },
    ]),
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: safeJson({
      role: "tool",
      content: stringifyStructured(toolExecution.output),
    }),
  };

  if (toolExecution.id) {
    attrs.tool_call_id = toolExecution.id;
  }
  if (toolExecution.isError) {
    attrs["error.message"] = stringifyStructured(toolExecution.output);
    attrs[STATUS_CODE_ATTR] = 500;
  }

  emitSpan(
    `${toolExecution.name}.tool`,
    attrs,
    startTime,
    toolExecution.isError ? stringifyStructured(toolExecution.output) : undefined,
    toolExecution.isError ? 500 : undefined,
  );
}

export function emitToolSpansFromMessages(messages: any[] | undefined): void {
  for (const toolExecution of extractToolExecutions(messages)) {
    try {
      emitToolSpan(toolExecution);
    } catch {
      // Never break the application.
    }
  }
}
