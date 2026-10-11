import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  TOGETHER_SYSTEM_NAME,
} from "./_constants.js";
import {
  dataValue,
  protect,
  type Capture,
  type CaptureOptions,
} from "./_capture.js";
import {
  applyTokenUsage,
  formatInputMessages,
  formatTextPrompt,
  formatTools,
  safeJson,
  summarizeRequestBody,
  toSerializableValue,
  resolveStatusCode,
} from "./_helpers.js";
import type { TogetherOperationSpec } from "./_types.js";

export interface Operation {
  span: Span;
  ctx: Context;
  capture: Capture;
  spec: TogetherOperationSpec;
  request: any;
  ended: boolean;
}
const usageKeys = {
  inputTokens: ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  outputTokens: ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  promptTokens: ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  completionTokens: ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  totalTokens: SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
};
export function startOperation(
  spec: TogetherOperationSpec,
  request: any,
  options: CaptureOptions,
): Operation {
  const parent = context.active();
  const model = dataValue(request, "model");
  const span = trace
    .getTracer(INSTRUMENTATION_LIBRARY_NAME, PACKAGE_VERSION)
    .startSpan(
      spec.spanName,
      {
        kind: SpanKind.CLIENT,
        attributes: {
          [SpanAttributes.TRACELOOP_ENTITY_NAME]: spec.spanName,
          [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
          [SpanAttributes.LLM_SYSTEM]: TOGETHER_SYSTEM_NAME,
          [SpanAttributes.LLM_REQUEST_TYPE]: spec.requestType,
          [RespanSpanAttributes.RESPAN_LOG_TYPE]: spec.logType,
          ...(typeof model === "string"
            ? { [SpanAttributes.LLM_REQUEST_MODEL]: model }
            : {}),
        },
      },
      parent,
    );
  const capture = protect(span, parent, options);
  const op: Operation = {
    span,
    ctx: trace.setSpan(parent, span),
    capture,
    spec,
    request: undefined,
    ended: false,
  };
  if (capture.inputs) {
    try {
      op.request = toSerializableValue(request);
    } catch {}
  }
  return op;
}
function requestAttrs(op: Operation): Record<string, any> {
  const request = op.request;
  const attrs: Record<string, any> = {};
  if (!op.capture.inputs || !request) return attrs;
  if (op.spec.kind === "chat") {
    const messages = formatInputMessages(request.messages);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(messages);
    const tools = formatTools(request.tools);
    if (tools.length)
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  } else if (op.spec.kind === "text")
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
      formatTextPrompt(request.prompt),
    );
  else if (op.spec.kind === "embedding")
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request.input);
  else
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
      summarizeRequestBody(request),
    );
  for (const [field, key] of [
    ["max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS],
    ["temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE],
    ["top_p", SpanAttributes.LLM_REQUEST_TOP_P],
    ["frequency_penalty", SpanAttributes.LLM_FREQUENCY_PENALTY],
    ["presence_penalty", SpanAttributes.LLM_PRESENCE_PENALTY],
  ]) {
    const value = dataValue(request, field);
    if (typeof value === "number" && Number.isFinite(value)) attrs[key] = value;
  }
  return attrs;
}
export function finishOperation(
  op: Operation,
  response?: any,
  error?: unknown,
  hasError = false,
): void {
  if (op.ended) return;
  op.ended = true;
  op.capture.observe();
  try {
    const attrs = requestAttrs(op);
    if (!hasError && response !== undefined && op.span.isRecording()) {
      const responseModel = dataValue(response, "model");
      if (typeof responseModel === "string")
        attrs[ATTR_GEN_AI_RESPONSE_MODEL] = responseModel;
      const responseId = dataValue(response, "id");
      if (typeof responseId === "string")
        attrs[ATTR_GEN_AI_RESPONSE_ID] = responseId;
      const usage = dataValue(response, "usage");
      if (usage && typeof usage === "object") {
        const safeUsage: Record<string, unknown> = {};
        for (const key of [
          "prompt_tokens",
          "completion_tokens",
          "input_tokens",
          "output_tokens",
          "total_tokens",
        ]) {
          const value = dataValue(usage, key);
          if (typeof value === "number" && Number.isFinite(value))
            safeUsage[key] = value;
        }
        applyTokenUsage(attrs, safeUsage, usageKeys);
        const details = dataValue(usage, "prompt_tokens_details"),
          completionDetails = dataValue(usage, "completion_tokens_details");
        const cached = dataValue(details, "cached_tokens"),
          reasoning = dataValue(completionDetails, "reasoning_tokens");
        if (typeof cached === "number" && Number.isFinite(cached))
          attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cached;
        if (typeof reasoning === "number" && Number.isFinite(reasoning))
          attrs[ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = reasoning;
      }
      if (op.capture.outputs) {
        if (op.spec.kind === "chat") {
          const outputs = Array.isArray(response?.choices)
            ? response.choices.map((choice: any) => {
                const message = choice.message ?? {};
                return {
                  ...toSerializableValue(message),
                  role: message.role ?? "assistant",
                };
              })
            : [];
          attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(outputs);
          const first = outputs[0];
          if (first) {
            attrs[`${ATTR_GEN_AI_COMPLETION}.0.role`] = first.role;
            if (first.content !== undefined)
              attrs[`${ATTR_GEN_AI_COMPLETION}.0.content`] =
                typeof first.content === "string"
                  ? first.content
                  : safeJson(first.content);
            if (first.tool_calls?.length)
              attrs[`${ATTR_GEN_AI_COMPLETION}.0.tool_calls`] = safeJson(
                first.tool_calls,
              );
          }
        } else if (op.spec.kind === "text")
          attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
            (response?.choices ?? []).map((choice: any) => ({
              role: "assistant",
              content: choice.text ?? "",
            })),
          );
        else if (op.spec.kind === "embedding")
          attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
            response?.data ?? response,
          );
        else
          attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
            toSerializableValue(response),
          );
      }
    }
    // Canonical bodies, schemas and all usage precede optional indexed messages.
    op.span.setAttributes(attrs);
    if (!hasError && op.capture.outputs && Array.isArray(response?.choices)) {
      const reasons = response.choices
        .map((choice: any) => choice.finish_reason)
        .filter((value: any) => typeof value === "string");
      if (reasons.length)
        op.span.setAttribute(ATTR_GEN_AI_RESPONSE_FINISH_REASONS, reasons);
    }
    if (op.capture.inputs && op.spec.kind === "chat")
      formatInputMessages(op.request?.messages).forEach((message, index) => {
        op.span.setAttribute(
          `${ATTR_GEN_AI_PROMPT}.${index}.role`,
          message.role,
        );
        if (message.content !== undefined)
          op.span.setAttribute(
            `${ATTR_GEN_AI_PROMPT}.${index}.content`,
            typeof message.content === "string"
              ? message.content
              : safeJson(message.content),
          );
        if (message.tool_calls?.length)
          op.span.setAttribute(
            `${ATTR_GEN_AI_PROMPT}.${index}.tool_calls`,
            safeJson(message.tool_calls),
          );
      });
    if (hasError) {
      op.span.setStatus({
        code: SpanStatusCode.ERROR,
        ...(op.capture.inputs && op.capture.outputs
          ? { message: error instanceof Error ? error.message : String(error) }
          : {}),
      });
      if (op.capture.inputs && op.capture.outputs)
        op.span.recordException(
          error instanceof Error ? error : new Error(String(error)),
        );
      const status =
        dataValue(error, "status") ?? dataValue(error, "statusCode");
      if (typeof status === "number" && Number.isFinite(status))
        op.span.setAttribute("status_code", status);
    }
  } catch {
  } finally {
    op.span.end();
    op.request = undefined;
  }
}
