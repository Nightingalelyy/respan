/**
 * Respan instrumentation plugin for BeeAI Framework.
 *
 * Wraps `@arizeai/openinference-instrumentation-beeai` in the Respan plugin
 * protocol and normalizes BeeAI event spans for Respan routing.
 *
 * ```typescript
 * import * as beeaiFramework from "beeai-framework";
 * import { BeeAIInstrumentation as OpenInferenceBeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
 * import { Respan } from "@respan/respan";
 * import { BeeAIInstrumentor } from "@respan/instrumentation-beeai";
 *
 * const respan = new Respan({
 *   instrumentations: [new BeeAIInstrumentor({
 *     sdkModule: beeaiFramework,
 *     instrumentationClass: OpenInferenceBeeAIInstrumentation,
 *   })],
 * });
 * await respan.initialize();
 * ```
 */

import { trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { BeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
import {
  INPUT_VALUE as OPENINFERENCE_INPUT_VALUE,
  LLM_MODEL_NAME,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_TOTAL,
  METADATA as OPENINFERENCE_METADATA,
  OUTPUT_VALUE as OPENINFERENCE_OUTPUT_VALUE,
} from "@arizeai/openinference-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

export type BeeAIInstrumentationClass = new (...args: any[]) => any;
type ProcessorOnStart = (span: ReadableSpan, parentContext: unknown) => void;
type ProcessorOnEnd = (span: ReadableSpan) => void;

const BEEAI_SCOPE_NAME = "@arizeai/openinference-instrumentation-beeai";
const BEEAI_TARGET = "target";
const BEEAI_DATA = "data";
const BEEAI_TRACE_ID = "traceId";
const BEEAI_VERSION = "beeai.version";
const OTEL_SCOPE_NAME = "otel.scope.name";
const MAX_PENDING_CHAT_INPUTS = 20;
const MAX_PENDING_CHAT_SPANS_PER_TRACE = 64;
const MAX_DROPPED_SPAN_PARENTS_PER_TRACE = 256;
const MAX_TRACKED_TRACES = 256;
const TRACE_STATE_TTL_MS = 5 * 60 * 1_000;
const OFF_CONTRACT_ALIASES = new Set([
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "tools",
  "tool_calls",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
]);

interface PendingChatSpan {
  span: ReadableSpan;
  exportSpan: ProcessorOnEnd;
}

const droppedSpanParentsByTrace = new Map<string, Map<string, string | undefined>>();
const workflowSpanIdsByTrace = new Map<string, string>();
const pendingChatInputsByTrace = new Map<string, unknown[]>();
const pendingChatSpansByTrace = new Map<string, PendingChatSpan[]>();
const traceStateTouchedAt = new Map<string, number>();

function clearTraceState(traceId: string, flushPending = true): void {
  const pendingSpans = pendingChatSpansByTrace.get(traceId);

  droppedSpanParentsByTrace.delete(traceId);
  workflowSpanIdsByTrace.delete(traceId);
  pendingChatInputsByTrace.delete(traceId);
  pendingChatSpansByTrace.delete(traceId);
  traceStateTouchedAt.delete(traceId);

  if (!flushPending || !pendingSpans) return;
  for (const { span, exportSpan } of pendingSpans) {
    try {
      exportSpan(span);
    } catch {
      // State eviction must not block other spans from exporting.
    }
  }
}

function exportPendingChatSpan({ span, exportSpan }: PendingChatSpan): void {
  try {
    exportSpan(span);
  } catch {
    // State compaction must not block other spans from exporting.
  }
}

function pruneTraceState(now = Date.now()): void {
  for (const [traceId, touchedAt] of Array.from(traceStateTouchedAt.entries())) {
    if (now - touchedAt > TRACE_STATE_TTL_MS) {
      clearTraceState(traceId);
    }
  }

  while (traceStateTouchedAt.size > MAX_TRACKED_TRACES) {
    const oldestTraceId = traceStateTouchedAt.keys().next().value as string | undefined;
    if (!oldestTraceId) break;
    clearTraceState(oldestTraceId);
  }
}

function touchTraceState(traceId: string): void {
  traceStateTouchedAt.delete(traceId);
  traceStateTouchedAt.set(traceId, Date.now());
  pruneTraceState();
}

function setDefault(attrs: Record<string, any>, key: string, value: any): void {
  if (attrs[key] === undefined) attrs[key] = value;
}

function safeJsonStr(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function parseJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function asRecord(value: unknown): Record<string, any> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as Record<string, any>;
}

function isMeaningfulStructuredValue(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.length > 0;
  if (Array.isArray(value)) return value.some((item) => isMeaningfulStructuredValue(item));
  if (typeof value === "object") {
    return Object.values(value as Record<string, unknown>).some((item) =>
      isMeaningfulStructuredValue(item),
    );
  }
  return true;
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  for (const value of values) {
    if (value !== undefined) return value;
  }
  return undefined;
}

function getInstrumentationScopeName(span: ReadableSpan): string {
  return (
    ((span as any).instrumentationScope?.name as string | undefined) ??
    ((span as any).instrumentationLibrary?.name as string | undefined) ??
    ((span as any).attributes?.[OTEL_SCOPE_NAME] as string | undefined) ??
    ""
  );
}

function isBeeAIChatStartTarget(target: unknown): target is string {
  return (
    typeof target === "string" &&
    target.startsWith("backend.") &&
    target.endsWith(".chat.start")
  );
}

function getBeeAIEventLogType(target: unknown): string | undefined {
  if (typeof target !== "string") return undefined;

  if (target.startsWith("agent.")) return RespanLogType.AGENT;
  if (target.startsWith("tool.")) return RespanLogType.TOOL;

  if (target.startsWith("backend.") && target.includes(".chat.")) {
    return RespanLogType.CHAT;
  }
  if (target.startsWith("backend.") && target.includes(".embedding.")) {
    return RespanLogType.EMBEDDING;
  }

  return undefined;
}

function inferProviderFromTarget(target: unknown): string | undefined {
  if (typeof target !== "string") return undefined;
  const match = /^backend\.([^.]+)\.(?:chat|embedding)\./.exec(target);
  return match?.[1]?.toLowerCase();
}

function setTokenAttributes(
  attrs: Record<string, any>,
  usage: Record<string, any> | undefined,
): void {
  const promptTokens = firstDefined(
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS],
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS],
    usage?.promptTokens,
    usage?.prompt_tokens,
    usage?.inputTokens,
    usage?.input_tokens,
    attrs[LLM_TOKEN_COUNT_PROMPT],
  );
  const completionTokens = firstDefined(
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS],
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS],
    usage?.completionTokens,
    usage?.completion_tokens,
    usage?.outputTokens,
    usage?.output_tokens,
    attrs[LLM_TOKEN_COUNT_COMPLETION],
  );
  const totalTokens = firstDefined(
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS],
    usage?.totalTokens,
    usage?.total_tokens,
    attrs[LLM_TOKEN_COUNT_TOTAL],
    promptTokens !== undefined && completionTokens !== undefined
      ? Number(promptTokens) + Number(completionTokens)
      : undefined,
  );

  if (promptTokens !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_USAGE_PROMPT_TOKENS, promptTokens);
    setDefault(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, promptTokens);
  }
  if (completionTokens !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, completionTokens);
    setDefault(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, completionTokens);
  }
  if (totalTokens !== undefined) {
    setDefault(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, totalTokens);
  }
}

function getSpanTraceKey(span: ReadableSpan, attrs: Record<string, any>): string | undefined {
  const spanContext = typeof (span as any).spanContext === "function"
    ? (span as any).spanContext()
    : undefined;
  // BeeAI emits related lifecycle events as separate OTEL spans, so its
  // framework trace id is the correlation key when available.
  const traceId = firstDefined(attrs[BEEAI_TRACE_ID], spanContext?.traceId);
  return typeof traceId === "string" && traceId.length > 0 ? traceId : undefined;
}

function getOtelSpanContext(span: ReadableSpan): { traceId?: string; spanId?: string } | undefined {
  return typeof (span as any).spanContext === "function"
    ? (span as any).spanContext()
    : undefined;
}

function getOtelTraceId(span: ReadableSpan): string | undefined {
  const traceId = getOtelSpanContext(span)?.traceId;
  return typeof traceId === "string" && traceId.length > 0 ? traceId : undefined;
}

function getOtelSpanId(span: ReadableSpan): string | undefined {
  const spanId = getOtelSpanContext(span)?.spanId;
  return typeof spanId === "string" && spanId.length > 0 ? spanId : undefined;
}

function getOtelParentSpanId(span: ReadableSpan): string | undefined {
  const parentSpanId =
    (span as any).parentSpanId ?? (span as any).parentSpanContext?.spanId;
  return typeof parentSpanId === "string" && parentSpanId.length > 0
    ? parentSpanId
    : undefined;
}

function rememberDroppedSpanParent(span: ReadableSpan): void {
  const traceId = getOtelTraceId(span);
  const spanId = getOtelSpanId(span);
  if (!traceId || !spanId) return;

  const traceParents = droppedSpanParentsByTrace.get(traceId) ?? new Map();
  traceParents.delete(spanId);
  traceParents.set(spanId, getOtelParentSpanId(span));
  while (traceParents.size > MAX_DROPPED_SPAN_PARENTS_PER_TRACE) {
    const oldestSpanId = traceParents.keys().next().value as string | undefined;
    if (!oldestSpanId) break;
    traceParents.delete(oldestSpanId);
  }
  droppedSpanParentsByTrace.set(traceId, traceParents);
  touchTraceState(traceId);
}

function isWorkflowSpan(span: ReadableSpan, attrs: Record<string, any>): boolean {
  const spanKind = attrs[SpanAttributes.TRACELOOP_SPAN_KIND];
  return (
    (typeof spanKind === "string" && spanKind.toLowerCase() === "workflow") ||
    span.name.endsWith(".workflow.workflow")
  );
}

function rememberWorkflowSpan(span: ReadableSpan, attrs: Record<string, any>): void {
  if (!isWorkflowSpan(span, attrs)) return;

  const traceId = getOtelTraceId(span);
  const spanId = getOtelSpanId(span);
  if (!traceId || !spanId) return;

  workflowSpanIdsByTrace.set(traceId, spanId);
  touchTraceState(traceId);
}

function resolveExportParentSpanId(span: ReadableSpan): string | undefined {
  const traceId = getOtelTraceId(span);
  let parentSpanId = getOtelParentSpanId(span);
  if (!traceId || !parentSpanId) return parentSpanId;

  const traceParents = droppedSpanParentsByTrace.get(traceId);
  if (!traceParents) return parentSpanId;
  touchTraceState(traceId);

  const visited = new Set<string>();
  while (parentSpanId && traceParents.has(parentSpanId) && !visited.has(parentSpanId)) {
    visited.add(parentSpanId);
    parentSpanId = traceParents.get(parentSpanId);
  }

  return parentSpanId;
}

function reparentFromDroppedSpans(span: ReadableSpan): void {
  const currentParentSpanId = getOtelParentSpanId(span);
  const traceId = getOtelTraceId(span);
  const workflowSpanId = traceId ? workflowSpanIdsByTrace.get(traceId) : undefined;
  const resolvedParentSpanId = workflowSpanId ?? resolveExportParentSpanId(span);
  if (resolvedParentSpanId === currentParentSpanId) return;

  Object.defineProperty(span, "parentSpanId", {
    value: resolvedParentSpanId,
    writable: false,
    configurable: true,
    enumerable: true,
  });
  // OTEL 2.x reads parentSpanContext on the wire, not parentSpanId.
  Object.defineProperty(span, "parentSpanContext", {
    value: resolvedParentSpanId
      ? {
          traceId,
          spanId: resolvedParentSpanId,
          traceFlags: span.spanContext().traceFlags,
          isRemote: false,
        }
      : undefined,
    writable: false,
    configurable: true,
    enumerable: true,
  });
}

function enqueuePendingChatInput(
  span: ReadableSpan,
  attrs: Record<string, any>,
  input: unknown,
): void {
  if (input === undefined) return;

  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return;

  const queue = pendingChatInputsByTrace.get(traceKey) ?? [];
  const serializedInput = safeJsonStr(input);
  if (queue.length > 0 && safeJsonStr(queue[queue.length - 1]) === serializedInput) {
    return;
  }

  queue.push(input);
  if (queue.length > MAX_PENDING_CHAT_INPUTS) {
    queue.shift();
  }
  pendingChatInputsByTrace.set(traceKey, queue);
  touchTraceState(traceKey);
}

function dequeuePendingChatInput(
  span: ReadableSpan,
  attrs: Record<string, any>,
): unknown {
  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return undefined;

  const queue = pendingChatInputsByTrace.get(traceKey);
  if (!queue || queue.length === 0) return undefined;
  touchTraceState(traceKey);

  const input = queue.shift();
  if (queue.length === 0) {
    pendingChatInputsByTrace.delete(traceKey);
  }
  return input;
}

function getStateMessages(state: Record<string, any> | undefined): unknown[] | undefined {
  const memory = asRecord(state?.memory);
  return Array.isArray(memory?.messages) ? memory.messages : undefined;
}

function normalizeToolCall(block: Record<string, any>): Record<string, unknown> {
  return {
    id: block.toolCallId,
    type: "function",
    function: {
      name: block.toolName,
      arguments: safeJsonStr(block.args ?? {}),
    },
  };
}

function normalizeToolResult(block: Record<string, any>): Record<string, unknown> {
  return {
    tool_call_id: block.toolCallId,
    name: block.toolName,
    content: block.result,
    is_error: Boolean(block.isError),
  };
}

function normalizeBeeAIMessage(message: unknown): unknown {
  const record = asRecord(message);
  if (!record) return message;

  const normalized: Record<string, unknown> = {};
  if (typeof record.role === "string") {
    normalized.role = record.role;
  }

  if (typeof record.content === "string") {
    normalized.content = record.content;
    return normalized;
  }

  const content = Array.isArray(record.content) ? record.content : undefined;
  if (!content) return normalized.role ? normalized : message;

  const textParts: string[] = [];
  const toolCalls: Record<string, unknown>[] = [];
  const toolResults: Record<string, unknown>[] = [];

  for (const blockValue of content) {
    const block = asRecord(blockValue);
    if (!block) continue;

    if (block.type === "text" && block.text !== undefined) {
      textParts.push(String(block.text));
    } else if (block.type === "tool-call") {
      toolCalls.push(normalizeToolCall(block));
    } else if (block.type === "tool-result") {
      toolResults.push(normalizeToolResult(block));
    }
  }

  if (textParts.length > 0) {
    normalized.content = textParts.join("\n");
  }
  if (toolCalls.length > 0) {
    if (normalized.content === undefined) {
      normalized.content = "";
    }
    normalized.tool_calls = toolCalls;
  }
  if (toolResults.length === 1 && normalized.role === "tool") {
    Object.assign(normalized, toolResults[0]);
  } else if (toolResults.length > 0) {
    normalized.tool_results = toolResults;
  }

  return normalized;
}

function normalizeMessages(messages: unknown[] | undefined): unknown[] | undefined {
  return messages?.map((message) => normalizeBeeAIMessage(message));
}

function getMessageContentValue(message: unknown): unknown {
  const record = asRecord(message);
  const content = record?.content;
  if (!Array.isArray(content) || content.length !== 1) return normalizeBeeAIMessage(message);

  const block = asRecord(content[0]);
  if (!block) return normalizeBeeAIMessage(message);
  if (block.type === "text" && block.text !== undefined) return block.text;
  if (block.type === "tool-result" && block.result !== undefined) {
    return block.isError === undefined
      ? block.result
      : { result: block.result, is_error: Boolean(block.isError) };
  }
  if (block.type === "tool-call") {
    return { tool_call: normalizeToolCall(block) };
  }

  return normalizeBeeAIMessage(message);
}

function getAgentStateInput(state: Record<string, any> | undefined): unknown {
  if (!state) return undefined;
  const messages = getStateMessages(state);
  if (messages) {
    const hasOutput = getStateResultValue(state) !== undefined || getLastStateMessageValue(state) !== undefined;
    const inputMessages = hasOutput && messages.length > 0 ? messages.slice(0, -1) : messages;
    return {
      iteration: state.iteration,
      messages: normalizeMessages(inputMessages),
    };
  }
  return state;
}

function getChatInputFromState(state: Record<string, any> | undefined): unknown {
  const messages = getStateMessages(state);
  return messages ? normalizeMessages(messages) : undefined;
}

function sameToolCalls(left: unknown, right: unknown): boolean {
  if (left === undefined || right === undefined) return false;
  return safeJsonStr(left) === safeJsonStr(right);
}

function matchesAssistantOutput(message: unknown, output: unknown): boolean {
  const messageRecord = asRecord(message);
  const outputRecord = asRecord(output);
  if (!messageRecord || !outputRecord || messageRecord.role !== "assistant") return false;

  if (sameToolCalls(messageRecord.tool_calls, outputRecord.tool_calls)) {
    return true;
  }

  return (
    typeof messageRecord.content === "string" &&
    typeof outputRecord.content === "string" &&
    messageRecord.content === outputRecord.content
  );
}

function getPendingChatInputFromState(
  state: Record<string, any> | undefined,
  pendingSpan: ReadableSpan,
): unknown {
  const messages = normalizeMessages(getStateMessages(state));
  if (!messages || messages.length === 0) return undefined;

  const pendingAttrs = (pendingSpan as any).attributes as Record<string, any> | undefined;
  const output = parseJson(pendingAttrs?.[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]);

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (matchesAssistantOutput(messages[index], output)) {
      return index > 0 ? messages.slice(0, index) : undefined;
    }
  }

  const lastMessage = asRecord(messages[messages.length - 1]);
  if (lastMessage?.role === "assistant") {
    return messages.length > 1 ? messages.slice(0, -1) : undefined;
  }

  return messages;
}

function setChatInputAttributes(span: ReadableSpan, input: unknown): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs || input === undefined) return;

  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonStr(input);
  setChatPromptAttributes(attrs, input, true);
}

function queuePendingChatSpan(
  span: ReadableSpan,
  attrs: Record<string, any>,
  exportSpan: ProcessorOnEnd,
  traceKey = getSpanTraceKey(span, attrs),
): void {
  if (!traceKey) return;

  const queue = pendingChatSpansByTrace.get(traceKey) ?? [];
  if (!queue.some((entry) => entry.span === span)) {
    queue.push({ span, exportSpan });
    while (queue.length > MAX_PENDING_CHAT_SPANS_PER_TRACE) {
      const oldestPendingSpan = queue.shift();
      if (oldestPendingSpan) exportPendingChatSpan(oldestPendingSpan);
    }
    pendingChatSpansByTrace.set(traceKey, queue);
    touchTraceState(traceKey);
  }
}

function flushPendingChatSpansFromState(
  span: ReadableSpan,
  attrs: Record<string, any>,
  state: Record<string, any> | undefined,
): void {
  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return;

  const queue = pendingChatSpansByTrace.get(traceKey);
  if (!queue || queue.length === 0) return;
  touchTraceState(traceKey);

  const remaining: PendingChatSpan[] = [];
  for (const entry of queue) {
    const pendingSpan = entry.span;
    const pendingAttrs = (pendingSpan as any).attributes as Record<string, any> | undefined;
    const input = getPendingChatInputFromState(state, pendingSpan);
    if (!pendingAttrs || input === undefined) {
      remaining.push(entry);
      continue;
    }

    setChatInputAttributes(pendingSpan, input);
    entry.exportSpan(pendingSpan);
  }

  if (remaining.length > 0) {
    pendingChatSpansByTrace.set(traceKey, remaining);
  } else {
    pendingChatSpansByTrace.delete(traceKey);
  }
}

function flushAllPendingChatSpans(): void {
  for (const queue of pendingChatSpansByTrace.values()) {
    for (const { span, exportSpan } of queue) {
      try {
        exportSpan(span);
      } catch {
        // Best-effort flush during deactivation.
      }
    }
  }
  pendingChatSpansByTrace.clear();
}

function shouldDelayMissingChatInputSpan(span: ReadableSpan): boolean {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return false;

  return (
    attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] === RespanLogType.CHAT &&
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] === undefined &&
    typeof attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] === "string" &&
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME].endsWith(".success")
  );
}

function getLastStateMessageValue(state: Record<string, any> | undefined): unknown {
  const messages = getStateMessages(state);
  if (!messages || messages.length === 0) return undefined;
  return getMessageContentValue(messages[messages.length - 1]);
}

function getStateResultValue(state: Record<string, any> | undefined): unknown {
  if (!state || state.result === undefined) return undefined;
  return getMessageContentValue(state.result);
}

function getMatchingToolResult(
  state: Record<string, any> | undefined,
  toolCallMsg: Record<string, any> | undefined,
): unknown {
  const toolCallId = toolCallMsg?.toolCallId;
  if (!toolCallId) return undefined;

  const messages = getStateMessages(state);
  if (!messages) return undefined;

  for (const message of messages) {
    const content = asRecord(message)?.content;
    if (!Array.isArray(content)) continue;

    for (const blockValue of content) {
      const block = asRecord(blockValue);
      if (
        block?.type === "tool-result" &&
        block.toolCallId === toolCallId &&
        block.result !== undefined
      ) {
        return block.isError === undefined
          ? block.result
          : { result: block.result, is_error: Boolean(block.isError) };
      }
    }
  }

  return undefined;
}

function isFinalAnswerTool(toolCallMsg: Record<string, any> | undefined): boolean {
  const name = toolCallMsg?.toolName;
  return typeof name === "string" && name.toLowerCase() === "final_answer";
}

function normalizeInputValue(input: unknown): unknown {
  if (Array.isArray(input)) {
    return normalizeMessages(input);
  }

  const record = asRecord(input);
  if (record && Array.isArray(record.messages)) {
    return normalizeMessages(record.messages);
  }
  return input;
}

function normalizeMeaningfulInputValue(input: unknown): unknown {
  const normalized = normalizeInputValue(input);
  return isMeaningfulStructuredValue(normalized) ? normalized : undefined;
}

function normalizeOutputValue(output: unknown): unknown {
  if (Array.isArray(output)) {
    const messages = normalizeMessages(output);
    return messages && messages.length === 1 ? messages[0] : messages;
  }

  const record = asRecord(output);
  if (record && Array.isArray(record.messages)) {
    const messages = normalizeMessages(record.messages);
    return messages && messages.length === 1 ? messages[0] : messages;
  }
  if (record && record.content !== undefined && record.role !== undefined) {
    return normalizeBeeAIMessage(record);
  }

  return output;
}

function clearChatPromptAttributes(attrs: Record<string, any>): void {
  for (const key of Object.keys(attrs)) {
    if (key.startsWith(`${ATTR_GEN_AI_PROMPT}.`)) {
      delete attrs[key];
    }
  }
}

function setChatPromptAttributes(
  attrs: Record<string, any>,
  input: unknown,
  overwrite = false,
): void {
  const messages = Array.isArray(input)
    ? input
    : asRecord(input)?.role !== undefined
      ? [input]
      : undefined;
  if (!messages) return;

  if (overwrite) {
    clearChatPromptAttributes(attrs);
  }

  const setPromptAttribute = (key: string, value: unknown) => {
    if (overwrite) {
      attrs[key] = value;
    } else {
      setDefault(attrs, key, value);
    }
  };

  for (const [index, message] of messages.entries()) {
    const record = asRecord(message);
    if (!record) continue;

    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    if (typeof record.role === "string") {
      setPromptAttribute(`${prefix}.role`, record.role);
    }
    if (typeof record.content === "string") {
      setPromptAttribute(`${prefix}.content`, record.content);
    }
    if (typeof record.name === "string") {
      setPromptAttribute(`${prefix}.name`, record.name);
    }
    if (typeof record.tool_call_id === "string") {
      setPromptAttribute(`${prefix}.tool_call_id`, record.tool_call_id);
    }
    if (record.tool_calls !== undefined) {
      setPromptAttribute(`${prefix}.tool_calls`, safeJsonStr(record.tool_calls));
    }
  }
}

function setChatCompletionAttributes(
  attrs: Record<string, any>,
  output: unknown,
): void {
  const outputRecord = asRecord(output);
  const messages = Array.isArray(output)
    ? output
    : outputRecord?.role === "assistant"
      ? [output]
      : undefined;
  const assistantMessage = messages
    ?.map((message) => asRecord(message))
    .find((message) => message?.role === "assistant");
  if (!assistantMessage) return;

  const completionPrefix = `${ATTR_GEN_AI_COMPLETION}.0`;

  setDefault(attrs, `${completionPrefix}.role`, "assistant");
  setDefault(
    attrs,
    `${completionPrefix}.content`,
    typeof assistantMessage.content === "string" ? assistantMessage.content : "",
  );

  if (assistantMessage.tool_calls !== undefined) {
    setDefault(
      attrs,
      `${completionPrefix}.tool_calls`,
      safeJsonStr(assistantMessage.tool_calls),
    );
  }
}

function markSpanDropped(attrs: Record<string, any>): void {
  attrs[RespanSpanAttributes.RESPAN_PROCESSORS] = [];
}

function dropSpan(span: ReadableSpan, attrs: Record<string, any>): void {
  rememberDroppedSpanParent(span);
  markSpanDropped(attrs);
}

function shouldDropTarget(target: string): boolean {
  return target.endsWith(".start") || target.endsWith(".finish");
}

function isBeeAIFrameworkParentSpan(span: ReadableSpan, attrs: Record<string, any>): boolean {
  return span.name === "beeai-framework-main" && (
    attrs[BEEAI_VERSION] !== undefined || attrs.source !== undefined
  );
}

function getOpenInferenceInput(attrs: Record<string, any>): unknown {
  return parseJson(attrs[OPENINFERENCE_INPUT_VALUE]);
}

function getOpenInferenceOutput(attrs: Record<string, any>): unknown {
  return parseJson(attrs[OPENINFERENCE_OUTPUT_VALUE]);
}

function cacheBeeAIStartSpan(span: ReadableSpan): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return;

  rememberWorkflowSpan(span, attrs);

  if (span.name === "beeai-framework-main" || isBeeAIFrameworkParentSpan(span, attrs)) {
    rememberDroppedSpanParent(span);
    return;
  }

  if (getInstrumentationScopeName(span) !== BEEAI_SCOPE_NAME) return;

  const target = attrs[BEEAI_TARGET];
  const data = asRecord(parseJson(attrs[BEEAI_DATA]));
  const metadata = asRecord(parseJson(attrs[OPENINFERENCE_METADATA]));
  const state = asRecord(firstDefined(data?.state, metadata?.state));

  if (typeof target === "string" && target === "agent.toolCalling.start") {
    cacheChatInputFromAgentState(span, attrs, state);
  }

  if (typeof target === "string" && shouldDropTarget(target)) {
    rememberDroppedSpanParent(span);
  }

  if (!isBeeAIChatStartTarget(target)) return;

  const directInput = firstDefined(data?.input, getOpenInferenceInput(attrs));
  if (directInput !== undefined) {
    enqueuePendingChatInput(span, attrs, normalizeMeaningfulInputValue(directInput));
  }
}

function cacheChatInputFromAgentState(
  span: ReadableSpan,
  attrs: Record<string, any>,
  state: Record<string, any> | undefined,
): void {
  enqueuePendingChatInput(span, attrs, getChatInputFromState(state));
}

function cleanupBeeAIRawAttributes(attrs: Record<string, any>): void {
  delete attrs[BEEAI_TARGET];
  delete attrs[BEEAI_DATA];
  delete attrs[OPENINFERENCE_METADATA];
  delete attrs[BEEAI_TRACE_ID];
  delete attrs[BEEAI_VERSION];
  delete attrs.source;
  delete attrs[OPENINFERENCE_INPUT_VALUE];
  delete attrs[OPENINFERENCE_OUTPUT_VALUE];
  delete attrs["input.mime_type"];
  delete attrs["output.mime_type"];
  delete attrs[LLM_MODEL_NAME];
  delete attrs[LLM_PROVIDER];
  delete attrs[LLM_SYSTEM];
  delete attrs[LLM_TOKEN_COUNT_PROMPT];
  delete attrs[LLM_TOKEN_COUNT_COMPLETION];
  delete attrs[LLM_TOKEN_COUNT_TOTAL];
  delete attrs[`${OPENINFERENCE_METADATA}.model_name`];

  for (const key of Object.keys(attrs)) {
    if (key.startsWith("llm.input_messages.") || key.startsWith("llm.output_messages.")) {
      delete attrs[key];
    }
  }

  for (const key of OFF_CONTRACT_ALIASES) {
    delete attrs[key];
  }
}

function setInputOutputAttributes(
  span: ReadableSpan,
  attrs: Record<string, any>,
  logType: string,
  target: unknown,
  data: Record<string, any> | undefined,
  value: Record<string, any> | undefined,
): void {
  const metadata = asRecord(parseJson(attrs[OPENINFERENCE_METADATA]));
  const state = asRecord(firstDefined(data?.state, metadata?.state));
  const toolCallMsg = asRecord(firstDefined(data?.toolCallMsg, metadata?.toolCallMsg));
  const targetValue = typeof target === "string" ? target : "";

  const directInput = firstDefined(
    data?.input,
    value?.input,
    getOpenInferenceInput(attrs),
    toolCallMsg?.args,
  );
  const normalizedDirectInput = directInput !== undefined
    ? normalizeMeaningfulInputValue(directInput)
    : undefined;
  const cachedChatInput = logType === RespanLogType.CHAT && targetValue.endsWith(".success")
    ? dequeuePendingChatInput(span, attrs)
    : undefined;
  const stateChatInput = logType === RespanLogType.CHAT
    ? getChatInputFromState(state)
    : undefined;
  const input = logType === RespanLogType.CHAT
    ? firstDefined(cachedChatInput, stateChatInput, normalizedDirectInput)
    : firstDefined(
        normalizedDirectInput,
        logType === RespanLogType.AGENT ? getAgentStateInput(state) : undefined,
      );
  if (input !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonStr(input);
    if (logType === RespanLogType.CHAT) {
      setChatPromptAttributes(attrs, input, true);
    }
  }

  const finalAnswerOutput = targetValue.includes("finalAnswer.") && isFinalAnswerTool(toolCallMsg)
    ? getStateResultValue(state)
    : undefined;
  const directOutput = firstDefined(
    value?.messages,
    getOpenInferenceOutput(attrs),
    data?.output,
    value?.output,
    value?.result,
  );
  const output = firstDefined(
    finalAnswerOutput,
    directOutput !== undefined ? normalizeOutputValue(directOutput) : undefined,
    getMatchingToolResult(state, toolCallMsg),
    logType === RespanLogType.AGENT
      ? firstDefined(getStateResultValue(state), getLastStateMessageValue(state))
      : undefined,
  );
  if (output !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStr(output);
    if (logType === RespanLogType.CHAT) {
      setChatCompletionAttributes(attrs, output);
    }
  }
}

function hasMeaningfulCanonicalEntityContent(attrs: Record<string, any>): boolean {
  return (
    isMeaningfulStructuredValue(
      parseJson(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]),
    ) ||
    isMeaningfulStructuredValue(
      parseJson(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]),
    )
  );
}

function translateBeeAIEventSpan(span: ReadableSpan): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return;

  // OITracer intentionally starts the underlying OTEL span without attributes
  // and applies them immediately afterwards. Re-run start-event caching here,
  // when ReadableSpan attributes are complete, while retaining onStart support
  // for instrumentors that populate attributes eagerly.
  cacheBeeAIStartSpan(span);

  if (isBeeAIFrameworkParentSpan(span, attrs)) {
    dropSpan(span, attrs);
    return;
  }

  if (getInstrumentationScopeName(span) !== BEEAI_SCOPE_NAME) return;

  const target = attrs[BEEAI_TARGET];
  const data = asRecord(parseJson(attrs[BEEAI_DATA]));
  const metadata = asRecord(parseJson(attrs[OPENINFERENCE_METADATA]));
  const state = asRecord(firstDefined(data?.state, metadata?.state));

  if (typeof target === "string" && target === "agent.toolCalling.success") {
    flushPendingChatSpansFromState(span, attrs, state);
    cacheChatInputFromAgentState(span, attrs, state);
  }

  if (typeof target === "string" && shouldDropTarget(target)) {
    dropSpan(span, attrs);
    return;
  }

  const logType = getBeeAIEventLogType(target);
  if (!logType) return;

  setDefault(attrs, RespanSpanAttributes.RESPAN_LOG_TYPE, logType);
  if (logType === RespanLogType.CHAT || logType === RespanLogType.EMBEDDING) {
    setDefault(attrs, SpanAttributes.LLM_REQUEST_TYPE, logType);
  }

  setDefault(
    attrs,
    SpanAttributes.TRACELOOP_ENTITY_NAME,
    typeof target === "string" && target.length > 0 ? target : span.name,
  );
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, span.name);

  const value = asRecord(data?.value);
  const usage = asRecord(firstDefined(value?.usage, data?.usage));
  setTokenAttributes(attrs, usage);

  const provider = firstDefined(
    attrs[ATTR_GEN_AI_SYSTEM],
    attrs[LLM_PROVIDER],
    attrs[LLM_SYSTEM],
    inferProviderFromTarget(target),
  );
  if (typeof provider === "string" && provider.length > 0) {
    setDefault(attrs, ATTR_GEN_AI_SYSTEM, provider.toLowerCase());
  }

  const model = firstDefined(
    attrs[ATTR_GEN_AI_REQUEST_MODEL],
    attrs[LLM_MODEL_NAME],
    attrs[`${OPENINFERENCE_METADATA}.model_name`],
    data?.model,
    value?.model,
    metadata?.model_name,
    metadata?.modelName,
  );
  if (model !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_REQUEST_MODEL, model);
  }

  setInputOutputAttributes(span, attrs, logType, target, data, value);
  if (
    target === "agent.toolCalling.success" &&
    !hasMeaningfulCanonicalEntityContent(attrs)
  ) {
    dropSpan(span, attrs);
    cleanupBeeAIRawAttributes(attrs);
    return;
  }
  reparentFromDroppedSpans(span);
  cleanupBeeAIRawAttributes(attrs);
}


interface InstrumentationDelegate {
  activate(): void;
  deactivate(): void;
}

type DelegateFactory = (
  instrumentationClass: BeeAIInstrumentationClass,
  sdkModule: Record<string, unknown>,
) => InstrumentationDelegate | Promise<InstrumentationDelegate>;

interface OpenInferenceModule {
  OpenInferenceInstrumentor: new (
    instrumentationClass: BeeAIInstrumentationClass,
    sdkModule?: Record<string, unknown>,
  ) => InstrumentationDelegate;
}

export interface BeeAIInstrumentorOptions {
  /**
   * Optional BeeAI module object. Pass this in ESM/bundled environments when
   * the OpenInference instrumentor needs to patch a specific module instance.
   */
  sdkModule?: Record<string, unknown>;
  /**
   * Optional OpenInference BeeAI instrumentation constructor.
   *
   * Pass the constructor resolved by the application when a linked package,
   * pnpm workspace, or bundler can install more than one `beeai-framework`
   * module instance. Keeping this constructor in the same dependency realm as
   * `sdkModule` preserves the upstream instrumentor's `instanceof ChatModel`
   * checks and therefore its model, input/output, and usage serialization.
   */
  instrumentationClass?: BeeAIInstrumentationClass;
  /**
   * Internal extension point for tests and custom OpenInference delegate wiring.
   */
  delegateFactory?: DelegateFactory;
}

export class BeeAIInstrumentor {
  public readonly name = "beeai";

  private readonly _sdkModule?: Record<string, unknown>;
  private readonly _instrumentationClass: BeeAIInstrumentationClass;
  private readonly _delegateFactory?: DelegateFactory;
  private _delegate: InstrumentationDelegate | null = null;
  private _activationPromise: Promise<void> | null = null;
  private _activationGeneration = 0;
  private _activationRequested = false;
  private _ownsTranslatorHook = false;

  private static _translatorHookRefCount = 0;
  private static _patchedProcessor: any = null;
  private static _originalProcessorOnStart: ProcessorOnStart | null = null;
  private static _wrappedProcessorOnStart: ProcessorOnStart | null = null;
  private static _originalProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _wrappedProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _trackedBeeAISpans = new Set<ReadableSpan>();
  private static _pendingDelegateDeactivations = new Set<InstrumentationDelegate>();
  private static _preDelegateProcessor: any = null;
  private static _preDelegateProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _expectedNestedDelegateRestoreOnEnd: ProcessorOnEnd | null = null;

  constructor(options: BeeAIInstrumentorOptions = {}) {
    this._sdkModule = options.sdkModule;
    this._instrumentationClass =
      options.instrumentationClass ?? BeeAIInstrumentation;
    this._delegateFactory = options.delegateFactory;
  }

  async activate(): Promise<void> {
    this._activationRequested = true;
    if (this._delegate) {
      return;
    }

    const pendingActivation = this._activationPromise;
    if (pendingActivation) {
      await pendingActivation;
      if (this._activationRequested && !this._delegate) {
        await this.activate();
      }
      return;
    }

    const generation = this._activationGeneration;
    const activationPromise = this._activateGeneration(generation);
    this._activationPromise = activationPromise;
    try {
      await activationPromise;
    } finally {
      if (this._activationPromise === activationPromise) {
        this._activationPromise = null;
      }
    }
  }

  deactivate(): void {
    this._activationRequested = false;
    this._activationGeneration += 1;

    const delegate = this._delegate;
    this._delegate = null;
    if (this._ownsTranslatorHook && delegate) {
      // The OpenInference delegate owns the processor handler wrapped by the
      // BeeAI translator. Restore the outer hook first so the delegate can
      // safely restore its own handler. If BeeAI spans are still open, both
      // restorations are deferred until those spans finish.
      BeeAIInstrumentor._pendingDelegateDeactivations.add(delegate);
      this._releaseTranslatorHookOwnership();
    } else {
      this._releaseTranslatorHookOwnership();
      delegate?.deactivate();
    }
  }

  private async _activateGeneration(generation: number): Promise<void> {
    let delegate: InstrumentationDelegate | null = null;
    let activationAttempted = false;
    let ownsTranslatorHook = false;

    try {
      const sdkModule = this._sdkModule ?? (await this._loadBeeAIFramework());
      if (!this._isActivationCurrent(generation)) return;

      delegate = await this._createDelegate(sdkModule);
      if (!this._isActivationCurrent(generation)) return;

      const processorBeforeDelegate = BeeAIInstrumentor._getActiveSpanProcessor();
      const processorOnEndBeforeDelegate = typeof processorBeforeDelegate?.onEnd === "function"
        ? processorBeforeDelegate.onEnd as ProcessorOnEnd
        : null;
      activationAttempted = true;
      delegate.activate();
      const expectedNestedDelegateRestoreOnEnd =
        (delegate.constructor as any)?._originalProcessorOnEnd as ProcessorOnEnd | null;
      if (!this._isActivationCurrent(generation)) {
        delegate.deactivate();
        return;
      }

      if (BeeAIInstrumentor._installTranslatorHook()) {
        if (!BeeAIInstrumentor._preDelegateProcessor && processorOnEndBeforeDelegate) {
          BeeAIInstrumentor._preDelegateProcessor = processorBeforeDelegate;
          BeeAIInstrumentor._preDelegateProcessorOnEnd = processorOnEndBeforeDelegate;
          BeeAIInstrumentor._expectedNestedDelegateRestoreOnEnd =
            expectedNestedDelegateRestoreOnEnd;
        }
        BeeAIInstrumentor._translatorHookRefCount += 1;
        ownsTranslatorHook = true;
      }

      if (!this._isActivationCurrent(generation)) {
        if (ownsTranslatorHook) {
          BeeAIInstrumentor._pendingDelegateDeactivations.add(delegate);
          BeeAIInstrumentor._releaseTranslatorHookReference();
        } else {
          delegate.deactivate();
        }
        return;
      }

      this._delegate = delegate;
      this._ownsTranslatorHook = ownsTranslatorHook;
    } catch (error) {
      if (ownsTranslatorHook) {
        if (delegate) {
          BeeAIInstrumentor._pendingDelegateDeactivations.add(delegate);
        }
        BeeAIInstrumentor._releaseTranslatorHookReference();
      }
      if (delegate && activationAttempted && !ownsTranslatorHook) {
        try {
          delegate.deactivate();
        } catch {
          // Preserve the activation error so a later activate() can retry.
        }
      }
      throw error;
    }
  }

  private _isActivationCurrent(generation: number): boolean {
    return this._activationRequested && this._activationGeneration === generation;
  }

  private _releaseTranslatorHookOwnership(): void {
    if (!this._ownsTranslatorHook) return;
    this._ownsTranslatorHook = false;
    BeeAIInstrumentor._releaseTranslatorHookReference();
  }

  private static _releaseTranslatorHookReference(): void {
    BeeAIInstrumentor._translatorHookRefCount = Math.max(
      0,
      BeeAIInstrumentor._translatorHookRefCount - 1,
    );
    BeeAIInstrumentor._maybeRestoreTranslatorHookAfterDrain();
  }

  private static _getActiveSpanProcessor(): any {
    const tracerProvider = trace.getTracerProvider() as any;
    return (
      tracerProvider?.activeSpanProcessor ??
      tracerProvider?._activeSpanProcessor ??
      tracerProvider?._delegate?.activeSpanProcessor ??
      tracerProvider?._delegate?._activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?.activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?._activeSpanProcessor
    );
  }

  private static _shouldTrackBeeAISpan(span: ReadableSpan): boolean {
    return (
      getInstrumentationScopeName(span) === BEEAI_SCOPE_NAME ||
      span.name === "beeai-framework-main"
    );
  }

  private static _clearCompletedTraceState(span: ReadableSpan): void {
    const attrs = ((span as any).attributes ?? {}) as Record<string, any>;
    if (!isWorkflowSpan(span, attrs)) return;

    const traceId = getOtelTraceId(span);
    if (traceId) clearTraceState(traceId);
  }

  private static _maybeRestoreTranslatorHookAfterDrain(): void {
    if (
      BeeAIInstrumentor._translatorHookRefCount !== 0 ||
      BeeAIInstrumentor._trackedBeeAISpans.size !== 0
    ) return;

    const outerOnEndRestored = BeeAIInstrumentor._patchedProcessor
      ? BeeAIInstrumentor._restoreTranslatorHook()
      : false;
    BeeAIInstrumentor._deactivatePendingDelegates(outerOnEndRestored);
  }

  private static _deactivatePendingDelegates(outerOnEndRestored: boolean): void {
    const pendingDelegates = Array.from(BeeAIInstrumentor._pendingDelegateDeactivations);
    BeeAIInstrumentor._pendingDelegateDeactivations.clear();
    try {
      for (const delegate of pendingDelegates) {
        try {
          delegate.deactivate();
        } catch (error) {
          console.warn(
            "[respan] BeeAIInstrumentor: deferred OpenInference deactivation failed.",
            error,
          );
        }
      }
    } finally {
      const processor = BeeAIInstrumentor._preDelegateProcessor;
      const originalOnEnd = BeeAIInstrumentor._preDelegateProcessorOnEnd;
      const expectedNestedRestoreOnEnd =
        BeeAIInstrumentor._expectedNestedDelegateRestoreOnEnd;
      if (
        outerOnEndRestored &&
        processor &&
        originalOnEnd &&
        expectedNestedRestoreOnEnd &&
        processor.onEnd === expectedNestedRestoreOnEnd
      ) {
        processor.onEnd = originalOnEnd;
      }
      BeeAIInstrumentor._preDelegateProcessor = null;
      BeeAIInstrumentor._preDelegateProcessorOnEnd = null;
      BeeAIInstrumentor._expectedNestedDelegateRestoreOnEnd = null;
    }
  }

  private static _installTranslatorHook(): boolean {
    const processor = BeeAIInstrumentor._getActiveSpanProcessor();
    if (!processor || typeof processor.onEnd !== "function") {
      return false;
    }

    if (BeeAIInstrumentor._patchedProcessor === processor) {
      return true;
    }

    BeeAIInstrumentor._restoreTranslatorHook();

    const originalProcessorOnEnd = processor.onEnd as ProcessorOnEnd;
    const originalProcessorOnStart = typeof processor.onStart === "function"
      ? processor.onStart as ProcessorOnStart
      : null;
    const callOriginalProcessorOnEnd = (span: ReadableSpan) =>
      originalProcessorOnEnd.call(processor, span);
    const wrappedProcessorOnStart = originalProcessorOnStart
      ? (span: ReadableSpan, parentContext: unknown) => {
          if (BeeAIInstrumentor._translatorHookRefCount > 0) {
            if (BeeAIInstrumentor._shouldTrackBeeAISpan(span)) {
              BeeAIInstrumentor._trackedBeeAISpans.add(span);
            }
            try {
              cacheBeeAIStartSpan(span);
            } catch {
              // Translation must never block span export.
            }
          }
          return originalProcessorOnStart.call(processor, span, parentContext);
        }
      : null;
    const wrappedProcessorOnEnd = (span: ReadableSpan) => {
      const wasTracked = BeeAIInstrumentor._trackedBeeAISpans.delete(span);
      const shouldTranslate =
        BeeAIInstrumentor._translatorHookRefCount > 0 || wasTracked;
      const attrs = ((span as any).attributes ?? {}) as Record<string, any>;
      const beeAITraceKey = getSpanTraceKey(span, attrs);
      let shouldDelayExport = false;

      try {
        if (shouldTranslate) {
          translateBeeAIEventSpan(span);
          if (shouldDelayMissingChatInputSpan(span)) {
            queuePendingChatSpan(
              span,
              attrs,
              callOriginalProcessorOnEnd,
              beeAITraceKey,
            );
            shouldDelayExport = true;
          }
        }
      } catch {
        // Translation must never block span export.
      }

      if (shouldDelayExport) {
        BeeAIInstrumentor._maybeRestoreTranslatorHookAfterDrain();
        return;
      }

      try {
        return callOriginalProcessorOnEnd(span);
      } finally {
        BeeAIInstrumentor._clearCompletedTraceState(span);
        BeeAIInstrumentor._maybeRestoreTranslatorHookAfterDrain();
      }
    };

    if (wrappedProcessorOnStart) {
      processor.onStart = wrappedProcessorOnStart;
    }
    processor.onEnd = wrappedProcessorOnEnd;
    BeeAIInstrumentor._patchedProcessor = processor;
    BeeAIInstrumentor._originalProcessorOnStart = originalProcessorOnStart;
    BeeAIInstrumentor._wrappedProcessorOnStart = wrappedProcessorOnStart;
    BeeAIInstrumentor._originalProcessorOnEnd = originalProcessorOnEnd;
    BeeAIInstrumentor._wrappedProcessorOnEnd = wrappedProcessorOnEnd;
    return true;
  }

  private static _restoreTranslatorHook(): boolean {
    const processor = BeeAIInstrumentor._patchedProcessor;
    const originalOnStart = BeeAIInstrumentor._originalProcessorOnStart;
    const wrappedOnStart = BeeAIInstrumentor._wrappedProcessorOnStart;
    const originalOnEnd = BeeAIInstrumentor._originalProcessorOnEnd;
    const wrappedOnEnd = BeeAIInstrumentor._wrappedProcessorOnEnd;
    let restoredOnEnd = false;

    if (processor && originalOnStart) {
      if (!wrappedOnStart || processor.onStart === wrappedOnStart) {
        processor.onStart = originalOnStart;
      } else {
        console.warn(
          "[respan] BeeAIInstrumentor: active span processor onStart was modified externally; original handler could not be restored.",
        );
      }
    }

    if (processor && originalOnEnd) {
      if (!wrappedOnEnd || processor.onEnd === wrappedOnEnd) {
        processor.onEnd = originalOnEnd;
        restoredOnEnd = true;
      } else {
        console.warn(
          "[respan] BeeAIInstrumentor: active span processor onEnd was modified externally; original handler could not be restored.",
        );
      }
    }

    flushAllPendingChatSpans();
    pendingChatInputsByTrace.clear();
    pendingChatSpansByTrace.clear();
    droppedSpanParentsByTrace.clear();
    workflowSpanIdsByTrace.clear();
    traceStateTouchedAt.clear();
    BeeAIInstrumentor._trackedBeeAISpans.clear();
    BeeAIInstrumentor._patchedProcessor = null;
    BeeAIInstrumentor._originalProcessorOnStart = null;
    BeeAIInstrumentor._wrappedProcessorOnStart = null;
    BeeAIInstrumentor._originalProcessorOnEnd = null;
    BeeAIInstrumentor._wrappedProcessorOnEnd = null;
    return restoredOnEnd;
  }

  private async _createDelegate(
    sdkModule: Record<string, unknown>,
  ): Promise<InstrumentationDelegate> {
    if (this._delegateFactory) {
      return this._delegateFactory(this._instrumentationClass, sdkModule);
    }

    const { OpenInferenceInstrumentor } = await importOpenInferenceModule();
    return new OpenInferenceInstrumentor(this._instrumentationClass, sdkModule);
  }

  private async _loadBeeAIFramework(): Promise<Record<string, unknown>> {
    try {
      return (await import("beeai-framework")) as Record<string, unknown>;
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(
        `BeeAIInstrumentor requires beeai-framework to be installed, or a sdkModule option to be provided. ${reason}`,
      );
    }
  }
}

async function importOpenInferenceModule(): Promise<OpenInferenceModule> {
  const dynamicImport = new Function("specifier", "return import(specifier)") as (
    specifier: string,
  ) => Promise<OpenInferenceModule>;
  return dynamicImport("@respan/instrumentation-openinference");
}

export { BeeAIInstrumentor as BeeAIInstrumentation };
