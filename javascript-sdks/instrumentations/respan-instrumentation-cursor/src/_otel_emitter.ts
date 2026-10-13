import {
  context,
  SpanStatusCode,
  type Span,
  type Attributes,
} from "@opentelemetry/api";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { getPropagatedAttributes } from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { ATTR_GEN_AI_TOOL_CALL_ID } from "@opentelemetry/semantic-conventions/incubating";
import { startCapture, type Capture } from "./_capture.js";

type RecordValue = Record<string, any>;
export interface CursorRunState {
  capture: Capture;
  finished: boolean;
  metadata: RecordValue;
  output: string[];
  pendingTools: Map<string, { capture: Capture; name: string }>;
  completedTools: Set<string>;
  seen: WeakSet<object>;
}
export function createCursorRunState(
  agentName: string,
  content: boolean,
  message?: unknown,
  agent?: RecordValue,
  options?: RecordValue,
): CursorRunState {
  // Start the real span, including the sampler, before inspecting any payload.
  const capture = startCapture(
    `${agentName}.agent`,
    {
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: agentName,
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.AGENT,
    },
    content,
  );
  const state: CursorRunState = {
    capture,
    finished: false,
    metadata: {},
    output: [],
    pendingTools: new Map(),
    completedTools: new Set(),
    seen: new WeakSet(),
  };
  if (capture.allowed()) {
    const inherited = getPropagatedAttributes()?.metadata;
    if (inherited && typeof inherited === "object")
      state.metadata = { ...inherited };
    setIdentity(state, agent);
    if (message !== undefined)
      capture.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        json(message),
      );
    const model = options?.model ?? agent?.model;
    if (model && typeof model.id === "string")
      state.metadata.cursor_model = model.id;
    registerCursorToolDefinitions(state, options);
  }
  return state;
}
export function registerCursorToolDefinitions(
  state: CursorRunState,
  options?: RecordValue,
): void {
  if (!state.capture.allowed()) return;
  const custom = options?.local?.customTools;
  if (!custom || typeof custom !== "object") return;
  state.metadata.cursor_tools = Object.entries(custom).map(([name, value]) => {
    const tool = value as RecordValue;
    return {
      name,
      ...(tool.description !== undefined
        ? { description: tool.description }
        : {}),
      ...(tool.inputSchema !== undefined
        ? { inputSchema: tool.inputSchema }
        : {}),
      ...(tool.outputSchema !== undefined
        ? { outputSchema: tool.outputSchema }
        : {}),
      ...(tool.annotations !== undefined
        ? { annotations: tool.annotations }
        : {}),
    };
  });
}
function setIdentity(state: CursorRunState, value?: RecordValue): void {
  if (!state.capture.allowed() || !value) return;
  for (const [source, destination] of [
    ["agentId", "cursor_agent_id"],
    ["id", "cursor_run_id"],
    ["requestId", "cursor_request_id"],
  ] as const)
    if (typeof value[source] === "string")
      state.metadata[destination] = value[source];
  if (typeof value.model?.id === "string")
    state.metadata.cursor_model = value.model.id;
}
export function recordCursorRunResult(
  state: CursorRunState,
  value: unknown,
): void {
  if (!value || typeof value !== "object") return;
  const result = value as RecordValue;
  if (state.capture.allowed()) {
    setIdentity(state, result);
    if (typeof result.result === "string")
      state.capture.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        result.result,
      );
    if (typeof result.status === "string")
      state.metadata.cursor_status = result.status;
    if (result.usage && typeof result.usage === "object") {
      const usage: RecordValue = {};
      for (const key of [
        "inputTokens",
        "outputTokens",
        "cacheReadTokens",
        "cacheWriteTokens",
        "totalTokens",
        "reasoningTokens",
      ])
        if (
          typeof result.usage[key] === "number" &&
          Number.isFinite(result.usage[key])
        )
          usage[key] = result.usage[key];
      if (Object.keys(usage).length) state.metadata.cursor_usage = usage;
    }
  }
  if (result.status === "error" || result.status === "cancelled") {
    state.capture.span.setStatus({
      code: SpanStatusCode.ERROR,
      ...(state.capture.allowed() && typeof result.error?.message === "string"
        ? { message: result.error.message }
        : {}),
    });
  }
}
export function markCursorRunError(
  state: CursorRunState,
  error: unknown,
): void {
  state.capture.span.setStatus({ code: SpanStatusCode.ERROR });
  if (state.capture.allowed() && error instanceof Error)
    state.capture.span.recordException(error);
}
export function trackCursorMessage(state: CursorRunState, raw: unknown): void {
  if (state.finished || !raw || typeof raw !== "object") return;
  if (state.seen.has(raw)) return;
  state.seen.add(raw);
  const message = raw as RecordValue;
  if (message.type === "tool_call") {
    trackTool(
      state,
      message.call_id,
      message.name,
      message.status,
      () => message.args,
      () => message.result,
    );
    return;
  }
  if (!state.capture.allowed()) return;
  if (typeof message.agent_id === "string")
    state.metadata.cursor_agent_id = message.agent_id;
  if (typeof message.run_id === "string")
    state.metadata.cursor_run_id = message.run_id;
  if (message.type === "usage") {
    recordCursorRunResult(state, { usage: message.usage });
    return;
  }
  if (message.type === "request") {
    if (typeof message.request_id === "string")
      state.metadata.cursor_request_id = message.request_id;
    return;
  }
  if (message.type === "system") {
    if (typeof message.model?.id === "string")
      state.metadata.cursor_model = message.model.id;
    return;
  }
  if (message.type === "assistant")
    for (const block of message.message?.content ?? [])
      if (block.type === "text" && typeof block.text === "string")
        state.output.push(block.text);
  if (message.type === "thinking" || message.type === "task")
    state.capture.span.addEvent(`cursor.${message.type}`, {
      payload: json(message),
    });
  if (message.type === "status") {
    state.metadata.cursor_status = message.status;
    if (["ERROR", "CANCELLED", "EXPIRED"].includes(message.status))
      state.capture.span.setStatus({ code: SpanStatusCode.ERROR });
  }
}
export function trackCursorCallback(
  state: CursorRunState,
  kind: string,
  raw: unknown,
): void {
  if (!state.capture.allowed() || !raw || typeof raw !== "object") return;
  const payload = raw as RecordValue;
  if (kind === "onDelta" && payload.update) {
    const update = payload.update;
    if (
      update.type === "tool-call-started" ||
      update.type === "tool-call-completed"
    )
      trackTool(
        state,
        update.callId,
        update.toolCall?.name ?? update.toolCall?.type,
        update.type === "tool-call-started"
          ? "running"
          : update.toolCall?.result?.status === "error"
            ? "error"
            : "completed",
        () => update.toolCall?.args,
        () => update.toolCall?.result,
      );
  }
}
function trackTool(
  state: CursorRunState,
  callId: unknown,
  name: unknown,
  status: unknown,
  args: () => unknown,
  result: () => unknown,
): void {
  if (typeof name !== "string") return;
  const id = typeof callId === "string" && callId ? callId : undefined;
  if (id && state.completedTools.has(id)) return;
  if (status === "running") {
    if (!id || state.pendingTools.has(id)) return;
    const capture = startCapture(
      `${name}.tool`,
      {
        [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
        [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
        [ATTR_GEN_AI_TOOL_CALL_ID]: id,
      },
      state.capture.contentAllowed(),
      state.capture.ctx,
    );
    if (capture.allowed())
      capture.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        json({ name, arguments: args() }),
      );
    state.pendingTools.set(id, { capture, name });
    return;
  }
  const pending = id ? state.pendingTools.get(id) : undefined;
  const capture =
    pending?.capture ??
    startCapture(
      `${name}.tool`,
      {
        [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
        [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
        ...(id ? { [ATTR_GEN_AI_TOOL_CALL_ID]: id } : {}),
      },
      state.capture.contentAllowed(),
      state.capture.ctx,
    );
  if (capture.allowed()) {
    if (!pending)
      capture.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        json({ name, arguments: args() }),
      );
    const output = result();
    if (output !== undefined)
      capture.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        json(output),
      );
  }
  if (status === "error")
    capture.span.setStatus({ code: SpanStatusCode.ERROR });
  capture.end();
  if (id) {
    state.pendingTools.delete(id);
    state.completedTools.add(id);
  }
}
export function emitCursorToolExecution(
  state: CursorRunState,
  name: string,
  args: unknown,
  contextValue: unknown,
  result?: unknown,
  error?: unknown,
  failed = false,
): void {
  const callId =
    contextValue && typeof contextValue === "object"
      ? (contextValue as RecordValue).toolCallId
      : undefined;
  trackTool(
    state,
    callId,
    name,
    failed ||
      (result &&
        typeof result === "object" &&
        (result as RecordValue).isError === true)
      ? "error"
      : "completed",
    () => args,
    () => (failed ? error : result),
  );
}
export function emitFinalCursorRunSpans(state: CursorRunState): void {
  if (state.finished) return;
  state.finished = true;
  for (const { capture } of state.pendingTools.values()) {
    capture.span.setStatus({ code: SpanStatusCode.ERROR });
    capture.end();
  }
  state.pendingTools.clear();
  if (state.capture.allowed()) {
    if (state.output.length)
      state.capture.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        state.output.join(""),
      );
    state.capture.span.setAttribute(
      RespanSpanAttributes.RESPAN_METADATA,
      json(state.metadata),
    );
  }
  state.capture.end();
}
export function json(value: unknown): string {
  const seen = new WeakSet<object>();
  try {
    return (
      JSON.stringify(value, (_key, next) => {
        if (typeof next === "bigint") return next.toString();
        if (next && typeof next === "object") {
          if (seen.has(next)) return "[Circular]";
          seen.add(next);
        }
        return next;
      }) ?? "null"
    );
  } catch {
    return "[Unserializable]";
  }
}
