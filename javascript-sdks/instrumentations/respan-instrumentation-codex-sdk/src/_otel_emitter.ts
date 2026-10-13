import {
  context,
  trace,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { ContentGuard } from "./_privacy.js";
import { data, json, record, snapshot, type Row } from "./_data.js";

const tracer = () => trace.getTracer("@respan/instrumentation-codex-sdk");
export interface CodexEmitterOptions {
  agentName?: string;
  captureItemSpans?: boolean;
  workflowName?: string;
  traceContent?: boolean;
}
interface ItemState {
  span: Span;
  guard: ContentGuard;
  completed: boolean;
}
export interface CodexTurnState {
  span: Span;
  guard: ContentGuard;
  context: Context;
  items: Map<string, ItemState>;
  metadata: Row;
  captureItemSpans: boolean;
  ended: boolean;
  eventsObserved: boolean;
  input?: unknown;
  output?: string;
}
export function createCodexTurnState(
  input: unknown,
  thread: unknown,
  options: CodexEmitterOptions,
): CodexTurnState {
  const parent = context.active();
  const name = options.agentName?.trim() || "codex-agent";
  const span = tracer().startSpan(
    name,
    { attributes: base(name, RespanLogType.AGENT) },
    parent,
  );
  const guard = new ContentGuard(span, options.traceContent !== false, parent);
  const state: CodexTurnState = {
    span,
    guard,
    context: trace.setSpan(parent, span),
    items: new Map(),
    metadata: {},
    captureItemSpans: options.captureItemSpans !== false,
    ended: false,
    eventsObserved: false,
  };
  if (options.workflowName)
    guard.internal(() =>
      span.setAttribute(
        RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
        options.workflowName!,
      ),
    );
  const model = data(data(thread, "_threadOptions"), "model");
  if (typeof model === "string") state.metadata.codex_model = model;
  threadMetadata(state, thread);
  if (guard.allowed()) {
    state.input = snapshot(
      typeof input === "string"
        ? [{ role: "user", content: input }]
        : [{ role: "user", content: input }],
    );
    guard.internal(() =>
      span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        json(state.input),
      ),
    );
  }
  return state;
}
export function trackCodexEvent(state: CodexTurnState, event: unknown): void {
  state.guard.observe();
  state.eventsObserved = true;
  switch (data(event, "type")) {
    case "thread.started": {
      const id = data(event, "thread_id");
      if (typeof id === "string")
        state.guard.internal(() =>
          state.span.setAttribute(RespanSpanAttributes.RESPAN_THREADS_ID, id),
        );
      break;
    }
    case "turn.completed":
      usage(state, data(event, "usage"));
      break;
    case "turn.failed":
      fail(state.span, state.guard, data(event, "error"));
      break;
    case "error":
      fail(state.span, state.guard, event);
      break;
    case "item.started":
    case "item.updated":
    case "item.completed":
      trackItem(
        state,
        data(event, "item"),
        data(event, "type") === "item.completed",
      );
      break;
  }
}
export function trackCodexRunResult(
  state: CodexTurnState,
  result: unknown,
): void {
  state.guard.observe();
  usage(state, data(result, "usage"));
  if (!state.eventsObserved) {
    const items = data(result, "items");
    if (Array.isArray(items))
      for (const item of items) trackItem(state, item, true);
  }
  const output = data(result, "finalResponse");
  if (state.guard.allowed() && typeof output === "string")
    state.output = output;
}
export function finishCodexTurn(state: CodexTurnState, thread: unknown): void {
  if (state.ended) return;
  state.ended = true;
  state.guard.observe();
  threadMetadata(state, thread);
  state.guard.metadata(state.metadata);
  if (state.guard.allowed() && state.output !== undefined)
    state.guard.internal(() =>
      state.span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        json(state.output),
      ),
    );
  for (const item of state.items.values())
    if (!item.completed) {
      item.completed = true;
      if (item.guard.prepare()) item.span.end();
    }
  if (state.guard.prepare()) state.span.end();
}
export function fail(span: Span, guard: ContentGuard, error: unknown): void {
  const message = guard.allowed() ? data(error, "message") : undefined;
  guard.internal(() =>
    span.setStatus({
      code: SpanStatusCode.ERROR,
      ...(typeof message === "string" ? { message } : {}),
    }),
  );
}
function threadMetadata(state: CodexTurnState, thread: unknown): void {
  const id = data(thread, "_id");
  if (typeof id === "string")
    state.guard.internal(() =>
      state.span.setAttribute(RespanSpanAttributes.RESPAN_THREADS_ID, id),
    );
}
function usage(state: CodexTurnState, value: unknown): void {
  const fields: Row = {};
  for (const key of [
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
  ]) {
    const count = data(value, key);
    if (typeof count === "number" && Number.isFinite(count) && count >= 0)
      fields[key] = count;
  }
  if (Object.keys(fields).length) state.metadata.codex_turn_usage = fields;
}
function trackItem(
  state: CodexTurnState,
  item: unknown,
  completed: boolean,
): void {
  const type = data(item, "type");
  if (type === "agent_message") {
    const text = state.guard.allowed() ? data(item, "text") : undefined;
    if (completed && typeof text === "string") state.output = text;
    return;
  }
  if (!state.captureItemSpans || typeof type !== "string") return;
  const logType = ["command_execution", "mcp_tool_call", "web_search"].includes(
    type,
  )
    ? RespanLogType.TOOL
    : ["reasoning", "file_change", "todo_list", "error"].includes(type)
      ? RespanLogType.TASK
      : undefined;
  if (!logType) return;
  const id = data(item, "id");
  if (typeof id !== "string") return;
  let current = state.items.get(id);
  if (!current) {
    const name =
      type === "mcp_tool_call"
        ? `mcp__${string(data(item, "server"), "mcp")}__${string(data(item, "tool"), "tool")}`
        : type;
    const span = tracer().startSpan(
      name,
      {
        attributes: {
          ...base(name, logType),
        },
      },
      state.context,
    );
    const guard = new ContentGuard(
      span,
      state.guard.captureAllowed(),
      state.context,
      state.guard,
    );
    current = { span, guard, completed: false };
    state.items.set(id, current);
    current.guard.metadata({ codex_item_id: id, codex_item_type: type });
  }
  if (current.completed) return;
  current.guard.observe();
  if (
    data(item, "status") === "failed" ||
    type === "error" ||
    record(data(item, "error"))
  )
    fail(current.span, current.guard, data(item, "error") ?? item);
  if (current.guard.allowed()) {
    const payload = snapshot(item) as Row;
    let input: unknown;
    let output: unknown;
    switch (type) {
      case "command_execution":
        input = {
          name: "command_execution",
          arguments: { command: payload.command },
        };
        output = {
          aggregated_output: payload.aggregated_output,
          exit_code: payload.exit_code,
          status: payload.status,
        };
        break;
      case "mcp_tool_call":
        input = {
          name: `mcp__${payload.server}__${payload.tool}`,
          arguments: payload.arguments,
        };
        output = payload.error ?? payload.result;
        break;
      case "web_search":
        input = { name: "web_search", arguments: { query: payload.query } };
        break;
      case "reasoning":
        output = payload.text;
        break;
      case "file_change":
        input = payload.changes;
        output = { changes: payload.changes, status: payload.status };
        break;
      case "todo_list":
        output = payload.items;
        break;
      case "error":
        output = { message: payload.message };
        break;
    }
    current.guard.internal(() => {
      if (input !== undefined)
        current!.span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_INPUT,
          json(input),
        );
      if (output !== undefined)
        current!.span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
          json(output),
        );
    });
  }
  if (completed) {
    current.completed = true;
    if (current.guard.prepare()) current.span.end();
  }
}
function base(name: string, type: RespanLogType): Record<string, string> {
  return {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: type,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
  };
}
function string(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}
