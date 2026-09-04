/**
 * Respan instrumentation plugin for the pi coding agent
 * (`@earendil-works/pi-coding-agent`).
 *
 * pi has no global patch point: its runtime hands lifecycle events to
 * extensions (`pi.on(...)`) and to SDK subscribers (`session.subscribe(...)`).
 * `PiInstrumentor` therefore exposes two adapters that drive the same
 * `PiSessionTracer` state machine:
 *
 * - `instrumentor.extension` — a pi extension factory. Pass it to
 *   `new DefaultResourceLoader({ extensionFactories: [instrumentor.extension] })`
 *   or load it with `pi -e`. Every invocation creates its own tracer.
 * - `instrumentor.attach(session)` — subscribe to an `AgentSession`'s event
 *   stream. One tracer per session; returns a detach function.
 *
 * Both are fail-open: handlers never throw into pi, and spans are only
 * emitted while the instrumentor is active (i.e. after `Respan.initialize()`).
 */

import { hrTime } from "@opentelemetry/core";
import type { HrTime } from "@opentelemetry/api";
import type { RespanInstrumentation } from "@respan/respan";

import { debugLog } from "./_debug.js";
import {
  PiSessionTracer,
  type PiMetadataValue,
  type PiTraceScope,
  type PiTracerOptions,
} from "./_otel_emitter.js";
import { detectPiVersion } from "./_pi_version.js";
import type {
  PiAgentSessionLike,
  PiExtensionAPI,
  PiExtensionContextLike,
  PiExtensionFactory,
  PiToolDefinitionLike,
} from "./_pi_types.js";

export type {
  PiPromptCapture,
  PiSessionInfo,
  PiTraceScope,
  PiTracerOptions,
  PiMetadataValue,
} from "./_otel_emitter.js";
export {
  PiSessionTracer,
  PACKAGE_VERSION,
  PI_INSTRUMENTATION_NAME,
  sessionTraceId,
} from "./_otel_emitter.js";
export type {
  PiAgentMessage,
  PiAgentSessionLike,
  PiAssistantMessage,
  PiExtensionAPI,
  PiExtensionContextLike,
  PiExtensionFactory,
  PiModelLike,
  PiToolDefinitionLike,
  PiToolResultMessage,
  PiUsage,
  PiUserMessage,
} from "./_pi_types.js";
export { detectPiVersion } from "./_pi_version.js";

export interface PiInstrumentorOptions
  extends Omit<PiTracerOptions, "emit" | "enabled" | "threadIdentifier"> {}

/** Per-session correlation overrides for `PiInstrumentor.attach()`. */
export interface PiAttachOverrides {
  threadIdentifier?: string;
  customerIdentifier?: string;
  metadata?: Record<string, PiMetadataValue>;
}

interface AttachedSession {
  tracer: PiSessionTracer;
  unsubscribe: () => void;
  ref: WeakRef<PiAgentSessionLike>;
}

type RecordValue = Record<string, unknown>;

export class PiInstrumentor implements RespanInstrumentation {
  public readonly name = "pi";

  /**
   * pi extension factory. Each invocation creates its own `PiSessionTracer`
   * (one per pi session / process) and registers the `pi.on` handlers that
   * drive it. Handlers are registered synchronously and never throw.
   */
  public readonly extension: PiExtensionFactory;

  private readonly _options: PiInstrumentorOptions;
  /**
   * Attached sessions are held weakly (lookup by session object, plus weak
   * refs for enumeration): a session that is disposed without calling the
   * detach function must not keep its message history alive for the life of
   * the process.
   */
  private readonly _attached = new WeakMap<PiAgentSessionLike, AttachedSession>();
  private readonly _attachedRefs = new Set<WeakRef<PiAgentSessionLike>>();
  private readonly _extensionTracers = new Set<WeakRef<PiSessionTracer>>();
  private _active = false;

  constructor(options: PiInstrumentorOptions = {}) {
    this._options = { ...options };
    this.extension = (pi: PiExtensionAPI): void => {
      this._registerExtension(pi);
    };
  }

  activate(): void {
    this._active = true;
  }

  /**
   * Closes every open run — emitting its workflow/agent spans and any
   * interrupted chat/tool spans while emission is still enabled —
   * unsubscribes every `attach()`ed session, then stops emitting.
   * `Respan.shutdown()` calls this before the final flush, so a run cut short
   * by a shutdown still gets its root span instead of orphaned children.
   */
  deactivate(): void {
    for (const ref of [...this._attachedRefs]) {
      const session = ref.deref();
      if (session) {
        this.detach(session);
      } else {
        this._attachedRefs.delete(ref);
      }
    }
    for (const ref of [...this._extensionTracers]) {
      ref.deref()?.onSessionShutdown({ reason: "deactivate" });
    }
    this._active = false;
  }

  isActive(): boolean {
    return this._active;
  }

  /** Configured trace scope (`"run"` unless `"session"` was requested). */
  get traceScope(): PiTraceScope {
    return this._options.traceScope === "session" ? "session" : "run";
  }

  /** Number of live tracers (attached sessions + live extension runtimes). */
  get activeSessionCount(): number {
    pruneWeakRefs(this._attachedRefs);
    pruneWeakRefs(this._extensionTracers);
    return this._attachedRefs.size + this._extensionTracers.size;
  }

  /**
   * SDK alternative to the extension: subscribe to an `AgentSession`'s event
   * stream. One tracer per session (keyed by the session object); overrides
   * pin correlation per session. Returns a detach function that unsubscribes
   * and drops the tracer.
   */
  attach(session: PiAgentSessionLike, overrides: PiAttachOverrides = {}): () => void {
    if (!session || typeof session.subscribe !== "function") {
      throw new TypeError(
        "PiInstrumentor.attach() requires an AgentSession-like object with subscribe().",
      );
    }
    this.detach(session);

    const tracer = this._createTracer(overrides);
    const state: { turnStartedAt?: HrTime } = {};
    bindSession(tracer, session);
    tracer.setToolDefinitions(readSessionTools(session));
    tracer.onSessionStart();

    const listener = (event: unknown): void => {
      try {
        handleSessionEvent(tracer, session, state, event);
      } catch (error) {
        debugLog("session event handler failed", error);
      }
    };
    let unsubscribe: () => void;
    try {
      const returned = session.subscribe(listener);
      unsubscribe = typeof returned === "function" ? returned : () => undefined;
    } catch (error) {
      debugLog("session.subscribe() failed", error);
      unsubscribe = () => undefined;
    }

    const entry: AttachedSession = { tracer, unsubscribe, ref: new WeakRef(session) };
    this._attached.set(session, entry);
    this._attachedRefs.add(entry.ref);
    return () => {
      if (this._attached.get(session) !== entry) {
        return;
      }
      this._attached.delete(session);
      this._attachedRefs.delete(entry.ref);
      try {
        unsubscribe();
      } catch (error) {
        debugLog("unsubscribe failed", error);
      }
      tracer.onSessionShutdown({ reason: "detach" });
    };
  }

  /** Detach a previously attached session. Returns `true` when it was attached. */
  detach(session: PiAgentSessionLike): boolean {
    const entry = this._attached.get(session);
    if (!entry) {
      return false;
    }
    this._attached.delete(session);
    this._attachedRefs.delete(entry.ref);
    try {
      entry.unsubscribe();
    } catch (error) {
      debugLog("unsubscribe failed", error);
    }
    entry.tracer.onSessionShutdown({ reason: "detach" });
    return true;
  }

  private _createTracer(overrides: PiAttachOverrides = {}): PiSessionTracer {
    return new PiSessionTracer({
      ...this._options,
      threadIdentifier: overrides.threadIdentifier,
      customerIdentifier: overrides.customerIdentifier ?? this._options.customerIdentifier,
      metadata: { ...(this._options.metadata ?? {}), ...(overrides.metadata ?? {}) },
      enabled: () => this._active,
    });
  }

  private _registerExtension(pi: PiExtensionAPI): void {
    const tracer = this._createTracer();
    const ref = new WeakRef(tracer);
    pruneWeakRefs(this._extensionTracers);
    this._extensionTracers.add(ref);

    const on = (
      event: string,
      handler: (event: RecordValue, ctx: PiExtensionContextLike | undefined) => void,
    ): void => {
      pi.on(event, (payload: unknown, ctx: PiExtensionContextLike | undefined) => {
        try {
          handler(isRecord(payload) ? payload : {}, ctx);
        } catch (error) {
          debugLog(`pi handler for ${event} failed`, error);
        }
        // Never return a value: pi interprets handler results (message
        // replacement, context edits, ...).
      });
    };

    const bind = (ctx: PiExtensionContextLike | undefined): void => {
      if (!ctx) {
        return;
      }
      const sessionManager = ctx.sessionManager;
      tracer.setSession({
        sessionId: callSafely(() => sessionManager?.getSessionId?.()),
        sessionFile: callSafely(() => sessionManager?.getSessionFile?.()),
        cwd: typeof ctx.cwd === "string" ? ctx.cwd : undefined,
      });
      tracer.setModel(ctx.model);
      tracer.setThinkingLevel(ctx.thinkingLevel);
    };
    const refreshTools = (): void => {
      tracer.setToolDefinitions(readExtensionTools(pi));
    };

    on("session_start", (event, ctx) => {
      bind(ctx);
      tracer.setSession({ piVersion: detectPiVersion() });
      refreshTools();
      tracer.onSessionStart({ reason: stringValue(event.reason) });
    });
    on("before_agent_start", (event, ctx) => {
      bind(ctx);
      refreshTools();
      tracer.onBeforeAgentStart({ prompt: event.prompt, systemPrompt: event.systemPrompt });
    });
    on("agent_start", (_event, ctx) => {
      bind(ctx);
      tracer.onAgentStart();
    });
    on("context", (event) => {
      tracer.onContext(event.messages);
    });
    on("turn_start", (event) => {
      tracer.onTurnStart({ turnIndex: event.turnIndex });
    });
    on("turn_end", (event) => {
      tracer.onTurnEnd({ turnIndex: event.turnIndex, message: event.message, toolResults: event.toolResults });
    });
    on("message_start", (event) => {
      tracer.onMessageStart(event.message);
    });
    on("message_update", (event) => {
      tracer.onMessageUpdate({ message: event.message, assistantMessageEvent: event.assistantMessageEvent });
    });
    on("message_end", (event) => {
      tracer.onMessageEnd(event.message);
    });
    on("tool_execution_start", (event) => {
      tracer.onToolExecutionStart({ toolCallId: event.toolCallId, toolName: event.toolName, args: event.args });
    });
    on("tool_execution_end", (event) => {
      tracer.onToolExecutionEnd({
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        result: event.result,
        isError: event.isError,
      });
    });
    on("agent_end", (event) => {
      tracer.onAgentEnd({ messages: event.messages, willRetry: event.willRetry });
    });
    on("agent_settled", () => {
      tracer.onAgentSettled();
    });
    on("model_select", (event) => {
      tracer.setModel(event.model);
    });
    on("thinking_level_select", (event) => {
      tracer.setThinkingLevel(event.level);
    });
    on("session_before_compact", (event) => {
      const preparation = isRecord(event.preparation) ? event.preparation : undefined;
      tracer.onCompactionStart({
        reason: event.reason,
        willRetry: event.willRetry,
        tokensBefore: preparation?.tokensBefore,
      });
    });
    on("session_compact", (event) => {
      const entry = isRecord(event.compactionEntry) ? event.compactionEntry : undefined;
      tracer.onCompactionEnd({
        summary: entry?.summary,
        tokensBefore: entry?.tokensBefore,
        firstKeptEntryId: entry?.firstKeptEntryId,
        reason: event.reason,
        willRetry: event.willRetry,
      });
    });
    on("session_compact_failed", (event) => {
      tracer.onCompactionEnd({
        reason: event.reason,
        willRetry: event.willRetry,
        error: event.errorMessage,
        aborted: event.aborted,
      });
    });
    on("session_before_tree", (event) => {
      const preparation = isRecord(event.preparation) ? event.preparation : undefined;
      tracer.onBranchSummaryStart({
        userWantsSummary: preparation?.userWantsSummary,
        targetId: preparation?.targetId,
        oldLeafId: preparation?.oldLeafId,
        label: preparation?.label,
      });
    });
    on("session_tree", (event) => {
      tracer.onBranchSummaryEnd({
        newLeafId: event.newLeafId,
        oldLeafId: event.oldLeafId,
        summaryEntry: event.summaryEntry,
      });
    });
    on("session_shutdown", (event) => {
      tracer.onSessionShutdown({ reason: stringValue(event.reason) });
      this._extensionTracers.delete(ref);
    });
  }
}

/**
 * Convenience: `new PiInstrumentor(options)` that is already active. Useful
 * when the OTEL pipeline is initialized elsewhere and you just need the
 * extension factory.
 */
export function createPiExtension(options: PiInstrumentorOptions = {}): PiExtensionFactory {
  const instrumentor = new PiInstrumentor(options);
  instrumentor.activate();
  return instrumentor.extension;
}

// ── Subscribe (SDK) adapter ───────────────────────────────────────────────

function handleSessionEvent(
  tracer: PiSessionTracer,
  session: PiAgentSessionLike,
  state: { turnStartedAt?: HrTime },
  event: unknown,
): void {
  if (!isRecord(event) || typeof event.type !== "string") {
    return;
  }
  switch (event.type) {
    case "agent_start":
      bindSession(tracer, session);
      tracer.setToolDefinitions(readSessionTools(session));
      tracer.onAgentStart();
      break;
    case "agent_end":
      tracer.onAgentEnd({ messages: event.messages, willRetry: event.willRetry === true });
      break;
    case "auto_retry_end":
      // A cancelled / exhausted retry cycle ends a run kept open by willRetry.
      tracer.onAutoRetryEnd({ success: event.success, finalError: event.finalError });
      break;
    case "agent_settled":
      tracer.onAgentSettled();
      break;
    case "turn_start":
      state.turnStartedAt = hrTime();
      tracer.onTurnStart({ turnIndex: event.turnIndex });
      break;
    case "turn_end":
      tracer.onTurnEnd({ turnIndex: event.turnIndex, message: event.message, toolResults: event.toolResults });
      break;
    case "message_start":
      if (isAssistantMessage(event.message)) {
        ensureLlmInput(tracer, session, state);
      }
      tracer.onMessageStart(event.message);
      break;
    case "message_update":
      tracer.onMessageUpdate({ message: event.message, assistantMessageEvent: event.assistantMessageEvent });
      break;
    case "message_end":
      if (isAssistantMessage(event.message)) {
        ensureLlmInput(tracer, session, state);
      }
      tracer.onMessageEnd(event.message);
      break;
    case "tool_execution_start":
      tracer.onToolExecutionStart({ toolCallId: event.toolCallId, toolName: event.toolName, args: event.args });
      break;
    case "tool_execution_end":
      tracer.onToolExecutionEnd({
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        result: event.result,
        isError: event.isError,
      });
      break;
    case "compaction_start":
      tracer.onCompactionStart({ reason: event.reason });
      break;
    case "compaction_end": {
      const result = isRecord(event.result) ? event.result : undefined;
      tracer.onCompactionEnd({
        summary: result?.summary,
        tokensBefore: result?.tokensBefore,
        tokensAfter: result?.estimatedTokensAfter,
        firstKeptEntryId: result?.firstKeptEntryId,
        reason: event.reason,
        willRetry: event.willRetry,
        error: event.errorMessage,
        aborted: event.aborted,
      });
      break;
    }
    case "thinking_level_changed":
      tracer.setThinkingLevel(event.level);
      break;
    default:
      break;
  }
}

/**
 * Subscribe mode has no `context` event: snapshot `session.messages` when
 * the assistant message starts (pi appends messages to agent state on
 * `message_end`, so at that moment the state is exactly the LLM input). The
 * LLM start time is the preceding `turn_start`.
 */
function ensureLlmInput(
  tracer: PiSessionTracer,
  session: PiAgentSessionLike,
  state: { turnStartedAt?: HrTime },
): void {
  if (!tracer.hasOpenRun || tracer.hasPendingLlm) {
    return;
  }
  tracer.onContext(readSessionMessages(session), { startTime: state.turnStartedAt });
  state.turnStartedAt = undefined;
}

function readSessionMessages(session: PiAgentSessionLike): unknown[] {
  const messages = callSafely(() => session.messages);
  if (!Array.isArray(messages)) {
    return [];
  }
  const snapshot = [...messages];
  // An LLM input never ends with an assistant message; a trailing one is the
  // (partial or just-finalized) response itself.
  while (snapshot.length > 0 && isAssistantMessage(snapshot[snapshot.length - 1])) {
    snapshot.pop();
  }
  return snapshot;
}

function bindSession(tracer: PiSessionTracer, session: PiAgentSessionLike): void {
  tracer.setSession({
    sessionId: callSafely(() => session.sessionId),
    sessionFile: callSafely(() => session.sessionFile),
    cwd: callSafely(() => session.sessionManager?.getCwd?.()),
    piVersion: detectPiVersion(),
  });
  tracer.setModel(callSafely(() => session.model));
  tracer.setThinkingLevel(callSafely(() => session.thinkingLevel));
}

function readSessionTools(session: PiAgentSessionLike): PiToolDefinitionLike[] | undefined {
  const all = callSafely(() =>
    typeof session.getAllTools === "function" ? session.getAllTools() : undefined,
  );
  if (Array.isArray(all)) {
    const active = callSafely(() =>
      typeof session.getActiveToolNames === "function" ? session.getActiveToolNames() : undefined,
    );
    return filterTools(all, active);
  }
  const agent = callSafely(() => session.agent);
  const agentState = isRecord(agent) && isRecord(agent.state) ? agent.state : undefined;
  const tools = callSafely(() => agentState?.tools);
  return Array.isArray(tools) ? filterTools(tools, undefined) : undefined;
}

function readExtensionTools(pi: PiExtensionAPI): PiToolDefinitionLike[] | undefined {
  const all = callSafely(() => (typeof pi.getAllTools === "function" ? pi.getAllTools() : undefined));
  if (!Array.isArray(all)) {
    return undefined;
  }
  const active = callSafely(() =>
    typeof pi.getActiveTools === "function" ? pi.getActiveTools() : undefined,
  );
  return filterTools(all, active);
}

function filterTools(all: unknown[], active: unknown): PiToolDefinitionLike[] {
  const activeNames = Array.isArray(active)
    ? new Set(active.filter((name): name is string => typeof name === "string"))
    : undefined;
  const tools: PiToolDefinitionLike[] = [];
  for (const tool of all) {
    if (!isRecord(tool) || typeof tool.name !== "string" || !tool.name) {
      continue;
    }
    if (activeNames && !activeNames.has(tool.name)) {
      continue;
    }
    tools.push({
      name: tool.name,
      description: typeof tool.description === "string" ? tool.description : undefined,
      parameters: tool.parameters,
    });
  }
  return tools;
}

function isAssistantMessage(message: unknown): boolean {
  return isRecord(message) && message.role === "assistant";
}

function pruneWeakRefs<T extends object>(refs: Set<WeakRef<T>>): void {
  for (const ref of [...refs]) {
    if (ref.deref() === undefined) {
      refs.delete(ref);
    }
  }
}

function callSafely<T>(fn: () => T): T | undefined {
  try {
    return fn();
  } catch {
    return undefined;
  }
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function isRecord(value: unknown): value is RecordValue {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
