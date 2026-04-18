/**
 * Respan instrumentation plugin for the Anthropic SDK.
 *
 * Monkey-patches `messages.create()` and `messages.stream()` on the
 * Anthropic client prototype to emit OTEL spans with GenAI attributes.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { AnthropicInstrumentor } from "@respan/instrumentation-anthropic";
 *
 * const respan = new Respan({
 *   instrumentations: [new AnthropicInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import { context, trace, SpanKind, SpanStatusCode, TraceFlags } from "@opentelemetry/api";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const PACKAGE_VERSION = "1.1.0";
const STREAM_INSTRUMENTED = Symbol("respan.anthropic.stream.instrumented");
const TOOL_USE_JSON_BUFFER_KEY = Symbol("respan.anthropic.tool.use.json.buffer");

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

function safeJson(obj: any): string {
  try {
    return JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value,
    );
  } catch {
    return String(obj);
  }
}

function toSerializableValue(value: any): any {
  if (value === null || value === undefined) return value;
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  try {
    return JSON.parse(
      JSON.stringify(value, (_key, innerValue) =>
        typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
      ),
    );
  } catch {
    return value;
  }
}

// ---------------------------------------------------------------------------
// Content normalization
// ---------------------------------------------------------------------------

function normalizeContentBlock(block: any): string {
  if (typeof block === "string") return block;
  if (block && typeof block === "object") {
    if (typeof block.text === "string") return block.text;
    if (block.type === "image") return "[image]";
  }
  return "";
}

function stringifyStructured(value: any): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return safeJson(value);
}

function normalizeToolCallBlock(block: any): Record<string, any> | null {
  if (!block || typeof block !== "object" || block.type !== "tool_use") {
    return null;
  }

  return {
    id: block.id ?? "",
    type: "function",
    function: {
      name: block.name ?? "",
      arguments: safeJson(block.input ?? {}),
    },
  };
}

function normalizeToolResultBlock(block: any): Record<string, any> | null {
  if (!block || typeof block !== "object" || block.type !== "tool_result") {
    return null;
  }

  const normalized: Record<string, any> = {
    role: "tool",
    content: stringifyStructured(block.content ?? ""),
  };
  if (block.tool_use_id) normalized.tool_call_id = block.tool_use_id;
  if (block.is_error === true) normalized.is_error = true;
  return normalized;
}

function formatInputMessages(
  messages: any[],
  system?: any,
): any[] {
  const result: any[] = [];

  // System prompt
  if (system != null) {
    if (typeof system === "string") {
      result.push({ role: "system", content: system });
    } else if (Array.isArray(system)) {
      const parts = system
        .map((b: any) => {
          if (typeof b === "string") return b;
          if (b && typeof b.text === "string") return b.text;
          return String(b);
        })
        .filter(Boolean);
      result.push({ role: "system", content: parts.join("\n") });
    }
  }

  for (const msg of messages) {
    const role = msg?.role ?? "user";
    const content = msg?.content ?? "";

    if (!Array.isArray(content)) {
      result.push({ role, content });
      continue;
    }

    const textParts: string[] = [];
    const toolCalls: Record<string, any>[] = [];
    const toolResults: Record<string, any>[] = [];

    for (const block of content) {
      const text = normalizeContentBlock(block);
      if (text) textParts.push(text);

      const toolCall = normalizeToolCallBlock(block);
      if (toolCall) {
        toolCalls.push(toolCall);
        continue;
      }

      const toolResult = normalizeToolResultBlock(block);
      if (toolResult) {
        toolResults.push(toolResult);
      }
    }

    if (textParts.length > 0 || toolCalls.length > 0) {
      const normalizedMessage: Record<string, any> = {
        role,
        content: textParts.join("\n"),
      };
      if (toolCalls.length > 0) {
        normalizedMessage.tool_calls = toolCalls;
      }
      result.push(normalizedMessage);
    }

    result.push(...toolResults);
  }

  return result;
}

function formatOutput(message: any): string {
  const content = message?.content;
  if (!content) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  const parts: string[] = [];
  for (const block of content) {
    const text = normalizeContentBlock(block);
    if (text) parts.push(text);
  }
  return parts.join("\n");
}

function formatOutputMessage(message: any): Record<string, any> {
  const outputMessage: Record<string, any> = {
    role: "assistant",
    content: formatOutput(message),
  };

  const toolCalls = extractToolCalls(message);
  if (toolCalls) {
    outputMessage.tool_calls = toolCalls;
  }

  return outputMessage;
}

function extractToolCalls(message: any): any[] | null {
  const content = message?.content;
  if (!Array.isArray(content)) return null;

  const toolCalls: any[] = [];
  for (const block of content) {
    const toolCall = normalizeToolCallBlock(block);
    if (toolCall) toolCalls.push(toolCall);
  }

  return toolCalls.length ? toolCalls : null;
}

function extractToolCallsFromInputMessages(messages: any[] | undefined): any[] | null {
  if (!Array.isArray(messages)) return null;

  const toolCalls: any[] = [];
  for (const message of messages) {
    if (Array.isArray(message?.tool_calls)) {
      toolCalls.push(...message.tool_calls);
    }

    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      const toolCall = normalizeToolCallBlock(block);
      if (toolCall) toolCalls.push(toolCall);
    }
  }

  return toolCalls.length ? toolCalls : null;
}

function mergeToolCalls(...groups: Array<any[] | null | undefined>): any[] | null {
  const merged: any[] = [];
  const seen = new Set<string>();

  for (const group of groups) {
    for (const toolCall of group ?? []) {
      const key = safeJson(toolCall);
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(toolCall);
    }
  }

  return merged.length ? merged : null;
}

function extractToolExecutions(messages: any[] | undefined): Array<{
  id: string;
  name: string;
  input: any;
  output: any;
  isError: boolean;
}> {
  if (!Array.isArray(messages)) return [];

  const toolUses = new Map<string, { name: string; input: any }>();
  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      const toolCall = normalizeToolCallBlock(block);
      if (!toolCall) continue;
      toolUses.set(toolCall.id, {
        name: toolCall.function?.name ?? "tool",
        input: block.input ?? {},
      });
    }
  }

  const executions: Array<{
    id: string;
    name: string;
    input: any;
    output: any;
    isError: boolean;
  }> = [];

  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      if (!block || typeof block !== "object" || block.type !== "tool_result") continue;
      const toolUseId = block.tool_use_id ?? "";
      const toolUse = toolUses.get(toolUseId);
      executions.push({
        id: toolUseId,
        name: toolUse?.name ?? "tool",
        input: toolUse?.input ?? {},
        output: block.content ?? "",
        isError: block.is_error === true,
      });
    }
  }

  return executions;
}

function formatTools(tools: any[] | undefined): any[] | null {
  if (!tools || !tools.length) return null;

  const result: any[] = [];
  for (const tool of tools) {
    const entry: any = {
      type: "function",
      function: { name: tool.name ?? "" },
    };
    if (tool.description) entry.function.description = tool.description;
    if (tool.input_schema) entry.function.parameters = tool.input_schema;
    result.push(entry);
  }

  return result.length ? toSerializableValue(result) : null;
}

// ---------------------------------------------------------------------------
// Span construction (inline, matches _otel_emitter.ts pattern)
// ---------------------------------------------------------------------------

function buildReadableSpan(opts: {
  name: string;
  startTime: [number, number];
  endTime: [number, number];
  attributes: Record<string, any>;
  errorMessage?: string;
}): ReadableSpan {
  const activeSpan = trace.getSpan(context.active());
  const activeSpanContext = activeSpan?.spanContext();
  const traceId = activeSpanContext?.traceId ?? Array.from({ length: 32 }, () =>
    Math.floor(Math.random() * 16).toString(16),
  ).join("");
  const spanId = Array.from({ length: 16 }, () =>
    Math.floor(Math.random() * 16).toString(16),
  ).join("");
  const parentSpanId = activeSpanContext?.spanId;
  const traceFlags = activeSpanContext?.traceFlags ?? TraceFlags.SAMPLED;

  const status = opts.errorMessage
    ? { code: SpanStatusCode.ERROR, message: opts.errorMessage }
    : { code: SpanStatusCode.OK, message: "" };

  return {
    name: opts.name,
    kind: SpanKind.INTERNAL,
    spanContext: () => ({
      traceId,
      spanId,
      traceFlags,
      isRemote: false,
    }),
    parentSpanId,
    startTime: opts.startTime,
    endTime: opts.endTime,
    duration: hrTimeDuration(opts.startTime, opts.endTime),
    status,
    attributes: opts.attributes,
    links: [],
    events: [],
    resource: { attributes: {} } as any,
    instrumentationLibrary: {
      name: "@respan/instrumentation-anthropic",
      version: PACKAGE_VERSION,
    },
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as unknown as ReadableSpan;
}

function injectSpan(span: ReadableSpan): void {
  const tp = trace.getTracerProvider() as any;
  const processor =
    tp?.activeSpanProcessor ??
    tp?._delegate?.activeSpanProcessor ??
    tp?._delegate?._tracerProvider?.activeSpanProcessor;
  if (processor && typeof processor.onEnd === "function") {
    processor.onEnd(span);
  }
}

// ---------------------------------------------------------------------------
// Attribute building
// ---------------------------------------------------------------------------

function buildSpanAttrs(
  kwargs: Record<string, any>,
  message: any,
): Record<string, any> {
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: "anthropic.chat",
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "anthropic.chat",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_SYSTEM]: "anthropic",
  };

  // Model
  const model = message?.model ?? kwargs.model;
  if (model) attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = model;

  // Input
  const inputMsgs = formatInputMessages(kwargs.messages ?? [], kwargs.system);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);

  // Output
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([
    formatOutputMessage(message),
  ]);

  // Token usage
  if (message?.usage) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] =
      message.usage.input_tokens ?? 0;
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] =
      message.usage.output_tokens ?? 0;
  }

  // Tool calls
  const toolCalls = mergeToolCalls(
    extractToolCalls(message),
    extractToolCallsFromInputMessages(kwargs.messages),
  );
  if (toolCalls) attrs.tool_calls = safeJson(toolCalls);

  // Tool definitions
  const tools = formatTools(kwargs.tools);
  if (tools) attrs.tools = safeJson(tools);

  return attrs;
}

function buildErrorAttrs(kwargs: Record<string, any>): Record<string, any> {
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: "anthropic.chat",
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "anthropic.chat",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_SYSTEM]: "anthropic",
  };

  if (kwargs.model) attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = kwargs.model;

  const inputMsgs = formatInputMessages(kwargs.messages ?? [], kwargs.system);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);

  const toolCalls = extractToolCallsFromInputMessages(kwargs.messages);
  if (toolCalls) attrs.tool_calls = safeJson(toolCalls);

  const tools = formatTools(kwargs.tools);
  if (tools) attrs.tools = safeJson(tools);

  return attrs;
}

// ---------------------------------------------------------------------------
// Emit span helper
// ---------------------------------------------------------------------------

function emitSpan(
  attrs: Record<string, any>,
  startTime: [number, number],
  errorMessage?: string,
): void {
  try {
    const span = buildReadableSpan({
      name: "anthropic.chat",
      startTime,
      endTime: hrTime(),
      attributes: attrs,
      errorMessage,
    });
    injectSpan(span);
  } catch {
    // Never break the application
  }
}

function emitSuccessSpan(
  kwargs: Record<string, any>,
  startTime: [number, number],
  message: any,
): void {
  try {
    const attrs = buildSpanAttrs(kwargs, message);
    emitSpan(attrs, startTime);
  } catch {
    // Never break the application
  }
}

function emitErrorSpan(
  kwargs: Record<string, any>,
  startTime: [number, number],
  err: unknown,
): void {
  try {
    const attrs = buildErrorAttrs(kwargs);
    attrs["error.message"] = String(err);
    emitSpan(attrs, startTime, String(err));
  } catch {
    // Never break the application
  }
}

function emitToolSpan(toolExecution: {
  id: string;
  name: string;
  input: any;
  output: any;
  isError: boolean;
}): void {
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
  }

  const span = buildReadableSpan({
    name: `${toolExecution.name}.tool`,
    startTime,
    endTime: hrTime(),
    attributes: attrs,
    errorMessage: toolExecution.isError
      ? stringifyStructured(toolExecution.output)
      : undefined,
  });
  injectSpan(span);
}

function emitToolSpansFromMessages(messages: any[] | undefined): void {
  for (const toolExecution of extractToolExecutions(messages)) {
    try {
      emitToolSpan(toolExecution);
    } catch {
      // Never break the application
    }
  }
}

function cloneContentBlock(block: any): any {
  if (!block || typeof block !== "object") return block;
  if (Array.isArray(block)) return block.map((entry) => cloneContentBlock(entry));
  return { ...block };
}

function buildMessageFromStreamState(
  state: {
    message: any;
    usage: Record<string, any>;
    stopReason: string | null;
    stopSequence: string | null;
    contentBlocks: Map<number, any>;
  },
  kwargs: Record<string, any>,
): any {
  const content = Array.from(state.contentBlocks.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([, block]) => {
      const normalized = cloneContentBlock(block);
      if (normalized && typeof normalized === "object") {
        const jsonBuffer = normalized[TOOL_USE_JSON_BUFFER_KEY];
        delete normalized[TOOL_USE_JSON_BUFFER_KEY];

        if (
          normalized.type === "tool_use" &&
          typeof normalized.input === "string" &&
          typeof jsonBuffer === "string"
        ) {
          try {
            normalized.input = jsonBuffer.trim() ? JSON.parse(jsonBuffer) : {};
          } catch {
            normalized.input = jsonBuffer;
          }
        }
      }
      return normalized;
    });

  return {
    ...(state.message ?? {}),
    model: state.message?.model ?? kwargs.model,
    content,
    usage: state.usage,
    stop_reason: state.stopReason ?? state.message?.stop_reason ?? null,
    stop_sequence: state.stopSequence ?? state.message?.stop_sequence ?? null,
  };
}

function updateStreamState(
  state: {
    message: any;
    usage: Record<string, any>;
    stopReason: string | null;
    stopSequence: string | null;
    contentBlocks: Map<number, any>;
  },
  event: any,
): void {
  if (!event || typeof event !== "object") return;

  if (event.type === "message_start") {
    state.message = { ...(event.message ?? {}) };
    state.usage = { ...(event.message?.usage ?? {}) };

    if (Array.isArray(event.message?.content)) {
      for (const [index, block] of event.message.content.entries()) {
        state.contentBlocks.set(index, cloneContentBlock(block));
      }
    }
    return;
  }

  if (event.type === "content_block_start") {
    state.contentBlocks.set(event.index, cloneContentBlock(event.content_block));
    return;
  }

  if (event.type === "content_block_delta") {
    const existingBlock = state.contentBlocks.get(event.index) ?? {};
    const delta = event.delta ?? {};

    if (delta.type === "text_delta") {
      existingBlock.type ??= "text";
      existingBlock.text = `${existingBlock.text ?? ""}${delta.text ?? ""}`;
      state.contentBlocks.set(event.index, existingBlock);
      return;
    }

    if (delta.type === "input_json_delta") {
      existingBlock.type ??= "tool_use";
      const nextBuffer = `${existingBlock[TOOL_USE_JSON_BUFFER_KEY] ?? ""}${delta.partial_json ?? ""}`;
      existingBlock[TOOL_USE_JSON_BUFFER_KEY] = nextBuffer;

      try {
        existingBlock.input = nextBuffer.trim() ? JSON.parse(nextBuffer) : {};
      } catch {
        existingBlock.input = nextBuffer;
      }

      state.contentBlocks.set(event.index, existingBlock);
    }
    return;
  }

  if (event.type === "message_delta") {
    state.stopReason = event.delta?.stop_reason ?? state.stopReason;
    state.stopSequence = event.delta?.stop_sequence ?? state.stopSequence;
    if (event.usage && typeof event.usage === "object") {
      state.usage = { ...state.usage, ...event.usage };
    }
  }
}

function wrapStreamingCreateResult(
  streamResult: any,
  kwargs: Record<string, any>,
  startTime: [number, number],
): any {
  if (
    !streamResult ||
    typeof streamResult !== "object" ||
    streamResult[STREAM_INSTRUMENTED]
  ) {
    return streamResult;
  }

  Object.defineProperty(streamResult, STREAM_INSTRUMENTED, {
    value: true,
    configurable: true,
    enumerable: false,
  });

  const state = {
    message: null as any,
    usage: {} as Record<string, any>,
    stopReason: null as string | null,
    stopSequence: null as string | null,
    contentBlocks: new Map<number, any>(),
  };
  let hasEmitted = false;

  const emitFinalSpan = (error?: unknown) => {
    if (hasEmitted) return;
    hasEmitted = true;

    if (error) {
      emitErrorSpan(kwargs, startTime, error);
      return;
    }

    const message = buildMessageFromStreamState(state, kwargs);
    emitSuccessSpan(kwargs, startTime, message);
  };

  const originalAsyncIterator = streamResult[Symbol.asyncIterator]?.bind(streamResult);
  if (typeof originalAsyncIterator !== "function") {
    emitSuccessSpan(kwargs, startTime, streamResult);
    return streamResult;
  }

  streamResult[Symbol.asyncIterator] = function () {
    const iterator = originalAsyncIterator();

    return {
      async next(...args: any[]) {
        try {
          const result = await iterator.next(...args);
          if (result.done) {
            emitFinalSpan();
          } else {
            updateStreamState(state, result.value);
          }
          return result;
        } catch (err) {
          emitFinalSpan(err);
          throw err;
        }
      },

      async return(value?: any) {
        try {
          const result = typeof iterator.return === "function"
            ? await iterator.return(value)
            : { done: true, value };
          emitFinalSpan();
          return result;
        } catch (err) {
          emitFinalSpan(err);
          throw err;
        }
      },

      async throw(err?: any) {
        emitFinalSpan(err);
        if (typeof iterator.throw === "function") {
          return iterator.throw(err);
        }
        throw err;
      },

      [Symbol.asyncIterator]() {
        return this;
      },
    };
  };

  return streamResult;
}

function instrumentCreateResult(
  result: any,
  kwargs: Record<string, any>,
  startTime: [number, number],
): any {
  if (!result || typeof result !== "object") {
    return result;
  }

  let hasHandled = false;

  const handleSuccess = (value: any) => {
    if (kwargs?.stream === true) {
      return wrapStreamingCreateResult(value, kwargs, startTime);
    }

    if (!hasHandled) {
      hasHandled = true;
      emitSuccessSpan(kwargs, startTime, value);
    }
    return value;
  };

  const handleError = (err: unknown) => {
    if (hasHandled) return;
    hasHandled = true;
    emitErrorSpan(kwargs, startTime, err);
  };

  const originalThen = typeof result.then === "function" ? result.then.bind(result) : null;
  if (originalThen) {
    result.then = function (onfulfilled?: any, onrejected?: any) {
      return originalThen(
        (value: any) => {
          const instrumentedValue = handleSuccess(value);
          return onfulfilled ? onfulfilled(instrumentedValue) : instrumentedValue;
        },
        (reason: any) => {
          handleError(reason);
          if (onrejected) {
            return onrejected(reason);
          }
          throw reason;
        },
      );
    };
  }

  const originalCatch = typeof result.catch === "function" ? result.catch.bind(result) : null;
  if (originalCatch) {
    result.catch = function (onrejected?: any) {
      return originalCatch((reason: any) => {
        handleError(reason);
        if (onrejected) {
          return onrejected(reason);
        }
        throw reason;
      });
    };
  }

  const originalWithResponse =
    typeof result.withResponse === "function" ? result.withResponse.bind(result) : null;
  if (originalWithResponse) {
    result.withResponse = async function () {
      try {
        const response = await originalWithResponse();
        return {
          ...response,
          data: handleSuccess(response.data),
        };
      } catch (err) {
        handleError(err);
        throw err;
      }
    };
  }

  return result;
}

function findPackageDirectory(resolvedEntry: string): string | null {
  let currentDir = dirname(resolvedEntry);

  while (true) {
    if (existsSync(join(currentDir, "package.json"))) {
      return currentDir;
    }

    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) {
      return null;
    }
    currentDir = parentDir;
  }
}

function addAnthropicModuleCandidates(
  urls: Set<string>,
  resolverBase: string | URL,
): void {
  try {
    const require = createRequire(resolverBase);
    const resolvedEntry = require.resolve("@anthropic-ai/sdk");
    const packageDir = findPackageDirectory(resolvedEntry);

    if (!packageDir) return;

    for (const entryFile of ["index.mjs", "index.js"]) {
      const entryPath = join(packageDir, entryFile);
      if (existsSync(entryPath)) {
        urls.add(pathToFileURL(entryPath).href);
      }
    }
  } catch {
    // Ignore resolution failures for this candidate.
  }
}

async function loadAnthropicConstructors(): Promise<any[]> {
  const candidateUrls = new Set<string>();
  const runtimeResolutionBases = [
    join(process.cwd(), "__respan_runtime__.js"),
    process.env.INIT_CWD ? join(process.env.INIT_CWD, "__respan_init__.js") : null,
    process.argv[1] ?? null,
    import.meta.url,
  ].filter(Boolean) as Array<string | URL>;

  for (const resolutionBase of runtimeResolutionBases) {
    addAnthropicModuleCandidates(candidateUrls, resolutionBase);
  }

  const constructors: any[] = [];
  for (const moduleUrl of candidateUrls) {
    try {
      const importedModule = await import(moduleUrl);
      const Anthropic = importedModule?.default ?? importedModule;
      if (typeof Anthropic === "function" && !constructors.includes(Anthropic)) {
        constructors.push(Anthropic);
      }
    } catch {
      // Ignore candidate import failures so we can keep trying others.
    }
  }

  return constructors;
}

// ---------------------------------------------------------------------------
// Instrumentor
// ---------------------------------------------------------------------------

export class AnthropicInstrumentor {
  public readonly name = "anthropic";
  private static readonly _sharedState = {
    activeInstances: 0,
    patchedTargets: [] as Array<{
      messagesPrototype: any;
      originalCreate: any;
      originalStream: any;
    }>,
  };

  private _isInstrumented = false;

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const anthropicConstructors = await loadAnthropicConstructors();
    if (anthropicConstructors.length === 0) {
      console.warn(
        "[Respan] Failed to activate Anthropic instrumentation — @anthropic-ai/sdk not found",
      );
      return;
    }

    const sharedState = AnthropicInstrumentor._sharedState;

    try {
      for (const Anthropic of anthropicConstructors) {
        const tempClient = new Anthropic({ apiKey: "sk-placeholder" });
        const messagesProto = Object.getPrototypeOf(tempClient.messages);

        if (
          !messagesProto ||
          typeof messagesProto.create !== "function" ||
          sharedState.patchedTargets.some((target) => target.messagesPrototype === messagesProto)
        ) {
          continue;
        }

        const patchedTarget = {
          messagesPrototype: messagesProto,
          originalCreate: messagesProto.create,
          originalStream: typeof messagesProto.stream === "function"
            ? messagesProto.stream
            : null,
        };

        messagesProto.create = function (
          this: any,
          body: any,
          options?: any,
        ) {
          const startTime = hrTime();
          try {
            emitToolSpansFromMessages(body?.messages);
            const result = patchedTarget.originalCreate.call(this, body, options);
            return instrumentCreateResult(result, body, startTime);
          } catch (err: any) {
            emitErrorSpan(body, startTime, err);
            throw err;
          }
        };

        sharedState.patchedTargets.push(patchedTarget);
      }

      if (sharedState.patchedTargets.length === 0) {
        console.warn(
          "[Respan] Failed to activate Anthropic instrumentation — no compatible Messages prototypes found",
        );
        return;
      }

      sharedState.activeInstances += 1;
      this._isInstrumented = true;
    } catch (err) {
      console.warn("[Respan] Failed to activate Anthropic instrumentation:", err);
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = AnthropicInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0 || sharedState.patchedTargets.length === 0) return;

    try {
      for (const patchedTarget of sharedState.patchedTargets) {
        patchedTarget.messagesPrototype.create = patchedTarget.originalCreate;
        if (patchedTarget.originalStream) {
          patchedTarget.messagesPrototype.stream = patchedTarget.originalStream;
        }
      }
    } catch {
      /* ignore */
    }

    sharedState.patchedTargets = [];
  }
}
