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

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";

const PACKAGE_VERSION = "1.1.0";

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
    let content = msg?.content ?? "";

    if (Array.isArray(content)) {
      content = content
        .map((b: any) => normalizeContentBlock(b))
        .filter(Boolean)
        .join("\n");
    }

    result.push({ role, content });
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

function extractToolCalls(message: any): any[] | null {
  const content = message?.content;
  if (!Array.isArray(content)) return null;

  const toolCalls: any[] = [];
  for (const block of content) {
    if (block?.type !== "tool_use") continue;
    toolCalls.push({
      id: block.id ?? "",
      type: "function",
      function: {
        name: block.name ?? "",
        arguments: safeJson(block.input ?? {}),
      },
    });
  }

  return toolCalls.length ? toolCalls : null;
}

function formatTools(tools: any[] | undefined): string | null {
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

  return result.length ? safeJson(result) : null;
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
  // Generate random hex IDs
  const traceId = Array.from({ length: 32 }, () =>
    Math.floor(Math.random() * 16).toString(16),
  ).join("");
  const spanId = Array.from({ length: 16 }, () =>
    Math.floor(Math.random() * 16).toString(16),
  ).join("");

  const status = opts.errorMessage
    ? { code: SpanStatusCode.ERROR, message: opts.errorMessage }
    : { code: SpanStatusCode.OK, message: "" };

  return {
    name: opts.name,
    kind: SpanKind.INTERNAL,
    spanContext: () => ({
      traceId,
      spanId,
      traceFlags: 1,
      isRemote: false,
    }),
    parentSpanId: undefined,
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
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formatOutput(message);

  // Token usage
  if (message?.usage) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] =
      message.usage.input_tokens ?? 0;
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] =
      message.usage.output_tokens ?? 0;
  }

  // Tool calls
  const toolCalls = extractToolCalls(message);
  if (toolCalls) attrs[RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS] = safeJson(toolCalls);

  // Tool definitions
  const toolsJson = formatTools(kwargs.tools);
  if (toolsJson) attrs[RespanSpanAttributes.RESPAN_SPAN_TOOLS] = toolsJson;

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

// ---------------------------------------------------------------------------
// Instrumentor
// ---------------------------------------------------------------------------

let _originalCreate: any = null;
let _originalStream: any = null;
let _messagesPrototype: any = null;

export class AnthropicInstrumentor {
  public readonly name = "anthropic";
  private _isInstrumented = false;

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    let Anthropic: any;
    try {
      Anthropic = (await import("@anthropic-ai/sdk")).default;
    } catch {
      console.warn(
        "[Respan] Failed to activate Anthropic instrumentation — @anthropic-ai/sdk not found",
      );
      return;
    }

    // We need to get the Messages prototype. Create a temporary client
    // to access it, then patch the prototype so all instances are covered.
    try {
      const tempClient = new Anthropic({ apiKey: "sk-placeholder" });
      const messagesProto = Object.getPrototypeOf(tempClient.messages);
      _messagesPrototype = messagesProto;

      // Patch create
      if (!_originalCreate) {
        _originalCreate = messagesProto.create;
      }
      messagesProto.create = async function (
        this: any,
        body: any,
        options?: any,
      ) {
        const startTime = hrTime();
        try {
          const message = await _originalCreate.call(this, body, options);
          try {
            const attrs = buildSpanAttrs(body, message);
            emitSpan(attrs, startTime);
          } catch {
            // Never break the application
          }
          return message;
        } catch (err: any) {
          try {
            const attrs = buildErrorAttrs(body);
            emitSpan(attrs, startTime, String(err));
          } catch {
            // Never break the application
          }
          throw err;
        }
      };

      // Patch stream
      if (typeof messagesProto.stream === "function") {
        if (!_originalStream) {
          _originalStream = messagesProto.stream;
        }
        messagesProto.stream = function (
          this: any,
          body: any,
          options?: any,
        ) {
          const startTime = hrTime();
          const streamResult = _originalStream.call(this, body, options);
          let spanEmitted = false;

          const emitStreamSpan = (message: any) => {
            if (spanEmitted) return;
            spanEmitted = true;
            try {
              const attrs = buildSpanAttrs(body, message);
              emitSpan(attrs, startTime);
            } catch {
              // Never break the application
            }
          };

          // The Anthropic JS SDK stream returns a MessageStream which is
          // an async iterable with a .finalMessage() promise
          const originalFinalMessage = streamResult.finalMessage?.bind(streamResult);
          if (typeof originalFinalMessage === "function") {
            streamResult.finalMessage = async function () {
              const message = await originalFinalMessage();
              emitStreamSpan(message);
              return message;
            };
          }

          // Also hook into the finalMessage event if EventEmitter-style
          if (typeof streamResult.on === "function") {
            const subscribeMethod =
              typeof streamResult.once === "function" ? "once" : "on";
            streamResult[subscribeMethod]("finalMessage", (message: any) => {
              emitStreamSpan(message);
            });
          }

          return streamResult;
        };
      }

      this._isInstrumented = true;
    } catch (err) {
      console.warn("[Respan] Failed to activate Anthropic instrumentation:", err);
    }
  }

  deactivate(): void {
    if (!this._isInstrumented || !_messagesPrototype) return;

    try {
      if (_originalCreate) {
        _messagesPrototype.create = _originalCreate;
        _originalCreate = null;
      }
      if (_originalStream) {
        _messagesPrototype.stream = _originalStream;
        _originalStream = null;
      }
    } catch {
      /* ignore */
    }

    _messagesPrototype = null;
    this._isInstrumented = false;
  }
}
