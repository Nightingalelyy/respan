/**
 * Respan instrumentation plugin for LlamaIndex TypeScript.
 *
 * The plugin attaches event handlers to LlamaIndex's CallbackManager and emits
 * OTEL ReadableSpan objects into the active Respan tracing pipeline.
 */

import { hrTime } from "@opentelemetry/core";
import { RespanLogType } from "@respan/respan-sdk";
import {
  LLAMA_INDEX_EVENTS,
  type LlamaIndexEventName,
} from "./_constants.js";
import {
  LlamaIndexSpanEmitter,
  formatAgentStepId,
  formatTaskInput,
  formatTaskOutput,
} from "./_emitter.js";
import { sanitizeErrorMessage } from "./_helpers.js";

export interface LlamaIndexInstrumentorOptions {
  workflowName?: string;
  llamaIndexModule?: Record<string, any>;
}

type CallbackManagerLike = {
  on(event: string, handler: (event: any) => void): unknown;
  off(event: string, handler: (event: any) => void): unknown;
};

type HandlerBinding = {
  event: LlamaIndexEventName;
  handler: (event: any) => void;
};

type LlamaIndexSettingsLike = {
  llm?: unknown;
  withLLM?: (llm: unknown, fn: (...args: any[]) => any) => any;
};

type PatchedLlm = Record<string, any> & {
  chat: (...args: any[]) => any;
};

function eventDetail(event: any): Record<string, any> {
  return event?.detail && typeof event.detail === "object" ? event.detail : {};
}

function eventId(
  detail: Record<string, any>,
  fallbackPrefix: string,
  event?: any,
): string {
  return eventCorrelationId(detail, event) ??
    `${fallbackPrefix}-${Math.random().toString(16).slice(2)}`;
}

function eventCorrelationId(
  detail: Record<string, any>,
  event?: any,
): string | undefined {
  const value = detail.id ?? event?.reason?.id;
  return value === undefined || value === null || value === ""
    ? undefined
    : String(value);
}

function eventCaller(event: any): object | undefined {
  const callers = event?.reason?.computedCallers;
  if (!Array.isArray(callers)) return undefined;
  return callers.find(
    (caller) => caller && typeof caller === "object" && typeof caller.chat === "function",
  );
}

function callerModel(caller: object | undefined): string | undefined {
  if (!caller) return undefined;
  const callerRecord = caller as Record<string, any>;
  const model = callerRecord.model ?? callerRecord.metadata?.model;
  return typeof model === "string" && model.trim() ? model : undefined;
}

export class LlamaIndexInstrumentor {
  public readonly name = "llama-index";
  private readonly _options: LlamaIndexInstrumentorOptions;
  private readonly _emitter: LlamaIndexSpanEmitter;
  private _callbackManager: CallbackManagerLike | null = null;
  private _bindings: HandlerBinding[] = [];
  private _isInstrumented = false;
  private _settings: LlamaIndexSettingsLike | null = null;
  private _llmDescriptor?: PropertyDescriptor;
  private _llmDescriptorWasOwn = false;
  private _settingsWithLLM?: LlamaIndexSettingsLike["withLLM"];
  private _withLLMWasOwn = false;
  private readonly _patchedLlms = new Map<object, PropertyDescriptor | undefined>();

  constructor(options: LlamaIndexInstrumentorOptions = {}) {
    this._options = options;
    this._emitter = new LlamaIndexSpanEmitter({
      workflowName: options.workflowName,
    });
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    let llamaIndex: any = this._options.llamaIndexModule;
    try {
      llamaIndex ??= await import("llamaindex");
    } catch (error) {
      console.warn(
        "[respan] LlamaIndexInstrumentor failed to activate: install the `llamaindex` package.",
        error,
      );
      return;
    }

    const callbackManager = llamaIndex.Settings?.callbackManager;
    if (
      !callbackManager ||
      typeof callbackManager.on !== "function" ||
      typeof callbackManager.off !== "function"
    ) {
      console.warn(
        "[respan] LlamaIndexInstrumentor failed to activate: no compatible LlamaIndex CallbackManager found.",
      );
      return;
    }

    this._callbackManager = callbackManager;
    this._patchSettings(llamaIndex.Settings as LlamaIndexSettingsLike);
    this._bindings = this._buildHandlers();
    for (const binding of this._bindings) {
      callbackManager.on(binding.event, binding.handler);
    }
    this._isInstrumented = true;
  }

  deactivate(): void {
    if (!this._isInstrumented || !this._callbackManager) {
      return;
    }

    for (const binding of this._bindings) {
      try {
        this._callbackManager.off(binding.event, binding.handler);
      } catch {
        // Ignore removal failures; instrumentation must not break user code.
      }
    }

    this._emitter.flushPending({
      errorMessage: "LlamaIndex operation ended without a matching completion event",
      endTime: hrTime(),
    });
    this._restoreSettings();
    this._emitter.clear();
    this._bindings = [];
    this._callbackManager = null;
    this._isInstrumented = false;
  }

  private _buildHandlers(): HandlerBinding[] {
    return [
      {
        event: LLAMA_INDEX_EVENTS.LLM_START,
        handler: (event) => {
          const detail = eventDetail(event);
          const caller = eventCaller(event);
          this._emitter.startLLM({
            id: eventId(detail, "llm"),
            messages: detail.messages,
            startTime: hrTime(),
            model: callerModel(caller) ?? this._settingsModel(),
            caller,
          });
        },
      },
      {
        event: LLAMA_INDEX_EVENTS.LLM_END,
        handler: (event) => {
          const detail = eventDetail(event);
          this._emitter.endLLM({
            id: eventId(detail, "llm"),
            response: detail.response,
            endTime: hrTime(),
          });
        },
      },
      {
        event: LLAMA_INDEX_EVENTS.LLM_TOOL_CALL,
        handler: (event) => {
          const detail = eventDetail(event);
          this._emitter.recordToolCall({
            toolCall: detail.toolCall,
            startTime: hrTime(),
          });
        },
      },
      {
        event: LLAMA_INDEX_EVENTS.LLM_TOOL_RESULT,
        handler: (event) => {
          const detail = eventDetail(event);
          this._emitter.emitToolResult({
            toolCall: detail.toolCall,
            toolResult: detail.toolResult,
            endTime: hrTime(),
          });
        },
      },
      this._startEndBinding(
        LLAMA_INDEX_EVENTS.QUERY_START,
        LLAMA_INDEX_EVENTS.QUERY_END,
        "llamaindex.query",
        RespanLogType.WORKFLOW,
        "query",
      ),
      this._startEndBinding(
        LLAMA_INDEX_EVENTS.RETRIEVE_START,
        LLAMA_INDEX_EVENTS.RETRIEVE_END,
        "llamaindex.retrieve",
        RespanLogType.TASK,
        "retrieve",
      ),
      this._startEndBinding(
        LLAMA_INDEX_EVENTS.SYNTHESIZE_START,
        LLAMA_INDEX_EVENTS.SYNTHESIZE_END,
        "llamaindex.synthesize",
        RespanLogType.TASK,
        "synthesize",
      ),
      this._startEndBinding(
        LLAMA_INDEX_EVENTS.CHUNKING_START,
        LLAMA_INDEX_EVENTS.CHUNKING_END,
        "llamaindex.chunking",
        RespanLogType.TASK,
        "chunking",
      ),
      this._startEndBinding(
        LLAMA_INDEX_EVENTS.NODE_PARSING_START,
        LLAMA_INDEX_EVENTS.NODE_PARSING_END,
        "llamaindex.node_parsing",
        RespanLogType.TASK,
        "node-parsing",
      ),
      {
        event: LLAMA_INDEX_EVENTS.AGENT_START,
        handler: (event) => {
          const detail = eventDetail(event);
          const id = formatAgentStepId(detail) ?? eventId(detail, "agent");
          this._emitter.startRecord({
            id,
            name: "llamaindex.agent",
            logType: RespanLogType.AGENT,
            startTime: hrTime(),
            input: formatTaskInput(detail),
          });
        },
      },
      {
        event: LLAMA_INDEX_EVENTS.AGENT_END,
        handler: (event) => {
          const detail = eventDetail(event);
          const id = formatAgentStepId(detail) ?? eventId(detail, "agent");
          this._emitter.endRecord({
            id,
            output: formatTaskOutput(detail),
            endTime: hrTime(),
          });
        },
      },
    ].flat();
  }

  private _startEndBinding(
    startEvent: LlamaIndexEventName,
    endEvent: LlamaIndexEventName,
    name: string,
    logType: string,
    fallbackPrefix: string,
  ): HandlerBinding[] {
    const pendingIds: string[] = [];
    return [
      {
        event: startEvent,
        handler: (event) => {
          const detail = eventDetail(event);
          const id = eventId(detail, fallbackPrefix, event);
          pendingIds.push(id);
          this._emitter.startRecord({
            id,
            name,
            logType,
            startTime: hrTime(),
            input: formatTaskInput(detail),
          });
        },
      },
      {
        event: endEvent,
        handler: (event) => {
          const detail = eventDetail(event);
          const correlatedId = eventCorrelationId(detail, event);
          const correlatedIndex = correlatedId
            ? pendingIds.lastIndexOf(correlatedId)
            : -1;
          const id = correlatedIndex >= 0
            ? pendingIds.splice(correlatedIndex, 1)[0]
            : pendingIds.pop() ?? correlatedId ?? eventId(detail, fallbackPrefix, event);
          this._emitter.endRecord({
            id,
            output: formatTaskOutput(detail),
            endTime: hrTime(),
          });
        },
      },
    ];
  }

  private _patchSettings(settings: LlamaIndexSettingsLike): void {
    this._settings = settings;
    this._llmDescriptorWasOwn = Object.hasOwn(settings, "llm");
    let descriptorOwner: object | null = settings;
    while (descriptorOwner && !Object.getOwnPropertyDescriptor(descriptorOwner, "llm")) {
      descriptorOwner = Object.getPrototypeOf(descriptorOwner);
    }
    this._llmDescriptor = descriptorOwner
      ? Object.getOwnPropertyDescriptor(descriptorOwner, "llm")
      : undefined;
    const originalDescriptor = this._llmDescriptor;
    if (originalDescriptor?.get && originalDescriptor.set && originalDescriptor.configurable) {
      const instrumentor = this;
      Object.defineProperty(settings, "llm", {
        ...originalDescriptor,
        get: originalDescriptor.get,
        set(value: unknown) {
          originalDescriptor.set?.call(settings, instrumentor._patchLlm(value));
        },
      });
      try {
        this._patchLlm(originalDescriptor.get.call(settings));
      } catch {
        // LlamaIndex throws while no default LLM has been configured.
      }
    }

    if (typeof settings.withLLM === "function") {
      this._withLLMWasOwn = Object.hasOwn(settings, "withLLM");
      this._settingsWithLLM = settings.withLLM;
      const originalWithLLM = settings.withLLM;
      settings.withLLM = (llm, fn) => originalWithLLM.call(settings, this._patchLlm(llm), fn);
    }
  }

  private _settingsModel(): string | undefined {
    try {
      const llm = this._settings?.llm;
      return llm && typeof llm === "object" ? callerModel(llm) : undefined;
    } catch {
      return undefined;
    }
  }

  private _patchLlm(value: unknown): unknown {
    if (!value || typeof value !== "object") return value;
    const llm = value as PatchedLlm;
    if (typeof llm.chat !== "function" || this._patchedLlms.has(llm)) return llm;

    const ownDescriptor = Object.getOwnPropertyDescriptor(llm, "chat");
    const originalChat = llm.chat;
    const instrumentor = this;
    const wrappedChat = async function (this: object, ...args: any[]): Promise<any> {
      const caller = this;
      try {
        const response = await originalChat.apply(this, args);
        if (response && typeof response === "object" && Symbol.asyncIterator in response) {
          const originalIterator = response[Symbol.asyncIterator].bind(response);
          response[Symbol.asyncIterator] = async function* () {
            try {
              yield* originalIterator();
            } catch (error) {
              instrumentor._recordLlmFailure(caller, error);
              throw error;
            }
          };
        }
        return response;
      } catch (error) {
        instrumentor._recordLlmFailure(this, error);
        throw error;
      }
    };
    Object.defineProperty(llm, "chat", {
      configurable: true,
      enumerable: ownDescriptor?.enumerable ?? false,
      writable: true,
      value: wrappedChat,
    });
    this._patchedLlms.set(llm, ownDescriptor);
    return llm;
  }

  private _recordLlmFailure(caller: object, error: unknown): void {
    this._emitter.failLLMForCaller({
      caller,
      errorMessage: sanitizeErrorMessage(error),
      endTime: hrTime(),
    });
  }

  private _restoreSettings(): void {
    if (this._settings && this._llmDescriptor) {
      if (this._llmDescriptorWasOwn) {
        Object.defineProperty(this._settings, "llm", this._llmDescriptor);
      } else {
        delete (this._settings as Record<string, unknown>).llm;
      }
    }
    if (this._settings && this._settingsWithLLM) {
      if (this._withLLMWasOwn) {
        this._settings.withLLM = this._settingsWithLLM;
      } else {
        delete (this._settings as Record<string, unknown>).withLLM;
      }
    }
    for (const [llm, descriptor] of this._patchedLlms) {
      if (descriptor) {
        Object.defineProperty(llm, "chat", descriptor);
      } else {
        delete (llm as Record<string, unknown>).chat;
      }
    }
    this._patchedLlms.clear();
    this._settings = null;
    this._llmDescriptor = undefined;
    this._llmDescriptorWasOwn = false;
    this._settingsWithLLM = undefined;
    this._withLLMWasOwn = false;
  }
}

export { LlamaIndexSpanEmitter } from "./_emitter.js";
