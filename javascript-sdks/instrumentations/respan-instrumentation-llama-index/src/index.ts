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

function eventDetail(event: any): Record<string, any> {
  return event?.detail && typeof event.detail === "object" ? event.detail : {};
}

function eventId(detail: Record<string, any>, fallbackPrefix: string): string {
  return String(detail.id ?? `${fallbackPrefix}-${Math.random().toString(16).slice(2)}`);
}

export class LlamaIndexInstrumentor {
  public readonly name = "llama-index";
  private readonly _options: LlamaIndexInstrumentorOptions;
  private readonly _emitter: LlamaIndexSpanEmitter;
  private _callbackManager: CallbackManagerLike | null = null;
  private _bindings: HandlerBinding[] = [];
  private _isInstrumented = false;

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
          this._emitter.startLLM({
            id: eventId(detail, "llm"),
            messages: detail.messages,
            startTime: hrTime(),
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
    return [
      {
        event: startEvent,
        handler: (event) => {
          const detail = eventDetail(event);
          this._emitter.startRecord({
            id: eventId(detail, fallbackPrefix),
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
          this._emitter.endRecord({
            id: eventId(detail, fallbackPrefix),
            output: formatTaskOutput(detail),
            endTime: hrTime(),
          });
        },
      },
    ];
  }
}

export { LlamaIndexSpanEmitter } from "./_emitter.js";
